import pytest
from pydantic import ValidationError

from ugsl_ai_coach.integration.errors import (
    AdapterContractError, AnalysisIdConflict, ConcurrentUpdate, IdempotencyConflict,
    InvalidTransition, JobNotFound,
)
from ugsl_ai_coach.integration.lifecycle import transition_job
from ugsl_ai_coach.integration.models import AnalysisJob, AnalysisSubmission, JobAcceptance, JobState
from ugsl_ai_coach.integration.service import AnalysisIntegrationService
from .fakes import FakeDispatcher, FakeIdFactory, FakeRepository


@pytest.fixture
def wiring():
    repository, dispatcher, factory = FakeRepository(), FakeDispatcher(), FakeIdFactory()
    return AnalysisIntegrationService(repository, dispatcher, factory), repository, dispatcher, factory


def test_submit_get_and_duplicate_work(wiring, submission):
    service, repository, dispatcher, factory = wiring
    first = service.submit(submission)
    second = service.submit(submission)
    assert first == second == service.get(first.analysis_id)
    assert first.submission == submission
    assert first.state == JobState.SUBMITTED
    assert len(repository.jobs) == len(dispatcher.dispatched) == factory.calls == 1


@pytest.mark.parametrize("field,value", [
    ("attempt_id", "ATT-000002"), ("learner_video_ref", "changed-video"),
    ("reference_profile_ref", "changed-profile"),
])
def test_conflicting_key_rejected_without_side_effects(wiring, submission, field, value):
    service, repository, dispatcher, factory = wiring
    original = service.submit(submission)
    changed = AnalysisSubmission.model_validate({**submission.model_dump(), field: value})
    with pytest.raises(IdempotencyConflict):
        service.submit(changed)
    assert service.get(original.analysis_id) == original
    assert len(repository.jobs) == len(dispatcher.dispatched) == factory.calls == 1


@pytest.mark.parametrize("state", [JobState.PROCESSING, *[s for s in JobState if s.is_terminal]])
def test_duplicate_returns_current_job_without_rewriting_or_redispatch(wiring, submission, make_result, state):
    service, repository, dispatcher, factory = wiring
    submitted = service.submit(submission)
    current = service.mark_processing(submitted.analysis_id)
    if state.is_terminal:
        current = service.complete(submitted.analysis_id, make_result(state.value))
    assert service.submit(submission) == current
    assert len(repository.jobs) == len(dispatcher.dispatched) == factory.calls == 1


@pytest.mark.parametrize("state", [s for s in JobState if s.is_terminal])
def test_complete_preserves_traceability_and_frozen_evidence(wiring, submission, make_result, state):
    service, repository, dispatcher, factory = wiring
    first = service.submit(submission)
    processing = service.mark_processing(first.analysis_id)
    result = make_result(state.value)
    terminal = service.complete(first.analysis_id, result)
    assert terminal.analysis_id == result.analysis_id
    assert terminal.attempt_id == result.attempt_id == submission.attempt_id
    assert terminal.state.value == result.status.value
    assert first.state == JobState.SUBMITTED and processing.state == JobState.PROCESSING
    assert terminal.structured_analysis == result
    for model in (terminal, terminal.structured_analysis):
        field = next(iter(type(model).model_fields))
        with pytest.raises(ValidationError):
            setattr(model, field, getattr(model, field))


@pytest.mark.parametrize("state", [s for s in JobState if s.is_terminal])
def test_terminal_analysis_updates_rejected(wiring, submission, make_result, state):
    service, _, _, _ = wiring
    job = service.submit(submission)
    service.mark_processing(job.analysis_id)
    result = make_result(state.value)
    terminal = service.complete(job.analysis_id, result)
    for operation in (
        lambda: service.mark_processing(job.analysis_id),
        lambda: service.complete(job.analysis_id, result),
        lambda: service.complete(job.analysis_id, make_result("FAILED" if state != JobState.FAILED else "COMPLETED")),
    ):
        with pytest.raises(InvalidTransition):
            operation()
        assert service.get(job.analysis_id) == terminal


def test_skip_and_repeat_processing_rejected(wiring, submission, make_result):
    service, _, _, _ = wiring
    job = service.submit(submission)
    with pytest.raises(InvalidTransition):
        service.complete(job.analysis_id, make_result())
    service.mark_processing(job.analysis_id)
    with pytest.raises(InvalidTransition):
        service.mark_processing(job.analysis_id)


def test_invalid_result_does_not_change_stored_job(wiring, submission, make_result):
    service, _, _, _ = wiring
    job = service.submit(submission)
    processing = service.mark_processing(job.analysis_id)
    with pytest.raises(ValidationError):
        service.complete(job.analysis_id, make_result(attempt_id="ATT-000002"))
    assert service.get(job.analysis_id) == processing


def test_new_attempt_keeps_first_record(wiring, submission, make_result):
    service, repository, dispatcher, _ = wiring
    first = service.submit(submission)
    service.mark_processing(first.analysis_id)
    first = service.complete(first.analysis_id, make_result())
    second_request = AnalysisSubmission(
        attempt_id="ATT-000002", learner_video_ref="synthetic-video-2",
        reference_profile_ref=submission.reference_profile_ref, idempotency_key="synthetic-key-2",
    )
    second = service.submit(second_request)
    assert second.analysis_id != first.analysis_id and second.attempt_id != first.attempt_id
    assert service.get(first.analysis_id) == first
    assert len(repository.jobs) == len(dispatcher.dispatched) == 2


def test_atomic_accept_resolves_stale_lookup_without_duplicate_dispatch(submission, submitted):
    class StaleLookupRepository(FakeRepository):
        def __init__(self):
            super().__init__()
            self.hide_next = False

        def get_by_idempotency_key(self, key):
            if self.hide_next:
                self.hide_next = False
                return None
            return super().get_by_idempotency_key(key)
    repository = StaleLookupRepository()
    repository.accept(submitted)
    repository.hide_next = True
    dispatcher = FakeDispatcher()
    factory = FakeIdFactory()
    factory.calls = 1
    service = AnalysisIntegrationService(repository, dispatcher, factory)
    assert service.submit(submission) == submitted
    assert len(repository.jobs) == 1 and not dispatcher.dispatched


def test_atomic_accept_rejects_racing_conflicting_payload(submission, submitted):
    class RacingRepository(FakeRepository):
        def accept(self, candidate):
            super().accept(submitted)
            return super().accept(candidate)
    repository = RacingRepository()
    dispatcher = FakeDispatcher()
    service = AnalysisIntegrationService(repository, dispatcher, FakeIdFactory())
    changed = AnalysisSubmission.model_validate({**submission.model_dump(), "learner_video_ref": "changed"})
    with pytest.raises(IdempotencyConflict):
        service.submit(changed)
    assert len(repository.jobs) == 1 and not dispatcher.dispatched


def test_id_collision_never_overwrites_history(wiring, submission):
    service, repository, dispatcher, factory = wiring
    first = service.submit(submission)
    factory.calls = 0
    second = AnalysisSubmission.model_validate({**submission.model_dump(), "attempt_id": "ATT-000002", "idempotency_key": "new-key"})
    with pytest.raises(AnalysisIdConflict):
        service.submit(second)
    assert service.get(first.analysis_id) == first
    assert len(repository.jobs) == len(dispatcher.dispatched) == 1


def test_stale_snapshot_cannot_rewrite_a_winner(wiring, submission, make_result):
    service, repository, _, _ = wiring
    first = service.submit(submission)
    processing = service.mark_processing(first.analysis_id)
    winner = service.complete(first.analysis_id, make_result())
    losing_update = transition_job(processing, JobState.FAILED, structured_analysis=make_result("FAILED"))
    with pytest.raises(ConcurrentUpdate):
        repository.compare_and_set(processing, losing_update)
    assert service.get(first.analysis_id) == winner


def test_missing_job_and_invalid_id(wiring):
    service, _, _, _ = wiring
    with pytest.raises(JobNotFound):
        service.get("AN-999999")
    with pytest.raises(ValidationError):
        service.get("AN-1")
    with pytest.raises(JobNotFound):
        service.mark_processing("AN-999999")


def test_invalid_factory_output_not_stored_or_dispatched(submission):
    class InvalidFactory:
        def create(self):
            return "AN-uuid"
    repository, dispatcher = FakeRepository(), FakeDispatcher()
    service = AnalysisIntegrationService(repository, dispatcher, InvalidFactory())
    with pytest.raises(ValidationError):
        service.submit(submission)
    assert not repository.jobs and not dispatcher.dispatched


def test_dispatch_failure_is_visible_without_automatic_retry(submission):
    class BrokenDispatcher(FakeDispatcher):
        def dispatch(self, job):
            self.dispatched.append(job)
            raise RuntimeError("synthetic dispatch error")
    repository, dispatcher, factory = FakeRepository(), BrokenDispatcher(), FakeIdFactory()
    service = AnalysisIntegrationService(repository, dispatcher, factory)
    with pytest.raises(RuntimeError, match="dispatch error"):
        service.submit(submission)
    assert len(repository.jobs) == 1
    assert service.submit(submission).state == JobState.SUBMITTED
    assert len(dispatcher.dispatched) == 1


def test_storage_failure_is_not_swallowed_or_dispatched(submission):
    class BrokenRepository(FakeRepository):
        def accept(self, candidate):
            raise RuntimeError("synthetic storage error")
    dispatcher = FakeDispatcher()
    service = AnalysisIntegrationService(BrokenRepository(), dispatcher, FakeIdFactory())
    with pytest.raises(RuntimeError, match="storage error"):
        service.submit(submission)
    assert not dispatcher.dispatched


def test_adapter_cannot_return_another_job(wiring, submitted):
    service, repository, _, _ = wiring
    repository.jobs["AN-000002"] = submitted
    with pytest.raises(AdapterContractError):
        service.get("AN-000002")


def test_adapter_cannot_change_accepted_payload(submission):
    class BadRepository(FakeRepository):
        def accept(self, candidate):
            changed = AnalysisJob.model_validate({**candidate.model_dump(), "learner_video_ref": "changed"})
            return JobAcceptance(job=changed, created=True)
    dispatcher = FakeDispatcher()
    service = AnalysisIntegrationService(BadRepository(), dispatcher, FakeIdFactory())
    with pytest.raises(AdapterContractError):
        service.submit(submission)
    assert not dispatcher.dispatched


def test_service_revalidates_unchecked_submission(wiring, submission):
    service, repository, dispatcher, _ = wiring
    with pytest.raises(ValidationError):
        service.submit(submission.model_copy(update={"learner_video_ref": ""}))
    assert not repository.jobs and not dispatcher.dispatched


def test_whitespace_normalization_preserves_logical_idempotency(wiring, submission):
    service, repository, dispatcher, factory = wiring
    first = service.submit(submission)
    values = submission.model_dump()
    for field in ("learner_video_ref", "reference_profile_ref", "idempotency_key"):
        values[field] = f"  {values[field]}  "
    assert service.submit(AnalysisSubmission(**values)) == first
    assert len(repository.jobs) == len(dispatcher.dispatched) == factory.calls == 1


def test_repository_missing_snapshot_and_repeated_terminal_update(submitted, make_result):
    repository = FakeRepository()
    processing = transition_job(submitted, JobState.PROCESSING)
    with pytest.raises(JobNotFound):
        repository.compare_and_set(submitted, processing)
    repository.accept(submitted)
    repository.compare_and_set(submitted, processing)
    terminal = transition_job(processing, JobState.COMPLETED, structured_analysis=make_result())
    repository.compare_and_set(processing, terminal)
    with pytest.raises(InvalidTransition):
        repository.compare_and_set(terminal, terminal)
    assert repository.get(terminal.analysis_id) == terminal


def test_bad_update_response_is_rejected(submission):
    class BadUpdateRepository(FakeRepository):
        def compare_and_set(self, expected, replacement):
            return expected
    service = AnalysisIntegrationService(BadUpdateRepository(), FakeDispatcher(), FakeIdFactory())
    first = service.submit(submission)
    with pytest.raises(AdapterContractError):
        service.mark_processing(first.analysis_id)
