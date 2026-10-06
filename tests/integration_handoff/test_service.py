import pytest
from pydantic import ValidationError

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.integration.errors import AdapterContractError, AnalysisIdConflict, IdempotencyConflict, InvalidTransition, JobNotFound
from ugsl_ai_coach.integration.handoff.lifecycle import ClaimConflict
from ugsl_ai_coach.integration.handoff.models import JobWorkPair, WorkAcceptance, WorkState
from ugsl_ai_coach.integration.handoff.service import AnalysisHandoffService
from ugsl_ai_coach.integration.models import AnalysisSubmission, JobState
from ugsl_ai_coach.integration.service import CoachingIntegrationService
from tests.integration_contract.fakes import FakeCoachingRepository
from .fakes import FakeIdFactory, FakePersistence, FakeStore


def submit_and_claim(wiring, submission):
    service, persistence, _ = wiring
    service.submit(submission)
    return service.claim_next(now_ms=10, lease_duration_ms=10)


def test_acceptance_is_one_pair_without_dispatch_and_survives_simulated_restart(wiring, submission):
    service, persistence, factory = wiring
    job = service.submit(submission)
    pair = service.get(job.analysis_id)
    assert pair.job == job and pair.work.state == WorkState.PENDING
    assert pair.work.delivery_count == 0
    # A new adapter instance reconstructs a fake durable snapshot; no real disk is used.
    restarted_store = FakeStore.restore(persistence.store.snapshot())
    restarted = AnalysisHandoffService(FakePersistence(restarted_store), factory)
    claimed = restarted.claim_next(now_ms=100, lease_duration_ms=10)
    assert claimed.analysis_id == job.analysis_id and claimed.attempt_id == job.attempt_id
    assert len(restarted_store.pairs) == 1


def test_duplicate_submission_never_adds_work_or_delivery(wiring, submission):
    service, persistence, factory = wiring
    first = service.submit(submission)
    second = service.submit(submission)
    assert first == second
    assert len(persistence.store.pairs) == factory.calls == 1
    assert service.get(first.analysis_id).work.delivery_count == 0


@pytest.mark.parametrize("field,value", [("attempt_id", "ATT-000002"), ("learner_video_ref", "changed"), ("reference_profile_ref", "changed")])
def test_conflicting_payload_preserves_original_pair(wiring, submission, field, value):
    service, persistence, factory = wiring
    service.submit(submission)
    before = persistence.store.snapshot()
    changed = AnalysisSubmission.model_validate({**submission.model_dump(), field: value})
    with pytest.raises(IdempotencyConflict):
        service.submit(changed)
    assert persistence.store.snapshot() == before and factory.calls == 1


@pytest.mark.parametrize("begin_before_crash", [False, True])
def test_worker_crash_recovery_preserves_job_and_fences_old_worker(wiring, submission, make_result, begin_before_crash):
    service, persistence, factory = wiring
    first = submit_and_claim(wiring, submission)
    if begin_before_crash:
        service.begin(first, now_ms=11)
    before = service.get(first.analysis_id).job.model_dump_json()
    restored = FakePersistence(FakeStore.restore(persistence.store.snapshot()))
    restarted = AnalysisHandoffService(restored, factory)
    assert restarted.claim_next(now_ms=19, lease_duration_ms=10) is None
    second = restarted.claim_next(now_ms=20, lease_duration_ms=10)
    assert second.delivery_count == 2
    assert second.analysis_id == first.analysis_id and second.attempt_id == first.attempt_id
    assert restarted.get(first.analysis_id).job.model_dump_json() == before
    for operation in (
        lambda: restarted.begin(first, now_ms=21),
        lambda: restarted.complete(first, make_result(), now_ms=21),
        lambda: restored.begin(first, now_ms=21),
        lambda: restored.finish(first, make_result(), now_ms=21),
    ):
        with pytest.raises(ClaimConflict):
            operation()
    processing = restarted.begin(second, now_ms=21)
    assert processing.state == JobState.PROCESSING and processing.structured_analysis is None
    restarted.complete(second, make_result(), now_ms=22)
    assert len(restored.store.pairs) == 1


def test_one_active_worker_and_no_false_failure_on_delivery_recovery(wiring, submission):
    service, persistence, factory = wiring
    claim = submit_and_claim(wiring, submission)
    another_worker = AnalysisHandoffService(persistence, factory)
    assert another_worker.claim_next(now_ms=11, lease_duration_ms=10) is None
    processing = service.begin(claim, now_ms=11)
    retried = another_worker.claim_next(now_ms=20, lease_duration_ms=10)
    pair = service.get(claim.analysis_id)
    assert pair.job == processing and pair.job.structured_analysis is None
    assert pair.work.delivery_count == 2 and retried.analysis_id == claim.analysis_id
    assert service.begin(retried, now_ms=21) == processing


@pytest.mark.parametrize("status", ["COMPLETED", "UNANALYZABLE", "FAILED"])
def test_atomic_completion_closes_work_for_every_real_m2_outcome(wiring, submission, make_result, status):
    service, persistence, _ = wiring
    claim = submit_and_claim(wiring, submission)
    service.begin(claim, now_ms=11)
    result = make_result(status)
    terminal = service.complete(claim, result, now_ms=12)
    pair = service.get(claim.analysis_id)
    assert terminal.structured_analysis == result
    assert pair.job.state.value == status and pair.work.state == WorkState.COMPLETED
    assert pair.work.delivery_count == 1
    before = persistence.store.snapshot()
    assert service.claim_next(now_ms=100, lease_duration_ms=10) is None
    assert service.submit(submission) == terminal
    for operation in (
        lambda: service.begin(claim, now_ms=13),
        lambda: service.complete(claim, result, now_ms=13),
    ):
        with pytest.raises(ClaimConflict):
            operation()
    assert persistence.store.snapshot() == before


@pytest.mark.parametrize("stage", ["pending", "claimed", "processing", "terminal"])
def test_idempotency_returns_current_stage_without_redispatch(wiring, submission, make_result, stage):
    service, persistence, factory = wiring
    first = service.submit(submission)
    if stage != "pending":
        claimed = service.claim_next(now_ms=10, lease_duration_ms=10)
        if stage in ("processing", "terminal"):
            service.begin(claimed, now_ms=11)
        if stage == "terminal":
            service.complete(claimed, make_result(), now_ms=12)
    before = persistence.store.snapshot()
    assert service.submit(submission) == service.get(first.analysis_id).job
    assert persistence.store.snapshot() == before and factory.calls == 1


def test_invalid_results_and_completion_before_begin_do_not_commit(wiring, submission, make_result):
    service, persistence, _ = wiring
    claim = submit_and_claim(wiring, submission)
    with pytest.raises(InvalidTransition):
        service.complete(claim, make_result(), now_ms=11)
    service.begin(claim, now_ms=11)
    before = persistence.store.snapshot()
    for result in (
        make_result(analysis_id="AN-000002"), make_result(attempt_id="ATT-000002"),
        make_result().model_copy(update={"overall_score": None}),
    ):
        with pytest.raises(ValidationError):
            service.complete(claim, result, now_ms=12)
    assert persistence.store.snapshot() == before


def test_expired_claim_cannot_begin_or_finish_even_before_reclaim(wiring, submission, make_result):
    service, persistence, _ = wiring
    claimed = submit_and_claim(wiring, submission)
    service.begin(claimed, now_ms=11)
    before = persistence.store.snapshot()
    for operation in (
        lambda: service.begin(claimed, now_ms=20),
        lambda: service.complete(claimed, make_result(), now_ms=20),
    ):
        with pytest.raises(ClaimConflict):
            operation()
    assert persistence.store.snapshot() == before


def test_precommit_acceptance_failure_creates_neither_record(submission):
    class BrokenPersistence(FakePersistence):
        def before_commit(self, pair):
            raise RuntimeError("synthetic commit failure")
    persistence = BrokenPersistence()
    service = AnalysisHandoffService(persistence, FakeIdFactory())
    with pytest.raises(RuntimeError, match="commit failure"):
        service.submit(submission)
    assert not persistence.store.pairs
    assert persistence.get("AN-000001") is None


@pytest.mark.parametrize("operation", ["claim", "begin", "finish"])
def test_precommit_failures_never_leave_half_updated_pair(wiring, submission, make_result, operation):
    service, persistence, _ = wiring
    service.submit(submission)
    if operation != "claim":
        claimed = service.claim_next(now_ms=10, lease_duration_ms=10)
        if operation == "finish":
            service.begin(claimed, now_ms=11)
    before = persistence.store.snapshot()
    def fail(pair):
        raise RuntimeError("synthetic write error")
    persistence.before_commit = fail
    with pytest.raises(RuntimeError, match="write error"):
        if operation == "claim":
            service.claim_next(now_ms=10, lease_duration_ms=10)
        elif operation == "begin":
            service.begin(claimed, now_ms=11)
        else:
            service.complete(claimed, make_result(), now_ms=12)
    assert persistence.store.snapshot() == before


def test_id_collision_never_overwrites_prior_pair(wiring, submission):
    service, persistence, factory = wiring
    first = service.submit(submission)
    before = persistence.store.snapshot()
    factory.calls = 0
    second = AnalysisSubmission.model_validate({**submission.model_dump(), "idempotency_key": "new-key", "attempt_id": "ATT-000002"})
    with pytest.raises(AnalysisIdConflict):
        service.submit(second)
    assert persistence.store.snapshot() == before
    assert service.get(first.analysis_id).job == first


def test_racing_duplicate_acceptance_creates_one_work_item(submission):
    class RacingPersistence(FakePersistence):
        def accept(self, candidate):
            self.winner = super().accept(candidate)
            return super().accept(candidate)
    persistence = RacingPersistence()
    service = AnalysisHandoffService(persistence, FakeIdFactory())
    first = service.submit(submission)
    assert persistence.winner.created
    assert len(persistence.store.pairs) == 1
    assert service.get(first.analysis_id).work.delivery_count == 0


@pytest.mark.parametrize("field,value", [
    ("analysis_id", "AN-000002"), ("attempt_id", "ATT-000002"),
    ("learner_video_ref", "changed"), ("reference_profile_ref", "changed"),
])
def test_corrupted_stored_identity_rejected_not_repaired(wiring, submission, field, value):
    service, persistence, _ = wiring
    job = service.submit(submission)
    pair = persistence.store.pairs[job.analysis_id]
    corrupted = pair.model_copy(update={"work": pair.work.model_copy(update={field: value})})
    persistence.store.pairs[job.analysis_id] = corrupted
    for operation in (
        lambda: service.get(job.analysis_id), lambda: service.submit(submission),
        lambda: service.claim_next(now_ms=0, lease_duration_ms=10),
    ):
        with pytest.raises(ValidationError):
            operation()
    assert persistence.store.pairs[job.analysis_id] is corrupted


def test_missing_or_non_pair_state_cannot_be_acknowledged(wiring, submission):
    service, persistence, _ = wiring
    service.submit(submission)
    pair = persistence.store.pairs["AN-000001"]
    persistence.store.pairs["AN-000001"] = pair.model_copy(update={"work": None})
    with pytest.raises(ValidationError):
        service.submit(submission)
    with pytest.raises(ValidationError):
        service.claim_next(now_ms=0, lease_duration_ms=10)


def test_missing_and_invalid_identifiers(wiring):
    service, _, _ = wiring
    with pytest.raises(JobNotFound):
        service.get("AN-999999")
    with pytest.raises(ValidationError):
        service.get("AN-invalid")


@pytest.mark.parametrize("status", ["COMPLETED", "UNANALYZABLE", "FAILED"])
def test_later_coaching_uses_same_terminal_evidence_without_changing_work(wiring, submission, make_result, status):
    service, persistence, _ = wiring
    claim = submit_and_claim(wiring, submission)
    service.begin(claim, now_ms=11)
    result = make_result(status)
    terminal = service.complete(claim, result, now_ms=12)
    before = persistence.store.snapshot()
    feedback = generate_feedback(result, feedback_id="synthetic-feedback")
    records = FakeCoachingRepository()
    record = CoachingIntegrationService(persistence, records).record_feedback(claim.analysis_id, feedback)
    assert record.feedback == feedback and record.analysis_id == terminal.analysis_id
    assert persistence.store.snapshot() == before


def test_faulty_adapter_cannot_change_returned_lease_terms(wiring, submission):
    service, persistence, _ = wiring
    service.submit(submission)
    original_claim = persistence.claim_next
    def bad_claim(**kwargs):
        pair = original_claim(**kwargs)
        return JobWorkPair(job=pair.job, work=pair.work.model_copy(update={"lease_expires_at_ms": 1000}))
    persistence.claim_next = bad_claim
    with pytest.raises(AdapterContractError):
        service.claim_next(now_ms=10, lease_duration_ms=10)


def test_accept_then_crash_or_optional_wakeup_failure_keeps_discoverable_work(wiring, submission):
    service, persistence, factory = wiring
    with pytest.raises(RuntimeError, match="wakeup"):
        service.submit(submission)
        raise RuntimeError("synthetic optional wakeup failure after commit")
    restarted = AnalysisHandoffService(FakePersistence(FakeStore.restore(persistence.store.snapshot())), factory)
    assert restarted.submit(submission).analysis_id == "AN-000001"
    assert restarted.claim_next(now_ms=0, lease_duration_ms=10).delivery_count == 1


def test_atomic_port_cannot_accept_a_job_without_work(wiring, submission):
    service, persistence, _ = wiring
    job = service.submit(submission)
    snapshot = persistence.store.snapshot()
    broken = service.get(job.analysis_id).model_copy(update={"work": None})
    with pytest.raises(ValidationError):
        persistence.accept(broken)
    assert persistence.store.snapshot() == snapshot


def test_another_attempt_never_overwrites_completed_evidence(wiring, submission, make_result):
    service, persistence, _ = wiring
    first = submit_and_claim(wiring, submission)
    service.begin(first, now_ms=11)
    service.complete(first, make_result(), now_ms=12)
    original = service.get(first.analysis_id).model_dump_json()
    next_submission = AnalysisSubmission.model_validate({**submission.model_dump(),
        "attempt_id": "ATT-000002", "learner_video_ref": "synthetic-video-2", "idempotency_key": "new-key"})
    second = service.submit(next_submission)
    assert second.analysis_id == "AN-000002" and len(persistence.store.pairs) == 2
    assert service.get(first.analysis_id).model_dump_json() == original
    assert service.claim_next(now_ms=20, lease_duration_ms=10).analysis_id == second.analysis_id


def test_finish_commit_before_acknowledgement_loss_preserves_terminal_truth(wiring, submission, make_result):
    service, persistence, _ = wiring
    claimed = submit_and_claim(wiring, submission)
    service.begin(claimed, now_ms=11)
    finish = persistence.finish
    def lost_ack(expected, result, *, now_ms):
        finish(expected, result, now_ms=now_ms)
        raise RuntimeError("synthetic acknowledgement loss")
    persistence.finish = lost_ack
    with pytest.raises(RuntimeError, match="acknowledgement loss"):
        service.complete(claimed, make_result(), now_ms=12)
    pair = service.get(claimed.analysis_id)
    assert pair.job.state == JobState.COMPLETED and pair.work.state == WorkState.COMPLETED
    assert pair.job.structured_analysis == make_result()
    assert service.claim_next(now_ms=100, lease_duration_ms=10) is None
    assert service.submit(submission) == pair.job


@pytest.mark.parametrize("now,duration", [(-1, 10), (0, 0), (0, "10"), (0.5, 10)])
def test_invalid_claim_inputs_do_not_write(wiring, submission, now, duration):
    service, persistence, _ = wiring
    service.submit(submission)
    before = persistence.store.snapshot()
    with pytest.raises(ValidationError):
        service.claim_next(now_ms=now, lease_duration_ms=duration)
    assert persistence.store.snapshot() == before


def test_invalid_factory_and_unchecked_submission_are_rejected(submission):
    class BadFactory:
        def create(self):
            return "AN-uuid"
    persistence = FakePersistence()
    service = AnalysisHandoffService(persistence, BadFactory())
    with pytest.raises(ValidationError):
        service.submit(submission)
    with pytest.raises(ValidationError):
        service.submit(submission.model_copy(update={"learner_video_ref": ""}))
    assert not persistence.store.pairs
