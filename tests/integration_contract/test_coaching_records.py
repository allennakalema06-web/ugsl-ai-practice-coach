import pytest
from pydantic import ValidationError

from ugsl_ai_coach.domain.analysis import AnalysisStatus
from ugsl_ai_coach.integration.errors import (
    AdapterContractError, AnalysisNotTerminal, FeedbackConflict, JobNotFound,
)
from ugsl_ai_coach.integration.models import AnalysisJob, CoachingRecord, JobState
from ugsl_ai_coach.integration.service import AnalysisIntegrationService, CoachingIntegrationService
from .fakes import FakeCoachingRepository, FakeDispatcher, FakeIdFactory, FakeRepository


@pytest.fixture
def wiring():
    jobs, records = FakeRepository(), FakeCoachingRepository()
    analysis = AnalysisIntegrationService(jobs, FakeDispatcher(), FakeIdFactory())
    coaching = CoachingIntegrationService(jobs, records)
    return analysis, coaching, jobs, records


@pytest.mark.parametrize("state", [s for s in JobState if s.is_terminal])
def test_analysis_finishes_before_later_coaching_without_any_evidence_mutation(
    wiring, submission, make_result, make_feedback, state,
):
    analysis, coaching, jobs, records = wiring
    submitted = analysis.submit(submission)
    analysis.mark_processing(submitted.analysis_id)
    result = make_result(state.value)
    terminal = analysis.complete(submitted.analysis_id, result)
    terminal = jobs.get(terminal.analysis_id)
    assert records.get(terminal.analysis_id) is None
    before_job = terminal.model_dump_json()
    before_result = terminal.structured_analysis.model_dump_json()
    source_json = result.model_dump_json()
    record = coaching.record_feedback(terminal.analysis_id, make_feedback(result))
    assert record.analysis_id == terminal.analysis_id == record.feedback.analysis_id
    assert record.attempt_id == terminal.attempt_id == record.feedback.attempt_id
    assert records.get(record.analysis_id) == record
    assert jobs.get(terminal.analysis_id) is terminal
    assert jobs.get(terminal.analysis_id).model_dump_json() == before_job
    assert terminal.structured_analysis.model_dump_json() == before_result
    assert result.model_dump_json() == source_json


@pytest.mark.parametrize("state", [JobState.SUBMITTED, JobState.PROCESSING])
def test_feedback_before_terminal_evidence_is_rejected(wiring, submission, make_result, make_feedback, state):
    analysis, coaching, jobs, records = wiring
    job = analysis.submit(submission)
    if state == JobState.PROCESSING:
        job = analysis.mark_processing(job.analysis_id)
    with pytest.raises(AnalysisNotTerminal):
        coaching.record_feedback(job.analysis_id, make_feedback(make_result()))
    assert not records.records
    assert jobs.get(job.analysis_id) == job


def test_feedback_for_missing_analysis_rejected(wiring, make_result, make_feedback):
    _, coaching, _, records = wiring
    with pytest.raises(JobNotFound):
        coaching.record_feedback("AN-000001", make_feedback(make_result()))
    assert not records.records


@pytest.mark.parametrize("field,value", [
    ("analysis_id", "AN-000002"), ("attempt_id", "ATT-000002"),
    ("overall_status", AnalysisStatus.FAILED), ("summary", "You failed."),
    ("audio_text", "Unique audio-only information."),
])
def test_feedback_traceability_grounding_and_accessibility_rejected(
    wiring, submission, make_result, make_feedback, field, value,
):
    analysis, coaching, jobs, records = wiring
    job = analysis.submit(submission)
    analysis.mark_processing(job.analysis_id)
    result = make_result()
    terminal = analysis.complete(job.analysis_id, result)
    feedback = make_feedback(result).model_copy(update={field: value})
    with pytest.raises(ValueError):
        coaching.record_feedback(job.analysis_id, feedback)
    assert not records.records and jobs.get(job.analysis_id) == terminal


def test_identical_feedback_is_idempotent_and_conflicting_feedback_never_overwrites(
    wiring, submission, make_result, make_feedback,
):
    analysis, coaching, jobs, records = wiring
    job = analysis.submit(submission)
    analysis.mark_processing(job.analysis_id)
    result = make_result()
    terminal = analysis.complete(job.analysis_id, result)
    feedback = make_feedback(result)
    terminal = jobs.get(job.analysis_id)
    first = coaching.record_feedback(job.analysis_id, feedback)
    stored_record = records.get(job.analysis_id)
    assert coaching.record_feedback(job.analysis_id, feedback) == first
    assert records.get(job.analysis_id) is stored_record
    # Both outputs pass M5 validation, but a changed artifact ID is conflicting output.
    changed = feedback.model_copy(update={"feedback_id": "different-feedback"})
    with pytest.raises(FeedbackConflict):
        coaching.record_feedback(job.analysis_id, changed)
    assert len(records.records) == 1 and records.get(job.analysis_id) == first
    assert jobs.get(job.analysis_id) is terminal


@pytest.mark.parametrize("field,value", [
    ("analysis_id", "AN-000002"), ("attempt_id", "ATT-000002"),
    ("analysis_id", ""), ("attempt_id", ""), ("extra", 1),
])
def test_record_contract_rejects_mismatching_invalid_ids_and_extra_fields(make_result, make_feedback, field, value):
    feedback = make_feedback(make_result())
    values = dict(analysis_id=feedback.analysis_id, attempt_id=feedback.attempt_id, feedback=feedback)
    values[field] = value
    with pytest.raises(ValidationError):
        CoachingRecord(**values)


def test_record_and_nested_feedback_are_frozen_and_revalidated(make_result, make_feedback):
    feedback = make_feedback(make_result())
    record = CoachingRecord(analysis_id=feedback.analysis_id, attempt_id=feedback.attempt_id, feedback=feedback)
    for obj in (record, record.feedback, record.feedback.recommended_action):
        field = next(iter(type(obj).model_fields))
        with pytest.raises(ValidationError):
            setattr(obj, field, getattr(obj, field))
    assert CoachingRecord.model_validate_json(record.model_dump_json()) == record
    with pytest.raises(ValidationError):
        CoachingRecord(analysis_id=record.analysis_id, attempt_id=record.attempt_id,
                       feedback=feedback.model_copy(update={"summary": ""}))


@pytest.mark.parametrize("state", list(JobState))
def test_feedback_is_rejected_as_an_analysis_job_field(submitted, make_result, make_feedback, state):
    values = {**submitted.model_dump(), "state": state, "coaching_feedback": make_feedback(make_result())}
    if state.is_terminal:
        values["structured_analysis"] = make_result(state.value)
    with pytest.raises(ValidationError, match="Extra inputs"):
        AnalysisJob.model_validate(values)


def test_coaching_storage_error_propagates_without_mutating_analysis(wiring, submission, make_result, make_feedback):
    analysis, _, jobs, _ = wiring
    job = analysis.submit(submission)
    analysis.mark_processing(job.analysis_id)
    result = make_result()
    terminal = analysis.complete(job.analysis_id, result)
    class BrokenRecords:
        def append(self, record):
            raise RuntimeError("synthetic persistence error")
    with pytest.raises(RuntimeError, match="persistence error"):
        CoachingIntegrationService(jobs, BrokenRecords()).record_feedback(job.analysis_id, make_feedback(result))
    assert jobs.get(job.analysis_id) == terminal


def test_invalid_adapter_response_rejected(wiring, submission, make_result, make_feedback):
    analysis, _, jobs, _ = wiring
    job = analysis.submit(submission)
    analysis.mark_processing(job.analysis_id)
    result = make_result()
    terminal = analysis.complete(job.analysis_id, result)
    class BadRecords:
        def append(self, record):
            feedback = record.feedback.model_copy(update={"feedback_id": "different"})
            return CoachingRecord(analysis_id=record.analysis_id, attempt_id=record.attempt_id, feedback=feedback)
    with pytest.raises(AdapterContractError):
        CoachingIntegrationService(jobs, BadRecords()).record_feedback(job.analysis_id, make_feedback(result))
    assert jobs.get(job.analysis_id) == terminal
