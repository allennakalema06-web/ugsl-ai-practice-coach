import pytest
from pydantic import ValidationError

from ugsl_ai_coach.domain.analysis import AnalysisStatus
from ugsl_ai_coach.integration.models import AnalysisJob, AnalysisSubmission, JobAcceptance, JobState


def test_valid_submission_is_trimmed_and_opaque(submission):
    values = submission.model_dump()
    for field in ("learner_video_ref", "reference_profile_ref", "idempotency_key"):
        values[field] = "  arbitrary opaque value  "
    validated = AnalysisSubmission(**values)
    assert validated.learner_video_ref == "arbitrary opaque value"
    assert validated.reference_profile_ref == "arbitrary opaque value"
    assert validated.idempotency_key == "arbitrary opaque value"


@pytest.mark.parametrize("field,value", [
    ("attempt_id", ""), ("attempt_id", "ATT-1"), ("attempt_id", " ATT-000001 "),
    ("attempt_id", 123), ("learner_video_ref", ""), ("learner_video_ref", " \t\n"),
    ("learner_video_ref", b"bytes"), ("learner_video_ref", 3),
    ("reference_profile_ref", ""), ("reference_profile_ref", "  "),
    ("reference_profile_ref", ["profile"]), ("idempotency_key", ""),
    ("idempotency_key", "  "), ("idempotency_key", 3), ("extra", "forbidden"),
])
def test_invalid_submission_rejected(submission, field, value):
    with pytest.raises(ValidationError):
        AnalysisSubmission.model_validate({**submission.model_dump(), field: value})


@pytest.mark.parametrize("state", [JobState.SUBMITTED, JobState.PROCESSING])
def test_non_terminal_has_no_results(submitted, state):
    job = AnalysisJob.model_validate({**submitted.model_dump(), "state": state})
    assert job.structured_analysis is None
    assert "coaching_feedback" not in type(job).model_fields


@pytest.mark.parametrize("state", [JobState.SUBMITTED, JobState.PROCESSING])
def test_non_terminal_rejects_results(submitted, make_result, state):
    result = make_result()
    values = {**submitted.model_dump(), "state": state, "structured_analysis": result}
    with pytest.raises(ValidationError, match="Non-terminal"):
        AnalysisJob.model_validate(values)


@pytest.mark.parametrize("state", [s for s in JobState if s.is_terminal])
def test_terminal_reuses_matching_m2_without_coaching(submitted, make_result, state):
    result = make_result(state.value)
    job = AnalysisJob.model_validate({**submitted.model_dump(), "state": state,
        "structured_analysis": result})
    assert job.structured_analysis == result
    assert "coaching_feedback" not in job.model_dump()
    assert AnalysisJob.model_validate_json(job.model_dump_json()) == job


@pytest.mark.parametrize("state", [s for s in JobState if s.is_terminal])
def test_terminal_requires_analysis(submitted, state):
    with pytest.raises(ValidationError, match="require"):
        AnalysisJob.model_validate({**submitted.model_dump(), "state": state})


@pytest.mark.parametrize("state,result_status", [
    (job, result.value) for job in JobState if job.is_terminal
    for result in AnalysisStatus if job.value != result.value
])
def test_mismatched_terminal_status_rejected(submitted, make_result, state, result_status):
    with pytest.raises(ValidationError, match="status"):
        AnalysisJob.model_validate({**submitted.model_dump(), "state": state,
                                   "structured_analysis": make_result(result_status)})


@pytest.mark.parametrize("field,value", [("analysis_id", "AN-000002"), ("attempt_id", "ATT-000002")])
def test_analysis_identity_mismatch_rejected(submitted, make_result, field, value):
    result = make_result(**{field: value})
    with pytest.raises(ValidationError, match="identifiers"):
        AnalysisJob.model_validate({**submitted.model_dump(), "state": "COMPLETED", "structured_analysis": result})


def test_unchecked_nested_analysis_revalidated(submitted, make_result):
    result = make_result()
    with pytest.raises(ValidationError):
        AnalysisJob.model_validate({**submitted.model_dump(), "state": "COMPLETED",
            "structured_analysis": result.model_copy(update={"overall_score": None})})


@pytest.mark.parametrize("field,value", [
    ("analysis_id", "AN-1"), ("analysis_id", ""), ("state", "RETRYING"),
    ("learner_video_ref", " "), ("reference_profile_ref", " "),
    ("idempotency_key", " "), ("extra", True),
])
def test_invalid_job_fields(submitted, field, value):
    with pytest.raises(ValidationError):
        AnalysisJob.model_validate({**submitted.model_dump(), field: value})


def test_frozen_models_and_acceptance_constraints(submission, submitted):
    acceptance = JobAcceptance(job=submitted, created=True)
    for model in (submission, submitted, acceptance):
        field = next(iter(type(model).model_fields))
        with pytest.raises(ValidationError):
            setattr(model, field, getattr(model, field))
    with pytest.raises(ValidationError):
        JobAcceptance(job=submitted, created="true")
    with pytest.raises(ValidationError):
        JobAcceptance(job=submitted, created=True, extra=1)
    processing = AnalysisJob.model_validate({**submitted.model_dump(), "state": "PROCESSING"})
    with pytest.raises(ValidationError):
        JobAcceptance(job=processing, created=True)


def test_m2_statuses_unchanged():
    assert set(AnalysisStatus) == {AnalysisStatus.COMPLETED, AnalysisStatus.UNANALYZABLE, AnalysisStatus.FAILED}
    assert not JobState.SUBMITTED.is_terminal and not JobState.PROCESSING.is_terminal
