import pytest
from pydantic import ValidationError

from ugsl_ai_coach.integration.handoff.lifecycle import claim_work, complete_work
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem, JobWorkPair, WorkAcceptance
from ugsl_ai_coach.integration.models import AnalysisJob, JobState


@pytest.mark.parametrize("field,value", [
    ("analysis_id", ""), ("analysis_id", "AN-1"), ("attempt_id", "ATT-1"),
    ("attempt_id", ""), ("learner_video_ref", " "), ("learner_video_ref", b"media"),
    ("reference_profile_ref", ""), ("reference_profile_ref", 1),
    ("delivery_count", -1), ("delivery_count", 0.5), ("delivery_count", "0"),
    ("delivery_count", True), ("claimed_at_ms", -1), ("claimed_at_ms", 0.5),
    ("lease_expires_at_ms", "123"), ("state", "FAILED"), ("extra", 1),
    ("structured_analysis", {}), ("coaching_feedback", {}),
])
def test_invalid_work_contract(pending, field, value):
    with pytest.raises(ValidationError):
        AnalysisWorkItem.model_validate({**pending.model_dump(), field: value})


@pytest.mark.parametrize("changes", [
    {"state": "CLAIMED"},
    {"state": "CLAIMED", "delivery_count": 1, "claimed_at_ms": 1},
    {"state": "CLAIMED", "delivery_count": 0, "claimed_at_ms": 1, "lease_expires_at_ms": 2},
    {"state": "CLAIMED", "delivery_count": 1, "claimed_at_ms": 2, "lease_expires_at_ms": 1},
    {"state": "CLAIMED", "delivery_count": 1, "claimed_at_ms": 1, "lease_expires_at_ms": 1},
    {"claimed_at_ms": 1, "lease_expires_at_ms": 2},
    {"state": "COMPLETED", "delivery_count": 0},
    {"state": "COMPLETED", "delivery_count": 1, "claimed_at_ms": 1},
])
def test_lease_state_invariants(pending, changes):
    with pytest.raises(ValidationError):
        AnalysisWorkItem.model_validate({**pending.model_dump(), **changes})


def test_valid_models_frozen_and_round_trip(pending, submission):
    claimed = claim_work(pending, now_ms=10, lease_duration_ms=100)
    completed = complete_work(claimed, now_ms=11)
    job = AnalysisJob(analysis_id=pending.analysis_id, state=JobState.SUBMITTED, **submission.model_dump())
    pair = JobWorkPair(job=job, work=pending)
    acceptance = WorkAcceptance(pair=pair, created=True)
    for obj in (pending, claimed, completed, pair, acceptance):
        assert type(obj).model_validate_json(obj.model_dump_json()) == obj
        field = next(iter(type(obj).model_fields))
        with pytest.raises(ValidationError):
            setattr(obj, field, getattr(obj, field))
        with pytest.raises(ValidationError):
            type(obj).model_validate({**obj.model_dump(), "extra": 1})
    assert pending.delivery_count == 0 and completed.delivery_count == 1


@pytest.mark.parametrize("field,value", [
    ("analysis_id", "AN-000002"), ("attempt_id", "ATT-000002"),
    ("learner_video_ref", "changed-video"), ("reference_profile_ref", "changed-reference"),
])
def test_pair_identity_mismatch(pending, submission, field, value):
    job = AnalysisJob(analysis_id=pending.analysis_id, state="SUBMITTED", **submission.model_dump())
    work = pending.model_copy(update={field: value})
    with pytest.raises(ValidationError, match="match"):
        JobWorkPair(job=job, work=work)


@pytest.mark.parametrize("case", ["terminal_pending", "nonterminal_completed", "processing_undelivered"])
def test_pair_stage_inconsistency(pending, submission, make_result, case):
    job = AnalysisJob(analysis_id=pending.analysis_id, state="SUBMITTED", **submission.model_dump())
    work = pending
    if case == "terminal_pending":
        job = AnalysisJob.model_validate({**job.model_dump(), "state": "COMPLETED", "structured_analysis": make_result()})
    elif case == "nonterminal_completed":
        work = complete_work(claim_work(pending, now_ms=0, lease_duration_ms=10), now_ms=1)
    else:
        job = AnalysisJob.model_validate({**job.model_dump(), "state": "PROCESSING"})
    with pytest.raises(ValidationError):
        JobWorkPair(job=job, work=work)


def test_nested_unchecked_values_and_false_new_acceptance_rejected(pending, submission):
    job = AnalysisJob(analysis_id=pending.analysis_id, state="SUBMITTED", **submission.model_dump())
    with pytest.raises(ValidationError):
        JobWorkPair(job=job, work=pending.model_copy(update={"delivery_count": -1}))
    pair = JobWorkPair(job=job, work=claim_work(pending, now_ms=0, lease_duration_ms=10))
    with pytest.raises(ValidationError):
        WorkAcceptance(pair=pair, created=True)
    with pytest.raises(ValidationError):
        WorkAcceptance(pair=pair, created="false")
