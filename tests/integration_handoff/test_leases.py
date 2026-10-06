import pytest
from pydantic import ValidationError

from ugsl_ai_coach.integration.errors import InvalidTransition
from ugsl_ai_coach.integration.handoff.lifecycle import (
    ClaimConflict, WorkUnavailable, claim_work, complete_work, validate_active_claim, validate_work_transition,
)


def test_claim_reclaim_and_fencing_generation(pending):
    first = claim_work(pending, now_ms=10, lease_duration_ms=10)
    with pytest.raises(WorkUnavailable):
        claim_work(first, now_ms=19, lease_duration_ms=10)
    second = claim_work(first, now_ms=20, lease_duration_ms=10)
    assert second.delivery_count == 2 and first.delivery_count == 1 and pending.delivery_count == 0
    for field in ("analysis_id", "attempt_id", "learner_video_ref", "reference_profile_ref"):
        assert getattr(second, field) == getattr(first, field) == getattr(pending, field)
    with pytest.raises(ClaimConflict):
        validate_active_claim(second, first, now_ms=21)
    with pytest.raises(ClaimConflict):
        complete_work(first, now_ms=20)
    validate_work_transition(first, second, now_ms=20, lease_duration_ms=10)


@pytest.mark.parametrize("now,duration", [(-1, 10), (0.5, 10), ("0", 10), (True, 10), (0, 0), (0, -1), (0, 0.5), (0, "10")])
def test_invalid_clock_and_duration(pending, now, duration):
    with pytest.raises(ValidationError):
        claim_work(pending, now_ms=now, lease_duration_ms=duration)


def test_complete_and_closed_history(pending):
    claimed = claim_work(pending, now_ms=10, lease_duration_ms=10)
    completed = complete_work(claimed, now_ms=19)
    validate_work_transition(claimed, completed, now_ms=19)
    assert completed.delivery_count == 1
    assert completed.claimed_at_ms is None and completed.lease_expires_at_ms is None
    with pytest.raises(WorkUnavailable):
        claim_work(completed, now_ms=100, lease_duration_ms=10)
    with pytest.raises(ClaimConflict):
        complete_work(completed, now_ms=100)
    with pytest.raises(ClaimConflict):
        complete_work(pending, now_ms=10)
    with pytest.raises(ClaimConflict):
        validate_active_claim(claimed, claimed, now_ms=9)


@pytest.mark.parametrize("field,value", [
    ("analysis_id", "AN-000002"), ("attempt_id", "ATT-000002"),
    ("learner_video_ref", "changed-video"), ("reference_profile_ref", "changed-reference"),
    ("delivery_count", 2), ("claimed_at_ms", 9), ("lease_expires_at_ms", 21),
])
def test_transition_rejects_changed_identity_generation_or_lease(pending, field, value):
    replacement = claim_work(pending, now_ms=10, lease_duration_ms=10).model_copy(update={field: value})
    with pytest.raises(InvalidTransition):
        validate_work_transition(pending, replacement, now_ms=10, lease_duration_ms=10)


def test_invalid_transition_to_pending_and_unchecked_model(pending):
    claimed = claim_work(pending, now_ms=0, lease_duration_ms=10)
    with pytest.raises(InvalidTransition):
        validate_work_transition(claimed, pending, now_ms=11)
    with pytest.raises(ValidationError):
        claim_work(pending.model_copy(update={"delivery_count": -1}), now_ms=0, lease_duration_ms=10)
