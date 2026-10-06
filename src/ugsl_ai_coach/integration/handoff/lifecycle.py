"""Deterministic lease transitions using explicitly supplied trusted time values."""

from pydantic import TypeAdapter

from ugsl_ai_coach.domain.analysis import Timestamp
from ugsl_ai_coach.integration.errors import IntegrationError, InvalidTransition
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem, LeaseDuration, WorkState


class WorkUnavailable(IntegrationError):
    """Work is completed or already has an active claim."""


class ClaimConflict(IntegrationError):
    """The claim is stale, expired or incompatible with the authoritative clock."""


def validate_time(now_ms: int, lease_duration_ms: int | None = None) -> None:
    TypeAdapter(Timestamp).validate_python(now_ms)
    if lease_duration_ms is not None:
        TypeAdapter(LeaseDuration).validate_python(lease_duration_ms)


def claim_work(work: AnalysisWorkItem, *, now_ms: int, lease_duration_ms: int) -> AnalysisWorkItem:
    validate_time(now_ms, lease_duration_ms)
    work = AnalysisWorkItem.model_validate(work.model_dump(mode="python"))
    if work.state == WorkState.COMPLETED or (
        work.state == WorkState.CLAIMED and now_ms < work.lease_expires_at_ms
    ):
        raise WorkUnavailable("Work is not available for claim")
    return AnalysisWorkItem.model_validate({
        **work.model_dump(mode="python"), "state": WorkState.CLAIMED,
        "delivery_count": work.delivery_count + 1,
        "claimed_at_ms": now_ms, "lease_expires_at_ms": now_ms + lease_duration_ms,
    })


def validate_active_claim(
    current: AnalysisWorkItem, expected: AnalysisWorkItem, *, now_ms: int,
) -> None:
    validate_time(now_ms)
    current = AnalysisWorkItem.model_validate(current.model_dump(mode="python"))
    expected = AnalysisWorkItem.model_validate(expected.model_dump(mode="python"))
    if current != expected or current.state != WorkState.CLAIMED or not (
        current.claimed_at_ms <= now_ms < current.lease_expires_at_ms
    ):
        raise ClaimConflict("An exact current, unexpired claim is required")


def complete_work(work: AnalysisWorkItem, *, now_ms: int) -> AnalysisWorkItem:
    validate_active_claim(work, work, now_ms=now_ms)
    return AnalysisWorkItem.model_validate({
        **work.model_dump(mode="python"), "state": WorkState.COMPLETED,
        "claimed_at_ms": None, "lease_expires_at_ms": None,
    })


def validate_work_transition(
    previous: AnalysisWorkItem, replacement: AnalysisWorkItem, *, now_ms: int,
    lease_duration_ms: int | None = None,
) -> None:
    previous = AnalysisWorkItem.model_validate(previous.model_dump(mode="python"))
    replacement = AnalysisWorkItem.model_validate(replacement.model_dump(mode="python"))
    if replacement.state == WorkState.CLAIMED:
        if lease_duration_ms is None:
            raise InvalidTransition("Claim transition requires the requested lease duration")
        expected = claim_work(previous, now_ms=now_ms, lease_duration_ms=lease_duration_ms)
    elif replacement.state == WorkState.COMPLETED:
        expected = complete_work(previous, now_ms=now_ms)
    else:
        raise InvalidTransition("Work can only be claimed/reclaimed or completed")
    if replacement != expected:
        raise InvalidTransition("Work transition cannot change identity, references or delivery generation")
