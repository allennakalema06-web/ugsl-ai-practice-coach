"""M6E operational delivery contracts, separate from M2 analysis status."""

from datetime import datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from ugsl_ai_coach.domain.analysis import AnalysisId, AttemptId, ContractModel
from ugsl_ai_coach.integration.models import OpaqueReference


class CoachingWorkState(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"


class CoachingWork(ContractModel):
    analysis_id: AnalysisId
    attempt_id: AttemptId
    feedback_id: OpaqueReference
    state: CoachingWorkState
    delivery_count: int = Field(ge=0, strict=True)
    failure_count: int = Field(ge=0, strict=True)
    claimed_at: datetime | None
    lease_expires_at: datetime | None
    next_attempt_at: datetime
    last_error_code: str | None = None

    @model_validator(mode="after")
    def lease_shape(self) -> Self:
        times = (self.next_attempt_at, self.claimed_at, self.lease_expires_at)
        if any(t is not None and t.utcoffset() is None for t in times):
            raise ValueError("Coaching timestamps must be timezone aware")
        if self.last_error_code not in (None, "COACHING_PROCESSING_ERROR"):
            raise ValueError("Unsafe operational error code")
        if self.state == CoachingWorkState.CLAIMED:
            if (not self.delivery_count or self.claimed_at is None or self.lease_expires_at is None
                    or self.lease_expires_at <= self.claimed_at):
                raise ValueError("Claim requires a valid lease and generation")
        elif self.claimed_at is not None or self.lease_expires_at is not None:
            raise ValueError("Unclaimed work cannot retain lease ownership")
        return self


def retry_seconds(failures: int, base: int, maximum: int) -> int:
    if type(failures) is not int or failures < 0 or not 1 <= base <= maximum <= 86400:
        raise ValueError("Invalid retry bounds")
    return min(maximum, base * (2 ** min(failures, 17)))
