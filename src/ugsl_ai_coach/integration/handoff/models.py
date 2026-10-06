"""Work delivery metadata is separate from analysis evidence and learner performance."""

from enum import StrEnum
from typing import Any, Annotated, Self

from pydantic import Field, model_validator

from ugsl_ai_coach.domain.analysis import AnalysisId, AttemptId, ContractModel, Timestamp
from ugsl_ai_coach.integration.models import AnalysisJob, JobState, OpaqueReference

LeaseDuration = Annotated[int, Field(strict=True, gt=0)]


class WorkState(StrEnum):
    PENDING = "PENDING"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"


class AnalysisWorkItem(ContractModel):
    analysis_id: AnalysisId
    attempt_id: AttemptId
    learner_video_ref: OpaqueReference
    reference_profile_ref: OpaqueReference
    state: WorkState
    # Also a fencing generation: old workers cannot complete a newer delivery.
    delivery_count: int = Field(strict=True, ge=0)
    claimed_at_ms: Timestamp | None = None
    lease_expires_at_ms: Timestamp | None = None

    @model_validator(mode="after")
    def consistent_lease(self) -> Self:
        if self.state == WorkState.CLAIMED:
            if self.delivery_count < 1 or self.claimed_at_ms is None or self.lease_expires_at_ms is None:
                raise ValueError("CLAIMED work requires delivery_count and lease timestamps")
            if self.lease_expires_at_ms <= self.claimed_at_ms:
                raise ValueError("Lease expiry must follow claim time")
        else:
            if self.claimed_at_ms is not None or self.lease_expires_at_ms is not None:
                raise ValueError("Only CLAIMED work has active lease timestamps")
            if self.state == WorkState.COMPLETED and self.delivery_count < 1:
                raise ValueError("COMPLETED work must have been delivered")
        return self


class JobWorkPair(ContractModel):
    job: AnalysisJob
    work: AnalysisWorkItem

    @model_validator(mode="before")
    @classmethod
    def revalidate_nested(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = data.copy()
            for field, model in (("job", AnalysisJob), ("work", AnalysisWorkItem)):
                if isinstance(data.get(field), model):
                    data[field] = data[field].model_dump(mode="python")
        return data

    @model_validator(mode="after")
    def consistent_pair(self) -> Self:
        for field in ("analysis_id", "attempt_id", "learner_video_ref", "reference_profile_ref"):
            if getattr(self.job, field) != getattr(self.work, field):
                raise ValueError("Job/work identity and media references must match")
        if self.job.state.is_terminal != (self.work.state == WorkState.COMPLETED):
            raise ValueError("Terminal analysis and completed work must be committed together")
        if self.job.state == JobState.PROCESSING and self.work.delivery_count == 0:
            raise ValueError("Processing analysis requires previously delivered work")
        return self


class WorkAcceptance(ContractModel):
    pair: JobWorkPair
    created: bool = Field(strict=True)

    @model_validator(mode="before")
    @classmethod
    def revalidate_pair(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("pair"), JobWorkPair):
            return {**data, "pair": data["pair"].model_dump(mode="python")}
        return data

    @model_validator(mode="after")
    def new_acceptance_has_pending_work(self) -> Self:
        if self.created and (
            self.pair.job.state != JobState.SUBMITTED or
            self.pair.work.state != WorkState.PENDING or self.pair.work.delivery_count != 0
        ):
            raise ValueError("New acceptance requires SUBMITTED job and undelivered PENDING work")
        return self
