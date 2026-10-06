"""Immutable job contracts reusing M2 and M5 without changing their semantics."""

from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import Field, StringConstraints, model_validator

from ugsl_ai_coach.coaching.models import CoachingFeedback
from ugsl_ai_coach.domain.analysis import (
    AnalysisId, AttemptId, ContractModel, StructuredAnalysisResult,
)

OpaqueReference = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]


class JobState(StrEnum):
    SUBMITTED = "SUBMITTED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    UNANALYZABLE = "UNANALYZABLE"
    FAILED = "FAILED"

    @property
    def is_terminal(self) -> bool:
        return self in (self.COMPLETED, self.UNANALYZABLE, self.FAILED)


class AnalysisSubmission(ContractModel):
    attempt_id: AttemptId
    learner_video_ref: OpaqueReference
    reference_profile_ref: OpaqueReference
    idempotency_key: OpaqueReference


class AnalysisJob(ContractModel):
    """Analysis-stage lifecycle only; terminal does not imply downstream completion."""

    analysis_id: AnalysisId
    attempt_id: AttemptId
    state: JobState
    learner_video_ref: OpaqueReference
    reference_profile_ref: OpaqueReference
    idempotency_key: OpaqueReference
    structured_analysis: StructuredAnalysisResult | None = None

    @model_validator(mode="before")
    @classmethod
    def revalidate_nested_instances(cls, data: Any) -> Any:
        # Frozen instances can still originate from unchecked model_copy.
        if isinstance(data, dict):
            data = data.copy()
            for field, model in (
                ("structured_analysis", StructuredAnalysisResult),
            ):
                if isinstance(data.get(field), model):
                    data[field] = data[field].model_dump(mode="python")
        return data

    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        analysis = self.structured_analysis
        if not self.state.is_terminal:
            if analysis is not None:
                raise ValueError("Non-terminal jobs cannot contain analysis results")
            return self
        if analysis is None:
            raise ValueError("Terminal jobs require a structured analysis result")
        if analysis.status.value != self.state.value:
            raise ValueError("Job state must match the terminal analysis status")
        if analysis.analysis_id != self.analysis_id or analysis.attempt_id != self.attempt_id:
            raise ValueError("Analysis identifiers must match the job")
        return self

    @property
    def submission(self) -> AnalysisSubmission:
        """Canonical logical payload retained through every lifecycle state."""
        return AnalysisSubmission(
            attempt_id=self.attempt_id, learner_video_ref=self.learner_video_ref,
            reference_profile_ref=self.reference_profile_ref, idempotency_key=self.idempotency_key,
        )


class JobAcceptance(ContractModel):
    job: AnalysisJob
    created: bool = Field(strict=True)

    @model_validator(mode="after")
    def new_jobs_are_submitted(self) -> Self:
        if self.created and self.job.state != JobState.SUBMITTED:
            raise ValueError("Newly accepted jobs must be SUBMITTED")
        return self


class CoachingRecord(ContractModel):
    """One immutable downstream artifact per analysis; source validation is service-owned."""

    analysis_id: AnalysisId
    attempt_id: AttemptId
    feedback: CoachingFeedback

    @model_validator(mode="before")
    @classmethod
    def revalidate_feedback(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("feedback"), CoachingFeedback):
            data = {**data, "feedback": data["feedback"].model_dump(mode="python")}
        return data

    @model_validator(mode="after")
    def matching_identifiers(self) -> Self:
        if self.analysis_id != self.feedback.analysis_id or self.attempt_id != self.feedback.attempt_id:
            raise ValueError("Coaching record identifiers must match the feedback")
        return self
