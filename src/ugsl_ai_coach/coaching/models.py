"""Frozen coaching contracts. Semantic authorization belongs to validation.py."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from ugsl_ai_coach.domain.analysis import (
    AnalysisId, AnalysisStatus, AttemptId, BodyRegion, ContractModel,
    FindingId, FindingStatus, NormalizedValue, SkillCategory, Timestamp,
)

NonEmpty = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1)]
Movement = Literal[SkillCategory.MOVEMENT]


class ActionKind(StrEnum):
    REVIEW_MOVEMENT = "REVIEW_MOVEMENT"
    RETRY_ATTEMPT = "RETRY_ATTEMPT"
    RETRY_CAPTURE = "RETRY_CAPTURE"
    TRY_LATER = "TRY_LATER"


class RecommendedAction(ContractModel):
    kind: ActionKind
    message: NonEmpty


class CoachingPoint(ContractModel):
    finding_id: FindingId
    skill: Movement
    body_region: BodyRegion
    start_time_ms: Timestamp
    end_time_ms: Timestamp
    message: NonEmpty
    reason: NonEmpty

    @model_validator(mode="after")
    def ordered_interval(self) -> Self:
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must be >= start_time_ms")
        return self


class AuthorizedFact(ContractModel):
    """Safe status-derived explanation plus unchanged source prioritization data."""

    point: CoachingPoint
    source_status: FindingStatus
    severity: int = Field(strict=True, ge=0, le=3)
    confidence: NormalizedValue


class VisualAnnotation(ContractModel):
    finding_id: FindingId
    skill: Movement
    body_region: BodyRegion
    start_time_ms: Timestamp
    end_time_ms: Timestamp
    label: NonEmpty

    @model_validator(mode="after")
    def ordered_interval(self) -> Self:
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must be >= start_time_ms")
        return self


class ProviderMetadata(ContractModel):
    provider_type: NonEmpty
    policy_version: NonEmpty
    template_version: NonEmpty


class CoachingContext(ContractModel):
    """Trusted grounding output; providers cannot author or amend this context."""

    feedback_id: NonEmpty
    attempt_id: AttemptId
    analysis_id: AnalysisId
    overall_status: AnalysisStatus
    evidence_confidence: NormalizedValue | None
    strengths: tuple[AuthorizedFact, ...]
    corrections: tuple[AuthorizedFact, ...]
    observations: tuple[AuthorizedFact, ...]
    summary: NonEmpty
    recommended_action: RecommendedAction
    encouragement: NonEmpty
    policy_version: NonEmpty
    template_version: NonEmpty
    constraints: tuple[NonEmpty, ...]


class CoachingFeedback(ContractModel):
    feedback_id: NonEmpty
    attempt_id: AttemptId
    analysis_id: AnalysisId
    overall_status: AnalysisStatus
    summary: NonEmpty
    strengths: tuple[CoachingPoint, ...]
    corrections: tuple[CoachingPoint, ...]
    observations: tuple[CoachingPoint, ...]
    recommended_action: RecommendedAction
    encouragement: NonEmpty
    audio_text: NonEmpty | None = None
    visual_annotations: tuple[VisualAnnotation, ...]
    provider_metadata: ProviderMetadata


def accessible_text(feedback: CoachingFeedback) -> str:
    """Complete reading order; annotations repeat these points, never add facts."""
    parts = [feedback.summary]
    for point in (*feedback.corrections, *feedback.strengths, *feedback.observations):
        parts.extend((point.message, point.reason))
    parts.extend((feedback.recommended_action.message, feedback.encouragement))
    return "\n".join(parts)
