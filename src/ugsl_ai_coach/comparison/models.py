"""Frozen request, evidence, alignment, and comparison artifacts."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ugsl_ai_coach.cv.models import (
    ExtractionResult, Finite, HandLabel, NonNegativeInt, NormalizationConfig, Probability, SamplingConfig, VideoMetadata,
)
from ugsl_ai_coach.domain.analysis import AnalysisId, AttemptId, BodyRegion, FindingId, StructuredAnalysisResult

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ComparisonModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HandCorrespondence(ComparisonModel):
    """Caller-established pairing and anatomical region, never inferred here."""

    reference_label: HandLabel
    learner_label: HandLabel
    body_region: Literal[BodyRegion.LEFT_HAND, BodyRegion.RIGHT_HAND]
    establishment_basis: NonEmpty


class ComparisonRequest(ComparisonModel):
    analysis_id: AnalysisId
    attempt_id: AttemptId
    finding_id: FindingId = "F-001"
    reference_id: NonEmpty
    correspondence: HandCorrespondence | None = None


class ExtractionContext(ComparisonModel):
    """Lightweight M3 provenance without duplicating video pixels/landmark arrays."""

    video: VideoMetadata
    sampling: SamplingConfig
    normalization_config: NormalizationConfig
    coordinate_convention: Literal["unmirrored_image_xy_model_reported_handedness"]

    @classmethod
    def from_extraction(cls, extraction: ExtractionResult) -> Self:
        return cls(video=extraction.video, sampling=extraction.sampling,
                   normalization_config=extraction.normalization_config,
                   coordinate_convention=extraction.coordinate_convention)


class Availability(StrEnum):
    AVAILABLE = "AVAILABLE"
    MISSING_HAND = "MISSING_HAND"
    AMBIGUOUS_HAND = "AMBIGUOUS_HAND"
    UNNORMALIZED = "UNNORMALIZED"
    LOW_LABEL_CONFIDENCE = "LOW_LABEL_CONFIDENCE"
    MISSING_WRIST = "MISSING_WRIST"


class TrajectoryObservation(ComparisonModel):
    frame_index: NonNegativeInt
    timestamp_ms: Finite = Field(ge=0)
    availability: Availability
    x: Finite | None = None
    y: Finite | None = None
    hand_detection_index: NonNegativeInt | None = None

    @model_validator(mode="after")
    def available_coordinates(self) -> Self:
        if self.availability == Availability.AVAILABLE:
            if self.x is None or self.y is None or self.hand_detection_index is None:
                raise ValueError("Available observations require coordinates and detection index")
        elif self.x is not None or self.y is not None:
            raise ValueError("Missing observations must not contain coordinates")
        return self


class MovementTrajectory(ComparisonModel):
    reported_label: HandLabel
    wrist_landmark_index: Literal[0] = 0
    observations: tuple[TrajectoryObservation, ...]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if not self.observations:
            raise ValueError("A trajectory must retain sampled observations, including gaps")
        if any(b.timestamp_ms <= a.timestamp_ms or b.frame_index <= a.frame_index
               for a, b in zip(self.observations, self.observations[1:])):
            raise ValueError("Trajectory observations must be strictly ordered")
        return self

    @property
    def usable_points(self) -> tuple[TrajectoryObservation, ...]:
        return tuple(p for p in self.observations if p.availability == Availability.AVAILABLE)


class TrajectorySummary(ComparisonModel):
    sampled_count: int = Field(strict=True, gt=0)
    usable_count: NonNegativeInt
    usable_coverage: Probability
    observed_duration_ms: Finite | None = Field(default=None, ge=0)
    maximum_usable_gap_ms: Finite | None = Field(default=None, ge=0)


class EligibilityReport(ComparisonModel):
    reference: TrajectorySummary
    learner: TrajectorySummary
    reference_eligible: bool
    learner_eligible: bool
    evidence_confidence: Probability


class AlignmentPair(ComparisonModel):
    # Indices into each trajectory's actual usable_points, not interpolated samples.
    reference_index: NonNegativeInt
    learner_index: NonNegativeInt
    reference_frame_index: NonNegativeInt
    learner_frame_index: NonNegativeInt
    local_distance: Finite = Field(ge=0)


class AlignmentResult(ComparisonModel):
    local_metric: Literal["euclidean_normalized_xy"] = "euclidean_normalized_xy"
    cumulative_cost: Finite = Field(ge=0)
    mean_path_distance: Finite = Field(ge=0)
    path: tuple[AlignmentPair, ...]

    @property
    def path_length(self) -> int:
        return len(self.path)


class ComparisonMetrics(ComparisonModel):
    finding_id: FindingId
    reference: TrajectorySummary
    learner: TrajectorySummary
    dtw_cumulative_cost: Finite = Field(ge=0)
    dtw_mean_path_distance: Finite = Field(ge=0)
    alignment_path_length: int = Field(strict=True, gt=0)
    movement_similarity: Probability
    evidence_confidence: Probability


class OutcomeReason(StrEnum):
    CORRESPONDENCE_UNESTABLISHED = "CORRESPONDENCE_UNESTABLISHED"
    REFERENCE_INSUFFICIENT = "REFERENCE_INSUFFICIENT"
    LEARNER_INSUFFICIENT = "LEARNER_INSUFFICIENT"
    BOTH_INSUFFICIENT = "BOTH_INSUFFICIENT"
    ALIGNMENT_RESOURCE_LIMIT = "ALIGNMENT_RESOURCE_LIMIT"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


class InvalidComparisonInput(ValueError):
    """Invalid caller input; never converted into a learner result."""


class ComparisonProcessingError(RuntimeError):
    """Expected processing failures that can produce a FAILED M2 result."""


class AlignmentResourceLimit(ComparisonProcessingError):
    pass


class NumericalComparisonFailure(ComparisonProcessingError):
    pass
