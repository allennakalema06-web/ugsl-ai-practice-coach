"""PROVISIONAL ENGINEERING THRESHOLDS — NOT LINGUISTICALLY VALIDATED."""

from typing import Literal, Self

from pydantic import Field, model_validator

from ugsl_ai_coach.cv.models import Finite, Probability
from ugsl_ai_coach.domain.analysis import FindingStatus
from ugsl_ai_coach.comparison.models import ComparisonModel, NonEmpty


class FindingBand(ComparisonModel):
    status: Literal[FindingStatus.STRONG, FindingStatus.ACCEPTABLE, FindingStatus.NEEDS_IMPROVEMENT]
    severity: int = Field(strict=True, ge=0, le=3)


class ComparisonPolicy(ComparisonModel):
    algorithm_version: Literal["movement-wrist-dtw-v1"] = "movement-wrist-dtw-v1"
    policy_version: NonEmpty = "provisional-engineering-v1"
    linguistically_validated: Literal[False] = False
    minimum_usable_observations: int = Field(default=8, strict=True, ge=2)
    minimum_reference_coverage: Probability = 0.8
    minimum_learner_coverage: Probability = 0.8
    minimum_handedness_confidence: Probability = 0.5
    # Missing label confidence cannot establish reliable per-frame identity.
    maximum_alignment_cells: int = Field(default=1_000_000, strict=True, gt=0)
    similarity_distance_scale: Finite = Field(default=1.0, gt=0)
    strong_maximum_distance: Finite = Field(default=0.1, ge=0)
    acceptable_maximum_distance: Finite = Field(default=0.3, ge=0)
    strong_band: FindingBand = FindingBand(status=FindingStatus.STRONG, severity=0)
    acceptable_band: FindingBand = FindingBand(status=FindingStatus.ACCEPTABLE, severity=0)
    different_band: FindingBand = FindingBand(status=FindingStatus.NEEDS_IMPROVEMENT, severity=1)

    @model_validator(mode="after")
    def ordered_thresholds(self) -> Self:
        if self.acceptable_maximum_distance <= self.strong_maximum_distance:
            raise ValueError("acceptable_maximum_distance must exceed strong_maximum_distance")
        return self

    def classify(self, distance: float) -> FindingBand:
        # The algorithm delegates all status/severity mapping to this policy.
        import math
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Classification requires a finite non-negative distance")
        if distance <= self.strong_maximum_distance:
            return self.strong_band
        if distance <= self.acceptable_maximum_distance:
            return self.acceptable_band
        return self.different_band
