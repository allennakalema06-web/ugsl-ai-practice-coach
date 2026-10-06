"""Descriptive evidence sufficiency is independent of geometric similarity."""

import math

from ugsl_ai_coach.comparison.models import (
    EligibilityReport, MovementTrajectory, TrajectorySummary,
)
from ugsl_ai_coach.comparison.policy import ComparisonPolicy


def summarize(trajectory: MovementTrajectory) -> TrajectorySummary:
    points = trajectory.usable_points
    return TrajectorySummary(
        sampled_count=len(trajectory.observations), usable_count=len(points),
        usable_coverage=len(points) / len(trajectory.observations),
        observed_duration_ms=points[-1].timestamp_ms - points[0].timestamp_ms if points else None,
        maximum_usable_gap_ms=max((b.timestamp_ms - a.timestamp_ms for a, b in zip(points, points[1:])), default=None),
    )


def check_eligibility(reference: MovementTrajectory, learner: MovementTrajectory,
                      policy: ComparisonPolicy) -> EligibilityReport:
    ref, learn = summarize(reference), summarize(learner)
    minimum = policy.minimum_usable_observations
    # No coordinate distances, DTW costs, or similarity enter this formula.
    confidence = min(ref.usable_coverage, learn.usable_coverage,
                     min(1.0, ref.usable_count / minimum), min(1.0, learn.usable_count / minimum))
    return EligibilityReport(
        reference=ref, learner=learn,
        reference_eligible=ref.usable_count >= minimum and ref.usable_coverage >= policy.minimum_reference_coverage,
        learner_eligible=learn.usable_count >= minimum and learn.usable_coverage >= policy.minimum_learner_coverage,
        evidence_confidence=confidence,
    )


def movement_similarity(distance: float, distance_scale: float) -> float:
    """s/(s+d): identity=1, monotone decreasing, finite, no linguistic calibration."""
    if not math.isfinite(distance) or distance < 0 or not math.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError("Similarity requires finite non-negative distance and positive scale")
    # Equivalent to s/(s+d), avoiding overflow in s+d for large finite inputs.
    if distance <= distance_scale:
        return 1 / (1 + distance / distance_scale)
    ratio = distance_scale / distance
    return ratio / (1 + ratio)
