import math

import pytest
from pydantic import ValidationError

from ugsl_ai_coach.comparison.policy import ComparisonPolicy, FindingBand


@pytest.mark.parametrize("distance,status,severity", [
    (0, "STRONG", 0), (0.1, "STRONG", 0),
    (math.nextafter(0.1, math.inf), "ACCEPTABLE", 0), (0.3, "ACCEPTABLE", 0),
    (math.nextafter(0.3, math.inf), "NEEDS_IMPROVEMENT", 1),
])
def test_provisional_boundaries(distance, status, severity):
    band = ComparisonPolicy().classify(distance)
    assert (band.status, band.severity) == (status, severity)


def test_replace_thresholds_and_band_mapping():
    policy = ComparisonPolicy(policy_version="local-engineering-v2", strong_maximum_distance=0.2,
        acceptable_maximum_distance=0.5, different_band=FindingBand(status="NEEDS_IMPROVEMENT", severity=2))
    assert policy.classify(0.15).status == "STRONG"
    assert policy.classify(0.4).status == "ACCEPTABLE"
    assert policy.classify(0.6).severity == 2
    assert policy.algorithm_version == "movement-wrist-dtw-v1"
    assert policy.policy_version == "local-engineering-v2"


@pytest.mark.parametrize("field,value", [
    ("minimum_usable_observations", 1), ("minimum_usable_observations", True),
    ("minimum_reference_coverage", -0.1), ("minimum_learner_coverage", 1.1),
    ("minimum_handedness_confidence", float("nan")), ("similarity_distance_scale", 0),
    ("maximum_alignment_cells", 0), ("strong_maximum_distance", -1),
    ("acceptable_maximum_distance", 0.1), ("acceptable_maximum_distance", float("inf")),
    ("policy_version", "  "), ("algorithm_version", "pretend-v2"),
])
def test_invalid_policies(field, value):
    with pytest.raises(ValidationError):
        ComparisonPolicy(**{field: value})
