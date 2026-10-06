import math

import pytest

from ugsl_ai_coach.comparison.metrics import check_eligibility, movement_similarity
from ugsl_ai_coach.comparison.policy import ComparisonPolicy
from ugsl_ai_coach.comparison.trajectory import select_trajectory


def test_counts_coverage_duration_and_confidence(make_extraction):
    policy = ComparisonPolicy()
    reference = select_trajectory(make_extraction(), "Left", policy)
    learner = select_trajectory(make_extraction(missing=(1, 2)), "Left", policy)
    report = check_eligibility(reference, learner, policy)
    assert report.reference.usable_count == 10
    assert report.learner.usable_count == 8
    assert report.learner.usable_coverage == 0.8
    assert report.evidence_confidence == 0.8
    assert report.reference_eligible and report.learner_eligible
    assert report.learner.observed_duration_ms == 9 * 40.25
    assert report.learner.maximum_usable_gap_ms == 3 * 40.25


def test_count_sufficiency_affects_confidence(make_extraction):
    policy = ComparisonPolicy()
    small = select_trajectory(make_extraction([(0, 0)]), "Left", policy)
    report = check_eligibility(small, small, policy)
    assert report.evidence_confidence == 1 / 8
    assert not report.learner_eligible


def test_similarity_is_monotone_and_bounded():
    values = [movement_similarity(d, 1.0) for d in [0, 0.1, 1, 10, 1e308]]
    assert values[0] == 1
    assert values[2] == 0.5
    assert all(0 <= v <= 1 and math.isfinite(v) for v in values)
    assert values == sorted(values, reverse=True)
    assert movement_similarity(1e308, 1e308) == 0.5
    assert movement_similarity(1, 2) > movement_similarity(1, 1)


@pytest.mark.parametrize("distance,scale", [(-1, 1), (float("nan"), 1), (float("inf"), 1),
    (float("-inf"), 1), (1, 0), (1, -1), (1, float("inf")), (1, float("nan"))])
def test_invalid_similarity_inputs(distance, scale):
    with pytest.raises(ValueError):
        movement_similarity(distance, scale)
