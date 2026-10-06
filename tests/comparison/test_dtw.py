import math

import pytest

from ugsl_ai_coach.comparison.dtw import align_dtw
from ugsl_ai_coach.comparison.models import (
    AlignmentResourceLimit, InvalidComparisonInput, NumericalComparisonFailure, TrajectoryObservation,
)


def points(values):
    return tuple(TrajectoryObservation(frame_index=i, timestamp_ms=i * 40.0,
        availability="AVAILABLE", hand_detection_index=0, x=float(x), y=0.0) for i, x in enumerate(values))


def test_identical_and_different_paths():
    same = points([0, 1, 2])
    alignment = align_dtw(same, same, maximum_cells=100)
    assert alignment.cumulative_cost == alignment.mean_path_distance == 0
    assert [(p.reference_index, p.learner_index) for p in alignment.path] == [(0, 0), (1, 1), (2, 2)]
    assert align_dtw(same, points([10, 11, 12]), maximum_cells=100).mean_path_distance > 0


def test_different_lengths_and_speeds_align_without_interpolation():
    short = points([0, 1, 2])
    long = points([0, 0, 1, 1, 2, 2])
    result = align_dtw(short, long, maximum_cells=100)
    assert result.cumulative_cost == 0
    assert result.path_length == 6
    assert {p.reference_index for p in result.path} == set(range(3))
    assert {p.learner_index for p in result.path} == set(range(6))


def test_known_euclidean_cost_and_path_mean():
    a = points([0])
    b = (TrajectoryObservation(frame_index=0, timestamp_ms=0.0, availability="AVAILABLE",
                              hand_detection_index=0, x=3.0, y=4.0),)
    result = align_dtw(a, b, maximum_cells=1)
    assert result.cumulative_cost == result.mean_path_distance == 5


def test_minimum_cost_against_small_exhaustive_oracle():
    # Enumerate all monotone paths independently rather than reproducing DTW's recurrence.
    a, b = points([0, 2, 1]), points([0, 1, 2, 1])
    def paths(i, j, cost):
        cost += math.hypot(a[i].x - b[j].x, a[i].y - b[j].y)
        if i == len(a) - 1 and j == len(b) - 1:
            return [cost]
        candidates = []
        for di, dj in [(1, 0), (0, 1), (1, 1)]:
            if i + di < len(a) and j + dj < len(b):
                candidates.extend(paths(i + di, j + dj, cost))
        return candidates
    result = align_dtw(a, b, maximum_cells=100)
    assert result.cumulative_cost == min(paths(0, 0, 0))
    assert sum(p.local_distance for p in result.path) == pytest.approx(result.cumulative_cost)
    assert result.mean_path_distance == pytest.approx(result.cumulative_cost / result.path_length)


@pytest.mark.parametrize("left,right", [((), points([1])), (points([1]), ()), ((), ())])
def test_empty_inputs_rejected(left, right):
    with pytest.raises(InvalidComparisonInput):
        align_dtw(left, right, maximum_cells=100)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_input_rejected_even_if_validation_bypassed(value):
    invalid = TrajectoryObservation.model_construct(frame_index=0, timestamp_ms=0.0,
        availability="AVAILABLE", hand_detection_index=0, x=value, y=0.0)
    with pytest.raises(InvalidComparisonInput):
        align_dtw((invalid,), points([0]), maximum_cells=100)


def test_missing_input_rejected():
    missing = TrajectoryObservation(frame_index=0, timestamp_ms=0.0, availability="MISSING_HAND")
    with pytest.raises(InvalidComparisonInput):
        align_dtw((missing,), points([0]), maximum_cells=100)


def test_finite_inputs_with_overflow_raise_typed_failure():
    with pytest.raises(NumericalComparisonFailure):
        align_dtw(points([1e308]), points([-1e308]), maximum_cells=100)


def test_cell_budget_and_boundary():
    a = points([0, 1])
    assert align_dtw(a, a, maximum_cells=4).cumulative_cost == 0
    with pytest.raises(AlignmentResourceLimit):
        align_dtw(a, a, maximum_cells=3)


def test_inputs_unchanged_and_ties_deterministic():
    a, b = points([0, 0, 0]), points([0, 0])
    before = tuple(p.model_dump_json() for p in a + b)
    first = align_dtw(a, b, maximum_cells=100)
    assert first == align_dtw(a, b, maximum_cells=100)
    assert tuple(p.model_dump_json() for p in a + b) == before


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_rejected(budget):
    with pytest.raises(InvalidComparisonInput):
        align_dtw(points([0]), points([0]), maximum_cells=budget)
