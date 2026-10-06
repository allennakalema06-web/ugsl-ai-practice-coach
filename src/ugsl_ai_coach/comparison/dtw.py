"""Exact global DTW: Euclidean XY, rolling costs, bounded byte predecessor grid."""

import math
from collections.abc import Sequence

from ugsl_ai_coach.comparison.models import (
    AlignmentPair, AlignmentResourceLimit, AlignmentResult, Availability,
    InvalidComparisonInput, NumericalComparisonFailure, TrajectoryObservation,
)


def align_dtw(reference: Sequence[TrajectoryObservation], learner: Sequence[TrajectoryObservation],
              *, maximum_cells: int) -> AlignmentResult:
    if not reference or not learner:
        raise InvalidComparisonInput("DTW requires non-empty sequences")
    if type(maximum_cells) is not int or maximum_cells <= 0:
        raise InvalidComparisonInput("DTW maximum_cells must be a positive integer")
    for sequence in (reference, learner):
        if any(p.availability != Availability.AVAILABLE or p.x is None or p.y is None
               or not all(math.isfinite(v) for v in (p.x, p.y, p.timestamp_ms)) for p in sequence):
            raise InvalidComparisonInput("DTW requires actual finite available observations")
        if any(b.timestamp_ms <= a.timestamp_ms or b.frame_index <= a.frame_index
               for a, b in zip(sequence, sequence[1:])):
            raise InvalidComparisonInput("DTW sequences must have ordered timestamps and frame indices")
    n, m = len(reference), len(learner)
    if n * m > maximum_cells:
        raise AlignmentResourceLimit("Requested exact alignment exceeds configured cell budget")
    predecessors = bytearray(n * m)
    previous = [math.inf] * (m + 1)
    previous[0] = 0.0
    for i, a in enumerate(reference):
        current = [math.inf] * (m + 1)
        for j, b in enumerate(learner):
            distance = math.hypot(a.x - b.x, a.y - b.y)
            # Stable tie order: diagonal, advance reference, advance learner.
            choices = (previous[j], previous[j + 1], current[j])
            direction = min(range(3), key=choices.__getitem__)
            cost = distance + choices[direction]
            if not math.isfinite(cost):
                raise NumericalComparisonFailure("DTW distance or accumulated cost is non-finite")
            current[j + 1] = cost
            predecessors[i * m + j] = direction
        previous = current
    cost = previous[m]
    i, j, path = n - 1, m - 1, []
    while i >= 0 and j >= 0:
        a, b = reference[i], learner[j]
        path.append(AlignmentPair(reference_index=i, learner_index=j,
                                  reference_frame_index=a.frame_index, learner_frame_index=b.frame_index,
                                  local_distance=math.hypot(a.x - b.x, a.y - b.y)))
        direction = predecessors[i * m + j]
        if direction == 0:
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return AlignmentResult(cumulative_cost=cost, mean_path_distance=cost / len(path), path=tuple(path))
