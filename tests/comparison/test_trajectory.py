import pytest

from ugsl_ai_coach.comparison.models import Availability, InvalidComparisonInput
from ugsl_ai_coach.comparison.policy import ComparisonPolicy
from ugsl_ai_coach.comparison.trajectory import select_trajectory


def test_gaps_remain_absent_with_original_times(make_extraction):
    extraction = make_extraction(missing=(1,), unnormalized=(3,), duplicate=(5,))
    trajectory = select_trajectory(extraction, "Left", ComparisonPolicy())
    assert [p.frame_index for p in trajectory.usable_points] == [0, 2, 4, 6, 7, 8, 9]
    for index, state in [(1, Availability.MISSING_HAND), (3, Availability.UNNORMALIZED), (5, Availability.AMBIGUOUS_HAND)]:
        point = trajectory.observations[index]
        assert point.availability == state
        assert point.x is None and point.y is None
        assert point.timestamp_ms == extraction.frames[index].timestamp_ms


def test_no_label_substitution_or_mirroring(make_extraction):
    extraction = make_extraction(label="Right", offset=2)
    assert not select_trajectory(extraction, "Left", ComparisonPolicy()).usable_points
    chosen = select_trajectory(extraction, "Right", ComparisonPolicy())
    assert chosen.usable_points[0].x == 2
    assert chosen.usable_points[-1].x == 2.9


@pytest.mark.parametrize("confidence", [None, 0.49])
def test_missing_or_low_label_confidence_is_unusable(make_extraction, confidence):
    trajectory = select_trajectory(make_extraction(confidence=confidence), "Left", ComparisonPolicy())
    assert not trajectory.usable_points
    assert all(p.availability == Availability.LOW_LABEL_CONFIDENCE for p in trajectory.observations)


def test_label_confidence_threshold_is_inclusive(make_extraction):
    assert len(select_trajectory(make_extraction(confidence=0.5), "Left", ComparisonPolicy()).usable_points) == 10


def test_inconsistent_normalization_is_invalid_input(make_extraction):
    from ugsl_ai_coach.cv.models import ExtractionResult
    data = make_extraction().model_dump()
    data["frames"][0]["normalization"]["hands"] = ()
    with pytest.raises(InvalidComparisonInput):
        select_trajectory(ExtractionResult.model_validate(data), "Left", ComparisonPolicy())
