import numpy as np
import pytest
from pydantic import ValidationError

from ugsl_ai_coach.cv.models import SamplingConfig
from ugsl_ai_coach.cv.sampling import sample_frames
from ugsl_ai_coach.cv.video import DecodedFrame, VideoError, decode_frames, inspect_video


@pytest.mark.parametrize("fps,target,count,expected", [(60, 30, 12, 6), (15, 60, 12, 12), (29.97, 60, 12, 12)])
def test_deterministic_sampling_without_duplicates(make_video, fps, target, count, expected):
    path = make_video(fps, count)
    metadata = inspect_video(path)
    config = SamplingConfig(target_fps=target)
    first = list(sample_frames(decode_frames(path, metadata), config))
    second = list(sample_frames(decode_frames(path, metadata), config))
    assert len(first) == expected
    assert [(f.index, f.timestamp_ms) for f in first] == [(f.index, f.timestamp_ms) for f in second]
    assert len({f.index for f in first}) == expected
    assert all(b.timestamp_ms > a.timestamp_ms for a, b in zip(first, first[1:]))


def test_irregular_timestamps_skip_missed_slots():
    frames = [DecodedFrame(i, t, np.zeros((2, 2, 3), dtype=np.uint8)) for i, t in enumerate([0, 5, 40, 200, 210])]
    assert [f.timestamp_ms for f in sample_frames(frames, SamplingConfig(target_fps=25))] == [0, 40, 200]


@pytest.mark.parametrize("times", [[0, 0], [10, 5], [-1], [float("nan")]])
def test_invalid_time_order_rejected(times):
    frames = [DecodedFrame(i, t, np.zeros((2, 2, 3), dtype=np.uint8)) for i, t in enumerate(times)]
    with pytest.raises(VideoError):
        list(sample_frames(frames, SamplingConfig()))


@pytest.mark.parametrize("value", [0, -1, float("inf"), 1001])
def test_invalid_sampling_rate(value):
    with pytest.raises(ValidationError):
        SamplingConfig(target_fps=value)
