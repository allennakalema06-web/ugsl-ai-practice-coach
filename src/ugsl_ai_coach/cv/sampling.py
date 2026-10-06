"""Select actual frames by elapsed time; never duplicate or interpolate."""

from collections.abc import Iterable, Iterator
import math

from ugsl_ai_coach.cv.models import SamplingConfig
from ugsl_ai_coach.cv.video import DecodedFrame, VideoError


def sample_frames(frames: Iterable[DecodedFrame], config: SamplingConfig) -> Iterator[DecodedFrame]:
    period = 1000 / config.target_fps
    next_time, previous_time, previous_index = 0.0, -1.0, -1
    for frame in frames:
        if (not math.isfinite(frame.timestamp_ms) or frame.timestamp_ms < 0
                or frame.timestamp_ms <= previous_time or frame.index <= previous_index):
            raise VideoError("Sampling requires ordered unique frames and timestamps")
        previous_time, previous_index = frame.timestamp_ms, frame.index
        if frame.timestamp_ms + 1e-7 >= next_time:
            yield frame
            # Skip missed target slots rather than reusing this frame for them.
            next_time = (math.floor((frame.timestamp_ms + 1e-7) / period) + 1) * period
