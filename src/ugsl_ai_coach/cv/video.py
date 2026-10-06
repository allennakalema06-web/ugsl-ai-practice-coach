"""Local file validation and sequential decoding with explicit timestamp provenance."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np

from ugsl_ai_coach.cv.models import TimestampSource, VideoMetadata


class VideoError(ValueError):
    """A file cannot be used safely by the extraction pipeline."""


@dataclass(frozen=True)
class DecodedFrame:
    index: int
    timestamp_ms: float
    bgr: np.ndarray


@contextmanager
def _capture(path: Path):
    if not path.exists():
        raise VideoError("Video file does not exist")
    if not path.is_file():
        raise VideoError("Video path must be a regular file")
    capture = None
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise VideoError("Video cannot be opened by the decoder")
        yield capture
    except cv2.error as exc:
        raise VideoError("Video decoding failed") from exc
    finally:
        if capture is not None:
            capture.release()


def _frame_valid(frame, width: int, height: int) -> None:
    if frame is None or frame.size == 0:
        raise VideoError("Decoder returned an empty frame")
    if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
        raise VideoError("Video must decode to fixed-size 8-bit BGR frames")


def inspect_video(path: str | Path) -> VideoMetadata:
    # Scan once to verify usable frames and timestamp consistency, without storing pixels.
    with _capture(Path(path)) as capture:
        width, height, fps, reported_count = (
            capture.get(prop) for prop in (
                cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT,
            )
        )
        if not all(math.isfinite(v) for v in (width, height, fps, reported_count)):
            raise VideoError("Video metadata contains non-finite values")
        if not (0 < width <= 32768 and 0 < height <= 32768 and width.is_integer() and height.is_integer()):
            raise VideoError("Video dimensions are invalid or exceed decoder safety limits")
        if not 0 < fps <= 1000 or reported_count < 0:
            raise VideoError("Video frame rate or frame count is invalid")
        count, last, first = 0, 0.0, 0.0
        all_zero, monotonic = True, True
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            _frame_valid(frame, int(width), int(height))
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(timestamp) or timestamp < 0:
                raise VideoError("Video timestamps are invalid")
            if count == 0:
                first = timestamp
            elif timestamp <= last:
                monotonic = False
            all_zero = all_zero and timestamp == 0
            last = timestamp
            count += 1
        if count == 0:
            raise VideoError("Video contains no usable frames")
        if reported_count > count + 0.5:
            raise VideoError("Video ended before its reported frame count; possible truncated decode")
        if not monotonic and not all_zero:
            raise VideoError("Decoder timestamps are not strictly increasing")
        source = TimestampSource.FPS_ESTIMATE if all_zero else TimestampSource.DECODER
        duration = count * 1000 / fps if all_zero else last - first + 1000 / fps
        if not math.isfinite(duration) or duration <= 0:
            raise VideoError("Video duration is invalid")
        return VideoMetadata(width=int(width), height=int(height), fps=fps,
                             frame_count=count, duration_ms=duration, timestamp_source=source)


def decode_frames(path: str | Path, metadata: VideoMetadata) -> Iterator[DecodedFrame]:
    with _capture(Path(path)) as capture:
        count, previous, origin = 0, -1.0, None
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            _frame_valid(bgr, metadata.width, metadata.height)
            if metadata.timestamp_source == TimestampSource.FPS_ESTIMATE:
                timestamp = count * 1000 / metadata.fps
            else:
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
                if origin is None:
                    origin = timestamp
                timestamp -= origin
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
                raise VideoError("Video timestamps changed or are invalid during extraction")
            yield DecodedFrame(count, timestamp, bgr)
            previous = timestamp
            count += 1
        if count != metadata.frame_count:
            raise VideoError("Video frame count changed between inspection and extraction")
