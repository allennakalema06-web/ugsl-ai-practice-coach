import cv2
import numpy as np
import pytest

from ugsl_ai_coach.cv.video import VideoError, decode_frames, inspect_video


def test_valid_metadata_and_decoding(make_video):
    path = make_video(fps=24, count=12)
    metadata = inspect_video(path)
    assert (metadata.width, metadata.height, metadata.frame_count) == (64, 48, 12)
    assert metadata.fps == pytest.approx(24)
    assert metadata.duration_ms == pytest.approx(500)
    frames = list(decode_frames(path, metadata))
    assert [f.index for f in frames] == list(range(12))
    assert [f.timestamp_ms for f in frames] == pytest.approx([i * 1000 / 24 for i in range(12)])


def test_missing_video(tmp_path):
    with pytest.raises(VideoError, match="does not exist"):
        inspect_video(tmp_path / "missing.avi")


def test_directory_is_not_video(tmp_path):
    with pytest.raises(VideoError, match="regular file"):
        inspect_video(tmp_path)


def test_corrupt_video(tmp_path):
    path = tmp_path / "invalid.avi"
    path.write_bytes(b"invalid video bytes")
    with pytest.raises(VideoError):
        inspect_video(path)


class StubCapture:
    def __init__(self, overrides=None, timestamps=(0, 40, 80), empty=False):
        self.values = {cv2.CAP_PROP_FRAME_WIDTH: 64.0, cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
                       cv2.CAP_PROP_FPS: 25.0, cv2.CAP_PROP_FRAME_COUNT: float(len(timestamps))}
        self.values.update(overrides or {})
        self.timestamps, self.index, self.released, self.empty = timestamps, 0, False, empty

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps[self.index - 1]
        return self.values[prop]

    def read(self):
        if self.index == len(self.timestamps):
            return False, None
        self.index += 1
        return True, np.zeros((0, 0, 3) if self.empty else (48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


@pytest.mark.parametrize("property,value", [
    (cv2.CAP_PROP_FPS, 0), (cv2.CAP_PROP_FPS, -1), (cv2.CAP_PROP_FPS, float("nan")),
    (cv2.CAP_PROP_FPS, 1001), (cv2.CAP_PROP_FRAME_WIDTH, 0),
    (cv2.CAP_PROP_FRAME_HEIGHT, -1), (cv2.CAP_PROP_FRAME_WIDTH, 1.5),
    (cv2.CAP_PROP_FRAME_WIDTH, 40000), (cv2.CAP_PROP_FRAME_COUNT, -1),
])
def test_invalid_metadata_and_cleanup(monkeypatch, tmp_path, property, value):
    path = tmp_path / "stub.avi"
    path.touch()
    capture = StubCapture({property: value})
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(VideoError):
        inspect_video(path)
    assert capture.released


@pytest.mark.parametrize("timestamps", [(), (0, 40, 20), (0, float("inf")), (0, -1)])
def test_invalid_timestamps_or_no_frames(monkeypatch, tmp_path, timestamps):
    path = tmp_path / "stub.avi"
    path.touch()
    capture = StubCapture(timestamps=timestamps)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(VideoError):
        inspect_video(path)
    assert capture.released


def test_all_zero_timestamps_use_explicit_fps_estimate(monkeypatch, tmp_path):
    path = tmp_path / "stub.avi"
    path.touch()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: StubCapture(timestamps=(0, 0, 0)))
    metadata = inspect_video(path)
    assert metadata.timestamp_source == "fps_estimate"
    assert [f.timestamp_ms for f in decode_frames(path, metadata)] == [0, 40, 80]


def test_variable_frame_times_are_retained(monkeypatch, tmp_path):
    path = tmp_path / "stub.avi"
    path.touch()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: StubCapture(timestamps=(10, 50, 130)))
    metadata = inspect_video(path)
    assert metadata.timestamp_source == "decoder"
    assert [f.timestamp_ms for f in decode_frames(path, metadata)] == [0, 40, 120]


def test_empty_decoded_frame(monkeypatch, tmp_path):
    path = tmp_path / "stub.avi"
    path.touch()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: StubCapture(empty=True))
    with pytest.raises(VideoError, match="empty frame"):
        inspect_video(path)


def test_truncated_decode(monkeypatch, tmp_path):
    path = tmp_path / "stub.avi"
    path.touch()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: StubCapture({cv2.CAP_PROP_FRAME_COUNT: 100.0}))
    with pytest.raises(VideoError, match="truncated"):
        inspect_video(path)
