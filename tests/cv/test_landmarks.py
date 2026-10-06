from types import SimpleNamespace as NS
import sys

import numpy as np
import pytest

from ugsl_ai_coach.cv.landmarks import ExtractionError, MediaPipeExtractor, convert_results
from ugsl_ai_coach.cv.video import DecodedFrame


def mp_point(x=0.1, y=0.2, z=-0.3, visibility=None, presence=None):
    return NS(x=x, y=y, z=z, visibility=visibility, presence=presence)


def test_adapter_retains_complete_hands_pose_and_quality():
    hands = NS(hand_landmarks=[[mp_point(x=0.1)] * 21, [mp_point(x=0.9)] * 21],
               handedness=[[NS(category_name="Left", score=0.8)], [NS(category_name="Right", score=0.9)]])
    pose = NS(pose_landmarks=[[mp_point(visibility=0.7, presence=0.8)] * 33])
    raw = convert_results(hands, pose)
    assert [h.reported_handedness for h in raw.hands] == ["Left", "Right"]
    assert [h.handedness_confidence for h in raw.hands] == [0.8, 0.9]
    assert [h.landmarks[0].coordinates.x for h in raw.hands] == [0.1, 0.9]
    assert raw.hands[0].landmarks[0].coordinates.z == -0.3
    assert raw.hands[0].landmarks[0].visibility is None
    assert [p.index for p in raw.pose] == [11, 12, 13, 14, 15, 16]
    assert raw.pose[0].visibility == 0.7 and raw.pose[0].presence == 0.8
    assert len(raw.hands[0].landmarks) == 21


def test_missing_detections_stay_missing():
    raw = convert_results(NS(hand_landmarks=[], handedness=[]), NS(pose_landmarks=[]))
    assert raw.hands == () and raw.pose == ()


def test_duplicate_labels_do_not_overwrite_detections():
    hands = NS(hand_landmarks=[[mp_point()] * 21] * 2,
               handedness=[[NS(category_name="Left", score=0.8)]] * 2)
    assert len(convert_results(hands, NS(pose_landmarks=[])).hands) == 2


def test_missing_model_assets_fail_explicitly(tmp_path):
    with pytest.raises(ExtractionError, match="asset is missing"):
        with MediaPipeExtractor(tmp_path / "hand.task", tmp_path / "pose.task"):
            pass


def test_unopened_extractor_fails(tmp_path):
    extractor = MediaPipeExtractor(tmp_path / "hand.task", tmp_path / "pose.task")
    with pytest.raises(ExtractionError, match="context manager"):
        extractor.extract(DecodedFrame(0, 0, np.zeros((10, 10, 3), dtype=np.uint8)))


def test_tasks_receive_rgb_without_mirroring_and_close(monkeypatch, tmp_path):
    assets = [tmp_path / "hand.task", tmp_path / "pose.task"]
    for path in assets:
        path.touch()
    images, timestamps, closed = [], [], []
    class Task:
        def __init__(self, kind):
            self.kind = kind
        def detect_for_video(self, image, timestamp):
            images.append(image.copy())
            timestamps.append(timestamp)
            return NS(hand_landmarks=[], handedness=[]) if self.kind == "hand" else NS(pose_landmarks=[])
        def close(self):
            closed.append(self.kind)
    factories = {kind: NS(create_from_options=lambda options, kind=kind: Task(kind)) for kind in ("hand", "pose")}
    options = lambda **kwargs: kwargs
    mp = NS(tasks=NS(BaseOptions=options, vision=NS(
        HandLandmarker=factories["hand"], PoseLandmarker=factories["pose"],
        HandLandmarkerOptions=options, PoseLandmarkerOptions=options, RunningMode=NS(VIDEO="VIDEO"),
    )), ImageFormat=NS(SRGB="SRGB"), Image=lambda image_format, data: data)
    monkeypatch.setitem(sys.modules, "mediapipe", mp)
    bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
    with MediaPipeExtractor(*assets) as extractor:
        assert not extractor.extract(DecodedFrame(0, 1.2, bgr)).hands
        with pytest.raises(ExtractionError, match="strictly increasing"):
            extractor.extract(DecodedFrame(1, 1.3, bgr))
    assert timestamps == [1, 1]
    assert np.array_equal(images[0], bgr[:, :, ::-1])
    assert np.array_equal(bgr, np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8))
    assert closed == ["hand", "pose"]


def test_initialization_failure_closes_partial_resources(monkeypatch, tmp_path):
    import mediapipe as mp
    assets = [tmp_path / "hand.task", tmp_path / "pose.task"]
    for path in assets:
        path.touch()
    closed = []
    monkeypatch.setattr(mp.tasks.vision.HandLandmarker, "create_from_options", lambda _: NS(close=lambda: closed.append("hand")))
    def fail(_):
        raise RuntimeError("native initialization failed")
    monkeypatch.setattr(mp.tasks.vision.PoseLandmarker, "create_from_options", fail)
    with pytest.raises(ExtractionError, match="initialization failed"):
        with MediaPipeExtractor(*assets):
            pass
    assert closed == ["hand"]
