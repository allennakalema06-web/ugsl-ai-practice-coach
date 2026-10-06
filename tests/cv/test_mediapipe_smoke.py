from pathlib import Path

import numpy as np
import pytest

from ugsl_ai_coach.cv.landmarks import MediaPipeExtractor
from ugsl_ai_coach.cv.video import DecodedFrame


@pytest.mark.integration
def test_local_mediapipe_initialization_and_blank_frame():
    models = Path(__file__).resolve().parents[2] / ".models"
    hand = models / "hand_landmarker.task"
    pose = models / "pose_landmarker_lite.task"
    if not hand.is_file() or not pose.is_file():
        from ugsl_ai_coach.assets import model_paths
        hand, pose = model_paths()
    with MediaPipeExtractor(hand, pose) as extractor:
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        raw = extractor.extract(DecodedFrame(0, 0, blank))
        assert not raw.hands and not raw.pose
        assert not extractor.extract(DecodedFrame(1, 20, blank)).hands
