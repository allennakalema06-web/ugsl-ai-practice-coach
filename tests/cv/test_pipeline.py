import json

import pytest
from pydantic import ValidationError

from ugsl_ai_coach.cv.models import Coordinates, HandObservation, Landmark, RawObservation, SamplingConfig
from ugsl_ai_coach.cv.pipeline import extract_video


def hand(label):
    return HandObservation(reported_handedness=label, landmarks=tuple(
        Landmark(index=i, coordinates=Coordinates(x=0.1 + i / 100, y=0.2, z=-0.3)) for i in range(21)
    ))


class FakeExtractor:
    def extract(self, frame):
        pose = tuple(Landmark(index=i, coordinates=Coordinates(x=x, y=0.5)) for i, x in [(11, 0.25), (12, 0.75)])
        if frame.index == 0:
            return RawObservation(hands=(hand("Left"), hand("Right")), pose=pose)
        if frame.index == 1:
            return RawObservation(hands=(hand("Right"),))
        return RawObservation()


def test_pipeline_preserves_gaps_times_normalization_and_coverage(make_video):
    path = make_video(fps=15, count=4)
    result = extract_video(path, FakeExtractor())
    assert result.sampling.target_fps == 60
    assert len(result.frames) == 4
    assert [f.frame_index for f in result.frames] == [0, 1, 2, 3]
    assert result.frames[1].timestamp_ms > result.frames[0].timestamp_ms
    assert result.frames[1].raw.hands[0].reported_handedness == "Right"
    assert result.frames[2].raw.hands == ()  # Neither zero fill nor last-value carry.
    assert result.frames[1].normalization.status == "MISSING_SHOULDERS"
    assert result.frames[0].normalization.hands[0][0].coordinates.x == -0.8
    assert result.frames[0].raw.hands[0].landmarks[0].coordinates.z == -0.3
    assert result.coverage.model_dump() == {
        "sampled_frame_count": 4, "frames_with_pose": 1,
        "frames_with_left_hand": 1, "frames_with_right_hand": 2, "frames_normalized": 1,
    }
    assert result == extract_video(path, FakeExtractor())
    payload = json.loads(result.model_dump_json())
    forbidden = {"overall_score", "severity", "findings", "feedback", "skill", "performance_score"}
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    assert not keys(payload) & forbidden


def test_zero_detection_has_no_fake_landmarks(make_video):
    class EmptyExtractor:
        def extract(self, frame):
            return RawObservation()
    result = extract_video(make_video(), EmptyExtractor(), SamplingConfig(target_fps=15))
    assert len(result.frames) == 6
    assert result.coverage.frames_normalized == result.coverage.frames_with_pose == 0
    assert result.coverage.frames_with_left_hand == result.coverage.frames_with_right_hand == 0
    assert all(not f.raw.hands and not f.raw.pose and not f.normalization.hands for f in result.frames)


@pytest.mark.parametrize("change", ["reversed", "duplicate", "coverage"])
def test_result_rejects_inconsistent_trajectory(make_video, change):
    result = extract_video(make_video(), FakeExtractor()).model_dump()
    if change == "reversed":
        result["frames"] = tuple(reversed(result["frames"]))
    elif change == "duplicate":
        result["frames"] = (result["frames"][0], result["frames"][0])
    else:
        result["coverage"]["frames_with_left_hand"] = 100
    from ugsl_ai_coach.cv.models import ExtractionResult
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate(result)


def test_decoder_resources_close_when_extractor_fails(make_video, monkeypatch):
    import cv2
    captures = []
    original = cv2.VideoCapture
    def capture(path):
        resource = original(path)
        captures.append(resource)
        return resource
    monkeypatch.setattr(cv2, "VideoCapture", capture)
    class FailedExtractor:
        def extract(self, frame):
            raise RuntimeError("controlled extractor failure")
    with pytest.raises(RuntimeError, match="controlled"):
        extract_video(make_video(), FailedExtractor())
    assert len(captures) == 2
    assert all(not resource.isOpened() for resource in captures)
