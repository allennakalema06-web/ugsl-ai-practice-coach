"""MediaPipe Tasks adapter; local assets, unmirrored RGB, model labels preserved."""

from pathlib import Path
from typing import Protocol

import cv2
from pydantic import ValidationError

from ugsl_ai_coach.cv.models import Coordinates, HandLabel, HandObservation, Landmark, RawObservation
from ugsl_ai_coach.cv.video import DecodedFrame

POSE_INDICES = (11, 12, 13, 14, 15, 16)  # Anatomical shoulders, elbows, wrists.


class ExtractionError(RuntimeError):
    """Landmarker configuration or processing failed."""


class LandmarkExtractor(Protocol):
    def extract(self, frame: DecodedFrame) -> RawObservation: ...


def reported_hand_label(label: str) -> HandLabel:
    # Deliberately preserve Tasks classifier labels. No image-position inference
    # or undocumented anatomical inversion based on the legacy Solutions API.
    return HandLabel(label)


def _landmarks(points, indices) -> tuple[Landmark, ...]:
    return tuple(Landmark(index=index, coordinates=Coordinates(
        x=points[index].x, y=points[index].y, z=points[index].z),
        visibility=points[index].visibility, presence=points[index].presence,
    ) for index in indices)


def convert_results(hand_result, pose_result) -> RawObservation:
    """Preserve all detections, even ambiguous duplicate handedness labels."""
    if len(hand_result.hand_landmarks) != len(hand_result.handedness):
        raise ExtractionError("Hand coordinates and handedness results are inconsistent")
    hands = []
    for points, categories in zip(hand_result.hand_landmarks, hand_result.handedness):
        if len(points) != 21 or not categories:
            raise ExtractionError("Incomplete hand detection output")
        category = max(categories, key=lambda item: item.score)
        hands.append(HandObservation(
            reported_handedness=reported_hand_label(category.category_name),
            handedness_confidence=category.score, landmarks=_landmarks(points, range(21)),
        ))
    pose = ()
    if pose_result.pose_landmarks:
        if len(pose_result.pose_landmarks) != 1 or len(pose_result.pose_landmarks[0]) != 33:
            raise ExtractionError("Expected one complete pose result")
        pose = _landmarks(pose_result.pose_landmarks[0], POSE_INDICES)
    return RawObservation(hands=tuple(hands), pose=pose)


class MediaPipeExtractor:
    """One context-managed instance per video; VIDEO tracking on CPU."""

    def __init__(self, hand_model: str | Path, pose_model: str | Path):
        self.hand_model, self.pose_model = Path(hand_model), Path(pose_model)
        self._hand = self._pose = None
        self._last_timestamp = -1

    def __enter__(self):
        if self._hand is not None:
            raise ExtractionError("Extractor is already open")
        for path in (self.hand_model, self.pose_model):
            if not path.is_file():
                raise ExtractionError("A required local MediaPipe model asset is missing")
        import mediapipe as mp
        self._mp = mp
        vision = mp.tasks.vision
        try:
            self._hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.hand_model)),
                running_mode=vision.RunningMode.VIDEO, num_hands=2,
            ))
            self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.pose_model)),
                running_mode=vision.RunningMode.VIDEO, num_poses=1, output_segmentation_masks=False,
            ))
        except Exception as exc:
            self.close()
            raise ExtractionError("MediaPipe initialization failed") from exc
        self._last_timestamp = -1
        return self

    def extract(self, frame: DecodedFrame) -> RawObservation:
        if self._hand is None or self._pose is None:
            raise ExtractionError("Use MediaPipeExtractor inside its context manager")
        timestamp = round(frame.timestamp_ms)
        if timestamp <= self._last_timestamp:
            raise ExtractionError("MediaPipe requires strictly increasing integer millisecond timestamps")
        try:
            rgb = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = convert_results(self._hand.detect_for_video(image, timestamp),
                                     self._pose.detect_for_video(image, timestamp))
        except (ValueError, RuntimeError, ValidationError, cv2.error) as exc:
            raise ExtractionError("MediaPipe frame extraction failed") from exc
        self._last_timestamp = timestamp
        return result

    def close(self):
        # Close both resources even if one native close operation fails.
        hand, pose = self._hand, self._pose
        self._hand = self._pose = None
        try:
            if hand is not None:
                hand.close()
        finally:
            if pose is not None:
                pose.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
