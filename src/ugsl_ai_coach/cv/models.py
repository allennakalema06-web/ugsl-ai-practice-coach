"""Immutable, finite, time-aware extraction data."""

from enum import StrEnum
from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Finite = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Probability = Annotated[Finite, Field(ge=0, le=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


class ObservationModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Coordinates(ObservationModel):
    x: Finite
    y: Finite
    z: Finite | None = None


class NormalizedCoordinates(ObservationModel):
    # Only planar coordinates share a shoulder-relative origin across models.
    x: Finite
    y: Finite


class Landmark(ObservationModel):
    index: NonNegativeInt
    coordinates: Coordinates
    visibility: Probability | None = None
    presence: Probability | None = None


class HandLabel(StrEnum):
    LEFT = "Left"
    RIGHT = "Right"


class HandObservation(ObservationModel):
    reported_handedness: HandLabel
    handedness_confidence: Probability | None = None
    landmarks: tuple[Landmark, ...]

    @model_validator(mode="after")
    def complete_hand(self) -> Self:
        if {p.index for p in self.landmarks} != set(range(21)) or len(self.landmarks) != 21:
            raise ValueError("Detected hands must retain all 21 unique landmark indices")
        return self


class RawObservation(ObservationModel):
    hands: tuple[HandObservation, ...] = ()
    pose: tuple[Landmark, ...] = ()

    @model_validator(mode="after")
    def unique_pose(self) -> Self:
        if len({p.index for p in self.pose}) != len(self.pose):
            raise ValueError("Pose landmark indices must be unique")
        return self


class SamplingConfig(ObservationModel):
    target_fps: Finite = Field(default=60.0, gt=0, le=1000)


class TimestampSource(StrEnum):
    DECODER = "decoder"
    FPS_ESTIMATE = "fps_estimate"


class VideoMetadata(ObservationModel):
    width: int = Field(strict=True, gt=0, le=32768)
    height: int = Field(strict=True, gt=0, le=32768)
    fps: Finite = Field(gt=0, le=1000)
    frame_count: int = Field(strict=True, gt=0)
    duration_ms: Finite = Field(gt=0)
    timestamp_source: TimestampSource


class NormalizationConfig(ObservationModel):
    minimum_shoulder_width: Finite = Field(default=1e-6, gt=0)
    minimum_anchor_visibility: Probability = 0.5


class NormalizationStatus(StrEnum):
    NORMALIZED = "NORMALIZED"
    MISSING_SHOULDERS = "MISSING_SHOULDERS"
    LOW_ANCHOR_VISIBILITY = "LOW_ANCHOR_VISIBILITY"
    INVALID_SHOULDER_SCALE = "INVALID_SHOULDER_SCALE"


class NormalizedLandmark(ObservationModel):
    index: NonNegativeInt
    coordinates: NormalizedCoordinates


class NormalizationResult(ObservationModel):
    status: NormalizationStatus
    # x/y origin and scale in image-width units (y aspect-corrected).
    origin: NormalizedCoordinates | None = None
    shoulder_width: Finite | None = Field(default=None, gt=0)
    pose: tuple[NormalizedLandmark, ...] = ()
    # Same detection order as raw.hands; no second copy of labels/confidence.
    hands: tuple[tuple[NormalizedLandmark, ...], ...] = ()

    @model_validator(mode="after")
    def consistent_state(self) -> Self:
        if self.status == NormalizationStatus.NORMALIZED:
            if self.origin is None or self.shoulder_width is None:
                raise ValueError("Successful normalization requires origin and shoulder width")
        elif self.origin is not None or self.shoulder_width is not None or self.pose or self.hands:
            raise ValueError("Unavailable normalization must not contain transformed observations")
        return self


class FrameObservation(ObservationModel):
    frame_index: NonNegativeInt
    timestamp_ms: Finite = Field(ge=0)
    raw: RawObservation
    normalization: NormalizationResult


class CoverageSummary(ObservationModel):
    sampled_frame_count: NonNegativeInt
    frames_with_pose: NonNegativeInt
    frames_with_left_hand: NonNegativeInt
    frames_with_right_hand: NonNegativeInt
    frames_normalized: NonNegativeInt

    @classmethod
    def from_frames(cls, frames: Sequence[FrameObservation]) -> Self:
        return cls(
            sampled_frame_count=len(frames),
            frames_with_pose=sum(bool(f.raw.pose) for f in frames),
            frames_with_left_hand=sum(any(h.reported_handedness == HandLabel.LEFT for h in f.raw.hands) for f in frames),
            frames_with_right_hand=sum(any(h.reported_handedness == HandLabel.RIGHT for h in f.raw.hands) for f in frames),
            frames_normalized=sum(f.normalization.status == NormalizationStatus.NORMALIZED for f in frames),
        )


class ExtractionResult(ObservationModel):
    video: VideoMetadata
    sampling: SamplingConfig
    normalization_config: NormalizationConfig
    coordinate_convention: Literal["unmirrored_image_xy_model_reported_handedness"] = "unmirrored_image_xy_model_reported_handedness"
    frames: tuple[FrameObservation, ...]
    coverage: CoverageSummary

    @model_validator(mode="after")
    def ordered_frames(self) -> Self:
        if not self.frames:
            raise ValueError("Extraction must include at least one sampled frame")
        if any(b.timestamp_ms <= a.timestamp_ms or b.frame_index <= a.frame_index
               for a, b in zip(self.frames, self.frames[1:])):
            raise ValueError("Frames must have strictly increasing timestamps and indices")
        if self.frames[-1].frame_index >= self.video.frame_count:
            raise ValueError("Sampled frame indices must be within the decoded video")
        expected = CoverageSummary.from_frames(self.frames)
        if self.coverage != expected:
            raise ValueError("Coverage must describe the actual frame observations")
        return self
