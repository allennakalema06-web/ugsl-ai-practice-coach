import pytest

from ugsl_ai_coach.comparison.models import ComparisonRequest, HandCorrespondence
from ugsl_ai_coach.cv.models import (
    Coordinates, CoverageSummary, ExtractionResult, FrameObservation, HandObservation, Landmark,
    NormalizationConfig, NormalizationResult, NormalizedCoordinates, NormalizedLandmark,
    RawObservation, SamplingConfig, VideoMetadata,
)


@pytest.fixture
def comparison_request():
    return ComparisonRequest(analysis_id="AN-000001", attempt_id="ATT-000001", reference_id="synthetic-reference",
        correspondence=HandCorrespondence(reference_label="Left", learner_label="Left", body_region="LEFT_HAND",
                                          establishment_basis="Explicit synthetic fixture correspondence"))


@pytest.fixture
def make_extraction():
    def create(points=None, *, label="Left", missing=(), unnormalized=(), duplicate=(),
               times=None, confidence=0.9, offset=0):
        points = points if points is not None else [(i / 10, 0) for i in range(10)]
        times = times if times is not None else [i * 40.25 for i in range(len(points))]
        frames = []
        for index, ((x, y), timestamp) in enumerate(zip(points, times)):
            x += offset
            hand = HandObservation(reported_handedness=label, handedness_confidence=confidence, landmarks=tuple(
                Landmark(index=i, coordinates=Coordinates(x=x, y=y, z=100.0)) for i in range(21)))
            hands = () if index in missing else (hand, hand) if index in duplicate else (hand,)
            norm = NormalizationResult(status="MISSING_SHOULDERS") if index in unnormalized else NormalizationResult(
                status="NORMALIZED", origin=NormalizedCoordinates(x=0.5, y=0.5), shoulder_width=1.0,
                hands=tuple(tuple(NormalizedLandmark(index=i, coordinates=NormalizedCoordinates(x=x, y=y))
                                  for i in range(21)) for _ in hands))
            frames.append(FrameObservation(frame_index=index, timestamp_ms=timestamp,
                                           raw=RawObservation(hands=hands), normalization=norm))
        return ExtractionResult(video=VideoMetadata(width=100, height=100, fps=25.0, frame_count=len(frames),
            duration_ms=times[-1] + 40, timestamp_source="decoder"), sampling=SamplingConfig(),
            normalization_config=NormalizationConfig(), frames=tuple(frames), coverage=CoverageSummary.from_frames(frames))
    return create
