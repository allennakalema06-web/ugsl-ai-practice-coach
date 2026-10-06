"""Internal video-to-observations orchestration; no comparison or judgments."""

from pathlib import Path

from ugsl_ai_coach.cv.landmarks import LandmarkExtractor
from ugsl_ai_coach.cv.models import (
    CoverageSummary, ExtractionResult, FrameObservation,
    NormalizationConfig, SamplingConfig,
)
from ugsl_ai_coach.cv.normalization import normalize
from ugsl_ai_coach.cv.sampling import sample_frames
from ugsl_ai_coach.cv.video import decode_frames, inspect_video


def extract_video(path: str | Path, extractor: LandmarkExtractor,
                  sampling: SamplingConfig | None = None,
                  normalization: NormalizationConfig | None = None) -> ExtractionResult:
    sampling = sampling if sampling is not None else SamplingConfig()
    normalization = normalization if normalization is not None else NormalizationConfig()
    metadata = inspect_video(path)
    frames = []
    decoded = decode_frames(path, metadata)
    try:
        for frame in sample_frames(decoded, sampling):
            raw = extractor.extract(frame)
            frames.append(FrameObservation(
                frame_index=frame.index, timestamp_ms=frame.timestamp_ms, raw=raw,
                normalization=normalize(raw, metadata.width, metadata.height, normalization),
            ))
    finally:
        decoded.close()
    coverage = CoverageSummary.from_frames(frames)
    return ExtractionResult(video=metadata, sampling=sampling, normalization_config=normalization,
                            frames=tuple(frames), coverage=coverage)
