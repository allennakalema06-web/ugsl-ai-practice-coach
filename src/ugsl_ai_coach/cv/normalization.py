"""Planar signer-relative coordinates; raw z keeps its model-local meaning."""

import math

from ugsl_ai_coach.cv.models import (
    NormalizationConfig, NormalizationResult, NormalizationStatus,
    NormalizedCoordinates, NormalizedLandmark, RawObservation,
)


def normalize(raw: RawObservation, width: int, height: int,
              config: NormalizationConfig) -> NormalizationResult:
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    anchors = {p.index: p for p in raw.pose}
    if 11 not in anchors or 12 not in anchors:
        return NormalizationResult(status=NormalizationStatus.MISSING_SHOULDERS)
    left, right = anchors[11], anchors[12]
    if any(value is not None and value < config.minimum_anchor_visibility
           for p in (left, right) for value in (p.visibility, p.presence)):
        return NormalizationResult(status=NormalizationStatus.LOW_ANCHOR_VISIBILITY)
    aspect = height / width
    origin_x = left.coordinates.x / 2 + right.coordinates.x / 2
    origin_y = (left.coordinates.y / 2 + right.coordinates.y / 2) * aspect
    scale = math.hypot(right.coordinates.x - left.coordinates.x,
                       (right.coordinates.y - left.coordinates.y) * aspect)
    if not all(math.isfinite(v) for v in (scale, origin_x, origin_y)) or scale <= config.minimum_shoulder_width:
        return NormalizationResult(status=NormalizationStatus.INVALID_SHOULDER_SCALE)

    def transform(points):
        return tuple(NormalizedLandmark(index=p.index, coordinates=NormalizedCoordinates(
            x=(p.coordinates.x - origin_x) / scale,
            y=(p.coordinates.y * aspect - origin_y) / scale,
        )) for p in points)

    return NormalizationResult(
        status=NormalizationStatus.NORMALIZED,
        origin=NormalizedCoordinates(x=origin_x, y=origin_y), shoulder_width=scale,
        pose=transform(raw.pose), hands=tuple(transform(h.landmarks) for h in raw.hands),
    )
