import pytest
from pydantic import ValidationError

from ugsl_ai_coach.cv.models import Coordinates, Landmark, NormalizationConfig, RawObservation
from ugsl_ai_coach.cv.normalization import normalize


def point(index, x, y, z=0.25, visibility=None):
    return Landmark(index=index, coordinates=Coordinates(x=x, y=y, z=z), visibility=visibility)


def body(dx=0, dy=0, scale=1):
    return RawObservation(pose=tuple(point(i, x * scale + dx, y * scale + dy) for i, x, y in [
        (11, 0.25, 0.5), (12, 0.75, 0.5), (15, 1.0, 1.5),
    ]))


def test_origin_scale_aspect_and_raw_preservation():
    raw = body()
    before = raw.model_dump_json()
    result = normalize(raw, 200, 100, NormalizationConfig())
    assert result.status == "NORMALIZED"
    assert (result.origin.x, result.origin.y) == pytest.approx((0.5, 0.25))
    assert result.shoulder_width == pytest.approx(0.5)
    assert (result.pose[-1].coordinates.x, result.pose[-1].coordinates.y) == pytest.approx((1, 1))
    assert raw.model_dump_json() == before
    assert raw.pose[-1].coordinates.z == 0.25


@pytest.mark.parametrize("dx,dy,scale", [(0.2, -0.4, 1), (0, 0, 2), (-1, 0.3, 0.5)])
def test_translation_and_scale_invariance(dx, dy, scale):
    original = normalize(body(), 200, 100, NormalizationConfig())
    transformed = normalize(body(dx, dy, scale), 200, 100, NormalizationConfig())
    for a, b in zip(original.pose, transformed.pose):
        assert (a.coordinates.x, a.coordinates.y) == pytest.approx((b.coordinates.x, b.coordinates.y))


@pytest.mark.parametrize("indices", [(), (11,), (12,)])
def test_missing_shoulder_is_explicit(indices):
    result = normalize(RawObservation(pose=tuple(point(i, 0.5, 0.5) for i in indices)), 100, 100, NormalizationConfig())
    assert result.status == "MISSING_SHOULDERS"
    assert result.pose == () and result.hands == () and result.origin is None


@pytest.mark.parametrize("width", [0.0, 1e-9])
def test_unsafe_shoulder_scale(width):
    raw = RawObservation(pose=(point(11, 0.5, 0.5), point(12, 0.5 + width, 0.5)))
    result = normalize(raw, 100, 100, NormalizationConfig())
    assert result.status == "INVALID_SHOULDER_SCALE"
    assert result.shoulder_width is None


def test_low_visibility_anchor():
    raw = RawObservation(pose=(point(11, 0.25, 0.5, visibility=0.1), point(12, 0.75, 0.5)))
    assert normalize(raw, 100, 100, NormalizationConfig()).status == "LOW_ANCHOR_VISIBILITY"


def test_coordinates_are_not_clamped():
    result = normalize(body(), 100, 100, NormalizationConfig())
    assert result.pose[-1].coordinates.y == 2.0


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_coordinates_rejected(axis, value):
    values = {"x": 0.1, "y": 0.2, "z": 0.3, axis: value}
    with pytest.raises(ValidationError):
        Coordinates(**values)
