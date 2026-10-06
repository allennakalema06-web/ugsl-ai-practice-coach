import pytest
from pydantic import ValidationError

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.coaching.grounding import ground_analysis
from ugsl_ai_coach.coaching.models import CoachingFeedback, CoachingPoint, VisualAnnotation


@pytest.mark.parametrize("field,value", [
    ("feedback_id", ""), ("feedback_id", "  "), ("feedback_id", 3),
    ("analysis_id", "AN-1"), ("analysis_id", ""), ("attempt_id", "ATT-1"),
    ("summary", ""), ("encouragement", " "), ("audio_text", ""), ("unexpected", True),
])
def test_feedback_rejects_invalid_fields(make_analysis, field, value):
    values = generate_feedback(make_analysis(), feedback_id="FB-test").model_dump(mode="python")
    values[field] = value
    with pytest.raises(ValidationError):
        CoachingFeedback.model_validate(values)


@pytest.mark.parametrize("field,value", [
    ("finding_id", ""), ("finding_id", "F-1"), ("skill", "DIRECTION"),
    ("skill", "HANDSHAPE"), ("start_time_ms", -1), ("start_time_ms", "123"),
    ("start_time_ms", 0.5), ("end_time_ms", 1), ("message", ""), ("reason", " "),
    ("extra", 1),
])
def test_point_contract(make_analysis, field, value):
    values = generate_feedback(make_analysis(), feedback_id="FB-test").corrections[0].model_dump()
    values[field] = value
    with pytest.raises(ValidationError):
        CoachingPoint.model_validate(values)


@pytest.mark.parametrize("field,value", [("end_time_ms", 0), ("label", " "), ("extra", 1)])
def test_annotation_contract(make_analysis, field, value):
    values = generate_feedback(make_analysis(), feedback_id="FB-test").visual_annotations[0].model_dump()
    values[field] = value
    with pytest.raises(ValidationError):
        VisualAnnotation.model_validate(values)


def test_frozen_nested_models_and_roundtrip(make_analysis):
    context = ground_analysis(make_analysis(), "FB-test")
    feedback = generate_feedback(make_analysis(), feedback_id="FB-test")
    objects = [feedback, feedback.corrections[0], feedback.recommended_action,
               feedback.visual_annotations[0], feedback.provider_metadata,
               context, context.corrections[0], context.corrections[0].point]
    for obj in objects:
        field = next(iter(type(obj).model_fields))
        with pytest.raises(ValidationError):
            setattr(obj, field, getattr(obj, field))
        with pytest.raises(ValidationError):
            type(obj).model_validate({**obj.model_dump(), "extra": "forbidden"})
    assert isinstance(feedback.corrections, tuple)
    assert CoachingFeedback.model_validate_json(feedback.model_dump_json()) == feedback
