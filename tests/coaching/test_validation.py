import pytest
from pydantic import ValidationError

from ugsl_ai_coach.coaching.grounding import ground_analysis
from ugsl_ai_coach.coaching.models import ActionKind, RecommendedAction
from ugsl_ai_coach.coaching.providers.deterministic import DeterministicCoachingProvider
from ugsl_ai_coach.coaching.validation import GroundingViolation, validate_feedback
from ugsl_ai_coach.domain.analysis import AnalysisStatus, BodyRegion, FindingStatus, SkillCategory


def check(feedback, context):
    return validate_feedback(feedback, context, provider_type="deterministic")


@pytest.mark.parametrize("field,value", [
    ("feedback_id", "invented"), ("attempt_id", "ATT-999999"), ("analysis_id", "AN-999999"),
    ("overall_status", AnalysisStatus.FAILED),
    ("summary", "Your UgSL is 92% correct."), ("summary", "Excellent handshape."),
    ("summary", "Your movement goes upward."), ("encouragement", "You must redo this now."),
    ("encouragement", "You failed. Unlock the next level by passing."),
    ("encouragement", "Perfect signing!"), ("summary", "DTW cost is 0.4."),
    ("summary", "The room was too dark."), ("summary", "Your grammar is wrong."),
    ("audio_text", "Unique audio-only correction."),
])
def test_global_claims_rejected(make_analysis, field, value):
    context = ground_analysis(make_analysis(), "FB-test")
    feedback = DeterministicCoachingProvider().generate(context)
    with pytest.raises((GroundingViolation, ValidationError)):
        check(feedback.model_copy(update={field: value}), context)


@pytest.mark.parametrize("field,value", [
    ("finding_id", "F-999"), ("skill", SkillCategory.HANDSHAPE),
    ("body_region", BodyRegion.RIGHT_HAND), ("start_time_ms", 122),
    ("start_time_ms", 124), ("end_time_ms", 988), ("end_time_ms", 986),
    ("message", "Move your hand upward."), ("reason", "Your sign is incorrect."),
])
def test_point_source_and_wording_mismatch_rejected(make_analysis, field, value):
    context = ground_analysis(make_analysis(), "FB-test")
    feedback = DeterministicCoachingProvider().generate(context)
    point = feedback.corrections[0].model_copy(update={field: value})
    with pytest.raises((GroundingViolation, ValidationError)):
        check(feedback.model_copy(update={"corrections": (point,)}), context)


@pytest.mark.parametrize("field,value", [
    ("finding_id", "F-999"), ("skill", "DIRECTION"),
    ("body_region", BodyRegion.RIGHT_HAND), ("start_time_ms", 0),
    ("end_time_ms", 1000), ("label", "Excellent facial expression."),
])
def test_annotation_fabrication_rejected(make_analysis, field, value):
    context = ground_analysis(make_analysis(), "FB-test")
    feedback = DeterministicCoachingProvider().generate(context)
    annotation = feedback.visual_annotations[0].model_copy(update={field: value})
    with pytest.raises((GroundingViolation, ValidationError)):
        check(feedback.model_copy(update={"visual_annotations": (annotation,)}), context)


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "invent_strength", "swap_category", "omit_annotation", "duplicate_annotation"])
def test_no_omission_reclassification_or_unearned_praise(make_analysis, mutation):
    context = ground_analysis(make_analysis(), "FB-test")
    feedback = DeterministicCoachingProvider().generate(context)
    changes = {
        "omit": {"corrections": ()},
        "duplicate": {"corrections": feedback.corrections * 2},
        "invent_strength": {"strengths": feedback.corrections},
        "swap_category": {"corrections": (), "observations": feedback.corrections},
        "omit_annotation": {"visual_annotations": ()},
        "duplicate_annotation": {"visual_annotations": feedback.visual_annotations * 2},
    }
    with pytest.raises(GroundingViolation):
        check(feedback.model_copy(update=changes[mutation]), context)


@pytest.mark.parametrize("status", [AnalysisStatus.UNANALYZABLE, AnalysisStatus.FAILED])
@pytest.mark.parametrize("category", ["strengths", "corrections"])
def test_non_completed_rejects_performance_claims(make_analysis, status, category):
    context = ground_analysis(make_analysis(status), "FB-test")
    feedback = DeterministicCoachingProvider().generate(context)
    point = DeterministicCoachingProvider().generate(ground_analysis(make_analysis(), "FB-test")).corrections[0]
    with pytest.raises(GroundingViolation):
        check(feedback.model_copy(update={category: (point,)}), context)


def test_strong_cannot_authorize_correction(make_analysis, make_finding):
    context = ground_analysis(make_analysis(findings=(make_finding(FindingStatus.STRONG),)), "FB-test")
    feedback = DeterministicCoachingProvider().generate(context)
    with pytest.raises(GroundingViolation):
        check(feedback.model_copy(update={"corrections": feedback.strengths}), context)


@pytest.mark.parametrize("field", ["provider_type", "policy_version", "template_version"])
def test_metadata_matches_boundary(make_analysis, field):
    context = ground_analysis(make_analysis(), "FB-test")
    feedback = DeterministicCoachingProvider().generate(context)
    metadata = feedback.provider_metadata.model_copy(update={field: "invented"})
    with pytest.raises(GroundingViolation):
        check(feedback.model_copy(update={"provider_metadata": metadata}), context)


def test_action_is_authorized_not_just_typed(make_analysis):
    context = ground_analysis(make_analysis(), "FB-test")
    feedback = DeterministicCoachingProvider().generate(context)
    with pytest.raises(GroundingViolation):
        check(feedback.model_copy(update={"recommended_action": RecommendedAction(
            kind=ActionKind.RETRY_CAPTURE, message="You can retry because the room is dark.",
        )}), context)


def test_non_model_output_rejected(make_analysis):
    with pytest.raises(GroundingViolation):
        check({}, ground_analysis(make_analysis(), "FB-test"))
