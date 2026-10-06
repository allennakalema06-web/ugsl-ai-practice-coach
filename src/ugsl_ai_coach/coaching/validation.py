"""Fail-closed semantic validation against a trusted, closed-language context."""

from ugsl_ai_coach.coaching.models import CoachingContext, CoachingFeedback, accessible_text
from ugsl_ai_coach.coaching.providers.deterministic import annotations_for
from ugsl_ai_coach.domain.analysis import AnalysisStatus


class GroundingViolation(ValueError):
    """Provider output is not authorized. No unsafe feedback is returned."""


def validate_feedback(
    feedback: CoachingFeedback, context: CoachingContext, *, provider_type: str,
) -> CoachingFeedback:
    if not isinstance(feedback, CoachingFeedback):
        raise GroundingViolation("Provider must return CoachingFeedback")
    # Do not trust an instance made through unchecked model_copy/model_construct.
    feedback = CoachingFeedback.model_validate(feedback.model_dump(mode="python"))
    for field in (
        "feedback_id", "attempt_id", "analysis_id", "overall_status", "summary",
        "recommended_action", "encouragement",
    ):
        if getattr(feedback, field) != getattr(context, field):
            raise GroundingViolation(f"Unauthorized {field}")
    if feedback.overall_status != AnalysisStatus.COMPLETED and (
        feedback.strengths or feedback.corrections
    ):
        raise GroundingViolation("Non-completed analysis cannot produce performance claims")
    for category in ("strengths", "corrections", "observations"):
        expected = tuple(fact.point for fact in getattr(context, category))
        if getattr(feedback, category) != expected:
            raise GroundingViolation(f"Unauthorized {category}: source or wording mismatch")
    if feedback.visual_annotations != annotations_for(context):
        raise GroundingViolation("Unauthorized visual annotation")
    metadata = feedback.provider_metadata
    if (metadata.provider_type != provider_type or
            metadata.policy_version != context.policy_version or
            metadata.template_version != context.template_version):
        raise GroundingViolation("Unauthorized provider metadata")
    if feedback.audio_text is not None and feedback.audio_text != accessible_text(feedback):
        raise GroundingViolation("Audio must be an exact equivalent of accessible text")
    return feedback
