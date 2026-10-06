"""Offline provider; assembles only facts authorized by trusted grounding."""

from ugsl_ai_coach.coaching.models import (
    CoachingContext, CoachingFeedback, ProviderMetadata, VisualAnnotation,
    accessible_text,
)


def annotations_for(context: CoachingContext) -> tuple[VisualAnnotation, ...]:
    return tuple(VisualAnnotation(
        finding_id=fact.point.finding_id, skill=fact.point.skill,
        body_region=fact.point.body_region,
        start_time_ms=fact.point.start_time_ms, end_time_ms=fact.point.end_time_ms,
        label=fact.point.message,
    ) for fact in (*context.corrections, *context.strengths, *context.observations))


class DeterministicCoachingProvider:
    provider_type = "deterministic"

    def __init__(self, *, include_audio_text: bool = False) -> None:
        self.include_audio_text = include_audio_text

    def generate(self, context: CoachingContext) -> CoachingFeedback:
        feedback = CoachingFeedback(
            feedback_id=context.feedback_id, attempt_id=context.attempt_id,
            analysis_id=context.analysis_id, overall_status=context.overall_status,
            summary=context.summary,
            strengths=tuple(f.point for f in context.strengths),
            corrections=tuple(f.point for f in context.corrections),
            observations=tuple(f.point for f in context.observations),
            recommended_action=context.recommended_action,
            encouragement=context.encouragement,
            visual_annotations=annotations_for(context),
            provider_metadata=ProviderMetadata(
                provider_type=self.provider_type, policy_version=context.policy_version,
                template_version=context.template_version,
            ),
        )
        if self.include_audio_text:
            feedback = feedback.model_copy(update={"audio_text": accessible_text(feedback)})
        return feedback
