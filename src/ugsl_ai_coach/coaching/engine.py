"""M5 orchestration: M2 -> grounding -> provider -> validated feedback."""

from uuid import uuid4

from ugsl_ai_coach.coaching.grounding import ground_analysis
from ugsl_ai_coach.coaching.models import CoachingFeedback
from ugsl_ai_coach.coaching.providers.base import CoachingProvider
from ugsl_ai_coach.coaching.providers.deterministic import DeterministicCoachingProvider
from ugsl_ai_coach.coaching.validation import validate_feedback
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult


def generate_feedback(
    analysis: StructuredAnalysisResult, *, feedback_id: str | None = None,
    provider: CoachingProvider | None = None,
) -> CoachingFeedback:
    context = ground_analysis(analysis, feedback_id if feedback_id is not None else f"FB-{uuid4()}")
    selected_provider = provider if provider is not None else DeterministicCoachingProvider()
    feedback = selected_provider.generate(context)
    return validate_feedback(feedback, context, provider_type=selected_provider.provider_type)
