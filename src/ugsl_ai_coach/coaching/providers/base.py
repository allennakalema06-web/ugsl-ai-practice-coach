from typing import Protocol

from ugsl_ai_coach.coaching.models import CoachingContext, CoachingFeedback


class CoachingProvider(Protocol):
    @property
    def provider_type(self) -> str: ...

    def generate(self, context: CoachingContext) -> CoachingFeedback: ...
