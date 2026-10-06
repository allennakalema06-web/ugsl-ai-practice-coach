"""Versioned closed-language BAI policy; no metric interpretation or new scoring."""

from typing import Literal

from ugsl_ai_coach.coaching.models import ActionKind, RecommendedAction
from ugsl_ai_coach.domain.analysis import ContractModel, FindingStatus

CONSTRAINTS = (
    "Explain only supplied evidence; never independently evaluate UgSL or create findings.",
    "Only MOVEMENT is supported; no direction, handshape, orientation, location, timing, "
    "sequence, movement range, body position, grammar, meaning or non-manual claims.",
    "Preserve source finding identifiers, skill, body region, severity and timestamps.",
    "Engineering similarity and evidence confidence are not grades, mastery or correctness probabilities.",
    "Communicate uncertainty; insufficient evidence warrants generic retry, never a guessed cause.",
    "Preserve agency; no commands, urgency, progression locks, shame, threats, guilt or manipulation.",
    "Prioritize using supplied severity and confidence; provide one optional next action.",
    "Keep technical metrics outside learner text; audio may only repeat complete accessible text.",
    "Use only authorized v1 wording; reject all unsupported paraphrases rather than repair claims.",
)


class CoachingPolicy(ContractModel):
    policy_version: Literal["bai-coaching-v1"] = "bai-coaching-v1"
    template_version: Literal["movement-coaching-v1"] = "movement-coaching-v1"
    # A fixed bound is a presentation choice, not a performance threshold.
    maximum_points_per_category: Literal[1] = 1
    constraints: tuple[str, ...] = CONSTRAINTS


POLICY = CoachingPolicy()
LIMIT = "This provisional movement comparison does not establish complete UgSL correctness."
ENCOURAGEMENT = "Practice can be taken at your own pace."
COMPLETED_SUMMARY = "The analysis finished. The available movement evidence is described below."
EMPTY_SUMMARY = "The analysis finished, but there are no supported movement coaching points for this attempt."
UNCERTAIN_SUMMARY = "The analysis finished. Some movement evidence is limited; the available observations are described below."
UNANALYZABLE_SUMMARY = "I couldn't get enough reliable movement evidence from this attempt."
FAILED_SUMMARY = "I couldn't complete the analysis this time."

# No evidence strings or deviation are interpreted here. Status is authoritative.
MESSAGES = {
    FindingStatus.STRONG: "The available evidence suggests your movement path was close to the reference in this attempt.",
    FindingStatus.ACCEPTABLE: "The available evidence suggests the observed movement was within the current provisional comparison range.",
    FindingStatus.NEEDS_IMPROVEMENT: "The available evidence indicates a movement path difference from the reference in this attempt.",
    FindingStatus.WARNING: "The analysis flags movement evidence for review; it does not establish a performance correction.",
    FindingStatus.INSUFFICIENT_EVIDENCE: "There is insufficient reliable evidence to describe movement performance in this interval.",
}
REASONS = {
    **{status: LIMIT for status in (
        FindingStatus.STRONG, FindingStatus.ACCEPTABLE, FindingStatus.NEEDS_IMPROVEMENT,
    )},
    FindingStatus.WARNING: "The supplied warning does not establish a specific cause or movement change to make.",
    FindingStatus.INSUFFICIENT_EVIDENCE: "Missing evidence is not a judgment of your signing ability.",
}


def message_for(status: FindingStatus, *, limited_evidence: bool) -> str:
    """Incomplete sufficiency gets cautious wording, without a new score band."""
    if limited_evidence and status in (
        FindingStatus.STRONG, FindingStatus.ACCEPTABLE, FindingStatus.NEEDS_IMPROVEMENT,
    ):
        return MESSAGES[status].replace("The available evidence suggests", "The limited movement evidence suggests").replace(
            "The available evidence indicates", "The limited movement evidence suggests"
        )
    return MESSAGES[status]


ACTIONS = {
    ActionKind.REVIEW_MOVEMENT: RecommendedAction(
        kind=ActionKind.REVIEW_MOVEMENT,
        message="You can review the reference movement before another attempt.",
    ),
    ActionKind.RETRY_ATTEMPT: RecommendedAction(
        kind=ActionKind.RETRY_ATTEMPT,
        message="You can try another attempt when you're ready.",
    ),
    ActionKind.RETRY_CAPTURE: RecommendedAction(
        kind=ActionKind.RETRY_CAPTURE,
        message="You can try recording the attempt again when you're ready.",
    ),
    ActionKind.TRY_LATER: RecommendedAction(
        kind=ActionKind.TRY_LATER,
        message="You can try the analysis again later.",
    ),
}
