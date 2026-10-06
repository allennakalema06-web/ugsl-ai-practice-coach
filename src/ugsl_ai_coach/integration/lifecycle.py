"""Pure transitions; immutable snapshots require atomic adapter persistence."""

from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.integration.errors import InvalidTransition
from ugsl_ai_coach.integration.models import AnalysisJob, JobState


def validate_transition(previous: AnalysisJob, replacement: AnalysisJob) -> None:
    previous = AnalysisJob.model_validate(previous.model_dump(mode="python"))
    replacement = AnalysisJob.model_validate(replacement.model_dump(mode="python"))
    if previous.analysis_id != replacement.analysis_id or previous.submission != replacement.submission:
        raise InvalidTransition("Lifecycle updates cannot replace job identity or submission")
    legal = (
        previous.state == JobState.SUBMITTED and replacement.state == JobState.PROCESSING
    ) or (
        previous.state == JobState.PROCESSING and replacement.state.is_terminal
    )
    if not legal:
        raise InvalidTransition("Only SUBMITTED -> PROCESSING -> terminal transitions are allowed")


def transition_job(
    job: AnalysisJob, state: JobState, *,
    structured_analysis: StructuredAnalysisResult | None = None,
) -> AnalysisJob:
    job = AnalysisJob.model_validate(job.model_dump(mode="python"))
    replacement = AnalysisJob.model_validate({
        **job.model_dump(mode="python"), "state": state,
        "structured_analysis": structured_analysis,
    })
    validate_transition(job, replacement)
    return replacement
