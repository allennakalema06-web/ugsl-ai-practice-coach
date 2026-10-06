import pytest

from ugsl_ai_coach.integration.errors import InvalidTransition
from ugsl_ai_coach.integration.lifecycle import transition_job, validate_transition
from ugsl_ai_coach.integration.models import AnalysisJob, JobState


def state_job(submitted, state, make_result):
    return AnalysisJob.model_validate({**submitted.model_dump(), "state": state,
        "structured_analysis": make_result(state.value) if state.is_terminal else None})


@pytest.mark.parametrize("previous,next_state", [(a, b) for a in JobState for b in JobState])
def test_all_transition_pairs(submitted, make_result, previous, next_state):
    before = state_job(submitted, previous, make_result)
    after = state_job(submitted, next_state, make_result)
    legal = previous == JobState.SUBMITTED and next_state == JobState.PROCESSING or (
        previous == JobState.PROCESSING and next_state.is_terminal
    )
    if legal:
        validate_transition(before, after)
    else:
        with pytest.raises(InvalidTransition):
            validate_transition(before, after)


@pytest.mark.parametrize("state", [s for s in JobState if s.is_terminal])
def test_pure_transitions_preserve_history(submitted, make_result, state):
    original = submitted.model_dump_json()
    processing = transition_job(submitted, JobState.PROCESSING)
    terminal = transition_job(processing, state, structured_analysis=make_result(state.value))
    assert submitted.model_dump_json() == original
    assert submitted.state == JobState.SUBMITTED
    assert processing.state == JobState.PROCESSING
    assert terminal.submission == submitted.submission
    assert terminal.analysis_id == submitted.analysis_id


@pytest.mark.parametrize("field,value", [
    ("analysis_id", "AN-000002"), ("attempt_id", "ATT-000002"),
    ("learner_video_ref", "another-video"), ("reference_profile_ref", "another-profile"),
    ("idempotency_key", "another-key"),
])
def test_transition_cannot_reassign_identity_or_media(submitted, field, value):
    replacement = AnalysisJob.model_validate({**submitted.model_dump(), "state": "PROCESSING", field: value})
    with pytest.raises(InvalidTransition, match="identity"):
        validate_transition(submitted, replacement)
