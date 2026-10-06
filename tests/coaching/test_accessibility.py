import pytest

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.coaching.models import accessible_text
from ugsl_ai_coach.coaching.providers.deterministic import DeterministicCoachingProvider
from ugsl_ai_coach.domain.analysis import AnalysisStatus, FindingStatus


@pytest.mark.parametrize("status", list(AnalysisStatus))
def test_text_is_complete_without_audio_and_optional_audio_is_equivalent(make_analysis, status):
    analysis = make_analysis(status)
    feedback = generate_feedback(analysis, feedback_id="FB-test")
    spoken = generate_feedback(analysis, feedback_id="FB-test", provider=DeterministicCoachingProvider(include_audio_text=True))
    assert feedback.audio_text is None
    assert spoken.audio_text == accessible_text(feedback)
    assert spoken.model_dump(exclude={"audio_text"}) == feedback.model_dump(exclude={"audio_text"})
    assert feedback.summary in accessible_text(feedback)
    assert feedback.recommended_action.message in accessible_text(feedback)
    assert feedback.encouragement in accessible_text(feedback)


def test_every_point_and_annotation_exists_in_accessible_text(make_analysis, make_finding):
    analysis = make_analysis(findings=(
        make_finding(),
        make_finding(FindingStatus.STRONG, finding_id="F-002"),
        make_finding(FindingStatus.INSUFFICIENT_EVIDENCE, finding_id="F-003"),
    ))
    feedback = generate_feedback(analysis, feedback_id="FB-test")
    text = accessible_text(feedback)
    for point in (*feedback.corrections, *feedback.strengths, *feedback.observations):
        assert point.message in text and point.reason in text
    for annotation in feedback.visual_annotations:
        source = next(f for f in analysis.findings if f.finding_id == annotation.finding_id)
        assert (annotation.start_time_ms, annotation.end_time_ms, annotation.body_region) == (
            source.start_time_ms, source.end_time_ms, source.body_region,
        )
        assert annotation.label in text
