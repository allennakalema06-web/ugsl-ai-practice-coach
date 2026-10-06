import pytest

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.coaching.models import accessible_text
from ugsl_ai_coach.domain.analysis import AnalysisStatus, FindingStatus, SkillCategory


@pytest.mark.parametrize("status", list(FindingStatus))
def test_bai_wording_is_respectful_optional_and_free_of_jargon(make_analysis, make_finding, status):
    feedback = generate_feedback(make_analysis(findings=(make_finding(status),)), feedback_id="FB-test")
    text = accessible_text(feedback).lower()
    for forbidden in (
        "bad", "poor signer", "failure", "you failed", "wrong signing", "must", "now!",
        "hurry", "unlock", "certificate", "better than", "disappoint", "perfect", "dtw",
        "algorithm", "landmark", "coordinate", "bai-coaching-v1", "movement-coaching-v1",
        "%", "score", "probability", "mastery", "pass",
    ):
        assert forbidden not in text
    assert feedback.recommended_action.message.startswith("You can ")
    assert len(feedback.corrections) <= 1
    assert len(feedback.strengths) <= 1
    assert len(feedback.observations) <= 1
    assert "correct" not in feedback.recommended_action.message


def test_truthful_difference_without_identity_judgment(make_analysis):
    feedback = generate_feedback(make_analysis(), feedback_id="FB-test")
    assert "movement path difference" in feedback.corrections[0].message
    assert "your ability" not in feedback.corrections[0].message
    assert not feedback.strengths
    assert "does not establish complete UgSL correctness" in feedback.corrections[0].reason


def test_acceptable_neutral_not_failure(make_analysis, make_finding):
    feedback = generate_feedback(make_analysis(findings=(make_finding(FindingStatus.ACCEPTABLE),)), feedback_id="FB-test")
    assert "within the current provisional comparison range" in feedback.observations[0].message
    assert not feedback.strengths and not feedback.corrections


@pytest.mark.parametrize("status", [AnalysisStatus.UNANALYZABLE, AnalysisStatus.FAILED])
def test_unknown_cause_never_guessed(make_analysis, status):
    feedback = generate_feedback(make_analysis(status), feedback_id="FB-test")
    text = accessible_text(feedback).lower()
    for token in ("dark", "lighting", "visible", "frame", "resource", "stack", "exception", "bad", "incorrect"):
        assert token not in text
    assert not feedback.strengths and not feedback.corrections
    assert "couldn't" in feedback.summary


def test_insufficient_evidence_not_weakness(make_analysis, make_finding):
    feedback = generate_feedback(make_analysis(findings=(make_finding(FindingStatus.INSUFFICIENT_EVIDENCE),)), feedback_id="FB-test")
    assert not feedback.corrections
    assert "insufficient reliable evidence" in feedback.observations[0].message
    assert "not a judgment" in feedback.observations[0].reason


def test_no_direction_enum():
    assert "DIRECTION" not in SkillCategory.__members__


@pytest.mark.parametrize("score", [0.0, 0.43, 1.0])
def test_engineering_score_not_used_in_coaching(make_analysis, score):
    feedback = generate_feedback(make_analysis(overall_score=score), feedback_id="FB-test")
    expected = generate_feedback(make_analysis(overall_score=0.5), feedback_id="FB-test")
    assert feedback == expected


def test_all_confidence_levels_remain_provisional(make_analysis, make_finding):
    for confidence in (0.0, 0.3, 0.9, 1.0):
        feedback = generate_feedback(make_analysis(findings=(make_finding(FindingStatus.STRONG, confidence=confidence),)), feedback_id="FB-test")
        assert "suggests" in feedback.strengths[0].message
        assert "does not establish" in feedback.strengths[0].reason
