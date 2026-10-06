import pytest

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.integration.models import AnalysisJob, AnalysisSubmission, JobState


@pytest.fixture
def submission():
    return AnalysisSubmission(
        attempt_id="ATT-000001", learner_video_ref="synthetic-video-1",
        reference_profile_ref="synthetic-profile-1", idempotency_key="synthetic-key-1",
    )


@pytest.fixture
def submitted(submission):
    return AnalysisJob(analysis_id="AN-000001", state=JobState.SUBMITTED, **submission.model_dump())


@pytest.fixture
def make_result():
    def make(status="COMPLETED", *, analysis_id="AN-000001", attempt_id="ATT-000001"):
        return StructuredAnalysisResult(
            analysis_id=analysis_id, attempt_id=attempt_id, model_version="synthetic-m2-v1",
            status=status, overall_score=0.43 if status == "COMPLETED" else None,
            overall_confidence=None if status == "FAILED" else 0.9,
            findings=(),
        )
    return make


@pytest.fixture
def make_feedback():
    def make(result):
        return generate_feedback(result, feedback_id="synthetic-feedback-1")
    return make
