import pytest

from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem, WorkState
from ugsl_ai_coach.integration.handoff.service import AnalysisHandoffService
from ugsl_ai_coach.integration.models import AnalysisSubmission
from .fakes import FakeIdFactory, FakePersistence


@pytest.fixture
def submission():
    return AnalysisSubmission(attempt_id="ATT-000001", learner_video_ref="synthetic-video-1",
                              reference_profile_ref="synthetic-reference-1", idempotency_key="synthetic-key-1")


@pytest.fixture
def pending(submission):
    return AnalysisWorkItem(analysis_id="AN-000001", attempt_id=submission.attempt_id,
        learner_video_ref=submission.learner_video_ref, reference_profile_ref=submission.reference_profile_ref,
        state=WorkState.PENDING, delivery_count=0)


@pytest.fixture
def wiring():
    persistence, factory = FakePersistence(), FakeIdFactory()
    return AnalysisHandoffService(persistence, factory), persistence, factory


@pytest.fixture
def make_result():
    def make(status="COMPLETED", **overrides):
        values = dict(analysis_id="AN-000001", attempt_id="ATT-000001", model_version="synthetic-analysis-v1",
                      status=status, overall_score=0.43 if status == "COMPLETED" else None,
                      overall_confidence=None if status == "FAILED" else 0.9, findings=())
        values.update(overrides)
        return StructuredAnalysisResult(**values)
    return make
