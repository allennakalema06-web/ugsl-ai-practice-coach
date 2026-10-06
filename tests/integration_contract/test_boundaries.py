import ast
from pathlib import Path
import subprocess
import sys

import pytest

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.domain.analysis import AnalysisStatus, BodyRegion, Evidence, StructuredAnalysisResult, StructuredFinding
from ugsl_ai_coach.integration.models import AnalysisJob, JobState
from ugsl_ai_coach.integration.service import CoachingIntegrationService
from .fakes import FakeCoachingRepository, FakeRepository


def movement_result():
    return StructuredAnalysisResult(
        analysis_id="AN-000001", attempt_id="ATT-000001", model_version="synthetic-movement-v1",
        status=AnalysisStatus.COMPLETED, overall_score=0.43, overall_confidence=1.0,
        findings=(StructuredFinding(
            finding_id="F-001", skill="MOVEMENT", status="NEEDS_IMPROVEMENT",
            severity=1, confidence=1.0, body_region=BodyRegion.LEFT_HAND,
            start_time_ms=123, end_time_ms=987,
            evidence=Evidence(expected_value="synthetic reference", observed_value="synthetic observation", deviation=0.4),
        ),),
    )


def test_real_finding_traceability_through_integration(submitted):
    result = movement_result()
    feedback = generate_feedback(result, feedback_id="synthetic-feedback")
    job = AnalysisJob.model_validate({**submitted.model_dump(), "state": JobState.COMPLETED,
                                      "structured_analysis": result})
    jobs = FakeRepository()
    jobs.jobs[job.analysis_id] = job
    record = CoachingIntegrationService(jobs, FakeCoachingRepository()).record_feedback(job.analysis_id, feedback)
    point = record.feedback.corrections[0]
    source = job.structured_analysis.findings[0]
    assert point.finding_id == source.finding_id
    assert (point.skill, point.body_region, point.start_time_ms, point.end_time_ms) == (
        source.skill, source.body_region, source.start_time_ms, source.end_time_ms,
    )
    assert job.attempt_id == feedback.attempt_id == result.attempt_id
    assert job.analysis_id == feedback.analysis_id == result.analysis_id
    assert job.structured_analysis == result


@pytest.mark.parametrize("field,value", [
    ("finding_id", "F-999"), ("body_region", BodyRegion.RIGHT_HAND),
    ("start_time_ms", 0), ("end_time_ms", 1000), ("message", "Excellent handshape."),
])
def test_integration_does_not_loosen_m5_source_validation(submitted, field, value):
    result = movement_result()
    feedback = generate_feedback(result, feedback_id="synthetic-feedback")
    point = feedback.corrections[0].model_copy(update={field: value})
    feedback = feedback.model_copy(update={"corrections": (point,)})
    jobs = FakeRepository()
    jobs.jobs[submitted.analysis_id] = AnalysisJob.model_validate({**submitted.model_dump(),
        "state": "COMPLETED", "structured_analysis": result})
    records = FakeCoachingRepository()
    with pytest.raises(ValueError):
        CoachingIntegrationService(jobs, records).record_feedback(submitted.analysis_id, feedback)
    assert not records.records


def test_no_new_mounted_http_endpoint():
    from ugsl_ai_coach.main import create_app
    app = create_app()
    assert set(app.openapi()["paths"]) == {
        "/api/v1/health", "/api/v1/contracts/analysis",
    }


def test_integration_source_does_not_process_media_generate_feedback_or_choose_infrastructure():
    import ugsl_ai_coach.integration as integration
    root = Path(integration.__file__).parent
    allowed = {"enum", "typing", "pydantic", "ugsl_ai_coach.integration", "ugsl_ai_coach.domain.analysis",
               "ugsl_ai_coach.coaching.models", "ugsl_ai_coach.coaching.grounding", "ugsl_ai_coach.coaching.validation"}
    for source in root.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert any(node.module == name or node.module.startswith(name + ".") for name in allowed)
            elif isinstance(node, ast.Import):
                assert all(alias.name in allowed for alias in node.names)
            elif isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                assert name not in {"open", "generate_feedback", "compare_movement", "extract_video", "generate", "eval", "exec"}


def test_fresh_process_works_without_network_cv_queues_or_vendor_sdks():
    code = '''
import sys
import socket
class BlockImports:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("cv2", "mediapipe", "numpy", "ugsl_ai_coach.cv", "ugsl_ai_coach.comparison",
                   "ugsl_ai_coach.coaching.engine", "redis", "celery", "sqlalchemy", "boto3",
                   "openai", "google", "langchain", "gtts", "pyttsx3", "concurrent.futures",
                   "multiprocessing")
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            raise AssertionError("Forbidden import: " + fullname)
sys.meta_path.insert(0, BlockImports())
def blocked(*args, **kwargs):
    raise AssertionError("Network access forbidden")
socket.socket = blocked
socket.create_connection = blocked
from ugsl_ai_coach.integration.service import AnalysisIntegrationService, CoachingIntegrationService
from ugsl_ai_coach.integration.models import AnalysisSubmission
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.coaching.grounding import ground_analysis
from ugsl_ai_coach.coaching.providers.deterministic import DeterministicCoachingProvider
from tests.integration_contract.fakes import FakeRepository, FakeDispatcher, FakeIdFactory, FakeCoachingRepository
jobs = FakeRepository()
service = AnalysisIntegrationService(jobs, FakeDispatcher(), FakeIdFactory())
submission = AnalysisSubmission(attempt_id="ATT-000001", learner_video_ref="synthetic-video",
    reference_profile_ref="synthetic-profile", idempotency_key="synthetic-key")
job = service.submit(submission)
service.mark_processing(job.analysis_id)
result = StructuredAnalysisResult(analysis_id=job.analysis_id, attempt_id=job.attempt_id,
    model_version="synthetic", status="FAILED", findings=())
assert service.complete(job.analysis_id, result).state.value == "FAILED"
assert service.submit(submission).state.value == "FAILED"
feedback = DeterministicCoachingProvider().generate(ground_analysis(result, "synthetic-feedback"))
records = FakeCoachingRepository()
coaching = CoachingIntegrationService(jobs, records)
before = service.get(job.analysis_id).model_dump_json()
assert coaching.record_feedback(job.analysis_id, feedback).feedback == feedback
assert service.get(job.analysis_id).model_dump_json() == before
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
