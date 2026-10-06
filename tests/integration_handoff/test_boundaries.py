import subprocess
import sys


def test_fresh_process_handoff_needs_no_network_workers_or_infrastructure():
    code = '''
import sys
import socket
class BlockImports:
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ("cv2", "mediapipe", "numpy", "ugsl_ai_coach.cv", "ugsl_ai_coach.comparison",
                     "ugsl_ai_coach.coaching.engine", "redis", "celery", "sqlalchemy", "sqlite3",
                     "boto3", "google", "openai", "kafka", "pika", "multiprocessing",
                     "concurrent.futures", "gtts", "pyttsx3")
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("Forbidden import: " + fullname)
sys.meta_path.insert(0, BlockImports())
def blocked(*args, **kwargs):
    raise AssertionError("Network access forbidden")
socket.socket = blocked
socket.create_connection = blocked
from ugsl_ai_coach.integration.handoff.service import AnalysisHandoffService
from ugsl_ai_coach.integration.models import AnalysisSubmission
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from tests.integration_handoff.fakes import FakePersistence, FakeIdFactory, FakeStore
persistence = FakePersistence()
factory = FakeIdFactory()
service = AnalysisHandoffService(persistence, factory)
submission = AnalysisSubmission(attempt_id="ATT-000001", learner_video_ref="synthetic-video",
    reference_profile_ref="synthetic-reference", idempotency_key="synthetic-key")
service.submit(submission)
claim = service.claim_next(now_ms=0, lease_duration_ms=10)
service.begin(claim, now_ms=1)
restarted = AnalysisHandoffService(FakePersistence(FakeStore.restore(persistence.store.snapshot())), factory)
claim = restarted.claim_next(now_ms=10, lease_duration_ms=10)
restarted.begin(claim, now_ms=11)
result = StructuredAnalysisResult(analysis_id=claim.analysis_id, attempt_id=claim.attempt_id,
    model_version="synthetic", status="FAILED", findings=())
assert restarted.complete(claim, result, now_ms=12).structured_analysis == result
assert restarted.claim_next(now_ms=100, lease_duration_ms=10) is None
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
