import ast
from pathlib import Path
import subprocess
import sys

import pytest

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.coaching.grounding import ground_analysis
from ugsl_ai_coach.coaching.models import CoachingContext
from ugsl_ai_coach.coaching.providers.base import CoachingProvider
from ugsl_ai_coach.coaching.providers.deterministic import DeterministicCoachingProvider
from ugsl_ai_coach.coaching.validation import GroundingViolation


def test_identical_context_deterministic(make_analysis):
    context = ground_analysis(make_analysis(), "FB-test")
    provider = DeterministicCoachingProvider()
    assert provider.generate(context) == provider.generate(context)
    assert generate_feedback(make_analysis(), feedback_id="FB-test") == provider.generate(context)


def test_default_ids_generated_only_at_boundary(make_analysis):
    first, second = generate_feedback(make_analysis()), generate_feedback(make_analysis())
    assert first.feedback_id.startswith("FB-")
    assert first.feedback_id != second.feedback_id
    assert first.model_dump(exclude={"feedback_id"}) == second.model_dump(exclude={"feedback_id"})


def test_vendor_neutral_provider_receives_only_context(make_analysis):
    class OfflineProvider:
        provider_type = "offline-test"

        def generate(self, context):
            assert isinstance(context, CoachingContext)
            assert not hasattr(context, "overall_score")
            output = DeterministicCoachingProvider().generate(context)
            return output.model_copy(update={"provider_metadata": output.provider_metadata.model_copy(
                update={"provider_type": self.provider_type},
            )})
    provider: CoachingProvider = OfflineProvider()
    feedback = generate_feedback(make_analysis(), feedback_id="FB-test", provider=provider)
    assert feedback.provider_metadata.provider_type == "offline-test"


def test_unsafe_provider_rejected_without_repair(make_analysis):
    class UnsafeProvider:
        provider_type = "deterministic"

        def generate(self, context):
            return DeterministicCoachingProvider().generate(context).model_copy(update={"summary": "Perfect signing!"})
    with pytest.raises(GroundingViolation):
        generate_feedback(make_analysis(), provider=UnsafeProvider())


def test_provider_errors_propagate_without_learner_payload(make_analysis):
    class BrokenProvider:
        provider_type = "broken"

        def generate(self, context):
            raise RuntimeError("technical provider failure")
    with pytest.raises(RuntimeError, match="technical provider failure"):
        generate_feedback(make_analysis(), provider=BrokenProvider())


def test_offline_execution_blocks_network_media_and_external_providers():
    # A fresh process avoids already-imported CV modules masking boundary violations.
    code = '''
import sys
import socket
class BlockImports:
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ("cv2", "mediapipe", "numpy", "openai", "google", "httpx", "requests",
                     "ugsl_ai_coach.cv", "ugsl_ai_coach.comparison", "pyttsx3", "gtts")
        if any(fullname == name or fullname.startswith(name + ".") for name in forbidden):
            raise AssertionError("Forbidden import: " + fullname)
sys.meta_path.insert(0, BlockImports())
def blocked(*args, **kwargs):
    raise AssertionError("Network use forbidden")
socket.socket = blocked
socket.create_connection = blocked
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.coaching.engine import generate_feedback
analysis = StructuredAnalysisResult(analysis_id="AN-000001", attempt_id="ATT-000001",
    model_version="offline", status="COMPLETED", overall_score=0.43,
    overall_confidence=1.0, findings=())
assert generate_feedback(analysis, feedback_id="FB-test").audio_text is None
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_coaching_source_has_no_media_io_tts_or_vendor_imports():
    import ugsl_ai_coach.coaching as coaching
    root = Path(coaching.__file__).parent
    allowed = {"enum", "typing", "uuid", "pydantic", "ugsl_ai_coach.coaching", "ugsl_ai_coach.domain.analysis"}
    for file in root.rglob("*.py"):
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert any(node.module == name or node.module.startswith(name + ".") for name in allowed)
            elif isinstance(node, ast.Import):
                assert all(alias.name in allowed for alias in node.names)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"open", "eval", "exec", "__import__"}
