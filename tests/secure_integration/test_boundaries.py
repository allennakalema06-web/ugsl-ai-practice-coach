import json
import subprocess
import sys

import pytest

from ugsl_ai_coach.api.auth import StaticTokenAuthenticator, AuthenticationFailed
from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.worker.processor import create_production_processor
from ugsl_ai_coach.media.references import MediaConfigurationError


def test_imports_do_not_contact_database_or_storage():
    script = """
import psycopg, boto3, socket
def reject(*args, **kwargs):
    raise AssertionError('unexpected infrastructure I/O')
psycopg.connect = reject
boto3.client = reject
socket.create_connection = reject
import ugsl_ai_coach.main
import ugsl_ai_coach.infrastructure.object_store.s3
import ugsl_ai_coach.worker.processor
assert 'mediapipe' not in __import__('sys').modules
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_secret_comparison_is_constant_time(settings, monkeypatch):
    import ugsl_ai_coach.api.auth as module
    original = module.compare_digest
    calls = []
    def compare(a, b):
        calls.append((len(a), len(b)))
        return original(a, b)
    monkeypatch.setattr(module, "compare_digest", compare)
    authenticator = StaticTokenAuthenticator(settings)
    authenticator.authenticate(settings.service_token.get_secret_value())
    with pytest.raises(AuthenticationFailed):
        authenticator.authenticate("invalid")
    assert len(calls) == 2
    assert settings.service_token.get_secret_value() not in repr(settings)
    assert settings.service_token.get_secret_value() not in str(settings.model_dump())


def test_missing_models_fail_only_on_explicit_processor_invocation(settings):
    with pytest.raises(MediaConfigurationError):
        create_production_processor(settings)


@pytest.mark.parametrize("field,value", [("max_video_bytes", 0), ("max_video_bytes", 1024**3 + 1),
    ("max_reference_bytes", 0), ("max_reference_bytes", 50 * 1024**2 + 1)])
def test_media_size_configuration_is_bounded(field, value):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})
