import subprocess
import sys

import pytest
from pydantic import ValidationError

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.infrastructure.postgres.connection import DatabaseConfigurationError, connection_factory
from ugsl_ai_coach.infrastructure.postgres.persistence import PostgresAnalysisIdFactory


@pytest.mark.parametrize("url", [None, "", "   "])
def test_database_only_required_at_explicit_invocation(monkeypatch, url):
    monkeypatch.delenv("UGSL_DATABASE_URL", raising=False)
    settings = Settings(_env_file=None, database_url=url)
    with pytest.raises(DatabaseConfigurationError, match="UGSL_DATABASE_URL"):
        connection_factory(settings)


def test_config_secret_and_no_eager_connect(monkeypatch):
    monkeypatch.setenv("UGSL_DATABASE_URL", "postgresql://synthetic-secret@localhost/synthetic")
    settings = Settings(_env_file=None)
    assert "synthetic-secret" not in repr(settings)
    assert "synthetic-secret" not in str(settings.model_dump())
    assert callable(connection_factory(settings))


@pytest.mark.parametrize("field,value", [
    ("worker_lease_seconds", 0), ("worker_lease_seconds", 86401),
    ("worker_poll_seconds", 0), ("worker_poll_seconds", 61),
    ("worker_poll_seconds", float("inf")), ("worker_poll_seconds", float("nan")),
])
def test_worker_config_bounds(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_decimal_uuid_identity():
    factory = PostgresAnalysisIdFactory()
    ids = {factory.create() for _ in range(20)}
    assert len(ids) == 20
    assert all(value.startswith("AN-") and value[3:].isdigit() and len(value[3:]) >= 6 for value in ids)


def test_imports_never_connect():
    code = """
import psycopg
def reject(*args, **kwargs):
    raise AssertionError('unexpected connection')
psycopg.connect = reject
import ugsl_ai_coach.integration.handoff
import ugsl_ai_coach.infrastructure.postgres.persistence
import ugsl_ai_coach.infrastructure.postgres.coaching
import ugsl_ai_coach.infrastructure.postgres.migrate
import ugsl_ai_coach.worker.runtime
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
