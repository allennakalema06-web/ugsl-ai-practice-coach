import os
import re
from contextlib import contextmanager
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.infrastructure.postgres.migrate import migrate
from ugsl_ai_coach.infrastructure.postgres.persistence import PostgresAnalysisPersistence
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem, JobWorkPair
from ugsl_ai_coach.integration.models import AnalysisJob


@pytest.fixture
def make_pair():
    def make(number=1, **overrides):
        values = dict(analysis_id=f"AN-{number:06d}", attempt_id=f"ATT-{number:06d}",
                      learner_video_ref=f"synthetic-video-{number}",
                      reference_profile_ref=f"synthetic-profile-{number}",
                      idempotency_key=f"synthetic-key-{number}", state="SUBMITTED")
        values.update(overrides)
        job = AnalysisJob(**values)
        work = AnalysisWorkItem(analysis_id=job.analysis_id, attempt_id=job.attempt_id,
                                learner_video_ref=job.learner_video_ref,
                                reference_profile_ref=job.reference_profile_ref,
                                state="PENDING", delivery_count=0)
        return JobWorkPair(job=job, work=work)
    return make


@pytest.fixture
def make_result():
    def make(status="COMPLETED", **overrides):
        values = dict(analysis_id="AN-000001", attempt_id="ATT-000001", model_version="synthetic-v1",
                      status=status, overall_score=0.43 if status == "COMPLETED" else None,
                      overall_confidence=None if status == "FAILED" else 0.9, findings=())
        values.update(overrides)
        return StructuredAnalysisResult(**values)
    return make


@pytest.fixture
def db():
    url = os.environ.get("UGSL_TEST_DATABASE_URL", "").strip()
    if not url:
        pytest.skip("Real PostgreSQL requires explicit UGSL_TEST_DATABASE_URL")
    # Each test owns only its random schema, never existing application tables.
    # SQL.Identifier performs safe identifier composition, not string formatting.
    schema = "ugsl_m6c_test_" + uuid4().hex
    with psycopg.connect(url, row_factory=dict_row, connect_timeout=10) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    @contextmanager
    def connect():
        with psycopg.connect(url, row_factory=dict_row, connect_timeout=10) as conn:
            conn.execute("SELECT set_config('search_path', %s, false)", (schema,))
            conn.execute("SELECT set_config('statement_timeout', '10000', false)")
            yield conn
    try:
        yield connect
    finally:
        assert re.fullmatch(r"ugsl_m6c_test_[a-f0-9]{32}", schema)
        with psycopg.connect(url, row_factory=dict_row, connect_timeout=10) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def repo(db):
    migrate(db)
    return PostgresAnalysisPersistence(db)
