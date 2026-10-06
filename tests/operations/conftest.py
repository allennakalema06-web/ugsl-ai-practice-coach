from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.integration.coaching_work import CoachingWork

TOKEN = 'm6e-test-token-0123456789ABCDEFGHIJKLMNOP'


@pytest.fixture
def settings():
    return Settings(_env_file=None, service_token=TOKEN, object_store_bucket='synthetic-private-bucket')


@pytest.fixture
def result():
    return StructuredAnalysisResult(analysis_id='AN-000001', attempt_id='ATT-000001',
        model_version='synthetic', status='COMPLETED', overall_score=0.5, overall_confidence=1.0, findings=())


@pytest.fixture
def work():
    return CoachingWork(analysis_id='AN-000001', attempt_id='ATT-000001', feedback_id='FB-stable',
        state='CLAIMED', delivery_count=1, failure_count=0,
        claimed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        lease_expires_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        next_attempt_at=datetime(2026, 1, 1, tzinfo=timezone.utc))


class Cursor:
    def __init__(self, row=None, rows=None):
        self.row, self.rows = row, rows or []
    def fetchone(self): return self.row
    def fetchall(self): return self.rows


class SnapshotDB:
    def __init__(self): self.calls = []
    def execute(self, query, params=None):
        self.calls.append(query)
        if 'operational_worker_counts' in query:
            return Cursor(rows=[dict(worker='coaching', outcome='retry', event_count=3)])
        if "count(*) AS n" in query:
            return Cursor({'n': 2})
        return Cursor({'depth': 4, 'age': 40})
    @contextmanager
    def connect(self): yield self
