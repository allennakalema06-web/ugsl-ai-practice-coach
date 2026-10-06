from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from ugsl_ai_coach.infrastructure.postgres.rate_limit import PostgresRateLimiter
from ugsl_ai_coach.infrastructure.postgres.coaching_work import PostgresCoachingWork
from ugsl_ai_coach.integration.handoff.lifecycle import ClaimConflict
from .conftest import Cursor


class LimiterDB:
    def __init__(self, count=1, second=15, existing=True, expired=False):
        self.now = datetime(2026, 1, 1, 0, 1, second, tzinfo=timezone.utc)
        self.window = self.now.replace(second=0, microsecond=0)
        self.count, self.existing, self.expired = count, existing, expired
        self.calls = []
    def execute(self, query, params=None):
        self.calls.append((query, params))
        if 'INSERT' in query: return Cursor(None if self.existing else {'principal_digest': 'inserted'})
        if 'FOR UPDATE' in query:
            return Cursor({'request_count': self.count, 'window_start': self.window.replace(minute=0) if self.expired else self.window})
        if query.startswith('SELECT clock_timestamp'): return Cursor({'now': self.now})
        return Cursor()
    @contextmanager
    def connect(self): yield self


@pytest.mark.parametrize('count,existing,expired,allowed', [(1, False, False, True), (1, True, False, True),
    (2, True, False, False), (9999, True, False, False), (9999, True, True, True)])
def test_rate_threshold_window_and_bounded_counter(count, existing, expired, allowed):
    db = LimiterDB(count, existing=existing, expired=expired)
    result = PostgresRateLimiter(db.connect).check('trusted-service', 'submit', 2)
    assert result.allowed == allowed and result.retry_after == 45
    assert 'FOR UPDATE' in db.calls[1][0]
    assert db.calls[2][0] == 'SELECT clock_timestamp() AS now'
    assert db.calls[-1][1][1] <= 3
    assert 'trusted-service' not in str(db.calls)
    assert len(db.calls[0][1][0]) == 64


@pytest.mark.parametrize('second,retry', [(0, 60), (59, 1)])
def test_retry_after_db_clock(second, retry):
    db = LimiterDB(second=second)
    assert PostgresRateLimiter(db.connect).check('trusted', 'read', 1).retry_after == retry


def test_coaching_stale_generation_and_db_time_before_mutation(work):
    class DB:
        calls = []
        def execute(self, query, params=None):
            self.calls.append(query)
            if 'FOR UPDATE' in query: return Cursor(work.model_dump())
            return Cursor({'now': work.claimed_at})
        @contextmanager
        def connect(self): yield self
    db = DB()
    repo = PostgresCoachingWork(db.connect)
    with pytest.raises(ClaimConflict):
        with db.connect() as conn:
            current = repo._lock(conn, work)
            repo._active(conn, current, work.model_copy(update={'delivery_count': 2}))
    assert db.calls == ['SELECT * FROM coaching_work WHERE analysis_id = %s FOR UPDATE', 'SELECT clock_timestamp() AS now']


@pytest.mark.parametrize('now_field', ['lease_expires_at'])
def test_coaching_expired_lease_refused(work, now_field):
    class DB:
        def execute(self, *args): return Cursor({'now': getattr(work, now_field)})
    with pytest.raises(ClaimConflict): PostgresCoachingWork(None)._active(DB(), work, work)
