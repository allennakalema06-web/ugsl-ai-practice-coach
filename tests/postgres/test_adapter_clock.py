"""Offline tests of the adapter's authoritative-clock and fencing boundary."""

from contextlib import contextmanager

import pytest

from ugsl_ai_coach.infrastructure.postgres import persistence as module
from ugsl_ai_coach.infrastructure.postgres.serialization import to_timestamp
from ugsl_ai_coach.integration.handoff.lifecycle import ClaimConflict, claim_work
from ugsl_ai_coach.integration.handoff.models import JobWorkPair
from ugsl_ai_coach.integration.lifecycle import transition_job
from ugsl_ai_coach.integration.models import JobState


class Cursor:
    rowcount = 1
    def __init__(self, row):
        self.row = row
    def fetchone(self):
        return self.row


class Connection:
    def __init__(self, now=100):
        self.now = now
        self.statements = []
        self.rolled_back = False
    def execute(self, query, params=None):
        self.statements.append((query, params))
        return Cursor({"now": to_timestamp(self.now)} if "date_trunc" in query else {"analysis_id": "AN-000001"})
    @contextmanager
    def connect(self):
        try:
            yield self
        except BaseException:
            self.rolled_back = True
            raise


def wiring(monkeypatch, make_pair, now=100):
    pair = make_pair()
    work = claim_work(pair.work, now_ms=100, lease_duration_ms=1000)
    pair = JobWorkPair(job=pair.job, work=work)
    conn = Connection(now)
    monkeypatch.setattr(module, "read_pair", lambda conn, analysis_id: pair)
    return module.PostgresAnalysisPersistence(conn.connect), pair, conn


@pytest.mark.parametrize("hint", [None, 0, 9999999999999])
def test_begin_uses_db_clock_not_caller_hint(monkeypatch, make_pair, hint):
    repo, pair, conn = wiring(monkeypatch, make_pair, now=101)
    assert repo.begin(pair.work, now_ms=hint).job.state == "PROCESSING"
    assert "FOR UPDATE" in conn.statements[0][0]
    assert "clock_timestamp" in conn.statements[1][0]


@pytest.mark.parametrize("now", [99, 1100, 1101])
def test_db_clock_rejects_not_yet_active_or_expired_claim(monkeypatch, make_pair, now):
    repo, pair, conn = wiring(monkeypatch, make_pair, now=now)
    with pytest.raises(ClaimConflict):
        repo.begin(pair.work, now_ms=100)
    assert conn.rolled_back
    assert not any("UPDATE" in q and "FOR UPDATE" not in q for q, _ in conn.statements)


def test_stale_generation_rejected_before_mutation(monkeypatch, make_pair):
    repo, pair, conn = wiring(monkeypatch, make_pair, now=101)
    stale = pair.work.model_copy(update={"delivery_count": 2})
    with pytest.raises(ClaimConflict):
        repo.begin(stale)
    assert conn.rolled_back


def test_claim_returns_database_time_and_increments_once(monkeypatch, make_pair):
    pending = make_pair()
    conn = Connection(now=500)
    claimed = JobWorkPair(job=pending.job, work=claim_work(pending.work, now_ms=500, lease_duration_ms=1000))
    answers = iter([pending, claimed])
    monkeypatch.setattr(module, "read_pair", lambda conn, analysis_id: next(answers))
    repo = module.PostgresAnalysisPersistence(conn.connect)
    assert repo.claim_next(now_ms=0, lease_duration_ms=1000) == claimed
    assert "FOR UPDATE SKIP LOCKED" in conn.statements[0][0]
    assert conn.statements[2][1][:3] == (1, to_timestamp(500), to_timestamp(1500))


@pytest.mark.parametrize("duration", [0, -1, True, 86400001])
def test_invalid_lease_never_opens_connection(duration):
    def reject():
        raise AssertionError("connection must not be opened")
    with pytest.raises(ValueError):
        module.PostgresAnalysisPersistence(reject).claim_next(lease_duration_ms=duration)
