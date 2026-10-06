"""M6E real PostgreSQL release gate: never substitute a fake for transactions."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from hashlib import sha256
from importlib.resources import files
from threading import Barrier

import psycopg
import pytest

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.coaching.providers.deterministic import DeterministicCoachingProvider
from ugsl_ai_coach.infrastructure.postgres.coaching import PostgresCoachingRepository
from ugsl_ai_coach.infrastructure.postgres.coaching_work import PostgresCoachingWork
from ugsl_ai_coach.infrastructure.postgres.migrate import migrate
from ugsl_ai_coach.infrastructure.postgres.persistence import PostgresAnalysisPersistence
from ugsl_ai_coach.infrastructure.postgres.rate_limit import PostgresRateLimiter
from ugsl_ai_coach.integration.errors import FeedbackConflict
from ugsl_ai_coach.integration.handoff.lifecycle import ClaimConflict
from ugsl_ai_coach.integration.models import CoachingRecord
from ugsl_ai_coach.operations.metrics import Metrics
from ugsl_ai_coach.worker.coaching import CoachingWorker, CoachingProcessor
from ugsl_ai_coach.core.config import Settings
from .test_database import terminal, record, fault

pytestmark = pytest.mark.postgres


@pytest.mark.parametrize('status', ['COMPLETED', 'UNANALYZABLE', 'FAILED'])
def test_terminal_commit_schedules_coaching_atomically(repo, make_pair, make_result, status):
    result = make_result(status)
    before = terminal(repo, make_pair(), result)
    work = PostgresCoachingWork(repo.connect).get(result.analysis_id)
    assert work.state == 'PENDING' and work.delivery_count == 0
    assert work.attempt_id == result.attempt_id and work.feedback_id.startswith('FB-')
    assert PostgresAnalysisPersistence(repo.connect).get_pair(result.analysis_id) == before


def test_no_terminal_result_means_no_coaching_work(repo, make_pair):
    repo.accept(make_pair())
    claim = repo.claim_next(lease_duration_ms=60000)
    repo.begin(claim.work)
    assert PostgresCoachingWork(repo.connect).get(claim.job.analysis_id) is None


def test_terminal_rollback_also_rolls_back_scheduling(repo, make_pair, make_result):
    repo.accept(make_pair())
    claim = repo.claim_next(lease_duration_ms=60000)
    before = repo.begin(claim.work)
    with repo.connect() as conn: fault(conn, 'terminal')
    with pytest.raises(psycopg.Error): repo.finish(claim.work, make_result())
    assert repo.get_pair(claim.job.analysis_id) == before
    assert PostgresCoachingWork(repo.connect).get(claim.job.analysis_id) is None


def test_schedule_insert_failure_rolls_back_terminal_and_work(repo, make_pair, make_result):
    repo.accept(make_pair())
    claim = repo.claim_next(lease_duration_ms=60000)
    before = repo.begin(claim.work)
    with repo.connect() as conn:
        conn.execute("""CREATE FUNCTION refuse_coaching() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'synthetic fault'; END $$;
            CREATE TRIGGER refuse_coaching BEFORE INSERT ON coaching_work
            FOR EACH ROW EXECUTE FUNCTION refuse_coaching()""")
    with pytest.raises(psycopg.Error): repo.finish(claim.work, make_result())
    assert repo.get_pair(claim.job.analysis_id) == before
    assert PostgresCoachingWork(repo.connect).get(claim.job.analysis_id) is None


@pytest.mark.parametrize('has_feedback', [False, True])
def test_upgrade_backfill_preserves_history_and_migration_reruns(db, make_pair, make_result, tmp_path, monkeypatch, has_feedback):
    import ugsl_ai_coach.infrastructure.postgres.migrate as module
    source = files('ugsl_ai_coach.infrastructure.postgres').joinpath('migrations')
    target = tmp_path / 'migrations'
    target.mkdir()
    (target / '001_initial_integration.sql').write_bytes(source.joinpath('001_initial_integration.sql').read_bytes())
    with monkeypatch.context() as patch:
        patch.setattr(module, 'files', lambda package: tmp_path)
        assert migrate(db) == ('001_initial_integration.sql',)
    repo = PostgresAnalysisPersistence(db)
    result = make_result()
    before = terminal(repo, make_pair(), result)
    existing = record(result)
    if has_feedback: PostgresCoachingRepository(db).append(existing)
    assert migrate(db) == ('002_operational_hardening.sql',)
    assert migrate(db) == ()
    work = PostgresCoachingWork(db).get(result.analysis_id)
    assert work.state == ('COMPLETED' if has_feedback else 'PENDING')
    if has_feedback:
        assert work.feedback_id == existing.feedback.feedback_id
        assert PostgresCoachingRepository(db).get(result.analysis_id) == existing
    assert repo.get_pair(result.analysis_id) == before


def ready_work(repo, make_pair, make_result):
    result = make_result()
    terminal(repo, make_pair(), result)
    return PostgresCoachingWork(repo.connect), result


def test_concurrent_coaching_claims_one_owner(repo, make_pair, make_result):
    work, _ = ready_work(repo, make_pair, make_result)
    barrier = Barrier(2)
    def claim():
        barrier.wait()
        return PostgresCoachingWork(repo.connect).claim_next(lease_duration_ms=60000)
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _: claim(), range(2)))
    assert sum(c is not None for c in claims) == 1
    assert work.get('AN-000001').delivery_count == 1


def test_active_expired_claim_and_stale_generation(repo, make_pair, make_result):
    work, result = ready_work(repo, make_pair, make_result)
    old = work.claim_next(lease_duration_ms=1000)
    assert work.claim_next(lease_duration_ms=60000) is None
    with repo.connect() as conn: conn.execute('SELECT pg_sleep(1.05)')
    new = work.claim_next(lease_duration_ms=60000)
    assert new.delivery_count == 2 and new.feedback_id == old.feedback_id
    with pytest.raises(ClaimConflict): work.load_analysis(old)
    with pytest.raises(ClaimConflict): work.retry(old, base_seconds=1, max_seconds=2)
    with pytest.raises(ClaimConflict): work.complete(old, CoachingProcessor().process(old, result))
    assert work.load_analysis(new) == result


def test_retry_schedule_uses_db_time_survives_reconnect_and_bounds(repo, make_pair, make_result):
    work, _ = ready_work(repo, make_pair, make_result)
    claim = work.claim_next(lease_duration_ms=60000)
    with repo.connect() as conn: before = conn.execute('SELECT clock_timestamp() AS now').fetchone()['now']
    pending = work.retry(claim, base_seconds=1, max_seconds=2)
    with repo.connect() as conn: after = conn.execute('SELECT clock_timestamp() AS now').fetchone()['now']
    assert before + timedelta(seconds=1) <= pending.next_attempt_at <= after + timedelta(seconds=1)
    assert pending.failure_count == 1 and pending.last_error_code == 'COACHING_PROCESSING_ERROR'
    assert work.claim_next(lease_duration_ms=60000) is None
    assert PostgresCoachingWork(repo.connect).get(claim.analysis_id) == pending
    with repo.connect() as conn: conn.execute('SELECT pg_sleep(1.05)')
    again = work.claim_next(lease_duration_ms=60000)
    assert again.feedback_id == claim.feedback_id and again.delivery_count == 2
    assert (work.retry(again, base_seconds=1, max_seconds=2).next_attempt_at - after).total_seconds() >= 2


@pytest.mark.parametrize('invalid', [False, True])
def test_provider_failure_byte_for_byte_m2_unchanged(repo, make_pair, make_result, invalid):
    work, result = ready_work(repo, make_pair, make_result)
    with repo.connect() as conn:
        before = conn.execute('SELECT * FROM analysis_jobs WHERE analysis_id = %s', (result.analysis_id,)).fetchone()
    class Provider:
        provider_type = 'deterministic'
        def generate(self, context):
            if invalid: return DeterministicCoachingProvider().generate(context).model_copy(update={'summary': 'Invented claim'})
            raise RuntimeError('sensitive exception')
    worker = CoachingWorker(work, CoachingProcessor(Provider()), Settings(_env_file=None))
    assert not worker.run_once()
    with repo.connect() as conn:
        after = conn.execute('SELECT * FROM analysis_jobs WHERE analysis_id = %s', (result.analysis_id,)).fetchone()
        assert conn.execute('SELECT count(*) AS n FROM coaching_records').fetchone()['n'] == 0
    assert after == before
    assert work.get(result.analysis_id).state == 'PENDING'


def test_atomic_complete_restart_ack_loss_and_no_reopen(repo, make_pair, make_result):
    work, result = ready_work(repo, make_pair, make_result)
    claim = work.claim_next(lease_duration_ms=60000)
    feedback = CoachingProcessor().process(claim, result)
    work.complete(claim, feedback)
    fresh = PostgresCoachingWork(repo.connect)
    assert fresh.existing(claim) == feedback  # old claim after lost acknowledgement
    assert fresh.complete(claim, feedback) == feedback
    assert fresh.get(result.analysis_id).state == 'COMPLETED'
    assert fresh.claim_next(lease_duration_ms=60000) is None
    with pytest.raises(FeedbackConflict): fresh.complete(claim, record(result, 'different-feedback'))
    with pytest.raises(psycopg.errors.CheckViolation):
        with repo.connect() as conn:
            conn.execute("UPDATE coaching_work SET state = 'PENDING' WHERE analysis_id = %s", (result.analysis_id,))


def test_legacy_append_reconciles_without_provider(repo, make_pair, make_result):
    work, result = ready_work(repo, make_pair, make_result)
    claim = work.claim_next(lease_duration_ms=60000)
    feedback = record(result)
    PostgresCoachingRepository(repo.connect).append(feedback)
    assert work.existing(claim) == feedback
    assert work.get(result.analysis_id).state == 'COMPLETED'


def test_coaching_completion_failure_rolls_back_feedback_and_work(repo, make_pair, make_result):
    work, result = ready_work(repo, make_pair, make_result)
    claim = work.claim_next(lease_duration_ms=60000)
    with repo.connect() as conn:
        conn.execute("""CREATE FUNCTION fail_coaching_finish() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN IF NEW.state = 'COMPLETED' THEN RAISE EXCEPTION 'synthetic fault'; END IF;
            RETURN NEW; END $$;
            CREATE TRIGGER fail_coaching_finish BEFORE UPDATE ON coaching_work
            FOR EACH ROW EXECUTE FUNCTION fail_coaching_finish()""")
    with pytest.raises(psycopg.Error): work.complete(claim, CoachingProcessor().process(claim, result))
    assert work.get(result.analysis_id) == claim
    assert PostgresCoachingRepository(repo.connect).get(result.analysis_id) is None


def test_half_coaching_completion_rejected(repo, make_pair, make_result):
    work, result = ready_work(repo, make_pair, make_result)
    work.claim_next(lease_duration_ms=60000)
    with pytest.raises(psycopg.errors.CheckViolation):
        with repo.connect() as conn:
            conn.execute("UPDATE coaching_work SET state = 'COMPLETED', claimed_at = NULL, lease_expires_at = NULL")
    assert PostgresCoachingRepository(repo.connect).get(result.analysis_id) is None


def test_concurrent_limiter_instances_share_threshold_and_separate_buckets(repo):
    barrier = Barrier(8)
    def check():
        barrier.wait()
        return PostgresRateLimiter(repo.connect).check('synthetic-backend', 'submit', 3)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: check(), range(8)))
    assert sum(r.allowed for r in results) == 3
    assert all(1 <= r.retry_after <= 60 for r in results)
    limiter = PostgresRateLimiter(repo.connect)
    assert limiter.check('synthetic-backend', 'read', 1).allowed
    assert limiter.check('different-service', 'submit', 1).allowed
    with repo.connect() as conn:
        rows = conn.execute('SELECT * FROM service_rate_limits').fetchall()
        assert len(rows) == 3
        assert all(len(r['principal_digest']) == 64 for r in rows)


def test_limiter_window_expiry_and_restart_use_database_clock(repo):
    limiter = PostgresRateLimiter(repo.connect)
    assert limiter.check('synthetic', 'submit', 1).allowed
    assert not limiter.check('synthetic', 'submit', 1).allowed
    with repo.connect() as conn:
        conn.execute("UPDATE service_rate_limits SET window_start = date_trunc('minute', clock_timestamp()) - interval '1 minute'")
    assert PostgresRateLimiter(repo.connect).check('synthetic', 'submit', 1).allowed
    with repo.connect() as conn:
        row = conn.execute("SELECT *, date_trunc('minute', clock_timestamp()) AS expected FROM service_rate_limits").fetchone()
    assert row['window_start'] == row['expected'] and row['request_count'] == 1


def test_queue_snapshot_and_cross_process_worker_metrics(repo, make_pair, make_result):
    from ugsl_ai_coach.infrastructure.postgres.telemetry import PostgresWorkerMetrics
    terminal(repo, make_pair(), make_result())
    PostgresWorkerMetrics(repo.connect).outcome('coaching', 'failure')
    metrics = Metrics()
    metrics.snapshot(repo.connect)
    text = metrics.render().decode()
    assert 'ugsl_postgres_up 1.0' in text
    assert 'ugsl_queue_depth{worker="coaching"} 1.0' in text
    assert 'outcome="failure",worker="coaching"} 1.0' in text
    assert 'AN-000001' not in text


def test_real_analysis_processor_exception_creates_no_coaching_work(repo, make_pair):
    from ugsl_ai_coach.worker.runtime import AnalysisWorker
    repo.accept(make_pair())
    class Broken:
        def process(self, work): raise RuntimeError('synthetic infrastructure failure')
    assert not AnalysisWorker(repo, Broken(), Settings(_env_file=None)).run_once()
    assert repo.get('AN-000001').state == 'PROCESSING'
    assert repo.get('AN-000001').structured_analysis is None
    assert PostgresCoachingWork(repo.connect).get('AN-000001') is None


def test_invalid_feedback_cannot_be_written_by_delivery_adapter(repo, make_pair, make_result):
    from ugsl_ai_coach.coaching.validation import GroundingViolation
    work, result = ready_work(repo, make_pair, make_result)
    claim = work.claim_next(lease_duration_ms=60000)
    valid = CoachingProcessor().process(claim, result)
    invalid = valid.model_copy(update={'feedback': valid.feedback.model_copy(update={'summary': 'Invented claim'})})
    with pytest.raises(GroundingViolation): work.complete(claim, invalid)
    assert work.get(result.analysis_id) == claim
    assert PostgresCoachingRepository(repo.connect).get(result.analysis_id) is None


def test_worker_lost_completion_ack_converges_against_real_committed_truth(repo, make_pair, make_result):
    work, result = ready_work(repo, make_pair, make_result)
    class LostAck:
        def __getattr__(self, name): return getattr(work, name)
        def complete(self, claim, feedback):
            work.complete(claim, feedback)
            raise psycopg.OperationalError('synthetic lost acknowledgement')
    worker = CoachingWorker(LostAck(), CoachingProcessor(), Settings(_env_file=None))
    assert worker.run_once()
    assert work.get(result.analysis_id).state == 'COMPLETED'
    assert work.get(result.analysis_id).failure_count == 0
    assert PostgresCoachingRepository(repo.connect).get(result.analysis_id) is not None
