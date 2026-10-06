"""Real PostgreSQL only: synthetic data in disposable, explicitly owned schemas."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import psycopg
import pytest

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.coaching.validation import GroundingViolation
from ugsl_ai_coach.infrastructure.postgres.coaching import PostgresCoachingRepository
from ugsl_ai_coach.infrastructure.postgres.migrate import migrate
from ugsl_ai_coach.infrastructure.postgres.persistence import PostgresAnalysisPersistence
from ugsl_ai_coach.infrastructure.postgres.serialization import CorruptPersistence
from ugsl_ai_coach.integration.errors import AnalysisIdConflict, AnalysisNotTerminal, FeedbackConflict, IdempotencyConflict
from ugsl_ai_coach.integration.handoff.lifecycle import ClaimConflict
from ugsl_ai_coach.integration.models import CoachingRecord

pytestmark = pytest.mark.postgres


def fault(conn, event):
    if event == "insert":
        conn.execute("""CREATE FUNCTION forced_fault() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'synthetic insertion fault'; END $$;
            CREATE TRIGGER forced_fault BEFORE INSERT ON analysis_work
            FOR EACH ROW EXECUTE FUNCTION forced_fault()""")
    else:
        conn.execute("""CREATE FUNCTION forced_fault() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN IF NEW.state = 'COMPLETED' THEN RAISE EXCEPTION 'synthetic terminal fault';
            END IF; RETURN NEW; END $$;
            CREATE TRIGGER forced_fault BEFORE UPDATE ON analysis_work
            FOR EACH ROW EXECUTE FUNCTION forced_fault()""")


def terminal(repo, pair, result):
    repo.accept(pair)
    claimed = repo.claim_next(lease_duration_ms=60000)
    repo.begin(claimed.work)
    return repo.finish(claimed.work, result)


def record(result, identity="synthetic-feedback"):
    return CoachingRecord(analysis_id=result.analysis_id, attempt_id=result.attempt_id,
                          feedback=generate_feedback(result, feedback_id=identity))


def test_empty_database_migration_and_rerun(db):
    assert migrate(db) == ("001_initial_integration.sql", "002_operational_hardening.sql")
    assert migrate(db) == ()
    with db() as conn:
        assert conn.execute("SELECT count(*) AS n FROM schema_migrations").fetchone()["n"] == 2
        for table in ("analysis_jobs", "analysis_work", "coaching_records", "coaching_work", "service_rate_limits", "operational_worker_counts"):
            assert conn.execute("SELECT to_regclass(%s) AS name", (table,)).fetchone()["name"]


def test_failed_migration_rolls_back_and_not_tracked(db, tmp_path, monkeypatch):
    from ugsl_ai_coach.infrastructure.postgres import migrate as module
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "001_fault.sql").write_text("CREATE TABLE partial_migration (id INT); SELECT missing_column;", encoding="utf-8")
    monkeypatch.setattr(module, "files", lambda package: tmp_path)
    with pytest.raises(psycopg.Error):
        module.migrate(db)
    with db() as conn:
        assert conn.execute("SELECT to_regclass('partial_migration') AS t").fetchone()["t"] is None
        assert conn.execute("SELECT to_regclass('schema_migrations') AS t").fetchone()["t"] is None


def test_migration_checksum_mismatch_is_rejected(repo):
    with repo.connect() as conn:
        conn.execute("UPDATE schema_migrations SET checksum = 'synthetic-change'")
    with pytest.raises(CorruptPersistence):
        migrate(repo.connect)


def test_accept_commits_pair_and_reconnect_reads_it(repo, make_pair):
    candidate = make_pair()
    assert repo.accept(candidate).created
    fresh = PostgresAnalysisPersistence(repo.connect)
    assert fresh.get_pair(candidate.job.analysis_id) == candidate
    assert fresh.get(candidate.job.analysis_id) == candidate.job


def test_accept_failure_rolls_back_both(repo, make_pair):
    with repo.connect() as conn:
        fault(conn, "insert")
    with pytest.raises(psycopg.Error):
        repo.accept(make_pair())
    assert repo.get_pair("AN-000001") is None
    with repo.connect() as conn:
        assert conn.execute("SELECT count(*) AS n FROM analysis_work").fetchone()["n"] == 0


def test_idempotency_conflict_and_analysis_collision(repo, make_pair):
    original = make_pair()
    repo.accept(original)
    assert not repo.accept(make_pair(analysis_id="AN-000002")).created
    assert repo.get_pair("AN-000002") is None
    with pytest.raises(IdempotencyConflict):
        repo.accept(make_pair(analysis_id="AN-000003", learner_video_ref="different"))
    with pytest.raises(AnalysisIdConflict):
        repo.accept(make_pair(idempotency_key="another-key"))
    assert repo.get_pair("AN-000001") == original


def test_concurrent_acceptance_deduplicates(repo, make_pair):
    barrier = Barrier(2)
    def submit(number):
        barrier.wait(timeout=5)
        return repo.accept(make_pair(analysis_id=f"AN-{number:06d}"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(submit, [1, 2]))
    assert sorted(a.created for a in answers) == [False, True]
    assert answers[0].pair == answers[1].pair


def test_concurrent_workers_only_one_active_claim(repo, make_pair):
    repo.accept(make_pair())
    barrier = Barrier(2)
    def claim(_):
        barrier.wait(timeout=5)
        return repo.claim_next(lease_duration_ms=60000)
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(claim, [1, 2]))
    assert sum(answer is not None for answer in answers) == 1
    assert repo.get_pair("AN-000001").work.delivery_count == 1


def test_skip_locked_allows_other_work(repo, make_pair):
    repo.accept(make_pair())
    repo.accept(make_pair(2))
    with repo.connect() as locked:
        locked.execute("SELECT analysis_id FROM analysis_work WHERE analysis_id = %s FOR UPDATE", ("AN-000001",))
        assert repo.claim_next(lease_duration_ms=60000).job.analysis_id == "AN-000002"
    assert repo.claim_next(lease_duration_ms=60000).job.analysis_id == "AN-000001"


def test_db_clock_ignores_host_time_and_active_lease_is_excluded(repo, make_pair):
    repo.accept(make_pair())
    claimed = repo.claim_next(now_ms=0, lease_duration_ms=60000)
    assert claimed.work.claimed_at_ms > 0
    assert claimed.work.lease_expires_at_ms - claimed.work.claimed_at_ms == 60000
    assert repo.claim_next(now_ms=999999999999999999, lease_duration_ms=60000) is None
    assert repo.begin(claimed.work, now_ms=0).job.state == "PROCESSING"


def test_expired_claim_recovery_generation_fences_stale_worker(repo, make_pair, make_result):
    repo.accept(make_pair())
    old = repo.claim_next(lease_duration_ms=1000)
    repo.begin(old.work)
    with repo.connect() as conn:
        conn.execute("SELECT pg_sleep(1.05)")
    with pytest.raises(ClaimConflict):
        repo.begin(old.work)
    new = repo.claim_next(lease_duration_ms=60000)
    assert new.work.delivery_count == 2 and new.job.state == "PROCESSING"
    assert repo.begin(new.work) == new
    for action in (lambda: repo.begin(old.work), lambda: repo.finish(old.work, make_result())):
        with pytest.raises(ClaimConflict):
            action()
    assert repo.finish(new.work, make_result()).work.state == "COMPLETED"


@pytest.mark.parametrize("status", ["COMPLETED", "UNANALYZABLE", "FAILED"])
def test_atomic_terminal_result_and_restart_durability(repo, make_pair, make_result, status):
    result = make_result(status)
    completed = terminal(repo, make_pair(), result)
    assert completed.job.state == status and completed.work.state == "COMPLETED"
    assert completed.work.claimed_at_ms is None and completed.work.lease_expires_at_ms is None
    assert PostgresAnalysisPersistence(repo.connect).get_pair("AN-000001") == completed
    assert completed.job.structured_analysis == result
    assert repo.claim_next(lease_duration_ms=60000) is None


def test_terminal_failure_rolls_back_job_result_and_work(repo, make_pair, make_result):
    repo.accept(make_pair())
    claim = repo.claim_next(lease_duration_ms=60000)
    before = repo.begin(claim.work)
    with repo.connect() as conn:
        fault(conn, "terminal")
    with pytest.raises(psycopg.Error):
        repo.finish(claim.work, make_result())
    assert repo.get_pair("AN-000001") == before


def test_terminal_evidence_and_completed_work_immutable(repo, make_pair, make_result):
    completed = terminal(repo, make_pair(), make_result())
    with pytest.raises(ClaimConflict):
        repo.finish(completed.work, make_result())
    for statement in (
        "UPDATE analysis_jobs SET state = 'SUBMITTED', structured_analysis = NULL WHERE analysis_id = %s",
        "UPDATE analysis_jobs SET structured_analysis = jsonb_set(structured_analysis, '{model_version}', '\"changed\"') WHERE analysis_id = %s",
        "UPDATE analysis_work SET state = 'PENDING' WHERE analysis_id = %s",
        "DELETE FROM analysis_jobs WHERE analysis_id = %s",
    ):
        with pytest.raises(psycopg.errors.CheckViolation):
            with repo.connect() as conn:
                conn.execute(statement, ("AN-000001",))
    assert repo.get_pair("AN-000001") == completed


def test_coaching_idempotent_conflict_and_analysis_unchanged(repo, make_pair, make_result):
    result = make_result()
    completed = terminal(repo, make_pair(), result)
    records = PostgresCoachingRepository(repo.connect)
    feedback = record(result)
    assert records.append(feedback) == feedback
    assert records.append(feedback) == feedback
    assert PostgresCoachingRepository(repo.connect).get(result.analysis_id) == feedback
    with pytest.raises(FeedbackConflict):
        records.append(record(result, "synthetic-other-feedback"))
    for statement in ("UPDATE coaching_records SET feedback_id = 'changed' WHERE analysis_id = %s",
                      "DELETE FROM coaching_records WHERE analysis_id = %s"):
        with pytest.raises(psycopg.errors.CheckViolation):
            with repo.connect() as conn:
                conn.execute(statement, (result.analysis_id,))
    assert repo.get_pair(result.analysis_id) == completed


def test_coaching_requires_terminal_and_full_m5_validation(repo, make_pair, make_result):
    result = make_result()
    records = PostgresCoachingRepository(repo.connect)
    repo.accept(make_pair())
    with pytest.raises(AnalysisNotTerminal):
        records.append(record(result))
    claim = repo.claim_next(lease_duration_ms=60000)
    repo.begin(claim.work)
    repo.finish(claim.work, result)
    valid = record(result)
    invalid = valid.model_copy(update={"feedback": valid.feedback.model_copy(update={"summary": "Unauthorized claim"})})
    with pytest.raises(GroundingViolation):
        records.append(invalid)
    assert records.get(result.analysis_id) is None


def test_db_rejects_half_acceptance_and_half_terminal_commit(repo, make_pair):
    with pytest.raises(psycopg.errors.CheckViolation):
        with repo.connect() as conn:
            conn.execute("""INSERT INTO analysis_jobs (analysis_id, attempt_id, idempotency_key,
                learner_video_ref, reference_profile_ref, state)
                VALUES (%s, %s, %s, %s, %s, 'SUBMITTED')""",
                ("AN-000001", "ATT-000001", "synthetic-key", "synthetic-video", "synthetic-profile"))
    assert repo.get_pair("AN-000001") is None
    repo.accept(make_pair())
    claim = repo.claim_next(lease_duration_ms=60000)
    repo.begin(claim.work)
    with pytest.raises(psycopg.errors.CheckViolation):
        with repo.connect() as conn:
            conn.execute("""UPDATE analysis_work SET state = 'COMPLETED', claimed_at = NULL,
                lease_expires_at = NULL WHERE analysis_id = %s""", ("AN-000001",))
    assert repo.get_pair("AN-000001").work.state == "CLAIMED"
