"""M6B atomic persistence using PostgreSQL row locks and authoritative DB time."""

from uuid import uuid4

from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

from ugsl_ai_coach.domain.analysis import AnalysisId, StructuredAnalysisResult
from ugsl_ai_coach.infrastructure.postgres.connection import ConnectionFactory
from ugsl_ai_coach.infrastructure.postgres.serialization import (
    CorruptPersistence, pair_from_rows, payload, to_milliseconds, to_timestamp,
)
from ugsl_ai_coach.integration.errors import AnalysisIdConflict, IdempotencyConflict, JobNotFound
from ugsl_ai_coach.integration.handoff.lifecycle import (
    ClaimConflict, claim_work, complete_work, validate_active_claim, validate_time,
)
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem, JobWorkPair, WorkAcceptance
from ugsl_ai_coach.integration.lifecycle import transition_job
from ugsl_ai_coach.integration.models import JobState, OpaqueReference


# A single statement yields a consistent read snapshot of both records, including
# orphan work. Static identifiers only; all lookup values are bound parameters.
PAIR_SQL = """SELECT j.*, w.analysis_id AS work_analysis_id, w.attempt_id AS work_attempt_id,
    w.learner_video_ref AS work_learner_video_ref,
    w.reference_profile_ref AS work_reference_profile_ref, w.state AS work_state,
    w.delivery_count AS work_delivery_count, w.claimed_at AS work_claimed_at,
    w.lease_expires_at AS work_lease_expires_at
    FROM analysis_jobs j FULL JOIN analysis_work w ON j.analysis_id = w.analysis_id
    WHERE COALESCE(j.analysis_id, w.analysis_id) = %s"""


def read_pair(conn, analysis_id: str) -> JobWorkPair | None:
    row = conn.execute(PAIR_SQL, (analysis_id,)).fetchone()
    if row is None:
        return None
    work = None if row["work_analysis_id"] is None else {
        key: row["work_" + key] for key in (
            "analysis_id", "attempt_id", "learner_video_ref", "reference_profile_ref",
            "state", "delivery_count", "claimed_at", "lease_expires_at",
        )
    }
    return pair_from_rows(None if row["analysis_id"] is None else row, work)


def database_now(conn) -> int:
    return to_milliseconds(conn.execute(
        "SELECT date_trunc('milliseconds', clock_timestamp()) AS now").fetchone()["now"])


class PostgresAnalysisIdFactory:
    """Stateless decimal UUID identity, conforming to the existing AN digit contract."""

    def create(self) -> AnalysisId:
        return f"AN-{uuid4().int:06d}"


class PostgresAnalysisPersistence:
    """Production implementation of AnalysisPersistence.

    Optional now_ms preserves the M6B calling signature but is only schema-
    validated, never trusted for ownership. PostgreSQL supplies time after locks
    are acquired. The worker calls these operations without a host timestamp;
    deterministic AnalysisHandoffService claim-time assertions remain unchanged
    and are not used to coordinate this DB-clock production worker.
    """

    def __init__(self, connect: ConnectionFactory):
        self.connect = connect

    def get_pair(self, analysis_id: AnalysisId) -> JobWorkPair | None:
        analysis_id = TypeAdapter(AnalysisId).validate_python(analysis_id)
        with self.connect() as conn:
            return read_pair(conn, analysis_id)

    def get(self, analysis_id: AnalysisId):
        pair = self.get_pair(analysis_id)
        return None if pair is None else pair.job

    def get_by_idempotency_key(self, key: str) -> JobWorkPair | None:
        key = TypeAdapter(OpaqueReference).validate_python(key)
        with self.connect() as conn:
            row = conn.execute("SELECT analysis_id FROM analysis_jobs WHERE idempotency_key = %s",
                               (key,)).fetchone()
            return None if row is None else read_pair(conn, row["analysis_id"])

    def accept(self, candidate: JobWorkPair) -> WorkAcceptance:
        candidate = JobWorkPair.model_validate(candidate.model_dump(mode="python"))
        WorkAcceptance(pair=candidate, created=True)
        job, work = candidate.job, candidate.work
        try:
            with self.connect() as conn:
                inserted = conn.execute("""INSERT INTO analysis_jobs
                    (analysis_id, attempt_id, idempotency_key, learner_video_ref,
                     reference_profile_ref, state) VALUES (%s, %s, %s, %s, %s, 'SUBMITTED')
                    ON CONFLICT (idempotency_key) DO NOTHING RETURNING analysis_id""",
                    (job.analysis_id, job.attempt_id, job.idempotency_key,
                     job.learner_video_ref, job.reference_profile_ref)).fetchone()
                if inserted is None:
                    row = conn.execute("SELECT analysis_id FROM analysis_jobs WHERE idempotency_key = %s",
                                       (job.idempotency_key,)).fetchone()
                    if row is None:
                        raise CorruptPersistence("Idempotency record disappeared")
                    existing = read_pair(conn, row["analysis_id"])
                    if existing.job.submission != job.submission:
                        raise IdempotencyConflict("Idempotency key has a different submission")
                    return WorkAcceptance(pair=existing, created=False)
                conn.execute("""INSERT INTO analysis_work
                    (analysis_id, attempt_id, learner_video_ref, reference_profile_ref, state, delivery_count)
                    VALUES (%s, %s, %s, %s, 'PENDING', 0)""",
                    (work.analysis_id, work.attempt_id, work.learner_video_ref, work.reference_profile_ref))
                return WorkAcceptance(pair=read_pair(conn, job.analysis_id), created=True)
        except UniqueViolation as error:
            raise AnalysisIdConflict("Analysis identity already exists") from error

    def claim_next(self, *, lease_duration_ms: int, now_ms: int | None = None) -> JobWorkPair | None:
        validate_time(0 if now_ms is None else now_ms, lease_duration_ms)
        if lease_duration_ms > 86400000:
            raise ValueError("Production lease must be at most one day")
        with self.connect() as conn:
            row = conn.execute("""SELECT analysis_id FROM analysis_work
                WHERE state = 'PENDING' OR (state = 'CLAIMED' AND lease_expires_at <= clock_timestamp())
                ORDER BY created_at, analysis_id LIMIT 1 FOR UPDATE SKIP LOCKED""").fetchone()
            if row is None:
                return None
            pair = read_pair(conn, row["analysis_id"])
            if pair is None:
                raise CorruptPersistence("Claimed work has no analysis pair")
            now = database_now(conn)
            claimed = claim_work(pair.work, now_ms=now, lease_duration_ms=lease_duration_ms)
            conn.execute("""UPDATE analysis_work SET state = 'CLAIMED', delivery_count = %s,
                claimed_at = %s, lease_expires_at = %s, updated_at = clock_timestamp()
                WHERE analysis_id = %s AND delivery_count = %s
                AND (state = 'PENDING' OR (state = 'CLAIMED' AND lease_expires_at <= clock_timestamp()))""",
                (claimed.delivery_count, to_timestamp(claimed.claimed_at_ms),
                 to_timestamp(claimed.lease_expires_at_ms), claimed.analysis_id, pair.work.delivery_count))
            stored = read_pair(conn, claimed.analysis_id)
            if stored.work != claimed:
                raise ClaimConflict("Claim update did not preserve the expected generation")
            return stored

    def _locked_claim(self, conn, expected, now_ms):
        if now_ms is not None:
            validate_time(now_ms)
        expected = AnalysisWorkItem.model_validate(expected.model_dump(mode="python"))
        row = conn.execute("SELECT analysis_id FROM analysis_work WHERE analysis_id = %s FOR UPDATE",
                           (expected.analysis_id,)).fetchone()
        pair = read_pair(conn, expected.analysis_id)
        if pair is None:
            raise JobNotFound("Analysis job does not exist")
        if row is None:
            raise CorruptPersistence("Analysis has no work record")
        now = database_now(conn)
        validate_active_claim(pair.work, expected, now_ms=now)
        return pair, now

    def begin(self, expected: AnalysisWorkItem, *, now_ms: int | None = None) -> JobWorkPair:
        with self.connect() as conn:
            pair, _ = self._locked_claim(conn, expected, now_ms)
            if pair.job.state == JobState.PROCESSING:
                return pair
            replacement = transition_job(pair.job, JobState.PROCESSING)
            changed = conn.execute("""UPDATE analysis_jobs SET state = 'PROCESSING', updated_at = clock_timestamp()
                WHERE analysis_id = %s AND state = 'SUBMITTED' AND EXISTS
                (SELECT 1 FROM analysis_work WHERE analysis_id = %s AND state = 'CLAIMED'
                 AND delivery_count = %s AND lease_expires_at > clock_timestamp())""",
                (expected.analysis_id, expected.analysis_id, expected.delivery_count)).rowcount
            if changed != 1:
                raise ClaimConflict("Claim expired before processing could begin")
            return JobWorkPair(job=replacement, work=pair.work)

    def finish(self, expected: AnalysisWorkItem, result: StructuredAnalysisResult,
               *, now_ms: int | None = None) -> JobWorkPair:
        result = StructuredAnalysisResult.model_validate(result.model_dump(mode="python"))
        with self.connect() as conn:
            pair, now = self._locked_claim(conn, expected, now_ms)
            terminal = transition_job(pair.job, JobState(result.status.value), structured_analysis=result)
            completed = complete_work(pair.work, now_ms=now)
            changed = conn.execute("""UPDATE analysis_jobs SET state = %s, structured_analysis = %s,
                updated_at = clock_timestamp() WHERE analysis_id = %s AND state = 'PROCESSING'
                AND EXISTS (SELECT 1 FROM analysis_work WHERE analysis_id = %s AND state = 'CLAIMED'
                            AND delivery_count = %s AND lease_expires_at > clock_timestamp())""",
                (terminal.state.value, Jsonb(payload(result)), expected.analysis_id,
                 expected.analysis_id, expected.delivery_count)).rowcount
            if changed != 1:
                raise ClaimConflict("Claim expired before terminal persistence")
            changed = conn.execute("""UPDATE analysis_work SET state = 'COMPLETED', claimed_at = NULL,
                lease_expires_at = NULL, updated_at = clock_timestamp()
                WHERE analysis_id = %s AND state = 'CLAIMED' AND delivery_count = %s
                AND lease_expires_at > clock_timestamp()""",
                (expected.analysis_id, expected.delivery_count)).rowcount
            if changed != 1:
                raise ClaimConflict("Claim expired before work completion")
            return JobWorkPair(job=terminal, work=completed)
