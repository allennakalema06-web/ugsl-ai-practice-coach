"""Narrow coaching delivery operations using DB time, locks and fencing."""

from psycopg.types.json import Jsonb

from ugsl_ai_coach.infrastructure.postgres.coaching import validate_record
from ugsl_ai_coach.infrastructure.postgres.persistence import read_pair
from ugsl_ai_coach.infrastructure.postgres.serialization import coaching_from_row, payload, CorruptPersistence
from ugsl_ai_coach.integration.coaching_work import CoachingWork, retry_seconds
from ugsl_ai_coach.integration.errors import FeedbackConflict, AnalysisNotTerminal
from ugsl_ai_coach.integration.handoff.lifecycle import ClaimConflict
from ugsl_ai_coach.integration.models import CoachingRecord

FIELDS = tuple(CoachingWork.model_fields)


def work_from_row(row):
    return None if row is None else CoachingWork.model_validate({key: row[key] for key in FIELDS})


class PostgresCoachingWork:
    def __init__(self, connect):
        self.connect = connect

    def get(self, analysis_id):
        with self.connect() as conn:
            return work_from_row(conn.execute("SELECT * FROM coaching_work WHERE analysis_id = %s",
                                             (analysis_id,)).fetchone())

    def claim_next(self, *, lease_duration_ms):
        if type(lease_duration_ms) is not int or not 1 <= lease_duration_ms <= 86400000:
            raise ValueError("Invalid coaching lease")
        with self.connect() as conn:
            row = conn.execute("""SELECT * FROM coaching_work WHERE
                (state = 'PENDING' AND next_attempt_at <= clock_timestamp()) OR
                (state = 'CLAIMED' AND lease_expires_at <= clock_timestamp())
                ORDER BY next_attempt_at, created_at, analysis_id LIMIT 1 FOR UPDATE SKIP LOCKED""").fetchone()
            if row is None:
                return None
            return work_from_row(conn.execute("""WITH t AS (SELECT clock_timestamp() AS now)
                UPDATE coaching_work SET state = 'CLAIMED', delivery_count = delivery_count + 1,
                    claimed_at = t.now, lease_expires_at = t.now + %s * interval '1 millisecond',
                    updated_at = t.now FROM t WHERE analysis_id = %s RETURNING coaching_work.*""",
                (lease_duration_ms, row["analysis_id"])).fetchone())

    def _lock(self, conn, expected):
        expected = CoachingWork.model_validate(expected.model_dump(mode="python"))
        current = work_from_row(conn.execute("SELECT * FROM coaching_work WHERE analysis_id = %s FOR UPDATE",
                                            (expected.analysis_id,)).fetchone())
        if current is None:
            raise ClaimConflict("Coaching delivery does not exist")
        return current

    def _active(self, conn, current, expected):
        now = conn.execute("SELECT clock_timestamp() AS now").fetchone()["now"]
        if (current != expected or current.state != "CLAIMED"
                or not current.claimed_at <= now < current.lease_expires_at):
            raise ClaimConflict("Coaching claim is stale or expired")

    def _record(self, conn, analysis_id):
        row = conn.execute("SELECT * FROM coaching_records WHERE analysis_id = %s", (analysis_id,)).fetchone()
        if row is None:
            return None
        record = coaching_from_row(row)
        validate_record(conn, record)
        return record

    def existing(self, expected):
        """Reread committed truth before invoking a provider, including lost ack."""
        with self.connect() as conn:
            current = self._lock(conn, expected)
            record = self._record(conn, current.analysis_id)
            if record is not None:
                self._reconcile(conn, current)
                return record
            self._active(conn, current, expected)
            return None

    def load_analysis(self, expected):
        with self.connect() as conn:
            current = self._lock(conn, expected)
            self._active(conn, current, expected)
            pair = read_pair(conn, current.analysis_id)
            if pair is None or not pair.job.state.is_terminal:
                raise AnalysisNotTerminal("Coaching requires terminal evidence")
            if pair.job.attempt_id != current.attempt_id:
                raise CorruptPersistence("Coaching source identity mismatch")
            return pair.job.structured_analysis

    def _reconcile(self, conn, current):
        if current.state != "COMPLETED":
            conn.execute("""UPDATE coaching_work SET state = 'COMPLETED', claimed_at = NULL,
                lease_expires_at = NULL, updated_at = clock_timestamp() WHERE analysis_id = %s
                AND state <> 'COMPLETED'""", (current.analysis_id,))

    def complete(self, expected, record):
        record = CoachingRecord.model_validate(record.model_dump(mode="python"))
        with self.connect() as conn:
            current = self._lock(conn, expected)
            stored = self._record(conn, current.analysis_id)
            if stored is not None:
                if stored != record:
                    raise FeedbackConflict("Historical coaching differs")
                self._reconcile(conn, current)
                return stored
            self._active(conn, current, expected)
            if (record.analysis_id != current.analysis_id or record.attempt_id != current.attempt_id
                    or record.feedback.feedback_id != current.feedback_id):
                raise FeedbackConflict("Coaching must preserve assigned identity")
            validate_record(conn, record)
            inserted = conn.execute("""INSERT INTO coaching_records (analysis_id, attempt_id, feedback_id, feedback)
                SELECT %s, %s, %s, %s FROM coaching_work WHERE analysis_id = %s
                AND state = 'CLAIMED' AND delivery_count = %s AND lease_expires_at > clock_timestamp()
                RETURNING analysis_id""", (record.analysis_id, record.attempt_id,
                record.feedback.feedback_id, Jsonb(payload(record.feedback)), current.analysis_id,
                current.delivery_count)).fetchone()
            if inserted is None:
                raise ClaimConflict('Coaching lease expired before completion')
            # Migration trigger reconciles both this path and existing M6A append
            # within the record insertion transaction; deferred checks forbid halves.
            finished = work_from_row(conn.execute("SELECT * FROM coaching_work WHERE analysis_id = %s",
                                                 (current.analysis_id,)).fetchone())
            if finished.state != "COMPLETED":
                raise CorruptPersistence("Coaching completion was not atomic")
            return record

    def retry(self, expected, *, base_seconds, max_seconds):
        delay = retry_seconds(expected.failure_count, base_seconds, max_seconds)
        with self.connect() as conn:
            current = self._lock(conn, expected)
            if self._record(conn, current.analysis_id) is not None:
                self._reconcile(conn, current)
                return current
            self._active(conn, current, expected)
            row = conn.execute("""UPDATE coaching_work SET state = 'PENDING', failure_count = failure_count + 1,
                claimed_at = NULL, lease_expires_at = NULL, last_error_code = 'COACHING_PROCESSING_ERROR',
                next_attempt_at = clock_timestamp() + %s * interval '1 second', updated_at = clock_timestamp()
                WHERE analysis_id = %s AND state = 'CLAIMED' AND delivery_count = %s
                AND lease_expires_at > clock_timestamp() RETURNING *""",
                (delay, current.analysis_id, current.delivery_count)).fetchone()
            if row is None:
                raise ClaimConflict("Coaching lease expired before retry")
            return work_from_row(row)
