"""Append-only M6A coaching repository; grounded against immutable M2 evidence."""

from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

from ugsl_ai_coach.coaching.grounding import ground_analysis
from ugsl_ai_coach.coaching.validation import validate_feedback
from ugsl_ai_coach.domain.analysis import AnalysisId
from ugsl_ai_coach.infrastructure.postgres.connection import ConnectionFactory
from ugsl_ai_coach.infrastructure.postgres.persistence import read_pair
from ugsl_ai_coach.infrastructure.postgres.serialization import CorruptPersistence, coaching_from_row, payload
from ugsl_ai_coach.integration.errors import AnalysisNotTerminal, FeedbackConflict, JobNotFound
from ugsl_ai_coach.integration.models import CoachingRecord


def validate_record(conn, record: CoachingRecord) -> None:
    pair = read_pair(conn, record.analysis_id)
    if pair is None:
        raise JobNotFound("Analysis job does not exist")
    if not pair.job.state.is_terminal:
        raise AnalysisNotTerminal("Coaching requires terminal analysis")
    context = ground_analysis(pair.job.structured_analysis, feedback_id=record.feedback.feedback_id)
    validate_feedback(record.feedback, context, provider_type=record.feedback.provider_metadata.provider_type)


class PostgresCoachingRepository:
    def __init__(self, connect: ConnectionFactory):
        self.connect = connect

    def get(self, analysis_id: AnalysisId) -> CoachingRecord | None:
        analysis_id = TypeAdapter(AnalysisId).validate_python(analysis_id)
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM coaching_records WHERE analysis_id = %s", (analysis_id,)).fetchone()
            if row is None:
                return None
            record = coaching_from_row(row)
            try:
                validate_record(conn, record)
            except ValueError as error:
                raise CorruptPersistence("Stored coaching is not grounded in its source") from error
            return record

    def append(self, record: CoachingRecord) -> CoachingRecord:
        record = CoachingRecord.model_validate(record.model_dump(mode="python"))
        try:
            with self.connect() as conn:
                validate_record(conn, record)
                conn.execute("""INSERT INTO coaching_records
                    (analysis_id, attempt_id, feedback_id, feedback) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (analysis_id) DO NOTHING RETURNING analysis_id""",
                    (record.analysis_id, record.attempt_id, record.feedback.feedback_id,
                     Jsonb(payload(record.feedback)))).fetchone()
                stored = coaching_from_row(conn.execute(
                    "SELECT * FROM coaching_records WHERE analysis_id = %s", (record.analysis_id,)).fetchone())
                validate_record(conn, stored)
                if stored != record:
                    raise FeedbackConflict("Analysis already has different coaching")
                return stored
        except UniqueViolation as error:
            raise FeedbackConflict("Feedback identity already belongs to another analysis") from error
