"""Checked reconstruction of existing contracts, not parallel persistence models."""

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import ValidationError

from ugsl_ai_coach.coaching.models import CoachingFeedback
from ugsl_ai_coach.domain.analysis import ContractModel, StructuredAnalysisResult
from ugsl_ai_coach.integration.errors import AdapterContractError
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem, JobWorkPair
from ugsl_ai_coach.integration.models import AnalysisJob, CoachingRecord


class CorruptPersistence(AdapterContractError):
    """Stored state violates the authoritative typed contract. No automatic repair."""


EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def to_timestamp(milliseconds: int) -> datetime:
    return EPOCH + timedelta(milliseconds=milliseconds)


def to_milliseconds(value: datetime | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CorruptPersistence("Stored timestamp must be timezone-aware")
    delta = value - EPOCH
    microseconds = (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds
    if microseconds < 0 or microseconds % 1000:
        raise CorruptPersistence("Stored lease timestamp must be a non-negative exact millisecond")
    return microseconds // 1000


def payload(model: ContractModel) -> dict[str, Any]:
    # Revalidate even frozen instances: unchecked model_copy is possible.
    checked = type(model).model_validate(model.model_dump(mode="python"))
    return checked.model_dump(mode="json")


def analysis_from_json(value: Any) -> StructuredAnalysisResult:
    result = _read(StructuredAnalysisResult, value)
    if result.model_dump(mode="json") != value:
        raise CorruptPersistence("Stored analysis payload is not canonical")
    return result


def feedback_from_json(value: Any) -> CoachingFeedback:
    feedback = _read(CoachingFeedback, value)
    if feedback.model_dump(mode="json") != value:
        raise CorruptPersistence("Stored feedback payload is not canonical")
    return feedback


def _read(model, values):
    try:
        if not isinstance(values, dict):
            raise TypeError
        return model.model_validate(values)
    except (ValidationError, TypeError, ValueError) as error:
        raise CorruptPersistence("Stored payload violates its typed contract") from error


def job_from_row(row: dict) -> AnalysisJob:
    try:
        data = {key: row[key] for key in (
            "analysis_id", "attempt_id", "state", "learner_video_ref",
            "reference_profile_ref", "idempotency_key", "structured_analysis",
        )}
        if data["structured_analysis"] is not None:
            data["structured_analysis"] = analysis_from_json(data["structured_analysis"])
        job = _read(AnalysisJob, data)
        for key in ("analysis_id", "attempt_id", "learner_video_ref", "reference_profile_ref", "idempotency_key"):
            if data[key] != getattr(job, key):
                raise CorruptPersistence("Stored submission columns are not canonical")
        return job
    except KeyError as error:
        raise CorruptPersistence("Stored job columns are incomplete") from error


def work_from_row(row: dict) -> AnalysisWorkItem:
    try:
        data = {key: row[key] for key in (
            "analysis_id", "attempt_id", "learner_video_ref", "reference_profile_ref",
            "state", "delivery_count",
        )}
        data.update(claimed_at_ms=to_milliseconds(row["claimed_at"]),
                    lease_expires_at_ms=to_milliseconds(row["lease_expires_at"]))
        work = _read(AnalysisWorkItem, data)
        for key in ("analysis_id", "attempt_id", "learner_video_ref", "reference_profile_ref"):
            if data[key] != getattr(work, key):
                raise CorruptPersistence("Stored work columns are not canonical")
        return work
    except KeyError as error:
        raise CorruptPersistence("Stored work columns are incomplete") from error


def pair_from_rows(job: dict | None, work: dict | None) -> JobWorkPair | None:
    if job is None and work is None:
        return None
    if job is None or work is None:
        raise CorruptPersistence("Stored analysis requires both job and work")
    return _read(JobWorkPair, {"job": job_from_row(job), "work": work_from_row(work)})


def coaching_from_row(row: dict) -> CoachingRecord:
    try:
        feedback = feedback_from_json(row["feedback"])
        if row["feedback_id"] != feedback.feedback_id:
            raise CorruptPersistence("Stored feedback identity is inconsistent")
        return _read(CoachingRecord, {"analysis_id": row["analysis_id"],
                     "attempt_id": row["attempt_id"], "feedback": feedback})
    except KeyError as error:
        raise CorruptPersistence("Stored coaching columns are incomplete") from error
