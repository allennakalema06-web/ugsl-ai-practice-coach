from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.infrastructure.postgres.serialization import (
    CorruptPersistence, analysis_from_json, coaching_from_row, feedback_from_json,
    job_from_row, pair_from_rows, payload, to_milliseconds, to_timestamp, work_from_row,
)


def rows(pair):
    job = pair.job.model_dump(mode="json")
    work = pair.work.model_dump(mode="json")
    work["claimed_at"] = to_timestamp(work.pop("claimed_at_ms")) if pair.work.claimed_at_ms is not None else None
    work["lease_expires_at"] = to_timestamp(work.pop("lease_expires_at_ms")) if pair.work.lease_expires_at_ms is not None else None
    return job, work


@pytest.mark.parametrize("status", ["COMPLETED", "UNANALYZABLE", "FAILED"])
def test_m2_roundtrip(make_result, status):
    result = make_result(status)
    assert analysis_from_json(payload(result)) == result


def test_m5_roundtrip_and_identity(make_result):
    feedback = generate_feedback(make_result(), feedback_id="synthetic-feedback")
    assert feedback_from_json(payload(feedback)) == feedback
    row = dict(analysis_id=feedback.analysis_id, attempt_id=feedback.attempt_id,
               feedback_id=feedback.feedback_id, feedback=payload(feedback))
    assert coaching_from_row(row).feedback == feedback
    row["feedback_id"] = "other"
    with pytest.raises(CorruptPersistence):
        coaching_from_row(row)


@pytest.mark.parametrize("value", [None, [], "{}", {}, {"status": "DIRECTION"}])
@pytest.mark.parametrize("read", [analysis_from_json, feedback_from_json])
def test_invalid_json_is_explicit_corruption(read, value):
    with pytest.raises(CorruptPersistence):
        read(value)


def test_unchecked_result_is_revalidated(make_result):
    with pytest.raises(ValidationError):
        payload(make_result().model_copy(update={"overall_score": 5.0}))


def test_stored_payload_is_not_silently_normalized(make_result):
    value = payload(make_result())
    value["model_version"] = "  synthetic-v1  "
    with pytest.raises(CorruptPersistence):
        analysis_from_json(value)


@pytest.mark.parametrize("field", ["learner_video_ref", "reference_profile_ref", "idempotency_key"])
def test_stored_submission_is_not_silently_normalized(make_pair, field):
    job, _ = rows(make_pair())
    job[field] = " " + job[field]
    with pytest.raises(CorruptPersistence):
        job_from_row(job)


def test_pair_roundtrip_and_missing_rows(make_pair):
    pair = make_pair()
    job, work = rows(pair)
    assert pair_from_rows(job, work) == pair
    assert pair_from_rows(None, None) is None
    for a, b in [(job, None), (None, work)]:
        with pytest.raises(CorruptPersistence):
            pair_from_rows(a, b)


@pytest.mark.parametrize("change", [
    {"analysis_id": "AN-000002"}, {"attempt_id": "ATT-000002"},
    {"learner_video_ref": "different"}, {"state": "COMPLETED", "delivery_count": 1},
    {"delivery_count": -1}, {"state": "CLAIMED"},
])
def test_inconsistent_pair_is_rejected(make_pair, change):
    job, work = rows(make_pair())
    work.update(change)
    with pytest.raises(CorruptPersistence):
        pair_from_rows(job, work)


def test_terminal_job_requires_valid_matching_evidence(make_pair, make_result):
    job, _ = rows(make_pair())
    job["state"] = "COMPLETED"
    with pytest.raises(CorruptPersistence):
        job_from_row(job)
    job["structured_analysis"] = payload(make_result(analysis_id="AN-000002"))
    with pytest.raises(CorruptPersistence):
        job_from_row(job)


@pytest.mark.parametrize("read", [job_from_row, work_from_row, coaching_from_row])
def test_missing_columns(read):
    with pytest.raises(CorruptPersistence):
        read({})


@pytest.mark.parametrize("milliseconds", [0, 1, 999, 1700000000123])
def test_timestamp_roundtrip(milliseconds):
    assert to_milliseconds(to_timestamp(milliseconds)) == milliseconds
    assert to_milliseconds(to_timestamp(milliseconds).astimezone(timezone(timedelta(hours=3)))) == milliseconds


@pytest.mark.parametrize("timestamp", [datetime(2026, 1, 1),
    datetime(1969, 1, 1, tzinfo=timezone.utc),
    datetime(2026, 1, 1, microsecond=1, tzinfo=timezone.utc), "not-a-timestamp"])
def test_bad_timestamp(timestamp):
    with pytest.raises(CorruptPersistence):
        to_milliseconds(timestamp)
