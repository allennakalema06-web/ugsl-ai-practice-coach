"""Deterministic test adapters only; deliberately no production registration."""

from ugsl_ai_coach.integration.errors import (
    AnalysisIdConflict, ConcurrentUpdate, FeedbackConflict, IdempotencyConflict, JobNotFound,
)
from ugsl_ai_coach.integration.lifecycle import validate_transition
from ugsl_ai_coach.integration.models import AnalysisJob, CoachingRecord, JobAcceptance, JobState


class FakeRepository:
    """Sequential fake illustrates atomic-port semantics, not durable concurrency."""

    def __init__(self):
        self.jobs = {}
        self.keys = {}

    def get(self, analysis_id):
        return self.jobs.get(analysis_id)

    def get_by_idempotency_key(self, key):
        return self.get(self.keys[key]) if key in self.keys else None

    def accept(self, candidate):
        candidate = AnalysisJob.model_validate(candidate.model_dump(mode="python"))
        if candidate.state != JobState.SUBMITTED:
            raise ValueError("Only submitted candidates may be accepted")
        existing = self.get_by_idempotency_key(candidate.idempotency_key)
        if existing is not None:
            if existing.submission != candidate.submission:
                raise IdempotencyConflict("Conflicting logical submission")
            return JobAcceptance(job=existing, created=False)
        if candidate.analysis_id in self.jobs:
            raise AnalysisIdConflict("Analysis ID already exists")
        self.jobs[candidate.analysis_id] = candidate
        self.keys[candidate.idempotency_key] = candidate.analysis_id
        return JobAcceptance(job=candidate, created=True)

    def compare_and_set(self, expected, replacement):
        validate_transition(expected, replacement)
        current = self.get(expected.analysis_id)
        if current is None:
            raise JobNotFound("No job exists")
        if current != expected:
            raise ConcurrentUpdate("Stale expected snapshot")
        self.jobs[expected.analysis_id] = replacement
        return replacement


class FakeDispatcher:
    def __init__(self):
        self.dispatched = []

    def dispatch(self, job):
        self.dispatched.append(job)


class FakeIdFactory:
    def __init__(self):
        self.calls = 0

    def create(self):
        self.calls += 1
        return f"AN-{self.calls:06d}"


class FakeCoachingRepository:
    """Sequential append-only fake; production atomicity remains an adapter obligation."""

    def __init__(self):
        self.records = {}

    def get(self, analysis_id):
        return self.records.get(analysis_id)

    def append(self, record):
        record = CoachingRecord.model_validate(record.model_dump(mode="python"))
        existing = self.get(record.analysis_id)
        if existing is not None:
            if existing != record:
                raise FeedbackConflict("Analysis already has a different coaching artifact")
            return existing
        self.records[record.analysis_id] = record
        return record
