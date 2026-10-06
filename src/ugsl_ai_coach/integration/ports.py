"""Technology-neutral adapter obligations, including atomicity and immutability."""

from typing import Protocol

from ugsl_ai_coach.domain.analysis import AnalysisId
from ugsl_ai_coach.integration.models import AnalysisJob, CoachingRecord, JobAcceptance


class AnalysisJobReader(Protocol):
    """Read-only analysis access, also usable by separate coaching persistence."""

    def get(self, analysis_id: AnalysisId) -> AnalysisJob | None: ...


class AnalysisJobRepository(AnalysisJobReader, Protocol):
    """Future adapters own durable uniqueness and concurrency enforcement.

    Idempotency keys are unique in the adapter's configured namespace. Backend
    authentication/authorization and any tenancy scoping are later integration
    concerns; callers cannot infer access permission from possession of an ID.
    """

    def get_by_idempotency_key(self, key: str) -> AnalysisJob | None: ...

    def accept(self, candidate: AnalysisJob) -> JobAcceptance:
        """Atomically accept a SUBMITTED candidate or return the existing job.

        Same key + same canonical submission: existing job, created=False,
        regardless of its current lifecycle state. Same key + changed payload:
        raise IdempotencyConflict. A colliding analysis ID for a new key raises
        AnalysisIdConflict; never overwrite an earlier attempt/job. Concurrent
        accepts for a key must have exactly one created=True outcome.
        """
        ...

    def compare_and_set(self, expected: AnalysisJob, replacement: AnalysisJob) -> AnalysisJob:
        """Atomically apply only a valid transition to the exact stored snapshot.

        Validate both records and validate_transition before persisting. Raise
        JobNotFound for an absent ID, ConcurrentUpdate for a stale snapshot or
        InvalidTransition for a forbidden update. Never mutate terminal records,
        including repeated terminal updates. Coaching is stored separately.
        """
        ...


class AnalysisDispatcher(Protocol):
    def dispatch(self, job: AnalysisJob) -> None:
        """Hand newly accepted work to future background processing.

        No execution technology is implied. Dispatch exceptions propagate.
        Atomic persistence plus durable dispatch is not implemented in M6A;
        later wiring must address this failure window before production use.
        """
        ...


class AnalysisIdFactory(Protocol):
    def create(self) -> AnalysisId:
        """Allocate AN- followed by at least six digits; uniqueness is adapter-owned."""
        ...


class CoachingRecordRepository(Protocol):
    """Separate append-only records; persistence technology is deliberately unspecified."""

    def get(self, analysis_id: AnalysisId) -> CoachingRecord | None: ...

    def append(self, record: CoachingRecord) -> CoachingRecord:
        """Atomically insert one artifact per analysis or return its identical record.

        Same analysis plus exactly identical output (including feedback ID,
        metadata and optional audio): return the existing record, no duplicate.
        Any conflicting output: raise FeedbackConflict; never replace history.
        Concurrent writes must obey the same rule. Source evidence validation
        belongs to CoachingIntegrationService before append; adapters validate
        the record contract. No update/delete operation or job mutation exists.
        """
        ...
