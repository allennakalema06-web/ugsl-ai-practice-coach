"""Atomic durable-persistence obligations; transport and storage are unspecified."""

from typing import Protocol

from ugsl_ai_coach.domain.analysis import AnalysisId, StructuredAnalysisResult
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem, JobWorkPair, WorkAcceptance
from ugsl_ai_coach.integration.ports import AnalysisJobReader


class AnalysisPersistence(AnalysisJobReader, Protocol):
    """Production adapters must enforce these operations within durable atomic boundaries.

    A successful commit survives process restart. No job-only acceptance/update
    operation is exposed. Reads reject missing/mismatched job/work records rather
    than repair them. get() is a checked read-only view for separate M6A coaching.
    Time values are trusted shared UTC epoch milliseconds, supplied explicitly;
    future adapters must obtain/validate authoritative time, not trust clients.
    """

    def get_pair(self, analysis_id: AnalysisId) -> JobWorkPair | None: ...

    def get_by_idempotency_key(self, key: str) -> JobWorkPair | None: ...

    def accept(self, candidate: JobWorkPair) -> WorkAcceptance:
        """Atomically commit a new SUBMITTED job plus PENDING work, or deduplicate.

        Same key+payload returns existing pair, created=False, at any lifecycle
        stage. Conflicting key payload raises IdempotencyConflict. A new key with
        an existing analysis ID raises AnalysisIdConflict. Exactly one concurrent
        accept creates the pair. No partial commit or acknowledged job without
        work is permitted. Reject invalid/inconsistent existing records.
        """
        ...

    def claim_next(self, *, now_ms: int, lease_duration_ms: int) -> JobWorkPair | None:
        """Atomically claim eligible PENDING or expired CLAIMED work; otherwise None.

        Increment delivery_count as a fencing generation. Keep job state and all
        IDs/references unchanged. Only one active claimant; completed work is
        never eligible. Expiry is inclusive: now >= lease_expires_at. Selection
        ordering/fairness is adapter-owned. Validate pair and lease transition.
        """
        ...

    def begin(self, expected: AnalysisWorkItem, *, now_ms: int) -> JobWorkPair:
        """Atomically verify the exact active claim and begin/resume analysis.

        SUBMITTED -> PROCESSING; already PROCESSING resumes without rewriting
        its lifecycle/evidence. Terminal jobs cannot begin. Stale/expired claim
        raises ClaimConflict. Preserve work metadata; no independent job write.
        """
        ...

    def finish(
        self, expected: AnalysisWorkItem, result: StructuredAnalysisResult, *, now_ms: int,
    ) -> JobWorkPair:
        """Atomically verify claim, finalize PROCESSING job, and complete work.

        M2 status/IDs must match; apply M6A transition rules. Reject stale/expired
        deliveries and repeated terminal writes. Analysis FAILED is accepted only
        as an actual supplied M2 result, never synthesized from delivery errors.
        No partial terminal-job/work commit. Coaching is never part of this write.
        """
        ...
