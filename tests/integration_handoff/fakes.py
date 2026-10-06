"""Sequential NON-PRODUCTION reference adapter. Memory is NOT durable.

Snapshot reconstruction simulates persisted-state restart for contract tests only.
No locks, threads, processes, filesystem writes or production startup registration.
"""

from ugsl_ai_coach.integration.errors import AdapterContractError, AnalysisIdConflict, IdempotencyConflict, JobNotFound
from ugsl_ai_coach.integration.handoff.lifecycle import claim_work, complete_work, validate_active_claim, validate_time, validate_work_transition
from ugsl_ai_coach.integration.handoff.models import JobWorkPair, WorkAcceptance, WorkState
from ugsl_ai_coach.integration.lifecycle import transition_job
from ugsl_ai_coach.integration.models import JobState


class FakeStore:
    def __init__(self):
        self.pairs = {}

    def snapshot(self):
        return {key: pair.model_dump_json() for key, pair in self.pairs.items()}

    @classmethod
    def restore(cls, snapshot):
        store = cls()
        store.pairs = {key: JobWorkPair.model_validate_json(value) for key, value in snapshot.items()}
        for key, pair in store.pairs.items():
            if key != pair.job.analysis_id:
                raise AdapterContractError("Snapshot key does not match pair")
        return store


class FakePersistence:
    def __init__(self, store=None):
        self.store = store if store is not None else FakeStore()

    def get_pair(self, analysis_id):
        pair = self.store.pairs.get(analysis_id)
        if pair is None:
            return None
        pair = JobWorkPair.model_validate(pair.model_dump(mode="python"))
        if pair.job.analysis_id != analysis_id:
            raise AdapterContractError("Stored pair does not match key")
        return pair

    def get(self, analysis_id):
        pair = self.get_pair(analysis_id)
        return pair.job if pair is not None else None

    def get_by_idempotency_key(self, key):
        for analysis_id in sorted(self.store.pairs):
            pair = self.get_pair(analysis_id)
            if pair.job.idempotency_key == key:
                return pair
        return None

    def before_commit(self, pair):
        """Fault-injection seam: every check happens before the single commit."""

    def _commit(self, pair):
        pair = JobWorkPair.model_validate(pair.model_dump(mode="python"))
        self.before_commit(pair)
        # One combined pair, one state replacement. There is no separate job write.
        self.store.pairs = {**self.store.pairs, pair.job.analysis_id: pair}
        return pair

    def accept(self, candidate):
        candidate = WorkAcceptance(pair=candidate, created=True).pair
        existing = self.get_by_idempotency_key(candidate.job.idempotency_key)
        if existing is not None:
            if existing.job.submission != candidate.job.submission:
                raise IdempotencyConflict("Conflicting logical submission")
            return WorkAcceptance(pair=existing, created=False)
        if candidate.job.analysis_id in self.store.pairs:
            raise AnalysisIdConflict("Analysis ID already exists")
        return WorkAcceptance(pair=self._commit(candidate), created=True)

    def claim_next(self, *, now_ms, lease_duration_ms):
        validate_time(now_ms, lease_duration_ms)
        for analysis_id in sorted(self.store.pairs):
            pair = self.get_pair(analysis_id)
            work = pair.work
            if work.state == WorkState.COMPLETED or (
                work.state == WorkState.CLAIMED and now_ms < work.lease_expires_at_ms
            ):
                continue
            claimed = claim_work(work, now_ms=now_ms, lease_duration_ms=lease_duration_ms)
            validate_work_transition(work, claimed, now_ms=now_ms, lease_duration_ms=lease_duration_ms)
            return self._commit(JobWorkPair(job=pair.job, work=claimed))
        return None

    def _owned_pair(self, expected, now_ms):
        pair = self.get_pair(expected.analysis_id)
        if pair is None:
            raise JobNotFound("Job/work pair was not found")
        validate_active_claim(pair.work, expected, now_ms=now_ms)
        return pair

    def begin(self, expected, *, now_ms):
        pair = self._owned_pair(expected, now_ms)
        if pair.job.state == JobState.PROCESSING:
            return pair
        job = transition_job(pair.job, JobState.PROCESSING)
        return self._commit(JobWorkPair(job=job, work=pair.work))

    def finish(self, expected, result, *, now_ms):
        pair = self._owned_pair(expected, now_ms)
        job = transition_job(pair.job, JobState(result.status.value), structured_analysis=result)
        work = complete_work(pair.work, now_ms=now_ms)
        validate_work_transition(pair.work, work, now_ms=now_ms)
        return self._commit(JobWorkPair(job=job, work=work))


class FakeIdFactory:
    def __init__(self):
        self.calls = 0

    def create(self):
        self.calls += 1
        return f"AN-{self.calls:06d}"
