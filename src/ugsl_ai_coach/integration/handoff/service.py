"""M6B entry point: acceptance commits work; no immediate dispatch is required."""

from pydantic import TypeAdapter

from ugsl_ai_coach.domain.analysis import AnalysisId, StructuredAnalysisResult
from ugsl_ai_coach.integration.errors import AdapterContractError, IdempotencyConflict, JobNotFound
from ugsl_ai_coach.integration.handoff.lifecycle import (
    validate_active_claim, validate_time, validate_work_transition,
)
from ugsl_ai_coach.integration.handoff.models import AnalysisWorkItem, JobWorkPair, WorkAcceptance, WorkState
from ugsl_ai_coach.integration.handoff.ports import AnalysisPersistence
from ugsl_ai_coach.integration.lifecycle import transition_job
from ugsl_ai_coach.integration.models import AnalysisJob, AnalysisSubmission, JobState
from ugsl_ai_coach.integration.ports import AnalysisIdFactory


class AnalysisHandoffService:
    def __init__(self, persistence: AnalysisPersistence, id_factory: AnalysisIdFactory) -> None:
        self.persistence = persistence
        self.id_factory = id_factory

    def get(self, analysis_id: AnalysisId) -> JobWorkPair:
        analysis_id = TypeAdapter(AnalysisId).validate_python(analysis_id)
        pair = self.persistence.get_pair(analysis_id)
        if pair is None:
            raise JobNotFound("Analysis job/work pair was not found")
        pair = JobWorkPair.model_validate(pair.model_dump(mode="python"))
        if pair.job.analysis_id != analysis_id:
            raise AdapterContractError("Persistence returned a different job/work pair")
        return pair

    def submit(self, submission: AnalysisSubmission) -> AnalysisJob:
        submission = AnalysisSubmission.model_validate(submission.model_dump(mode="python"))
        existing = self.persistence.get_by_idempotency_key(submission.idempotency_key)
        if existing is not None:
            existing = JobWorkPair.model_validate(existing.model_dump(mode="python"))
            if existing.job.submission != submission:
                raise IdempotencyConflict("Idempotency key identifies a different submission")
            return existing.job
        job = AnalysisJob(analysis_id=self.id_factory.create(), state=JobState.SUBMITTED,
                          **submission.model_dump(mode="python"))
        work = AnalysisWorkItem(
            analysis_id=job.analysis_id, attempt_id=job.attempt_id,
            learner_video_ref=job.learner_video_ref, reference_profile_ref=job.reference_profile_ref,
            state=WorkState.PENDING, delivery_count=0,
        )
        candidate = JobWorkPair(job=job, work=work)
        accepted = self.persistence.accept(candidate)
        accepted = WorkAcceptance.model_validate(accepted.model_dump(mode="python"))
        if accepted.pair.job.submission != submission or (accepted.created and accepted.pair != candidate):
            raise AdapterContractError("Persistence acceptance does not match submission")
        return accepted.pair.job

    def claim_next(self, *, now_ms: int, lease_duration_ms: int) -> AnalysisWorkItem | None:
        validate_time(now_ms, lease_duration_ms)
        pair = self.persistence.claim_next(now_ms=now_ms, lease_duration_ms=lease_duration_ms)
        if pair is None:
            return None
        pair = JobWorkPair.model_validate(pair.model_dump(mode="python"))
        validate_active_claim(pair.work, pair.work, now_ms=now_ms)
        if pair.work.claimed_at_ms != now_ms or pair.work.lease_expires_at_ms != now_ms + lease_duration_ms:
            raise AdapterContractError("Persistence returned different lease terms")
        return pair.work

    def begin(self, claimed: AnalysisWorkItem, *, now_ms: int) -> AnalysisJob:
        claimed = AnalysisWorkItem.model_validate(claimed.model_dump(mode="python"))
        previous = self.get(claimed.analysis_id)
        validate_active_claim(previous.work, claimed, now_ms=now_ms)
        expected_job = previous.job if previous.job.state == JobState.PROCESSING else transition_job(
            previous.job, JobState.PROCESSING,
        )
        saved = self.persistence.begin(claimed, now_ms=now_ms)
        saved = JobWorkPair.model_validate(saved.model_dump(mode="python"))
        if saved.job != expected_job or saved.work != claimed:
            raise AdapterContractError("Persistence changed the begin operation")
        return saved.job

    def complete(
        self, claimed: AnalysisWorkItem, result: StructuredAnalysisResult, *, now_ms: int,
    ) -> AnalysisJob:
        claimed = AnalysisWorkItem.model_validate(claimed.model_dump(mode="python"))
        result = StructuredAnalysisResult.model_validate(result.model_dump(mode="python"))
        previous = self.get(claimed.analysis_id)
        validate_active_claim(previous.work, claimed, now_ms=now_ms)
        expected_job = transition_job(previous.job, JobState(result.status.value), structured_analysis=result)
        saved = self.persistence.finish(claimed, result, now_ms=now_ms)
        saved = JobWorkPair.model_validate(saved.model_dump(mode="python"))
        validate_work_transition(claimed, saved.work, now_ms=now_ms)
        if saved.work.state != WorkState.COMPLETED or saved.job != expected_job:
            raise AdapterContractError("Persistence changed the terminal result")
        return saved.job
