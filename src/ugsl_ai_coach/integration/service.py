"""Lifecycle coordination only; injected ports own storage, dispatch and IDs."""

from pydantic import TypeAdapter

from ugsl_ai_coach.coaching.grounding import ground_analysis
from ugsl_ai_coach.coaching.models import CoachingFeedback
from ugsl_ai_coach.coaching.validation import validate_feedback
from ugsl_ai_coach.domain.analysis import AnalysisId, StructuredAnalysisResult
from ugsl_ai_coach.integration.errors import AdapterContractError, AnalysisNotTerminal, IdempotencyConflict, JobNotFound
from ugsl_ai_coach.integration.lifecycle import transition_job
from ugsl_ai_coach.integration.models import AnalysisJob, AnalysisSubmission, CoachingRecord, JobAcceptance, JobState
from ugsl_ai_coach.integration.ports import AnalysisDispatcher, AnalysisIdFactory, AnalysisJobRepository, CoachingRecordRepository


def _get_job(repository: AnalysisJobRepository, analysis_id: AnalysisId) -> AnalysisJob:
    analysis_id = TypeAdapter(AnalysisId).validate_python(analysis_id)
    job = repository.get(analysis_id)
    if job is None:
        raise JobNotFound("Analysis job was not found")
    job = AnalysisJob.model_validate(job.model_dump(mode="python"))
    if job.analysis_id != analysis_id:
        raise AdapterContractError("Repository returned a different analysis job")
    return job


class AnalysisIntegrationService:
    def __init__(
        self, repository: AnalysisJobRepository, dispatcher: AnalysisDispatcher,
        id_factory: AnalysisIdFactory,
    ) -> None:
        self.repository = repository
        self.dispatcher = dispatcher
        self.id_factory = id_factory

    def submit(self, submission: AnalysisSubmission) -> AnalysisJob:
        submission = AnalysisSubmission.model_validate(submission.model_dump(mode="python"))
        existing = self.repository.get_by_idempotency_key(submission.idempotency_key)
        if existing is not None:
            existing = AnalysisJob.model_validate(existing.model_dump(mode="python"))
            if existing.submission != submission:
                raise IdempotencyConflict("Idempotency key already identifies a different submission")
            return existing
        candidate = AnalysisJob(
            analysis_id=self.id_factory.create(), state=JobState.SUBMITTED,
            **submission.model_dump(mode="python"),
        )
        # The fast lookup is not a lock: accept must resolve concurrent submissions.
        acceptance = self.repository.accept(candidate)
        acceptance = JobAcceptance.model_validate(acceptance.model_dump(mode="python"))
        if acceptance.job.submission != submission or (
            acceptance.created and acceptance.job != candidate
        ):
            raise AdapterContractError("Repository acceptance does not match the submission")
        if acceptance.created:
            self.dispatcher.dispatch(acceptance.job)
        return acceptance.job

    def get(self, analysis_id: AnalysisId) -> AnalysisJob:
        return _get_job(self.repository, analysis_id)

    def mark_processing(self, analysis_id: AnalysisId) -> AnalysisJob:
        return self._update(analysis_id, JobState.PROCESSING)

    def complete(
        self, analysis_id: AnalysisId, structured_analysis: StructuredAnalysisResult,
    ) -> AnalysisJob:
        structured_analysis = StructuredAnalysisResult.model_validate(
            structured_analysis.model_dump(mode="python"),
        )
        return self._update(
            analysis_id, JobState(structured_analysis.status.value),
            structured_analysis=structured_analysis,
        )

    def _update(
        self, analysis_id: AnalysisId, state: JobState, *,
        structured_analysis: StructuredAnalysisResult | None = None,
    ) -> AnalysisJob:
        previous = self.get(analysis_id)
        replacement = transition_job(
            previous, state, structured_analysis=structured_analysis,
        )
        saved = self.repository.compare_and_set(previous, replacement)
        saved = AnalysisJob.model_validate(saved.model_dump(mode="python"))
        if saved != replacement:
            raise AdapterContractError("Repository update does not match the requested transition")
        return saved


class CoachingIntegrationService:
    """Validate existing M5 output and append it after immutable analysis persistence."""

    def __init__(
        self, jobs: AnalysisJobRepository, records: CoachingRecordRepository,
    ) -> None:
        self.jobs = jobs
        self.records = records

    def record_feedback(self, analysis_id: AnalysisId, feedback: CoachingFeedback) -> CoachingRecord:
        job = _get_job(self.jobs, analysis_id)
        if not job.state.is_terminal or job.structured_analysis is None:
            raise AnalysisNotTerminal("Coaching requires terminal structured analysis evidence")
        record = CoachingRecord(
            analysis_id=job.analysis_id, attempt_id=job.attempt_id, feedback=feedback,
        )
        # Existing M5 validation only: no provider or coaching generation is invoked.
        context = ground_analysis(job.structured_analysis, record.feedback.feedback_id)
        validate_feedback(
            record.feedback, context, provider_type=record.feedback.provider_metadata.provider_type,
        )
        saved = self.records.append(record)
        saved = CoachingRecord.model_validate(saved.model_dump(mode="python"))
        if saved != record:
            raise AdapterContractError("Coaching repository returned a different artifact")
        return saved
