from typing import Literal

from ugsl_ai_coach.coaching.models import CoachingFeedback
from ugsl_ai_coach.domain.analysis import AnalysisId, AttemptId, ContractModel, StructuredAnalysisResult
from ugsl_ai_coach.integration.models import AnalysisJob, JobState


class AnalysisAccepted(ContractModel):
    analysis_id: AnalysisId
    attempt_id: AttemptId
    state: JobState


class AnalysisResponse(AnalysisAccepted):
    structured_analysis: StructuredAnalysisResult | None = None

    @classmethod
    def from_job(cls, job: AnalysisJob):
        job = AnalysisJob.model_validate(job.model_dump(mode="python"))
        return cls(analysis_id=job.analysis_id, attempt_id=job.attempt_id, state=job.state,
                   structured_analysis=job.structured_analysis)


class FeedbackPending(ContractModel):
    analysis_id: AnalysisId
    attempt_id: AttemptId
    state: Literal["PENDING"] = "PENDING"


class ApiError(ContractModel):
    code: str
    message: str
    retryable: bool


class ErrorEnvelope(ContractModel):
    error: ApiError
