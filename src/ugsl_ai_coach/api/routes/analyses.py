from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response, Security
from fastapi.responses import JSONResponse

from ugsl_ai_coach.api.application import AnalysisApiService, get_api_service
from ugsl_ai_coach.api.auth import ServicePrincipal, require_service
from ugsl_ai_coach.api.models import AnalysisAccepted, AnalysisResponse, ErrorEnvelope, FeedbackPending
from ugsl_ai_coach.coaching.models import CoachingFeedback
from ugsl_ai_coach.domain.analysis import AnalysisId
from ugsl_ai_coach.integration.models import AnalysisSubmission

router = APIRouter(prefix="/analyses", tags=["internal analyses"], responses={
    code: {"model": ErrorEnvelope} for code in (401, 403, 404, 409, 422, 500, 503)
})


def service(request: Request) -> AnalysisApiService:
    return get_api_service(request)


@router.post("", status_code=202, response_model=AnalysisAccepted, openapi_extra={"x-required-scopes": ["analysis:submit"]})
def submit_analysis(submission: AnalysisSubmission,
    principal: Annotated[ServicePrincipal, Security(require_service, scopes=["analysis:submit"])],
    application: Annotated[AnalysisApiService, Depends(service)]):
    job = application.submit(submission)
    return AnalysisAccepted(analysis_id=job.analysis_id, attempt_id=job.attempt_id, state=job.state)


@router.get("/{analysis_id}", response_model=AnalysisResponse,
            openapi_extra={"x-required-scopes": ["analysis:read"]})
def analysis_status(analysis_id: AnalysisId,
    principal: Annotated[ServicePrincipal, Security(require_service, scopes=["analysis:read"])],
    application: Annotated[AnalysisApiService, Depends(service)]):
    result = AnalysisResponse.from_job(application.get(analysis_id))
    return JSONResponse(content=result.model_dump(mode="json", exclude={"structured_analysis"}
                        if result.structured_analysis is None else set()))


@router.get("/{analysis_id}/feedback", response_model=CoachingFeedback | FeedbackPending,
            responses={202: {"model": FeedbackPending}}, openapi_extra={"x-required-scopes": ["feedback:read"]})
def feedback_status(analysis_id: AnalysisId, response: Response,
    principal: Annotated[ServicePrincipal, Security(require_service, scopes=["feedback:read"])],
    application: Annotated[AnalysisApiService, Depends(service)]):
    job, record = application.feedback(analysis_id)
    if record is None:
        response.status_code = 202
        return FeedbackPending(analysis_id=job.analysis_id, attempt_id=job.attempt_id)
    return record.feedback
