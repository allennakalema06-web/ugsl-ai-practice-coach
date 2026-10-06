"""Stable safe transport errors: never serialize upstream exception text."""

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg import OperationalError, InterfaceError

from ugsl_ai_coach.api.auth import AuthenticationFailed, AuthorizationFailed, ServiceAuthConfigurationError
from ugsl_ai_coach.api.models import ApiError, ErrorEnvelope
from ugsl_ai_coach.infrastructure.postgres.connection import DatabaseConfigurationError
from ugsl_ai_coach.integration.errors import JobNotFound, IdempotencyConflict, AnalysisIdConflict
from ugsl_ai_coach.media.ports import (
    LearnerMediaNotFound, ReferenceProfileNotFound, ObjectStoreUnavailable,
    UnsupportedLearnerVideo, MediaTooLarge, CorruptReferenceProfile, IncompatibleReferenceProfile,
)
from ugsl_ai_coach.media.references import InvalidMediaReference, MediaConfigurationError
from ugsl_ai_coach.infrastructure.postgres.rate_limit import RateLimited
from ugsl_ai_coach.operations.events import emit

# Type, status, stable code, calm message, retryability.
ERRORS = (
    (RateLimited, 429, "RATE_LIMITED", "Service request capacity is temporarily exceeded.", True),
    (AuthenticationFailed, 401, "AUTHENTICATION_FAILED", "Valid backend service credentials are required.", False),
    (AuthorizationFailed, 403, "AUTHORIZATION_FAILED", "The backend service lacks the required permission.", False),
    (RequestValidationError, 422, "VALIDATION_ERROR", "The request does not match the required contract.", False),
    (InvalidMediaReference, 422, "INVALID_MEDIA_REFERENCE", "Media references must use the required internal namespace.", False),
    (JobNotFound, 404, "ANALYSIS_NOT_FOUND", "Analysis was not found.", False),
    (IdempotencyConflict, 409, "IDEMPOTENCY_CONFLICT", "The submission key identifies a different request.", False),
    (AnalysisIdConflict, 409, "ANALYSIS_ID_CONFLICT", "An analysis identity conflict occurred. Retry the submission.", True),
    (OperationalError, 503, "DATABASE_UNAVAILABLE", "Analysis persistence is temporarily unavailable.", True),
    (InterfaceError, 503, "DATABASE_UNAVAILABLE", "Analysis persistence is temporarily unavailable.", True),
    (DatabaseConfigurationError, 503, "DATABASE_UNAVAILABLE", "Analysis persistence is unavailable.", True),
    (ServiceAuthConfigurationError, 503, "SERVICE_UNAVAILABLE", "The internal service is unavailable.", True),
    (MediaConfigurationError, 503, "SERVICE_UNAVAILABLE", "Media integration is unavailable.", True),
    (LearnerMediaNotFound, 404, "LEARNER_MEDIA_NOT_FOUND", "Learner media was not found.", False),
    (ReferenceProfileNotFound, 404, "REFERENCE_PROFILE_NOT_FOUND", "The reference profile was not found.", False),
    (ObjectStoreUnavailable, 503, "OBJECT_STORE_UNAVAILABLE", "Private media storage is temporarily unavailable.", True),
    (UnsupportedLearnerVideo, 422, "UNSUPPORTED_LEARNER_VIDEO", "The learner video format is unsupported or unreadable.", False),
    (MediaTooLarge, 422, "MEDIA_TOO_LARGE", "The media object exceeds the supported size limit.", False),
    (CorruptReferenceProfile, 503, "CORRUPT_REFERENCE_PROFILE", "The reference profile requires correction.", False),
    (IncompatibleReferenceProfile, 503, "INCOMPATIBLE_REFERENCE_PROFILE", "The reference profile is incompatible with this processor.", False),
)


async def safe_error(request: Request, error: Exception):
    status, code, message, retryable = 500, "INTERNAL_ERROR", "The service could not complete the request.", True
    for error_type, candidate_status, candidate_code, candidate_message, candidate_retryable in ERRORS:
        if isinstance(error, error_type):
            status, code, message, retryable = candidate_status, candidate_code, candidate_message, candidate_retryable
            break
    events = {401: "authentication_failed", 403: "authorization_denied", 409: "idempotency_conflict", 429: "rate_limited"}
    emit(events.get(status, code.lower()), error_code=code, status=status)
    envelope = ErrorEnvelope(error=ApiError(code=code, message=message, retryable=retryable))
    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else {}
    if isinstance(error, RateLimited):
        headers["Retry-After"] = str(error.retry_after)
    return JSONResponse(status_code=status, content=envelope.model_dump(mode="json"), headers=headers)


def install_error_handlers(app):
    for error_type, *_ in ERRORS:
        app.add_exception_handler(error_type, safe_error)
    app.add_exception_handler(Exception, safe_error)
    # ServerErrorMiddleware re-raises generic exceptions after producing 500,
    # allowing an ASGI server to print sensitive upstream text/tracebacks.
    # Catch before that boundary so safe JSON is the final transport outcome.
    @app.middleware("http")
    async def exception_boundary(request, call_next):
        try:
            return await call_next(request)
        except Exception as error:
            return await safe_error(request, error)
