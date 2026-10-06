"""Service identity only; platform user authentication remains upstream."""

from dataclasses import dataclass
from enum import StrEnum
from hmac import compare_digest
import re
from typing import Annotated, Protocol

from fastapi import Depends, Request, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, SecurityScopes

from ugsl_ai_coach.core.config import Settings


class Scope(StrEnum):
    SUBMIT = "analysis:submit"
    READ = "analysis:read"
    FEEDBACK = "feedback:read"
    METRICS = "metrics:read"


@dataclass(frozen=True)
class ServicePrincipal:
    identity: str
    scopes: frozenset[Scope]


class AuthenticationFailed(ValueError): pass
class AuthorizationFailed(ValueError): pass
class ServiceAuthConfigurationError(ValueError): pass


class ServiceAuthenticator(Protocol):
    def authenticate(self, token: str) -> ServicePrincipal: ...


class ServiceAuthorizer(Protocol):
    def authorize(self, principal: ServicePrincipal, required: list[str]) -> None: ...


class ScopeAuthorizer:
    def authorize(self, principal: ServicePrincipal, required: list[str]) -> None:
        if not set(required).issubset(principal.scopes):
            raise AuthorizationFailed("Service scope is required")


class StaticTokenAuthenticator:
    def __init__(self, settings: Settings):
        secret = settings.service_token
        token = None if secret is None else secret.get_secret_value()
        # At least 32 ASCII bearer characters; reject trivial/repeated examples.
        # Entropy cannot be measured from a string: operators must generate a
        # cryptographically random token (recommended 32 random bytes or more).
        if (token is None or not re.fullmatch(r"[A-Za-z0-9._~+/-]{32,256}={0,2}", token)
                or len(set(token)) < 12 or token.lower().startswith(("changeme", "example", "password"))):
            raise ServiceAuthConfigurationError("A strong UGSL_SERVICE_TOKEN is required")
        self._token = token.encode("ascii")

    def authenticate(self, token: str) -> ServicePrincipal:
        try:
            encoded = token.encode("ascii")
        except (AttributeError, UnicodeError):
            raise AuthenticationFailed("Service credentials are invalid") from None
        if not compare_digest(encoded, self._token):
            raise AuthenticationFailed("Service credentials are invalid")
        return ServicePrincipal("ugsl-platform-backend", frozenset(Scope))


bearer = HTTPBearer(auto_error=False, scheme_name="UgSLBackendService", description="Internal UgSL backend service token")


def get_authenticator(request: Request) -> ServiceAuthenticator:
    injected = getattr(request.app.state, "authenticator", None)
    return injected if injected is not None else StaticTokenAuthenticator(request.app.state.settings)


def require_service(
    request: Request,
    security_scopes: SecurityScopes,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(bearer)],
) -> ServicePrincipal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AuthenticationFailed("Service credentials are required")
    principal = get_authenticator(request).authenticate(credentials.credentials)
    authorizer: ServiceAuthorizer = getattr(request.app.state, "authorizer", None) or ScopeAuthorizer()
    authorizer.authorize(principal, security_scopes.scopes)
    if any(scope in security_scopes.scopes for scope in (Scope.SUBMIT, Scope.READ, Scope.FEEDBACK)):
        from ugsl_ai_coach.infrastructure.postgres.connection import connection_factory
        from ugsl_ai_coach.infrastructure.postgres.rate_limit import PostgresRateLimiter, RateLimited
        operation = "submit" if Scope.SUBMIT in security_scopes.scopes else "read"
        settings = request.app.state.settings
        limiter = getattr(request.app.state, "rate_limiter", None)
        if limiter is None:
            limiter = PostgresRateLimiter(connection_factory(settings))
        threshold = settings.submit_rate_limit_per_minute if operation == "submit" else settings.read_rate_limit_per_minute
        decision = limiter.check(principal.identity, operation, threshold)
        if not decision.allowed:
            from ugsl_ai_coach.operations.metrics import best_effort
            best_effort(lambda: request.app.state.metrics.rate.labels(operation).inc())
            raise RateLimited(decision.retry_after)
    return principal
