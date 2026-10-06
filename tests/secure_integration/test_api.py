import json
import logging

import pytest
from fastapi.testclient import TestClient
from psycopg import OperationalError

from ugsl_ai_coach.api.application import AnalysisApiService
from ugsl_ai_coach.api.auth import Scope, ServicePrincipal, StaticTokenAuthenticator, ServiceAuthConfigurationError
from ugsl_ai_coach.coaching.engine import generate_feedback
from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.integration.errors import AnalysisIdConflict
from ugsl_ai_coach.integration.handoff.service import AnalysisHandoffService
from ugsl_ai_coach.integration.models import CoachingRecord
from ugsl_ai_coach.main import create_app
from ugsl_ai_coach.media.references import MediaReferencePolicy
from integration_handoff.fakes import FakePersistence, FakeIdFactory
from .conftest import SYNTHETIC_TOKEN


class Records:
    def __init__(self):
        self.records, self.writes = {}, 0
    def get(self, analysis_id):
        return self.records.get(analysis_id)
    def append(self, record):
        self.writes += 1
        self.records[record.analysis_id] = record
        return record


@pytest.fixture
def wiring(settings):
    persistence, records = FakePersistence(), Records()
    handoff = AnalysisHandoffService(persistence, FakeIdFactory())
    app = create_app(settings)
    app.state.rate_limiter = AllowedLimiter()
    app.state.analysis_api_service = AnalysisApiService(handoff, records, MediaReferencePolicy(settings), lambda: Resolver())
    client = TestClient(app, raise_server_exceptions=False)
    client.headers["Authorization"] = "Bearer " + SYNTHETIC_TOKEN
    return client, app, handoff, persistence, records


def test_post_acceptance_and_duplicate(wiring, submission):
    client, app, handoff, persistence, records = wiring
    first = client.post("/api/v1/analyses", json=submission.model_dump())
    assert first.status_code == 202
    assert first.json() == {"analysis_id": "AN-000001", "attempt_id": "ATT-000001", "state": "SUBMITTED"}
    assert client.post("/api/v1/analyses", json=submission.model_dump()).json() == first.json()
    assert len(persistence.store.pairs) == 1


def test_conflict(wiring, submission):
    client, *_ = wiring
    client.post("/api/v1/analyses", json=submission.model_dump())
    changed = {**submission.model_dump(), "learner_video_ref": "learner-videos/other.avi"}
    response = client.post("/api/v1/analyses", json=changed)
    assert response.status_code == 409 and response.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"


@pytest.mark.parametrize("change", [{"analysis_id": "AN-000001"}, {"score": 1}, {"video": "base64"},
    {"learner_email": "synthetic@example.invalid"}, {"attempt_id": "invalid"}])
def test_malformed_body_safe_422(wiring, submission, change):
    client, *_ = wiring
    response = client.post("/api/v1/analyses", json={**submission.model_dump(), **change})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "synthetic@example" not in response.text


@pytest.mark.parametrize("path,method", [("/api/v1/analyses", "post"), ("/api/v1/analyses/AN-000001", "get"),
    ("/api/v1/analyses/AN-000001/feedback", "get")])
@pytest.mark.parametrize("token", [None, "Bearer invalid", "Basic invalid", "Bearer", "Bearer Ω"])
def test_missing_invalid_credentials_401(wiring, submission, path, method, token):
    client, *_ = wiring
    client.headers.pop("Authorization", None)
    headers = [] if token is None else [(b"Authorization", token.encode("utf-8"))]
    response = getattr(client, method)(path, headers=headers,
                                     **({"json": submission.model_dump()} if method == "post" else {}))
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "AUTHENTICATION_FAILED"
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("path,method", [("/api/v1/analyses", "post"), ("/api/v1/analyses/AN-000001", "get"),
    ("/api/v1/analyses/AN-000001/feedback", "get")])
def test_authenticated_without_scope_403(wiring, submission, path, method):
    client, app, *_ = wiring
    class Auth:
        def authenticate(self, token):
            return ServicePrincipal("synthetic-principal", frozenset())
    app.state.authenticator = Auth()
    response = getattr(client, method)(path, **({"json": submission.model_dump()} if method == "post" else {}))
    assert response.status_code == 403 and response.json()["error"]["code"] == "AUTHORIZATION_FAILED"


def result(status):
    return StructuredAnalysisResult(analysis_id="AN-000001", attempt_id="ATT-000001", model_version="synthetic",
        status=status, overall_score=0.43 if status == "COMPLETED" else None,
        overall_confidence=None if status == "FAILED" else 0.9, findings=())


@pytest.mark.parametrize("state", ["SUBMITTED", "PROCESSING", "COMPLETED", "UNANALYZABLE", "FAILED"])
def test_status_actual_state_and_terminal_m2(wiring, submission, state):
    client, _, handoff, persistence, _ = wiring
    handoff.submit(submission)
    if state != "SUBMITTED":
        claim = handoff.claim_next(now_ms=100, lease_duration_ms=1000)
        handoff.begin(claim, now_ms=101)
        if state != "PROCESSING":
            handoff.complete(claim, result(state), now_ms=102)
    response = client.get("/api/v1/analyses/AN-000001")
    assert response.status_code == 200 and response.json()["state"] == state
    assert "learner_video_ref" not in response.text and "idempotency_key" not in response.text
    if state in ("SUBMITTED", "PROCESSING"):
        assert "structured_analysis" not in response.json()
    else:
        assert response.json()["structured_analysis"] == result(state).model_dump(mode="json")


@pytest.mark.parametrize("suffix", ["", "/feedback"])
def test_unknown_analysis_404(wiring, suffix):
    response = wiring[0].get("/api/v1/analyses/AN-000001" + suffix)
    assert response.status_code == 404 and response.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"


def test_feedback_pending_available_and_side_effect_free(wiring, submission):
    client, _, handoff, persistence, records = wiring
    handoff.submit(submission)
    response = client.get("/api/v1/analyses/AN-000001/feedback")
    assert response.status_code == 202 and response.json()["state"] == "PENDING"
    claim = handoff.claim_next(now_ms=100, lease_duration_ms=1000)
    handoff.begin(claim, now_ms=101)
    handoff.complete(claim, result("COMPLETED"), now_ms=102)
    assert client.get("/api/v1/analyses/AN-000001/feedback").status_code == 202
    feedback = generate_feedback(result("COMPLETED"), feedback_id="synthetic-feedback")
    records.records["AN-000001"] = CoachingRecord(analysis_id="AN-000001", attempt_id="ATT-000001", feedback=feedback)
    snapshot = persistence.store.snapshot()
    for _ in range(2):
        response = client.get("/api/v1/analyses/AN-000001/feedback")
        assert response.status_code == 200 and response.json() == feedback.model_dump(mode="json")
    assert records.writes == 0 and persistence.store.snapshot() == snapshot


@pytest.mark.parametrize("error,status,code", [(OperationalError("synthetic-credential"), 503, "DATABASE_UNAVAILABLE"),
    (AnalysisIdConflict("synthetic-credential"), 409, "ANALYSIS_ID_CONFLICT"),
    (RuntimeError("synthetic-credential"), 500, "INTERNAL_ERROR")])
def test_safe_error_envelope(wiring, submission, monkeypatch, error, status, code, caplog):
    client, app, *_ = wiring
    def reject(*args):
        raise error
    monkeypatch.setattr(app.state.analysis_api_service, "submit", reject)
    response = client.post("/api/v1/analyses", json=submission.model_dump())
    assert response.status_code == status
    assert set(response.json()) == {"error"} and set(response.json()["error"]) == {"code", "message", "retryable"}
    assert response.json()["error"]["code"] == code
    assert "synthetic-credential" not in response.text + caplog.text
    assert SYNTHETIC_TOKEN not in response.text + caplog.text


@pytest.mark.parametrize("token", [None, "", " " * 40, "a" * 64, "password-0123456789ABCDEFGHIJKLMNOP", "short"])
def test_weak_configuration_rejected_only_when_used(token):
    settings = Settings(_env_file=None, service_token=token)
    assert TestClient(create_app(settings)).get("/api/v1/health").status_code == 200
    with pytest.raises(ServiceAuthConfigurationError):
        StaticTokenAuthenticator(settings)


def test_openapi_security_and_scopes(wiring):
    schema = wiring[0].get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["UgSLBackendService"]["scheme"] == "bearer"
    for path, method, scope in [("/api/v1/analyses", "post", "analysis:submit"),
        ("/api/v1/analyses/{analysis_id}", "get", "analysis:read"),
        ("/api/v1/analyses/{analysis_id}/feedback", "get", "feedback:read")]:
        operation = schema["paths"][path][method]
        assert operation["security"] and operation["x-required-scopes"] == [scope]
    assert "security" not in schema["paths"]["/api/v1/health"]["get"]


@pytest.mark.parametrize("reference", ["http://host/x", "https://host/x", "file:///tmp/x", "learner-videos/../x.avi",
    "reference-profiles/expert.json"])
def test_submit_ssrf_reference_rejected_before_acceptance(wiring, submission, reference):
    client, _, _, persistence, _ = wiring
    response = client.post("/api/v1/analyses", json={**submission.model_dump(), "learner_video_ref": reference})
    assert response.status_code == 422 and response.json()["error"]["code"] == "INVALID_MEDIA_REFERENCE"
    assert persistence.store.pairs == {}


def test_unexpected_errors_do_not_escape_to_asgi_server(wiring, submission, monkeypatch):
    _, app, *_ = wiring
    def reject(*args):
        raise RuntimeError("synthetic-upstream-secret")
    monkeypatch.setattr(app.state.analysis_api_service, "submit", reject)
    # If it escapes, this client re-raises instead of returning the safe response.
    client = TestClient(app, raise_server_exceptions=True)
    response = client.post("/api/v1/analyses", json=submission.model_dump(),
                           headers={"Authorization": "Bearer " + SYNTHETIC_TOKEN})
    assert response.status_code == 500 and "synthetic-upstream-secret" not in response.text


def test_unconfigured_health_and_authentication_failure_do_not_require_database():
    client = TestClient(create_app(Settings(_env_file=None, service_token=None, database_url=None)))
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/analyses/AN-000001").status_code == 401
    response = client.get("/api/v1/analyses/AN-000001", headers={"Authorization": "Bearer invalid"})
    assert response.status_code == 503 and response.json()["error"]["code"] == "SERVICE_UNAVAILABLE"


def test_configured_token_missing_database_maps_safely(settings):
    client = TestClient(create_app(settings.model_copy(update={"database_url": None})))
    response = client.get("/api/v1/analyses/AN-000001", headers={"Authorization": "Bearer " + SYNTHETIC_TOKEN})
    assert response.status_code == 503 and response.json()["error"]["code"] == "DATABASE_UNAVAILABLE"


def test_single_scope_cannot_authorize_other_operations(wiring, submission):
    client, app, *_ = wiring
    class Auth:
        def authenticate(self, token):
            return ServicePrincipal("synthetic-submit-only", frozenset({Scope.SUBMIT}))
    app.state.authenticator = Auth()
    assert client.post("/api/v1/analyses", json=submission.model_dump()).status_code == 202
    assert client.get("/api/v1/analyses/AN-000001").status_code == 403
    assert client.get("/api/v1/analyses/AN-000001/feedback").status_code == 403


class Resolver:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def pin_learner(self, key):
        from ugsl_ai_coach.media.pinned import PinnedObject
        return PinnedObject(kind="learner", key=key, version_id="synthetic-version").encode()
    def pin_reference(self, key):
        from ugsl_ai_coach.media.pinned import PinnedObject
        return PinnedObject(kind="reference", key=key, version_id="synthetic-version").encode()


class AllowedLimiter:
    def check(self, *args):
        from ugsl_ai_coach.infrastructure.postgres.rate_limit import RateDecision
        return RateDecision(True, 1)
