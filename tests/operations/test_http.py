import json
import logging
import re
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from psycopg import OperationalError

from ugsl_ai_coach.api.auth import ServicePrincipal, Scope
from ugsl_ai_coach.infrastructure.postgres.rate_limit import RateDecision
from ugsl_ai_coach.main import create_app
from ugsl_ai_coach.operations.metrics import Metrics
from .conftest import TOKEN, SnapshotDB


@pytest.fixture
def wiring(settings):
    app = create_app(settings)
    app.state.operations_connect = SnapshotDB().connect
    return TestClient(app), app


def test_correlation_generated_not_trusted_and_safe_logs(wiring, caplog):
    client, _ = wiring
    with caplog.at_level(logging.INFO, logger='ugsl_ai_coach.operations'):
        response = client.get('/api/v1/analyses/AN-000001?secret=media-key',
            headers={'X-Request-ID': 'learner@example.invalid', 'Authorization': 'Bearer secret-token'})
    assert response.status_code == 401
    assert re.fullmatch('[a-f0-9]{32}', response.headers['X-Request-ID'])
    events = [r for r in caplog.records if hasattr(r, 'event')]
    assert any(r.event == 'authentication_failed' for r in events)
    assert any(r.event == 'request_completed' and r.route == '/api/v1/analyses/{analysis_id}' for r in events)
    assert all(r.request_id == response.headers['X-Request-ID'] for r in events)
    for secret in ['secret-token', 'media-key', 'learner@example.invalid', 'AN-000001']:
        assert secret not in caplog.text
    assert client.get('/api/v1/health').headers['X-Request-ID'] != response.headers['X-Request-ID']


def test_rate_limit_precedes_storage_and_preserves_envelope(wiring):
    client, app = wiring
    calls = []
    class Limiter:
        def check(self, *args):
            calls.append(args)
            return RateDecision(False, 17)
    app.state.rate_limiter = Limiter()
    response = client.post('/api/v1/analyses', headers={'Authorization': 'Bearer ' + TOKEN},
        json={'attempt_id': 'ATT-000001', 'learner_video_ref': 'learner-videos/synthetic.mp4',
              'reference_profile_ref': 'reference-profiles/synthetic.json', 'idempotency_key': 'synthetic'})
    assert response.status_code == 429 and response.headers['Retry-After'] == '17'
    assert response.json()['error']['code'] == 'RATE_LIMITED' and response.json()['error']['retryable']
    assert calls == [('ugsl-platform-backend', 'submit', 60)]
    assert 'learner-videos' not in response.text and TOKEN not in response.text
    client.get('/api/v1/analyses/AN-000001', headers={'Authorization': 'Bearer ' + TOKEN})
    client.get('/api/v1/analyses/AN-000001/feedback', headers={'Authorization': 'Bearer ' + TOKEN})
    assert calls[1:] == [('ugsl-platform-backend', 'read', 600)] * 2


def test_limiter_db_failure_fails_closed(wiring):
    client, app = wiring
    class Broken:
        def check(self, *args): raise OperationalError('secret DSN')
    app.state.rate_limiter = Broken()
    response = client.get('/api/v1/analyses/AN-000001', headers={'Authorization': 'Bearer ' + TOKEN})
    assert response.status_code == 503 and response.json()['error']['code'] == 'DATABASE_UNAVAILABLE'
    assert 'secret' not in response.text


def test_metrics_scope_and_db_snapshot(wiring):
    client, app = wiring
    assert client.get('/metrics').status_code == 401
    class Auth:
        def authenticate(self, token): return ServicePrincipal('synthetic', frozenset({Scope.READ}))
    app.state.authenticator = Auth()
    assert client.get('/metrics', headers={'Authorization': 'Bearer test'}).status_code == 403
    del app.state.authenticator
    response = client.get('/metrics', headers={'Authorization': 'Bearer ' + TOKEN})
    assert response.status_code == 200 and 'text/plain' in response.headers['content-type']
    for expected in ['ugsl_postgres_up 1.0', 'ugsl_queue_depth{worker="analysis"} 4.0',
                     'ugsl_oldest_eligible_work_age_seconds{worker="coaching"} 40.0', 'ugsl_coaching_completed 2.0']:
        assert expected in response.text
    assert TOKEN not in response.text


def test_metrics_survive_database_failure(wiring):
    client, app = wiring
    def reject(): raise OperationalError('secret DSN')
    app.state.operations_connect = reject
    response = client.get('/metrics', headers={'Authorization': 'Bearer ' + TOKEN})
    assert response.status_code == 200 and 'ugsl_postgres_up 0.0' in response.text
    assert 'secret' not in response.text


@pytest.mark.parametrize('database_ok', [True, False])
def test_readiness_coarse_and_no_storage_download(wiring, database_ok, monkeypatch):
    import boto3
    def reject(*args, **kwargs): raise AssertionError('No storage download permitted')
    monkeypatch.setattr(boto3, 'client', reject)
    client, app = wiring
    if not database_ok:
        app.state.operations_connect = reject
    response = client.get('/api/v1/health/ready')
    assert response.status_code == (200 if database_ok else 503)
    assert response.json() == {'status': 'ready' if database_ok else 'not_ready'}
    assert client.get('/api/v1/health').status_code == 200


def test_unconfigured_readiness_fails_without_import_failure():
    from ugsl_ai_coach.core.config import Settings
    client = TestClient(create_app(Settings(_env_file=None)))
    assert client.get('/api/v1/health').status_code == 200
    assert client.get('/api/v1/health/ready').status_code == 503


def test_metric_labels_and_http_failures_are_bounded(wiring):
    client, app = wiring
    client.get('/api/v1/health')
    client.get('/api/v1/analyses/AN-123456')
    client.get('/unmatched-sensitive-path')
    text = app.state.metrics.render().decode()
    assert 'method="GET",route="/api/v1/health",status="200"' in text
    assert 'status="401"' in text and 'status="404"' in text
    for value in ['AN-123456', 'unmatched-sensitive-path', TOKEN]: assert value not in text
    for family in app.state.metrics.registry.collect():
        for sample in family.samples:
            assert not set(sample.labels) & {'analysis_id', 'attempt_id', 'feedback_id', 'request_id', 'token', 'media_key'}


def test_metrics_and_security_sink_failure_does_not_change_response(wiring, monkeypatch):
    client, app = wiring
    def reject(*args, **kwargs): raise RuntimeError('sink failure')
    monkeypatch.setattr(app.state.metrics, 'record_http', reject)
    import ugsl_ai_coach.operations.events as events
    monkeypatch.setattr(events.logger, 'info', reject)
    assert client.get('/api/v1/health').status_code == 200
    assert client.get('/api/v1/analyses/AN-000001').status_code == 401
