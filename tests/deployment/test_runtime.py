from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.deployment import runtime
from ugsl_ai_coach.deployment.configuration import StartupConfigurationError, validate_config


@pytest.fixture
def configured(monkeypatch):
    for name in ('UGSL_DATABASE_URL', 'UGSL_SERVICE_TOKEN', 'UGSL_OBJECT_STORE_BUCKET',
                 'UGSL_OBJECT_STORE_REGION', 'UGSL_HAND_MODEL_PATH', 'UGSL_POSE_MODEL_PATH'):
        monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None, environment='production', database_url='postgresql://localhost/disposable',
                    service_token='a-safe-test-token-with-at-least-32-characters',
                    object_store_bucket='private-fixture-bucket', object_store_region='eu-central-1')


@pytest.mark.parametrize('role', ['api', 'analysis-worker', 'coaching-worker', 'migrate'])
def test_missing_database_fails_without_secret(configured, role):
    secret = 'sensitive-dsn-password-marker'
    with pytest.raises(StartupConfigurationError) as error:
        validate_config(role, configured.model_copy(update={'database_url': None}))
    assert secret not in str(error.value)
    assert role in str(error.value)


@pytest.mark.parametrize('role', ['coaching-worker', 'migrate'])
def test_no_storage_or_api_secret_required(configured, role):
    settings = configured.model_copy(update={'service_token': None, 'object_store_bucket': None,
                                              'object_store_region': None})
    assert validate_config(role, settings) is settings


@pytest.mark.parametrize('role', ['api', 'analysis-worker'])
def test_production_rejects_insecure_storage(configured, role):
    with pytest.raises(StartupConfigurationError) as error:
        validate_config(role, configured.model_copy(update={
            'object_store_endpoint_url': 'http://sensitive-storage-host.invalid'}))
    assert 'sensitive-storage-host' not in str(error.value)


def test_bundled_defaults_and_explicit_missing_models(configured):
    from pathlib import Path
    settings = validate_config('analysis-worker', configured)
    assert Path(settings.hand_model_path).is_file() and Path(settings.pose_model_path).is_file()
    with pytest.raises(StartupConfigurationError):
        validate_config('analysis-worker', configured.model_copy(update={'hand_model_path': '/missing/operator-model.task'}))


def test_api_port_and_safe_single_process(configured, monkeypatch):
    import uvicorn
    calls = []
    monkeypatch.setenv('PORT', '12345')
    monkeypatch.setattr(uvicorn, 'run', lambda app, **kwargs: calls.append((app, kwargs)))
    runtime.run_role('api', configured)
    app, options = calls[0]
    assert options == dict(host='0.0.0.0', port=12345, workers=1, access_log=False,
                           log_config=None, timeout_graceful_shutdown=60, proxy_headers=False)
    assert app.state.settings is configured


@pytest.mark.parametrize('port', ['0', '65536', 'secret-port-value', '-1', '1.2'])
def test_invalid_port_safe(configured, monkeypatch, port):
    monkeypatch.setenv('PORT', port)
    with pytest.raises(StartupConfigurationError) as error:
        validate_config('api', configured)
    assert port not in str(error.value)


def test_analysis_uses_production_processor_and_closes_client(configured, monkeypatch):
    import ugsl_ai_coach.worker.processor as processor_module
    import ugsl_ai_coach.worker.runtime as worker_runtime
    calls = []
    processor = SimpleNamespace(media=SimpleNamespace(client=SimpleNamespace(close=lambda: calls.append('closed'))))
    monkeypatch.setattr(processor_module, 'create_production_processor',
                        lambda settings: calls.append(('production', settings)) or processor)
    monkeypatch.setattr(worker_runtime, 'run_postgres_worker',
                        lambda actual, settings: calls.append(('worker', actual, settings)))
    runtime.run_role('analysis-worker', configured)
    assert calls[0][0] == 'production' and calls[0][1].hand_model_path
    assert calls[1][1] is processor and calls[-1] == 'closed'


def test_coaching_uses_existing_deterministic_provider(configured, monkeypatch):
    import ugsl_ai_coach.worker.coaching as coaching
    import ugsl_ai_coach.worker.runtime as worker_runtime
    from ugsl_ai_coach.coaching.providers.deterministic import DeterministicCoachingProvider
    from ugsl_ai_coach.infrastructure.postgres.coaching_work import PostgresCoachingWork
    from ugsl_ai_coach.infrastructure.postgres.telemetry import PostgresWorkerMetrics
    calls = []
    class Worker:
        def __init__(self, work, processor, settings, metrics):
            assert isinstance(work, PostgresCoachingWork)
            assert isinstance(processor.provider, DeterministicCoachingProvider)
            assert isinstance(metrics, PostgresWorkerMetrics)
            assert settings is configured
        def run(self):
            calls.append('ran')
    monkeypatch.setattr(coaching, 'CoachingWorker', Worker)
    monkeypatch.setattr(worker_runtime, 'shutdown_signals', lambda worker: nullcontext())
    runtime.run_role('coaching-worker', configured)
    assert calls == ['ran']


def test_migrate_only_explicit_runner(configured, monkeypatch, capsys):
    import ugsl_ai_coach.infrastructure.postgres.migrate as migration
    calls = []
    monkeypatch.setattr(migration, 'migrate', lambda connect: calls.append(connect) or ['001', '002'])
    runtime.run_role('migrate', configured)
    assert len(calls) == 1 and callable(calls[0])
    assert capsys.readouterr().out == 'Migrations complete: 2 applied\n'


@pytest.mark.parametrize('command', ['api', 'analysis-worker', 'coaching-worker', 'migrate'])
def test_cli_routes(configured, monkeypatch, command):
    from ugsl_ai_coach.deployment.__main__ import main
    import ugsl_ai_coach.core.config as config
    calls = []
    monkeypatch.setattr(config, 'Settings', lambda: configured)
    monkeypatch.setattr(runtime, 'run_role', lambda role, settings: calls.append((role, settings)))
    assert main([command]) == 0 and calls == [(command, configured)]


def test_cli_safe_failure_and_check_only(configured, monkeypatch, capsys):
    from ugsl_ai_coach.deployment.__main__ import main
    import ugsl_ai_coach.core.config as config
    monkeypatch.setattr(config, 'Settings', lambda: configured)
    def fail(*args):
        raise RuntimeError('sensitive-password-do-not-print')
    monkeypatch.setattr(runtime, 'run_role', fail)
    assert main(['api']) == 1
    assert 'sensitive-password' not in capsys.readouterr().out
    assert main(['check-config', 'migrate']) == 0


def test_production_docs_disabled_contract_unchanged(configured):
    from ugsl_ai_coach.main import create_app
    with TestClient(create_app(configured)) as client:
        for path in ('/docs', '/redoc', '/openapi.json'):
            assert client.get(path).status_code == 404
        assert client.get('/api/v1/contracts/analysis').status_code == 200
    assert create_app(configured.model_copy(update={'environment': 'development'})).openapi_url == '/openapi.json'


def test_deployment_imports_and_help_have_no_infrastructure_io(tmp_path):
    import subprocess
    import sys
    code = '''import sys, socket, boto3, psycopg
def blocked(*args, **kwargs): raise AssertionError('Infrastructure IO during import/help')
socket.create_connection = blocked
boto3.client = blocked
psycopg.connect = blocked
import ugsl_ai_coach.deployment.runtime
import ugsl_ai_coach.deployment.e2e
from ugsl_ai_coach.deployment.__main__ import main
try: main([sys.argv[1], '--help'])
except SystemExit as error: assert error.code == 0
'''
    for role in ('api', 'analysis-worker', 'coaching-worker', 'migrate'):
        result = subprocess.run([sys.executable, '-c', code, role], cwd=tmp_path,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
