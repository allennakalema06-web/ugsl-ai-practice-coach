import fnmatch
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def apt_packages(command):
    tokens = shlex.split(command.replace('\\\n', ' '), comments=True)
    start = tokens.index('install', tokens.index('apt-get')) + 1
    packages = set()
    for token in tokens[start:]:
        if token in ('&&', ';', '|'):
            break
        if not token.startswith('-'):
            packages.add(token)
    return packages


def test_blueprint_topology_and_privilege():
    config = yaml.safe_load((ROOT / 'render.yaml').read_text())
    services, databases = config['services'], config['databases']
    assert len(services) == 3 and len(databases) == 1
    assert [s['type'] for s in services] == ['web', 'worker', 'worker']
    assert [s['name'] for s in services] == [f'ugsl-ai-coach-{name}' for name in
                                            ('api', 'analysis-worker', 'coaching-worker')]
    for service, role in zip(services, ('api', 'analysis-worker', 'coaching-worker')):
        assert service['runtime'] == 'docker' and service['region'] == 'frankfurt'
        assert service['plan'] == '0.5c-512mb' and service['numInstances'] == 1
        assert service['autoDeployTrigger'] == 'off' and 'scaling' not in service
        assert service['dockerCommand'] == f'python -m ugsl_ai_coach.deployment {role}'
        assert service['preDeployCommand'] == 'python -m ugsl_ai_coach.deployment migrate'
        assert service['maxShutdownDelaySeconds'] == (60 if role == 'api' else 300)
        variables = {v['key']: v for v in service['envVars']}
        assert variables['UGSL_DATABASE_URL']['fromDatabase'] == {
            'name': 'ugsl-ai-coach-db', 'property': 'connectionString'}
        assert variables['UGSL_ENVIRONMENT']['value'] == 'production'
        for key, value in variables.items():
            if key in ('UGSL_DATABASE_URL', 'UGSL_ENVIRONMENT', 'UGSL_API_DOCS_ENABLED'):
                continue
            assert value == {'key': key, 'sync': False}
        if role == 'coaching-worker':
            assert set(variables) == {'UGSL_DATABASE_URL', 'UGSL_ENVIRONMENT'}
        elif role == 'analysis-worker':
            assert 'UGSL_SERVICE_TOKEN' not in variables
    assert services[0]['healthCheckPath'] == '/api/v1/health'
    assert databases == [dict(name='ugsl-ai-coach-db', region='frankfurt', plan='0.1c-256mb',
                               postgresMajorVersion='18', ipAllowList=[])]
    assert not any(word in json.dumps(config).lower() for word in ('redis', 'keyvalue', 'celery', 'rabbitmq'))


def test_docker_install_and_non_root():
    docker = (ROOT / 'Dockerfile').read_text()
    assert docker.startswith('FROM python:3.13-slim')
    assert 'pip install --no-cache-dir .' in docker and '.[dev]' not in docker
    assert 'USER 10001:10001' in docker and 'install -d -o ugsl -g ugsl -m 0700 /var/tmp/ugsl' in docker
    assert 'PYTHONUNBUFFERED=1' in docker and 'PYTHONDONTWRITEBYTECODE=1' in docker
    assert 'CMD ["python", "-m", "ugsl_ai_coach.deployment", "api"]' in docker
    assert [line for line in docker.splitlines() if line.startswith('COPY ')] == [
        'COPY pyproject.toml README.md ./', 'COPY src/ ./src/']
    assert apt_packages(docker) == {'libgl1', 'libegl1', 'libgles2', 'libglib2.0-0', 'libportaudio2'}
    assert 'model_paths()' in docker and 'import cv2, mediapipe, psycopg, boto3, prometheus_client' in docker


def ignored(path):
    # This subset has no negations; a matched parent removes the whole subtree.
    patterns = [p.strip() for p in (ROOT / '.dockerignore').read_text().splitlines()
                if p.strip() and not p.startswith('#')]
    parts = path.split('/')
    candidates = ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
    return any(fnmatch.fnmatchcase(c, p) for c in candidates for p in patterns)


@pytest.mark.parametrize('path', ['.git/config', '.github/workflows/ci.yml', '.venv/a', '.env', '.env.production',
    '.codex-db-env.sh', 'src/.env.secret', 'src/.codex-db-env.sh', '.aws/credentials', 'src/.aws/credentials',
    '.local-test-tmp/a', '.pytest_cache/a', 'tests/a', 'src/__pycache__/a.pyc', 'coverage.xml', 'build/a',
    'dist/a.whl', '.models/hand_landmarker.task', 'local-data/learner.avi', 'src/private.mp4'])
def test_sensitive_files_excluded_from_context(path):
    assert ignored(path)


def test_runtime_assets_not_ignored():
    for name in ('hand_landmarker.task', 'pose_landmarker_lite.task', 'manifest.json', 'LICENSE', 'NOTICE.md'):
        assert not ignored('src/ugsl_ai_coach/assets/' + name)


def test_packaged_assets_keep_original_m3_hashes():
    from hashlib import sha256
    from ugsl_ai_coach.assets import model_paths
    assert [sha256(path.read_bytes()).hexdigest() for path in model_paths()] == [
        'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1',
        '59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a']


def test_prometheus_auth_file_and_delivery_boundary():
    config = yaml.safe_load((ROOT / 'ops/prometheus/prometheus.example.yml').read_text())
    scrape = config['scrape_configs'][0]
    assert scrape['scheme'] == 'https' and scrape['metrics_path'] == '/metrics'
    assert scrape['authorization'] == {'type': 'Bearer', 'credentials_file': '/run/secrets/ugsl-metrics-service-token'}
    assert 'credentials' not in scrape['authorization'] and 'alerting' not in config
    assert config['rule_files'] == ['/etc/prometheus/rules/alerts.yml']


def test_ci_release_gates():
    config = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    assert config['on'] == {'pull_request': None, 'push': {'branches': ['main']}}
    assert config['permissions'] == {'contents': 'read'}
    job = config['jobs']['test-and-container']
    assert job['services']['postgres']['image'] == 'postgres:18'
    assert job['env']['UGSL_TEST_DATABASE_URL']
    steps = job['steps']
    native = next(s for s in steps if s.get('name') == 'Minimal native runtime libraries')
    assert apt_packages(native['run']) == {'libgl1', 'libegl1', 'libgles2', 'libglib2.0-0', 'libportaudio2'}
    setup = next(s for s in steps if s.get('uses', '').startswith('actions/setup-python'))
    assert setup['with']['python-version'] == '3.13'
    runs = '\n'.join(s.get('run', '') for s in steps)
    for required in ('pytest --junitxml', 'ops/ci/verify_results.py', 'pip check', 'git diff --check',
                     'pip wheel', 'cd /tmp', 'docker build', 'docker run --rm'):
        assert required in runs
    assert not any(word in runs for word in ('docker push', 'render deploy', 'git push'))
    diagnostic = next(s for s in steps if s.get('name') == 'Real Hand and Pose initialization diagnostics')
    assert diagnostic['run'] == 'python -m ugsl_ai_coach.deployment.mediapipe_check'
    assert 'continue-on-error' not in diagnostic
    assert steps.index(diagnostic) < next(i for i, s in enumerate(steps) if s.get('name') == 'Complete suite with PostgreSQL')


@pytest.mark.parametrize('mode', ['pass', 'missing_env', 'skip', 'failure', 'undercollected'])
def test_ci_rejects_missing_or_skipped_postgres(tmp_path, monkeypatch, mode):
    spec = importlib.util.spec_from_file_location('ci_verify', ROOT / 'ops/ci/verify_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('UGSL_TEST_DATABASE_URL', 'disposable-ci-not-a-real-credential')
    if mode == 'missing_env': monkeypatch.delenv('UGSL_TEST_DATABASE_URL')
    root = ET.Element('testsuites')
    for i in range(120 if mode == 'undercollected' else 121):
        case = ET.SubElement(root, 'testcase', classname='tests.postgres.test_live', name=str(i))
        if i == 0 and mode in ('skip', 'failure'):
            ET.SubElement(case, 'skipped' if mode == 'skip' else 'failure')
    path = tmp_path / 'results.xml'
    ET.ElementTree(root).write(path)
    if mode == 'pass': module.verify(path)
    else:
        with pytest.raises(ValueError): module.verify(path)


def test_installed_wheel_outside_checkout(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for name in ('pyproject.toml', 'README.md'):
        shutil.copyfile(ROOT / name, source / name)
    shutil.copytree(ROOT / 'src', source / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    wheel_dir, installed, outside = (tmp_path / name for name in ('wheels', 'installed', 'outside'))
    outside.mkdir()
    def run(args, cwd, env=None):
        result = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout
    run(['-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--wheel-dir', str(wheel_dir), '.'], source)
    wheel, = wheel_dir.glob('*.whl')
    run(['-m', 'pip', 'install', '--no-deps', '--no-index', '--target', str(installed), str(wheel)], outside)
    env = dict(os.environ, PYTHONPATH=str(installed), EXPECTED_INSTALL=str(installed))
    output = run(['-c', '''import os
from pathlib import Path
import ugsl_ai_coach.main as main
from ugsl_ai_coach.assets import model_paths, extractor_identity
root = Path(os.environ['EXPECTED_INSTALL']).resolve()
assert Path(main.__file__).resolve().is_relative_to(root)
assert all(p.resolve().is_relative_to(root) for p in model_paths())
assert extractor_identity().startswith('mediapipe-tasks-1.0.1:')
import cv2, mediapipe, psycopg, boto3, prometheus_client
print('installed package and native imports verified')
'''], outside, env)
    assert 'installed package and native imports verified' in output
