import json
from types import SimpleNamespace

import pytest

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.deployment import e2e
from ugsl_ai_coach.domain.analysis import StructuredAnalysisResult
from ugsl_ai_coach.coaching.engine import generate_feedback


@pytest.fixture
def staging_environment(monkeypatch):
    for key, value in dict(UGSL_E2E_ENVIRONMENT='staging', UGSL_E2E_BASE_URL='https://staging.example.invalid',
        UGSL_E2E_ALLOWED_HOST='staging.example.invalid', UGSL_E2E_SERVICE_TOKEN='random-fixture-token-at-least-32-characters',
        UGSL_OBJECT_STORE_BUCKET='private-test-bucket', UGSL_OBJECT_STORE_REGION='eu-central-1').items():
        monkeypatch.setenv(key, value)


def test_default_cli_refuses_before_any_upload(monkeypatch, capsys):
    monkeypatch.delenv('UGSL_E2E_ENVIRONMENT', raising=False)
    monkeypatch.setattr(e2e, 'run', lambda *args, **kwargs: pytest.fail('Unsafe live execution'))
    assert e2e.main([]) == 1
    assert 'configuration' in capsys.readouterr().out


@pytest.mark.parametrize('base,allowed', [('http://staging.example.invalid', 'staging.example.invalid'),
    ('https://production.example.invalid', 'production.example.invalid'),
    ('https://staging.example.invalid/path', 'staging.example.invalid'),
    ('https://staging.example.invalid', 'other.example.invalid'),
    ('https://user:secret@staging.example.invalid', 'staging.example.invalid')])
def test_staging_host_refusals(staging_environment, monkeypatch, base, allowed):
    monkeypatch.setenv('UGSL_E2E_BASE_URL', base)
    monkeypatch.setenv('UGSL_E2E_ALLOWED_HOST', allowed)
    with pytest.raises(e2e.E2EError):
        e2e.StagingConfig.from_environment(acknowledge_staging=True)


def test_explicit_allowlist_and_ack_required(staging_environment):
    with pytest.raises(e2e.E2EError):
        e2e.StagingConfig.from_environment(acknowledge_staging=False)
    config = e2e.StagingConfig.from_environment(acknowledge_staging=True)
    assert config.base_url == 'https://staging.example.invalid'
    assert 'random-fixture-token' not in repr(config)


def test_redirect_refused_without_exposing_url():
    with pytest.raises(e2e.E2EError) as error:
        e2e.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://secret-host.invalid')
    assert 'secret-host' not in str(error.value)


@pytest.fixture
def flow_fixture():
    result = StructuredAnalysisResult(analysis_id='AN-000001', attempt_id='ATT-000001', model_version='fixture-v1',
        status='UNANALYZABLE', findings=[dict(finding_id='F-001', skill='MOVEMENT', status='INSUFFICIENT_EVIDENCE',
            severity=1, confidence=0.0, body_region='LEFT_HAND', start_time_ms=0, end_time_ms=900)],
        overall_score=None, overall_confidence=0.0)
    feedback = generate_feedback(result, feedback_id='FB-fixture')
    submission = dict(attempt_id='ATT-000001', learner_video_ref='learner-videos/e2e-staging/fixture/input.avi',
        reference_profile_ref='reference-profiles/e2e-staging/fixture/profile.json', idempotency_key='fixture-idempotent')
    identity = dict(analysis_id=result.analysis_id, attempt_id=result.attempt_id)
    responses = [(200, {}), (200, {}), (401, {}),
        (202, dict(**identity, state='SUBMITTED')), (202, dict(**identity, state='PROCESSING')),
        (200, dict(**identity, state='PROCESSING')), (200, dict(**identity, state=result.status,
                                                          structured_analysis=result.model_dump(mode='json'))),
        (202, dict(**identity, state='PENDING')), (200, feedback.model_dump(mode='json')),
        (200, feedback.model_dump(mode='json')), (401, {}), (200, b'ugsl_postgres_up 1.0\n'), (403, {})]
    return submission, [(code, body if isinstance(body, bytes) else json.dumps(body).encode()) for code, body in responses]


def test_public_contract_pending_terminal_feedback_and_scope(flow_fixture):
    submission, responses = flow_fixture
    class Client:
        config = SimpleNamespace(scope_denial_token='fixture-scope-token')
        def request(self, *args, **kwargs):
            return responses.pop(0)
    assert e2e.verify_flow(Client(), submission, expected_status='UNANALYZABLE', poll_seconds=0) == {
        'terminal_status': 'UNANALYZABLE', 'feedback': 'validated', 'metrics': 'authenticated',
        'metrics_scope_negative_check': 'passed'}
    assert not responses


@pytest.mark.parametrize('fault', ['identity', 'pinned', 'logical_key', 'metrics_labels', 'feedback_changed', 'unexpected_status'])
def test_contract_and_privacy_failures(flow_fixture, fault):
    submission, responses = flow_fixture
    if fault == 'identity':
        body = json.loads(responses[6][1]); body['attempt_id'] = 'ATT-000002'
        responses[6] = (200, json.dumps(body).encode())
    elif fault == 'pinned': responses[0] = (200, b'ugsl-object-v1.private-marker')
    elif fault == 'logical_key': responses[0] = (200, submission['learner_video_ref'].encode())
    elif fault == 'metrics_labels': responses[11] = (200, b'ugsl_postgres_up 1.0\nbad{analysis_id="x"} 1\n')
    elif fault == 'feedback_changed':
        body = json.loads(responses[9][1]); body['feedback_id'] = 'FB-changed'
        responses[9] = (200, json.dumps(body).encode())
    class Client:
        def request(self, *args, **kwargs): return responses.pop(0)
    with pytest.raises(e2e.E2EError):
        e2e.verify_flow(Client(), submission, expected_status='COMPLETED' if fault == 'unexpected_status' else 'UNANALYZABLE',
                         poll_seconds=0)


@pytest.mark.integration
def test_synthetic_reference_uses_real_packaged_m3_and_profile_loader(tmp_path):
    from ugsl_ai_coach.assets import extractor_identity
    from ugsl_ai_coach.comparison.policy import ComparisonPolicy
    from ugsl_ai_coach.media.profiles import load_profile, serialize_profile
    key = 'reference-profiles/e2e-staging/non-sensitive/profile.json'
    settings = Settings(_env_file=None)
    path, profile = e2e.synthetic_inputs(tmp_path, key, settings)
    assert path.is_file() and path.stat().st_size > 0
    checked = load_profile(serialize_profile(profile), profile_id=key, extractor_identity=extractor_identity(),
                           policy=ComparisonPolicy())
    assert checked == profile
    assert profile.trajectory.observations
    assert all(observation.availability.value == 'MISSING_HAND' for observation in profile.trajectory.observations)
    from ugsl_ai_coach.assets import model_paths
    from ugsl_ai_coach.cv.landmarks import MediaPipeExtractor
    from ugsl_ai_coach.cv.pipeline import extract_video
    from ugsl_ai_coach.comparison.engine import compare_precomputed_movement
    from ugsl_ai_coach.comparison.models import ComparisonRequest
    with MediaPipeExtractor(*model_paths()) as extractor:
        learner = extract_video(path, extractor)
    request = ComparisonRequest(analysis_id='AN-000001', attempt_id='ATT-000001',
                                reference_id=profile.reference_id, correspondence=profile.correspondence)
    result = compare_precomputed_movement(profile.trajectory, profile.context, learner, request, profile.policy).analysis
    assert result.status.value == 'UNANALYZABLE' and result.overall_score is None


@pytest.mark.parametrize('body', [b'random-fixture-token-at-least-32-characters', b'ugsl-object-v1.sensitive'])
def test_client_never_reports_sensitive_response(staging_environment, body):
    config = e2e.StagingConfig.from_environment(acknowledge_staging=True)
    client = e2e.ServiceClient(config)
    class Response:
        code = 200
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return body
    client.opener = SimpleNamespace(open=lambda *args, **kwargs: Response())
    with pytest.raises(e2e.E2EError) as error:
        client.request('GET', '/metrics')
    assert body.decode() not in str(error.value)


@pytest.fixture
def real_synthetic(tmp_path):
    return e2e.synthetic_inputs(tmp_path, 'reference-profiles/e2e-staging/source/profile.json', Settings(_env_file=None))


def test_operator_uploads_isolated_keys_and_prints_summary_only(staging_environment, real_synthetic, monkeypatch, capsys):
    from contextlib import nullcontext
    import ugsl_ai_coach.infrastructure.object_store.s3 as s3
    config = e2e.StagingConfig.from_environment(acknowledge_staging=True)
    media, profile = real_synthetic
    # The local real-M3 fixture is reused; no external store/client is constructed.
    monkeypatch.setattr(e2e, 'synthetic_inputs', lambda directory, key, settings:
        (media, profile.model_copy(update={'profile_id': key})))
    uploads, submissions = [], []
    class Store:
        bucket = 'private-test-bucket'
        client = SimpleNamespace(put_object=lambda **kwargs: uploads.append(kwargs))
    monkeypatch.setattr(s3, 'create_s3_media_store', lambda settings: nullcontext(Store()))
    monkeypatch.setattr(e2e, 'ServiceClient', lambda settings: None)
    monkeypatch.setattr(e2e, 'verify_flow', lambda client, submission, **kwargs:
        submissions.append(submission) or {'terminal_status': kwargs['expected_status']})
    e2e.run(config)
    e2e.run(config)
    assert len(uploads) == 4 and len(submissions) == 2
    assert submissions[0]['idempotency_key'] != submissions[1]['idempotency_key']
    for index, upload in enumerate(uploads):
        prefix = 'learner-videos/' if index % 2 == 0 else 'reference-profiles/'
        assert upload['Key'].startswith(prefix + 'e2e-staging/')
        assert len(upload['Metadata']['ugsl-e2e-run']) == 32
        assert 'ACL' not in upload
    output = capsys.readouterr().out
    assert 'synthetic_blank_real_m3' in output and 'UNANALYZABLE' in output
    assert all(upload['Key'] not in output for upload in uploads)
    assert config.token not in output


def test_supplied_profile_model_mismatch_fails_before_upload(staging_environment, real_synthetic, monkeypatch, tmp_path):
    from ugsl_ai_coach.media.profiles import serialize_profile
    from ugsl_ai_coach.media.ports import IncompatibleReferenceProfile
    import ugsl_ai_coach.infrastructure.object_store.s3 as s3
    config = e2e.StagingConfig.from_environment(acknowledge_staging=True)
    media, profile = real_synthetic
    invalid = tmp_path / 'incompatible.json'
    invalid.write_bytes(serialize_profile(profile.model_copy(update={'extractor_identity': 'wrong-model-version'})))
    monkeypatch.setattr(s3, 'create_s3_media_store', lambda settings: pytest.fail('Must not upload incompatible inputs'))
    with pytest.raises(IncompatibleReferenceProfile):
        e2e.run(config, learner_video=str(media), reference_profile=str(invalid), expected_status='UNANALYZABLE')
