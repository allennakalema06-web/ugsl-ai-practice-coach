"""Explicit operator-run staging probe. Normal tests never execute live requests."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from secrets import token_hex
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from ugsl_ai_coach.core.config import Settings


class E2EError(ValueError):
    pass


@dataclass(frozen=True, repr=False)
class StagingConfig:
    base_url: str
    token: str
    settings: Settings
    scope_denial_token: str | None = None

    @classmethod
    def from_environment(cls, *, acknowledge_staging: bool):
        if not acknowledge_staging or os.environ.get('UGSL_E2E_ENVIRONMENT') != 'staging':
            raise E2EError('Explicit staging acknowledgement is required')
        base = os.environ.get('UGSL_E2E_BASE_URL', '').rstrip('/')
        allowed = os.environ.get('UGSL_E2E_ALLOWED_HOST', '')
        parsed = urlsplit(base)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.hostname != allowed
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment
                or any(c.isspace() for c in base) or re.search(r'(^|[.-])prod(uction)?([.-]|$)', allowed)):
            raise E2EError('An explicitly allowlisted HTTPS staging host is required')
        token = os.environ.get('UGSL_E2E_SERVICE_TOKEN', '')
        settings = Settings(_env_file=None, environment='production', service_token=token)
        from ugsl_ai_coach.api.auth import StaticTokenAuthenticator
        StaticTokenAuthenticator(settings)
        from ugsl_ai_coach.infrastructure.object_store.s3 import S3MediaStore
        S3MediaStore(None, settings)
        endpoint = settings.object_store_endpoint_url
        if endpoint:
            store_url = urlsplit(endpoint)
            if (store_url.scheme != 'https' or not store_url.hostname or store_url.username or store_url.password
                    or store_url.query or store_url.fragment):
                raise E2EError('Staging storage requires a configured HTTPS endpoint')
        return cls(base, token, settings, os.environ.get('UGSL_E2E_NO_METRICS_SCOPE_TOKEN') or None)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise E2EError('Redirects are refused to protect service credentials')


class ServiceClient:
    def __init__(self, config):
        self.config = config
        self.opener = build_opener(NoRedirect())

    def request(self, method, path, payload=None, *, authenticated=True, token=None):
        headers = {'Content-Type': 'application/json'}
        if authenticated:
            headers['Authorization'] = 'Bearer ' + (token or self.config.token)
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        request = Request(self.config.base_url + path, data=data, headers=headers, method=method)
        try:
            response = self.opener.open(request, timeout=30)
        except HTTPError as error:
            response = error
        with response:
            body = response.read(10 * 1024 * 1024 + 1)
            if len(body) > 10 * 1024 * 1024:
                raise E2EError('Response exceeds verification limits')
            # Never print upstream responses, even if a deployment leaks secrets.
            if (any(secret.encode() in body for secret in (self.config.token, self.config.scope_denial_token) if secret)
                    or b'ugsl-object-v1.' in body):
                raise E2EError('Response privacy check failed')
            return response.code, body


def require(condition, message):
    if not condition:
        raise E2EError(message)


def synthetic_inputs(directory, profile_key, settings):
    """Real M3 extraction of blank synthetic frames; gaps honestly cause abstention."""
    import cv2
    import numpy as np
    from ugsl_ai_coach.assets import model_paths
    from ugsl_ai_coach.cv.landmarks import MediaPipeExtractor
    from ugsl_ai_coach.cv.pipeline import extract_video
    from ugsl_ai_coach.comparison.models import HandCorrespondence
    from ugsl_ai_coach.media.profiles import build_reference_profile
    hand, pose = model_paths()
    hand = Path(settings.hand_model_path) if settings.hand_model_path else hand
    pose = Path(settings.pose_model_path) if settings.pose_model_path else pose
    path = directory / 'input.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (64, 64))
    require(writer.isOpened(), 'Synthetic media encoder unavailable')
    try:
        for _ in range(10): writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    with MediaPipeExtractor(hand, pose) as extractor:
        extraction = extract_video(path, extractor)
    identity = 'mediapipe-tasks-1.0.1:' + sha256(hand.read_bytes()).hexdigest() + ':' + sha256(pose.read_bytes()).hexdigest()
    profile = build_reference_profile(extraction, profile_id=profile_key, reference_id='synthetic-staging-smoke',
        extractor_identity=identity, correspondence=HandCorrespondence(reference_label='Left', learner_label='Left',
            body_region='LEFT_HAND', establishment_basis='Synthetic no-hand infrastructure smoke; no anatomical claim'))
    return path, profile


def configured_identity(settings):
    from ugsl_ai_coach.assets import model_paths
    hand, pose = model_paths()
    hand = Path(settings.hand_model_path) if settings.hand_model_path else hand
    pose = Path(settings.pose_model_path) if settings.pose_model_path else pose
    return 'mediapipe-tasks-1.0.1:' + sha256(hand.read_bytes()).hexdigest() + ':' + sha256(pose.read_bytes()).hexdigest()


def verify_flow(client, submission, *, expected_status, timeout_seconds=900, poll_seconds=2):
    from ugsl_ai_coach.api.models import AnalysisAccepted, AnalysisResponse, FeedbackPending
    from ugsl_ai_coach.coaching.models import CoachingFeedback
    from ugsl_ai_coach.coaching.grounding import ground_analysis
    from ugsl_ai_coach.coaching.validation import validate_feedback
    from ugsl_ai_coach.integration.models import AnalysisSubmission
    submission = AnalysisSubmission.model_validate(submission)
    def call(method, path, payload=None, **kwargs):
        code, body = client.request(method, path, payload, **kwargs)
        require(all(key.encode() not in body for key in (submission.learner_video_ref, submission.reference_profile_ref)),
                'Object references must not be exposed')
        require(b'ugsl-object-v1.' not in body, 'Pinned references must not be exposed')
        return code, body
    for path in ('/api/v1/health', '/api/v1/health/ready'):
        require(call('GET', path, authenticated=False)[0] == 200, 'Health/readiness check failed')
    require(call('POST', '/api/v1/analyses', submission.model_dump(), authenticated=False)[0] == 401,
            'Submission must require service authentication')
    code, body = call('POST', '/api/v1/analyses', submission.model_dump())
    require(code == 202, 'Authorized submission was not accepted')
    accepted = AnalysisAccepted.model_validate_json(body)
    require(accepted.attempt_id == submission.attempt_id and accepted.state.value == 'SUBMITTED',
            'Accepted attempt traceability or initial state mismatch')
    code, repeated = call('POST', '/api/v1/analyses', submission.model_dump())
    repeat = AnalysisAccepted.model_validate_json(repeated) if code == 202 else None
    require(repeat is not None and repeat.analysis_id == accepted.analysis_id and repeat.attempt_id == submission.attempt_id,
            'Idempotent retry changed analysis identity')
    deadline = monotonic() + timeout_seconds
    path = '/api/v1/analyses/' + accepted.analysis_id
    while True:
        code, body = call('GET', path)
        require(code == 200, 'Analysis polling failed')
        status = AnalysisResponse.model_validate_json(body)
        require(status.analysis_id == accepted.analysis_id and status.attempt_id == submission.attempt_id,
                'Analysis traceability mismatch')
        if status.state.is_terminal:
            result = status.structured_analysis
            require(result is not None and result.status.value == status.state.value
                and result.analysis_id == accepted.analysis_id and result.attempt_id == submission.attempt_id,
                'Terminal M2 evidence mismatch')
            require(status.state.value == expected_status, 'Terminal result differs from declared expected path')
            break
        require(status.state.value in ('SUBMITTED', 'PROCESSING') and status.structured_analysis is None,
                'Nonterminal analysis contract mismatch')
        require(monotonic() < deadline, 'Analysis verification timed out')
        sleep(poll_seconds)
    while True:
        code, body = call('GET', path + '/feedback')
        if code == 200:
            feedback = CoachingFeedback.model_validate_json(body)
            require(feedback.analysis_id == accepted.analysis_id and feedback.attempt_id == submission.attempt_id,
                    'Feedback traceability mismatch')
            validate_feedback(feedback, ground_analysis(result, feedback.feedback_id),
                              provider_type=feedback.provider_metadata.provider_type)
            again_code, again_body = call('GET', path + '/feedback')
            require(again_code == 200 and CoachingFeedback.model_validate_json(again_body) == feedback,
                    'Historical feedback changed on reread')
            break
        require(code == 202, 'Feedback polling failed')
        pending = FeedbackPending.model_validate_json(body)
        require(pending.analysis_id == accepted.analysis_id and pending.attempt_id == submission.attempt_id,
                'Pending feedback traceability mismatch')
        require(monotonic() < deadline, 'Coaching verification timed out')
        sleep(poll_seconds)
    require(call('GET', '/metrics', authenticated=False)[0] == 401, 'Metrics must require authentication')
    code, metrics = call('GET', '/metrics')
    require(code == 200 and b'ugsl_postgres_up 1.0' in metrics, 'Authenticated healthy metrics unavailable')
    require(not re.search(rb'(analysis_id|attempt_id|feedback_id|request_id|media_key|token)\s*=', metrics),
            'High-cardinality metric labels detected')
    scope_check = 'not_configured'
    negative_token = getattr(getattr(client, 'config', None), 'scope_denial_token', None)
    if negative_token:
        require(call('GET', '/metrics', token=negative_token)[0] == 403, 'Metrics scope denial failed')
        scope_check = 'passed'
    return {'terminal_status': result.status.value, 'feedback': 'validated', 'metrics': 'authenticated',
            'metrics_scope_negative_check': scope_check}


def run(config, *, learner_video=None, reference_profile=None, expected_status=None, timeout_seconds=900):
    from ugsl_ai_coach.infrastructure.object_store.s3 import create_s3_media_store, VIDEO_TYPES
    from ugsl_ai_coach.media.profiles import ReferenceProfile, serialize_profile, load_profile
    from ugsl_ai_coach.comparison.policy import ComparisonPolicy
    from ugsl_ai_coach.media.references import MediaReferencePolicy
    require(1 <= timeout_seconds <= 3600, 'Verification timeout must be bounded')
    require(bool(learner_video) == bool(reference_profile), 'Supply both test media and its reference profile')
    run_id = token_hex(16)
    policy = MediaReferencePolicy(config.settings)
    learner_key = policy.learner(config.settings.learner_video_prefix + 'e2e-staging/' + run_id + '/input' +
        (Path(learner_video).suffix.lower() if learner_video else '.avi'))
    reference_key = policy.reference(config.settings.reference_profile_prefix + 'e2e-staging/' + run_id + '/profile.json')
    with TemporaryDirectory(prefix='ugsl-staging-e2e-') as local:
        if learner_video:
            require(expected_status in ('COMPLETED', 'UNANALYZABLE', 'FAILED'), 'Declare expected terminal path for supplied media')
            path = Path(learner_video)
            require(Path(reference_profile).stat().st_size <= config.settings.max_reference_bytes, 'Profile exceeds limits')
            original = Path(reference_profile).read_bytes()
            profile = ReferenceProfile.model_validate_json(original)
            load_profile(original, profile_id=profile.profile_id,
                         extractor_identity=configured_identity(config.settings), policy=ComparisonPolicy())
            profile = ReferenceProfile.model_validate({**profile.model_dump(), 'profile_id': reference_key})
        else:
            require(expected_status in (None, 'UNANALYZABLE'), 'Synthetic blank media must honestly abstain')
            expected_status = 'UNANALYZABLE'
            path, profile = synthetic_inputs(Path(local), reference_key, config.settings)
        require(path.suffix.lower() in VIDEO_TYPES and 0 < path.stat().st_size <= config.settings.max_video_bytes,
                'Test media format or size is unsupported')
        encoded = serialize_profile(profile)
        # The same production loader verifies schema/model/policy compatibility.
        identity = configured_identity(config.settings)
        load_profile(encoded, profile_id=reference_key, extractor_identity=identity, policy=ComparisonPolicy())
        require(len(encoded) <= config.settings.max_reference_bytes, 'Reference exceeds limits')
        with create_s3_media_store(config.settings) as store:
            with path.open('rb') as media:
                store.client.put_object(Bucket=store.bucket, Key=learner_key, Body=media,
                    ContentType=VIDEO_TYPES[path.suffix.lower()], Metadata={'ugsl-e2e-run': run_id})
            store.client.put_object(Bucket=store.bucket, Key=reference_key, Body=encoded,
                ContentType='application/json', Metadata={'ugsl-e2e-run': run_id})
        submission = dict(attempt_id=f'ATT-{int(run_id, 16):06d}', learner_video_ref=learner_key,
                          reference_profile_ref=reference_key, idempotency_key='staging-e2e-' + run_id)
        report = verify_flow(ServiceClient(config), submission, expected_status=expected_status,
                             timeout_seconds=timeout_seconds)
        report['input_path'] = 'operator_supplied_test_media' if learner_video else 'synthetic_blank_real_m3'
        print(json.dumps(report))  # No keys, identifiers, payloads or credentials.


def main(argv=None):
    parser = argparse.ArgumentParser(description='Operator-only staging E2E; uploads isolated non-sensitive test objects')
    parser.add_argument('--acknowledge-staging-uploads', action='store_true')
    parser.add_argument('--learner-video')
    parser.add_argument('--reference-profile')
    parser.add_argument('--expected-status', choices=['COMPLETED', 'UNANALYZABLE', 'FAILED'])
    parser.add_argument('--timeout-seconds', type=int, default=900)
    args = parser.parse_args(argv)
    try:
        config = StagingConfig.from_environment(acknowledge_staging=args.acknowledge_staging_uploads)
        run(config, learner_video=args.learner_video, reference_profile=args.reference_profile,
            expected_status=args.expected_status, timeout_seconds=args.timeout_seconds)
        return 0
    except Exception:
        print('Staging E2E failed: configuration, contract or infrastructure verification failed')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
