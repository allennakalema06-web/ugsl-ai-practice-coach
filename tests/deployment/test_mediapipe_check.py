import json
from types import SimpleNamespace

import pytest

from ugsl_ai_coach.deployment import mediapipe_check as diagnostic


@pytest.mark.integration
def test_real_packaged_hand_and_pose_diagnostic(capsys):
    assert diagnostic.main([]) == 0
    output = capsys.readouterr().out
    assert 'Initializing: HandLandmarker' in output
    assert 'Initializing: PoseLandmarker' in output
    assert 'MediaPipe: 1.0.1' in output
    assert 'Exception' not in output


def test_exception_chain_preserves_inner_cause():
    try:
        try:
            raise OSError('missing-native-library.so')
        except OSError as inner:
            raise RuntimeError('Initialization failed') from inner
    except RuntimeError as error:
        assert diagnostic.exception_chain(error) == [
            {'exception_class': 'RuntimeError', 'exception_message': 'Initialization failed'},
            {'exception_class': 'OSError', 'exception_message': 'missing-native-library.so'}]


def test_both_children_checked_secrets_excluded_and_output_bounded(monkeypatch, capsys):
    secret = 'test-sensitive-credential-marker'
    monkeypatch.setenv('UGSL_DATABASE_URL', secret)
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', secret)
    calls = []
    def child(command, **kwargs):
        calls.append(command[-1])
        assert set(kwargs['env']).issubset({'PATH', 'SYSTEMROOT', 'WINDIR', 'PYTHONPATH'})
        assert kwargs['capture_output'] is True
        record = [{'exception_class': 'RuntimeError', 'exception_message': 'Native failure ' + secret}]
        return SimpleNamespace(returncode=1, stdout=json.dumps(record), stderr='Discard native output ' + secret)
    monkeypatch.setattr(diagnostic.subprocess, 'run', child)
    assert diagnostic.main([]) == 1
    output = capsys.readouterr().out
    assert calls == ['HandLandmarker', 'PoseLandmarker']
    assert secret not in output and 'Discard native output' not in output
    assert 'RuntimeError' in output and '[redacted]' in output
