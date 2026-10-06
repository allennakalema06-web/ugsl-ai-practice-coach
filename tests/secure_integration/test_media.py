from contextlib import contextmanager
from io import BytesIO

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.infrastructure.object_store.s3 import S3MediaStore, create_s3_media_store
from ugsl_ai_coach.media.ports import (LearnerMediaNotFound, ReferenceProfileNotFound, ObjectStoreUnavailable,
    MediaTooLarge, UnsupportedLearnerVideo, CorruptReferenceProfile)
from ugsl_ai_coach.media.references import MediaReferencePolicy, InvalidMediaReference, MediaConfigurationError
from .conftest import FakeS3


@pytest.mark.parametrize("value", ["http://host/video.avi", "https://host/video.mp4", "ftp://host/x", "file:///C:/x",
    "//host/video", "C:/video.avi", "C:\\video.avi", "/tmp/video.avi", "learner-videos/../x.avi",
    "learner-videos/./x.avi", "learner-videos//x.avi", "learner-videos/%2e%2e/x.avi",
    "learner-videos/x?url=http://host", "learner-videos/x#fragment", "learner-videos/x\n.avi",
    "learner-videos/", "reference-profiles/x.json", "another-bucket/x.avi"])
def test_learner_reference_boundary(settings, value):
    with pytest.raises(InvalidMediaReference):
        MediaReferencePolicy(settings).learner(value)


@pytest.mark.parametrize("value", ["learner-videos/x.avi", "reference-profiles/../x.json", "https://host/x.json",
                                  "reference-profiles/%2fx.json", "/reference-profiles/x.json"])
def test_reference_namespace(settings, value):
    with pytest.raises(InvalidMediaReference):
        MediaReferencePolicy(settings).reference(value)


@pytest.mark.parametrize("learner,reference", [("same/", "same/"), ("a/", "a/b/"), ("bad", "good/"), ("../", "good/")])
def test_namespace_configuration(learner, reference):
    with pytest.raises(MediaConfigurationError):
        MediaReferencePolicy(Settings(_env_file=None, learner_video_prefix=learner, reference_profile_prefix=reference))


def test_custom_namespace_and_fixed_bucket(settings):
    configured = settings.model_copy(update={"learner_video_prefix": "private/learners/", "reference_profile_prefix": "private/experts/"})
    client = FakeS3()
    store = S3MediaStore(client, configured)
    with learner_video(store, "private/learners/synthetic.avi"):
        pass
    assert all(arguments["Bucket"] == "synthetic-private-bucket" for _, arguments in client.calls)


@pytest.mark.parametrize("failure", [False, True])
def test_temp_cleanup_and_revision_pinning(s3, failure):
    client, store = s3
    path = None
    try:
        with learner_video(store, "learner-videos/synthetic.avi") as downloaded:
            path = downloaded
            assert path.read_bytes() == client.data
            assert path.name == "input.avi" and "learner-videos" not in str(path)
            if failure:
                raise RuntimeError("synthetic processor failure")
    except RuntimeError:
        assert failure
    assert path is not None and not path.exists() and not path.parent.exists()
    assert all(body.closed for body in client.bodies)
    assert client.calls[-1][1]["IfMatch"] == '"synthetic-etag"'


def test_object_version_is_pinned(s3):
    client, store = s3
    client.head_override["VersionId"] = "synthetic-version"
    with learner_video(store, "learner-videos/synthetic.avi"):
        pass
    assert client.calls[-1][1]["VersionId"] == "synthetic-version"
    assert "IfMatch" not in client.calls[-1][1]


def test_size_limit_before_download(settings):
    client = FakeS3(b"too large")
    store = S3MediaStore(client, settings.model_copy(update={"max_video_bytes": 3}))
    with pytest.raises(MediaTooLarge):
        with learner_video(store, "learner-videos/synthetic.avi"):
            pass
    assert [name for name, _ in client.calls] == ["head"]


@pytest.mark.parametrize("content_type", ["application/octet-stream", "text/html", "image/png", "video/webm"])
def test_unsupported_type_before_download(settings, content_type):
    client = FakeS3(content_type=content_type)
    with pytest.raises(UnsupportedLearnerVideo):
        with learner_video(S3MediaStore(client, settings), "learner-videos/synthetic.avi"):
            pass
    assert [name for name, _ in client.calls] == ["head"]


@pytest.mark.parametrize("extension", ["webm", "mov", "txt", "exe"])
def test_unsupported_extension_no_storage_request(s3, extension):
    client, store = s3
    with pytest.raises(UnsupportedLearnerVideo):
        with learner_video(store, "learner-videos/x." + extension):
            pass
    assert client.calls == []


@pytest.mark.parametrize("metadata", [{"ContentLength": 0}, {"ContentLength": True},
    {"ContentEncoding": "gzip"}, {"ETag": None}])
def test_invalid_metadata(s3, metadata):
    client, store = s3
    client.head_override.update(metadata)
    with pytest.raises((UnsupportedLearnerVideo, ObjectStoreUnavailable)):
        with learner_video(store, "learner-videos/x.avi"):
            pass
    assert len(client.calls) == 1


def test_changed_get_metadata_closes_stream(s3):
    client, store = s3
    client.get_override["ContentLength"] = 100
    with pytest.raises(UnsupportedLearnerVideo):
        with learner_video(store, "learner-videos/x.avi"):
            pass
    assert client.bodies[0].closed


def test_stream_exceeds_declared_length_closes_and_cleans(settings):
    client = FakeS3(b"longer than declared")
    client.head_override["ContentLength"] = 3
    client.get_override["ContentLength"] = 3
    with pytest.raises(MediaTooLarge):
        with learner_video(S3MediaStore(client, settings), "learner-videos/x.avi"):
            pass
    assert client.bodies[0].closed


def test_truncated_stream(s3):
    client, store = s3
    client.head_override["ContentLength"] = len(client.data) + 1
    client.get_override["ContentLength"] = len(client.data) + 1
    with pytest.raises(ObjectStoreUnavailable):
        with learner_video(store, "learner-videos/x.avi"):
            pass
    assert client.bodies[0].closed


@pytest.mark.parametrize("method,reference,expected", [
    ("video", "learner-videos/x.avi", LearnerMediaNotFound),
    ("profile", "reference-profiles/x.json", ReferenceProfileNotFound),
])
def test_missing_objects_distinguished(settings, method, reference, expected):
    client = FakeS3()
    client.error = ClientError({"Error": {"Code": "NoSuchKey", "Message": "synthetic-sensitive-key"}}, "HeadObject")
    store = S3MediaStore(client, settings)
    with pytest.raises(expected) as caught:
        if method == "video":
            with learner_video(store, reference): pass
        else:
            reference_profile(store, reference)
    assert "synthetic-sensitive-key" not in str(caught.value)


@pytest.mark.parametrize("error", [ClientError({"Error": {"Code": "AccessDenied", "Message": "synthetic-secret"}}, "HeadObject"),
    EndpointConnectionError(endpoint_url="https://synthetic-sensitive-host")])
def test_storage_error_redacted(settings, error):
    client = FakeS3()
    client.error = error
    with pytest.raises(ObjectStoreUnavailable) as caught:
        with learner_video(S3MediaStore(client, settings), "learner-videos/x.avi"): pass
    assert "synthetic-secret" not in str(caught.value) and "synthetic-sensitive-host" not in str(caught.value)


def test_reference_json_bounded_and_closed(settings):
    client = FakeS3(b'{"synthetic":true}', "application/json")
    assert reference_profile(S3MediaStore(client, settings), "reference-profiles/x.json") == client.data
    assert client.bodies[0].closed
    small = settings.model_copy(update={"max_reference_bytes": 1})
    with pytest.raises(MediaTooLarge):
        reference_profile(S3MediaStore(client, small), "reference-profiles/x.json")


@pytest.mark.parametrize("endpoint", ["file:///tmp", "https://user:synthetic-secret@host", "https://host?bucket=other", "https://host#x"])
def test_endpoint_is_configured_not_request_supplied(settings, endpoint):
    with pytest.raises(MediaConfigurationError):
        create_s3_media_store(settings.model_copy(update={"object_store_endpoint_url": endpoint}))


def test_default_aws_endpoint_and_standard_credential_chain(settings, monkeypatch):
    import boto3
    calls = []
    client = FakeS3()
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: calls.append((args, kwargs)) or client)
    create_s3_media_store(settings)
    arguments = calls[0][1]
    assert arguments["endpoint_url"] is None
    assert "aws_access_key_id" not in arguments and "aws_secret_access_key" not in arguments


def test_botocore_stream_failure_is_safe_and_closed(settings):
    from botocore.response import StreamingBody
    client = FakeS3(b"abc")
    stream = StreamingBody(BytesIO(b"ab"), 3)
    client.get_override["Body"] = stream
    with pytest.raises(ObjectStoreUnavailable):
        with learner_video(S3MediaStore(client, settings), "learner-videos/x.avi"):
            pass
    assert stream._raw_stream.closed


def test_blank_optional_endpoint_uses_native_aws(settings, monkeypatch):
    import boto3
    calls = []
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: calls.append(kwargs) or FakeS3())
    create_s3_media_store(settings.model_copy(update={"object_store_endpoint_url": "", "object_store_region": ""}))
    assert calls[0]["endpoint_url"] is None and calls[0]["region_name"] is None


def learner_video(store, key):
    return store.learner_video(store.pin_learner(key))


def reference_profile(store, key):
    return store.reference_profile(store.pin_reference(key))
