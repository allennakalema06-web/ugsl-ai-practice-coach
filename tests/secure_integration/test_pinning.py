"""Submission-time historical identity; never resolve a replacement at delivery."""

from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from ugsl_ai_coach.api.application import AnalysisApiService
from ugsl_ai_coach.infrastructure.object_store.s3 import S3MediaStore
from ugsl_ai_coach.integration.handoff.service import AnalysisHandoffService
from ugsl_ai_coach.media.pinned import PinnedObject, parse_pinned
from ugsl_ai_coach.media.ports import LearnerMediaNotFound, ReferenceProfileNotFound, ObjectStoreUnavailable
from ugsl_ai_coach.media.references import InvalidMediaReference, MediaReferencePolicy
from ugsl_ai_coach.main import create_app
from integration_handoff.fakes import FakePersistence, FakeIdFactory
from .conftest import SYNTHETIC_TOKEN
from .test_api import Records
from .test_processor import composition


class HistoricalS3:
    def __init__(self, versioned=True):
        self.versioned = versioned
        self.objects, self.latest, self.calls, self.bodies = {}, {}, [], []
    def put(self, key, data, content_type):
        version = str(len(self.objects) + 1)
        item = {"data": data, "ContentType": content_type, "ContentLength": len(data),
                "ETag": '"' + sha256(data).hexdigest() + '"'}
        if self.versioned:
            item["VersionId"] = version
        self.objects[key, version] = item
        self.latest[key] = version
    def close(self): pass
    def selected(self, operation, arguments):
        self.calls.append((operation, arguments))
        key = arguments["Key"]
        version = arguments.get("VersionId", self.latest.get(key))
        item = self.objects.get((key, version))
        if item is None:
            raise ClientError({"Error": {"Code": "NoSuchVersion"}}, operation)
        if arguments.get("IfMatch", item["ETag"]) != item["ETag"]:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, operation)
        return item
    def head_object(self, **arguments):
        return {k: v for k, v in self.selected("head", arguments).items() if k != "data"}
    def get_object(self, **arguments):
        item = self.selected("get", arguments)
        body = BytesIO(item["data"])
        self.bodies.append(body)
        return {**{k: v for k, v in item.items() if k != "data"}, "Body": body}


@pytest.fixture
def pinned_wiring(settings, submission):
    def create(versioned=True):
        objects = HistoricalS3(versioned)
        objects.put(submission.learner_video_ref, b"media A", "video/x-msvideo")
        objects.put(submission.reference_profile_ref, b'{"expert":"A"}', "application/json")
        store = S3MediaStore(objects, settings)
        persistence = FakePersistence()
        service = AnalysisApiService(AnalysisHandoffService(persistence, FakeIdFactory()), Records(),
                                     MediaReferencePolicy(settings), lambda: store)
        app = create_app(settings)
        app.state.analysis_api_service = service
        client = TestClient(app)
        client.headers["Authorization"] = "Bearer " + SYNTHETIC_TOKEN
        return client, service, persistence, store, objects
    return create


@pytest.mark.parametrize("versioned", [True, False])
def test_acceptance_persists_both_canonical_identities(pinned_wiring, submission, settings, versioned):
    client, _, persistence, _, objects = pinned_wiring(versioned)
    response = client.post("/api/v1/analyses", json=submission.model_dump())
    assert response.status_code == 202
    pair = persistence.get_pair(response.json()["analysis_id"])
    for kind, field in [("learner", "learner_video_ref"), ("reference", "reference_profile_ref")]:
        value = getattr(pair.job, field)
        identity = parse_pinned(value, kind, MediaReferencePolicy(settings))
        assert identity.key == getattr(submission, field)
        assert identity.version_id is not None if versioned else identity.version_id is None
        assert identity.etag is not None and getattr(pair.work, field) == value
        assert value not in response.text and identity.key not in response.text
    assert [operation for operation, _ in objects.calls] == ["head", "head"]
    status = client.get("/api/v1/analyses/AN-000001")
    assert "ugsl-object" not in status.text and "synthetic-private-bucket" not in status.text


@pytest.mark.parametrize("kind", ["learner", "reference"])
def test_versioned_overwrite_retrieves_accepted_A_not_B(pinned_wiring, submission, settings, kind):
    client, _, persistence, store, objects = pinned_wiring()
    assert client.post("/api/v1/analyses", json=submission.model_dump()).status_code == 202
    job = persistence.get("AN-000001")
    key = submission.learner_video_ref if kind == "learner" else submission.reference_profile_ref
    value = job.learner_video_ref if kind == "learner" else job.reference_profile_ref
    identity = parse_pinned(value, kind, MediaReferencePolicy(settings))
    objects.put(key, b"replacement B", "video/x-msvideo" if kind == "learner" else "application/json")
    objects.calls.clear()
    if kind == "learner":
        with store.learner_video(value) as path:
            assert path.read_bytes() == b"media A"
        assert not path.exists()
    else:
        assert store.reference_profile(value) == b'{"expert":"A"}'
    assert all(arguments.get("VersionId") == identity.version_id for _, arguments in objects.calls)
    assert all(body.closed for body in objects.bodies)


@pytest.mark.parametrize("kind", ["learner", "reference"])
def test_unversioned_overwrite_refuses_replacement(pinned_wiring, submission, kind):
    client, _, persistence, store, objects = pinned_wiring(False)
    client.post("/api/v1/analyses", json=submission.model_dump())
    job = persistence.get("AN-000001")
    key = submission.learner_video_ref if kind == "learner" else submission.reference_profile_ref
    value = job.learner_video_ref if kind == "learner" else job.reference_profile_ref
    objects.put(key, b"replacement B", "video/x-msvideo" if kind == "learner" else "application/json")
    objects.calls.clear()
    with pytest.raises(ObjectStoreUnavailable):
        if kind == "learner":
            with store.learner_video(value): pytest.fail("Replacement must never reach processing")
        else:
            store.reference_profile(value)
    assert len(objects.calls) == 1 and "IfMatch" in objects.calls[0][1]
    assert objects.bodies == []


@pytest.mark.parametrize("versioned", [True, False])
@pytest.mark.parametrize("changed_kind", ["learner", "reference"])
def test_idempotency_same_identity_then_changed_evidence_409(pinned_wiring, submission, versioned, changed_kind):
    client, _, persistence, _, objects = pinned_wiring(versioned)
    first = client.post("/api/v1/analyses", json=submission.model_dump())
    before = persistence.store.snapshot()
    assert client.post("/api/v1/analyses", json=submission.model_dump()).json() == first.json()
    key = submission.learner_video_ref if changed_kind == "learner" else submission.reference_profile_ref
    objects.put(key, b"changed evidence", "video/x-msvideo" if changed_kind == "learner" else "application/json")
    response = client.post("/api/v1/analyses", json=submission.model_dump())
    assert response.status_code == 409 and response.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    assert persistence.store.snapshot() == before


@pytest.mark.parametrize("kind,expected", [("learner", LearnerMediaNotFound), ("reference", ReferenceProfileNotFound)])
def test_missing_historical_version_never_falls_back(pinned_wiring, submission, settings, kind, expected):
    client, _, persistence, store, objects = pinned_wiring()
    client.post("/api/v1/analyses", json=submission.model_dump())
    job = persistence.get("AN-000001")
    value = job.learner_video_ref if kind == "learner" else job.reference_profile_ref
    identity = parse_pinned(value, kind, MediaReferencePolicy(settings))
    objects.put(identity.key, b"replacement B", "video/x-msvideo" if kind == "learner" else "application/json")
    del objects.objects[identity.key, identity.version_id]
    objects.calls.clear()
    with pytest.raises(expected):
        if kind == "learner":
            with store.learner_video(value): pytest.fail("Historical version is absent")
        else: store.reference_profile(value)
    assert len(objects.calls) == 1 and objects.calls[0][1]["VersionId"] == identity.version_id


def test_unpinned_worker_reference_rejected_without_storage(pinned_wiring, submission):
    _, _, _, store, objects = pinned_wiring()
    with pytest.raises(InvalidMediaReference):
        with store.learner_video(submission.learner_video_ref): pass
    with pytest.raises(InvalidMediaReference):
        store.reference_profile(submission.reference_profile_ref)
    assert objects.calls == []


@pytest.mark.parametrize("value", ["ugsl-object-v1.invalid", "http://host/file", "learner-videos/x.avi"])
def test_invalid_canonical_wire_rejected(settings, value):
    with pytest.raises(InvalidMediaReference):
        parse_pinned(value, "learner", MediaReferencePolicy(settings))


def test_pinned_namespace_cannot_bypass_security(settings):
    policy = MediaReferencePolicy(settings)
    for kind, key in [("learner", "reference-profiles/x.json"), ("reference", "learner-videos/x.avi"),
                      ("learner", "learner-videos/../x.avi"), ("learner", "http://host/x.avi")]:
        value = PinnedObject(kind=kind, key=key, etag='"A"').encode()
        with pytest.raises(InvalidMediaReference):
            parse_pinned(value, kind, policy)


@pytest.mark.parametrize("versioned", [True, False])
def test_actual_worker_uses_accepted_evidence_or_refuses_after_overwrite(composition, settings, submission, versioned):
    from ugsl_ai_coach.worker.processor import ProductionAnalysisProcessor
    from ugsl_ai_coach.worker.runtime import AnalysisWorker
    prior, _, _, original, _, extractor, _ = composition
    objects = HistoricalS3(versioned)
    objects.put(submission.learner_video_ref, original.video.data, "video/x-msvideo")
    objects.put(submission.reference_profile_ref, original.reference.data, "application/json")
    store = S3MediaStore(objects, settings)
    persistence = FakePersistence()
    api = AnalysisApiService(AnalysisHandoffService(persistence, FakeIdFactory()), Records(),
                             MediaReferencePolicy(settings), lambda: store)
    job = api.submit(submission)
    objects.put(submission.learner_video_ref, b"replacement invalid video B", "video/x-msvideo")
    objects.put(submission.reference_profile_ref, b"replacement invalid profile B", "application/json")
    processor = ProductionAnalysisProcessor(store, lambda: extractor,
                                           extractor_identity=prior.extractor_identity, settings=settings)
    class WorkerStore:
        def claim_next(self, *, lease_duration_ms):
            return persistence.claim_next(now_ms=100, lease_duration_ms=lease_duration_ms)
        def begin(self, expected):
            return persistence.begin(expected, now_ms=101)
        def finish(self, expected, result):
            return persistence.finish(expected, result, now_ms=102)
    assert AnalysisWorker(WorkerStore(), processor, settings).run_once() == versioned
    saved = persistence.get_pair(job.analysis_id)
    if versioned:
        assert saved.job.state == "COMPLETED" and saved.job.structured_analysis.overall_score == 1.0
    else:
        assert saved.job.state == "PROCESSING" and saved.job.structured_analysis is None
        assert saved.work.state == "CLAIMED"  # lease recovery, no fabricated FAILED


def test_second_pin_failure_never_accepts_half_submission(pinned_wiring, submission):
    client, _, persistence, _, objects = pinned_wiring()
    del objects.objects[submission.reference_profile_ref, objects.latest[submission.reference_profile_ref]]
    response = client.post("/api/v1/analyses", json=submission.model_dump())
    assert response.status_code == 404 and response.json()["error"]["code"] == "REFERENCE_PROFILE_NOT_FOUND"
    assert persistence.store.pairs == {}


def test_failed_authentication_does_not_resolve_objects(pinned_wiring, submission):
    client, _, persistence, _, objects = pinned_wiring()
    client.headers["Authorization"] = "Bearer invalid"
    assert client.post("/api/v1/analyses", json=submission.model_dump()).status_code == 401
    assert objects.calls == [] and persistence.store.pairs == {}


def test_missing_version_between_head_and_get_does_not_fall_back(pinned_wiring, submission, settings):
    client, _, persistence, store, objects = pinned_wiring()
    client.post("/api/v1/analyses", json=submission.model_dump())
    value = persistence.get("AN-000001").learner_video_ref
    identity = parse_pinned(value, "learner", MediaReferencePolicy(settings))
    original_get = objects.get_object
    def remove_then_get(**arguments):
        del objects.objects[identity.key, identity.version_id]
        return original_get(**arguments)
    objects.get_object = remove_then_get
    objects.put(identity.key, b"replacement B", "video/x-msvideo")
    objects.calls.clear()
    with pytest.raises(LearnerMediaNotFound):
        with store.learner_video(value): pass
    assert [op for op, _ in objects.calls] == ["head", "get"]
    assert all(args["VersionId"] == identity.version_id for _, args in objects.calls)


def test_unversioned_success_uses_accepted_etag_in_get(pinned_wiring, submission, settings):
    client, _, persistence, store, objects = pinned_wiring(False)
    client.post("/api/v1/analyses", json=submission.model_dump())
    value = persistence.get("AN-000001").learner_video_ref
    identity = parse_pinned(value, "learner", MediaReferencePolicy(settings))
    objects.calls.clear()
    with store.learner_video(value) as path:
        assert path.read_bytes() == b"media A"
    assert all(args["IfMatch"] == identity.etag for _, args in objects.calls)


def test_provider_ignoring_get_condition_is_rejected_and_body_closed(pinned_wiring, submission):
    client, _, persistence, store, objects = pinned_wiring(False)
    client.post("/api/v1/analyses", json=submission.model_dump())
    original_get = objects.get_object
    def changed_response(**arguments):
        response = original_get(**arguments)
        response["ETag"] = '"different-identity"'
        return response
    objects.get_object = changed_response
    with pytest.raises(ObjectStoreUnavailable):
        with store.learner_video(persistence.get("AN-000001").learner_video_ref): pass
    assert objects.bodies[-1].closed
