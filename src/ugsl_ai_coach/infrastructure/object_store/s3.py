"""Bounded private retrieval, pinned HEAD/GET revision, unpredictable temporary files."""

from contextlib import contextmanager
from pathlib import Path
import re
from urllib.parse import urlsplit

from botocore.exceptions import BotoCoreError, ClientError

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.media.ports import (
    LearnerMediaNotFound, ReferenceProfileNotFound, ObjectStoreUnavailable,
    MediaTooLarge, UnsupportedLearnerVideo, CorruptReferenceProfile,
)
from ugsl_ai_coach.media.references import MediaConfigurationError, MediaReferencePolicy
from ugsl_ai_coach.media.pinned import PinnedObject, parse_pinned
from ugsl_ai_coach.media.temporary import TemporaryMedia

VIDEO_TYPES = {".mp4": "video/mp4", ".avi": "video/x-msvideo"}


class S3MediaStore:
    def __init__(self, client, settings: Settings):
        self.policy = MediaReferencePolicy(settings)
        if not settings.object_store_bucket or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", settings.object_store_bucket):
            raise MediaConfigurationError("A fixed private object-store bucket is required")
        self.client, self.bucket = client, settings.object_store_bucket
        self.max_video_bytes, self.max_reference_bytes = settings.max_video_bytes, settings.max_reference_bytes

    def _call(self, operation, missing, **arguments):
        try:
            return getattr(self.client, operation)(Bucket=self.bucket, **arguments)
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NoSuchVersion", "NotFound"):
                raise missing("The requested private object was not found") from None
            raise ObjectStoreUnavailable("Private object retrieval failed") from None
        except BotoCoreError:
            raise ObjectStoreUnavailable("Private object storage is unavailable") from None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.client.close()

    def _validate_metadata(self, metadata, maximum, content_type, unsupported):
        length = metadata.get("ContentLength")
        if type(length) is not int or length <= 0:
            raise unsupported("Object metadata is invalid")
        if length > maximum:
            raise MediaTooLarge("Object exceeds the configured size limit")
        if metadata.get("ContentType") != content_type:
            raise unsupported("Object content type is unsupported")
        if metadata.get("ContentEncoding") not in (None, "identity"):
            raise unsupported("Encoded objects are unsupported")
        return length

    def _pin(self, key, kind, maximum, content_type, missing, unsupported):
        metadata = self._call("head_object", missing, Key=key)
        self._validate_metadata(metadata, maximum, content_type, unsupported)
        try:
            version = metadata.get("VersionId")
            return PinnedObject(kind=kind, key=key, version_id=None if version in (None, "null") else version,
                                etag=metadata.get("ETag")).encode()
        except ValueError:
            raise ObjectStoreUnavailable("Immutable object identity metadata is required") from None

    def pin_learner(self, reference: str) -> str:
        key = self.policy.learner(reference)
        suffix = Path(key).suffix.lower()
        if suffix not in VIDEO_TYPES:
            raise UnsupportedLearnerVideo("Only MP4 or AVI video containers are accepted")
        return self._pin(key, "learner", self.max_video_bytes, VIDEO_TYPES[suffix], LearnerMediaNotFound, UnsupportedLearnerVideo)

    def pin_reference(self, reference: str) -> str:
        key = self.policy.reference(reference)
        if not key.endswith(".json"):
            raise CorruptReferenceProfile("Reference profiles require versioned JSON")
        return self._pin(key, "reference", self.max_reference_bytes, "application/json", ReferenceProfileNotFound, CorruptReferenceProfile)

    def _verify_identity(self, metadata, pinned):
        if ((pinned.version_id and metadata.get("VersionId") != pinned.version_id)
                or (pinned.etag and metadata.get("ETag") != pinned.etag)):
            from ugsl_ai_coach.operations.events import emit
            emit('storage_identity_violation', error_code='OBJECT_IDENTITY_MISMATCH')
            raise ObjectStoreUnavailable("The accepted object identity is unavailable or changed")

    @contextmanager
    def _body(self, pinned, maximum, content_type, missing, unsupported):
        # Never resolve latest: both HEAD and GET use the identity accepted at POST.
        conditions = pinned.conditions()
        metadata = self._call("head_object", missing, Key=pinned.key, **conditions)
        self._verify_identity(metadata, pinned)
        length = self._validate_metadata(metadata, maximum, content_type, unsupported)
        response = self._call("get_object", missing, Key=pinned.key, **conditions)
        body = response.get("Body")
        if body is None:
            raise ObjectStoreUnavailable("Object retrieval returned no body")
        try:
            self._verify_identity(response, pinned)
            if response.get("ContentLength") != length or response.get("ContentType") != content_type:
                raise unsupported("Object metadata changed during retrieval")
            if response.get("ContentEncoding") not in (None, "identity"):
                raise unsupported("Encoded objects are unsupported")
            yield body, length
        finally:
            body.close()

    def _copy(self, body, length, maximum, write):
        total = 0
        try:
            while True:
                chunk = body.read(min(65536, maximum - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum or total > length:
                    raise MediaTooLarge("Object stream exceeds its allowed size")
                write(chunk)
        except BotoCoreError:
            raise ObjectStoreUnavailable("Private object stream was interrupted") from None
        if total != length:
            raise ObjectStoreUnavailable("Private object stream was incomplete")

    @contextmanager
    def learner_video(self, reference: str):
        pinned = parse_pinned(reference, "learner", self.policy)
        key = pinned.key
        suffix = Path(key).suffix.lower()
        if suffix not in VIDEO_TYPES:
            raise UnsupportedLearnerVideo("Only MP4 or AVI video containers are accepted")
        with self._body(pinned, self.max_video_bytes, VIDEO_TYPES[suffix], LearnerMediaNotFound, UnsupportedLearnerVideo) as (body, length):
            with TemporaryMedia().directory() as directory:
                path = Path(directory) / ("input" + suffix)  # no caller filename/key in local path
                with path.open("xb") as output:
                    self._copy(body, length, self.max_video_bytes, output.write)
                yield path

    def reference_profile(self, reference: str) -> bytes:
        pinned = parse_pinned(reference, "reference", self.policy)
        key = pinned.key
        if not key.endswith(".json"):
            raise CorruptReferenceProfile("Reference profiles require versioned JSON")
        with self._body(pinned, self.max_reference_bytes, "application/json", ReferenceProfileNotFound, CorruptReferenceProfile) as (body, length):
            chunks = []
            self._copy(body, length, self.max_reference_bytes, chunks.append)
            return b"".join(chunks)


def create_s3_media_store(settings: Settings) -> S3MediaStore:
    # Validate configuration before asking Boto's standard provider chain for credentials.
    S3MediaStore(None, settings)
    endpoint = settings.object_store_endpoint_url or None
    if endpoint:
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment or any(c.isspace() for c in endpoint)):
            raise MediaConfigurationError("The configured object-store endpoint is invalid")
    import boto3
    from botocore.config import Config
    try:
        client = boto3.client("s3", region_name=settings.object_store_region or None, endpoint_url=endpoint,
                             config=Config(connect_timeout=10, read_timeout=30,
                                           retries={"mode": "standard", "max_attempts": 3}))
    except BotoCoreError:
        raise ObjectStoreUnavailable("Private object storage could not be initialized") from None
    return S3MediaStore(client, settings)
