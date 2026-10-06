"""Canonical non-URL object identity, stored in existing opaque string fields."""

from base64 import urlsafe_b64encode, b64decode
import re
from typing import Literal, Self

from pydantic import Field, model_validator

from ugsl_ai_coach.domain.analysis import ContractModel
from ugsl_ai_coach.media.references import InvalidMediaReference, MediaReferencePolicy

PREFIX = "ugsl-object-v1."


class PinnedObject(ContractModel):
    kind: Literal["learner", "reference"]
    key: str = Field(strict=True, min_length=1, max_length=1024)
    version_id: str | None = Field(default=None, strict=True, min_length=1, max_length=1024)
    etag: str | None = Field(default=None, strict=True, min_length=1, max_length=1024)

    @model_validator(mode="after")
    def immutable_identity(self) -> Self:
        if self.version_id == "null" or (self.version_id is None and self.etag is None):
            raise ValueError("A non-null immutable object identity is required")
        for value in (self.version_id, self.etag):
            if value is not None and any(ord(c) < 33 or ord(c) > 126 for c in value):
                raise ValueError("Object identities must be printable ASCII without whitespace")
        if self.etag is not None and not re.fullmatch(r'"[!#-~]+"', self.etag):
            raise ValueError("A strong quoted ETag is required")
        return self

    def encode(self) -> str:
        checked = PinnedObject.model_validate(self.model_dump(mode="python"))
        return PREFIX + urlsafe_b64encode(checked.model_dump_json().encode()).decode().rstrip("=")

    def conditions(self) -> dict[str, str]:
        return {"VersionId": self.version_id} if self.version_id else {"IfMatch": self.etag}


def parse_pinned(value: str, kind: str, policy: MediaReferencePolicy) -> PinnedObject:
    try:
        if not isinstance(value, str) or not value.startswith(PREFIX) or len(value) > 8192:
            raise ValueError
        encoded = value[len(PREFIX):]
        data = b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        pinned = PinnedObject.model_validate_json(data)
        if pinned.kind != kind or pinned.encode() != value:
            raise ValueError
        (policy.learner if kind == "learner" else policy.reference)(pinned.key)
        return pinned
    except (ValueError, TypeError):
        raise InvalidMediaReference("A canonical pinned object reference is required") from None
