"""Role-specific configuration validation, without connections or secret output."""

import os
from pathlib import Path
from urllib.parse import urlsplit

from psycopg.conninfo import conninfo_to_dict

from ugsl_ai_coach.core.config import Settings

ROLES = ('api', 'analysis-worker', 'coaching-worker', 'migrate')


class StartupConfigurationError(ValueError):
    pass


def api_port() -> int:
    value = os.environ.get('PORT', '8000')
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 65535:
        raise StartupConfigurationError('PORT must be a valid listening port')
    return int(value)


def validate_config(role: str, settings: Settings) -> Settings:
    if role not in ROLES:
        raise StartupConfigurationError('Unknown runtime role')
    try:
        from ugsl_ai_coach.infrastructure.postgres.connection import connection_factory
        connection_factory(settings)  # Validates presence only; no connection.
        conninfo_to_dict(settings.database_url.get_secret_value())
        if role in ('api', 'analysis-worker'):
            from ugsl_ai_coach.infrastructure.object_store.s3 import S3MediaStore
            S3MediaStore(None, settings)
            endpoint = settings.object_store_endpoint_url
            if endpoint:
                parsed = urlsplit(endpoint)
                if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password
                        or parsed.query or parsed.fragment or any(c.isspace() for c in endpoint)
                        or (settings.environment == 'production' and parsed.scheme != 'https')):
                    raise ValueError('Unsafe storage endpoint')
        if role == 'api':
            from ugsl_ai_coach.api.auth import StaticTokenAuthenticator
            StaticTokenAuthenticator(settings)
            api_port()
        if role == 'analysis-worker':
            from ugsl_ai_coach.assets import model_paths
            hand, pose = model_paths()
            # Explicit custom assets remain supported; never silently replace
            # an operator-provided model. Production profiles must match hashes.
            settings = settings.model_copy(update={
                'hand_model_path': settings.hand_model_path or str(hand),
                'pose_model_path': settings.pose_model_path or str(pose),
            })
            if not all(Path(p).is_file() for p in (settings.hand_model_path, settings.pose_model_path)):
                raise ValueError('Local model assets unavailable')
    except Exception:
        raise StartupConfigurationError(f'Required configuration invalid for {role}') from None
    return settings
