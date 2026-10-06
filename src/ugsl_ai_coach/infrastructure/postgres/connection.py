"""Short-lived context-managed connections, without a pool or import-time I/O."""

from collections.abc import Callable
from contextlib import AbstractContextManager

import psycopg
from psycopg.rows import dict_row

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.integration.errors import AdapterContractError

ConnectionFactory = Callable[[], AbstractContextManager[psycopg.Connection]]


class DatabaseConfigurationError(AdapterContractError):
    """PostgreSQL was explicitly requested without connection configuration."""


def connection_factory(settings: Settings) -> ConnectionFactory:
    secret = settings.database_url
    if secret is None or not secret.get_secret_value().strip():
        raise DatabaseConfigurationError("UGSL_DATABASE_URL is required for PostgreSQL runtime")
    # Keep the secret in the closure; never include it in repr/log output.
    def connect():
        return psycopg.connect(secret.get_secret_value(), row_factory=dict_row, connect_timeout=10)
    return connect
