"""Multi-instance fixed minute window; one row per trusted service/operation."""

from dataclasses import dataclass
from hashlib import sha256
from math import ceil
from typing import Literal
from psycopg import Error, OperationalError


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    retry_after: int


class RateLimited(ValueError):
    def __init__(self, retry_after):
        self.retry_after = max(1, min(60, int(retry_after)))
        super().__init__("Service capacity limit exceeded")


class PostgresRateLimiter:
    def __init__(self, connect):
        self.connect = connect

    def check(self, principal: str, operation: Literal['submit', 'read'], limit: int) -> RateDecision:
        try:
            return self._check(principal, operation, limit)
        except Error:
            # Query/transaction failures never silently disable admission control.
            raise OperationalError('Service capacity persistence unavailable') from None

    def _check(self, principal, operation, limit):
        if not isinstance(principal, str) or not 1 <= len(principal) <= 256:
            raise ValueError("Invalid trusted service principal")
        if operation not in ('submit', 'read') or type(limit) is not int or not 1 <= limit <= 1000000:
            raise ValueError("Invalid capacity bucket")
        digest = sha256(principal.encode('utf-8')).hexdigest()
        with self.connect() as conn:
            inserted = conn.execute("""INSERT INTO service_rate_limits
                (principal_digest, operation, window_start, request_count)
                VALUES (%s, %s, date_trunc('minute', clock_timestamp()), 1)
                ON CONFLICT DO NOTHING RETURNING principal_digest""", (digest, operation)).fetchone()
            row = conn.execute("""SELECT * FROM service_rate_limits
                WHERE principal_digest = %s AND operation = %s FOR UPDATE""", (digest, operation)).fetchone()
            # Clock read after acquiring the lock, rather than before a blocking statement.
            now = conn.execute("SELECT clock_timestamp() AS now").fetchone()['now']
            window = now.replace(second=0, microsecond=0)
            count = row['request_count']
            if inserted is not None or row['window_start'] != window:
                count = 1
            else:
                count = min(limit + 1, count + 1)
            conn.execute("""UPDATE service_rate_limits SET window_start = %s, request_count = %s
                WHERE principal_digest = %s AND operation = %s""", (window, count, digest, operation))
            return RateDecision(count <= limit, max(1, ceil(60 - now.second - now.microsecond / 1000000)))
