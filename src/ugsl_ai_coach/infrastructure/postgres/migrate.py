"""Explicit, ordered, checksum-tracked transactional migrations.

Run: python -m ugsl_ai_coach.infrastructure.postgres.migrate
"""

from hashlib import sha256
from importlib.resources import files

from ugsl_ai_coach.core.config import Settings
from ugsl_ai_coach.infrastructure.postgres.connection import ConnectionFactory, connection_factory
from ugsl_ai_coach.infrastructure.postgres.serialization import CorruptPersistence


def migrate(connect: ConnectionFactory) -> tuple[str, ...]:
    migrations = sorted(files(__package__).joinpath("migrations").iterdir(), key=lambda p: p.name)
    applied = []
    # Serialize runners before creating the tracking table, too. Each invocation
    # is one transaction: a failed file rolls back schema AND tracking entries.
    with connect() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(684706300)")
        conn.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            name TEXT PRIMARY KEY, checksum TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp())""")
        known = {r["name"]: r["checksum"] for r in conn.execute(
            "SELECT name, checksum FROM schema_migrations").fetchall()}
        available = {p.name for p in migrations if p.name.endswith(".sql")}
        if set(known) - available:
            raise CorruptPersistence("Database has unknown migration versions")
        for path in migrations:
            if not path.name.endswith(".sql"):
                continue
            script = path.read_text(encoding="utf-8")
            checksum = sha256(script.encode("utf-8")).hexdigest()
            if path.name in known:
                if known[path.name] != checksum:
                    raise CorruptPersistence("An applied migration checksum has changed")
                continue
            # Trusted packaged static SQL, no user values or identifiers substituted.
            conn.execute(script, prepare=False)
            conn.execute("INSERT INTO schema_migrations (name, checksum) VALUES (%s, %s)",
                         (path.name, checksum))
            applied.append(path.name)
    return tuple(applied)


def main() -> None:
    applied = migrate(connection_factory(Settings()))
    print(f"PostgreSQL migrations complete: {len(applied)} applied")


if __name__ == "__main__":
    main()
