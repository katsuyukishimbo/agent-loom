"""Phase 1b — pgvector-backed EpisodicStore.

Drop-in implementation of the `EpisodicStore` protocol from `store.py`. The
`MemoryHub` accepts either store via constructor injection; no other module
changes.

Design choices worth noting:

1. **Two-stage retrieval**. The HNSW/IVFFlat index gives us cosine-ranked
   candidates, but R × I × R wants recency and importance multiplied in. We
   pull `top_k * over_fetch` rows from the ANN index, then re-rank in Python
   using `rir_score`. `over_fetch=3` is enough that the recency-weighted top
   rarely lives below the cosine cut-off.

2. **`SET LOCAL ivfflat.probes`**. With ~1k rows the default `probes=1` misses
   relevant vectors; bumping to 10 per-transaction restores recall without
   raising the floor for unrelated sessions. The film_benchmark_mvp project
   ran into the same gotcha.

3. **References-count update in the same transaction**. We bump
   `references_count` and `last_referenced_at` for every returned episode in
   one `UPDATE ... WHERE id IN (...)` call. Doing it per-row would multiply
   round-trips by `top_k`.

4. **Async via psycopg3**. `psycopg.AsyncConnection` keeps the existing async
   `EpisodicStore` shape. We hand-write SQL rather than pulling in SQLAlchemy
   ORM — the table is tiny and the queries are tighter without ORM ceremony.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from agent_loom.memory.store import Episode, rir_score


def _default_database_url() -> str:
    """Resolve DATABASE_URL, falling back to the docker-compose default.

    Why a helper: tests and the benchmark script both need the same default,
    and embedding the literal in three places invites drift.
    """
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://agent_loom:agent_loom@localhost:5434/agent_loom",
    )


def _to_psycopg_url(url: str) -> str:
    """Strip the SQLAlchemy `+psycopg` suffix so raw psycopg can consume it.

    The .env.example uses `postgresql+psycopg://...` because Alembic / future
    SQLAlchemy code wants that form. psycopg itself expects plain
    `postgresql://...`. One translator beats educating every caller.
    """
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


def _vector_literal(vec: list[float]) -> str:
    """Format a float list as the pgvector text literal '[1.0,2.0,...]'.

    Why a string and not psycopg's adapter: we don't want to take a dependency
    on `pgvector.psycopg` for one query; the text form is documented and
    stable.
    """
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class PgvectorEpisodicStore:
    """Postgres + pgvector implementation of the `EpisodicStore` Protocol.

    Construct with `PgvectorEpisodicStore(database_url=...)` or rely on the
    DATABASE_URL env var. Connections are opened lazily per call — Phase 1b
    runs at ~1 store call per agent step so pooling would be premature.
    """

    over_fetch: int = 3
    # `probes` controls how many IVFFlat lists are scanned. Default 1 is
    # built for huge tables; at N <= ~10k it loses recall badly. 100 means
    # "scan every list" given our lists=100 index, which is the safest
    # setting until we have benchmarks proving a lower value still wins
    # for our recall@k target.
    ivfflat_probes: int = 100

    def __init__(self, database_url: str | None = None) -> None:
        self._url = _to_psycopg_url(database_url or _default_database_url())

    # --- helpers ------------------------------------------------------

    async def _connect(self):  # type: ignore[no-untyped-def]
        """Open a fresh async connection.

        Imported lazily so unit tests in fake mode don't pay the psycopg import
        cost. The conftest forces fake mode globally; nothing in fake mode
        should ever reach this code path.
        """
        import psycopg  # local import: see docstring

        return await psycopg.AsyncConnection.connect(self._url, autocommit=False)

    @staticmethod
    def _row_to_episode(row: tuple) -> Episode:
        """Map a SELECT row back to an Episode.

        Column order MUST match every SELECT in this file. Keeping the helper
        single-source-of-truth makes that easy to enforce.
        """
        (
            episode_id,
            content,
            importance,
            references_count,
            created_at,
            last_referenced_at,
            embedding_text,
            metadata,
            source,
        ) = row

        if embedding_text is None:
            embedding: list[float] | None = None
        else:
            # pgvector returns vectors as the string `[1,2,3]`; parse defensively.
            inner = embedding_text.strip().lstrip("[").rstrip("]")
            embedding = [float(x) for x in inner.split(",")] if inner else []

        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        return Episode(
            episode_id=episode_id,
            content=content,
            importance=float(importance),
            references_count=int(references_count),
            created_at=created_at,
            last_referenced_at=last_referenced_at,
            embedding=embedding,
            metadata=metadata or {},
            source=source,
        )

    # --- EpisodicStore protocol --------------------------------------

    async def write(self, episode: Episode) -> Episode:
        """Persist `episode`. ON CONFLICT updates the row (idempotent writes).

        Why ON CONFLICT: tests may write the same episode twice. Without
        UPSERT semantics the second write would raise PK violation and force
        every caller to wrap inserts in try/except.
        """
        emb_text = _vector_literal(episode.embedding) if episode.embedding else None
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO episodes (
                        episode_id, content, importance, references_count,
                        created_at, last_referenced_at, embedding, metadata, source
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s::jsonb, %s)
                    ON CONFLICT (episode_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        importance = EXCLUDED.importance,
                        references_count = EXCLUDED.references_count,
                        last_referenced_at = EXCLUDED.last_referenced_at,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        source = EXCLUDED.source
                    """,
                    (
                        str(episode.episode_id),
                        episode.content,
                        episode.importance,
                        episode.references_count,
                        episode.created_at,
                        episode.last_referenced_at,
                        emb_text,
                        json.dumps(episode.metadata),
                        episode.source,
                    ),
                )
            await conn.commit()
        finally:
            await conn.close()
        return episode

    async def recall(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        now: datetime | None = None,
    ) -> list[Episode]:
        """Top-K by R × I × R.

        Two-phase: ANN index narrows to candidates → Python re-ranks with
        recency × importance × relevance. The references_count / timestamp
        update for every winning episode happens in the same transaction.
        """
        now = now or datetime.utcnow()
        emb_text = _vector_literal(query_embedding)

        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                # Bump probes for this transaction only (small dataset fix).
                await cur.execute(f"SET LOCAL ivfflat.probes = {self.ivfflat_probes}")

                # ANN narrow. Cosine distance via `<=>` (vector_cosine_ops).
                # We over-fetch because R × I × R may re-rank an item that was
                # 6th by relevance up to 1st once recency/importance applies.
                fetch_n = max(top_k * self.over_fetch, top_k)
                await cur.execute(
                    """
                    SELECT
                        episode_id, content, importance, references_count,
                        created_at, last_referenced_at, embedding::text, metadata, source
                    FROM episodes
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (emb_text, fetch_n),
                )
                rows = await cur.fetchall()

                candidates = [self._row_to_episode(r) for r in rows]

                # Python-side re-rank — same formula as InMemoryEpisodicStore.
                scored = [
                    (rir_score(ep, query_embedding, now=now), ep) for ep in candidates
                ]
                scored.sort(key=lambda pair: pair[0], reverse=True)
                winners = [ep for _, ep in scored[:top_k]]

                if winners:
                    winner_ids = [str(ep.episode_id) for ep in winners]
                    await cur.execute(
                        """
                        UPDATE episodes
                        SET references_count = references_count + 1,
                            last_referenced_at = %s
                        WHERE episode_id = ANY(%s::uuid[])
                        """,
                        (now, winner_ids),
                    )
                    # Reflect the bump on the returned objects so callers see
                    # the same state the DB now holds.
                    for ep in winners:
                        ep.references_count += 1
                        ep.last_referenced_at = now

            await conn.commit()
        finally:
            await conn.close()

        return winners

    async def list_all(self) -> list[Episode]:
        """Diagnostic / E2E-test helper. Not on the hot path."""
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        episode_id, content, importance, references_count,
                        created_at, last_referenced_at, embedding::text, metadata, source
                    FROM episodes
                    ORDER BY created_at ASC
                    """
                )
                rows = await cur.fetchall()
        finally:
            await conn.close()
        return [self._row_to_episode(r) for r in rows]

    # --- maintenance helpers (not part of the Protocol) --------------

    async def truncate(self) -> None:
        """Empty the table. Tests call this between cases for isolation."""
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute("TRUNCATE TABLE episodes")
            await conn.commit()
        finally:
            await conn.close()

    async def count(self) -> int:
        conn = await self._connect()
        try:
            async with conn.cursor() as cur:
                await cur.execute("SELECT count(*) FROM episodes")
                row = await cur.fetchone()
        finally:
            await conn.close()
        return int(row[0]) if row else 0


# --- env-driven factory --------------------------------------------------


def database_url_or_none() -> str | None:
    """Return DATABASE_URL only if it is set AND looks like a real URL.

    Used by hello_harness to decide between pgvector and in-memory mode at
    startup. We deliberately don't ping the DB here — that would slow every
    cold start. The first store operation will raise loud and clear if the URL
    is unreachable.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    if not url.startswith(("postgresql://", "postgresql+psycopg://")):
        return None
    return url


async def reachable(url: str | None = None, timeout: float = 1.0) -> bool:
    """Quick connect-and-close test. Used by tests to decide skip-vs-run.

    Returns False on any exception — the caller doesn't care WHY the DB is
    unreachable, only that it is.
    """
    target = _to_psycopg_url(url or _default_database_url())
    try:
        import psycopg

        conn = await psycopg.AsyncConnection.connect(target, connect_timeout=int(timeout) or 1)
        await conn.close()
        return True
    except Exception:
        return False


__all__: list[str] = [
    "PgvectorEpisodicStore",
    "database_url_or_none",
    "reachable",
]
