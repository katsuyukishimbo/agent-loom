"""initial episodes table + IVFFlat index

Revision ID: 20260517_0000
Revises:
Create Date: 2026-05-17

Episodes mirror `agent_loom.memory.store.Episode`. The IVFFlat index uses
`vector_cosine_ops` because R × I × R's relevance term is cosine similarity.

Why lists = 100: pgvector's published guidance is `rows / 1000`. For the
1k-row Phase 1b target this puts each list near optimal size. Future tuning
(Phase 2+) can ALTER the index without touching the table.

Why no schema-level UNIQUE on metadata: metadata is jsonb and we want cheap
appends. Phase 2's KG migration will add edge tables and the lookup indexes
they need.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260517_0000"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Enable pgvector. The pgvector/pgvector:pg16 image ships the extension
    #    binary but the schema-level CREATE EXTENSION still has to run once.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Episodes table — column-by-column matches Episode pydantic model so a
    #    SQL row maps 1:1 to a Python instance.
    op.execute(
        """
        CREATE TABLE episodes (
            episode_id          UUID PRIMARY KEY,
            content             TEXT NOT NULL,
            importance          DOUBLE PRECISION NOT NULL CHECK (importance >= 0 AND importance <= 10),
            references_count    INTEGER NOT NULL DEFAULT 0,
            created_at          TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
            last_referenced_at  TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
            embedding           vector(1536),
            metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
            source              TEXT NOT NULL DEFAULT 'executor'
        )
        """
    )

    # 3. IVFFlat ANN index for cosine similarity. `lists = 100` is the pgvector
    #    rule-of-thumb for ~1k rows. The companion `SET LOCAL ivfflat.probes`
    #    used at query time (see store_pg.py) compensates for small datasets
    #    where the index would otherwise underperform.
    op.execute(
        """
        CREATE INDEX episodes_embedding_idx
        ON episodes
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )

    # 4. Secondary index for diagnostic queries / future TTL sweeps.
    op.execute(
        "CREATE INDEX episodes_last_referenced_idx ON episodes (last_referenced_at)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS episodes_last_referenced_idx")
    op.execute("DROP INDEX IF EXISTS episodes_embedding_idx")
    op.execute("DROP TABLE IF EXISTS episodes")
    # We intentionally leave the `vector` extension installed — other apps on
    # the same database may rely on it.
