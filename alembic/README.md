# Alembic migrations

```
./scripts/dev_db_up.sh
alembic upgrade head
```

The `episodes` table mirrors `agent_loom.memory.store.Episode`. We use a
hand-written DDL rather than SQLAlchemy autogenerate so the pgvector-specific
bits (IVFFlat index, `vector_cosine_ops`, `lists = 100`) are visible in diff.
