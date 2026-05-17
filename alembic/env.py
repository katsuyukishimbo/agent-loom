"""Alembic environment.

Reads DATABASE_URL from .env or the environment. We deliberately do NOT use
SQLAlchemy MetaData autogenerate here — the schema lives in the migration
file as plain SQL so reviewers see the exact DDL (including the IVFFlat index
options pgvector needs) without crawling ORM models.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# Load .env if present, so `alembic upgrade head` works without manual export.
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()
except ImportError:  # pragma: no cover — python-dotenv is optional.
    pass

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Read DATABASE_URL at runtime so dev/CI can override per-process.
database_url = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://agent_loom:agent_loom@localhost:5434/agent_loom",
)
config.set_main_option("sqlalchemy.url", database_url)

# Phase 1b ships only one table whose schema is hand-written in the migration
# file. No declarative metadata to autogenerate against.
target_metadata = None


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
