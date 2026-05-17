#!/usr/bin/env bash
# Bring up Postgres + pgvector for local dev.
#
# Why TMPDIR=/tmp: Docker Desktop on macOS occasionally fails to build/pull
# images when TMPDIR points at the per-user /var/folders/... path. Pinning it
# to /tmp avoids "no space left on device" surprises (same workaround the
# film_benchmark_mvp project uses).
set -euo pipefail

cd "$(dirname "$0")/.."

export TMPDIR=/tmp

echo "[dev_db_up] docker compose up -d"
docker compose up -d

echo "[dev_db_up] Waiting for db healthcheck..."
for i in $(seq 1 30); do
  status=$(docker inspect -f '{{.State.Health.Status}}' agent_loom_pg 2>/dev/null || echo "unknown")
  if [ "$status" = "healthy" ]; then
    echo "[dev_db_up] db is healthy."
    break
  fi
  sleep 1
done

if [ "$status" != "healthy" ]; then
  echo "[dev_db_up] WARNING: db did not become healthy within 30s. Check 'docker compose logs db'." >&2
  exit 1
fi

echo "[dev_db_up] Done. DATABASE_URL=postgresql+psycopg://agent_loom:agent_loom@localhost:5434/agent_loom"
echo "[dev_db_up] Next: alembic upgrade head"
