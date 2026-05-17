#!/usr/bin/env bash
# Tear down the local Postgres container.
#
# Use -v to drop the named volume too (full reset). Without -v the data
# survives between up/down cycles which is what you usually want for iterating
# on migrations without reseeding.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "${1:-}" = "--purge" ]; then
  echo "[dev_db_down] docker compose down -v (volumes dropped)"
  docker compose down -v
else
  echo "[dev_db_down] docker compose down (volumes preserved; pass --purge to drop)"
  docker compose down
fi
