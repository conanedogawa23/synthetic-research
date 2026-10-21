#!/usr/bin/env bash
# Copy this checkout onto the Sydney host and restart the API.
# The host .env, data, and virtualenv stay where they are.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
target="/opt/winimy/synthetic-research"
python="${target}/.venv/bin/python"

if [[ ! -x "$python" ]]; then
  echo "API virtualenv is missing at ${python}" >&2
  exit 1
fi

PYTHONPATH="$root" "$python" -m unittest discover -s "$root/tests" -v
rsync -a --delete \
  --exclude .env \
  --exclude .venv \
  --exclude data \
  --exclude .git \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  "${root}/" "${target}/"
sudo systemctl restart winimy-api
sudo systemctl is-active winimy-api
