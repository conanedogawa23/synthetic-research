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

if ! command -v uvx >/dev/null 2>&1 && [[ ! -x "${HOME}/.local/bin/uvx" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

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
