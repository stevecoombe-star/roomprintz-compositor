#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source ".env"
  set +a
fi

if [[ -f ".env.local" ]]; then
  set -a
  # shellcheck source=/dev/null
  source ".env.local"
  set +a
fi

# Sensible defaults for local dev if not explicitly set in env files.
export DEBUG_ROOMPRINTZ_PROMPT="${DEBUG_ROOMPRINTZ_PROMPT:-1}"
export DEBUG_ROOMPRINTZ_STAGE3_PROMPT="${DEBUG_ROOMPRINTZ_STAGE3_PROMPT:-0}"
export VIBODE_LOG_PROMPTS="${VIBODE_LOG_PROMPTS:-0}"

echo "-------------------------------------------------------"
echo "Vibode Compositor Dev Server"
echo "DEBUG_ROOMPRINTZ_PROMPT=${DEBUG_ROOMPRINTZ_PROMPT}"
echo "DEBUG_ROOMPRINTZ_STAGE3_PROMPT=${DEBUG_ROOMPRINTZ_STAGE3_PROMPT}"
echo "VIBODE_LOG_PROMPTS=${VIBODE_LOG_PROMPTS}"
echo "VIBODE_DUMP_ANNOTATED_IMAGE=${VIBODE_DUMP_ANNOTATED_IMAGE:-0}"
echo "VIBODE_DEBUG_DIR=${VIBODE_DEBUG_DIR:-}"
echo "VIBODE_STRICT=${VIBODE_STRICT:-0}"
echo "-------------------------------------------------------"

exec uvicorn main:app --reload --port 8000
