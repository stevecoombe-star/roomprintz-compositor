#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source ".venv/bin/activate"
fi

# The compositor is the sole authority for server-side research gates.
# Load repository values first, then local overrides, matching scripts/dev.sh.
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

is_truthy() {
  case "${1:-}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn]) return 0 ;;
    *) return 1 ;;
  esac
}

gate_state() {
  if is_truthy "${1:-}"; then
    printf 'enabled'
  else
    printf 'disabled'
  fi
}

LISTEN_HOST="${AFC_SR1_RESEARCH_HOST:-127.0.0.1}"
LISTEN_PORT="${AFC_SR1_RESEARCH_PORT:-8000}"
READER_STATE="$(gate_state "${AFC_SR1_TR2_READER_ENABLED:-}")"
PLACEMENT_STATE="$(gate_state "${AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED:-}")"

echo "-------------------------------------------------------"
echo "AFC-SR1 Certified Research Compositor"
echo "reader gate: ${READER_STATE}"
echo "placement gate: ${PLACEMENT_STATE}"
echo "listen: ${LISTEN_HOST}:${LISTEN_PORT}"
echo "-------------------------------------------------------"

if [[ "${READER_STATE}" != "enabled" ]]; then
  echo "AFC_SR1_TR2_READER_ENABLED must be one of: 1, true, yes, on" >&2
  exit 2
fi
if [[ "${PLACEMENT_STATE}" != "enabled" ]]; then
  echo "AFC_SR1_TS0_CHILD_PLACEMENT_ENABLED must be one of: 1, true, yes, on" >&2
  exit 2
fi

# Certified research launches are intentionally stable and never auto-reload.
exec uvicorn main:app --host "${LISTEN_HOST}" --port "${LISTEN_PORT}"
