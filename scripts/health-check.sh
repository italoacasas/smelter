#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/load-config.sh"

PASS=0
FAIL=0

check() {
  local name="$1" method="$2" url="$3" data="${4:-}"
  printf "%-20s " "$name"

  if [[ "$method" == "GET" ]]; then
    status=$(curl -sf -o /dev/null -w "%{http_code}" "$url" 2>/dev/null) || status="000"
  else
    status=$(curl -sf -o /dev/null -w "%{http_code}" \
      -H "Content-Type: application/json" \
      -d "$data" "$url" 2>/dev/null) || status="000"
  fi

  if [[ "$status" == "200" ]]; then
    echo "OK ($status)"
    PASS=$((PASS + 1))
  else
    echo "FAIL ($status)"
    FAIL=$((FAIL + 1))
  fi
}

TARGET_INSTANCES="${INSTANCE:-$ACTIVE_INSTANCES}"

for instance in $TARGET_INSTANCES; do
  export INSTANCE="$instance"
  source "$SCRIPT_DIR/load-config.sh"
  BASE="http://localhost:$PORT"
  MODEL="$MODEL_ID"

  echo "==> Health check [$INSTANCE_NAME]: $BASE"
  echo ""

  echo "Container / transport"
  check "/health"       GET  "$BASE/health"
  check "/v1/models"    GET  "$BASE/v1/models"

  echo ""
  echo "Model readiness"
  check "/api/tags"     GET  "$BASE/api/tags"
  check "/api/show"     POST "$BASE/api/show"      "{\"model\":\"$MODEL\"}"
  check "/api/generate" POST "$BASE/api/generate"  "{\"model\":\"$MODEL\",\"prompt\":\"ping\",\"stream\":false}"
  check "/v1/chat/completions" POST "$BASE/v1/chat/completions" "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"stream\":false}"

  echo ""
done

unset INSTANCE

echo "Results: $PASS passed, $FAIL failed"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
