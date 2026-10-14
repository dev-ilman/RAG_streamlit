#!/usr/bin/env bash
# End-to-end smoke test against a running stack.
#
#   scripts/smoke.sh [base_url]     # default http://localhost:8000
#
# Drives the same path a user takes: register, ingest, wait for indexing, ask
# a question, and confirm the answer came back with citations attached.
set -euo pipefail

BASE="${1:-http://localhost:8000}"
API="${BASE}/api/v1"
JAR="$(mktemp)"
EMAIL="smoke-$(date +%s)-$RANDOM@example.com"
PASSWORD="smoke-password-123"

trap 'rm -f "$JAR"' EXIT

step() { printf "\n\033[1;36m==> %s\033[0m\n" "$1"; }
fail() { printf "\033[1;31mFAIL: %s\033[0m\n" "$1" >&2; exit 1; }

command -v jq >/dev/null || fail "jq is required"

step "Health"
curl -fsS "${API}/health/ready" | jq -e '.database == "ok"' >/dev/null \
  || fail "database is not ready"

step "Register ${EMAIL}"
TOKEN="$(curl -fsS -c "$JAR" -X POST "${API}/auth/register" \
  -H 'Content-Type: application/json' \
  -d "{\"email\":\"${EMAIL}\",\"password\":\"${PASSWORD}\"}" | jq -r '.access_token')"
[ -n "$TOKEN" ] && [ "$TOKEN" != "null" ] || fail "no access token returned"
AUTH=(-H "Authorization: Bearer ${TOKEN}")

step "Refresh rotates the session"
curl -fsS -b "$JAR" -c "$JAR" -X POST "${API}/auth/refresh" | jq -e '.access_token' >/dev/null \
  || fail "refresh did not return a new access token"

step "Ingest a document"
DOC_ID="$(curl -fsS "${AUTH[@]}" -X POST "${API}/documents/text" \
  -H 'Content-Type: application/json' \
  -d '{"title":"Smoke corpus","content":"Solid electrolyte interphase growth consumes cyclable lithium and is the dominant calendar ageing mechanism in graphite anodes. Lithium plating occurs at low temperature and high charge rates, and it is largely irreversible."}' \
  | jq -r '.id')"
[ -n "$DOC_ID" ] || fail "ingestion did not return a document id"

step "Wait for indexing"
for attempt in $(seq 1 40); do
  STATUS="$(curl -fsS "${AUTH[@]}" "${API}/documents/${DOC_ID}" | jq -r '.status')"
  [ "$STATUS" = "ready" ] && break
  [ "$STATUS" = "failed" ] && fail "ingestion failed: $(curl -fsS "${AUTH[@]}" "${API}/documents/${DOC_ID}" | jq -r '.error')"
  sleep 1
done
[ "$STATUS" = "ready" ] || fail "document stuck in status '${STATUS}'"

step "Retrieval returns the passage"
curl -fsS "${AUTH[@]}" -X POST "${API}/documents/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"what consumes cyclable lithium","top_k":3}' \
  | jq -e '.hits | length > 0' >/dev/null || fail "search returned no hits"

step "Ask a grounded question"
SESSION_ID="$(curl -fsS "${AUTH[@]}" -X POST "${API}/chat/sessions" \
  -H 'Content-Type: application/json' -d '{}' | jq -r '.id')"

ANSWER="$(curl -fsS "${AUTH[@]}" -X POST "${API}/chat/sessions/${SESSION_ID}/ask" \
  -H 'Content-Type: application/json' \
  -d '{"message":"What consumes cyclable lithium?"}')"

echo "$ANSWER" | jq -e '.assistant_message.citations | length > 0' >/dev/null \
  || fail "the answer carried no citations"
echo "$ANSWER" | jq -r '.assistant_message.content' | head -c 300
echo

step "History is persisted"
curl -fsS "${AUTH[@]}" "${API}/chat/sessions/${SESSION_ID}" \
  | jq -e '[.messages[].role] == ["user","assistant"]' >/dev/null \
  || fail "both turns were not persisted"

step "Cleanup"
curl -fsS "${AUTH[@]}" -X DELETE "${API}/documents/${DOC_ID}" -o /dev/null
curl -fsS -b "$JAR" -X POST "${API}/auth/logout" -o /dev/null

printf "\n\033[1;32mSmoke test passed.\033[0m\n"
