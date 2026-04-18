#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

source "$SCRIPT_DIR/load-config.sh"

if [[ -z "${INSTANCE:-}" ]]; then
  INSTANCE_COUNT=$(echo "$ACTIVE_INSTANCES" | wc -w)
  if [[ "$INSTANCE_COUNT" == "1" ]]; then
    export INSTANCE="$ACTIVE_INSTANCES"
    source "$SCRIPT_DIR/load-config.sh"
  else
    echo "Error: active workload has $INSTANCE_COUNT instances ($ACTIVE_INSTANCES). Pass INSTANCE=<name>." >&2
    exit 1
  fi
fi

MODEL="$MODEL_ID"
BASE="http://localhost:$PORT"

# Check dependencies
for cmd in jq curl; do
  if ! command -v "$cmd" &>/dev/null; then
    echo "Error: $cmd is required but not installed."
    exit 1
  fi
done

# Colors
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
GREEN='\033[32m'
RED='\033[31m'
RESET='\033[0m'

# Check if service is running
printf "${DIM}Connecting to ${BASE}...${RESET}"
if ! curl -sf "$BASE/v1/models" &>/dev/null; then
  printf "\r${RED}Error: Service not reachable at ${BASE}${RESET}\n"
  echo "       Start it with: make start"
  exit 1
fi
printf "\r\033[K"

# Header
echo ""
printf "${BOLD}${CYAN}  Smelter Chat${RESET}\n"
printf "${DIM}  Instance: ${INSTANCE_NAME}${RESET}\n"
printf "${DIM}  Model: ${MODEL}${RESET}\n"
printf "${DIM}  Type /clear to reset, /info for stats, exit to quit${RESET}\n"
echo ""

MESSAGES="[]"
TURN=0
TOTAL_PROMPT_TOKENS=0
TOTAL_COMPLETION_TOKENS=0

print_separator() {
  printf "${DIM}"
  printf '%*s\n' "${COLUMNS:-80}" '' | tr ' ' '─'
  printf "${RESET}"
}

while true; do
  printf "${BOLD}${GREEN}  > ${RESET}"
  read -r user_input || break

  # Handle empty input
  if [[ -z "$user_input" ]]; then
    continue
  fi

  # Handle commands
  case "$user_input" in
    exit|quit|/exit|/quit)
      break
      ;;
    /clear)
      MESSAGES="[]"
      TURN=0
      printf "\n${DIM}  Context cleared.${RESET}\n\n"
      continue
      ;;
    /info)
      MSG_COUNT=$(echo "$MESSAGES" | jq 'length')
      printf "\n${DIM}  Messages: ${MSG_COUNT} | Turns: ${TURN}${RESET}\n"
      printf "${DIM}  Prompt tokens: ${TOTAL_PROMPT_TOKENS} | Completion tokens: ${TOTAL_COMPLETION_TOKENS}${RESET}\n\n"
      continue
      ;;
    /*)
      printf "\n${DIM}  Unknown command. Available: /clear, /info, exit${RESET}\n\n"
      continue
      ;;
  esac

  # Append user message
  MESSAGES=$(echo "$MESSAGES" | jq --arg msg "$user_input" '. + [{"role": "user", "content": $msg}]')

  # Show thinking indicator
  printf "${DIM}  ...${RESET}"

  # Send request and measure time
  START_NS=$(date +%s%N)
  RESPONSE=$(curl -sf "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n --arg model "$MODEL" --argjson messages "$MESSAGES" \
      '{model: $model, messages: $messages, stream: false}')" 2>/dev/null) || {
    printf "\r\033[K${RED}  Request failed. Is the service still running?${RESET}\n\n"
    # Remove the failed user message
    MESSAGES=$(echo "$MESSAGES" | jq '.[:-1]')
    continue
  }
  END_NS=$(date +%s%N)

  # Clear thinking indicator
  printf "\r\033[K"

  # Extract response
  ASSISTANT_MSG=$(echo "$RESPONSE" | jq -r '.choices[0].message.content')

  # Print response with wrapping
  echo ""
  echo "$ASSISTANT_MSG" | while IFS= read -r line; do
    printf "  %s\n" "$line"
  done

  # Stats
  COMPLETION_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.completion_tokens // 0')
  PROMPT_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens // 0')
  ELAPSED_S=$(awk "BEGIN {printf \"%.2f\", ($END_NS - $START_NS) / 1000000000}")

  TOTAL_PROMPT_TOKENS=$((TOTAL_PROMPT_TOKENS + PROMPT_TOKENS))
  TOTAL_COMPLETION_TOKENS=$((TOTAL_COMPLETION_TOKENS + COMPLETION_TOKENS))
  TURN=$((TURN + 1))

  echo ""
  if [[ "$COMPLETION_TOKENS" -gt 0 ]]; then
    TPS=$(awk "BEGIN {printf \"%.1f\", $COMPLETION_TOKENS / $ELAPSED_S}")
    printf "${DIM}  ${COMPLETION_TOKENS} tokens | ${TPS} tok/s | ${ELAPSED_S}s${RESET}\n"
  fi
  echo ""

  # Append assistant message to history
  MESSAGES=$(echo "$MESSAGES" | jq --arg msg "$ASSISTANT_MSG" '. + [{"role": "assistant", "content": $msg}]')
done

echo ""
if [[ $TOTAL_COMPLETION_TOKENS -gt 0 ]]; then
  printf "${DIM}  Session: ${TURN} turns, ${TOTAL_COMPLETION_TOKENS} tokens generated${RESET}\n"
fi
printf "${DIM}  Goodbye!${RESET}\n\n"
