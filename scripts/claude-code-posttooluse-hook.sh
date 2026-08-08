#!/usr/bin/env bash
# Claude Code PostToolUse hook: pipes a tool's real output through squishi
# and replaces it with the compressed result, before it ever enters
# context. Per Anthropic's own docs (code.claude.com/docs/en/costs,
# "Offload processing to hooks and skills"), this is the officially
# recommended shape for reducing context bloat from verbose tool output
# — a mechanical PreToolUse/PostToolUse filter, not a replacement for
# /compact (which is an LLM summarization call over the whole
# conversation, a fundamentally different, more expensive mechanism).
#
# Skips small output outright (cheap length check here, before ever
# spawning squishi) — squishi's own internal thresholds already pass
# small content through unchanged, but there's no reason to pay even a
# process-spawn cost on a two-line `pwd` result.
#
# Requires: squishi and jq on PATH.

set -euo pipefail

MIN_BYTES=2000

input="$(cat)"
tool_output="$(jq -r '.tool_output // empty' <<<"$input")"

if [[ -z "$tool_output" ]] || [[ ${#tool_output} -lt $MIN_BYTES ]]; then
    exit 0
fi

if ! command -v squishi >/dev/null 2>&1; then
    # squishi not installed — don't block the tool result over a missing
    # optional dependency, just pass through unmodified.
    exit 0
fi

compressed="$(squishi <<<"$tool_output" 2>/dev/null)" || exit 0

jq -n --arg output "$compressed" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    updatedToolOutput: $output
  }
}'
