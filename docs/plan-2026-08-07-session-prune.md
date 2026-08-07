# Plan: `session_prune` — structural pruning for squishi (Rust)

Resumes the idea already speced in `README.md`'s "Remembered: the
total-recall-kit ruleset" section, pressure-tested by the technical board
(2026-08-07, see `docs/ideation/agent-stack-architecture/`) and ranked
the highest-priority real gap: squishi's existing shape-based compressors
barely touch real session transcripts (measured earlier this session
against an actual 11MB transcript — the `PlainText` path had zero
qualifying blocks; everything reads as code/diff/log-shaped). Session
pruning operates on a transcript's *structure* (which tool ran, on what
path, superseded by what), not any single blob's text shape.

## Goal

A new squishi module that parses a Claude Code session transcript
(JSONL) and reports which tool-call results are prunable under five
structural rules, without mutating anything by default.

## Context — real transcript shape, confirmed by reading this session's
own live transcript file directly, not assumed

- One JSON record per line. Top-level keys include `type`, `message`,
  `uuid`, `timestamp`, `toolUseResult`, `cwd`, `sessionId`.
- `message.content` is a list of blocks when the message carries tool
  activity:
  - `{"type": "tool_use", "id": "toolu_...", "name": "Read", "input": {...}}`
    — `input.file_path` present for `Read`/`Write`/`Edit`; absent for
    other tools (`Bash`, `Grep`, ...).
  - `{"type": "tool_result", "tool_use_id": "toolu_...", "content": "...", "is_error": bool}`
    — appears in a later `user`-role message, `content` is a plain
    string.
- Real values confirmed: `Read` input is `{file_path, limit?}`; `Write`
  input is `{file_path, content}`; `Edit` input is `{file_path,
  old_string, new_string, replace_all?}`.
- Parsing must be defensive: unknown/malformed lines are skipped, never a
  hard error — transcript JSONL shape isn't a versioned contract.

## Non-goals

- No live transcript mutation. Confirmed earlier this session (real
  hooks-reference check): no Claude Code hook can rewrite a past
  transcript entry; `PostToolUse`'s `updatedToolOutput` only applies to
  the *current* tool call. v1 is a CLI tool only — a stats report by
  default, an explicit pruned *copy* via `--write`, never mutating the
  original. An advisory hook layer is a possible later addition, not
  part of this plan.
- Rule 2 ("supersede write by read") ships but flagged **off by
  default** — board consensus (Fowler): real false-positive risk (a
  write's content a later read didn't fully re-verify), ship behind a
  flag until real usage data exists, not blocked entirely.

## Steps

1. **`src/session_prune.rs`** — new module, sibling to `content_detect.rs`.
   - `SessionItem` — a flat, ordered list built from parsing: enough per
     item to run all five rules (line index, tool name, path if any,
     tool_use_id linkage, tool_result content + is_error + byte length).
   - `parse(jsonl: &str) -> Vec<SessionItem>` — one pass, defensive
     (`serde_json::Value`-based, skip anything that doesn't parse or
     doesn't match the known shapes).
   *Check*: parse a real trimmed excerpt of this session's own
   transcript (not synthetic-only) and assert the expected tool calls/
   results are extracted correctly.

2. **Five rule functions, independent, each `&[SessionItem] ->
   Vec<PruneCandidate>`, own unit tests:**
   - `dedupe_latest_read` — an older `Read` of a path is prunable once a
     newer `Read` of the same path exists later.
   - `supersede_write_by_read` (flagged off by default at the CLI layer,
     not in the function itself — the function is pure and always
     available) — a `Write`/`Edit`'s own tool result is prunable once a
     later `Read` of the same path exists.
   - `drop_redundant_errors` — an error tool-result is prunable if an
     identical `(tool name, content)` error already appeared earlier.
   - `prune_old_large_outputs` — tool-result content over `min_bytes`
     is prunable once it's no longer within the last `window` items (a
     recency window over item count, not a wall-clock age cutoff).
   - `collapse_task_launches` — repeated "running in background with
     ID:" tool-results collapse to the latest one.
   *Check*: each rule gets a real-fixture test (trimmed excerpts from
   this session's own transcript, covering a real instance of that
   rule's pattern) plus a negative test (content that must NOT be
   flagged, e.g. two different paths' reads never collide).

3. **CLI wiring** (`src/main.rs`): `--session-prune <path>` (a flag
   taking a transcript path, following the same "flag not subcommand"
   precedent `--doctor` established — squishi's `Cli` has no `Commands`
   enum). `--include-rule-2` opts into the flagged-off rule. `--json`
   (existing flag) for structured output; default is a human-readable
   stats report (rule name, candidate count, total prunable bytes).
   `--write <out-path>` emits a pruned copy of the transcript (drops the
   flagged tool-result contents, replaces with a short marker, same
   spirit as `base64_strip`'s marker) — never mutates the input.
   *Check*: run against this session's own real (large) transcript file,
   confirm real, sane stats; confirm `--write` produces valid JSONL with
   the same line count (pruned content replaced, not removed — must stay
   parseable transcript shape).

## Validation

`cargo test` + `cargo clippy --all-targets -- -D warnings` + `cargo fmt
--check`, plus a real run against this session's own live transcript
file (multi-thousand lines) — not just the trimmed fixtures.

## Risks

- Real transcript schema isn't versioned — defensive parsing (skip, not
  panic) is required, not optional; a test proves it (a deliberately
  malformed line in a fixture must not abort parsing of the rest).
- Rule 2 stays off by default per board consensus — must not be
  silently promoted to default-on without real usage data first.
- `session_to_trm.py` (Python, in mindforge) is the known next
  consumer of this — explicitly out of scope for this plan (user:
  "write session_prune in rust first then come back for
  session_to_trm"). Noting here so the follow-on isn't lost: that
  script should eventually be ported into squishi itself (Rust), not
  stay a separate Python tool in a different repo.
