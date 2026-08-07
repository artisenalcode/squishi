# Plan: port `session_to_trm.py` to Rust — squishi does extraction+digest+
compression, total-recall owns the stage call

Executes the deferred half of the session-compression thread (board
review 2026-08-07, `session_prune` shipped first per explicit
sequencing). Ports `mindforge/tools/session_to_trm.py` (140 lines) and
the extraction half of `extract_claude_sessions.py` (193 lines) out of
Python, into Rust, split across the two tools that already own the two
real responsibilities — not merged into one.

## Goal

`squishi` gains a mode that turns a raw Claude Code transcript into a
compressed, ready-to-stage digest (content + metadata), on stdout as
JSON. `total-recall` gains a new subcommand that calls squishi for that,
then stages the result itself — squishi's own explicit, repeatedly-
stated boundary ("compresses text, never stores/retrieves — that's
total-recall's job") stays intact; the Python script's actual `trm
stage` subprocess call moves to the tool that already owns staging.

## Real finding, checked before designing: `session_prune` doesn't
compose with this pipeline

`extract_session_text` (the existing Python logic, being ported) only
keeps `message.content` blocks with `type == "text"` from `user`/
`assistant` records — `tool_use`/`tool_result` blocks are discarded
outright, unconditionally, regardless of pruning. `session_prune`
(shipped separately, same day) only touches `tool_result` content.
**These operate on disjoint fields of the same transcript** — running
`session_prune` before extraction has zero effect on the extracted text.
Not integrated here; `session_prune` remains a separate tool for a
different real need (reducing tool-call noise in a transcript kept in
its own JSONL shape), not a pre-pass for this prose-digest pipeline.

## Context — ported logic, read in full from the real Python source

- `extract_session_text(path, max_chars)`: walks JSONL, tracks
  `sessionId`/`cwd`/first+last `timestamp` across all record types
  (not just text-bearing ones), filters to `type in (user, assistant)`,
  pulls `message.content` as either a plain string or a list of blocks
  (`type == "text"` only), strips a trailing `<system-reminder>...`
  block (everything from that tag to the end of the string — Python's
  `re.DOTALL` substitution, not just the same line), joins as
  `"USER: {text}"` / `"ASSISTANT: {text}"` lines separated by blank
  lines, tracks `turn_count`. Truncates in the *middle* (keep head +
  tail, drop the middle) if the joined text exceeds `max_chars`,
  flagging `truncated: true`.
- `build_digest_content(compressed_text, meta)`: a fixed-format header
  (`SESSION DIGEST <id>`, then a `---`-delimited metadata block:
  `type: session-digest`, `session_id`, `cwd`, `first_ts`, `last_ts`,
  `turn_count`) followed by the compressed text.
- `stage_in_session_bank`: calls `trm stage --reason <r> --source
  direct`, content via stdin, **with the subprocess's `cwd` set to the
  session's own original project directory** — so `trm`'s own git-
  remote/cwd bank resolution picks the right bank, not a reimplemented
  copy of that logic. This is the one piece of real behavior the Rust
  port must preserve exactly: the bank resolution must happen against
  the *session's* cwd, not whatever directory the ingest command itself
  is run from.
- `reason` string format: `"Claude Code session {id} in {cwd} ({n} text
  turns) — squishi-compressed transcript, judge which durable facts are
  worth keeping"`.

## Steps

### Part 1 — squishi: `src/session_digest.rs`

1. `SessionMeta { session_id: Option<String>, cwd: Option<String>,
   first_ts: Option<String>, last_ts: Option<String>, turn_count: usize,
   truncated: bool, raw_bytes: usize }`.
2. `extract_session_text(jsonl: &str, max_chars: usize) -> (String,
   SessionMeta)` — faithful port of the Python function above. Char-
   safe truncation (Rust byte-slicing mid-UTF8-char panics; Python's
   `len()`/slicing is codepoint-based) — slice via a `Vec<char>` or
   `char_indices`, not raw byte indexing.
   *Check*: a real trimmed excerpt of this session's own transcript
   (same fixture-sourcing discipline as `session_prune`'s tests) —
   assert turns extracted, system-reminder stripped, meta fields
   correct; a synthetic over-`max_chars` case proves the middle-
   truncation shape.
3. `build_digest_content(compressed: &str, meta: &SessionMeta) ->
   String` — same header format as the Python version.
4. CLI wiring (`src/main.rs`): `--session-digest <TRANSCRIPT_PATH>`
   (flag, not subcommand — same precedent as `--doctor`/
   `--session-prune`), `--max-chars` (default 100_000, matching Python).
   Extracted text is compressed via squishi's **existing** `route()` —
   the same generic dedup/shape compression the Python version already
   gets by piping through the real `squishi` CLI; nothing new to build
   there. Default output: the digest content, bare, on stdout (pipeable
   directly). `--json`: `{content, session_id, cwd, first_ts, last_ts,
   turn_count, truncated, raw_bytes, chars_before, chars_after}` — the
   full contract `trm ingest-session` needs (content plus every meta
   field required to build the `reason` string and resolve the bank).
   *Check*: run against this session's own real transcript; confirm
   sane, real output and that normal compress/`--doctor`/
   `--session-prune` paths are unaffected (full existing test suite).

### Part 2 — total-recall: `trm ingest-session <path>`

5. New `Commands::IngestSession { path: PathBuf }` (or similar) in
   `src/main.rs`. Shells out to `squishi --session-digest <path>
   --json` (subprocess — total-recall doesn't depend on squishi's crate,
   consistent with how `session_to_trm.py` already called the real
   `squishi` binary on PATH), parses the JSON contract from step 4.
   *Check*: a real run against this session's own transcript produces
   the expected digest fields.
6. Build the `reason` string (same format as the Python version) from
   the parsed meta. Call `handover::stage` **in-process** (not by
   shelling out to `trm stage` on itself) with the bank resolved from
   `meta.cwd`, not the `trm ingest-session` process's own cwd — the one
   piece of real behavior from the Python version that must be
   preserved exactly (`bank::resolve_bank_id(None, Path::new(&meta.cwd))`
   at the ingest call site, not the ambient `std::env::current_dir()`
   every other `trm` command uses).
   *Check*: a real run against this session's own transcript, executed
   from an unrelated `cwd`, stages into the *session's* bank (verified
   against `bank::resolve_bank_id`'s real output for that cwd), not
   whatever bank the ambient shell cwd would resolve to.
7. `CORE_DOCS` update (`trm skill get core`) — new `## Ingest session`
   section, matching the existing per-command doc pattern.

## Non-goals

- No bulk/resumable multi-session scan (`extract_claude_sessions.py`'s
  own `main()` — walks `~/.claude/projects/*/*.jsonl`, keeps a state
  file of already-staged sessions). Out of scope for this plan; a
  single-session `trm ingest-session <path>` is the unit being ported.
  A bulk-scan wrapper (shell loop, or a later `--all` flag) can reuse
  this once it exists — not built speculatively here.
- `session_prune` is not wired into this pipeline — see the real finding
  above.
- `mindforge/tools/session_to_trm.py` and `extract_claude_sessions.py`
  are not deleted by this plan — that's a separate, explicit follow-on
  once the Rust path is proven equivalent on real transcripts.

## Validation

- squishi: `cargo test`, `cargo clippy --all-targets -- -D warnings`,
  `cargo fmt --check`. Real run against this session's own transcript.
- total-recall: same three gates. Real run against this session's own
  transcript from an unrelated cwd, confirming correct cross-cwd bank
  resolution.

## Risks

- **UTF-8 truncation correctness** — the one real port-fidelity risk
  (Python string slicing is codepoint-based and never panics; Rust byte
  slicing does on a non-boundary index). Must be covered by a real test
  with multi-byte characters in the fixture, not just ASCII.
- **cwd-based bank resolution at the ingest call site** is a real
  behavioral requirement, not a nice-to-have — the whole point of
  matching the Python version's `cwd=session_cwd` subprocess argument.
  Must be a real, verified test (run from directory A, session claims
  cwd B, confirm it lands in B's bank), not assumed from the code
  reading correct.
