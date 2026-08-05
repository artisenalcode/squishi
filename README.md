# squishi

Rust-native text compressor. Pure functions, zero dependencies beyond
`clap`/`regex`, no subprocess, no network, no storage.

## Boundary

squishi compresses text. It does not store or retrieve anything — that's
`total-recall`'s job (formerly `mf`, `~/Code/_labs/mindforge/total-recall`).
Don't add a cache, a database, or a retrieval marker here; if a caller
needs to get back what was compressed away, that's a `total-recall` bank
entry, not squishi's concern.

## What it does today

- `line_dedup` — collapse runs of >5 identical consecutive lines. Safe,
  lossless-in-spirit pre-pass; never destroys non-repeating structure.
- `log_compress` — classify each line (error/warn/info), score it, keep
  errors/warnings (first + last + highest-scoring middle, capped),
  summary lines (`=== N passed/failed ===` etc.), and context around each
  kept line. Emits `[N lines omitted: X error, Y warn]` for the rest.

Both arrived at by reading `headroom`'s LogCompressor mechanism (classify
→ score → select → format) to understand *why* it gets real compression
on repetitive logs, then implementing squishi's own version in plain
Rust — no Python, no venv, no external process.

```bash
squishi compress "<text>"
```

Pipeline: dedup first (free); if still over ~2000 chars, run log_compress
on the deduped text. Prints `{"compressed", "source", "chars_before",
"chars_after"}`.

## Remembered: the total-recall-kit ruleset (not built yet)

`total-recall-kit` (deleted from `_labs`, evaluated for value first) had a
second, different class of rule — not text compression, but **session
pruning**: which whole tool-call results in a harness's conversation
history can be dropped because a *later* item already supersedes them.
Worth remembering as a distinct future mode, since it operates on
structured session items (role + tool metadata), not raw text:

- **Dedupe latest read** — an older `read` of a file is prunable once a
  newer `read` of the same path exists in the session.
- **Supersede write by read** — a `write`'s tool output is prunable once
  a later `read` of the same path verifies it landed correctly.
- **Drop redundant errors** — an error tool-output is prunable if an
  identical `(tool, content)` error already appeared earlier.
- **Prune old large tool outputs** — outputs over N chars are prunable
  once they're no longer within the last K messages (a recency window,
  not an age cutoff).
- **Collapse task launches** — repeated "background task launched" logs
  collapse to the latest one.

None of this is implementable without a harness feeding squishi
structured session data (role, tool name, path, timestamp) — today
squishi only sees raw text. If a harness integration ever wants this,
it's a new module (`session_prune`, say), not a bolt-on to `line_dedup`
or `log_compress`.

## Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```
