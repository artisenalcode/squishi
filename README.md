# squishi

Rust-native text compressor. Detects content shape, routes to the
matching technique. No subprocess for its default path, no network, no
storage.

## Boundary

squishi compresses text. It does not store or retrieve anything — that's
`total-recall`'s job (formerly `mf`, `~/Code/_labs/total-recall`, now its
own standalone repo). Don't add a cache, a database, or a retrieval
marker here; if a caller needs to get back what was compressed away,
that's a `total-recall` bank entry, not squishi's concern.

## What it does today

```bash
squishi compress "<text>"
```

Detects content shape (`content_detect`) and routes:

- **JSON** (`json_compress`) — dedupes exact-duplicate array elements,
  caps large arrays to first-5 + last-5 + a dropped-count marker. A
  single object (nothing repeatable) passes through unchanged.
- **Search results** (`search_compress`) — groups grep/ripgrep-style
  `path:line:text` output by file, caps to 5 matches/file with an
  omitted-count marker, preserves per-file grouping even when input is
  interleaved.
- **Log/build output** (`log_compress`) — classifies each line
  (error/warn/info/summary), scores it, keeps errors/warnings (first +
  last + highest-scoring middle, capped) plus summary lines and context
  around each kept line. Emits `[N lines omitted: X error, Y warn]` for
  the rest.
- **Everything else** (`line_dedup`) — collapses runs of >5 identical
  consecutive lines. Safe, lossless-in-spirit; never destroys
  non-repeating structure. Content that doesn't match Json/SearchResults/
  Log gets a real sub-classification via `magika` (rust/html/diff/csv/
  markdown/... — see below) rather than a blind "PlainText" label, even
  though compression behavior is the same dedup-only pass for now.

`log_compress` and the router shape were arrived at by reading
`headroom`'s ContentRouter/LogCompressor mechanism (classify → score →
select → format) to understand *why* it gets real compression, then
implementing squishi's own version in plain Rust — no Python, no venv,
no external process for any of the above.

### Detection: regex first, Magika as fallback — deliberately, not arbitrarily

The fast regex/parse checks (Json/SearchResults/Log) run first and are
authoritative — measured against real Magika output and found *more*
precise for these three: Magika labeled a single compact JSON object
`"jsonl"` and had no dedicated label for ad-hoc application logs at all
(fell back to `"txt"`). Magika only gets consulted when those checks
find nothing — where it's a genuine upgrade over the old blind
`PlainText` catch-all: real file-type labels (rust, html, diff, csv,
markdown, ...) instead of one undifferentiated bucket.

Cost: ~111ms to load the model (`google/magika`'s official Rust crate —
3.1MB, bundled directly in the crate, zero network calls, unlike
Kompress's on-demand download) plus ~28ms/classification — only paid on
the fallback path, never on content the regex checks already classified.

Prints `{"compressed", "kind", "source", "chars_before", "chars_after",
...}` (extra fields vary by which compressor ran).

## `kompress` — real, tested, not wired in

`src/kompress.rs` is a native Rust port of headroom's Kompress: a
learned per-word keep/drop classifier (ModernBERT,
`chopratejas/kompress-v2-base`, downloaded via `hf-hub` + run locally via
`ort`/ONNX Runtime — the same stack `total-recall`'s embeddings already
prove out, just applied to a different model). Correctness-verified
against the real ONNX I/O contract (not assumed from headroom's Python
source) and against real content: on genuinely filler-heavy prose it
gets real, meaningful compression (412 words → 327, keeping content
words, dropping "that"/"of"/"a"-type filler).

**Deliberately not wired into the default `compress` path.** Every
invocation is a fresh process with no state to reuse, so every call pays
the full ONNX session-load cost from scratch — measured at ~7.4s just to
load, before any inference. That's fine for `line_dedup`/`log_compress`/
etc. (all complete in milliseconds) but not worth triggering for
`compress`'s default path, and not worth an unconditional model download
baked into normal usage. A daemon architecture would amortize the load
cost across calls; nothing here needs that yet. Revisit if/when a real
use case shows up — the module, its tests (`#[ignore]`d by default, run
explicitly with `cargo test -- --ignored`), and the verification probes
(`examples/probe_kompress.rs`, `examples/probe_tokenizer.rs`,
`examples/probe_scores.rs`) stay in the repo either way.

Two real levers if this gets revisited, neither tried yet: `ort`'s
`with_optimized_model_path` (serialize the post-optimization graph once,
skip re-optimizing on every cold start) as a no-daemon fix, or a
long-running process that loads the session once and serves requests
over a local socket — the standard fix for "expensive load, cheap
inference," but a real architecture change (squishi stops being a
stateless one-shot CLI).

`ort` is pinned to `=2.0.0-rc.12` (was `=2.0.0-rc.13`) — `magika` hard-
pins `rc.12` even on its latest unreleased source, and `[patch]` can't
bypass an exact pin from the same registry without forking. Confirmed
API-compatible for everything squishi uses (`kompress.rs` needed zero
changes), so this was the pragmatic fix over forking.

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
cargo test                    # fast — kompress's real-model tests are #[ignore]d
cargo test -- --ignored       # slow — downloads/runs the real kompress model
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```
