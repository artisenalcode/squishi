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
- **Git diffs / unified diffs** (`diff_compress`) — caps file count
  (keeps the heaviest by change volume), caps hunks/file (first + last +
  top-scored middle — scored by change density, priority keywords, and
  query-context word overlap), trims context lines around each `+`/`-`.
  See below for the port rationale and real measurements.
- **Plain prose** (`line_dedup` + `semantic_dedup`) — line-dedup first,
  then, if the result is still over 2000 chars, sentence-level paraphrase
  dedup (see below). Under the threshold, dedup alone is the whole pass —
  no model load, stays fast.
- **Everything else** (`line_dedup`) — collapses runs of >5 identical
  consecutive lines. Safe, lossless-in-spirit; never destroys
  non-repeating structure. Content that doesn't match Json/SearchResults/
  Log/PlainText gets a real sub-classification via `magika` (rust/html/
  diff/csv/markdown/... — see below) rather than a blind catch-all, even
  though compression behavior is the same dedup-only pass for now (these
  are structured formats, not prose — sentence-level paraphrase dedup
  doesn't apply).

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

Diff detection is also a fast regex tier, checked before Log: a
`diff --git`/`--combined`/`--cc` header, or a naked `--- a/`/`--- /dev/null`
file marker paired with an `@@` hunk header (unified diff without the git
wrapper). Has to run before the Log check — diff hunks routinely contain
words like `fail`/`error` in test code, which would otherwise misroute
them.

Prints `{"compressed", "kind", "source", "chars_before", "chars_after",
...}` (extra fields vary by which compressor ran).

## `diff_compress` — real port of headroom's `DiffCompressor`, CCR stripped

Unlike `log_compress`/`json_compress`/`search_compress` (own design,
same shape as headroom's mechanism), this one is a much closer port of
`crates/headroom-core/src/transforms/diff_compressor.rs`: the parser,
hunk-relevance scorer (change density + priority-keyword regex +
query-context word overlap), first+last+top-scored-middle hunk
selection, and context-line trimming are all carried over near-verbatim
— that logic isn't headroom-proxy-specific, it's just correct diff
handling. What's stripped entirely: the CCR cache-key/retrieval-marker
machinery (MD5 hash, `[N lines compressed to M. Retrieve: hash=...]`
marker, `CcrStore` persistence) and the `DiffCompressorStats`
observability sidecar (tracing spans, per-file drop stats) — squishi has
no store (total-recall's job) and no daemon to keep counters in.

Config defaults are headroom's own (`max_files: 20`, `max_hunks_per_file:
10`, `max_context_lines: 2`) — kept as-is after real measurement, not
re-tuned on a guess. Tested against real headroom commits, not just
synthetic fixtures:

- A typical PR-sized diff (20 files, ≤9 hunks in any single file — under
  both caps): 84,448 → 78,515 chars, **7.0% reduction**, `hunks_removed:
  0`. All the savings came from trimming git's default 3-line context
  down to 2 — neither structural cap fires on an ordinary diff, since
  headroom's thresholds were tuned for an LLM-proxy's worst-case
  traffic, not typical code-review-sized diffs.
- An oversized diff (24 files, one file with >10 hunks): 100,898 →
  84,748 chars, **16.0% reduction** — file cap dropped 4 files, hunk cap
  dropped 9 hunks. Output verified intact: commit header, rename/mode
  markers, hunk structure, summary footer all correct.
- Cost is pure CPU (regex + string ops, no ML, no subprocess): ~27-34ms
  cold (includes one-time regex compilation) for the largest diff
  tested (100KB/24 files), ~6-7ms warm steady-state. Three orders of
  magnitude cheaper than `semantic_dedup`'s ONNX pass — unlike Kompress,
  there was never a real cost-vs-benefit tension here to resolve.
- Conclusion: defaults are fine to ship as-is. The savings ceiling on
  everyday diffs is modest by design (headroom's caps rarely engage
  below outlier-sized changesets) — retune `max_hunks_per_file`/
  `max_files`/`max_context_lines` later against real governator traffic
  if that turns out to matter, not against a guess made before any
  traffic exists.

## `semantic_dedup` — wired in, replaces the earlier Kompress port

Sentence-level paraphrase dedup: collapse the same idea restated in
different words, not just exact-duplicate lines (`line_dedup`'s job).
Same model and greedy single-pass clustering algorithm as
`advisory/tools/dedupe_semantic.py` (`sentence-transformers/all-MiniLM-
L6-v2`, STS-tuned), ported to raw Rust `ort` — not `fastembed`:
`fastembed` hard-pins `ort =2.0.0-rc.13`, incompatible with `magika`'s
hard pin on `=2.0.0-rc.12` in the same crate, and measured slower
besides (~1.07s warm via `fastembed` vs ~587-708ms warm via raw `ort`
for the same model). Downloaded via `hf-hub` + run locally via
`ort`/ONNX Runtime, cached after first use — same stack `total-recall`'s
embeddings already prove out, just the raw API instead of the wrapper.

Wired into the default `compress` path for `PlainText`: line-dedup
runs first, and only if the result stays over 2000 chars does
`semantic_dedup` load the model and run. Sentences are split, embedded,
mean-pooled + L2-normalized (the standard sentence-transformers recipe —
raw ONNX output is per-token `last_hidden_state`, not pre-pooled),
then greedily deduped by cosine similarity (threshold 0.80, matching
`dedupe_semantic.py`'s default). Sentences under 8 or over 40 words are
never dropped — too short to be a meaningful whole-sentence paraphrase
comparison. Measured on real paraphrase-heavy content: 21 sentences → 11,
2035 chars → 1049 (~48% reduction), ~2.3s cold (includes first-run model
download + load).

Replaces the earlier native Kompress port (`src/kompress.rs`, removed):
word-level keep/drop showed real compression only on genuinely
filler-heavy prose and cost 4-18s per call with no state to reuse across
invocations. This model is smaller (~90MB vs 261-601MB), faster
(sub-second warm vs multi-second), and does a fundamentally more useful
job for this content — comparing whole sentences to each other, not
scoring words in isolation. Concluded not worth the cost for what it was
trying to do; removed rather than kept dormant.

If the model is unavailable (offline, first-run download fails), falls
back to the line-dedup result rather than failing outright —
`source: "dedup-semantic-unavailable"` in the JSON output signals this.

`ort` is pinned to `=2.0.0-rc.12` — `magika` hard-pins `rc.12` even on
its latest unreleased source, and `[patch]` can't bypass an exact pin
from the same registry without forking.

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
cargo test                    # fast — semantic_dedup's real-model tests are #[ignore]d
cargo test -- --ignored       # slow — downloads/runs the real MiniLM model
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```
