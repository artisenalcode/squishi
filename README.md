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
squishi "<text>"
cat file | squishi          # or pipe via stdin — no shell-argument size limit
```

Every call first runs a zero-model, unconditional pre-pass
(`base64_strip`) that strips base64-encoded blobs — inline data-URIs
and long standalone base64 runs — before shape detection even runs.
Not a `ContentKind` of its own: a blob can appear inside JSON, logs,
diffs, or plain text alike, the same way MCE's `Layer1Pruner` runs
ahead of its shape-aware routing (audited `~/Code/_labs/audit-repos/MCE`,
2026-08-07 — squishi had no base64 handling at all before this). A
matched blob is replaced with `[... squishi pruned: base64 blob
removed, N chars ...]`; `--json` output reports `base64_blobs_removed`
when anything was stripped. Two thresholds, calibrated against real
fixtures, not guessed: a `data:...;base64,`-prefixed blob only needs
20 chars of payload to count (the prefix itself is the high-confidence
signal); an unprefixed standalone run needs 500+ chars — found by
testing a real JWT, whose ~200-char base64url payload segment matched
and was wrongly stripped at an earlier, lower threshold before this
was raised.

Then, `content_detect` classifies the (now blob-stripped) content shape
and routes:

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

## `--doctor` — self-diagnostics

`squishi --doctor` checks this tool's own real failure surface instead of
compressing anything: binary identity (which build is actually running,
via `current_exe()` + crate version), whether magika loads, where the
semantic-dedup model cache (`hf_hub::Api::new()`) actually resolves to and
whether the model files are already cached there, whether semantic-dedup
itself loads, and a best-effort proxy signal for the PostToolUse hook
(file presence + `last-input.json`'s mtime — explicitly *not* a
registration check; there's no reliable programmatic way to confirm a
plugin-registered hook is live, see `anthropics/claude-code#84439`).

```
squishi --doctor          # full run, real model loads
squishi --doctor --quick  # skips magika/semantic-dedup loads
squishi --doctor --json   # structured output, same --json flag as compress
```

Exit code 1 if any check fails, 0 otherwise (warnings are fine). Not a
subcommand — `squishi doctor` would collide with compressing the literal
string `"doctor"` — so it's a flag, ignoring `text` when set. No `--fix`:
unlike `total-recall`'s own `doctor` (same `Check`/`Status` shape,
separate implementation — no shared crate yet, only two real
implementations exist), squishi has no repairable persistent state: no
locks, no bank dirs, nothing that gets stuck.

Real finding surfaced by this check, not assumed: `hf_hub::Api::new()`
(what `semantic_dedup::SemanticDedup::load` actually calls) resolves via
`Cache::default()`, which always uses `~/.cache/huggingface/hub` —
`HF_HOME` is silently ignored by that call path. `--doctor` reports this
explicitly when `HF_HOME` is set but wouldn't take effect.

## `session_prune` — structural pruning for Claude Code session transcripts

A different problem from every compressor above: those operate on a
single blob's *text shape*; `session_prune` operates on a transcript's
*structure* (which tool ran, on what path, superseded by what). Real
measurement found squishi's shape compressors barely touch real session
transcripts — the one path that loads the expensive semantic model,
`PlainText`, had zero qualifying blocks in a real 4713-line coding
session; everything reads as code/diff/log-shaped. Session pruning is
what actually finds signal in that content.

Originally speced from `total-recall-kit` (deleted from `_labs`,
evaluated for value first), pressure-tested by a technical-advisor board
review (2026-08-07) before being built:

- **Dedupe latest read** — an older `Read` of a path is prunable once a
  newer `Read` of the same path exists later in the session.
- **Supersede write by read** *(off by default — see below)* — a
  `Write`/`Edit`'s own tool result is prunable once a later `Read` of
  the same path verifies it landed correctly.
- **Drop redundant errors** — an error tool-result is prunable if an
  identical `(tool, content)` error already appeared earlier.
- **Prune old large tool outputs** — outputs over `--min-bytes` are
  prunable once they're no longer within the last `--window` session
  items (a recency window over item count, not a wall-clock cutoff).
- **Collapse task launches** — repeated "running in background with
  ID:" tool-results collapse to the latest one.

```
squishi --session-prune <transcript.jsonl>                  # stats report, mutates nothing
squishi --session-prune <transcript.jsonl> --json           # structured stats
squishi --session-prune <transcript.jsonl> --write out.jsonl  # a pruned COPY, original untouched
squishi --session-prune <transcript.jsonl> --include-rule-2   # also run the flagged rule
```

**Rule 2 ships off by default.** The board review flagged real
false-positive risk — a write's content a later read didn't fully
re-verify — so it's opt-in via `--include-rule-2` until real usage data
justifies flipping the default.

**No live transcript mutation, ever.** Confirmed against the real Claude
Code hooks reference: no hook can rewrite a past transcript entry —
`PostToolUse`'s `updatedToolOutput` only applies to the *current* tool
call. `--write` always produces a new copy; the input transcript is
never touched, by this tool or (today) by anything that could feed its
output back into a live session automatically.

Not a subcommand — same "flag not subcommand" reasoning as `--doctor`.

## `session_digest` — extraction + compression for total-recall staging

Rust port of `mindforge/tools/session_to_trm.py` (Python, being retired)
plus the extraction half of `extract_claude_sessions.py`. Pulls
human/assistant prose out of a Claude Code session transcript (drops
`tool_use`/`tool_result` blocks entirely, strips trailing
`<system-reminder>` injections), compresses it through squishi's normal
`route()` path, and builds a ready-to-stage digest with a metadata
header (`session_id`, `cwd`, `first_ts`, `last_ts`, `turn_count`).

```
squishi --session-digest <transcript.jsonl>          # digest text on stdout
squishi --session-digest <transcript.jsonl> --json   # + session_id/cwd/turn_count/... for a caller
squishi --session-digest <transcript.jsonl> --max-chars 50000
```

**Deliberately not layered on `session_prune`.** Real finding, checked
before assuming they'd compose: `session_prune` only touches
`tool_result` content; `extract_session_text` discards all
`tool_use`/`tool_result` blocks unconditionally, regardless of pruning.
They operate on disjoint fields of the same transcript — running one
before the other has no effect on the output.

**Boundary held**: this is extraction and compression only. squishi
never calls `trm` or anything storage-shaped — `total-recall`'s own
`trm ingest-session <path>` is the caller that runs this and stages the
result, keeping the actual `stage` call in the tool that already owns
storage.

## Development

```bash
cargo test                    # fast — semantic_dedup's real-model tests are #[ignore]d
cargo test -- --ignored       # slow — downloads/runs the real MiniLM model
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```
