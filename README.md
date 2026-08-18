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

Every call first runs two zero-model, unconditional pre-passes before
shape detection runs at all:

- **`line_number_strip`** — strips a real Claude Code `Read`-tool-shaped
  `N\t` line-number prefix (`cat -n` style, no padding), if and only if
  every line matches it with strictly sequential numbers starting at 1.
  Real finding, not a guess (`governator-proxy`'s Step 2 live-API check,
  2026-08-18): without this, `Read` output — Claude Code's single most
  common tool call — got classified `Other("tsv")` by Magika (a kind
  squishi has no compressor for) and every line-anchored fast-path regex
  (Diff/Log/SearchResults) misfired, since a real diff header or log
  keyword on line N no longer starts the line once `"N\t"` is in front
  of it. Real before/after on the same captured content: 9290→9291 chars
  (no reduction) with the prefix present, 8910→121 chars with it
  stripped. `--json` output reports `line_numbers_stripped: true` when
  it fires.
- **`base64_strip`** — strips base64-encoded blobs — inline data-URIs
  and long standalone base64 runs. Not a `ContentKind` of its own: a
  blob can appear inside JSON, logs, diffs, or plain text alike, the
  same way MCE's `Layer1Pruner` runs ahead of its shape-aware routing
  (audited `~/Code/_labs/audit-repos/MCE`, 2026-08-07 — squishi had no
  base64 handling at all before this). A matched blob is replaced with
  `[... squishi pruned: base64 blob removed, N chars ...]`; `--json`
  output reports `base64_blobs_removed` when anything was stripped. Two
  thresholds, calibrated against real fixtures, not guessed: a
  `data:...;base64,`-prefixed blob only needs 20 chars of payload to
  count (the prefix itself is the high-confidence signal); an
  unprefixed standalone run needs 500+ chars — found by testing a real
  JWT, whose ~200-char base64url payload segment matched and was
  wrongly stripped at an earlier, lower threshold before this was
  raised.

Then, `content_detect` classifies the (now line-number- and
blob-stripped) content shape and routes:

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
  Log/PlainText gets a real sub-classification via Google's Magika model
  (rust/html/diff/csv/markdown/... — see below) rather than a blind
  catch-all, even
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

Runs Google's real, unmodified `standard_v3_3` model (the same weights
the official `magika` crate ships) directly through `candle-onnx`,
instead of depending on that crate — squishi previously depended on it
for exactly this, but it hard-pins an `ort` (ONNX Runtime) prerelease
that repeatedly collided with other tools in this workspace; see
`docs/ideation/ort-dependency-consistency/2026-08-18-ort-pin-and-bottleneck-plan.md`
for the investigation, and `src/content_detect.rs`'s module doc comment
for exactly which ops candle-onnx needed patched in to run this graph.
The model (3.1MB) is embedded straight into the binary
(`assets/magika-standard_v3_3.onnx`, zero network calls, unlike
Kompress's on-demand download); the byte-feature-extraction algorithm
and the per-label threshold/canonicalization table
(`src/magika_labels.rs`) are ported line-for-line from the real
`magika` crate's own source so classifications stay identical to what
the official CLI reports. Cost: decodes once per process (well under a
millisecond, cached in a `LazyLock`) plus per-classification inference
time — only paid on the fallback path, never on content the regex
checks already classified.

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
L6-v2`, STS-tuned) — not `fastembed`: `fastembed` hard-pinned an `ort`
version incompatible with what `magika` (this crate's other ONNX
consumer at the time) required, and measured slower besides (~1.07s
warm via `fastembed` vs ~587-708ms warm via raw `ort` for the same
model). Ported off raw `ort` onto `candle` entirely on 2026-08-18 (see
the Detection section above and
`docs/ideation/ort-dependency-consistency/2026-08-18-ort-pin-and-bottleneck-plan.md`)
— squishi no longer depends on `ort` at all, so it can't re-collide with
another tool's `ort` resolution again. Re-measured against the same real
model before the switch: cosine similarity 1.0 against the old `ort`
output (bit-for-bit equivalent embeddings, not just "close enough"), and
~40% faster (candle loads real HF safetensors directly, no ONNX
conversion step). Downloaded via `hf-hub`, cached after first use.

Wired into the default `compress` path for `PlainText`: line-dedup
runs first, and only if the result stays over 2000 chars does
`semantic_dedup` load the model and run. Sentences are split, embedded,
mean-pooled + L2-normalized (the standard sentence-transformers recipe —
the raw model output is per-token `last_hidden_state`, not pre-pooled),
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

## `--doctor` — self-diagnostics

`squishi --doctor` checks this tool's own real failure surface instead of
compressing anything: binary identity (which build is actually running,
via `current_exe()` + crate version), whether the embedded Magika model
classifies real content correctly, where the semantic-dedup model cache
(`hf_hub::Api::new()`) actually resolves to and
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
Code hooks reference (code.claude.com/docs/en/hooks, 2026-08-18): no hook
can rewrite a past transcript entry at all — `PostToolUse` has no field
that replaces or edits tool output ("`PostToolUse` hooks can't undo
actions since the tool has already executed" is the doc's own wording);
its only output mechanism, `hookSpecificOutput.additionalContext`,
appends text alongside the original result, never removes or replaces it.
An earlier version of this note cited a `updatedToolOutput` field as the
scoping reason — that field does not exist in the real API, found the
hard way building `agents-brain`'s now-retired PostToolUse compress hook.
`--write` always produces a new copy regardless; the input transcript is
never touched, by this tool or by anything else, structurally not just by
convention.

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

## `--level` — how hard each compressor pushes

`conservative` / `default` / `aggressive` tune every threshold that used
to be a fixed constant: `semantic_dedup`'s paraphrase-similarity cutoff,
`diff_compress`'s context/hunk/file caps, `log_compress`'s error/warning/
context/total-line caps, and `json_compress`'s keep-edge cap. `default`
reproduces this repo's pre-`--level` behavior exactly — same numbers as
always, byte-for-byte.

```
squishi --level aggressive <text>
squishi --level conservative --json <text>
```

Values picked from real before/after measurement on real fixtures, not
guessed — same standard as `diff_compress`'s own numbers above:

| Surface | Conservative | Default | Aggressive | Fixture |
|---|---|---|---|---|
| `diff_compress` | 215,374 → 202,805 chars (5.8%) | → 135,729 (37.0%) | → 69,411 (67.8%) | real headroom commit (`eaf5980b`, 215KB) |
| `log_compress` | 607,751 → 24,375 chars (96.0%) | → 7,495 (98.8%) | → 2,346 (99.6%) | real system journal, error/warning lines |
| `json_compress` | 138,204 → 5,523 chars (96.0%) | → 2,787 (98.0%) | → 1,169 (99.2%) | real 400-element JSON array (graphify's own graph.json nodes) |
| `semantic_dedup` | 181,315 → 181,082 chars (0.1%) | → 180,036 (0.7%) | → 176,836 (2.5%) | real YouTube auto-caption transcript, 181KB |

`semantic_dedup`'s savings stay modest at every level on well-formed
prose — expected, not a bug: it only drops genuinely near-duplicate
sentences, and curated/edited text has little of that to begin with. The
gain shows up on repetitive raw material (auto-captions, meeting
transcripts), which is exactly what the fixture above is.

## `--session-stats` — cumulative real savings across a session

Scans a Claude Code session transcript (JSONL) for every squishi
`--json` call it contains and reports real cumulative `chars_before`/
`chars_after`, broken down by content `kind` and as a grand total.
Read-only, never mutates the transcript.

```
squishi --session-stats <transcript.jsonl>
squishi --session-stats <transcript.jsonl> --json
```

Matches `ToolResultItem.content` (from `session_prune`'s own transcript
parser) against squishi's own five-key `--json` contract
(`compressed`/`kind`/`source`/`chars_before`/`chars_after`) rather than
identifying squishi calls by tool name or command text — `ToolUseItem
.path` only captures Read/Write/Edit's `input.file_path`, not a Bash
command, so name/command matching isn't reliable. This means it's
invocation-method-agnostic (works whether squishi ran via Bash, a
wrapper, anything), but also means a transcript with **zero** matching
entries is reported honestly as zero calls, not an error — most
transcripts squishi doesn't actively instrument will look like that, and
that's the correct answer, not a false negative.

## Development

Building `candle-onnx` (used by `content_detect`'s Magika path) needs a
real `protoc` (Protocol Buffers compiler) binary on `PATH` or pointed to
via `PROTOC=/path/to/protoc` — its `build.rs` calls `prost_build` at
compile time. Not needed by anything else in this crate. No sudo
required to get one: download a static release from
[protocolbuffers/protobuf releases](https://github.com/protocolbuffers/protobuf/releases)
and drop it somewhere on `PATH` (e.g. `~/.local/bin/protoc`).

```bash
cargo test                    # fast — semantic_dedup/punctuation_restore's real-model tests are #[ignore]d
cargo test -- --ignored       # slow — downloads/runs the real MiniLM + punctuation models
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```
