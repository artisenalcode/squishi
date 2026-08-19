# Plan: port headroom's SmartCrusher row-selection + lossless compaction into squishi

Triggered from governator work (2026-08-19): governator dropped its
Python `headroom` escalation tier entirely (squishi already replicates
headroom's Log/JSON/Search compressors), but two real capabilities in
headroom's `SmartCrusher` never had a squishi equivalent:

1. **Statistics-driven row selection** — squishi's `json_compress.rs`
   caps large arrays to a blind first-N/last-N with a verified-fact
   marker for what's dropped. SmartCrusher instead scores every row
   (rarity, structural-outlier, error-keyword) and preserves the
   *interesting* ones — an error buried at index 40 of a 100-row array
   survives; squishi's blind edge-keep drops it today.
2. **Lossless CSV-schema/markdown-kv compaction** — for a cleanly
   tabular array, headroom can render the whole thing as CSV+schema
   (or markdown-kv) instead of dropping any rows at all, when that
   rendering saves ≥30% of bytes vs the minified JSON. Squishi has no
   lossless-rendering path today; it only ever drops or keeps items
   verbatim.

CacheAligner (headroom's other module) was investigated and ruled out —
it's a detector-only system-prompt linter (warns about UUIDs/JWTs/
timestamps destabilizing a cache prefix), never mutates or shrinks
text. Not a compression feature; nothing to port.

## Source of truth

Real implementation, not the retired Python: `headroom`'s Rust core at
`/home/alvin/Code/_labs/headroom/crates/headroom-core/src/transforms/smart_crusher/`
(11.4k lines across 19 files, 388 unit tests) plus its
`compaction/` subdirectory (6 files, ~2.7k lines:
`classifier.rs`, `compactor.rs`, `formatter.rs`, `ir.rs`, `mod.rs`,
`walker.rs`). The Python shim at `headroom/transforms/smart_crusher.py`
is a PyO3 wrapper — read it only for the public config surface
(`SmartCrusherConfig` field names/defaults), not the algorithm; the
algorithm lives in Rust.

Verified against squishi's current code (not assumed) before writing
this plan:

- `src/json_compress.rs` already does dedup + blind edge-cap + an
  invariant-disclosure marker (`crate::invariants`) that states verified
  facts (constant fields, ranges, dense-ID coverage) about what a
  *single contiguous* drop range hides. Any change to which indices get
  dropped has to keep feeding that machinery a contiguous range per gap,
  or its coverage claims go stale — see step 3.
- Only `src/main.rs` calls `compress_json_array` today; it owns the
  `source: &'static str` label in its `Output` struct (`"json"`,
  `"dedup+log"`, etc.). Governator's `cli/src/dispatch.rs:99` just
  prefixes that value (`format!("squishi:{}", squishi_result.source)`)
  — it is not the integration point. Step 5's wiring lands in
  `main.rs`, not governator.
- squishi has no CCR (blob-recovery) store. `pixel.rs` and `toon.rs`
  both note CCR registration as the *caller's* job, never squishi's own.
  headroom's lossless CSV-schema path leans on a CCR store for opaque
  cells (`compaction/ir.rs`'s `CellValue::OpaqueRef` — a hash pointer,
  not the original bytes). Without a store to point at, squishi can't
  reproduce that path — see step 4's scope cut.

## Scope decision (why this is its own project, not a same-session add)

Full fidelity would mean porting all 19 files: Shannon-entropy ID
detection, sequential-pattern detection (with a documented "BUG #2"
zero-padding fix), Pareto-based rare-value outlier detection (with a
documented "BUG #3" cardinality-cap fix), change-point preservation,
TOIN-learning hooks (not applicable — squishi has no learning-loop
equivalent, must be dropped, not stubbed), plus the whole compaction/
walker+IR+formatter subsystem. That's real multi-week work, not a
same-session addition. This plan scopes a v1 that captures the two
capabilities above without the statistical apparatus headroom built
them on, then leaves a documented extension point for later fidelity.

## Non-goals (v1)

- No TOIN / learning-loop port — squishi has no equivalent concept and
  no learning store; every headroom module that touches
  `use_feedback_hints` / `toin_confidence_threshold` gets read for the
  *selection logic* only, the learning call sites are dropped.
- No entropy-based or sequential-pattern field classification
  (`statistics.rs`) — real, useful, but a separate increment. v1's
  outlier scoring uses only `outliers.rs`'s two direct signals (rare
  structural fields, rare categorical values via the Pareto check) plus
  `error_keywords.rs`'s literal keyword scan — no entropy/UUID/
  sequential detection layered on top yet.
- No `factor_out_constants`, no `include_summaries`, no relevance-query
  biasing (`bias` param, `query_context`) — headroom features with no
  current squishi call site; adding them speculatively would violate
  this repo's own no-speculative-abstraction rule.
- CSV-schema is the only compaction format for v1 (drop
  markdown-kv). Headroom's own default is `csv-schema`; markdown-kv is
  an opt-in "trade tokens for model read accuracy" mode with no
  evidence squishi's callers need it yet.
- No opaque-cell (`CellValue::OpaqueRef`) substitution — headroom
  replaces base64/HTML/long-string cells with a CCR hash pointer,
  recoverable only because a CCR store exists downstream. squishi has
  no such store (confirmed: `pixel.rs`/`toon.rs` both treat CCR
  registration as the caller's job, never squishi's). v1 doesn't need
  the substitution at all: a long string cell renders verbatim in its
  CSV cell instead (`json_scalar_to_csv` already handles any string
  length), which keeps the render genuinely byte-lossless with no store
  required. This isn't a token-budget freebie — a large blob usually
  makes CSV-schema *bigger* than the minified-JSON source once the
  schema header and CSV quoting overhead are added — but that's caught
  by the existing ≥30%-savings gate (step 5), not by excluding the cell
  up front. The only cells v1's classifier actually excludes are
  non-scalar ones it has no rendering for at all: nested objects,
  arrays, and stringified-JSON (all three need the recursive `Nested`
  cell type this plan already cuts, see step 4).

## Steps

1. **`src/error_keywords.rs`** — literal port of headroom's
   `error_keywords.rs` (77 lines, a `const` keyword list) — smallest,
   zero-risk first step, unblocks step 2.

2. **`src/outliers.rs`** — port `detect_structural_outliers` and
   `detect_rare_status_values` (with the documented Bug #3 Pareto fix
   already applied upstream — port the fixed version, not the naive
   cap) from headroom's `outliers.rs`. Port
   `detect_error_items_for_preservation` using step 1's keyword list.
   Bring headroom's own unit tests over verbatim (already-written
   fixtures for the exact edge cases: uniform distributions, bimodal
   distributions, cardinality-above-cap) — don't re-derive test cases
   from scratch.

3. **Row-selection integration in `json_compress.rs`** — replace the
   current blind `seen[..keep_edge]` / `seen[len-keep_edge..]` slice.
   Two sub-problems the original plan left open, resolved here:

   - **Budget arbitration.** Step 2's detectors can flag more items than
     any sane budget (e.g. schema drift across a loosely-typed API
     response can flag most of the array as a "rare field" outlier).
     Cap total interest-flagged survivors at `keep_edge * 2` (same
     order of magnitude as the edge budget, so worst case total kept
     is bounded at 2x today's cap, not unbounded). When flagged items
     exceed the cap, prioritize `detect_error_items_for_preservation`
     hits first (most actionable), then `detect_rare_status_values`,
     then `detect_structural_outliers`'s rare-field signal, truncating
     within a tier by ascending original index for determinism.
   - **Non-contiguous drops.** Interest survivors can sit anywhere in
     the array, so the dropped set is no longer one contiguous middle
     slice — it's a set of indices with gaps. Keep array order stable
     (don't hoist interest items to the edges): compute
     `kept = edge_indices ∪ interest_indices`, walk the original array
     in order, and emit one marker per *contiguous run* of dropped
     indices rather than a single marker for the whole middle. Each
     run stays contiguous internally, so `build_marker` /
     `invariants::describe`'s coverage-range logic (which assumes a
     contiguous slice) keeps working unmodified per-run — it's called
     once per gap instead of once total. Add a test with two flagged
     items on opposite ends of a large array to prove two separate
     markers render, each with correct per-gap coverage facts.

4. **`src/compaction.rs`** — new module, scoped to CSV-schema only.
   `render_csv_schema(items: &[Value]) -> Option<String>`, ported from
   `compaction/compactor.rs` (schema building, `[N]{col:type,...}`
   column-frequency ordering) and `compaction/formatter.rs`
   (`CsvSchemaFormatter`'s actual row rendering). Declines (`None`,
   caller falls back to the lossy path) only for: fewer than 2 items, a
   non-object item, or any nested-object/array/stringified-JSON cell —
   the three shapes that need the recursive `Nested` cell type this v1
   doesn't have. Everything else renders, including a sparse table when
   keys don't line up across rows: headroom's own `core_ratio` check
   exists only to decide whether to *try* discriminator-bucketing a
   heterogeneous array, and headroom itself falls through to a sparse
   table when no clean discriminator is found rather than declining
   (confirmed by reading `compactor.rs`'s `compact()` and its own
   `stable_field_ordering` test, which exercises exactly this path).
   v1 has no bucketing, so the ratio has nothing left to gate — it goes
   straight to the sparse-table fallback headroom already uses. Skip
   `classifier.rs` too, not just `ir.rs`/`walker.rs`: v1 does no CCR
   substitution (see the updated non-goal above), so it has no use for
   opaque-vs-scalar sub-classification, only "is this cell a JSON
   scalar" — a one-line check, not the base64/HTML heuristics
   `classifier.rs` exists for.

5. **Wire into `compress_json_array`** — after dedup, before row
   selection: if the array classifies as tabular (step 4) AND
   CSV-schema rendering saves ≥30% bytes vs minified JSON (headroom's
   own `lossless_min_savings_ratio` default), return the CSV-schema
   rendering instead of the lossy drop-and-mark path. This is a
   genuinely different output shape (a string, not a JSON array) —
   `JsonCompressResult` needs a variant or a `rendering: CompactionKind`
   field. The actual caller that needs updating is `src/main.rs`'s
   `ContentKind::Json` match arm (line ~368) — it owns the `source`
   label squishi emits (`"json"`, `"dedup+log"`, etc.); governator's
   `dispatch.rs` only prefixes that value with `"squishi:"` and needs
   no changes of its own.

6. **Fixture-based parity spot-check** — not full byte-parity (v1
   intentionally diverges from headroom's exact algorithm), but pull 3-5
   of headroom's own test fixtures (bimodal status distribution, rare
   structural field, error-buried-in-middle) and confirm squishi's v1
   preserves the same rows headroom would, even though the underlying
   scoring differs.

7. **`cargo test` + `cargo clippy -D warnings` + `cargo fmt --check`**
   before calling this done — this repo's own bar, not a governator
   import.

## Later increments (not this plan)

- Entropy/sequential/UUID field classification (`statistics.rs`) for
  smarter outlier scoring.
- markdown-kv as a second compaction format.
- Change-point preservation (`anchors.rs`) — currently headroom's most
  sophisticated selection signal, deferred because it depends on the
  entropy/sequential classifiers from the deferred `statistics.rs`.
