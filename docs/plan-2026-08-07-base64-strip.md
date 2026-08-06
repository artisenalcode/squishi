# Plan: deterministic base64-blob stripping (option A from 2026-08-07 /think)

## Goal

Give squishi a zero-model, always-on pre-pass that strips base64-encoded blobs
(inline data-URIs and standalone runs) from input text before shape detection
and compression run — closing the real gap found auditing MCE's `Layer1Pruner`:
squishi currently has no base64 handling at all, so a tool response containing
an embedded base64 image (a screenshot API result, a data-URI in HTML/JSON)
passes through nearly unchanged.

## Context

- `src/main.rs::route(text: &str) -> (ContentKind, Output)` (line 52) is the
  single entry point: calls `detect(text)`, matches on `ContentKind`, runs the
  matching compressor. This is where the new pre-pass plugs in — before
  `detect()` runs, not as a new `ContentKind` (a base64 blob can appear
  *inside* JSON, logs, diffs, or plain text; it's an orthogonal concern to
  shape, the same way MCE's `Layer1Pruner` runs unconditionally before its
  Layer 2 semantic router, not as a shape category of its own).
- `src/main.rs::main()` calls `build_output(&text, &kind, output)` with the
  **original**, pre-strip `text` for `chars_before` — that stays unchanged;
  only `route()`'s internals need to work on the stripped text.
- Existing marker-string convention to match exactly (`src/line_dedup.rs:30`):
  `"[... squishi pruned: {} identical lines collapsed ...]\n"`. New marker
  should read the same family: `"[... squishi pruned: base64 blob removed,
  {n} chars ...]"`.
- Existing skip-threshold convention (`src/main.rs:32-34`,
  `SKIP_LOG_COMPRESS_UNDER_CHARS` etc.) — a `MIN_BLOB_CHARS` constant for what
  counts as "long enough to be worth stripping" fits the same pattern.
- Rust's `regex` crate (already a dependency, used in `content_detect.rs`)
  **does not support lookaround** (`(?<!...)`, `(?!...)`), unlike the Python
  `re` MCE's own pattern uses. Not needed here anyway: requiring a minimum
  length (`{100,}`) already makes greedy matching consume the maximal
  contiguous base64-alphabet run on its own — lookaround in MCE's version is
  a Python-idiom belt-and-suspenders, not a structural requirement.
- Real, honest risk (same one MCE's own design accepts, not unique to
  squishi): a long dense alphanumeric run without base64 padding — a commit
  SHA concatenation, a JWT, minified code with no whitespace — can false-match
  a standalone base64 pattern that isn't actually a blob. Requiring `=`
  padding would cut this risk but also miss real base64 whose length happens
  to be a multiple of 4 (no padding). MCE accepts the same tradeoff
  (`{0,2}` padding, optional). Plan keeps that choice, flagged in Risks
  and gated on a real measurement in Validation, not assumed safe.

## Steps

1. **New module `src/base64_strip.rs`.** `pub fn strip_base64_blobs(text: &str)
   -> (String, usize)` — returns `(stripped_text, blobs_removed_count)`. Two
   patterns, matching MCE's shape:
   - data-URI form: `data:[a-zA-Z0-9/+.-]+;base64,[A-Za-z0-9+/\n]{20,}={0,2}`
   - standalone form: `[A-Za-z0-9+/]{100,}={0,2}` (a `MIN_BLOB_CHARS = 100`
     constant, not a magic number)
   Apply data-URI pattern first (more specific, higher-confidence), then the
   standalone pattern on the result — replace each match with
   `[... squishi pruned: base64 blob removed, {n} chars ...]` (n = matched
   blob's original length), track total match count across both passes.
   *Check*: unit tests — a real data-URI (`data:image/png;base64,iVBORw0KG...`)
   gets replaced with one marker; a standalone 200-char base64 run gets
   replaced; short base64-looking strings (under 100 chars) are left alone;
   content with no base64 at all is returned byte-identical with count 0.

2. **False-positive calibration, real not assumed.** Before wiring into
   `route()`, run `strip_base64_blobs` against: (a) a real base64 image blob
   (data-URI form, from an actual screenshot/tool-response fixture — not
   synthetic), (b) a real long non-base64 token that could false-match — a
   real JWT, a real commit-hash-dense `git log --oneline` chunk, a real
   minified-JS sample. Report what actually happens, don't assume the regex
   is precise enough — this is the step that decides whether the standalone
   pattern's `{0,2}` optional padding is acceptable as-is or needs tightening
   (e.g. requiring `=` padding, or raising `MIN_BLOB_CHARS`) before it ships.
   *Check*: real output pasted for both cases, not inferred from reading the
   regex.

3. **Wire into `route()`** (`src/main.rs:52`): first line becomes
   `let (text, base64_removed) = base64_strip::strip_base64_blobs(text);` —
   shadow the incoming `&str` with the stripped `String`, so `detect()` and
   every match arm below operate on the stripped version unchanged otherwise.
   After the `match &kind { ... }` block produces `output`, if
   `base64_removed > 0`, insert `"base64_blobs_removed"` into
   `output.detail` (matching the existing per-kind detail-field convention —
   `elements_before/after`, `lines_before/after`, etc.).
   *Check*: existing `route()`-level tests (`json_array_routes_to_json_compressor`,
   `plain_text_under_threshold_only_dedups`, etc., `src/main.rs:266-351`)
   still pass unmodified — this step must not change behavior for content
   with no base64 in it, only add to it.

4. **New `route()`-level tests** for the base64 path specifically: a JSON
   object containing a base64 data-URI string value survives as valid JSON
   with the blob replaced (proves the marker text doesn't break JSON string
   syntax); a plain-text blob gets stripped and `base64_blobs_removed`
   appears in `--json` output; content with no base64 has no
   `base64_blobs_removed` key at all (not a zero value — matches existing
   convention of only including kind-relevant detail fields).
   *Check*: `cargo test`, new tests green, `--json` output inspected for the
   exact field shape.

5. **README update** — add base64-stripping to the "What it does today" list
   (matching the existing per-compressor bullet format), noting it's an
   unconditional pre-pass, not a `ContentKind`, and citing the MCE-audit
   origin briefly (this project's own convention — every design choice in
   this README cites where it came from and why).

## Validation

- `cargo test` (fast tier — no model involved, this pass is pure regex/string
  work, unlike `semantic_dedup`).
- `cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`.
- Step 2's real calibration output, pasted — the actual gate on whether this
  ships as designed or needs the padding/length tightened first.

## Risks

- **False positives on non-base64 dense alphanumeric runs** (see Context) —
  gated on step 2's real measurement, not shipped on assumption.
- **Marker text inside a JSON string value stays syntactically valid JSON**
  only because the marker itself contains no unescaped quotes/backslashes —
  true today (`[... squishi pruned: base64 blob removed, N chars ...]` has
  neither), but worth a literal round-trip test (step 4) rather than trusting
  the reasoning alone.
- **Ordering interacts with `detect()`**: stripping before detection changes
  what `detect()` sees — generally an improvement (Magika isn't confused by
  base64 noise), but a document that's *mostly* base64 with a small amount of
  real structure could reclassify differently post-strip than pre-strip.
  Not expected to be a real problem (the whole point is the base64 was noise,
  not signal) but worth being aware of if a detection-routing test ever fails
  unexpectedly after this change.
