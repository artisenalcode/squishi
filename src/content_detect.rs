//! Classify content by shape so the right compressor gets picked, instead
//! of always running the same fixed pipeline regardless of what's inside.
//! Same idea as headroom's ContentRouter detection step — own design.
//!
//! Combination strategy (deliberate, not arbitrary): the fast regex/parse
//! checks run first and are authoritative for Json/SearchResults/Log —
//! measured against real Magika output and found *more* precise for
//! these three: Magika labeled a single JSON object "jsonl" and had no
//! dedicated label for ad-hoc application logs at all (finding made with
//! the original `magika`-crate probe, since superseded by this module's
//! own tests below, which exercise the same real model). Magika only
//! gets consulted for whatever those checks don't confidently classify —
//! where it's a genuine improvement over the old blind "PlainText"
//! catch-all (real labels: rust, html, diff, csv, markdown, ...).
//!
//! 2026-08-18: ported off the official `magika` crate (and `ort`
//! underneath it) onto `candle-onnx` running the real, unmodified
//! `standard_v3_3` model — the same weights the official crate ships,
//! embedded straight into the binary (`assets/magika-standard_v3_3.onnx`,
//! sourced from the real `google/magika` repo, sha256
//! fe2d2eb49c5f88a9e0a6c048e15d6ffdf86235519c2afc535044de433169ec8c).
//! Motivation: `magika` 1.1.0 hard-pins an exact `ort` prerelease that
//! conflicts with what other tools in this workspace resolve to — see
//! docs/ideation/ort-dependency-consistency/2026-08-18-ort-pin-and-bottleneck-plan.md
//! for the full investigation. Dropping `ort`/`magika` entirely removes
//! that conflict at the source instead of chasing version numbers.
//!
//! candle-onnx 0.11.0 on crates.io can't run this graph as-is (it's
//! missing the Int32-input, Max, Reciprocal, and GlobalMaxPool ops the
//! model actually uses) — patched on a fork (see `Cargo.toml`'s
//! `[patch.crates-io]`) and verified against the real model before this
//! code was written: exact label match against the real magika CLI on
//! real files (rust source scores 0.9999454 at the "rust" row, a
//! markdown file scores 0.99917597 at "markdown", both well clear of
//! their real thresholds). The byte-feature-extraction algorithm below
//! and the per-label threshold/overwrite table in `magika_labels.rs` are
//! ported line-for-line from the real `magika` crate's own
//! `rust/lib/src/{input,file,model,content}.rs` (Apache-2.0) — not
//! guessed, not simplified, so classifications stay identical to what
//! the real magika CLI would report.

use crate::magika_labels::MAGIKA_LABELS;
use candle_core::{DType, Device, Tensor};
use candle_onnx::onnx::ModelProto;
use prost::Message;
use regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentKind {
    Json,
    SearchResults,
    Log,
    Diff,
    PlainText,
    /// Magika's raw label for content that isn't Json/SearchResults/Log
    /// and isn't generic prose either (e.g. "rust", "html", "diff",
    /// "csv", "markdown"). No dedicated compressor per label yet —
    /// callers currently treat this the same as PlainText for
    /// compression — but the real classification is preserved for
    /// observability and future routing.
    Other(String),
}

static SEARCH_RESULT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\S+:\d+:").unwrap());
static LOG_LEVEL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(error|warn(ing)?|fatal|fail(ed)?)\b").unwrap());
/// `diff --git`/`--combined`/`--cc` headers, or a naked `--- a/`/`--- /dev/null`
/// hunk-file marker (unified diff without the git wrapper, e.g. `diff -u`
/// output). Checked before Log: diff hunks routinely contain words like
/// "fail"/"error" in test code, which would otherwise misroute them.
static DIFF_HEADER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^diff --(git a/|combined |cc )").unwrap());
static NAKED_HUNK_OLD_FILE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^--- (a/.+|/dev/null)$").unwrap());
static HUNK_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^@@ -\d").unwrap());

/// Magika's label for generic, structureless text — mapped to
/// ContentKind::PlainText rather than Other("txt"), since that's exactly
/// what PlainText already means.
const MAGIKA_PLAIN_TEXT_LABEL: &str = "txt";

/// The real magika `standard_v3_3` model, embedded so classification
/// never needs a model download or a filesystem lookup at runtime.
const MODEL_BYTES: &[u8] = include_bytes!("../assets/magika-standard_v3_3.onnx");

/// Decoded once, reused across every `magika_label()` call in the
/// process — this is a strict improvement over the pre-migration code,
/// which rebuilt a whole `ort::Session` (the documented ~111ms cost) on
/// every single call. `None` if the embedded bytes ever fail to decode,
/// which should never happen for a byte-identical asset checked into
/// this repo — treated as "AI classification unavailable" rather than a
/// panic, matching how a missing/corrupt model was already handled
/// before this migration.
static MODEL: LazyLock<Option<ModelProto>> = LazyLock::new(|| ModelProto::decode(MODEL_BYTES).ok());

/// Real `config.min.json` constants for magika's `standard_v3_3` model —
/// see `rust/lib/src/config.rs`/`model.rs` in the real `magika` crate.
const BEG_SIZE: usize = 1024;
const END_SIZE: usize = 1024;
const BLOCK_SIZE: usize = 4096;
const PADDING_TOKEN: i64 = 256;
/// If the model still sees padding at this offset, the real content
/// captured at the front of the file is too short for magika to have
/// bothered training on — the real crate falls back to a rule (UTF-8 ->
/// Txt, else -> Unknown) rather than trusting the model. Content here is
/// always a Rust `&str`, so that rule always resolves to Txt, which is
/// exactly what `magika_label()` returning `None` already maps to below
/// — no separate rule path needed.
const MIN_FILE_SIZE_FOR_DL: usize = 8;

pub fn detect(content: &str) -> ContentKind {
    let trimmed = content.trim_start();

    if (trimmed.starts_with('[') || trimmed.starts_with('{'))
        && serde_json::from_str::<serde_json::Value>(trimmed).is_ok()
    {
        return ContentKind::Json;
    }

    let lines: Vec<&str> = content.lines().collect();
    if !lines.is_empty() {
        let search_lines = lines
            .iter()
            .filter(|l| SEARCH_RESULT_RE.is_match(l))
            .count();
        if search_lines * 2 >= lines.len() {
            return ContentKind::SearchResults;
        }

        let has_diff_header = lines.iter().any(|l| DIFF_HEADER_RE.is_match(l));
        let has_naked_hunk = lines.iter().any(|l| NAKED_HUNK_OLD_FILE_RE.is_match(l))
            && lines.iter().any(|l| HUNK_MARKER_RE.is_match(l));
        if has_diff_header || has_naked_hunk {
            return ContentKind::Diff;
        }

        let log_lines = lines.iter().filter(|l| LOG_LEVEL_RE.is_match(l)).count();
        if log_lines > 0 {
            return ContentKind::Log;
        }
    }

    match magika_label(content) {
        Some(label) if label == MAGIKA_PLAIN_TEXT_LABEL => ContentKind::PlainText,
        Some(label) => ContentKind::Other(label),
        // Magika unavailable/failed, or too little content to trust the
        // model on — don't hard-fail detection over an optional
        // enrichment signal, fall back to the old behavior.
        None => ContentKind::PlainText,
    }
}

fn magika_label(content: &str) -> Option<String> {
    let model = MODEL.as_ref()?;
    let features = extract_features(content.as_bytes())?;
    let input = Tensor::from_vec(features, (1, BEG_SIZE + END_SIZE), &Device::Cpu).ok()?;
    let inputs = HashMap::from([("bytes".to_string(), input)]);
    let outputs = candle_onnx::simple_eval(model, inputs).ok()?;
    let target = outputs.get("target_label")?;
    let scores: Vec<f32> = target
        .to_dtype(DType::F32)
        .ok()?
        .flatten_all()
        .ok()?
        .to_vec1()
        .ok()?;

    // Same tie-breaking as the real crate's `FileType::convert`: on a
    // tie, the later index wins (`scores[best].max(x) == x` is true for
    // x >= scores[best]).
    let mut best = 0usize;
    for (i, &x) in scores.iter().enumerate() {
        if scores[best].max(x) == x {
            best = i;
        }
    }
    let entry = MAGIKA_LABELS.get(best)?;
    let label = if scores[best] < entry.threshold {
        entry.low_confidence_label
    } else {
        entry.overwrite_label.unwrap_or(entry.label)
    };
    Some(label.to_string())
}

/// Real byte-feature extraction, ported from the magika crate's
/// `rust/lib/src/input.rs::extract_features_async` (synchronous here —
/// content is always already in memory, no file I/O to overlap).
fn extract_features(content: &[u8]) -> Option<Vec<i64>> {
    let file_len = content.len();
    if file_len == 0 {
        return None;
    }
    let buffer_size = BLOCK_SIZE.min(file_len);
    let beg = strip_prefix(&content[..buffer_size]);
    let end = strip_suffix(&content[file_len - buffer_size..]);

    let mut features = vec![PADDING_TOKEN; BEG_SIZE + END_SIZE];
    let (beg_slice, end_slice) = features.split_at_mut(BEG_SIZE);
    copy_features(beg_slice, beg, false);
    copy_features(end_slice, end, true);

    if features[MIN_FILE_SIZE_FOR_DL - 1] == PADDING_TOKEN {
        return None;
    }
    Some(features)
}

/// `right_align == false`: content lands at the start of `dst`, padding
/// trails at the end (used for the file's beginning). `right_align ==
/// true`: content lands at the end of `dst`, padding leads at the start
/// (used for the file's end) — matches `input.rs::copy_features`'s
/// `align` parameter (0 vs 1) exactly.
fn copy_features(dst: &mut [i64], src: &[u8], right_align: bool) {
    let len = dst.len().min(src.len());
    let dst_start = if right_align { dst.len() - len } else { 0 };
    let src_start = if right_align { src.len() - len } else { 0 };
    for (d, s) in dst[dst_start..dst_start + len]
        .iter_mut()
        .zip(&src[src_start..src_start + len])
    {
        *d = *s as i64;
    }
}

fn is_whitespace(b: u8) -> bool {
    b.is_ascii_whitespace() || b == 0x0b
}

fn strip_prefix(xs: &[u8]) -> &[u8] {
    let mut i = 0;
    while i < xs.len() && is_whitespace(xs[i]) {
        i += 1;
    }
    &xs[i..]
}

fn strip_suffix(xs: &[u8]) -> &[u8] {
    let mut j = xs.len();
    while j > 0 && is_whitespace(xs[j - 1]) {
        j -= 1;
    }
    &xs[..j]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_json_array() {
        assert_eq!(detect(r#"[{"a": 1}, {"a": 2}]"#), ContentKind::Json);
    }

    #[test]
    fn detects_json_object() {
        assert_eq!(detect(r#"{"key": "value"}"#), ContentKind::Json);
    }

    #[test]
    fn detects_search_results() {
        let content = "src/main.rs:10:fn main() {\nsrc/lib.rs:5:pub fn foo() {}\n";
        assert_eq!(detect(content), ContentKind::SearchResults);
    }

    #[test]
    fn detects_log_output() {
        let content = "starting up\nERROR: connection refused\nretrying\n";
        assert_eq!(detect(content), ContentKind::Log);
    }

    #[test]
    fn malformed_json_like_content_falls_through() {
        // starts with { but isn't valid JSON — must not misclassify.
        // Also not log/search-shaped, so this exercises the Magika
        // fallback path for real.
        let content = "{ this is not json, just a sentence with a brace }";
        let kind = detect(content);
        assert!(
            kind == ContentKind::PlainText || matches!(kind, ContentKind::Other(_)),
            "expected PlainText or Other(_), got {kind:?}"
        );
    }

    #[test]
    fn falls_back_to_plain_text_for_generic_prose() {
        let content = "just a normal paragraph of prose with no special structure.";
        assert_eq!(detect(content), ContentKind::PlainText);
    }

    #[test]
    fn classifies_rust_source_via_magika() {
        let content = "fn main() {\n    let x = 5;\n    println!(\"{}\", x);\n}\n";
        assert_eq!(detect(content), ContentKind::Other("rust".to_string()));
    }

    #[test]
    fn classifies_html_via_magika() {
        let content = "<html><head><title>Test</title></head><body><h1>Hi</h1></body></html>";
        assert_eq!(detect(content), ContentKind::Other("html".to_string()));
    }

    #[test]
    fn detects_git_diff_via_fast_regex() {
        // The fast regex tier now catches this before it ever reaches
        // Magika — diffs are a stronger, cheaper-to-check signal than a
        // probabilistic classifier, same reasoning as Json/SearchResults/
        // Log above.
        let content = "diff --git a/src/main.rs b/src/main.rs\nindex abc..def 100644\n--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1,3 +1,3 @@\n-let x = 5;\n+let x = 10;\n";
        assert_eq!(detect(content), ContentKind::Diff);
    }

    #[test]
    fn detects_naked_hunk_without_git_header() {
        let content =
            "--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,2 @@\n-old line\n+new line\n context\n";
        assert_eq!(detect(content), ContentKind::Diff);
    }

    #[test]
    fn diff_containing_error_keywords_is_not_misrouted_to_log() {
        let content = "diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n@@ -1,2 +1,2 @@\n-def test_fail():\n+def test_error_handling():\n";
        assert_eq!(detect(content), ContentKind::Diff);
    }

    #[test]
    fn very_short_content_falls_back_to_plain_text() {
        // Fewer than MIN_FILE_SIZE_FOR_DL real bytes -- magika's own rule
        // path, not a model call.
        assert_eq!(detect("hi"), ContentKind::PlainText);
    }
}
