//! Classify content by shape so the right compressor gets picked, instead of always running the same fixed pipeline.
//!
//! The fast regex/parse checks run first and are authoritative for Json/SearchResults/Log -- measured more precise than Magika for these three (Magika labeled a single JSON object "jsonl" and has no dedicated log label at all). Magika only gets consulted for whatever those checks don't confidently classify.
//!
//! Runs on `candle-onnx` with the real, unmodified `standard_v3_3` model (embedded in `assets/magika-standard_v3_3.onnx`, same weights the official `magika` crate ships), not the official crate itself -- `magika` 1.1.0 hard-pins an `ort` prerelease that conflicts with this workspace's other tools. candle-onnx 0.11.0 can't run this graph as-is (missing ops), patched on a fork (see `Cargo.toml`'s `[patch.crates-io]`) and verified against the real magika CLI: exact label match on real files, well clear of thresholds. The feature-extraction algorithm and `magika_labels.rs`'s threshold table are ported line-for-line from magika's own Rust source (Apache-2.0), not guessed.

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
    /// Magika's raw label for content that isn't Json/SearchResults/Log or generic prose (e.g. "rust", "html", "diff"). No dedicated compressor yet -- callers treat it like PlainText, but the real label is preserved for observability.
    Other(String),
}

static SEARCH_RESULT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\S+:\d+:").unwrap());
static LOG_LEVEL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(error|warn(ing)?|fatal|fail(ed)?)\b").unwrap());
/// `diff --git`/`--combined`/`--cc` headers, or a naked `--- a/` hunk-file marker. Checked before Log: diff hunks routinely contain words like "fail"/"error" in test code.
static DIFF_HEADER_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^diff --(git a/|combined |cc )").unwrap());
static NAKED_HUNK_OLD_FILE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^--- (a/.+|/dev/null)$").unwrap());
static HUNK_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^@@ -\d").unwrap());

/// Magika's label for generic, structureless text -- mapped to ContentKind::PlainText rather than Other("txt"), since that's exactly what PlainText means.
const MAGIKA_PLAIN_TEXT_LABEL: &str = "txt";

/// The real magika `standard_v3_3` model, embedded so classification never needs a download or filesystem lookup at runtime.
const MODEL_BYTES: &[u8] = include_bytes!("../assets/magika-standard_v3_3.onnx");

/// Decoded once, reused across every call. `None` if the embedded bytes ever fail to decode (shouldn't happen for a checked-in asset) -- treated as "AI classification unavailable", not a panic.
static MODEL: LazyLock<Option<ModelProto>> = LazyLock::new(|| ModelProto::decode(MODEL_BYTES).ok());

/// `config.min.json` constants for magika's `standard_v3_3` model.
const BEG_SIZE: usize = 1024;
const END_SIZE: usize = 1024;
const BLOCK_SIZE: usize = 4096;
const PADDING_TOKEN: i64 = 256;
/// If the model still sees padding at this offset, content is too short for magika's model to be trusted -- the real crate falls back to a UTF-8/Unknown rule, which for a Rust `&str` always resolves to Txt, exactly what `magika_label()` returning `None` already maps to.
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
        // Magika unavailable/failed or content too short -- don't hard-fail an optional enrichment signal.
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

    // Same tie-breaking as the real crate: on a tie, the later index wins.
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

/// Byte-feature extraction, ported from magika's `input.rs::extract_features_async` (synchronous here since content is always already in memory).
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

/// `right_align == false`: content at the start of `dst`, padding trails (file's beginning). `right_align == true`: content at the end, padding leads (file's end) -- matches magika's own `align` parameter.
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
        // Starts with { but isn't valid JSON, and isn't log/search-shaped -- exercises the Magika fallback path.
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
        // The fast regex tier catches this before it ever reaches Magika -- a stronger, cheaper signal than a probabilistic classifier.
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
        // Fewer than MIN_FILE_SIZE_FOR_DL bytes -- magika's rule path, not a model call.
        assert_eq!(detect("hi"), ContentKind::PlainText);
    }
}
