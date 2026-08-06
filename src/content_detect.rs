//! Classify content by shape so the right compressor gets picked, instead
//! of always running the same fixed pipeline regardless of what's inside.
//! Same idea as headroom's ContentRouter detection step — own design.
//!
//! Combination strategy (deliberate, not arbitrary): the fast regex/parse
//! checks run first and are authoritative for Json/SearchResults/Log —
//! measured against real Magika output and found *more* precise for
//! these three: Magika labeled a single JSON object "jsonl" and had no
//! dedicated label for ad-hoc application logs at all (see
//! examples/probe_magika.rs). Magika only gets consulted for whatever
//! those checks don't confidently classify — where it's a genuine
//! improvement over the old blind "PlainText" catch-all (real labels:
//! rust, html, diff, csv, markdown, ...) at a real cost (~111ms model
//! load, only paid on this fallback path, never on the fast paths).

use magika::Session;
use regex::Regex;
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
        // Magika unavailable/failed — don't hard-fail detection over an
        // optional enrichment signal, fall back to the old behavior.
        None => ContentKind::PlainText,
    }
}

fn magika_label(content: &str) -> Option<String> {
    let mut session = Session::new().ok()?;
    let result = session.identify_content_sync(content.as_bytes()).ok()?;
    Some(result.info().label.to_string())
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
}
