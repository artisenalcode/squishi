//! Classify content by shape so the right compressor gets picked, instead
//! of always running the same fixed pipeline regardless of what's inside.
//! Same idea as headroom's ContentRouter detection step — own design.

use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentKind {
    Json,
    SearchResults,
    Log,
    PlainText,
}

static SEARCH_RESULT_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\S+:\d+:").unwrap());
static LOG_LEVEL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(error|warn(ing)?|fatal|fail(ed)?)\b").unwrap());

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

        let log_lines = lines.iter().filter(|l| LOG_LEVEL_RE.is_match(l)).count();
        if log_lines > 0 {
            return ContentKind::Log;
        }
    }

    ContentKind::PlainText
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
    fn falls_back_to_plain_text() {
        let content = "just a normal paragraph of prose with no special structure.";
        assert_eq!(detect(content), ContentKind::PlainText);
    }

    #[test]
    fn malformed_json_like_content_falls_through() {
        // starts with { but isn't valid JSON — must not misclassify.
        let content = "{ this is not json, just a sentence with a brace }";
        assert_eq!(detect(content), ContentKind::PlainText);
    }
}
