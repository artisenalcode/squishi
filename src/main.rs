use clap::Parser;
use serde_json::{Map, Value};
use squishi::content_detect::{ContentKind, detect};
use squishi::diff_compress::{DiffCompressConfig, compress_diff};
use squishi::json_compress::compress_json_array;
use squishi::line_dedup::dedupe_line_runs;
use squishi::log_compress::{LogCompressConfig, compress_log};
use squishi::search_compress::compress_search_results;
use squishi::semantic_dedup::SemanticDedup;

#[derive(Parser)]
#[command(
    name = "squishi",
    about = "Rust-native text compressor — detects content shape, routes to the right technique. No store, no retrieve, that's total-recall's job"
)]
struct Cli {
    /// Text to compress.
    text: String,
}

const SKIP_LOG_COMPRESS_UNDER_CHARS: usize = 2000;
const SKIP_DIFF_COMPRESS_UNDER_CHARS: usize = 2000;
const SKIP_SEMANTIC_DEDUP_UNDER_CHARS: usize = 2000;
const PARAPHRASE_THRESHOLD: f32 = 0.80; // matches dedupe_semantic.py's default

struct Output {
    compressed: String,
    source: &'static str,
    /// Extra fields flattened into the top-level JSON output alongside
    /// compressed/kind/source/chars_before/chars_after — shape varies by
    /// which compressor ran (elements_before/after, lines_before/after,
    /// files_affected/hunks_removed, sentences_before/after, ...).
    detail: Map<String, Value>,
}

/// The full routing decision: detect content shape, pick and run the
/// matching compressor. Pulled out of `main` so it's callable from tests —
/// this is the actual logic governator's squishi.rs wrapper depends on,
/// and it had zero direct test coverage before this existed only inline
/// in `main()`.
fn route(text: &str) -> (ContentKind, Output) {
    let kind = detect(text);

    let output = match &kind {
        ContentKind::Json => match compress_json_array(text) {
            Some(result) => Output {
                compressed: result.content,
                source: "json",
                detail: Map::from_iter([
                    (
                        "elements_before".to_string(),
                        Value::from(result.original_elements),
                    ),
                    (
                        "elements_after".to_string(),
                        Value::from(result.compressed_elements),
                    ),
                ]),
            },
            // Valid JSON but not an array (e.g. a single object) —
            // nothing repeatable to compress, pass through unchanged.
            None => Output {
                compressed: text.to_string(),
                source: "json-passthrough",
                detail: Map::new(),
            },
        },
        ContentKind::SearchResults => {
            let result = compress_search_results(text);
            Output {
                compressed: result.content,
                source: "search",
                detail: Map::from_iter([
                    (
                        "lines_before".to_string(),
                        Value::from(result.original_lines),
                    ),
                    (
                        "lines_after".to_string(),
                        Value::from(result.compressed_lines),
                    ),
                ]),
            }
        }
        ContentKind::Log => {
            let deduped = dedupe_line_runs(text);
            if deduped.len() <= SKIP_LOG_COMPRESS_UNDER_CHARS {
                Output {
                    compressed: deduped,
                    source: "dedup",
                    detail: Map::new(),
                }
            } else {
                let result = compress_log(&deduped, &LogCompressConfig::default());
                Output {
                    detail: Map::from_iter([
                        (
                            "lines_before".to_string(),
                            Value::from(result.original_line_count),
                        ),
                        (
                            "lines_after".to_string(),
                            Value::from(result.compressed_line_count),
                        ),
                    ]),
                    compressed: result.content,
                    source: "dedup+log",
                }
            }
        }
        ContentKind::Diff => {
            if text.len() <= SKIP_DIFF_COMPRESS_UNDER_CHARS {
                Output {
                    compressed: text.to_string(),
                    source: "passthrough",
                    detail: Map::new(),
                }
            } else {
                let result = compress_diff(text, "", &DiffCompressConfig::default());
                Output {
                    detail: Map::from_iter([
                        (
                            "files_affected".to_string(),
                            Value::from(result.files_affected),
                        ),
                        (
                            "hunks_removed".to_string(),
                            Value::from(result.hunks_removed),
                        ),
                    ]),
                    compressed: result.content,
                    source: "diff",
                }
            }
        }
        ContentKind::PlainText => {
            let deduped = dedupe_line_runs(text);
            if deduped.len() <= SKIP_SEMANTIC_DEDUP_UNDER_CHARS {
                Output {
                    compressed: deduped,
                    source: "dedup",
                    detail: Map::new(),
                }
            } else {
                match SemanticDedup::load()
                    .and_then(|mut d| d.dedupe(&deduped, PARAPHRASE_THRESHOLD))
                {
                    Ok(result) => Output {
                        detail: Map::from_iter([
                            (
                                "sentences_before".to_string(),
                                Value::from(result.original_sentences),
                            ),
                            (
                                "sentences_after".to_string(),
                                Value::from(result.kept_sentences),
                            ),
                        ]),
                        compressed: result.content,
                        source: "dedup+semantic",
                    },
                    // Model unavailable (offline, first-run download
                    // failed) — line-dedup's result is still real
                    // compression, use it rather than failing outright.
                    Err(e) => Output {
                        compressed: deduped,
                        source: "dedup-semantic-unavailable",
                        detail: Map::from_iter([("semantic_error".to_string(), Value::from(e))]),
                    },
                }
            }
        }
        // Other(_) carries Magika's real classification (rust/html/diff/
        // csv/markdown/...) — structured formats, not prose, so sentence-
        // level paraphrase dedup doesn't apply; line_dedup only.
        ContentKind::Other(_) => Output {
            compressed: dedupe_line_runs(text),
            source: "dedup",
            detail: Map::new(),
        },
    };

    (kind, output)
}

fn main() {
    let cli = Cli::parse();
    let (kind, output) = route(&cli.text);

    let mut json = Map::new();
    json.insert(
        "compressed".to_string(),
        Value::from(output.compressed.clone()),
    );
    json.insert("kind".to_string(), Value::from(format!("{kind:?}")));
    json.insert("source".to_string(), Value::from(output.source));
    json.insert("chars_before".to_string(), Value::from(cli.text.len()));
    json.insert(
        "chars_after".to_string(),
        Value::from(output.compressed.len()),
    );
    for (k, v) in output.detail {
        json.insert(k, v);
    }

    println!("{}", Value::Object(json));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detail_str<'a>(v: &'a Value, key: &str) -> &'a str {
        v.get(key).and_then(Value::as_str).unwrap()
    }

    #[test]
    fn json_array_routes_to_json_compressor() {
        let (kind, output) = route(r#"[{"a":1},{"a":1},{"a":1}]"#);
        assert_eq!(kind, ContentKind::Json);
        assert_eq!(output.source, "json");
        assert!(output.detail.contains_key("elements_before"));
    }

    #[test]
    fn json_object_passes_through_unchanged() {
        let input = r#"{"key":"value"}"#;
        let (kind, output) = route(input);
        assert_eq!(kind, ContentKind::Json);
        assert_eq!(output.source, "json-passthrough");
        assert_eq!(output.compressed, input);
        assert!(output.detail.is_empty());
    }

    #[test]
    fn search_results_route_to_search_compressor() {
        let input = "src/main.rs:10:fn main() {\nsrc/lib.rs:5:pub fn foo() {}\n";
        let (kind, output) = route(input);
        assert_eq!(kind, ContentKind::SearchResults);
        assert_eq!(output.source, "search");
    }

    #[test]
    fn log_under_threshold_only_dedups() {
        let input = "starting up\nERROR: connection refused\nretrying\n";
        let (kind, output) = route(input);
        assert_eq!(kind, ContentKind::Log);
        assert_eq!(output.source, "dedup");
        assert!(output.detail.is_empty());
    }

    #[test]
    fn log_over_threshold_runs_log_compressor() {
        let mut input = String::new();
        for i in 0..120 {
            input.push_str(&format!("ERROR: failure number {i}\n"));
        }
        assert!(input.len() > SKIP_LOG_COMPRESS_UNDER_CHARS);
        let (kind, output) = route(&input);
        assert_eq!(kind, ContentKind::Log);
        assert_eq!(output.source, "dedup+log");
        assert!(output.detail.contains_key("lines_before"));
        assert!(output.detail.contains_key("lines_after"));
    }

    #[test]
    fn diff_under_threshold_passes_through() {
        let input = "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b";
        assert!(input.len() <= SKIP_DIFF_COMPRESS_UNDER_CHARS);
        let (kind, output) = route(input);
        assert_eq!(kind, ContentKind::Diff);
        assert_eq!(output.source, "passthrough");
        assert_eq!(output.compressed, input);
    }

    #[test]
    fn diff_over_threshold_runs_diff_compressor() {
        let mut input = String::from("diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n");
        for i in 0..40 {
            let start = i * 100 + 1;
            input.push_str(&format!("@@ -{0},6 +{0},6 @@\n", start));
            input.push_str(&format!(
                " ctx_a_{i}\n ctx_b_{i}\n-old_{i}\n+new_{i}\n ctx_c_{i}\n ctx_d_{i}\n"
            ));
        }
        assert!(input.len() > SKIP_DIFF_COMPRESS_UNDER_CHARS);
        let (kind, output) = route(&input);
        assert_eq!(kind, ContentKind::Diff);
        assert_eq!(output.source, "diff");
        assert!(output.detail.contains_key("files_affected"));
        assert!(output.detail.contains_key("hunks_removed"));
    }

    #[test]
    fn plain_text_under_threshold_only_dedups() {
        let input = "just a short paragraph of prose with no special structure.";
        let (kind, output) = route(input);
        assert_eq!(kind, ContentKind::PlainText);
        assert_eq!(output.source, "dedup");
    }

    #[test]
    fn other_content_kind_only_dedups() {
        let input = "fn main() {\n    let x = 5;\n    println!(\"{}\", x);\n}\n";
        let (kind, output) = route(input);
        assert!(matches!(kind, ContentKind::Other(_)));
        assert_eq!(output.source, "dedup");
    }

    #[test]
    fn top_level_output_is_valid_json_with_expected_fields() {
        let (kind, output) = route(r#"[{"a":1},{"a":1}]"#);
        let mut json = Map::new();
        json.insert(
            "compressed".to_string(),
            Value::from(output.compressed.clone()),
        );
        json.insert("kind".to_string(), Value::from(format!("{kind:?}")));
        json.insert("source".to_string(), Value::from(output.source));
        json.insert("chars_before".to_string(), Value::from(2usize));
        json.insert(
            "chars_after".to_string(),
            Value::from(output.compressed.len()),
        );
        for (k, v) in output.detail {
            json.insert(k, v);
        }
        let value = Value::Object(json);
        assert_eq!(detail_str(&value, "kind"), "Json");
        assert_eq!(detail_str(&value, "source"), "json");
        assert!(value.get("chars_before").unwrap().is_u64());
    }

    #[test]
    fn adversarial_content_survives_json_round_trip() {
        // Embedded quotes, backslashes, control characters — the exact
        // class of input hand-rolled `format!("{:?}", ...)` escaping was
        // never verified against a real JSON parser for.
        let input = "line one\n\"quoted\"\tand a \\backslash\\ and more prose to clear \
            the plain-text threshold so dedup runs and this string round-trips through \
            the actual compression path rather than a trivial passthrough, which is the \
            realistic case this test needs to cover for real adversarial content handling.";
        let (_, output) = route(input);
        let mut json = Map::new();
        json.insert("compressed".to_string(), Value::from(output.compressed));
        let serialized = Value::Object(json).to_string();
        let reparsed: Value = serde_json::from_str(&serialized).expect("must be valid JSON");
        assert!(reparsed["compressed"].as_str().unwrap().contains("quoted"));
    }
}
