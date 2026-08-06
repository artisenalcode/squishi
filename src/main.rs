use clap::Parser;
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
    detail: String,
}

fn main() {
    let cli = Cli::parse();
    let kind = detect(&cli.text);

    let output = match &kind {
        ContentKind::Json => match compress_json_array(&cli.text) {
            Some(result) => Output {
                compressed: result.content,
                source: "json",
                detail: format!(
                    "\"elements_before\":{},\"elements_after\":{}",
                    result.original_elements, result.compressed_elements
                ),
            },
            // Valid JSON but not an array (e.g. a single object) —
            // nothing repeatable to compress, pass through unchanged.
            None => Output {
                compressed: cli.text.clone(),
                source: "json-passthrough",
                detail: String::new(),
            },
        },
        ContentKind::SearchResults => {
            let result = compress_search_results(&cli.text);
            Output {
                compressed: result.content,
                source: "search",
                detail: format!(
                    "\"lines_before\":{},\"lines_after\":{}",
                    result.original_lines, result.compressed_lines
                ),
            }
        }
        ContentKind::Log => {
            let deduped = dedupe_line_runs(&cli.text);
            if deduped.len() <= SKIP_LOG_COMPRESS_UNDER_CHARS {
                Output {
                    compressed: deduped,
                    source: "dedup",
                    detail: String::new(),
                }
            } else {
                let result = compress_log(&deduped, &LogCompressConfig::default());
                Output {
                    detail: format!(
                        "\"lines_before\":{},\"lines_after\":{}",
                        result.original_line_count, result.compressed_line_count
                    ),
                    compressed: result.content,
                    source: "dedup+log",
                }
            }
        }
        ContentKind::Diff => {
            if cli.text.len() <= SKIP_DIFF_COMPRESS_UNDER_CHARS {
                Output {
                    compressed: cli.text.clone(),
                    source: "passthrough",
                    detail: String::new(),
                }
            } else {
                let result = compress_diff(&cli.text, "", &DiffCompressConfig::default());
                Output {
                    detail: format!(
                        "\"files_affected\":{},\"hunks_removed\":{}",
                        result.files_affected, result.hunks_removed
                    ),
                    compressed: result.content,
                    source: "diff",
                }
            }
        }
        ContentKind::PlainText => {
            let deduped = dedupe_line_runs(&cli.text);
            if deduped.len() <= SKIP_SEMANTIC_DEDUP_UNDER_CHARS {
                Output {
                    compressed: deduped,
                    source: "dedup",
                    detail: String::new(),
                }
            } else {
                match SemanticDedup::load()
                    .and_then(|mut d| d.dedupe(&deduped, PARAPHRASE_THRESHOLD))
                {
                    Ok(result) => Output {
                        detail: format!(
                            "\"sentences_before\":{},\"sentences_after\":{}",
                            result.original_sentences, result.kept_sentences
                        ),
                        compressed: result.content,
                        source: "dedup+semantic",
                    },
                    // Model unavailable (offline, first-run download
                    // failed) — line-dedup's result is still real
                    // compression, use it rather than failing outright.
                    Err(e) => Output {
                        compressed: deduped,
                        source: "dedup-semantic-unavailable",
                        detail: format!("\"semantic_error\":{e:?}"),
                    },
                }
            }
        }
        // Other(_) carries Magika's real classification (rust/html/diff/
        // csv/markdown/...) — structured formats, not prose, so sentence-
        // level paraphrase dedup doesn't apply; line_dedup only.
        ContentKind::Other(_) => Output {
            compressed: dedupe_line_runs(&cli.text),
            source: "dedup",
            detail: String::new(),
        },
    };

    let detail_json = if output.detail.is_empty() {
        String::new()
    } else {
        format!(",{}", output.detail)
    };

    println!(
        "{{\"compressed\":{:?},\"kind\":{:?},\"source\":{:?},\"chars_before\":{},\"chars_after\":{}{}}}",
        output.compressed,
        format!("{:?}", kind),
        output.source,
        cli.text.len(),
        output.compressed.len(),
        detail_json
    );
}
