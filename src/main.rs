use clap::Parser;
use squishi::content_detect::{ContentKind, detect};
use squishi::json_compress::compress_json_array;
use squishi::line_dedup::dedupe_line_runs;
use squishi::log_compress::{LogCompressConfig, compress_log};
use squishi::search_compress::compress_search_results;

// kompress (src/kompress.rs) is deliberately NOT wired in here. It's real,
// tested, and correctness-verified (word-level keep/drop, matches
// headroom's algorithm), but every invocation pays a ~7-18s ONNX
// session-load cost with no state to reuse between one-shot CLI calls —
// not worth triggering an unconditional model download for every
// PlainText compress until there's an actual use case that justifies
// that cost (a daemon architecture would amortize it; nothing here needs
// that yet). Revisit if/when that use case shows up.

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

struct Output {
    compressed: String,
    source: &'static str,
    detail: String,
}

fn main() {
    let cli = Cli::parse();
    let kind = detect(&cli.text);

    let output = match kind {
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
        ContentKind::PlainText => Output {
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
