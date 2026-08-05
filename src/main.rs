mod line_dedup;
mod log_compress;

use clap::Parser;
use line_dedup::dedupe_line_runs;
use log_compress::{LogCompressConfig, compress_log};

#[derive(Parser)]
#[command(
    name = "squishi",
    about = "Rust-native text compressor — no store, no retrieve, that's total-recall's job"
)]
struct Cli {
    /// Text to compress.
    text: String,
}

const SKIP_LOG_COMPRESS_UNDER_CHARS: usize = 2000;

fn main() {
    let cli = Cli::parse();

    let deduped = dedupe_line_runs(&cli.text);

    if deduped.len() <= SKIP_LOG_COMPRESS_UNDER_CHARS {
        println!(
            "{{\"compressed\":{:?},\"source\":\"dedup\",\"chars_before\":{},\"chars_after\":{}}}",
            deduped,
            cli.text.len(),
            deduped.len()
        );
        return;
    }

    let result = compress_log(&deduped, &LogCompressConfig::default());
    println!(
        "{{\"compressed\":{:?},\"source\":\"dedup+log\",\"chars_before\":{},\"chars_after\":{},\"lines_before\":{},\"lines_after\":{}}}",
        result.content,
        cli.text.len(),
        result.content.len(),
        result.original_line_count,
        result.compressed_line_count
    );
}
