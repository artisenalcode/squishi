use clap::Parser;
use serde_json::{Map, Value};
use squishi::base64_strip::strip_base64_blobs;
use squishi::content_detect::{ContentKind, detect};
use squishi::diff_compress::{DiffCompressConfig, compress_diff};
use squishi::doctor;
use squishi::json_compress::compress_json_array;
use squishi::line_dedup::dedupe_line_runs;
use squishi::log_compress::{LogCompressConfig, compress_log};
use squishi::search_compress::compress_search_results;
use squishi::semantic_dedup::SemanticDedup;
use squishi::session_digest;
use squishi::session_prune;

#[derive(Parser)]
#[command(
    name = "squishi",
    about = "Rust-native text compressor — detects content shape, routes to the right technique. No store, no retrieve, that's total-recall's job"
)]
struct Cli {
    /// Text to compress. Omit to read from stdin instead — the harness-
    /// agnostic path: no shell-argument size limit, no quoting/escaping
    /// fragility on large multi-line content (a big diff, log, or file
    /// read), unlike passing content as a positional argument.
    text: Option<String>,

    /// Emit the full JSON contract (compressed/kind/source/chars_before/
    /// chars_after/...) instead of the default bare compressed text.
    /// governator and other programmatic consumers want this; a harness
    /// piping content through for its own use generally just wants the
    /// compressed text back.
    #[arg(long)]
    json: bool,

    /// Run self-diagnostics instead of compressing `text` (ignored when
    /// set). Not a subcommand: squishi's Cli has no Commands enum, and
    /// `squishi doctor` would be ambiguous with compressing the literal
    /// string "doctor" — a flag has no such collision.
    #[arg(long)]
    doctor: bool,

    /// With --doctor, skip the two expensive model-load checks (magika,
    /// semantic-dedup). Ignored without --doctor.
    #[arg(long)]
    quick: bool,

    /// Run structural session pruning against a Claude Code transcript
    /// (JSONL) instead of compressing `text` (ignored when set). Same
    /// "flag not subcommand" reasoning as --doctor.
    #[arg(long, value_name = "TRANSCRIPT_PATH")]
    session_prune: Option<std::path::PathBuf>,

    /// With --session-prune, also run rule 2 (supersede a Write/Edit's
    /// result once a later Read of the same path exists) — off by
    /// default per the technical-board review's real false-positive
    /// concern (2026-08-07). Ignored without --session-prune.
    #[arg(long)]
    include_rule_2: bool,

    /// With --session-prune, minimum tool-result byte size counted by
    /// the "prune old large outputs" rule.
    #[arg(long, default_value_t = 2000)]
    min_bytes: usize,

    /// With --session-prune, how many of the most recent session items
    /// are exempt from the "prune old large outputs" rule.
    #[arg(long, default_value_t = 200)]
    window: usize,

    /// With --session-prune, write a pruned *copy* of the transcript to
    /// this path instead of printing a stats report. Never mutates the
    /// input transcript.
    #[arg(long, value_name = "OUT_PATH")]
    write: Option<std::path::PathBuf>,

    /// Extract human/assistant prose from a Claude Code session
    /// transcript (JSONL), compress it, and build a ready-to-stage
    /// digest — instead of compressing `text` (ignored when set). Same
    /// "flag not subcommand" reasoning as --doctor. Extraction and
    /// compression only; squishi never calls total-recall itself
    /// (`trm ingest-session` is the caller that stages the result).
    #[arg(long, value_name = "TRANSCRIPT_PATH")]
    session_digest: Option<std::path::PathBuf>,

    /// With --session-digest, the extracted-text truncation cap (middle
    /// dropped, head+tail kept), matching session_to_trm.py's default.
    #[arg(long, default_value_t = 100_000)]
    max_chars: usize,
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
    // Unconditional pre-pass, not a ContentKind of its own — a base64 blob
    // (an embedded screenshot, a data-URI) can appear inside any shape
    // detect() classifies below, so it's stripped before detection runs,
    // the same way MCE's Layer1Pruner runs ahead of its shape-aware
    // routing (audited 2026-08-07, see docs/plan-2026-08-07-base64-strip.md).
    let (text, base64_blobs_removed) = strip_base64_blobs(text);
    let text = text.as_str();

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

    let mut output = output;
    if base64_blobs_removed > 0 {
        output.detail.insert(
            "base64_blobs_removed".to_string(),
            Value::from(base64_blobs_removed),
        );
    }

    (kind, output)
}

/// Builds the CLI's full JSON output contract — governator's
/// `squishi.rs` wrapper depends on exactly these top-level fields (plus
/// whatever `detail` flattens in). A plain `Map<String, Value>`, not a
/// `#[derive(Serialize)]` struct: real `Value` types (not string
/// formatting) already give correct escaping on their own — a derive
/// macro buys nothing here beyond what this one shared function already
/// gets by existing, and it's not worth a proc-macro dependency for a
/// secondary, opt-in output path (`--json`; the default is bare text).
fn build_output(text: &str, kind: &ContentKind, output: Output) -> Map<String, Value> {
    let chars_after = output.compressed.len();
    let mut json = Map::new();
    json.insert("compressed".to_string(), Value::from(output.compressed));
    json.insert("kind".to_string(), Value::from(format!("{kind:?}")));
    json.insert("source".to_string(), Value::from(output.source));
    json.insert("chars_before".to_string(), Value::from(text.len()));
    json.insert("chars_after".to_string(), Value::from(chars_after));
    json.extend(output.detail);
    json
}

/// Resolve the content to compress: the positional argument if given,
/// otherwise stdin. Refuses to block reading from an interactive
/// terminal with neither — that's almost always a forgotten argument,
/// not someone about to type input, and hanging silently is the worst
/// failure mode for a tool meant to sit in an automated pipeline.
fn read_input(cli: &Cli) -> Result<String, String> {
    if let Some(text) = &cli.text {
        return Ok(text.clone());
    }

    if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return Err(
            "no text argument given and stdin is a terminal (not a pipe) — \
             pass text as an argument or pipe content in, e.g. `cat file | squishi`"
                .to_string(),
        );
    }

    let mut buf = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)
        .map_err(|e| format!("failed to read stdin: {e}"))?;
    Ok(buf)
}

fn main() {
    let cli = Cli::parse();

    if cli.doctor {
        let report = doctor::run(cli.quick);
        if cli.json {
            println!("{}", report.to_json());
        } else {
            for check in &report.checks {
                let label = match check.status {
                    doctor::Status::Pass => "PASS",
                    doctor::Status::Warn => "WARN",
                    doctor::Status::Fail => "FAIL",
                    doctor::Status::Skipped => "SKIP",
                };
                println!("{label}  {}: {}", check.name, check.message);
            }
        }
        std::process::exit(if report.has_failures() { 1 } else { 0 });
    }

    if let Some(path) = &cli.session_prune {
        let jsonl = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: failed to read {}: {e}", path.display());
                std::process::exit(2);
            }
        };
        let items = session_prune::parse(&jsonl);
        let candidates = session_prune::run(&items, cli.include_rule_2, cli.min_bytes, cli.window);

        if let Some(out_path) = &cli.write {
            let pruned = session_prune::apply_pruning(&jsonl, &candidates);
            if let Err(e) = std::fs::write(out_path, &pruned) {
                eprintln!("error: failed to write {}: {e}", out_path.display());
                std::process::exit(2);
            }
        }

        if cli.json {
            let mut by_rule: Map<String, Value> = Map::new();
            for c in &candidates {
                let entry = by_rule
                    .entry(c.rule.to_string())
                    .or_insert_with(|| Value::from(0));
                *entry = Value::from(entry.as_i64().unwrap_or(0) + 1);
            }
            let mut json = Map::new();
            json.insert("candidates".to_string(), Value::from(candidates.len()));
            json.insert(
                "bytes_prunable".to_string(),
                Value::from(candidates.iter().map(|c| c.bytes).sum::<usize>()),
            );
            json.insert("by_rule".to_string(), Value::Object(by_rule));
            println!("{}", Value::Object(json));
        } else {
            let mut by_rule: std::collections::BTreeMap<&str, (usize, usize)> =
                std::collections::BTreeMap::new();
            for c in &candidates {
                let entry = by_rule.entry(c.rule).or_default();
                entry.0 += 1;
                entry.1 += c.bytes;
            }
            for (rule, (count, bytes)) in &by_rule {
                println!("{rule}: {count} candidates, {bytes} bytes prunable");
            }
            let total_bytes: usize = candidates.iter().map(|c| c.bytes).sum();
            println!(
                "total: {} candidates, {total_bytes} bytes prunable",
                candidates.len()
            );
        }
        return;
    }

    if let Some(path) = &cli.session_digest {
        let jsonl = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: failed to read {}: {e}", path.display());
                std::process::exit(2);
            }
        };
        let (extracted, meta) = session_digest::extract_session_text(&jsonl, cli.max_chars);
        if extracted.trim().is_empty() {
            eprintln!("nothing to digest (empty session)");
            std::process::exit(1);
        }
        let chars_before = extracted.len();
        let (_, compress_output) = route(&extracted);
        let compressed = compress_output.compressed;
        let chars_after = compressed.len();
        let content = session_digest::build_digest_content(&compressed, &meta);

        if cli.json {
            let mut json = Map::new();
            json.insert("content".to_string(), Value::from(content));
            json.insert(
                "session_id".to_string(),
                meta.session_id.map(Value::from).unwrap_or(Value::Null),
            );
            json.insert(
                "cwd".to_string(),
                meta.cwd.map(Value::from).unwrap_or(Value::Null),
            );
            json.insert(
                "first_ts".to_string(),
                meta.first_ts.map(Value::from).unwrap_or(Value::Null),
            );
            json.insert(
                "last_ts".to_string(),
                meta.last_ts.map(Value::from).unwrap_or(Value::Null),
            );
            json.insert("turn_count".to_string(), Value::from(meta.turn_count));
            json.insert("truncated".to_string(), Value::from(meta.truncated));
            json.insert("raw_bytes".to_string(), Value::from(meta.raw_bytes));
            json.insert("chars_before".to_string(), Value::from(chars_before));
            json.insert("chars_after".to_string(), Value::from(chars_after));
            println!("{}", Value::Object(json));
        } else {
            println!("{content}");
        }
        return;
    }

    let text = match read_input(&cli) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    let (kind, output) = route(&text);
    let json = build_output(&text, &kind, output);

    if cli.json {
        println!("{}", Value::Object(json));
    } else {
        println!("{}", json["compressed"].as_str().unwrap_or_default());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn a_base64_blob_inside_json_is_stripped_and_stays_valid_json() {
        let blob = "A".repeat(500);
        let input = format!(r#"{{"image": "{blob}", "name": "test"}}"#);
        let (kind, output) = route(&input);
        assert_eq!(kind, ContentKind::Json);
        assert_eq!(output.detail["base64_blobs_removed"], Value::from(1));
        let parsed: Value = serde_json::from_str(&output.compressed)
            .expect("compressed output should still be valid JSON");
        assert_eq!(parsed["name"], "test");
        assert!(parsed["image"].as_str().unwrap().contains("squishi pruned"));
    }

    #[test]
    fn a_base64_blob_in_plain_text_is_reported_in_json_output() {
        let blob = "A".repeat(500);
        let input = format!("here is a blob: {blob} end");
        let (kind, output) = route(&input);
        assert_eq!(kind, ContentKind::PlainText);
        assert_eq!(output.detail["base64_blobs_removed"], Value::from(1));
        assert!(!output.compressed.contains(&blob));
    }

    #[test]
    fn content_with_no_base64_has_no_base64_blobs_removed_key() {
        let (_, output) = route("just a normal paragraph of prose with no special structure.");
        assert!(
            !output.detail.contains_key("base64_blobs_removed"),
            "base64_blobs_removed should be absent, not zero, when nothing was stripped"
        );
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
        let text = r#"[{"a":1},{"a":1}]"#;
        let (kind, output) = route(text);
        let cli_output = build_output(text, &kind, output);
        let serialized = serde_json::to_string(&cli_output).unwrap();
        let value: Value = serde_json::from_str(&serialized).expect("must be valid JSON");

        assert_eq!(value["kind"], "Json");
        assert_eq!(value["source"], "json");
        assert!(value["chars_before"].is_u64());
        assert!(value["elements_before"].is_u64());
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
        let (kind, output) = route(input);
        let cli_output = build_output(input, &kind, output);
        let serialized = serde_json::to_string(&cli_output).unwrap();
        let reparsed: Value = serde_json::from_str(&serialized).expect("must be valid JSON");
        assert!(reparsed["compressed"].as_str().unwrap().contains("quoted"));
    }
}
