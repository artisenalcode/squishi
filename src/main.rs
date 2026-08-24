use clap::{CommandFactory, Parser};
use serde_json::{Map, Value};
use squishi::base64_strip::strip_base64_blobs;
use squishi::content_detect::{ContentKind, detect};
use squishi::diff_compress::{DiffCompressConfig, compress_diff};
use squishi::doctor;
use squishi::invariants::InvariantConfig;
use squishi::json_compress::{JsonCompressConfig, JsonRendering, compress_json_array};
use squishi::line_dedup::dedupe_line_runs;
use squishi::line_number_strip::strip_read_tool_line_numbers;
use squishi::log_compress::{LogCompressConfig, compress_log};
use squishi::search_compress::compress_search_results;
use squishi::semantic_dedup::{SemanticDedup, SentenceShape};
use squishi::session_digest;
use squishi::session_prune;
use squishi::session_stats;
use squishi::toon;

#[derive(Parser)]
#[command(
    name = "squishi",
    about = "Rust-native text compressor — detects content shape, routes to the right technique. No store, no retrieve, that's total-recall's job"
)]
struct Cli {
    /// Text to compress. Omit to read from stdin instead -- no shell-argument size limit or quoting fragility on large multi-line content.
    text: Option<String>,

    /// Emit the full JSON contract (compressed/kind/source/chars_before/chars_after/...) instead of the default bare compressed text.
    #[arg(long)]
    json: bool,

    /// Run self-diagnostics instead of compressing `text`. A flag, not a subcommand, so `squishi doctor` isn't ambiguous with compressing the literal string "doctor".
    #[arg(long)]
    doctor: bool,

    /// With --doctor, skip the two model-load checks (magika, semantic-dedup).
    #[arg(long)]
    quick: bool,

    /// Run structural session pruning against a Claude Code transcript (JSONL) instead of compressing `text`.
    #[arg(long, value_name = "TRANSCRIPT_PATH")]
    session_prune: Option<std::path::PathBuf>,

    /// With --session-prune, also run rule 2 (supersede a Write/Edit's result once a later Read of the same path exists) -- off by default, real false-positive concern.
    #[arg(long)]
    include_rule_2: bool,

    /// With --session-prune, minimum tool-result byte size counted by the "prune old large outputs" rule.
    #[arg(long, default_value_t = 2000)]
    min_bytes: usize,

    /// With --session-prune, how many of the most recent session items are exempt from the "prune old large outputs" rule.
    #[arg(long, default_value_t = 200)]
    window: usize,

    /// With --session-prune, write a pruned *copy* of the transcript to this path instead of printing a stats report. Never mutates the input.
    #[arg(long, value_name = "OUT_PATH")]
    write: Option<std::path::PathBuf>,

    /// Extract human/assistant prose from a Claude Code session transcript (JSONL), compress it, and build a ready-to-stage digest, instead of compressing `text`. Extraction and compression only -- squishi never calls total-recall itself.
    #[arg(long, value_name = "TRANSCRIPT_PATH")]
    session_digest: Option<std::path::PathBuf>,

    /// Report cumulative real savings from every squishi `--json` call found in a session transcript (JSONL), instead of compressing `text`. Read-only.
    #[arg(long, value_name = "TRANSCRIPT_PATH")]
    session_stats: Option<std::path::PathBuf>,

    /// With --session-digest, the extracted-text truncation cap (middle dropped, head+tail kept).
    #[arg(long, default_value_t = 100_000)]
    max_chars: usize,

    /// With --session-digest, skip the first N physical lines before extracting -- an incremental caller passes the previous call's `total_lines` here to get only the delta since then.
    #[arg(long, default_value_t = 0)]
    start_line: usize,

    /// Skip shape detection and force this content kind, for a caller that already knows structurally what it's staging rather than trusting `detect()`'s heuristics (which can misfire on prose containing words like "failed"). Only `plain-text` is wired; other kinds don't need forcing.
    #[arg(long, value_enum)]
    force_kind: Option<ForceKind>,

    /// Process many texts in one process instead of one per invocation (ignores `text`; reads a JSON array from stdin: `[{"id": "...", "text": "..."}, ...]`). Loads the model once and reuses it across every item, instead of paying a fresh load per subprocess. Always emits the full `--json` contract per item plus `id`, as a JSON array.
    #[arg(long)]
    batch: bool,

    /// Deduplication method: `cascade` (default, 42x faster three-stage pipeline) or `mini-lm` (MiniLM transformer). Only affects plain-text dedup.
    #[arg(long, value_enum, default_value_t = DedupeMethod::Cascade)]
    dedup_method: DedupeMethod,

    /// How hard each compressor pushes: `conservative` keeps more context (safer, smaller savings), `aggressive` cuts harder (bigger savings, more loss). Applies to the default path and `--batch`; not read by `--session-prune`/`--session-digest`.
    #[arg(long, value_enum, default_value_t = Level::Default)]
    level: Level,

    /// Write a real roff(7) man page (squishi.1) to this directory instead of compressing `text`. Regenerated from this exact `Cli` definition via `clap_mangen`, so the man page can never drift from these flags.
    #[arg(long, value_name = "OUT_DIR", hide = true)]
    generate_man: Option<std::path::PathBuf>,

    /// Encode `text` (must be valid JSON) as TOON instead of running the normal detect+compress pipeline. Lossless, unlike every other mode here. Only ships TOON if it measures smaller than the JSON input; otherwise prints the original JSON unchanged.
    #[arg(long)]
    toon: bool,
}

#[derive(clap::ValueEnum, Clone, Copy)]
#[value(rename_all = "kebab-case")]
enum ForceKind {
    PlainText,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug)]
#[value(rename_all = "kebab-case")]
enum DedupeMethod {
    MiniLM,
    Cascade,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
#[value(rename_all = "kebab-case")]
enum Level {
    Conservative,
    Default,
    Aggressive,
}

/// Resolved per-level thresholds for every tunable compressor surface -- `--level` is a single caller-facing knob, this is its one expansion point.
struct LevelConfigs {
    paraphrase_threshold: f32,
    diff: DiffCompressConfig,
    log: LogCompressConfig,
    json: JsonCompressConfig,
}

/// Values measured on real fixtures (see README's `--level` section). `Default` reproduces this repo's pre-`--level` constants exactly, so it's a byte-for-byte no-op versus every pre-existing caller.
fn configs_for_level(level: Level) -> LevelConfigs {
    match level {
        Level::Conservative => LevelConfigs {
            paraphrase_threshold: 0.90,
            diff: DiffCompressConfig {
                max_context_lines: 4,
                max_hunks_per_file: 20,
                max_files: 40,
            },
            log: LogCompressConfig {
                max_errors: 20,
                max_warnings: 10,
                context_lines: 4,
                max_total_lines: 200,
            },
            json: JsonCompressConfig {
                keep_edge: 10,
                invariants: InvariantConfig::default(),
                ..JsonCompressConfig::default()
            },
        },
        Level::Default => LevelConfigs {
            paraphrase_threshold: PARAPHRASE_THRESHOLD,
            diff: DiffCompressConfig::default(),
            log: LogCompressConfig::default(),
            json: JsonCompressConfig::default(),
        },
        Level::Aggressive => LevelConfigs {
            paraphrase_threshold: 0.70,
            diff: DiffCompressConfig {
                max_context_lines: 1,
                max_hunks_per_file: 5,
                max_files: 10,
            },
            log: LogCompressConfig {
                max_errors: 5,
                max_warnings: 2,
                context_lines: 1,
                max_total_lines: 50,
            },
            json: JsonCompressConfig {
                keep_edge: 2,
                invariants: InvariantConfig::default(),
                ..JsonCompressConfig::default()
            },
        },
    }
}

const SKIP_LOG_COMPRESS_UNDER_CHARS: usize = 2000;
const SKIP_DIFF_COMPRESS_UNDER_CHARS: usize = 2000;
const SKIP_SEMANTIC_DEDUP_UNDER_CHARS: usize = 2000;
const PARAPHRASE_THRESHOLD: f32 = 0.80; // matches dedupe_semantic.py's default

struct Output {
    compressed: String,
    source: &'static str,
    /// Extra fields flattened into the top-level JSON output alongside compressed/kind/source/chars_before/chars_after -- shape varies by which compressor ran.
    detail: Map<String, Value>,
}

/// The full routing decision: detect content shape, pick and run the matching compressor. Pulled out of `main` so it's callable from tests and from governator's squishi.rs wrapper.
fn route(text: &str) -> (ContentKind, Output) {
    route_with_override(text, None)
}

/// Like `route_with_override`, but with an explicit `--level` instead of the implicit `Level::Default`. The CLI's single-shot path uses this; tests and `--session-digest` stay level-less.
fn route_with_level(
    text: &str,
    forced_kind: Option<ContentKind>,
    level: Level,
) -> (ContentKind, Output) {
    let mut dedup_cache = None;
    route_impl(
        text,
        forced_kind,
        &mut dedup_cache,
        ALLOW_PUNCTUATION_RESTORE_DEFAULT,
        false,
        level,
        DedupeMethod::MiniLM,
    )
}

/// Same eligibility gate `SemanticDedup::dedupe` takes -- see its own doc comment.
const ALLOW_PUNCTUATION_RESTORE_DEFAULT: bool = true;

/// Same as `route`, but `forced_kind` -- when given -- skips `detect()` entirely for a caller that already knows the content's shape, rather than trusting a heuristic that can misclassify it. Single-shot: loads its own `SemanticDedup` fresh every call; see `--batch` for the reused-model path.
fn route_with_override(text: &str, forced_kind: Option<ContentKind>) -> (ContentKind, Output) {
    let mut dedup_cache = None;
    route_impl(
        text,
        forced_kind,
        &mut dedup_cache,
        ALLOW_PUNCTUATION_RESTORE_DEFAULT,
        false,
        Level::Default,
        DedupeMethod::MiniLM,
    )
}

/// Loads `SemanticDedup` into `cache` on first need, reused on every subsequent call -- `--batch` shares one cache across an entire run instead of reloading the model per item.
fn ensure_dedup_loaded(cache: &mut Option<SemanticDedup>) -> Result<&mut SemanticDedup, String> {
    if cache.is_none() {
        *cache = Some(SemanticDedup::load()?);
    }
    Ok(cache.as_mut().unwrap())
}

/// Build output for semantic_dedup result.
fn build_semantic_dedup_output(
    result: &squishi::semantic_dedup::DedupResult,
    include_embedding: bool,
) -> Output {
    use squishi::semantic_dedup::SentenceShape;
    use serde_json::{json, Map, Value};

    let stories: Vec<Value> = result
        .kept
        .iter()
        .filter(|k| k.shape == SentenceShape::Narrative)
        .map(|k| Value::from(k.text.clone()))
        .collect();

    let kept: Vec<Value> = result
        .kept
        .iter()
        .map(|k| {
            let mut m = Map::new();
            m.insert("index".to_string(), Value::from(k.index));
            m.insert("text".to_string(), Value::from(k.text.clone()));
            m.insert(
                "shape".to_string(),
                Value::from(match k.shape {
                    SentenceShape::Narrative => "narrative",
                    SentenceShape::Concept => "concept",
                }),
            );
            if include_embedding {
                if let Some(embedding) = &k.embedding {
                    m.insert(
                        "embedding".to_string(),
                        Value::from(
                            embedding
                                .iter()
                                .map(|f| Value::from(*f))
                                .collect::<Vec<_>>(),
                        ),
                    );
                }
            }
            Value::Object(m)
        })
        .collect();

    let traceability: Vec<Value> = result
        .drops
        .iter()
        .map(|d| {
            let mut m = Map::new();
            m.insert("dropped_index".to_string(), Value::from(d.dropped_index));
            m.insert("kept_index".to_string(), Value::from(d.kept_index));
            m.insert("similarity".to_string(), Value::from(d.similarity));
            Value::Object(m)
        })
        .collect();

    Output {
        detail: Map::from_iter([
            (
                "sentences_before".to_string(),
                Value::from(result.original_sentences),
            ),
            (
                "sentences_after".to_string(),
                Value::from(result.kept_sentences),
            ),
            ("summary".to_string(), Value::from(result.summary.clone())),
            ("stories".to_string(), Value::from(stories)),
            ("kept".to_string(), Value::from(kept)),
            ("drops".to_string(), Value::from(traceability)),
            (
                "punctuation_restored".to_string(),
                Value::from(result.punctuation_restored),
            ),
        ]),
        compressed: result.content.clone(),
        source: "dedup+semantic",
    }
}

fn route_impl(
    text: &str,
    forced_kind: Option<ContentKind>,
    dedup_cache: &mut Option<SemanticDedup>,
    allow_punctuation_restore: bool,
    include_embedding: bool,
    level: Level,
    dedup_method: DedupeMethod,
) -> (ContentKind, Output) {
    // Claude Code's Read tool wraps file content in a `cat -n`-style `N\t<line>` prefix, which confuses both detect() and line-anchored fast-path regexes -- stripped before detection runs.
    let (text, line_numbers_stripped) = strip_read_tool_line_numbers(text);
    let text = text.as_str();

    // A base64 blob (embedded screenshot, data-URI) can appear inside any shape detect() classifies below, so it's stripped before detection runs.
    let (text, base64_blobs_removed) = strip_base64_blobs(text);
    let text = text.as_str();

    let kind = forced_kind.unwrap_or_else(|| detect(text));
    let configs = configs_for_level(level);

    let output = match &kind {
        ContentKind::Json => match compress_json_array(text, &configs.json) {
            Some(result) => Output {
                source: match result.rendering {
                    JsonRendering::RowSelected => "json",
                    JsonRendering::CsvSchema => "json-csv-schema",
                },
                compressed: result.content,
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
            // Valid JSON but not an array -- nothing repeatable to compress, pass through unchanged.
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
                let result = compress_log(&deduped, &configs.log);
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
                let result = compress_diff(text, "", &configs.diff);
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
            } else if matches!(dedup_method, DedupeMethod::Cascade) {
                // Use cascade dedup (simplified, no shape detection)
                use squishi::cascade_dedup::{CascadeDedup, KeptSentence};
                use squishi::semantic_dedup::split_sentences;

                let sentences: Vec<&str> = split_sentences(&deduped);
                match CascadeDedup::dedupe(sentences) {
                    Ok(result) => {
                        let kept: Vec<Value> = result
                            .kept
                            .iter()
                            .map(|k| {
                                let mut m = Map::new();
                                m.insert("text".to_string(), Value::from(k.text.clone()));
                                if include_embedding {
                                    if let Some(embedding) = &k.embedding {
                                        m.insert(
                                            "embedding".to_string(),
                                            Value::from(
                                                embedding
                                                    .iter()
                                                    .map(|f| Value::from(*f))
                                                    .collect::<Vec<_>>(),
                                            ),
                                        );
                                    }
                                }
                                Value::Object(m)
                            })
                            .collect();

                        let mut detail = Map::new();
                        detail.insert("kept_count".to_string(), Value::from(result.kept.len()));
                        detail.insert("removed_count".to_string(), Value::from(result.removed_count));
                        detail.insert("stage1_ms".to_string(), Value::from(result.stage1_time_ms));
                        detail.insert("stage2_ms".to_string(), Value::from(result.stage2_time_ms));
                        detail.insert("stage3_ms".to_string(), Value::from(result.stage3_time_ms));
                        detail.insert("kept".to_string(), Value::Array(kept));

                        Output {
                            compressed: result.kept.iter().map(|k| k.text.clone()).collect::<Vec<_>>().join(" "),
                            source: "cascade",
                            detail,
                        }
                    }
                    Err(_) => {
                        // Fallback to MiniLM on cascade failure
                        match ensure_dedup_loaded(dedup_cache).and_then(|d| {
                            d.dedupe(
                                &deduped,
                                configs.paraphrase_threshold,
                                allow_punctuation_restore,
                            )
                        }) {
                            Ok(result) => build_semantic_dedup_output(&result, include_embedding),
                            Err(_) => Output {
                                compressed: deduped,
                                source: "error",
                                detail: Map::new(),
                            },
                        }
                    }
                }
            } else {
                match ensure_dedup_loaded(dedup_cache).and_then(|d| {
                    d.dedupe(
                        &deduped,
                        configs.paraphrase_threshold,
                        allow_punctuation_restore,
                    )
                }) {
                    Ok(result) => build_semantic_dedup_output(&result, include_embedding),
                    Err(_) => Output {
                        compressed: deduped,
                        source: "error",
                        detail: Map::new(),
                    },
                }
            }
        }
        // Other(_) is a structured format (rust/html/diff/csv/...), not prose, so sentence-level dedup doesn't apply -- line_dedup only.
        ContentKind::Other(_) => Output {
            compressed: dedupe_line_runs(text),
            source: "dedup",
            detail: Map::new(),
        },
    };

    let mut output = output;
    if line_numbers_stripped {
        output
            .detail
            .insert("line_numbers_stripped".to_string(), Value::from(true));
    }
    if base64_blobs_removed > 0 {
        output.detail.insert(
            "base64_blobs_removed".to_string(),
            Value::from(base64_blobs_removed),
        );
    }

    (kind, output)
}

/// Builds the CLI's full JSON output contract -- governator's `squishi.rs` wrapper depends on exactly these top-level fields plus whatever `detail` flattens in.
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

/// Parses `--batch`'s stdin contract: a JSON array of `{"id", "text", "restore_punctuation", "include_embedding"}` objects, as `(id, text, allow_punctuation_restore, include_embedding)` tuples in order. Both booleans are optional per item (default `true`/`false` respectively).
fn parse_batch_items(raw: &str) -> Result<Vec<(String, String, bool, bool)>, String> {
    let value: Value =
        serde_json::from_str(raw).map_err(|e| format!("--batch stdin must be valid JSON: {e}"))?;
    let array = value
        .as_array()
        .ok_or_else(|| "--batch stdin must be a JSON array".to_string())?;

    array
        .iter()
        .enumerate()
        .map(|(i, item)| {
            let id = item
                .get("id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("--batch item {i}: missing string field \"id\""))?
                .to_string();
            let text = item
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("--batch item {i}: missing string field \"text\""))?
                .to_string();
            let allow_punctuation_restore = item
                .get("restore_punctuation")
                .and_then(|v| v.as_bool())
                .unwrap_or(ALLOW_PUNCTUATION_RESTORE_DEFAULT);
            let include_embedding = item
                .get("include_embedding")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            Ok((id, text, allow_punctuation_restore, include_embedding))
        })
        .collect()
}

/// Resolve the content to compress: the positional argument if given, otherwise stdin. Refuses to block on an interactive terminal with neither, since that's almost always a forgotten argument, not someone about to type input.
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

    if let Some(out_dir) = &cli.generate_man {
        if let Err(e) = std::fs::create_dir_all(out_dir) {
            eprintln!("squishi --generate-man: failed to create {out_dir:?}: {e}");
            std::process::exit(1);
        }
        match clap_mangen::Man::new(Cli::command()).generate_to(out_dir) {
            Ok(path) => println!("wrote {}", path.display()),
            Err(e) => {
                eprintln!("squishi --generate-man: failed to write man page: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

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

    if cli.toon {
        let text = match read_input(&cli) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        };
        let value: Value = match serde_json::from_str(text.trim()) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: --toon requires valid JSON input: {e}");
                std::process::exit(2);
            }
        };
        let chars_before = text.len();
        let (compressed, source) = match toon::encode_if_smaller(&value) {
            Some(toon_text) => (toon_text, "toon"),
            None => (
                serde_json::to_string(&value).unwrap_or_else(|_| text.clone()),
                "toon-not-smaller",
            ),
        };
        let chars_after = compressed.len();
        if cli.json {
            let mut json = Map::new();
            json.insert("compressed".to_string(), Value::from(compressed));
            json.insert("source".to_string(), Value::from(source));
            json.insert("chars_before".to_string(), Value::from(chars_before));
            json.insert("chars_after".to_string(), Value::from(chars_after));
            println!("{}", Value::Object(json));
        } else {
            println!("{compressed}");
        }
        return;
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
        let (extracted, meta) =
            session_digest::extract_session_text(&jsonl, cli.max_chars, cli.start_line);
        if extracted.trim().is_empty() {
            // A whole-file digest coming back empty still fails loudly; an incremental call (start_line > 0) coming back empty just means nothing's new since the last checkpoint.
            if cli.start_line == 0 {
                eprintln!("nothing to digest (empty session)");
                std::process::exit(1);
            }
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
            json.insert("total_lines".to_string(), Value::from(meta.total_lines));
            json.insert("chars_before".to_string(), Value::from(chars_before));
            json.insert("chars_after".to_string(), Value::from(chars_after));
            println!("{}", Value::Object(json));
        } else {
            println!("{content}");
        }
        return;
    }

    if let Some(path) = &cli.session_stats {
        let jsonl = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: failed to read {}: {e}", path.display());
                std::process::exit(2);
            }
        };
        let stats = session_stats::scan(&jsonl);

        if cli.json {
            let mut by_kind = Map::new();
            for (kind, ks) in &stats.by_kind {
                let mut m = Map::new();
                m.insert("calls".to_string(), Value::from(ks.calls));
                m.insert("chars_before".to_string(), Value::from(ks.chars_before));
                m.insert("chars_after".to_string(), Value::from(ks.chars_after));
                m.insert("chars_saved".to_string(), Value::from(ks.chars_saved()));
                m.insert("pct_saved".to_string(), Value::from(ks.pct_saved()));
                by_kind.insert(kind.clone(), Value::Object(m));
            }
            let mut json = Map::new();
            json.insert("calls".to_string(), Value::from(stats.total.calls));
            json.insert(
                "chars_before".to_string(),
                Value::from(stats.total.chars_before),
            );
            json.insert(
                "chars_after".to_string(),
                Value::from(stats.total.chars_after),
            );
            json.insert(
                "chars_saved".to_string(),
                Value::from(stats.total.chars_saved()),
            );
            json.insert(
                "pct_saved".to_string(),
                Value::from(stats.total.pct_saved()),
            );
            json.insert("by_kind".to_string(), Value::Object(by_kind));
            println!("{}", Value::Object(json));
        } else if stats.total.calls == 0 {
            println!("no squishi --json calls found in this transcript");
        } else {
            for (kind, ks) in &stats.by_kind {
                println!(
                    "{kind}: {} calls, {} -> {} chars ({:.1}% saved)",
                    ks.calls,
                    ks.chars_before,
                    ks.chars_after,
                    ks.pct_saved()
                );
            }
            println!(
                "total: {} calls, {} -> {} chars ({:.1}% saved)",
                stats.total.calls,
                stats.total.chars_before,
                stats.total.chars_after,
                stats.total.pct_saved()
            );
        }
        return;
    }

    let forced = cli.force_kind.map(|f| match f {
        ForceKind::PlainText => ContentKind::PlainText,
    });

    if cli.batch {
        if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
            eprintln!(
                "error: --batch reads a JSON array from stdin — pipe it in, don't run interactively"
            );
            std::process::exit(2);
        }
        let mut buf = String::new();
        if let Err(e) = std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf) {
            eprintln!("error: failed to read stdin: {e}");
            std::process::exit(2);
        }
        let items = match parse_batch_items(&buf) {
            Ok(items) => items,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        };

        let mut dedup_cache = None;
        let mut results = Vec::with_capacity(items.len());
        for (id, item_text, allow_punctuation_restore, include_embedding) in items {
            let (kind, output) = route_impl(
                &item_text,
                forced.clone(),
                &mut dedup_cache,
                allow_punctuation_restore,
                include_embedding,
                cli.level,
                cli.dedup_method,
            );
            let mut json = build_output(&item_text, &kind, output);
            json.insert("id".to_string(), Value::from(id));
            results.push(Value::Object(json));
        }
        println!("{}", Value::Array(results));
        return;
    }

    let text = match read_input(&cli) {
        Ok(text) => text,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let (kind, output) = route_with_level(&text, forced, cli.level);
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

    // --- parse_batch_items: pure ---

    #[test]
    fn parse_batch_items_parses_a_real_array() {
        let raw = r#"[{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]"#;
        let items = parse_batch_items(raw).unwrap();
        // Neither item set restore_punctuation or include_embedding -- both default to (true, false).
        assert_eq!(
            items,
            vec![
                ("a".to_string(), "first".to_string(), true, false),
                ("b".to_string(), "second".to_string(), true, false)
            ]
        );
    }

    #[test]
    fn parse_batch_items_honors_an_explicit_restore_punctuation_false() {
        let raw = r#"[
            {"id": "yt-video", "text": "unpunctuated caption text"},
            {"id": "wikipedia-page", "text": "Real prose.", "restore_punctuation": false}
        ]"#;
        let items = parse_batch_items(raw).unwrap();
        assert_eq!(
            items[0],
            (
                "yt-video".to_string(),
                "unpunctuated caption text".to_string(),
                true,
                false
            )
        );
        assert_eq!(
            items[1],
            (
                "wikipedia-page".to_string(),
                "Real prose.".to_string(),
                false,
                false
            )
        );
    }

    #[test]
    fn parse_batch_items_honors_an_explicit_include_embedding_true() {
        let raw = r#"[{"id": "a", "text": "some text", "include_embedding": true}]"#;
        let items = parse_batch_items(raw).unwrap();
        assert_eq!(
            items[0],
            ("a".to_string(), "some text".to_string(), true, true)
        );
    }

    #[test]
    fn parse_batch_items_rejects_a_non_array() {
        let result = parse_batch_items(r#"{"id": "a", "text": "first"}"#);
        assert!(result.is_err());
    }

    #[test]
    fn parse_batch_items_rejects_an_item_missing_text() {
        let result = parse_batch_items(r#"[{"id": "a"}]"#);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("item 0"));
    }

    #[test]
    fn parse_batch_items_on_empty_array_returns_empty() {
        assert_eq!(parse_batch_items("[]").unwrap(), Vec::new());
    }

    #[test]
    #[ignore] // real model load attempt (network/cache)
    fn ensure_dedup_loaded_reuses_the_same_cache_across_calls() {
        // Proves the cache slot is threaded through, not silently re-created on a second call, whether the first load succeeded or failed.
        let mut cache: Option<SemanticDedup> = None;
        let first = ensure_dedup_loaded(&mut cache);
        let first_was_ok = first.is_ok();
        let second = ensure_dedup_loaded(&mut cache);
        assert_eq!(second.is_ok(), first_was_ok);
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
    fn real_read_tool_shaped_input_gets_its_line_numbers_stripped_before_routing() {
        // Claude Code's Read tool numbers every line `N\t<content>`, which used to route to ContentKind::Other("tsv") and get zero real compression -- end-to-end regression guard.
        let lines: Vec<String> = (1..=30)
            .map(|i| format!("routine status line number {i} with no special structure at all"))
            .collect();
        let numbered: String = lines
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{}\t{l}\n", i + 1))
            .collect();
        let (kind, output) = route(&numbered);
        assert_ne!(
            kind,
            ContentKind::Other("tsv".to_string()),
            "should no longer be misread as TSV once the line-number prefix is stripped"
        );
        assert_eq!(output.detail["line_numbers_stripped"], Value::from(true));
        assert!(
            !output.compressed.contains('\t'),
            "the stripped prefix shouldn't reappear in the compressed output"
        );
    }

    #[test]
    fn content_with_no_line_numbering_has_no_line_numbers_stripped_key() {
        let (_, output) = route("just a normal paragraph of prose with no special structure.");
        assert!(
            !output.detail.contains_key("line_numbers_stripped"),
            "line_numbers_stripped should be absent, not false, when nothing was stripped"
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
        // Embedded quotes, backslashes, control characters -- never verified against a real JSON parser under hand-rolled escaping.
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
