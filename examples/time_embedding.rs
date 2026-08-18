//! One-off timing harness: load `SemanticDedup`, run it on a real
//! transcript file, print load time / total `dedupe()` time / per-stage
//! `embed_batch` breakdown. Sibling to `time_punctuation.rs`, same real
//! transcript as input — see docs/ideation/ort-dependency-consistency/
//! 2026-08-19-candle-cpu-profiling-plan.md's Step 1.
//!
//! Deliberately calls the real, full `dedupe()` (not a bare
//! embed-only path) with `allow_punctuation_restore: true` — this is
//! genuinely how production calls it on an unpunctuated transcript
//! (see main.rs's own call site), so the reported `dedupe time` also
//! includes real punctuation restoration and the greedy keep/drop
//! loop, not just embedding. `stage_timings()` isolates just the
//! `embed_batch` sub-stages (tokenize / build tensors / forward /
//! postprocess) from that larger real call, which is the number this
//! investigation actually cares about.

use squishi::semantic_dedup::SemanticDedup;
use std::env;
use std::fs;
use std::time::Instant;

/// Matches main.rs's own `Level::Default` (`PARAPHRASE_THRESHOLD`'s
/// real value there) — not independently chosen here.
const PARAPHRASE_THRESHOLD: f32 = 0.80;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).expect("usage: time_embedding <file>");
    let raw = fs::read_to_string(&path)?;
    // Same real-content shape as time_punctuation.rs: strip YAML
    // frontmatter and the leading '# Title' line.
    let content = raw
        .splitn(3, "---\n")
        .nth(2)
        .unwrap_or(&raw)
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let word_count = content.split_whitespace().count();
    println!("file: {path}");
    println!("word count: {word_count}");

    let load_start = Instant::now();
    let mut dedup = SemanticDedup::load()?;
    println!("load time: {:?}", load_start.elapsed());

    let dedupe_start = Instant::now();
    let result = dedup.dedupe(&content, PARAPHRASE_THRESHOLD, true)?;
    let elapsed = dedupe_start.elapsed();
    println!(
        "dedupe time: {elapsed:?} (real full call: punctuation restore + embed + greedy keep/drop, NOT embedding alone)"
    );
    println!(
        "sentences: {} original, {} kept, {} dropped",
        result.original_sentences,
        result.kept_sentences,
        result.original_sentences - result.kept_sentences
    );
    println!(
        "punctuation restored first: {}",
        result.punctuation_restored
    );

    let stages = dedup.stage_timings();
    println!("\n--- embed_batch per-stage breakdown (summed across all batches) ---");
    println!("tokenize:      {:?}", stages.tokenize);
    println!("build_tensors: {:?}", stages.build_tensors);
    println!("forward:       {:?}", stages.forward);
    println!("postprocess:   {:?}", stages.postprocess);
    println!(
        "stage total:   {:?} ({:.1}% of dedupe time)",
        stages.total(),
        100.0 * stages.total().as_secs_f64() / elapsed.as_secs_f64()
    );

    Ok(())
}
