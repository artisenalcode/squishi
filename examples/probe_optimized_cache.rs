//! Honest probe: does ort's `with_optimized_model_path` (cache the
//! post-optimization graph, skip re-optimizing on the next cold start)
//! actually help — even for a small model like Magika's (3.1MB), not
//! just the hypothesis that it would? Uses the real model.onnx magika
//! embeds (same file, read directly from the crate's cached source
//! rather than through magika's own Builder, which doesn't expose this
//! ort option).

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path: PathBuf = env::args()
        .nth(1)
        .expect("usage: probe_optimized_cache <path-to-model.onnx>")
        .into();

    let optimized_path = env::temp_dir().join("squishi-probe-optimized.onnx");
    let _ = std::fs::remove_file(&optimized_path); // clean slate

    // --- Baseline: default optimization, load from the original file ---
    let t0 = Instant::now();
    let mut builder = Session::builder()?;
    let _session = builder.commit_from_file(&model_path)?;
    println!(
        "baseline (default optimization, from original file): {:?}",
        t0.elapsed()
    );

    // --- First run with with_optimized_model_path set: performs
    // optimization AND serializes the result to optimized_path ---
    let t1 = Instant::now();
    let mut builder = Session::builder()?;
    builder = builder.with_optimized_model_path(&optimized_path)?;
    let _session = builder.commit_from_file(&model_path)?;
    println!(
        "first run with with_optimized_model_path set (optimizes + saves): {:?}",
        t1.elapsed()
    );
    println!(
        "optimized file written: {} ({} bytes)",
        optimized_path.exists(),
        std::fs::metadata(&optimized_path)
            .map(|m| m.len())
            .unwrap_or(0)
    );

    // --- Second run: load the ALREADY-optimized file directly, with
    // optimization disabled since it's already been done ---
    let t2 = Instant::now();
    let mut builder = Session::builder()?;
    builder = builder.with_optimization_level(GraphOptimizationLevel::Disable)?;
    let _session = builder.commit_from_file(&optimized_path)?;
    println!(
        "loading the pre-optimized graph directly (opt disabled): {:?}",
        t2.elapsed()
    );

    Ok(())
}
