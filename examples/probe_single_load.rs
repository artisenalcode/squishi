//! Single, isolated model load — for a fair fresh-process comparison
//! (squishi is a one-shot CLI, so same-process warmup effects between
//! multiple loads in one run don't reflect real usage). Run this binary
//! twice, as two separate processes: once against the original model
//! with default optimization, once against a pre-optimized file with
//! optimization disabled.

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path: PathBuf = env::args()
        .nth(1)
        .expect("usage: probe_single_load <path-to-model.onnx> [--skip-optimization]")
        .into();
    let skip_optimization = env::args().any(|a| a == "--skip-optimization");

    let t0 = Instant::now();
    let mut builder = Session::builder()?;
    if skip_optimization {
        builder = builder.with_optimization_level(GraphOptimizationLevel::Disable)?;
    }
    let _session = builder.commit_from_file(&model_path)?;
    println!(
        "load ({}): {:?}",
        if skip_optimization {
            "optimization disabled, pre-optimized file"
        } else {
            "default optimization"
        },
        t0.elapsed()
    );

    Ok(())
}
