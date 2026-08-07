//! One-off probe: confirm the real ONNX I/O contract for the
//! punctuation-restoration model before writing any integration code —
//! same discipline as `probe_single_load.rs`/`semantic_dedup.rs`'s own
//! module doc ("confirmed via examples probe, not assumed"). Downloads
//! via hf_hub, same as the real module will.

use hf_hub::api::sync::Api;
use ort::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model("ldenoue/fullstop-punctuation-multilang-large".to_string());

    println!("fetching model.onnx (small, graph-only)...");
    let small = repo.get("onnx/model.onnx")?;
    println!(
        "  -> {} ({} bytes)",
        small.display(),
        std::fs::metadata(&small)?.len()
    );

    println!("fetching model_quantized.onnx (real weights)...");
    let quantized = repo.get("onnx/model_quantized.onnx")?;
    println!(
        "  -> {} ({} bytes)",
        quantized.display(),
        std::fs::metadata(&quantized)?.len()
    );

    println!("loading session from model_quantized.onnx...");
    let session = Session::builder()?.commit_from_file(&quantized)?;

    println!("\n=== inputs ===");
    for input in session.inputs() {
        println!("  {} : {:?}", input.name(), input);
    }
    println!("\n=== outputs ===");
    for output in session.outputs() {
        println!("  {} : {:?}", output.name(), output);
    }

    Ok(())
}
