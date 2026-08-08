//! One-off probe: confirm the real ONNX I/O contract for
//! `ldenoue/distilbert-punctuator` before wiring it in — same discipline
//! as `probe_punctuation.rs` for the current large model.

use hf_hub::api::sync::Api;
use ort::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model("ldenoue/distilbert-punctuator".to_string());

    println!("fetching config.json...");
    let config_path = repo.get("config.json")?;
    println!("{}", std::fs::read_to_string(&config_path)?);

    println!("fetching onnx/model_quantized.onnx...");
    let quantized = repo.get("onnx/model_quantized.onnx")?;
    println!(
        "  -> {} ({} bytes)",
        quantized.display(),
        std::fs::metadata(&quantized)?.len()
    );

    println!("fetching tokenizer.json...");
    let tok = repo.get("tokenizer.json")?;
    println!("  -> {} ({} bytes)", tok.display(), std::fs::metadata(&tok)?.len());

    println!("loading session...");
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
