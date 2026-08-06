//! Throwaway probe: download the Kompress ONNX model, load it, print its
//! real input/output tensor names and shapes. Confirms (or corrects) the
//! I/O contract read out of headroom's Python source before any inference
//! code gets written against an assumption.

use hf_hub::api::sync::Api;
use ort::session::Session;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api = Api::new()?;
    let repo = api.model("chopratejas/kompress-v2-base".to_string());

    println!("downloading onnx/kompress-int8-wo.onnx ...");
    let path = repo.get("onnx/kompress-int8-wo.onnx")?;
    println!("downloaded to {}", path.display());

    let mut builder = Session::builder()?;
    let session = builder.commit_from_file(&path)?;

    println!("\n=== inputs ===");
    for input in session.inputs() {
        println!("{}: {:?}", input.name(), input.dtype());
    }

    println!("\n=== outputs ===");
    for output in session.outputs() {
        println!("{}: {:?}", output.name(), output.dtype());
    }

    Ok(())
}
