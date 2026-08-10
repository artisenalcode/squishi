//! One-off timing harness: is squishi's ONNX session under-threaded?
//! `semantic_dedup.rs`'s `Session::builder()` never calls
//! `with_intra_threads`/`with_parallel_execution` -- it runs on
//! whatever ORT's own defaults pick. Found while chasing a real
//! bottleneck in `persona`'s pipeline: `dedup-raw` against a real
//! 54-source corpus showed only ~308% average CPU utilization on an
//! 8-core box (953s cumulative CPU / 312s wall), well short of
//! saturating available cores.
//!
//! This duplicates the minimal batch-embed path from `embed_batch`
//! (which is private, not part of the crate's public API) against a
//! real ONNX session, timing default-threaded vs. explicitly-threaded
//! configurations over the same batch count/size squishi actually uses
//! (`EMBED_BATCH_SIZE = 32`). Not wired into the library either way
//! until this measurement says which config is actually faster.

use hf_hub::api::sync::Api;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use std::time::Instant;
use tokenizers::Tokenizer;

const MODEL_REPO: &str = "Qdrant/all-MiniLM-L6-v2-onnx";
const ONNX_FILENAME: &str = "model.onnx";
const EMBEDDING_DIM: usize = 384;
const BATCH_SIZE: usize = 32;
const NUM_BATCHES: usize = 50; // 1,600 sentences -- comparable order of magnitude to a real corpus run

fn sample_sentences(n: usize) -> Vec<String> {
    // Varied-length filler in MIN_WORDS..MAX_WORDS's real range -- pure
    // throughput probe, content doesn't matter, only token count shape does.
    (0..n)
        .map(|i| {
            format!(
                "This is sample sentence number {i} used only to measure embedding throughput under different thread configurations today."
            )
        })
        .collect()
}

fn run_batches(
    session: &mut Session,
    tokenizer: &mut Tokenizer,
    sentences: &[String],
) -> Result<(), String> {
    tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));
    for chunk in sentences.chunks(BATCH_SIZE) {
        let texts: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
        let encodings = tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| e.to_string())?;

        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        let mut ids = Vec::with_capacity(batch_size * seq_len);
        let mut mask = Vec::with_capacity(batch_size * seq_len);
        let mut type_ids = Vec::with_capacity(batch_size * seq_len);
        for e in &encodings {
            ids.extend(e.get_ids().iter().map(|&id| id as i64));
            mask.extend(e.get_attention_mask().iter().map(|&m| m as i64));
            type_ids.extend(e.get_type_ids().iter().map(|&t| t as i64));
        }

        let input_ids = TensorRef::from_array_view(([batch_size, seq_len], ids.as_slice()))
            .map_err(|e| e.to_string())?;
        let attention_mask = TensorRef::from_array_view(([batch_size, seq_len], mask.as_slice()))
            .map_err(|e| e.to_string())?;
        let token_type_ids =
            TensorRef::from_array_view(([batch_size, seq_len], type_ids.as_slice()))
                .map_err(|e| e.to_string())?;

        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])
            .map_err(|e| e.to_string())?;
        let (_shape, hidden) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;

        // Pool, same as embed_batch, so this isn't measuring a cheaper op.
        for b in 0..batch_size {
            let mut pooled = vec![0f32; EMBEDDING_DIM];
            let mut mask_sum = 0f32;
            for s in 0..seq_len {
                if mask[b * seq_len + s] == 0 {
                    continue;
                }
                mask_sum += 1.0;
                let base = (b * seq_len + s) * EMBEDDING_DIM;
                for d in 0..EMBEDDING_DIM {
                    pooled[d] += hidden[base + d];
                }
            }
            if mask_sum > 0.0 {
                for v in &mut pooled {
                    *v /= mask_sum;
                }
            }
        }
    }
    tokenizer.with_padding(None);
    Ok(())
}

fn bench(label: &str, session: &mut Session, tokenizer: &mut Tokenizer, sentences: &[String]) {
    // One warmup batch -- first call pays lazy init cost unrelated to
    // steady-state throughput.
    let _ = run_batches(
        session,
        tokenizer,
        &sentences[..BATCH_SIZE.min(sentences.len())],
    );

    let start = Instant::now();
    run_batches(session, tokenizer, sentences).expect("bench run failed");
    let elapsed = start.elapsed();
    println!(
        "{label}: {elapsed:?} total, {:.1} sentences/sec ({NUM_BATCHES} batches of {BATCH_SIZE})",
        sentences.len() as f64 / elapsed.as_secs_f64()
    );
}

fn main() -> Result<(), String> {
    let api = Api::new().map_err(|e| e.to_string())?;
    let repo = api.model(MODEL_REPO.to_string());
    let onnx_path = repo.get(ONNX_FILENAME).map_err(|e| e.to_string())?;
    let tokenizer_path = repo.get("tokenizer.json").map_err(|e| e.to_string())?;

    let sentences = sample_sentences(NUM_BATCHES * BATCH_SIZE);
    let cores = std::thread::available_parallelism()
        .map_err(|e| e.to_string())?
        .get();
    println!("available_parallelism: {cores}");

    // Config 1: exactly what semantic_dedup.rs does today -- no explicit
    // thread/parallelism config, pure ORT defaults.
    {
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
        let mut session = Session::builder()
            .map_err(|e| e.to_string())?
            .commit_from_file(&onnx_path)
            .map_err(|e| e.to_string())?;
        bench(
            "default (current squishi config)",
            &mut session,
            &mut tokenizer,
            &sentences,
        );
    }

    // Config 2: explicit intra-op threads = available cores.
    {
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
        let mut session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_intra_threads(cores)
            .map_err(|e| e.to_string())?
            .commit_from_file(&onnx_path)
            .map_err(|e| e.to_string())?;
        bench(
            &format!("with_intra_threads({cores})"),
            &mut session,
            &mut tokenizer,
            &sentences,
        );
    }

    // Config 3: explicit intra threads + parallel execution mode + max
    // graph optimization, in case any of those (not just thread count)
    // is what's actually left on the table.
    {
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
        let mut session = Session::builder()
            .map_err(|e| e.to_string())?
            .with_intra_threads(cores)
            .map_err(|e| e.to_string())?
            .with_inter_threads(cores)
            .map_err(|e| e.to_string())?
            .with_parallel_execution(true)
            .map_err(|e| e.to_string())?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| e.to_string())?
            .commit_from_file(&onnx_path)
            .map_err(|e| e.to_string())?;
        bench(
            &format!("with_intra_threads({cores}) + inter + parallel_execution + opt_level3"),
            &mut session,
            &mut tokenizer,
            &sentences,
        );
    }

    Ok(())
}
