//! Sentence-level paraphrase dedup: collapse the same idea restated in
//! different words, not just exact-duplicate lines (line_dedup's job).
//! Same model and algorithm as `advisory/tools/dedupe_semantic.py`
//! (sentence-transformers/all-MiniLM-L6-v2, STS-tuned, greedy single-pass
//! clustering by cosine threshold) — ported to raw Rust `ort`, not
//! `fastembed`: `fastembed` hard-pins `ort =2.0.0-rc.13`, incompatible
//! with `magika`'s hard pin on `=2.0.0-rc.12` in the same crate, and
//! measured slower besides (~1.07s warm vs ~587-708ms via raw `ort` for
//! the same model — see examples/probe_single_load.rs). Replaces the
//! earlier Kompress port (removed): word-level keep/drop showed real
//! compression only on genuinely filler-heavy prose and cost 4-18s per
//! call with no state to reuse; this model is smaller (~90MB vs
//! 261-601MB), faster (sub-second warm vs multi-second), and does a
//! fundamentally different, more broadly useful job — comparing
//! sentences to each other, not scoring words in isolation.
//!
//! ONNX contract (confirmed via examples probe, not assumed): inputs
//! `input_ids`/`attention_mask`/`token_type_ids` (int64, `[batch, seq]`),
//! output `last_hidden_state` (float32, `[batch, seq, 384]`) — raw
//! per-token states, not pre-pooled. Mean-pool over the sequence
//! dimension using the attention mask, then L2-normalize — the standard
//! sentence-transformers recipe, matching what `dedupe_semantic.py`'s
//! `fastembed.TextEmbedding` does internally on the Python side.

use hf_hub::api::sync::Api;
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

const MODEL_REPO: &str = "Qdrant/all-MiniLM-L6-v2-onnx";
const ONNX_FILENAME: &str = "model.onnx";
const EMBEDDING_DIM: usize = 384;

// Matches dedupe_semantic.py's MIN_W/MAX_W: sentences shorter than this
// are usually fragments (headers, list markers), longer ones are
// paragraphs where "paraphrase of the whole thing" stops being a
// meaningful comparison.
const MIN_WORDS: usize = 8;
const MAX_WORDS: usize = 40;

/// Splits on whitespace immediately following `.`/`!`/`?` — the `regex`
/// crate has no look-behind support, so this is a manual scan rather than
/// the `(?<=[.!?])\s+` pattern it can't express.
fn split_sentences(content: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let mut chars = content.char_indices().peekable();

    while let Some((_, c)) = chars.next() {
        if matches!(c, '.' | '!' | '?')
            && let Some(&(next_i, next_c)) = chars.peek()
            && next_c.is_whitespace()
        {
            sentences.push(content[start..next_i].trim());
            start = next_i;
        }
    }
    let tail = content[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }

    sentences
}

pub struct SemanticDedup {
    session: Session,
    tokenizer: Tokenizer,
}

pub struct DedupResult {
    pub original_sentences: usize,
    pub kept_sentences: usize,
    pub content: String,
}

impl SemanticDedup {
    pub fn load() -> Result<Self, String> {
        let api = Api::new().map_err(|e| e.to_string())?;
        let repo = api.model(MODEL_REPO.to_string());

        let onnx_path = repo.get(ONNX_FILENAME).map_err(|e| e.to_string())?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| e.to_string())?;

        let mut builder = Session::builder().map_err(|e| e.to_string())?;
        let session = builder
            .commit_from_file(&onnx_path)
            .map_err(|e| e.to_string())?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;

        Ok(Self { session, tokenizer })
    }

    /// Splits `content` into sentences, greedily drops any sentence whose
    /// cosine similarity to an already-kept sentence exceeds `threshold`,
    /// reassembles the survivors in original order. Sentences outside
    /// MIN_WORDS..MAX_WORDS are never dropped (not meaningfully
    /// comparable as whole-sentence paraphrases) — always kept as-is.
    pub fn dedupe(&mut self, content: &str, threshold: f32) -> Result<DedupResult, String> {
        let sentences: Vec<&str> = split_sentences(content.trim());
        let original_sentences = sentences.len();

        let mut kept: Vec<&str> = Vec::new();
        let mut kept_embeddings: Vec<Vec<f32>> = Vec::new();

        for sentence in &sentences {
            let word_count = sentence.split_whitespace().count();
            if !(MIN_WORDS..=MAX_WORDS).contains(&word_count) {
                kept.push(sentence);
                continue;
            }

            let embedding = self.embed(sentence)?;
            let is_paraphrase_of_kept = kept_embeddings
                .iter()
                .any(|k| cosine_similarity(&embedding, k) >= threshold);

            if !is_paraphrase_of_kept {
                kept.push(sentence);
                kept_embeddings.push(embedding);
            }
        }

        Ok(DedupResult {
            original_sentences,
            kept_sentences: kept.len(),
            content: kept.join(" "),
        })
    }

    fn embed(&mut self, text: &str) -> Result<Vec<f32>, String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| e.to_string())?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
        let seq_len = ids.len();

        let input_ids = TensorRef::from_array_view(([1, seq_len], ids.as_slice()))
            .map_err(|e| e.to_string())?;
        let attention_mask = TensorRef::from_array_view(([1, seq_len], mask.as_slice()))
            .map_err(|e| e.to_string())?;
        let token_type_ids = TensorRef::from_array_view(([1, seq_len], type_ids.as_slice()))
            .map_err(|e| e.to_string())?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "token_type_ids" => token_type_ids,
            ])
            .map_err(|e| e.to_string())?;

        let (_shape, hidden) = outputs["last_hidden_state"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;

        // Mean-pool over the sequence dimension, masking out padding —
        // the standard sentence-transformers pooling recipe. hidden is
        // flat [seq_len * EMBEDDING_DIM]; mask[i] gates token i.
        let mut pooled = vec![0f32; EMBEDDING_DIM];
        let mut mask_sum = 0f32;
        for (i, &m) in mask.iter().enumerate() {
            if m == 0 {
                continue;
            }
            mask_sum += 1.0;
            for d in 0..EMBEDDING_DIM {
                pooled[d] += hidden[i * EMBEDDING_DIM + d];
            }
        }
        if mask_sum > 0.0 {
            for v in &mut pooled {
                *v /= mask_sum;
            }
        }

        // L2-normalize, matching dedupe_semantic.py's embed().
        let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut pooled {
                *v /= norm;
            }
        }

        Ok(pooled)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Both already L2-normalized, so dot product alone is cosine
    // similarity — but compute the full form anyway rather than assume
    // callers never pass unnormalized vectors in.
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real-model tests — slow (downloads/loads a real ONNX model on a
    // cold cache), require network on first run. #[ignore]d so `cargo
    // test` stays fast by default; run explicitly with
    // `cargo test -- --ignored`.

    #[test]
    #[ignore]
    fn identical_sentence_repeated_collapses_to_one() {
        let mut d = SemanticDedup::load().unwrap();
        let content = "The quarterly report shows strong growth in revenue. \
                        The quarterly report shows strong growth in revenue. \
                        The quarterly report shows strong growth in revenue.";
        let result = d.dedupe(content, 0.80).unwrap();
        assert_eq!(result.original_sentences, 3);
        assert_eq!(result.kept_sentences, 1);
    }

    #[test]
    #[ignore]
    fn true_paraphrases_collapse() {
        let mut d = SemanticDedup::load().unwrap();
        let content = "The system failed to connect to the database server. \
                        The database server could not be reached by the system.";
        let result = d.dedupe(content, 0.80).unwrap();
        assert_eq!(result.original_sentences, 2);
        assert_eq!(result.kept_sentences, 1);
    }

    #[test]
    #[ignore]
    fn distinct_sentences_both_survive() {
        let mut d = SemanticDedup::load().unwrap();
        let content = "The database connection failed after three retries. \
                        Quarterly revenue grew by twelve percent this year.";
        let result = d.dedupe(content, 0.80).unwrap();
        assert_eq!(result.original_sentences, 2);
        assert_eq!(result.kept_sentences, 2);
    }

    #[test]
    #[ignore]
    fn short_fragments_are_never_dropped() {
        let mut d = SemanticDedup::load().unwrap();
        // "Yes." / "Okay." are under MIN_WORDS — never candidates for
        // dropping, regardless of similarity.
        let content = "Yes. Yes. Yes.";
        let result = d.dedupe(content, 0.80).unwrap();
        assert_eq!(result.original_sentences, 3);
        assert_eq!(result.kept_sentences, 3);
    }
}
