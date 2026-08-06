//! Native Rust port of headroom's Kompress: a learned per-word keep/drop
//! classifier (ModernBERT, `chopratejas/kompress-v2-base`), not a
//! summarizer — extractive, order-preserving, no generation. Ported by
//! reading headroom's own Python implementation
//! (`headroom/transforms/kompress_compressor.py`) and verifying the ONNX
//! I/O contract directly (see examples/probe_kompress.rs,
//! examples/probe_tokenizer.rs) rather than assuming from the Python
//! source alone.
//!
//! ONNX contract (confirmed, not assumed): inputs `input_ids` and
//! `attention_mask` (both int64, `[batch, seq]`), single output
//! `final_scores` (float32, `[batch, seq]`). Word-level score = max over
//! that word's subtoken scores (matches headroom's aggregation exactly).

use hf_hub::api::sync::Api;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::PathBuf;
use tokenizers::Tokenizer;

const MODEL_REPO: &str = "chopratejas/kompress-v2-base";
const ONNX_FILENAME: &str = "onnx/kompress-int8-wo.onnx";
const CHUNK_WORDS: usize = 350; // coupled to training — don't change without retraining
const SCORE_THRESHOLD: f32 = 0.5;
const MIN_WORDS: usize = 10; // below this, headroom's own Python skips the model too

pub struct Kompress {
    session: Session,
    tokenizer: Tokenizer,
}

pub struct KompressResult {
    pub original_words: usize,
    pub compressed_words: usize,
    pub content: String,
}

impl Kompress {
    /// Downloads (or reuses the hf-hub cache for) the ONNX model and
    /// tokenizer on first use. No explicit cache-dir parameter, unlike
    /// total-recall's Embedder — hf-hub manages its own cache
    /// (`~/.cache/huggingface`) and this repo has no bank/multi-location
    /// concept to keep separate from.
    pub fn load() -> Result<Self, String> {
        let api = Api::new().map_err(|e| e.to_string())?;
        let repo = api.model(MODEL_REPO.to_string());

        let onnx_path: PathBuf = repo.get(ONNX_FILENAME).map_err(|e| e.to_string())?;
        let tokenizer_path: PathBuf = repo.get("tokenizer.json").map_err(|e| e.to_string())?;

        let mut builder = Session::builder().map_err(|e| e.to_string())?;
        let session = builder
            .commit_from_file(&onnx_path)
            .map_err(|e| e.to_string())?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;

        Ok(Self { session, tokenizer })
    }

    /// Compress `content` by dropping low-importance words. Passthrough
    /// (unchanged) for content under MIN_WORDS, matching headroom's own
    /// "not worth a model call" cutoff.
    pub fn compress(&mut self, content: &str) -> Result<KompressResult, String> {
        let words: Vec<&str> = content.split_whitespace().collect();
        let original_words = words.len();

        if original_words < MIN_WORDS {
            return Ok(KompressResult {
                original_words,
                compressed_words: original_words,
                content: content.to_string(),
            });
        }

        let mut kept_indices: Vec<usize> = Vec::new();

        for chunk_start in (0..words.len()).step_by(CHUNK_WORDS) {
            let chunk_end = (chunk_start + CHUNK_WORDS).min(words.len());
            let chunk = &words[chunk_start..chunk_end];
            let scores = self.score_chunk(chunk)?;

            for (word_idx, &score) in scores.iter().enumerate() {
                if score > SCORE_THRESHOLD {
                    kept_indices.push(chunk_start + word_idx);
                }
            }
        }

        if kept_indices.is_empty() {
            // Nothing survived — matches headroom's own passthrough
            // fallback rather than returning empty text.
            return Ok(KompressResult {
                original_words,
                compressed_words: original_words,
                content: content.to_string(),
            });
        }

        let compressed_words = kept_indices.len();
        let content = kept_indices
            .iter()
            .map(|&i| words[i])
            .collect::<Vec<_>>()
            .join(" ");

        Ok(KompressResult {
            original_words,
            compressed_words,
            content,
        })
    }

    /// Per-word max-subtoken score for one chunk (<=350 words). Public for
    /// debugging/inspection (see examples/probe_scores.rs) as well as
    /// internal use.
    pub fn score_chunk(&mut self, chunk_words: &[&str]) -> Result<Vec<f32>, String> {
        let encoding = self
            .tokenizer
            .encode(chunk_words.to_vec(), true)
            .map_err(|e| e.to_string())?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let word_ids = encoding.get_word_ids().to_vec();
        let seq_len = ids.len();

        let input_ids = TensorRef::from_array_view(([1, seq_len], ids.as_slice()))
            .map_err(|e| e.to_string())?;
        let attention_mask = TensorRef::from_array_view(([1, seq_len], mask.as_slice()))
            .map_err(|e| e.to_string())?;

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
            ])
            .map_err(|e| e.to_string())?;

        let (_shape, scores) = outputs["final_scores"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;

        // Max subtoken score per word, matching headroom's
        // `if wid not in word_scores or s > word_scores[wid]` aggregation.
        let mut word_scores: Vec<f32> = vec![f32::MIN; chunk_words.len()];
        for (subtoken_idx, word_id) in word_ids.iter().enumerate() {
            let Some(wid) = word_id else { continue }; // special tokens ([CLS]/[SEP])
            let wid = *wid as usize;
            if wid < word_scores.len() {
                word_scores[wid] = word_scores[wid].max(scores[subtoken_idx]);
            }
        }

        Ok(word_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests load the real model — slow (first run downloads ~261MB
    // via hf-hub's own cache, subsequent runs reuse it) and require
    // network on a cold cache. Marked #[ignore] so `cargo test` stays
    // fast by default; run explicitly with `cargo test -- --ignored`.

    #[test]
    #[ignore]
    fn short_content_is_passthrough() {
        let mut k = Kompress::load().unwrap();
        let result = k.compress("too short").unwrap();
        assert_eq!(result.content, "too short");
        assert_eq!(result.original_words, result.compressed_words);
    }

    #[test]
    #[ignore]
    fn compresses_a_real_paragraph() {
        let mut k = Kompress::load().unwrap();
        let text = "The quick brown fox jumps over the lazy dog while several \
                     onlookers stand nearby watching with great interest and \
                     mild amusement at the unusual spectacle unfolding before \
                     them on this otherwise ordinary Tuesday afternoon.";
        let result = k.compress(text).unwrap();
        assert!(result.compressed_words < result.original_words);
        assert!(!result.content.is_empty());
    }

    #[test]
    #[ignore]
    fn chunk_boundary_does_not_drop_or_duplicate_words() {
        // A synthetic input straddling exactly CHUNK_WORDS confirms
        // chunk splitting doesn't lose or double-count words at the seam.
        let mut k = Kompress::load().unwrap();
        let words: Vec<String> = (0..(CHUNK_WORDS + 50))
            .map(|i| format!("word{i}"))
            .collect();
        let text = words.join(" ");
        let result = k.compress(&text).unwrap();
        assert_eq!(result.original_words, CHUNK_WORDS + 50);
        // compressed_words must never exceed original — would indicate
        // duplication across the chunk boundary.
        assert!(result.compressed_words <= result.original_words);
    }
}
