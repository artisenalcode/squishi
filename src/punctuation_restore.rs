//! Restores sentence-ending/comma punctuation on unpunctuated prose (real YouTube auto-captions have none). Non-LLM: an XLM-RoBERTa token-classification model, one label per token from a fixed 6-class set, the same shape as any NER model.
//!
//! Model: `oliverguhr/fullstop-punctuation-multilingual-base` (12 layers/768 hidden). `id2label`: 0=none, 1=".", 2=",", 3="?", 4="-", 5=":".
//!
//! Runs on `candle` (F32), real weights fetched via `hf-hub`, using `candle_transformers`' `XLMRobertaModel` plus a hand-added token-classification head loaded from the same checkpoint (the crate ships `XLMRobertaForSequenceClassification`, not the per-token head this module needs). Verified against the prior ONNX path: exact per-token argmax agreement on a real test sentence, ~2.5x faster wall time. Not quantized here -- candle's Q8_0 measured correct but not faster on this hardware, so the earlier ONNX quantization's speedup isn't matched, an accepted open tradeoff for now.

use crate::stage_timing::StageTimings;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config as XlmRobertaConfig, XLMRobertaModel};
use hf_hub::api::sync::Api;
use std::time::Instant;
use tokenizers::Tokenizer;

const MODEL_REPO: &str = "oliverguhr/fullstop-punctuation-multilingual-base";

/// The model's own `id2label` (config.json), index-matched to argmax position in the logits' last dimension.
const LABELS: [&str; 6] = ["", ".", ",", "?", "-", ":"];
/// Labels after which the next word should be capitalized -- the model only predicts punctuation, not true-casing.
const SENTENCE_ENDERS: [&str; 2] = [".", "?"];

/// The model's position-embedding ceiling is 514; chunk well under that in *words*, not subtokens, since a word can expand to several subtokens under SentencePiece.
const CHUNK_WORDS: usize = 300;

/// Chunks per model forward pass, padded to the batch's own longest chunk -- measured faster than one call per chunk. 8 balances per-call overhead against padding waste on a short last chunk.
const BATCH_SIZE: usize = 8;

pub struct PunctuationRestorer {
    model: XLMRobertaModel,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    tokenizer: Tokenizer,
    device: Device,
    pad_id: u32,
    /// Per-stage timing across `restore_batch` calls in the current `restore()`, reset each call.
    stage_timings: StageTimings,
}

impl PunctuationRestorer {
    pub fn load() -> Result<Self, String> {
        let api = Api::new().map_err(|e| e.to_string())?;
        let repo = api.model(MODEL_REPO.to_string());

        let safetensors_path = repo.get("model.safetensors").map_err(|e| e.to_string())?;
        let config_path = repo.get("config.json").map_err(|e| e.to_string())?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| e.to_string())?;

        let device = Device::Cpu;
        let config: XlmRobertaConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        // SAFETY: mmap of a real checkpoint fetched/cached from hf-hub, same trust boundary as every other model this crate loads.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], DType::F32, &device)
                .map_err(|e| e.to_string())?
        };
        // The checkpoint's encoder lives under "roberta.", with "classifier.weight"/"classifier.bias" (a plain per-token linear head) at the top level -- xlm_roberta.rs ships the base encoder but not this head, so it's applied by hand in `restore_batch`.
        let model = XLMRobertaModel::new(&config, vb.pp("roberta")).map_err(|e| e.to_string())?;
        let classifier_weight = vb
            .get((LABELS.len(), config.hidden_size), "classifier.weight")
            .map_err(|e| e.to_string())?;
        let classifier_bias = vb
            .get(LABELS.len(), "classifier.bias")
            .map_err(|e| e.to_string())?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
        // XLM-RoBERTa's pad token is "<pad>", id 1, not 0 -- read from the tokenizer rather than hardcoding.
        let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(1);

        Ok(Self {
            model,
            classifier_weight,
            classifier_bias,
            tokenizer,
            device,
            pad_id,
            stage_timings: StageTimings::default(),
        })
    }

    /// Restores punctuation and sentence-start capitalization, chunking by word count to stay inside the model's position limit and batching `BATCH_SIZE` chunks per forward pass. A chunk boundary mid-sentence is the one known artifact (no overlap/fusion), acceptable next to the word-window fallback this replaces.
    pub fn restore(&mut self, content: &str) -> Result<String, String> {
        self.stage_timings = StageTimings::default();
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return Ok(String::new());
        }

        let chunk_texts: Vec<String> = words
            .chunks(CHUNK_WORDS)
            .map(|chunk_words| chunk_words.join(" "))
            .collect();

        let mut restored_chunks = Vec::with_capacity(chunk_texts.len());
        for batch in chunk_texts.chunks(BATCH_SIZE) {
            restored_chunks.extend(self.restore_batch(batch)?);
        }
        Ok(restored_chunks.join(" "))
    }

    pub fn stage_timings(&self) -> &StageTimings {
        &self.stage_timings
    }

    /// Runs one or more chunks through a single forward pass, padded to the batch's own longest chunk rather than the model's 514 ceiling. Padded positions get `attention_mask = 0` and are never read back out, since `word_ids` is `None` for pad tokens and the per-word label mapping already skips `None`.
    fn restore_batch(&mut self, chunks: &[String]) -> Result<Vec<String>, String> {
        let tokenize_start = Instant::now();
        let encodings: Vec<_> = chunks
            .iter()
            .map(|c| self.tokenizer.encode(c.as_str(), true))
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;
        self.stage_timings.tokenize += tokenize_start.elapsed();

        let build_tensors_start = Instant::now();
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);
        let batch_size = encodings.len();

        let mut ids = vec![self.pad_id; batch_size * max_len];
        let mut mask = vec![0u32; batch_size * max_len];
        for (row, encoding) in encodings.iter().enumerate() {
            let row_ids = encoding.get_ids();
            let row_mask = encoding.get_attention_mask();
            let offset = row * max_len;
            for (col, (&id, &m)) in row_ids.iter().zip(row_mask.iter()).enumerate() {
                ids[offset + col] = id;
                mask[offset + col] = m;
            }
        }

        let input_ids = Tensor::from_vec(ids, (batch_size, max_len), &self.device)
            .map_err(|e| e.to_string())?;
        let attention_mask = Tensor::from_vec(mask, (batch_size, max_len), &self.device)
            .map_err(|e| e.to_string())?;
        let token_type_ids = Tensor::zeros((batch_size, max_len), DType::U32, &self.device)
            .map_err(|e| e.to_string())?;
        self.stage_timings.build_tensors += build_tensors_start.elapsed();

        let forward_start = Instant::now();
        let hidden = self
            .model
            .forward(
                &input_ids,
                &attention_mask,
                &token_type_ids,
                None,
                None,
                None,
            )
            .map_err(|e| e.to_string())?;
        let logits = hidden
            .broadcast_matmul(&self.classifier_weight.t().map_err(|e| e.to_string())?)
            .and_then(|l| l.broadcast_add(&self.classifier_bias))
            .map_err(|e| e.to_string())?;
        self.stage_timings.forward += forward_start.elapsed();

        let postprocess_start = Instant::now();
        let mut results = Vec::with_capacity(batch_size);
        for (row, (chunk, encoding)) in chunks.iter().zip(encodings.iter()).enumerate() {
            let row_len = encoding.get_ids().len();
            let word_ids = encoding.get_word_ids();

            // One predicted label per real (non-padded) subtoken -- argmax over the last dim.
            let token_labels: Vec<usize> = (0..row_len)
                .map(|i| {
                    let slice: Vec<f32> = logits
                        .i((row, i))
                        .and_then(|t| t.to_vec1())
                        .map_err(|e| e.to_string())?;
                    Ok::<usize, String>(
                        slice
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .map(|(idx, _)| idx)
                            .unwrap_or(0),
                    )
                })
                .collect::<Result<_, _>>()?;

            // A word's punctuation is decided by its LAST subtoken's prediction, since a word can span multiple subtokens under SentencePiece.
            let mut word_label: Vec<usize> = vec![0; chunk.split_whitespace().count()];
            for (token_index, word_id) in word_ids.iter().enumerate().take(row_len) {
                if let Some(w) = word_id {
                    let w = *w as usize;
                    if w < word_label.len() {
                        word_label[w] = token_labels[token_index];
                    }
                }
            }

            results.push(reconstruct(chunk, &word_label));
        }

        self.stage_timings.postprocess += postprocess_start.elapsed();
        Ok(results)
    }
}

/// Rejoins the original words with predicted punctuation and capitalization applied -- real word text, not detokenized subwords, so no SentencePiece "▁" artifacts survive.
fn reconstruct(original: &str, word_labels: &[usize]) -> String {
    let words: Vec<&str> = original.split_whitespace().collect();
    let mut out = String::new();
    let mut capitalize_next = true;

    for (i, word) in words.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        if capitalize_next {
            out.push_str(&capitalize_first(word));
        } else {
            out.push_str(word);
        }
        capitalize_next = false;

        if let Some(&label) = word_labels.get(i)
            && label != 0
            && let Some(mark) = LABELS.get(label)
        {
            out.push_str(mark);
            if SENTENCE_ENDERS.contains(mark) {
                capitalize_next = true;
            }
        }
    }
    out
}

fn capitalize_first(word: &str) -> String {
    let mut chars = word.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- reconstruct/capitalize_first: pure, no model ---

    #[test]
    fn reconstruct_inserts_period_and_capitalizes_next_word() {
        let original = "hello world how are you";
        // labels are index-aligned to words: "world" (index 1) gets ".", rest get 0 (none).
        let labels = [0, 1, 0, 0, 0];
        let result = reconstruct(original, &labels);
        assert_eq!(result, "Hello world. How are you");
    }

    #[test]
    fn reconstruct_handles_comma_without_capitalizing() {
        let original = "first second third";
        let labels = [0, 2, 0]; // "second" gets a comma
        let result = reconstruct(original, &labels);
        assert_eq!(result, "First second, third");
    }

    #[test]
    fn reconstruct_with_no_punctuation_labels_just_capitalizes_first_word() {
        let original = "no punctuation predicted here at all";
        let labels = [0, 0, 0, 0, 0, 0];
        let result = reconstruct(original, &labels);
        assert_eq!(result, "No punctuation predicted here at all");
    }

    #[test]
    fn reconstruct_question_mark_also_triggers_capitalization() {
        let original = "are you sure yes i am";
        let labels = [0, 0, 3, 0, 0, 0]; // "sure" gets "?"
        let result = reconstruct(original, &labels);
        assert_eq!(result, "Are you sure? Yes i am");
    }

    #[test]
    fn capitalize_first_handles_unicode() {
        assert_eq!(capitalize_first("étude"), "Étude");
    }

    #[test]
    fn capitalize_first_on_empty_string_is_empty() {
        assert_eq!(capitalize_first(""), "");
    }

    // --- real-model tests: slow, need network + ~562MB download on cold cache. #[ignore]d, run explicitly with `-- --ignored`.

    #[test]
    #[ignore]
    fn restores_punctuation_on_real_unpunctuated_prose() {
        let mut r = PunctuationRestorer::load().unwrap();
        let text = "hello and welcome to the show today we are going to talk about \
                     neuroscience and how the brain processes emotion this is a really \
                     interesting topic because most people think emotions live in the brain \
                     but actually they live in the body";
        let restored = r.restore(text).unwrap();
        // Not exact-string (model output isn't perfectly deterministic-by-inspection): must insert at least one sentence-ending mark and start capitalized.
        assert!(
            restored.contains('.') || restored.contains('?'),
            "expected at least one sentence-ending mark in: {restored}"
        );
        assert!(restored.starts_with(|c: char| c.is_uppercase()));
    }

    #[test]
    #[ignore]
    fn long_input_is_chunked_and_fully_restored_without_dropping_words() {
        let mut r = PunctuationRestorer::load().unwrap();
        let words: Vec<&str> = std::iter::repeat_n("word", 700).collect();
        let text = words.join(" ");
        let restored = r.restore(&text).unwrap();
        // Every original word must survive -- chunking must not drop or duplicate content at chunk boundaries.
        let restored_word_count = restored.split_whitespace().count();
        assert_eq!(restored_word_count, 700);
    }
}
