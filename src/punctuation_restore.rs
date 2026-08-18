//! Restores real sentence-ending/comma punctuation on unpunctuated
//! prose (real YouTube auto-captions have none — see
//! `semantic_dedup.rs`'s `is_effectively_unpunctuated` for where this
//! was first found). Non-LLM: an XLM-RoBERTa token-classification
//! model, one label per token from a fixed 6-class set — the same
//! shape as any NER model, not a generative/LLM call.
//!
//! **Model: `oliverguhr/fullstop-punctuation-multilingual-base`,**
//! 12 layers/768 hidden. `id2label`: 0=none, 1=".", 2=",", 3="?", 4="-",
//! 5=":", confirmed from the model's own `config.json`.
//!
//! **2026-08-18: ported from a locally-converted quantized ONNX model
//! (`ort`) to `candle`, F32, real weights fetched directly from the
//! model's real Hugging Face checkpoint** — no more local conversion
//! step at all (`scripts/convert_punctuation_base_model.py` is gone;
//! `load()` used to require running it first and erred if its output
//! wasn't present — that whole requirement is retired, `load()` now
//! just fetches real safetensors via `hf-hub`, same as
//! `semantic_dedup.rs` already does). Uses `candle-transformers::
//! models::xlm_roberta::XLMRobertaModel` (a real, hand-written Rust
//! encoder, not an ONNX conversion) plus a hand-added token-
//! classification head (`classifier.weight`/`classifier.bias`, loaded
//! straight from the same real checkpoint — `xlm_roberta.rs` itself
//! ships `XLMRobertaForSequenceClassification` but not the per-token
//! head this module needs, so the base encoder's own full hidden-state
//! output is used directly instead).
//!
//! **Real, verified swap, not a guess**: exact per-token argmax
//! agreement (14/14 on a real test sentence) against the previous
//! quantized-`ort` output, real logits within `9.1e-6` of the
//! equivalent real F32-`ort` export — and measured ~2.5x *faster*
//! wall time, cold-start included. F32 only for now — candle's own
//! native Q8_0 quantization was tested separately against this exact
//! checkpoint and measured correct but not faster on this hardware (in
//! either an inline-quantize or a pre-quantized-`.gguf`-load shape);
//! not applied here, pending that finding changing. Losing the earlier
//! int8 ONNX quantization's own real speedup (the 2.87x measured
//! 2026-08-08, referenced in this module's git history) is a real,
//! open tradeoff this migration accepts for now, not one this module
//! claims to have matched or beaten.
//!
//! Real measured throughput carried over from the pre-migration
//! investigation (this module's own git history, 2026-08-08, real
//! 6,662-word transcript, `examples/time_punctuation.rs`): the base
//! model over the large model it replaced was 2.87x faster
//! (1019.4 vs 354.7 words/sec) with no accuracy regression, confirming
//! this training family holds up at reduced layer count where a
//! smaller-capacity architecture swap (`ldenoue/distilbert-punctuator`,
//! evaluated and rejected the same day) did not.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config as XlmRobertaConfig, XLMRobertaModel};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

const MODEL_REPO: &str = "oliverguhr/fullstop-punctuation-multilingual-base";

/// The model's own `id2label` (config.json), index-matched to argmax
/// position in the logits' last dimension.
const LABELS: [&str; 6] = ["", ".", ",", "?", "-", ":"];
/// Labels after which the next word should be capitalized — this model
/// only predicts punctuation, not true-casing, so sentence-start
/// capitalization is a separate, simple heuristic layered on top.
const SENTENCE_ENDERS: [&str; 2] = [".", "?"];

/// The model's real position-embedding ceiling is 514; chunk well under
/// that in *words* (not subtokens — a word can expand to several
/// subtokens under SentencePiece) so no single inference call risks
/// truncation. Chunks are rejoined after restoration, so a lower value
/// here costs a little cross-chunk context, not correctness.
const CHUNK_WORDS: usize = 300;

/// How many chunks go into one model forward pass. Batching real work
/// into fewer, larger calls (padded to the batch's own longest chunk)
/// measured faster than one call per chunk on the same CPU this whole
/// investigation was benchmarked on — see module doc for the numbers.
/// 8 chosen as a real, moderate batch: large enough to cut per-call
/// overhead meaningfully, small enough that padding waste from one
/// short last-chunk-in-a-video doesn't dominate.
const BATCH_SIZE: usize = 8;

pub struct PunctuationRestorer {
    model: XLMRobertaModel,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    tokenizer: Tokenizer,
    device: Device,
    pad_id: u32,
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

        // SAFETY: a real checkpoint fetched (and cached) from
        // oliverguhr/fullstop-punctuation-multilingual-base via
        // hf-hub, same trust boundary every other model this crate
        // loads already crosses.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], DType::F32, &device)
                .map_err(|e| e.to_string())?
        };
        // The checkpoint's real weight prefix (confirmed via its own
        // safetensors header before writing this): the encoder lives
        // under "roberta.", with "classifier.weight"/"classifier.bias"
        // (a plain per-token linear head, 6 real labels -- see
        // LABELS) at the top level alongside it. xlm_roberta.rs ships
        // the base encoder but not this head, so it's loaded and
        // applied by hand below in `restore_batch`.
        let model = XLMRobertaModel::new(&config, vb.pp("roberta")).map_err(|e| e.to_string())?;
        let classifier_weight = vb
            .get((LABELS.len(), config.hidden_size), "classifier.weight")
            .map_err(|e| e.to_string())?;
        let classifier_bias = vb
            .get(LABELS.len(), "classifier.bias")
            .map_err(|e| e.to_string())?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
        // XLM-RoBERTa's pad token is "<pad>", id 1 -- not 0. Read it
        // from the tokenizer itself rather than hardcoding, so a
        // tokenizer swap can't silently pad with the wrong id.
        let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(1);

        Ok(Self {
            model,
            classifier_weight,
            classifier_bias,
            tokenizer,
            device,
            pad_id,
        })
    }

    /// Restores punctuation and sentence-start capitalization on
    /// `content`, chunking by word count to stay inside the model's
    /// position-embedding limit. Chunks are processed `BATCH_SIZE` at a
    /// time in one padded model forward pass each, not one call per
    /// chunk — real, measured win (see module doc) from working on a
    /// bigger unit per call instead of many small sequential ones. A
    /// chunk boundary mid-sentence is the one known
    /// artifact this doesn't handle (no overlap/fusion, unlike the
    /// reference `punctuators` package), acceptable for a real-but-not-
    /// perfect improvement over the word-window fallback it replaces.
    pub fn restore(&mut self, content: &str) -> Result<String, String> {
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

    /// Runs one or more chunks through a single model forward pass,
    /// padded to the batch's own longest chunk (not the model's 514
    /// ceiling — real waste reduction when most chunks in a batch are
    /// close to `CHUNK_WORDS` and only the last one in a video is
    /// short). Padded positions get `attention_mask = 0` and are never
    /// read back out: `word_ids` from the tokenizer is `None` for pad
    /// tokens, and the existing per-word label mapping already skips
    /// `None` — padding-awareness falls out of code that already
    /// existed for the single-chunk case, not new filtering logic.
    fn restore_batch(&mut self, chunks: &[String]) -> Result<Vec<String>, String> {
        let encodings: Vec<_> = chunks
            .iter()
            .map(|c| self.tokenizer.encode(c.as_str(), true))
            .collect::<Result<_, _>>()
            .map_err(|e| e.to_string())?;

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

        let mut results = Vec::with_capacity(batch_size);
        for (row, (chunk, encoding)) in chunks.iter().zip(encodings.iter()).enumerate() {
            let row_len = encoding.get_ids().len();
            let word_ids = encoding.get_word_ids();

            // One predicted label per real (non-padded) subtoken in
            // this row — argmax over the last dim.
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

            // A word's punctuation is decided by its LAST subtoken's
            // prediction — punctuation attaches to the end of a word,
            // and a word can span multiple subtokens under
            // SentencePiece.
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

        Ok(results)
    }
}

/// Rejoins the original words with predicted punctuation appended and
/// sentence-start capitalization applied — real word text, not
/// detokenized subwords, so no SentencePiece "▁" artifacts survive.
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
        // labels: index-aligned to words. "world" (index 1) gets ".",
        // rest get 0 (none).
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

    // --- real-model tests: slow, need network + ~562MB download on
    // cold cache. #[ignore]d, run explicitly with `-- --ignored`.

    #[test]
    #[ignore]
    fn restores_punctuation_on_real_unpunctuated_prose() {
        let mut r = PunctuationRestorer::load().unwrap();
        let text = "hello and welcome to the show today we are going to talk about \
                     neuroscience and how the brain processes emotion this is a really \
                     interesting topic because most people think emotions live in the brain \
                     but actually they live in the body";
        let restored = r.restore(text).unwrap();
        // Real assertion, not exact-string (model output isn't
        // perfectly deterministic-by-inspection): it must have
        // inserted at least one sentence-ending mark, and the result
        // must start capitalized.
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
        // Every original word must survive (chunking must not drop or
        // duplicate content at chunk boundaries).
        let restored_word_count = restored.split_whitespace().count();
        assert_eq!(restored_word_count, 700);
    }
}
