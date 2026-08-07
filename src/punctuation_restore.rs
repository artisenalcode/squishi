//! Restores real sentence-ending/comma punctuation on unpunctuated
//! prose (real YouTube auto-captions have none — see
//! `semantic_dedup.rs`'s `is_effectively_unpunctuated` for where this
//! was first found). Non-LLM: a standard XLM-RoBERTa-large token-
//! classification model (`ldenoue/fullstop-punctuation-multilang-large`,
//! an ONNX mirror of `oliverguhr/fullstop-punctuation-multilang-large`),
//! one label per token from a fixed 6-class set — the same shape as any
//! NER model, not a generative/LLM call.
//!
//! ONNX contract (confirmed via examples/probe_punctuation.rs, not
//! assumed): inputs `input_ids`/`attention_mask` (int64,
//! `[batch, seq]`) — no `token_type_ids`, XLM-RoBERTa doesn't use
//! segment ids. Output `logits` (float32, `[batch, seq, 6]`), one of:
//! 0=none, 1=".", 2=",", 3="?", 4="-", 5=":" (from the model's own
//! `config.json` `id2label`).
//!
//! Real cost, flagged rather than hidden: this model is ~562MB
//! quantized — 6x `semantic_dedup.rs`'s MiniLM (~90MB). A real,
//! material disk/first-run-download cost, accepted because the
//! alternative (the word-window fallback) produces genuinely choppy,
//! mid-thought-cut text — see `semantic_dedup.rs`'s own module doc.

use hf_hub::api::sync::Api;
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

const MODEL_REPO: &str = "ldenoue/fullstop-punctuation-multilang-large";
const ONNX_FILENAME: &str = "onnx/model_quantized.onnx";

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

pub struct PunctuationRestorer {
    session: Session,
    tokenizer: Tokenizer,
}

impl PunctuationRestorer {
    pub fn load() -> Result<Self, String> {
        let api = Api::new().map_err(|e| e.to_string())?;
        let repo = api.model(MODEL_REPO.to_string());

        let onnx_path = repo.get(ONNX_FILENAME).map_err(|e| e.to_string())?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| e.to_string())?;

        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .commit_from_file(&onnx_path)
            .map_err(|e| e.to_string())?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;

        Ok(Self { session, tokenizer })
    }

    /// Restores punctuation and sentence-start capitalization on
    /// `content`, chunking by word count to stay inside the model's
    /// position-embedding limit. Chunks are processed independently and
    /// rejoined — a chunk boundary mid-sentence is the one known
    /// artifact this doesn't handle (no overlap/fusion, unlike the
    /// reference `punctuators` package), acceptable for a real-but-not-
    /// perfect improvement over the word-window fallback it replaces.
    pub fn restore(&mut self, content: &str) -> Result<String, String> {
        let words: Vec<&str> = content.split_whitespace().collect();
        if words.is_empty() {
            return Ok(String::new());
        }

        let mut restored_chunks = Vec::new();
        for chunk_words in words.chunks(CHUNK_WORDS) {
            let chunk_text = chunk_words.join(" ");
            restored_chunks.push(self.restore_chunk(&chunk_text)?);
        }
        Ok(restored_chunks.join(" "))
    }

    fn restore_chunk(&mut self, chunk: &str) -> Result<String, String> {
        let encoding = self
            .tokenizer
            .encode(chunk, true)
            .map_err(|e| e.to_string())?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();
        let word_ids: Vec<Option<u32>> = encoding.get_word_ids().to_vec();
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

        let (shape, logits) = outputs["logits"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let num_labels = *shape.last().unwrap_or(&(LABELS.len() as i64)) as usize;

        // One predicted label per subtoken (argmax over the last dim).
        let token_labels: Vec<usize> = (0..seq_len)
            .map(|i| {
                let start = i * num_labels;
                let slice = &logits[start..start + num_labels];
                slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        // A word's punctuation is decided by its LAST subtoken's
        // prediction — punctuation attaches to the end of a word, and a
        // word can span multiple subtokens under SentencePiece.
        let mut word_label: Vec<usize> = vec![0; chunk.split_whitespace().count()];
        for (token_index, word_id) in word_ids.iter().enumerate() {
            if let Some(w) = word_id {
                let w = *w as usize;
                if w < word_label.len() {
                    word_label[w] = token_labels[token_index];
                }
            }
        }

        Ok(reconstruct(chunk, &word_label))
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
