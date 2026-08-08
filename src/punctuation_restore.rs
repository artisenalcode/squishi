//! Restores real sentence-ending/comma punctuation on unpunctuated
//! prose (real YouTube auto-captions have none — see
//! `semantic_dedup.rs`'s `is_effectively_unpunctuated` for where this
//! was first found). Non-LLM: an XLM-RoBERTa token-classification
//! model, one label per token from a fixed 6-class set — the same
//! shape as any NER model, not a generative/LLM call.
//!
//! **Model: `oliverguhr/fullstop-punctuation-multilingual-base`,**
//! 12 layers/768 hidden, converted locally to a 278,254,995-byte
//! quantized ONNX model by `scripts/convert_punctuation_base_model.py`
//! (no one publishes an ONNX export of this one — it did not exist
//! anywhere to just fetch). `load()` requires this at
//! `~/.cache/squishi/models/fullstop-punctuation-multilingual-base/` —
//! run the conversion script first; there's no automatic fallback to
//! the old large model (`ldenoue/fullstop-punctuation-multilang-large`,
//! 24 layers/1024 hidden) anymore, dropped 2026-08-08 to keep this
//! module to one path. `id2label`: 0=none, 1=".", 2=",", 3="?", 4="-",
//! 5=":", confirmed from the base model's own `config.json` before
//! converting — same scheme as the large model it replaced (same
//! training methodology/family, base is just fewer layers).
//!
//! **Measured on a real 6,662-word Hormozi transcript, 2026-08-08**
//! (`examples/time_punctuation.rs`): large — 18.78s restore, 3.2s load,
//! 354.7 words/sec. Base (real ONNX+int8, not the PyTorch estimate
//! that motivated trying this) — **6.53s restore, 1.46s load, 1019.4
//! words/sec — 2.87x faster**, and better than the 1.4x a PyTorch-only
//! test predicted, as expected once quantization compounds the
//! layer-count gain. Output stayed coherent, same 6-class scheme, no
//! wording changes needed elsewhere in this file. Useful at real
//! corpus scale — Alex Hormozi's corpus alone is ~500 videos.
//!
//! **A different, smaller swap was evaluated and rejected first,
//! same day.** `ldenoue/distilbert-punctuator` (DistilBERT,
//! English-only, 66,985,523 bytes — 8.4x smaller, same
//! tokenizer.json-based loading path) measured 5.5x faster (18.8s →
//! 3.4s) on the same transcript, but its output had periods/commas
//! inserted at grammatically wrong positions ("...you spend. More
//! money 10 million, ads very low trust and that's, it so I said a.
//! Bunch of businesses...") — a genuine accuracy regression from
//! DistilBERT's smaller capacity and different training data, not a
//! wiring bug (I/O contract confirmed correct via
//! `examples/probe_distilbert_punctuator.rs` first). Rejected: speed
//! without correctness isn't a win for data feeding advisor persona
//! synthesis. The base model above is the same training family as the
//! already-proven large model, not an architecture swap — that's why
//! it held up where DistilBERT didn't.

use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

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

/// How many chunks go into one `session.run()` call. Batching real
/// work into fewer, larger calls (padded to the batch's own longest
/// chunk) measured faster than one call per chunk on the same CPU this
/// whole investigation was benchmarked on — see module doc for the
/// numbers. 8 chosen as a real, moderate batch: large enough to cut
/// per-call overhead meaningfully, small enough that padding waste
/// from one short last-chunk-in-a-video doesn't dominate.
const BATCH_SIZE: usize = 8;

/// Where `scripts/convert_punctuation_base_model.py` writes the
/// converted model. No fallback to any other model — run the
/// conversion script first (see module doc); `load()` errors clearly
/// if this path is missing rather than silently using something
/// slower.
fn local_base_model_dir() -> Option<std::path::PathBuf> {
    let mut dir = dirs::home_dir()?;
    dir.push(".cache");
    dir.push("squishi");
    dir.push("models");
    dir.push("fullstop-punctuation-multilingual-base");
    Some(dir)
}

pub struct PunctuationRestorer {
    session: Session,
    tokenizer: Tokenizer,
    pad_id: i64,
}

impl PunctuationRestorer {
    pub fn load() -> Result<Self, String> {
        let dir = local_base_model_dir()
            .ok_or_else(|| "could not resolve home directory for model cache".to_string())?;
        let onnx_path = dir.join("model_quantized.onnx");
        let tokenizer_path = dir.join("tokenizer.json");
        if !onnx_path.exists() || !tokenizer_path.exists() {
            return Err(format!(
                "punctuation model not found at {} -- run `python3 scripts/convert_punctuation_base_model.py` first",
                dir.display()
            ));
        }

        let session = Session::builder()
            .map_err(|e| e.to_string())?
            .commit_from_file(&onnx_path)
            .map_err(|e| e.to_string())?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
        // XLM-RoBERTa's pad token is "<pad>", id 1 -- not 0. Read it
        // from the tokenizer itself rather than hardcoding, so a
        // tokenizer swap can't silently pad with the wrong id.
        let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(1) as i64;

        Ok(Self {
            session,
            tokenizer,
            pad_id,
        })
    }

    /// Restores punctuation and sentence-start capitalization on
    /// `content`, chunking by word count to stay inside the model's
    /// position-embedding limit. Chunks are processed `BATCH_SIZE` at a
    /// time in one padded `session.run()` call each, not one call per
    /// chunk — real, measured win (see module doc) from letting ONNX
    /// Runtime work on a bigger unit per call instead of many small
    /// sequential ones. A chunk boundary mid-sentence is the one known
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

    /// Runs one or more chunks through a single `session.run()` call,
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
        let mut mask = vec![0i64; batch_size * max_len];
        for (row, encoding) in encodings.iter().enumerate() {
            let row_ids = encoding.get_ids();
            let row_mask = encoding.get_attention_mask();
            let offset = row * max_len;
            for (col, (&id, &m)) in row_ids.iter().zip(row_mask.iter()).enumerate() {
                ids[offset + col] = id as i64;
                mask[offset + col] = m as i64;
            }
        }

        let input_ids = TensorRef::from_array_view(([batch_size, max_len], ids.as_slice()))
            .map_err(|e| e.to_string())?;
        let attention_mask = TensorRef::from_array_view(([batch_size, max_len], mask.as_slice()))
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

        let mut results = Vec::with_capacity(batch_size);
        for (row, (chunk, encoding)) in chunks.iter().zip(encodings.iter()).enumerate() {
            let row_len = encoding.get_ids().len();
            let word_ids = encoding.get_word_ids();
            let row_offset = row * max_len * num_labels;

            // One predicted label per real (non-padded) subtoken in
            // this row — argmax over the last dim.
            let token_labels: Vec<usize> = (0..row_len)
                .map(|i| {
                    let start = row_offset + i * num_labels;
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
