//! Sentence-level paraphrase dedup: cosine similarity over sentence-transformers/all-MiniLM-L6-v2 embeddings via candle, not `ort` (avoids magika's pinned-version conflict) and not the old Kompress word-scorer (slower, word- not sentence-scoped).
//!
//! Model contract, confirmed by comparing real output against the previous `ort` path: inputs `input_ids`/`token_type_ids`/attention mask, output raw per-token hidden states (`[batch, seq, 384]`), mean-pooled over the attention mask then L2-normalized.

use crate::stage_timing::StageTimings;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::Api;
use regex::Regex;
use std::sync::LazyLock;
use std::time::Instant;
use tokenizers::Tokenizer;

const MODEL_REPO: &str = "sentence-transformers/all-MiniLM-L6-v2";
const EMBEDDING_DIM: usize = 384;

// Outside this range, a sentence isn't meaningfully comparable as a whole-sentence paraphrase (too short = fragment, too long = paragraph) -- always kept as-is.
const MIN_WORDS: usize = 8;
const MAX_WORDS: usize = 40;

/// Below this density, treat text as effectively unpunctuated -- real ASR transcripts can have zero sentence-ending marks in the whole document.
const MIN_PUNCTUATION_PER_100_WORDS: f32 = 1.0;
/// Word-window fallback chunk size, kept inside MIN_WORDS..MAX_WORDS so fallback chunks stay dedup-eligible like real sentences.
const FALLBACK_WINDOW_WORDS: usize = 20;

/// Sentences per `embed_batch` call, bounding peak compute for documents with tens of thousands of eligible sentences; 32 is the conventional default, not independently tuned.
const EMBED_BATCH_SIZE: usize = 32;

/// Caps how many prior kept embeddings a sentence is compared against (dedup and summary centrality alike), bounding O(n^2) cost on very large documents; true duplicates are overwhelmingly local, so windowing by recency loses little in practice.
const MAX_COMPARISON_WINDOW: usize = 500;

fn is_effectively_unpunctuated(content: &str) -> bool {
    let word_count = content.split_whitespace().count();
    if word_count == 0 {
        return false;
    }
    let marks = content.matches(['.', '!', '?']).count();
    (marks as f32 / word_count as f32) * 100.0 < MIN_PUNCTUATION_PER_100_WORDS
}

/// Manual scan, not regex: the `regex` crate has no look-behind to express "whitespace following .!?" directly.
fn split_on_punctuation(content: &str) -> Vec<&str> {
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

/// Chunks whitespace-delimited words into fixed-size groups, returning borrowed slices of `content`.
fn split_by_word_window(content: &str, window: usize) -> Vec<&str> {
    let mut word_spans: Vec<(usize, usize)> = Vec::new();
    let mut word_start: Option<usize> = None;
    for (i, c) in content.char_indices() {
        if c.is_whitespace() {
            if let Some(s) = word_start.take() {
                word_spans.push((s, i));
            }
        } else if word_start.is_none() {
            word_start = Some(i);
        }
    }
    if let Some(s) = word_start {
        word_spans.push((s, content.len()));
    }

    word_spans
        .chunks(window)
        .filter_map(|group| {
            let start = group.first()?.0;
            let end = group.last()?.1;
            Some(content[start..end].trim())
        })
        .collect()
}

/// Real sentence split when punctuated, word-window fallback otherwise (see `SemanticDedup::dedupe` for the primary punctuation-restoration path this is the fallback tier of).
fn split_sentences(content: &str) -> Vec<&str> {
    if is_effectively_unpunctuated(content) {
        split_by_word_window(content, FALLBACK_WINDOW_WORDS)
    } else {
        split_on_punctuation(content)
    }
}

pub struct SemanticDedup {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    /// Lazily loaded on first unpunctuated input, since the ~562MB punctuation model shouldn't be paid for unconditionally.
    punctuation_restorer: Option<crate::punctuation_restore::PunctuationRestorer>,
    punctuation_load_attempted: bool,
    stage_timings: StageTimings,
}

/// Deterministic regex heuristic, not a model: Narrative = first-person/reported-speech opener ("I had a client", "she said"), Concept = everything else. A signal, not an authoritative label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SentenceShape {
    Narrative,
    Concept,
}

static NARRATIVE_MARKER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(I had|I remember|I once|I worked with|I saw|I met|there was a|there was an|she said|he said|they said|told me|my (client|patient|daughter|son|friend|colleague)|years ago|one day|a client of mine)\b",
    )
    .unwrap()
});

fn classify_shape(sentence: &str) -> SentenceShape {
    if NARRATIVE_MARKER_RE.is_match(sentence) {
        SentenceShape::Narrative
    } else {
        SentenceShape::Concept
    }
}

/// A survivor of dedup, with enough to trace it back to its position in the original text and to rank/filter it downstream.
#[derive(Debug, Clone)]
pub struct KeptSentence {
    pub text: String,
    /// Position in the original sentence split -- the caller's traceability anchor back to the source document.
    pub index: usize,
    pub shape: SentenceShape,
    /// Embedding already computed during dedup, carried forward for downstream reuse; `None` for out-of-range sentences that were never embedded.
    pub embedding: Option<Vec<f32>>,
}

/// A sentence collapsed into an existing kept sentence as a paraphrase.
#[derive(Debug, Clone)]
pub struct Drop {
    pub dropped_index: usize,
    pub kept_index: usize,
    pub similarity: f32,
}

pub struct DedupResult {
    pub original_sentences: usize,
    pub kept_sentences: usize,
    pub content: String,
    pub kept: Vec<KeptSentence>,
    pub drops: Vec<Drop>,
    /// Extractive summary: kept sentences ranked by mean cosine similarity to every other kept sentence (TextRank/LexRank), in original document order.
    pub summary: Vec<String>,
    /// Whether real punctuation restoration produced these sentence units, vs. the word-window fallback -- a quality signal for the caller.
    pub punctuation_restored: bool,
}

impl SemanticDedup {
    pub fn load() -> Result<Self, String> {
        let api = Api::new().map_err(|e| e.to_string())?;
        let repo = api.model(MODEL_REPO.to_string());

        let safetensors_path = repo.get("model.safetensors").map_err(|e| e.to_string())?;
        let config_path = repo.get("config.json").map_err(|e| e.to_string())?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| e.to_string())?;

        let device = Device::Cpu;
        let config: BertConfig = serde_json::from_str(
            &std::fs::read_to_string(&config_path).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;

        // SAFETY: mmap of a real checkpoint fetched/cached from hf-hub, same trust boundary as every other model this crate loads.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], DType::F32, &device)
                .map_err(|e| e.to_string())?
        };
        let model = BertModel::load(vb, &config).map_err(|e| e.to_string())?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
        tokenizer.with_padding(None);

        Ok(Self {
            model,
            tokenizer,
            device,
            punctuation_restorer: None,
            punctuation_load_attempted: false,
            stage_timings: StageTimings::default(),
        })
    }

    /// Splits `content` into sentences, greedily drops any whose cosine similarity to an already-kept sentence exceeds `threshold`, reassembles survivors in original order. `allow_punctuation_restore` lets the caller assert eligibility independent of `is_effectively_unpunctuated`'s content check (e.g. total-recall marks Wikipedia/git sources as never needing it).
    pub fn dedupe(
        &mut self,
        content: &str,
        threshold: f32,
        allow_punctuation_restore: bool,
    ) -> Result<DedupResult, String> {
        self.stage_timings = StageTimings::default();
        let content = content.trim();

        // `restored` is declared outside the `if` so `text_for_split`'s borrow can outlive the branch.
        let restored;
        let (text_for_split, punctuation_restored) =
            if allow_punctuation_restore && is_effectively_unpunctuated(content) {
                match self
                    .punctuation_restorer()
                    .and_then(|r| r.restore(content).ok())
                {
                    Some(r) => {
                        restored = r;
                        (restored.as_str(), true)
                    }
                    None => (content, false),
                }
            } else {
                (content, false)
            };

        let sentences: Vec<&str> = split_sentences(text_for_split);
        let original_sentences = sentences.len();

        // Batch-embed every eligible sentence up front; embedding is independent per sentence, so precomputing doesn't change the sequential keep/drop loop's semantics below.
        let eligible_texts: Vec<&str> = sentences
            .iter()
            .filter(|s| (MIN_WORDS..=MAX_WORDS).contains(&s.split_whitespace().count()))
            .copied()
            .collect();
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(eligible_texts.len());
        for chunk in eligible_texts.chunks(EMBED_BATCH_SIZE) {
            embeddings.extend(self.embed_batch(chunk)?);
        }
        let mut embeddings = embeddings.into_iter();

        let mut kept: Vec<KeptSentence> = Vec::new();
        let mut drops: Vec<Drop> = Vec::new();
        // Parallel to `kept`: which `sentences` index each embedding belongs to, so a drop can record which survivor it collapsed into.
        let mut kept_embeddings: Vec<(usize, Vec<f32>)> = Vec::new();

        for (index, sentence) in sentences.iter().enumerate() {
            let word_count = sentence.split_whitespace().count();
            if !(MIN_WORDS..=MAX_WORDS).contains(&word_count) {
                kept.push(KeptSentence {
                    text: sentence.to_string(),
                    index,
                    shape: classify_shape(sentence),
                    embedding: None,
                });
                continue;
            }

            let embedding = embeddings.next().ok_or_else(|| {
                "embed_batch returned fewer embeddings than eligible sentences (internal bug)"
                    .to_string()
            })?;
            let best_match = best_match(&embedding, &kept_embeddings);

            match best_match {
                Some((kept_index, similarity)) if similarity >= threshold => {
                    drops.push(Drop {
                        dropped_index: index,
                        kept_index,
                        similarity,
                    });
                }
                _ => {
                    kept.push(KeptSentence {
                        text: sentence.to_string(),
                        index,
                        shape: classify_shape(sentence),
                        embedding: Some(embedding.clone()),
                    });
                    kept_embeddings.push((index, embedding));
                }
            }
        }

        let summary = extractive_summary(&kept, &kept_embeddings);

        Ok(DedupResult {
            original_sentences,
            kept_sentences: kept.len(),
            content: kept
                .iter()
                .map(|k| k.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
            kept,
            drops,
            summary,
            punctuation_restored,
        })
    }

    pub fn stage_timings(&self) -> &StageTimings {
        &self.stage_timings
    }

    /// Lazily loads the punctuation model on first need, caching success or failure on `self` so a failed load never retries mid-call.
    fn punctuation_restorer(
        &mut self,
    ) -> Option<&mut crate::punctuation_restore::PunctuationRestorer> {
        if !self.punctuation_load_attempted {
            self.punctuation_load_attempted = true;
            self.punctuation_restorer =
                crate::punctuation_restore::PunctuationRestorer::load().ok();
        }
        self.punctuation_restorer.as_mut()
    }

    /// One forward pass per batch instead of per sentence -- unbatched, a multi-hour livestream transcript's dedup took tens of minutes of CPU inference. Padding is toggled off again afterward so `Tokenizer` state doesn't leak into other callers expecting unpadded single-sequence encoding.
    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let tokenize_start = Instant::now();
        self.tokenizer
            .with_padding(Some(tokenizers::PaddingParams::default()));
        let encode_result = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| e.to_string());
        self.tokenizer.with_padding(None);
        let encodings = encode_result?;
        self.stage_timings.tokenize += tokenize_start.elapsed();

        let build_tensors_start = Instant::now();
        let batch_size = encodings.len();
        let seq_len = encodings[0].get_ids().len();

        let mut ids = Vec::with_capacity(batch_size * seq_len);
        let mut mask = Vec::with_capacity(batch_size * seq_len);
        let mut type_ids = Vec::with_capacity(batch_size * seq_len);
        for e in &encodings {
            ids.extend(e.get_ids().iter().copied());
            mask.extend(e.get_attention_mask().iter().copied());
            type_ids.extend(e.get_type_ids().iter().copied());
        }

        let input_ids = Tensor::from_vec(ids, (batch_size, seq_len), &self.device)
            .map_err(|e| e.to_string())?;
        let attention_mask = Tensor::from_vec(mask.clone(), (batch_size, seq_len), &self.device)
            .map_err(|e| e.to_string())?;
        let token_type_ids = Tensor::from_vec(type_ids, (batch_size, seq_len), &self.device)
            .map_err(|e| e.to_string())?;
        self.stage_timings.build_tensors += build_tensors_start.elapsed();

        let forward_start = Instant::now();
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(|e| e.to_string())?;
        let hidden: Vec<f32> = hidden
            .flatten_all()
            .map_err(|e| e.to_string())?
            .to_vec1()
            .map_err(|e| e.to_string())?;
        let hidden = hidden.as_slice();
        self.stage_timings.forward += forward_start.elapsed();

        let postprocess_start = Instant::now();
        let mut result = Vec::with_capacity(batch_size);
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

            let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut pooled {
                    *v /= norm;
                }
            }
            result.push(pooled);
        }
        self.stage_timings.postprocess += postprocess_start.elapsed();

        Ok(result)
    }
}

/// Below this many embedded-kept sentences, "the most central sentence" isn't a meaningful signal (e.g. with 2 sentences, each is 100% similar to "the other one").
const MIN_SENTENCES_FOR_SUMMARY: usize = 4;
/// Extractive summary size as a fraction of embedded-kept sentences, clamped to [MIN, MAX] -- a starting calibration, not a derived constant.
const SUMMARY_FRACTION: f32 = 0.2;
const SUMMARY_MIN: usize = 3;
const SUMMARY_MAX: usize = 10;

/// TextRank/LexRank-style extractive summary: rank each embedded kept sentence by mean cosine similarity to every other (centrality), take the top slice, return in original document order rather than similarity-rank order. "Every other" is capped at MAX_COMPARISON_WINDOW landmarks (see that constant); below the cap this is the exact all-pairs computation.
fn extractive_summary(kept: &[KeptSentence], kept_embeddings: &[(usize, Vec<f32>)]) -> Vec<String> {
    if kept_embeddings.len() < MIN_SENTENCES_FOR_SUMMARY {
        return Vec::new();
    }

    let stride = kept_embeddings.len().div_ceil(MAX_COMPARISON_WINDOW).max(1);

    let mut scored: Vec<(usize, f32)> = kept_embeddings
        .iter()
        .map(|(index, emb)| {
            let others: Vec<&Vec<f32>> = kept_embeddings
                .iter()
                .step_by(stride)
                .filter(|(i, _)| i != index)
                .map(|(_, e)| e)
                .collect();
            let mean_sim = if others.is_empty() {
                0.0
            } else {
                others
                    .iter()
                    .map(|o| cosine_similarity(emb, o))
                    .sum::<f32>()
                    / others.len() as f32
            };
            (*index, mean_sim)
        })
        .collect();

    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    let take = ((kept_embeddings.len() as f32 * SUMMARY_FRACTION).round() as usize)
        .clamp(SUMMARY_MIN, SUMMARY_MAX.min(kept_embeddings.len()));

    let mut top_indices: Vec<usize> = scored.into_iter().take(take).map(|(i, _)| i).collect();
    top_indices.sort_unstable();

    top_indices
        .into_iter()
        .filter_map(|i| kept.iter().find(|k| k.index == i).map(|k| k.text.clone()))
        .collect()
}

/// Finds the most-similar already-kept embedding, searching only the most recent MAX_COMPARISON_WINDOW of `kept_embeddings`. Pure/standalone so it's testable with synthetic vectors, without a real model.
fn best_match(embedding: &[f32], kept_embeddings: &[(usize, Vec<f32>)]) -> Option<(usize, f32)> {
    let window_start = kept_embeddings.len().saturating_sub(MAX_COMPARISON_WINDOW);
    kept_embeddings[window_start..]
        .iter()
        .map(|(kept_index, k)| (*kept_index, cosine_similarity(embedding, k)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    // Both already L2-normalized so a dot product alone would do, but compute the full form anyway rather than assume callers never pass unnormalized vectors.
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

    // Real-model tests are slow (network + model load on a cold cache) -- #[ignore]d so `cargo test` stays fast by default; run explicitly with `cargo test -- --ignored`.

    #[test]
    #[ignore]
    fn identical_sentence_repeated_collapses_to_one() {
        let mut d = SemanticDedup::load().unwrap();
        let content = "The quarterly report shows strong growth in revenue. \
                        The quarterly report shows strong growth in revenue. \
                        The quarterly report shows strong growth in revenue.";
        let result = d.dedupe(content, 0.80, true).unwrap();
        assert_eq!(result.original_sentences, 3);
        assert_eq!(result.kept_sentences, 1);
        assert_eq!(result.kept[0].index, 0);
    }

    #[test]
    #[ignore]
    fn true_paraphrases_collapse() {
        let mut d = SemanticDedup::load().unwrap();
        let content = "The system failed to connect to the database server. \
                        The database server could not be reached by the system.";
        let result = d.dedupe(content, 0.80, true).unwrap();
        assert_eq!(result.original_sentences, 2);
        assert_eq!(result.kept_sentences, 1);
        assert_eq!(result.drops.len(), 1);
        assert_eq!(result.drops[0].dropped_index, 1);
        assert_eq!(result.drops[0].kept_index, 0);
        assert!(result.drops[0].similarity >= 0.80);
    }

    #[test]
    #[ignore]
    fn distinct_sentences_both_survive() {
        let mut d = SemanticDedup::load().unwrap();
        let content = "The database connection failed after three retries. \
                        Quarterly revenue grew by twelve percent this year.";
        let result = d.dedupe(content, 0.80, true).unwrap();
        assert_eq!(result.original_sentences, 2);
        assert_eq!(result.kept_sentences, 2);
        assert!(result.drops.is_empty());
    }

    #[test]
    #[ignore]
    fn short_fragments_are_never_dropped() {
        let mut d = SemanticDedup::load().unwrap();
        // "Yes." is under MIN_WORDS -- never a drop candidate regardless of similarity.
        let content = "Yes. Yes. Yes.";
        let result = d.dedupe(content, 0.80, true).unwrap();
        assert_eq!(result.original_sentences, 3);
        assert_eq!(result.kept_sentences, 3);
        assert!(result.summary.is_empty());
    }

    #[test]
    #[ignore]
    fn kept_sentences_carry_their_already_computed_embedding_forward() {
        let mut d = SemanticDedup::load().unwrap();
        let content = "The database connection failed after three consecutive retry attempts. \
                        Quarterly revenue grew by twelve percent compared to last year. \
                        Yes.";
        let result = d.dedupe(content, 0.80, true).unwrap();
        assert_eq!(result.kept_sentences, 3);

        let real_sentences: Vec<_> = result
            .kept
            .iter()
            .filter(|k| k.text.starts_with("The database") || k.text.starts_with("Quarterly"))
            .collect();
        assert_eq!(real_sentences.len(), 2);
        for k in &real_sentences {
            let embedding = k
                .embedding
                .as_ref()
                .unwrap_or_else(|| panic!("expected Some(embedding) for {:?}", k.text));
            assert_eq!(embedding.len(), 384, "all-MiniLM-L6-v2's real dimension");
        }

        let short = result.kept.iter().find(|k| k.text == "Yes.").unwrap();
        assert!(short.embedding.is_none());
    }

    #[test]
    #[ignore]
    fn allow_punctuation_restore_false_blocks_restoration_even_on_unpunctuated_text() {
        let mut d = SemanticDedup::load().unwrap();
        let content: String = std::iter::repeat_n("word ", 60).collect();
        let result = d.dedupe(&content, 0.80, false).unwrap();
        assert!(!result.punctuation_restored);
        // Still falls back to the word-window split -- disabling restoration doesn't mean the content goes unprocessed.
        assert!(result.original_sentences > 1);
    }

    // --- split_sentences fallback: pure, no model ---

    #[test]
    fn punctuated_text_uses_real_sentence_splitting() {
        let content = "First sentence here. Second sentence here. Third one too.";
        let result = split_sentences(content);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn unpunctuated_text_falls_back_to_word_windows() {
        let words: Vec<&str> = std::iter::repeat_n("word", 60).collect();
        let content = words.join(" ");
        let result = split_sentences(&content);
        assert_eq!(result.len(), 3);
        for chunk in &result {
            assert_eq!(chunk.split_whitespace().count(), 20);
        }
    }

    #[test]
    fn unpunctuated_fallback_preserves_all_words_in_order() {
        let content = "the quick brown fox jumps over the lazy dog and then runs away very quickly into the dark forest without looking back even once more";
        let joined_words: Vec<&str> = content.split_whitespace().collect();
        let result = split_sentences(content);
        let rejoined: Vec<&str> = result.iter().flat_map(|s| s.split_whitespace()).collect();
        assert_eq!(rejoined, joined_words);
    }

    #[test]
    fn sparse_punctuation_still_counts_as_unpunctuated() {
        let mut words: Vec<&str> = std::iter::repeat_n("word", 200).collect();
        words[100] = "word.";
        let content = words.join(" ");
        let result = split_sentences(&content);
        assert!(
            result.len() > 2,
            "expected word-window fallback (multiple chunks), got {} chunk(s)",
            result.len()
        );
    }

    // --- classify_shape: pure, no model, real corpus-derived examples ---

    #[test]
    fn narrative_opener_i_had_a_client() {
        let s = "I had a client who wanted to lose an incredible amount of body weight to fit into a wedding dress.";
        assert_eq!(classify_shape(s), SentenceShape::Narrative);
    }

    #[test]
    fn narrative_reported_speech_she_said() {
        let s = "She said you didn't look at the poop, you looked at the patient.";
        assert_eq!(classify_shape(s), SentenceShape::Narrative);
    }

    #[test]
    fn narrative_years_ago_opener() {
        let s = "Years ago I joined a group which was quite small but looking impressive.";
        assert_eq!(classify_shape(s), SentenceShape::Narrative);
    }

    #[test]
    fn concept_plain_claim_is_not_narrative() {
        let s = "Context tells the genes what they need to do and what not to do under those circumstances.";
        assert_eq!(classify_shape(s), SentenceShape::Concept);
    }

    #[test]
    fn concept_framework_definition_is_not_narrative() {
        let s = "Desirability is scored zero to ten, and below an eight the goal will not survive resistance.";
        assert_eq!(classify_shape(s), SentenceShape::Concept);
    }

    // --- extractive_summary: pure, synthetic embeddings, no model ---

    fn unit_vec(mut v: Vec<f32>) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }

    #[test]
    fn summary_empty_below_minimum_sentence_count() {
        let kept = vec![
            KeptSentence {
                text: "a".into(),
                index: 0,
                shape: SentenceShape::Concept,
                embedding: None,
            },
            KeptSentence {
                text: "b".into(),
                index: 1,
                shape: SentenceShape::Concept,
                embedding: None,
            },
        ];
        let embeddings = vec![(0, unit_vec(vec![1.0, 0.0])), (1, unit_vec(vec![0.0, 1.0]))];
        assert!(extractive_summary(&kept, &embeddings).is_empty());
    }

    #[test]
    fn summary_favors_the_most_central_sentence_and_stays_in_document_order() {
        // Three near-identical vectors (a cluster) plus one outlier: cluster members are each highly similar to the other two (central); the outlier is dissimilar to everything (peripheral).
        let kept = vec![
            KeptSentence {
                text: "cluster A".into(),
                index: 0,
                shape: SentenceShape::Concept,
                embedding: None,
            },
            KeptSentence {
                text: "cluster B".into(),
                index: 1,
                shape: SentenceShape::Concept,
                embedding: None,
            },
            KeptSentence {
                text: "cluster C".into(),
                index: 2,
                shape: SentenceShape::Concept,
                embedding: None,
            },
            KeptSentence {
                text: "outlier".into(),
                index: 3,
                shape: SentenceShape::Concept,
                embedding: None,
            },
        ];
        let embeddings = vec![
            (0, unit_vec(vec![1.0, 0.05, 0.0])),
            (1, unit_vec(vec![0.95, 0.1, 0.0])),
            (2, unit_vec(vec![0.9, 0.0, 0.05])),
            (3, unit_vec(vec![0.0, 0.0, 1.0])),
        ];
        // SUMMARY_MIN=3 forces all but the least-central sentence in -- the outlier (index 3) must be the one excluded.
        let summary = extractive_summary(&kept, &embeddings);
        assert_eq!(summary.len(), 3);
        assert!(!summary.contains(&"outlier".to_string()));
        assert_eq!(summary, vec!["cluster A", "cluster B", "cluster C"]);
    }

    #[test]
    fn best_match_ignores_duplicates_outside_the_comparison_window() {
        let query = unit_vec(vec![1.0, 0.0, 0.0]);
        let mut kept_embeddings: Vec<(usize, Vec<f32>)> = vec![(0, query.clone())];
        for i in 1..=MAX_COMPARISON_WINDOW {
            let angle = i as f32;
            kept_embeddings.push((i, unit_vec(vec![0.0, angle.sin(), angle.cos()])));
        }
        let (best_index, similarity) = best_match(&query, &kept_embeddings).unwrap();
        assert_ne!(
            best_index, 0,
            "the true duplicate at index 0 is outside the window"
        );
        assert!(similarity.abs() < 1e-6);
    }

    #[test]
    fn best_match_finds_duplicate_inside_the_comparison_window() {
        let query = unit_vec(vec![1.0, 0.0]);
        let mut kept_embeddings: Vec<(usize, Vec<f32>)> = (0..MAX_COMPARISON_WINDOW - 1)
            .map(|i| (i, unit_vec(vec![0.0, 1.0])))
            .collect();
        let dup_index = kept_embeddings.len();
        kept_embeddings.push((dup_index, query.clone()));

        let (best_index, similarity) = best_match(&query, &kept_embeddings).unwrap();
        assert_eq!(best_index, dup_index);
        assert!(similarity > 0.999);
    }

    #[test]
    fn summary_handles_more_kept_sentences_than_the_comparison_window() {
        // Regression guard for the O(k^2) fix: must complete and produce a bounded summary rather than hang or panic.
        let n = MAX_COMPARISON_WINDOW * 2 + 3;
        let kept: Vec<KeptSentence> = (0..n)
            .map(|i| KeptSentence {
                text: format!("sentence {i}"),
                index: i,
                shape: SentenceShape::Concept,
                embedding: None,
            })
            .collect();
        let embeddings: Vec<(usize, Vec<f32>)> = (0..n)
            .map(|i| {
                let angle = i as f32 * 0.001;
                (i, unit_vec(vec![angle.cos(), angle.sin()]))
            })
            .collect();

        let summary = extractive_summary(&kept, &embeddings);
        assert!(!summary.is_empty());
        assert!(summary.len() <= SUMMARY_MAX);
    }
}
