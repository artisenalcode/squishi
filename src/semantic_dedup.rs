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
//! **2026-08-18: ported from raw `ort` to `candle` (real, verified swap,
//! not a guess)** — real cosine-similarity comparison against the
//! previous `ort` path on real sentences measured `1.0000000000`
//! (float32-rounding-level agreement) and ~40% *faster* wall time,
//! cold-start included. Uses `candle-transformers::models::bert::
//! BertModel` (a real, hand-written Rust implementation, not an ONNX
//! conversion) loaded from the model's real original safetensors
//! checkpoint (`sentence-transformers/all-MiniLM-L6-v2`) rather than the
//! ONNX mirror the `ort` path used — no ONNX involved at all now, so
//! none of `magika`/`fastembed`'s `ort`-version-pin conflicts apply to
//! this path anymore either. F32 only for now — candle's own native
//! Q8_0 quantization was tested separately (real, correct, but not
//! faster on this hardware for the punctuation model; not applied here
//! either, pending that finding changing).
//!
//! Model contract (confirmed via a real comparison against the previous
//! `ort` output, not assumed): inputs `input_ids`/`token_type_ids`
//! (`u32`, real `candle_core::Tensor`, `[batch, seq]`) plus an attention
//! mask, output the encoder's raw per-token hidden states
//! (`[batch, seq, 384]`) — not pre-pooled. Mean-pool over the sequence
//! dimension using the attention mask, then L2-normalize — the same
//! standard sentence-transformers recipe the `ort` path already used,
//! unchanged (this module's own dedup/summary logic depends on exactly
//! this contract, not on how the embedding was actually computed).

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

// Matches dedupe_semantic.py's MIN_W/MAX_W: sentences shorter than this
// are usually fragments (headers, list markers), longer ones are
// paragraphs where "paraphrase of the whole thing" stops being a
// meaningful comparison.
const MIN_WORDS: usize = 8;
const MAX_WORDS: usize = 40;

/// Below this many sentence-ending marks per 100 words, treat the text
/// as effectively unpunctuated. Real YouTube auto-caption transcripts
/// confirmed to have ZERO `.`/`!`/`?` in the actual caption body — not a
/// hypothetical edge case, this is what raw ASR output actually looks
/// like (found 2026-08-07 testing this against real Dr. Roy Sugarman
/// transcripts: the only punctuation in the whole file was in YAML
/// frontmatter, none in ~1,900 words of transcript). Below this
/// density, punctuation-splitting would treat the entire document as
/// one "sentence" — worse than useless for dedup/shape/summary, which
/// all need real sentence-sized units to operate on.
const MIN_PUNCTUATION_PER_100_WORDS: f32 = 1.0;
/// Fallback chunk size when falling back to word-window splitting —
/// inside MIN_WORDS..MAX_WORDS so fallback chunks are still eligible
/// for embedding/dedup/shape-classification like real sentences are.
const FALLBACK_WINDOW_WORDS: usize = 20;

/// Sentences per `embed_batch` ONNX call. Bounds peak memory/compute per
/// call rather than one unbounded batch for a document with tens of
/// thousands of eligible sentences (real case found 2026-08-10: a
/// multi-hour livestream transcript with ~500K words). 32 is a
/// conventional sentence-transformer batch size, not independently
/// tuned here.
const EMBED_BATCH_SIZE: usize = 32;

/// Cap on how many previously-kept embeddings a new sentence is compared
/// against, in both the keep/drop loop and the extractive-summary
/// centrality ranking. Without this, each is O(k) work per sentence over
/// k already-kept sentences — O(n^2) total across a document, since k
/// grows with n when duplication is low. Harmless at ordinary sizes
/// (dozens/hundreds of sentences) but real at the ~500K-word transcript
/// scale noted above: tens of thousands of kept sentences would mean
/// hundreds of millions of 384-dim cosine-similarity calls. Bounding to
/// the most recent/sampled MAX_COMPARISON_WINDOW trades exhaustive
/// whole-document comparison for bounded compute — the same tradeoff
/// log_compress.rs already makes with its first/last/top-N capping,
/// rather than exhaustive processing of arbitrarily large input. True
/// duplicates in transcripts are overwhelmingly local (a speaker
/// repeating themselves within a session), so windowing by recency loses
/// little in practice. Not independently tuned — a round number well
/// above what any test or the original ~81K-word corpora need.
const MAX_COMPARISON_WINDOW: usize = 500;

fn is_effectively_unpunctuated(content: &str) -> bool {
    let word_count = content.split_whitespace().count();
    if word_count == 0 {
        return false;
    }
    let marks = content.matches(['.', '!', '?']).count();
    (marks as f32 / word_count as f32) * 100.0 < MIN_PUNCTUATION_PER_100_WORDS
}

/// Splits on whitespace immediately following `.`/`!`/`?` — the `regex`
/// crate has no look-behind support, so this is a manual scan rather than
/// the `(?<=[.!?])\s+` pattern it can't express.
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

/// Groups whitespace-delimited words into fixed-size chunks and returns
/// the original substring spanning each group — a real `&str` slice of
/// `content`, not a rebuilt/owned string, same borrow shape as
/// `split_on_punctuation`. Manual char scan rather than a crate
/// dependency (`unicode-segmentation`) for one whitespace-boundary walk.
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

/// Sentence-shaped units for dedup, given text that's already had a
/// chance at punctuation restoration (see `SemanticDedup::dedupe` —
/// this is the fallback tier, not the primary path anymore): real
/// sentence splitting when the text has real punctuation, a fixed-size
/// word-window split when it still doesn't (restoration unavailable or
/// failed).
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
    /// Lazily loaded on first unpunctuated input — most callers never
    /// hit unpunctuated text in a given call, so the ~562MB punctuation
    /// model shouldn't be paid for unconditionally on every `dedupe()`.
    punctuation_restorer: Option<crate::punctuation_restore::PunctuationRestorer>,
    punctuation_load_attempted: bool,
    /// Per-stage timing accumulated across `embed_batch` calls within
    /// the current `dedupe()` — see `stage_timing.rs`'s own doc comment.
    /// Reset at the start of each `dedupe()` call.
    stage_timings: StageTimings,
}

/// Deterministic, regex-based — same heuristic style as
/// `content_detect.rs`'s shape detection, not a model. Narrative: a
/// first-person-past or reported-speech opener ("I had a client",
/// "she said", "years ago", ...) — the shape recurring stories take in
/// this corpus (see advisory/knowledge/wiki/roy-sugarman-enrichment/*.md,
/// every story-shaped entry opens this way). Concept: everything else —
/// a stated claim/mechanism, not an anecdote. Unvalidated beyond this
/// corpus's real examples; callers should treat it as a signal, not an
/// authoritative label (same caveat `content_detect.rs` carries for its
/// own regex tiers).
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

/// A survivor of dedup, with enough to trace it back to its position in
/// the original (caller-supplied) text and to rank/filter it downstream.
#[derive(Debug, Clone)]
pub struct KeptSentence {
    pub text: String,
    /// Position in the original sentence split (0-based) — the caller's
    /// traceability anchor back to the source document.
    pub index: usize,
    pub shape: SentenceShape,
    /// The embedding already computed for this sentence during dedup,
    /// carried forward rather than discarded -- a caller doing further
    /// embedding-based work downstream (e.g. cross-document clustering)
    /// can reuse this instead of re-embedding the same text a second
    /// time. `None` for a sentence outside `MIN_WORDS..MAX_WORDS`
    /// (kept unconditionally, never embedded at all -- see the
    /// `dedupe()` loop below).
    pub embedding: Option<Vec<f32>>,
}

/// A sentence collapsed into an existing kept sentence as a paraphrase —
/// the same collapse-mapping `dedupe_semantic.py`'s report produced,
/// previously discarded in this Rust port.
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
    /// Extractive, non-LLM "summary": the kept sentences ranked highest
    /// by centrality (mean cosine similarity to the other embedded kept
    /// sentences — the standard TextRank/LexRank scoring, reusing the
    /// embeddings already computed for dedup rather than a second pass),
    /// in original document order. Only sentences that were actually
    /// embedded (within MIN_WORDS..MAX_WORDS) are eligible — short
    /// always-kept fragments aren't meaningfully rankable in isolation
    /// and are never central to anything by construction.
    pub summary: Vec<String>,
    /// Whether real punctuation restoration ran and produced the
    /// sentence units this result is based on, vs. the word-window
    /// fallback (restoration unavailable, or input already had real
    /// punctuation and never needed it). A real quality signal for the
    /// caller — word-window-derived sentences are choppier.
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

        // SAFETY: a real checkpoint fetched (and cached) from
        // sentence-transformers/all-MiniLM-L6-v2 via hf-hub, the same
        // trust boundary every other model this crate loads already
        // crosses.
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

    /// Splits `content` into sentences, greedily drops any sentence whose
    /// cosine similarity to an already-kept sentence exceeds `threshold`,
    /// reassembles the survivors in original order. Sentences outside
    /// MIN_WORDS..MAX_WORDS are never dropped (not meaningfully
    /// comparable as whole-sentence paraphrases) — always kept as-is.
    ///
    /// `allow_punctuation_restore` is a caller-asserted eligibility gate,
    /// independent of `is_effectively_unpunctuated`'s content-based
    /// check: even genuinely unpunctuated content never gets the
    /// ~562MB model invoked if the caller sets this false. Real use
    /// case — total-recall's persona ingestion knows structurally which
    /// sources are raw YouTube auto-captions (need it) vs. Wikipedia/git
    /// commit messages (already have real punctuation) — belt-and-
    /// suspenders alongside the content check, not a replacement for it.
    pub fn dedupe(
        &mut self,
        content: &str,
        threshold: f32,
        allow_punctuation_restore: bool,
    ) -> Result<DedupResult, String> {
        self.stage_timings = StageTimings::default();
        let content = content.trim();

        // Try real punctuation restoration first when the input needs
        // it and the caller allows it — `restored` has to be declared
        // here (not inside the `if`) so `text_for_split`'s borrow can
        // outlive the branch.
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

        // Batch-embed every eligible sentence up front, in chunks of
        // EMBED_BATCH_SIZE -- one ONNX forward pass per chunk instead of
        // one per sentence (see EMBED_BATCH_SIZE's doc comment for why).
        // Embedding is independent per sentence; only the greedy
        // keep/drop decision below is sequential, so precomputing these
        // doesn't change that loop's semantics or output, only how the
        // embeddings were produced.
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
        // Parallel to kept_embeddings: which `sentences` index (and thus
        // which `kept` entry) each embedding belongs to, so a drop can
        // record which survivor it was actually collapsed into.
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

    /// Per-stage timing summed across every `embed_batch` call made
    /// during the most recent `dedupe()` — see `stage_timing.rs`. Only
    /// read by the timing examples; not part of the production contract.
    pub fn stage_timings(&self) -> &StageTimings {
        &self.stage_timings
    }

    /// Lazily loads the punctuation-restoration model on first need —
    /// most `dedupe()` calls never hit unpunctuated input, so its
    /// ~562MB shouldn't be paid unconditionally. Cached on `self` for
    /// the rest of this `SemanticDedup`'s lifetime once attempted
    /// (success or failure) — never retries a failed load mid-call.
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

    /// Embeds a batch of sentences in one ONNX forward pass, replacing
    /// the original one-sentence-at-a-time `embed()` (removed 2026-08-10:
    /// no longer had any callers once `dedupe()` switched to this).
    /// Real fix, not a hypothetical optimization -- found reingesting a
    /// channel with several multi-hour livestream transcripts (tens of
    /// thousands of sentences each), where one full transformer forward
    /// pass per sentence turned a single document's dedup into tens of
    /// minutes of unbatched CPU inference. The ONNX contract already
    /// supports a batch dimension (`[batch, seq]` in, `[batch, seq, 384]`
    /// out -- see this module's own doc comment) -- this was purely a
    /// missed optimization at the call site, not a model limitation.
    /// Returns embeddings in the same order as `texts`.
    ///
    /// Uses the tokenizer's own batch padding (`with_padding`, default
    /// `BatchLongest`/right-padded/`pad_id: 0`) so every sequence in the
    /// batch shares one seq_len, which ONNX's fixed-shape `[batch, seq]`
    /// tensors require. Padding is toggled back off afterward so the
    /// `Tokenizer`'s state doesn't leak into any other caller that
    /// expects unpadded single-sequence encoding.
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

/// Minimum embedded-kept sentences before a summary is worth producing —
/// below this, "the most central sentence" isn't a meaningful signal
/// (e.g. 2 sentences: each is 100% similar to "the other sentence").
const MIN_SENTENCES_FOR_SUMMARY: usize = 4;
/// Extractive summary size, as a fraction of embedded-kept sentences,
/// clamped to [MIN, MAX] — same shape as the dedup threshold: a starting
/// calibration on this corpus, not a derived constant.
const SUMMARY_FRACTION: f32 = 0.2;
const SUMMARY_MIN: usize = 3;
const SUMMARY_MAX: usize = 10;

/// TextRank/LexRank-style extractive summary: rank each embedded kept
/// sentence by its mean cosine similarity to every *other* embedded kept
/// sentence (centrality), take the top slice, return in original
/// document order — not similarity-rank order, so the summary still
/// reads as a coherent excerpt rather than a shuffled highlight reel.
/// Reuses the embeddings `dedupe()` already computed; no second model
/// pass. Deliberately non-LLM: this is a ranking of what's already
/// there, not synthesis — see semantic_dedup.rs's module doc for the
/// boundary this keeps with actual summarization.
///
/// "Every other" is capped at MAX_COMPARISON_WINDOW landmarks (a fixed
/// stride through `kept_embeddings`, same landmarks for every sentence)
/// rather than a true all-pairs mean — see that constant's doc comment.
/// Below the cap (`stride == 1`) this is exactly the original all-pairs
/// computation; only documents with more kept sentences than the window
/// get the approximation.
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

/// Finds the most-similar already-kept embedding to `embedding`, searching
/// only the most recent `MAX_COMPARISON_WINDOW` of `kept_embeddings` (see
/// that constant's doc comment for why). Pure/standalone so it's testable
/// with synthetic vectors, without a real model.
fn best_match(embedding: &[f32], kept_embeddings: &[(usize, Vec<f32>)]) -> Option<(usize, f32)> {
    let window_start = kept_embeddings.len().saturating_sub(MAX_COMPARISON_WINDOW);
    kept_embeddings[window_start..]
        .iter()
        .map(|(kept_index, k)| (*kept_index, cosine_similarity(embedding, k)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
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
        // The second sentence collapsed into the first — traceable, not
        // silently discarded.
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
        // "Yes." / "Okay." are under MIN_WORDS — never candidates for
        // dropping, regardless of similarity.
        let content = "Yes. Yes. Yes.";
        let result = d.dedupe(content, 0.80, true).unwrap();
        assert_eq!(result.original_sentences, 3);
        assert_eq!(result.kept_sentences, 3);
        // Too few sentences for a summary to mean anything, and none
        // were ever embedded (all under MIN_WORDS) — summary stays empty.
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

        // The two real (within MIN_WORDS..MAX_WORDS) sentences were
        // embedded during dedup -- that embedding should be carried
        // forward on KeptSentence, not discarded, so a downstream
        // caller never has to re-embed the same text.
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

        // "Yes." is under MIN_WORDS -- kept unconditionally, never
        // embedded at all.
        let short = result.kept.iter().find(|k| k.text == "Yes.").unwrap();
        assert!(short.embedding.is_none());
    }

    #[test]
    #[ignore]
    fn allow_punctuation_restore_false_blocks_restoration_even_on_unpunctuated_text() {
        let mut d = SemanticDedup::load().unwrap();
        // Real shape: unpunctuated (would trigger restoration if
        // allowed), but the caller explicitly disallows it here --
        // real use case: total-recall marking a Wikipedia/git-commit
        // source, which structurally never needs it, false.
        let content: String = std::iter::repeat_n("word ", 60).collect();
        let result = d.dedupe(&content, 0.80, false).unwrap();
        assert!(!result.punctuation_restored);
        // Still falls back to the word-window split -- disabling
        // restoration doesn't mean the content goes unprocessed.
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
        // Real shape: a run-on transcript with zero sentence-ending
        // punctuation, matching actual YouTube auto-captions.
        let words: Vec<&str> = std::iter::repeat_n("word", 60).collect();
        let content = words.join(" ");
        let result = split_sentences(&content);
        // 60 words / 20-word window = 3 chunks, not 1 giant "sentence".
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
        // One period buried in 200 words is below the density floor —
        // still falls back, doesn't get treated as "has real sentences".
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
    // (Narrative examples adapted from advisory/knowledge/wiki/
    // roy-sugarman-enrichment/*.md's own sourced quotes — this session's
    // real persona corpus, not invented fixtures.)

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
        // Three near-identical vectors (a cluster) + one outlier. The
        // cluster members are each highly similar to the *other two*
        // cluster members (high mean similarity = central); the outlier
        // is dissimilar to everything (low mean similarity = peripheral).
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
        // SUMMARY_MIN=3 forces all but the least-central sentence in —
        // the outlier (index 3) must be the one excluded.
        let summary = extractive_summary(&kept, &embeddings);
        assert_eq!(summary.len(), 3);
        assert!(!summary.contains(&"outlier".to_string()));
        // Still returned in original document order (0, 1, 2), not
        // similarity-rank order.
        assert_eq!(summary, vec!["cluster A", "cluster B", "cluster C"]);
    }

    #[test]
    fn best_match_ignores_duplicates_outside_the_comparison_window() {
        // A true duplicate of `query` at index 0, followed by exactly
        // MAX_COMPARISON_WINDOW filler entries orthogonal to `query`
        // (third component always 0, so cosine similarity to `query` is
        // always exactly 0, regardless of the filler's other two
        // components). Once the filler pushes index 0 outside the
        // window, `best_match` must not find it.
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
        // Regression guard for the O(k^2) fix: this many kept sentences
        // used to mean MAX_COMPARISON_WINDOW^2-scale pairwise work in
        // `extractive_summary`; it must still complete and produce a
        // valid, bounded summary rather than hang or panic.
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
