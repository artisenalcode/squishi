//! Throwaway probe: confirm the Kompress tokenizer's word_ids() alignment
//! matches what headroom's Python `tokenizer(chunk_words,
//! is_split_into_words=True, ...)` produces — a subtle mismatch here
//! would silently corrupt every downstream score.

use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let api = Api::new()?;
    let repo = api.model("chopratejas/kompress-v2-base".to_string());
    let tokenizer_path = repo.get("tokenizer.json")?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)?;

    // A word ("unbelievable") likely to split into multiple subtokens,
    // surrounded by short common words likely to stay whole — checks that
    // word_ids() correctly maps N subtokens back to 1 word index.
    let words = ["this", "is", "unbelievable", "news", "today"];
    let encoding = tokenizer.encode(words.to_vec(), true)?;

    println!("tokens: {:?}", encoding.get_tokens());
    println!("word_ids: {:?}", encoding.get_word_ids());
    println!("input_ids: {:?}", encoding.get_ids());

    Ok(())
}
