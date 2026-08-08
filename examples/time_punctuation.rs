//! One-off timing harness: load the punctuation restorer, restore a
//! real file's content, print load time / restore time / word and
//! chunk counts. Used to baseline the current large model against a
//! real Hormozi transcript before evaluating a smaller replacement.

use squishi::punctuation_restore::PunctuationRestorer;
use std::env;
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).expect("usage: time_punctuation <file>");
    let raw = fs::read_to_string(&path)?;
    // Strip YAML frontmatter and the leading '# Title' line, same shape
    // real content takes when it reaches the restorer.
    let content = raw
        .splitn(3, "---\n")
        .nth(2)
        .unwrap_or(&raw)
        .lines()
        .filter(|l| !l.starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ");
    let word_count = content.split_whitespace().count();
    println!("file: {path}");
    println!("word count: {word_count}");

    let load_start = Instant::now();
    let mut restorer = PunctuationRestorer::load()?;
    println!("load time: {:?}", load_start.elapsed());

    let restore_start = Instant::now();
    let restored = restorer.restore(&content)?;
    let elapsed = restore_start.elapsed();
    println!("restore time: {elapsed:?}");
    println!("chunks: {}", word_count.div_ceil(300));
    println!(
        "words/sec: {:.1}",
        word_count as f64 / elapsed.as_secs_f64()
    );
    println!("\n--- first 500 chars of restored output ---");
    println!("{}", &restored[..restored.len().min(500)]);

    Ok(())
}
