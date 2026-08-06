//! Debug probe: dump raw score distribution for a real chunk to check for
//! degenerate output (all-keep, all-drop, NaN) before trusting the
//! threshold logic.

use squishi::kompress::Kompress;
use std::time::Instant;

fn main() -> Result<(), String> {
    let t0 = Instant::now();
    let mut k = Kompress::load()?;
    eprintln!("load took {:?}", t0.elapsed());

    let words: Vec<&str> =
        "This is unique sentence number 0 describing something different each time. \
                This is unique sentence number 1 describing something different each time. \
                This is unique sentence number 2 describing something different each time."
            .split_whitespace()
            .collect();

    let t1 = Instant::now();
    let scores = k.score_chunk(&words)?;
    eprintln!("score_chunk took {:?}", t1.elapsed());

    for (word, score) in words.iter().zip(scores.iter()) {
        println!("{score:.3}  {word}");
    }

    let min = scores.iter().cloned().fold(f32::MAX, f32::min);
    let max = scores.iter().cloned().fold(f32::MIN, f32::max);
    let mean: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
    eprintln!("min={min:.3} max={max:.3} mean={mean:.3}");

    Ok(())
}
