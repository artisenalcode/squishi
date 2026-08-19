//! Text-to-image rendering ("pixel-mode"), squishi's port of pxpipe's technique (github.com/teamchong/pxpipe, MIT): render dense text as a bitmap PNG instead of shipping it as tokens.
//!
//! Rendering primitive only -- delivery is out of scope. Claude Code has an open platform bug (anthropics/claude-code#31208) turning an MCP tool's image response into base64 text instead of a native image block, the opposite of the point; no live consumer is wired to this module yet.
//!
//! This file never imports or shells out to `trm`. Pixel-mode is lossy, so a real deployment needs CCR registration of the original text, but that's the caller's job, not this module's.
//!
//! Eligibility is a safety gate, not just a density heuristic: pxpipe states its own limitation of being unsafe for byte-exact recall of short strings (hashes, secrets, identifiers, code). The eligible set is exactly `Json`/`Log`/`PlainText` -- `Diff` and anything Magika calls `Other(_)` are refused outright, regardless of density.

use crate::content_detect::{self, ContentKind};
use image::{GrayImage, ImageFormat, Luma};
use spleen_font::{FONT_5X8, PSF2Font};
use std::io::Cursor;

/// Spleen's 5×8 glyph cell, the face pxpipe uses for Claude specifically.
pub const GLYPH_WIDTH: u32 = 5;
pub const GLYPH_HEIGHT: u32 = 8;

/// pxpipe's page geometry for Claude: 312 columns, 1568×728px pages.
pub const COLUMNS: u32 = 312;
pub const PAGE_WIDTH: u32 = 1568;
pub const PAGE_HEIGHT: u32 = 728;

/// `728 / 8 == 91` exactly, so stacked pages need no vertical padding between them.
pub const ROWS_PER_PAGE: u32 = PAGE_HEIGHT / GLYPH_HEIGHT;

/// `312 * 5 == 1560`, 8px short of the real 1568px page width -- not a typo, pxpipe's page leaves a margin around the glyph grid rather than filling edge-to-edge. Split evenly, 4px a side.
const H_MARGIN: u32 = (PAGE_WIDTH - COLUMNS * GLYPH_WIDTH) / 2;

// Page geometry is fixed at compile time, so its internal consistency is a compile-time fact, not a runtime test.
const _: () = assert!(ROWS_PER_PAGE * GLYPH_HEIGHT == PAGE_HEIGHT);
const _: () = assert!(COLUMNS * GLYPH_WIDTH <= PAGE_WIDTH);

/// Chars-per-token threshold below which content is "dense" enough to render as pixels. squishi has no access to Claude's real tokenizer (Anthropic doesn't publish one), so this is calibrated against `estimate_chars_per_token`'s own proxy measurement on real fixtures rather than pxpipe's own ~1/~3.5 numbers (measured against a real tokenizer this module doesn't have). Measured: dense JSON 1.951 chars/token, source code 3.375, README prose 5.203 -- `2.5` sits with healthy margin on both sides.
const PROFITABLE_CHARS_PER_TOKEN: f64 = 2.5;

/// Proxy for "characters per token" without a real Claude tokenizer. Every maximal run of alphanumeric characters counts as one chunk, every other non-whitespace character counts as its own chunk, whitespace contributes chars but no chunk -- `chars_per_token = total_chars / chunk_count`. Punctuation-dense content drives this down, long alphabetic runs (prose) drive it up, the same direction pxpipe's real gate cares about.
fn estimate_chars_per_token(text: &str) -> f64 {
    let total_chars = text.chars().count();
    if total_chars == 0 {
        return f64::INFINITY;
    }

    let mut chunks = 0usize;
    let mut in_word = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if !in_word {
                chunks += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if !ch.is_whitespace() {
                chunks += 1;
            }
        }
    }

    total_chars as f64 / chunks.max(1) as f64
}

/// pxpipe's density gate, ported as a heuristic proxy (see [`estimate_chars_per_token`]). Declines empty input.
pub fn is_profitable(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    estimate_chars_per_token(text) <= PROFITABLE_CHARS_PER_TOKEN
}

/// The safety gate pxpipe states about itself: never eligible for content unsafe to lose byte-exact recall on. `SearchResults` is excluded too -- file:line references are the kind of short identifier this isn't safe for.
pub fn eligible_kind(kind: &ContentKind) -> bool {
    matches!(
        kind,
        ContentKind::Json | ContentKind::Log | ContentKind::PlainText
    )
}

/// Render `text` onto Spleen-5×8, 312-column, 1568×728px pages and encode as a single PNG. Content needing more than one page's row capacity spills onto additional pages, stacked vertically into one taller image (`render_to_png` returns one `Vec<u8>`, unlike pxpipe's separate per-page attachments -- no consumer needs discrete pages yet).
///
/// A `\n` always starts a new row; a line longer than `COLUMNS` wraps onto additional rows.
pub fn render_to_png(text: &str) -> Vec<u8> {
    let rows = wrap_into_rows(text);
    let page_count = rows.len().div_ceil(ROWS_PER_PAGE as usize).max(1);
    let height = PAGE_HEIGHT * page_count as u32;

    let mut image = GrayImage::from_pixel(PAGE_WIDTH, height, Luma([255u8]));
    let mut font = PSF2Font::new(FONT_5X8).expect("bundled Spleen 5x8 PSF2 data is always valid");

    for (row_idx, row) in rows.iter().enumerate() {
        let y0 = row_idx as u32 * GLYPH_HEIGHT;
        for (col_idx, ch) in row.chars().enumerate().take(COLUMNS as usize) {
            let x0 = H_MARGIN + col_idx as u32 * GLYPH_WIDTH;
            blit_glyph(&mut image, &mut font, ch, x0, y0);
        }
    }

    encode_png(&image)
}

/// Split `text` into display rows: `\n` forces a row break, and any line longer than `COLUMNS` wraps onto as many additional rows as it needs.
fn wrap_into_rows(text: &str) -> Vec<String> {
    let mut rows = Vec::new();
    for line in text.split('\n') {
        if line.is_empty() {
            rows.push(String::new());
            continue;
        }
        let chars: Vec<char> = line.chars().collect();
        for chunk in chars.chunks(COLUMNS as usize) {
            rows.push(chunk.iter().collect());
        }
    }
    rows
}

/// Blit one glyph's pixels at `(x0, y0)`. Characters the bundled font has no glyph for are left blank rather than erroring.
fn blit_glyph(image: &mut GrayImage, font: &mut PSF2Font, ch: char, x0: u32, y0: u32) {
    let mut buf = [0u8; 4];
    let bytes = ch.encode_utf8(&mut buf).as_bytes();
    let Some(glyph) = font.glyph_for_utf8(bytes) else {
        return;
    };
    for (row_y, row) in glyph.enumerate() {
        for (col_x, on) in row.enumerate() {
            if on {
                image.put_pixel(x0 + col_x as u32, y0 + row_y as u32, Luma([0u8]));
            }
        }
    }
}

fn encode_png(image: &GrayImage) -> Vec<u8> {
    let mut bytes = Vec::new();
    image
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .expect("PNG-encoding an in-memory GrayImage cannot fail");
    bytes
}

/// Composes detection + both gates + render, mirroring `toon::encode_if_smaller`'s "returns `None` on decline" shape.
pub fn render_if_profitable(text: &str) -> Option<Vec<u8>> {
    let kind = content_detect::detect(text);
    if !eligible_kind(&kind) {
        return None;
    }
    if !is_profitable(text) {
        return None;
    }
    Some(render_to_png(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Real fixtures -----------------------------------------------

    // Dense: a real excerpt of graphify's own graph.json, minified to one line per record.
    const DENSE_JSON_FIXTURE: &str = r#"{"id":"src/toon.rs::encode","kind":"function","file":"src/toon.rs","line":42,"community":3,"degree":17,"betweenness":0.083,"tags":["json","codec","public-api"]}"#;

    // Sparse: a real excerpt of this crate's own README.md prose.
    const SPARSE_PROSE_FIXTURE: &str = "squishi compresses text. It does not store or \
        retrieve anything - that is total-recall's job. Compression should always be \
        measured against real fixtures, not synthetic examples built to flatter a format, \
        and every regression found in review should be fixed, not just documented and left \
        in place for someone else to trip over later.";

    // Code: a real excerpt of this crate's own source (this very file).
    const CODE_FIXTURE: &str = "pub fn is_profitable(text: &str) -> bool {\n    \
        if text.is_empty() {\n        return false;\n    }\n    \
        estimate_chars_per_token(text) <= PROFITABLE_CHARS_PER_TOKEN\n}";

    const DIFF_FIXTURE: &str = "diff --git a/src/pixel.rs b/src/pixel.rs\n\
        --- a/src/pixel.rs\n+++ b/src/pixel.rs\n@@ -1,3 +1,4 @@\n \
        use image::GrayImage;\n+use std::io::Cursor;\n";

    // --- estimate_chars_per_token / is_profitable ---------------------

    #[test]
    fn dense_json_measures_well_under_the_profitability_threshold() {
        let density = estimate_chars_per_token(DENSE_JSON_FIXTURE);
        // Real measured number, recorded so a future heuristic change has something concrete to diff against.
        assert!(
            density < PROFITABLE_CHARS_PER_TOKEN,
            "dense JSON measured {density} chars/token, expected well under {PROFITABLE_CHARS_PER_TOKEN}"
        );
        assert!(is_profitable(DENSE_JSON_FIXTURE));
    }

    #[test]
    fn sparse_prose_is_declined() {
        let density = estimate_chars_per_token(SPARSE_PROSE_FIXTURE);
        assert!(
            density > PROFITABLE_CHARS_PER_TOKEN,
            "sparse prose measured {density} chars/token, expected well over {PROFITABLE_CHARS_PER_TOKEN}"
        );
        assert!(!is_profitable(SPARSE_PROSE_FIXTURE));
    }

    #[test]
    fn empty_text_is_never_profitable() {
        assert!(!is_profitable(""));
    }

    // --- eligible_kind / render_if_profitable (the safety gate) -------

    #[test]
    fn code_classified_content_is_refused_regardless_of_density() {
        // Proves the eligibility gate is what's under test, not density: real source is dense code, so density alone would pass it.
        let kind = content_detect::detect(CODE_FIXTURE);
        assert!(
            !eligible_kind(&kind),
            "expected code content to be classified ineligible, got {kind:?}"
        );
        assert!(render_if_profitable(CODE_FIXTURE).is_none());
    }

    #[test]
    fn diff_classified_content_is_refused_regardless_of_density() {
        let kind = content_detect::detect(DIFF_FIXTURE);
        assert_eq!(kind, ContentKind::Diff);
        assert!(!eligible_kind(&kind));
        assert!(render_if_profitable(DIFF_FIXTURE).is_none());
    }

    #[test]
    fn sparse_prose_is_eligible_but_declined_on_density() {
        // PlainText is eligible -- it's density, not kind, that declines it here.
        let kind = content_detect::detect(SPARSE_PROSE_FIXTURE);
        assert_eq!(kind, ContentKind::PlainText);
        assert!(eligible_kind(&kind));
        assert!(render_if_profitable(SPARSE_PROSE_FIXTURE).is_none());
    }

    #[test]
    fn dense_eligible_content_renders() {
        assert!(render_if_profitable(DENSE_JSON_FIXTURE).is_some());
    }

    // --- render_to_png: real, decodable PNG output ---------------------

    #[test]
    fn renders_a_single_page_png_with_real_dimensions() {
        let png = render_to_png(DENSE_JSON_FIXTURE);
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "not a real PNG signature");

        let decoded = image::load_from_memory(&png).expect("rendered bytes must decode as PNG");
        assert_eq!(decoded.width(), PAGE_WIDTH);
        assert_eq!(decoded.height(), PAGE_HEIGHT);
    }

    #[test]
    fn content_longer_than_one_page_spills_onto_a_second_page() {
        // Text guaranteed to exceed one page's row capacity.
        let text = "x".repeat(COLUMNS as usize * (ROWS_PER_PAGE as usize + 1));
        let png = render_to_png(&text);
        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!(decoded.width(), PAGE_WIDTH);
        assert_eq!(decoded.height(), PAGE_HEIGHT * 2);
    }

    #[test]
    fn empty_text_still_renders_one_blank_page() {
        let png = render_to_png("");
        let decoded = image::load_from_memory(&png).unwrap();
        assert_eq!(decoded.width(), PAGE_WIDTH);
        assert_eq!(decoded.height(), PAGE_HEIGHT);
        // Fully blank (white) -- no glyphs were blitted.
        assert!(decoded.to_luma8().pixels().all(|p| p.0[0] == 255));
    }

    #[test]
    fn rendering_actually_draws_dark_pixels_for_real_text() {
        let png = render_to_png("hello");
        let decoded = image::load_from_memory(&png).unwrap().to_luma8();
        assert!(
            decoded.pixels().any(|p| p.0[0] == 0),
            "expected at least one black pixel from rendering real glyphs"
        );
    }

    #[test]
    fn newline_forces_a_row_break_not_a_wrapped_glyph() {
        let rows = wrap_into_rows("ab\ncd");
        assert_eq!(rows, vec!["ab".to_string(), "cd".to_string()]);
    }

    #[test]
    fn long_line_wraps_at_the_column_limit() {
        let line = "x".repeat(COLUMNS as usize + 3);
        let rows = wrap_into_rows(&line);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].chars().count(), COLUMNS as usize);
        assert_eq!(rows[1].chars().count(), 3);
    }
}
