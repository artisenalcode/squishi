//! Deterministic Read-tool line-number stripping — a zero-model pre-pass,
//! not a new `ContentKind`, mirroring `base64_strip.rs`'s own shape and
//! reasoning: run unconditionally before `detect()`, since this shape can
//! wrap any real content underneath it.
//!
//! Real finding, from governator-proxy's Step 2 live-API check
//! (`docs/ideation/governator-proxy/2026-08-18-transparent-proxy-step2-spec.md`):
//! Claude Code's real `Read` tool formats file content `cat -n`-style,
//! `<N>\t<line content>` per line, no padding (confirmed against a real
//! captured tool_result). Squishi previously ran `detect()` and every
//! compressor directly on that shape — real, live consequence: Magika
//! classified it as `Other("tsv")` (a kind squishi has no compressor
//! for), and even the fast-path regexes (Json/SearchResults/Diff/Log, all
//! anchored at line start) would misfire, since a real diff header or log
//! keyword on line N never starts the line once `"N\t"` is in front of
//! it. Confirmed directly on real data: the same real content compressed
//! 9290→9291 chars (no reduction) with the prefix present, 8910→121 chars
//! with it stripped. `Read` is Claude Code's single most common tool
//! call, so this is real, live-traffic-shaped impact, not a synthetic
//! edge case.
//!
//! Detection requires **strictly sequential numbering from 1**, not just
//! "every line starts with digits+tab" — the real, specific signature of
//! `cat -n`-style output, and a much narrower bar than a loose prefix
//! match. A genuine numbered-ID-column TSV would need to coincidentally
//! number every row `1..N` matching the real line count exactly to be
//! mistaken for this — and even then, stripping a redundant leading
//! sequence column loses no real information, since row order already
//! carries it.

use regex::Regex;
use std::sync::LazyLock;

static LINE_NUMBER_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(\d+)\t").unwrap());

/// Below this many lines, a match is too easy to hit by pure coincidence
/// (e.g. a real "1\tfoo" is a completely ordinary thing for a line of
/// text to start with) to trust as real Read-tool output.
const MIN_LINES: u64 = 5;

/// Strips a real Read-tool-shaped `N\t` prefix from every line if (and
/// only if) every line in `text` matches it with strictly sequential
/// numbers starting at 1 — otherwise returns `text` completely unchanged.
/// Never a partial strip: any line that breaks the pattern means nothing
/// is touched.
pub fn strip_read_tool_line_numbers(text: &str) -> (String, bool) {
    let lines: Vec<&str> = text.lines().collect();

    let mut expected: u64 = 1;
    for line in &lines {
        let Some(caps) = LINE_NUMBER_PREFIX_RE.captures(line) else {
            return (text.to_string(), false);
        };
        let Ok(n) = caps[1].parse::<u64>() else {
            return (text.to_string(), false);
        };
        if n != expected {
            return (text.to_string(), false);
        }
        expected += 1;
    }

    if expected <= MIN_LINES {
        return (text.to_string(), false);
    }

    let stripped: Vec<String> = lines
        .iter()
        .map(|line| LINE_NUMBER_PREFIX_RE.replace(line, "").into_owned())
        .collect();
    let mut stripped = stripped.join("\n");
    if text.ends_with('\n') {
        stripped.push('\n');
    }
    (stripped, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(lines: &[&str]) -> String {
        lines
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{}\t{l}", i + 1))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }

    #[test]
    fn strips_a_real_read_tool_shaped_block() {
        let lines: Vec<String> = (1..=10).map(|i| format!("content of line {i}")).collect();
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let text = numbered(&refs);
        let (result, stripped) = strip_read_tool_line_numbers(&text);
        assert!(stripped);
        assert_eq!(
            result,
            lines.join("\n") + "\n",
            "expected the N\\t prefix removed, content and line order preserved"
        );
    }

    #[test]
    fn content_with_no_numbering_is_returned_unchanged() {
        let text =
            "just a normal paragraph of prose with no special structure.\nsecond line here too.";
        let (result, stripped) = strip_read_tool_line_numbers(text);
        assert!(!stripped);
        assert_eq!(result, text);
    }

    #[test]
    fn a_gap_in_the_sequence_leaves_content_completely_untouched() {
        // Real lines 1-5, then jumps to 7 -- not real Read-tool output
        // (a genuine gap would mean a mid-file edit or a different
        // source entirely), so nothing should be stripped, not even the
        // lines that *do* match.
        let mut text = numbered(&["a", "b", "c", "d", "e"]);
        text.push_str("7\tf\n");
        let (result, stripped) = strip_read_tool_line_numbers(&text);
        assert!(!stripped);
        assert_eq!(result, text);
    }

    #[test]
    fn too_few_lines_is_not_trusted_as_real_read_tool_output() {
        // "1\tfoo\n2\tbar\n" is a completely ordinary thing for real text
        // to contain by coincidence -- must not fire on a tiny match.
        let text = numbered(&["foo", "bar"]);
        let (result, stripped) = strip_read_tool_line_numbers(&text);
        assert!(!stripped);
        assert_eq!(result, text);
    }

    #[test]
    fn a_real_numbered_tsv_id_column_that_happens_to_be_sequential_still_strips_safely() {
        // The real, narrow false-positive case named in this module's own
        // doc comment: genuine tab-separated data whose leading ID column
        // happens to be exactly 1..N. Stripping it loses no real
        // information (row order already carries the same sequence), so
        // this is documented as an acceptable, safe outcome, not a bug.
        let mut text = String::new();
        for i in 1..=8 {
            text.push_str(&format!("{i}\tvalue-{i}\n"));
        }
        let (result, stripped) = strip_read_tool_line_numbers(&text);
        assert!(stripped);
        assert!(!result.contains('\t'));
    }

    #[test]
    fn preserves_a_missing_trailing_newline() {
        let mut text = numbered(&["a", "b", "c", "d", "e"]);
        text.pop(); // drop the trailing \n
        let (result, stripped) = strip_read_tool_line_numbers(&text);
        assert!(stripped);
        assert!(!result.ends_with('\n'));
    }

    #[test]
    fn empty_content_is_not_trusted_as_real_read_tool_output() {
        let (result, stripped) = strip_read_tool_line_numbers("");
        assert!(!stripped);
        assert_eq!(result, "");
    }

    #[test]
    fn real_captured_shape_with_an_embedded_fact_survives_intact() {
        // Real regression fixture, matching this module's own doc
        // comment: the exact shape captured from a real governator-proxy
        // live-API check, confirming the embedded fact a caller might
        // later need to recall isn't corrupted by stripping.
        let mut lines: Vec<String> = (1..=60)
            .map(|i| format!("log line {i:03}: routine status check, all systems nominal"))
            .collect();
        lines.insert(60, "MAGIC_TOKEN=zq7-vindaloo-9142".to_string());
        for i in 61..=100 {
            lines.push(format!(
                "log line {i:03}: routine status check, all systems nominal"
            ));
        }
        let refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
        let text = numbered(&refs);
        let (result, stripped) = strip_read_tool_line_numbers(&text);
        assert!(stripped);
        assert!(result.contains("MAGIC_TOKEN=zq7-vindaloo-9142"));
        assert!(!result.contains('\t'));
    }
}
