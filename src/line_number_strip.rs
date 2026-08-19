//! Deterministic Read-tool line-number stripping -- a zero-model pre-pass, not a new `ContentKind`, run unconditionally before `detect()` since this shape can wrap any real content underneath it.
//!
//! Claude Code's `Read` tool formats file content `cat -n`-style, `<N>\t<line content>` per line. Left unstripped, Magika classifies it as `Other("tsv")` (no compressor for that) and even the fast-path regexes misfire since a diff header or log keyword on line N no longer starts the line once `"N\t"` is in front of it. Measured on real data: 9290→9291 chars (no reduction) with the prefix present, 8910→121 stripped.
//!
//! Detection requires strictly sequential numbering from 1, not just "every line starts with digits+tab" -- a genuine numbered-ID-column TSV would need to coincidentally number every row `1..N` to be mistaken for this, and even then stripping a redundant leading sequence column loses no real information.

use regex::Regex;
use std::sync::LazyLock;

static LINE_NUMBER_PREFIX_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(\d+)\t").unwrap());

/// Below this many lines, a match is too easy to hit by pure coincidence to trust as real Read-tool output.
const MIN_LINES: u64 = 5;

/// Strips a Read-tool-shaped `N\t` prefix from every line only if every line matches it with strictly sequential numbers starting at 1 -- otherwise returns `text` completely unchanged, never a partial strip.
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
        // Lines 1-5, then jumps to 7 -- not real Read-tool output, so nothing should be stripped, not even the lines that do match.
        let mut text = numbered(&["a", "b", "c", "d", "e"]);
        text.push_str("7\tf\n");
        let (result, stripped) = strip_read_tool_line_numbers(&text);
        assert!(!stripped);
        assert_eq!(result, text);
    }

    #[test]
    fn too_few_lines_is_not_trusted_as_real_read_tool_output() {
        // "1\tfoo\n2\tbar\n" is ordinary enough to occur by coincidence -- must not fire on a tiny match.
        let text = numbered(&["foo", "bar"]);
        let (result, stripped) = strip_read_tool_line_numbers(&text);
        assert!(!stripped);
        assert_eq!(result, text);
    }

    #[test]
    fn a_real_numbered_tsv_id_column_that_happens_to_be_sequential_still_strips_safely() {
        // Genuine TSV whose leading ID column happens to be exactly 1..N. Stripping loses no real information (row order already carries the sequence) -- a safe outcome, not a bug.
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
        // Confirms an embedded fact a caller might need to recall isn't corrupted by stripping.
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
