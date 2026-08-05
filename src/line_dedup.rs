//! Zero-dependency line-run deduplication.
//! No network, no subprocess, no ML — pure text transform.
//!
//! No head/tail truncation here, deliberately: squishi's pipeline hands
//! anything still over budget to `log_compress`, whose scoring-based
//! selection already bounds output size without the data loss a blind
//! truncation would cause. Don't add truncation back here unless
//! something in the pipeline actually needs a fallback for it.

const DEDUPE_RUN_THRESHOLD: usize = 5; // collapse runs of MORE than this many identical lines

/// Collapse runs of identical lines only — safe as a pre-pass before any
/// further compression, since it doesn't destroy non-repeating structure.
pub fn dedupe_line_runs(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut result = String::new();
    let mut i = 0;

    while i < lines.len() {
        let current_line = lines[i];
        let mut count = 1;
        while i + count < lines.len() && lines[i + count] == current_line {
            count += 1;
        }

        if count > DEDUPE_RUN_THRESHOLD {
            result.push_str(current_line);
            result.push('\n');
            result.push_str(&format!(
                "[... squishi pruned: {} identical lines collapsed ...]\n",
                count - 1
            ));
            i += count;
        } else {
            result.push_str(current_line);
            result.push('\n');
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_content_stays_empty() {
        assert_eq!(dedupe_line_runs(""), "");
    }

    #[test]
    fn short_content_is_unchanged_besides_trailing_newline() {
        let content = "Just a short line.\nAnother line.";
        assert_eq!(
            dedupe_line_runs(content),
            "Just a short line.\nAnother line.\n"
        );
    }

    #[test]
    fn runs_of_five_identical_lines_are_not_collapsed() {
        let content = "a\na\na\na\na\n";
        assert!(!dedupe_line_runs(content).contains("squishi pruned"));
    }

    #[test]
    fn runs_over_five_identical_lines_are_collapsed() {
        let content = "a\na\na\na\na\na\n";
        assert!(dedupe_line_runs(content).contains("squishi pruned: 5 identical lines collapsed"));
    }

    #[test]
    fn dedupe_never_lossily_truncates_non_repeating_content() {
        let mut content = String::new();
        for i in 0..2000 {
            content.push_str(&format!("This is a unique line number {i}\n"));
        }
        let deduped = dedupe_line_runs(&content);
        assert!(!deduped.contains("omitted"));
        assert_eq!(deduped.len(), content.len());
    }
}
