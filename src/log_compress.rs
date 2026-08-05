//! Log/build-output compression: classify each line's importance, keep
//! the important ones plus context, drop the rest with a summary marker.
//!
//! Own design, not a port — arrived at after reading headroom's
//! LogCompressor mechanism (classify → score → select → format) to
//! understand *why* it gets real compression on repetitive logs, then
//! implementing squishi's own version of the same shape in plain Rust,
//! no external process.

use regex::Regex;
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineLevel {
    Error,
    Warn,
    Info,
    Unknown,
}

struct ClassifiedLine<'a> {
    index: usize,
    content: &'a str,
    level: LineLevel,
    is_summary: bool,
    score: u8,
}

static ERROR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\b(error|fatal|critical|fail(ed)?)\b").unwrap());
static WARN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\bwarn(ing)?\b").unwrap());
static SUMMARY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(={3,}|-{3,}|\d+ (passed|failed|skipped)|TOTAL|Summary)").unwrap()
});

fn classify(index: usize, content: &str) -> ClassifiedLine<'_> {
    let level = if ERROR_RE.is_match(content) {
        LineLevel::Error
    } else if WARN_RE.is_match(content) {
        LineLevel::Warn
    } else if !content.trim().is_empty() {
        LineLevel::Info
    } else {
        LineLevel::Unknown
    };
    let is_summary = SUMMARY_RE.is_match(content.trim_start());
    let score = match level {
        LineLevel::Error => 100,
        LineLevel::Warn => 50,
        LineLevel::Info => 10,
        LineLevel::Unknown => 1,
    } + if is_summary { 40 } else { 0 };

    ClassifiedLine {
        index,
        content,
        level,
        is_summary,
        score,
    }
}

pub struct LogCompressConfig {
    pub max_errors: usize,
    pub max_warnings: usize,
    pub context_lines: usize,
    pub max_total_lines: usize,
}

impl Default for LogCompressConfig {
    fn default() -> Self {
        Self {
            max_errors: 10,
            max_warnings: 5,
            context_lines: 2,
            max_total_lines: 100,
        }
    }
}

pub struct LogCompressResult {
    pub original_line_count: usize,
    pub compressed_line_count: usize,
    pub content: String,
}

pub fn compress_log(content: &str, config: &LogCompressConfig) -> LogCompressResult {
    let lines: Vec<&str> = content.lines().collect();
    let classified: Vec<ClassifiedLine> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| classify(i, l))
        .collect();

    let errors: Vec<&ClassifiedLine> = classified
        .iter()
        .filter(|l| l.level == LineLevel::Error)
        .collect();
    let warnings: Vec<&ClassifiedLine> = classified
        .iter()
        .filter(|l| l.level == LineLevel::Warn)
        .collect();
    let summaries: Vec<&ClassifiedLine> = classified.iter().filter(|l| l.is_summary).collect();

    let mut selected: Vec<usize> = Vec::new();
    selected.extend(select_first_last_and_top(&errors, config.max_errors));
    selected.extend(select_first_last_and_top(&warnings, config.max_warnings));
    selected.extend(summaries.iter().map(|l| l.index));

    // Nothing flagged as error/warn/summary — there's no signal to justify
    // dropping any line, so don't. Only content with actual important/
    // unimportant contrast gets compressed.
    if selected.is_empty() {
        return LogCompressResult {
            original_line_count: lines.len(),
            compressed_line_count: lines.len(),
            content: content.to_string(),
        };
    }

    // Context lines around every selected index.
    let mut with_context: Vec<usize> = Vec::new();
    for &idx in &selected {
        let start = idx.saturating_sub(config.context_lines);
        let end = (idx + config.context_lines + 1).min(lines.len());
        with_context.extend(start..end);
    }
    with_context.sort_unstable();
    with_context.dedup();

    // Fixed budget cap (not headroom's adaptive sizer — deliberately
    // simpler for v1): if still over budget, keep the highest-scoring
    // lines within the context-expanded set.
    let final_indices: Vec<usize> = if with_context.len() > config.max_total_lines {
        let mut scored: Vec<(usize, u8)> = with_context
            .iter()
            .map(|&idx| (idx, classified[idx].score))
            .collect();
        scored.sort_by_key(|b| std::cmp::Reverse(b.1));
        scored.truncate(config.max_total_lines);
        let mut idxs: Vec<usize> = scored.into_iter().map(|(idx, _)| idx).collect();
        idxs.sort_unstable();
        idxs
    } else {
        with_context
    };

    let mut output_lines: Vec<String> = final_indices
        .iter()
        .map(|&i| classified[i].content.to_string())
        .collect();

    let omitted = lines.len() - final_indices.len();
    if omitted > 0 {
        let error_count = errors.len();
        let warn_count = warnings.len();
        output_lines.push(format!(
            "[{omitted} lines omitted: {error_count} error, {warn_count} warn]"
        ));
    }

    let compressed = output_lines.join("\n");
    LogCompressResult {
        original_line_count: lines.len(),
        compressed_line_count: final_indices.len(),
        content: compressed,
    }
}

fn select_first_last_and_top(lines: &[&ClassifiedLine], max_count: usize) -> Vec<usize> {
    if lines.len() <= max_count {
        return lines.iter().map(|l| l.index).collect();
    }

    let mut selected: Vec<usize> = Vec::new();
    if let Some(first) = lines.first() {
        selected.push(first.index);
    }
    if let Some(last) = lines.last()
        && !selected.contains(&last.index)
    {
        selected.push(last.index);
    }

    let mut remaining: Vec<&&ClassifiedLine> = lines
        .iter()
        .filter(|l| !selected.contains(&l.index))
        .collect();
    remaining.sort_by_key(|b| std::cmp::Reverse(b.score));
    let take = max_count.saturating_sub(selected.len());
    selected.extend(remaining.into_iter().take(take).map(|l| l.index));

    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_all_lines_when_under_budget() {
        let content = "line one\nline two\nline three";
        let result = compress_log(content, &LogCompressConfig::default());
        assert_eq!(result.compressed_line_count, 3);
        assert!(!result.content.contains("omitted"));
    }

    #[test]
    fn caps_repeated_errors_to_max_errors_plus_first_last() {
        let mut content = String::new();
        for i in 0..50 {
            content.push_str(&format!("ERROR: failure number {i}\n"));
        }
        let config = LogCompressConfig {
            max_errors: 5,
            context_lines: 0,
            ..Default::default()
        };
        let result = compress_log(&content, &config);
        // 5 max_errors selected + 1 omission marker line.
        assert!(result.compressed_line_count <= 7);
        assert!(result.content.contains("failure number 0"));
        assert!(result.content.contains("failure number 49"));
        assert!(result.content.contains("omitted"));
    }

    #[test]
    fn keeps_summary_lines_regardless_of_error_cap() {
        let mut content = String::new();
        for i in 0..50 {
            content.push_str(&format!("ERROR: failure number {i}\n"));
        }
        content.push_str("=== 50 failed, 0 passed ===\n");
        let config = LogCompressConfig {
            max_errors: 3,
            context_lines: 0,
            ..Default::default()
        };
        let result = compress_log(&content, &config);
        assert!(result.content.contains("50 failed"));
    }

    #[test]
    fn context_lines_surround_selected_errors() {
        let content = "info before\nERROR: boom\ninfo after";
        let config = LogCompressConfig {
            context_lines: 1,
            ..Default::default()
        };
        let result = compress_log(content, &config);
        assert!(result.content.contains("info before"));
        assert!(result.content.contains("ERROR: boom"));
        assert!(result.content.contains("info after"));
    }
}
