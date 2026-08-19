//! Unified-diff compression: cap file count and hunks-per-file, trim context lines around each change, keep the highest-signal hunks when a file has too many.
//!
//! squishi has no store (that's total-recall's job), so this drops any cache/retrieve-marker machinery a fuller port might carry.

use regex::Regex;
use std::collections::BTreeSet;
use std::sync::LazyLock;

// ─── Scoring constants ──────────────────────────────────────────────────
// A reasonable starting point, not derived from squishi's own traffic. Revisit after seeing real compression behavior.

const SCORE_CHANGE_DENSITY_WEIGHT: f64 = 0.03;
const SCORE_CHANGE_DENSITY_CAP: f64 = 0.3;
const SCORE_CONTEXT_WORD_WEIGHT: f64 = 0.2;
const SCORE_CONTEXT_MIN_WORD_LEN: usize = 2;
const SCORE_PRIORITY_PATTERN_BOOST: f64 = 0.3;
const SCORE_TOTAL_CAP: f64 = 1.0;

static HUNK_HEADER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^(?:",
        r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@",
        r"|",
        r"@@@ -\d+(?:,\d+)? -\d+(?:,\d+)? \+\d+(?:,\d+)? @@@",
        r"|",
        r"@@@@ -\d+(?:,\d+)? -\d+(?:,\d+)? -\d+(?:,\d+)? \+\d+(?:,\d+)? @@@@",
        r")(.*)$"
    ))
    .unwrap()
});
static HUNK_NEW_RANGE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\+(\d+)").unwrap());
static DIFF_GIT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^diff --git a/(.+) b/(.+)$").unwrap());
static DIFF_COMBINED_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^diff --combined (.+)$").unwrap());
static DIFF_CC_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^diff --cc (.+)$").unwrap());
static OLD_FILE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^--- (a/(.+)|/dev/null)$").unwrap());
static NEW_FILE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\+\+\+ (b/(.+)|/dev/null)$").unwrap());
static BINARY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^Binary files .+ differ$").unwrap());
static PRIORITY_PATTERNS: LazyLock<[Regex; 3]> = LazyLock::new(|| {
    [
        Regex::new(r"(?i)\b(error|exception|fail(?:ed|ure)?|fatal|critical|crash|panic)\b")
            .unwrap(),
        Regex::new(r"(?i)\b(important|note|todo|fixme|hack|xxx|bug|fix)\b").unwrap(),
        Regex::new(r"(?i)\b(security|auth|password|secret|token)\b").unwrap(),
    ]
});

fn is_diff_header(line: &str) -> bool {
    DIFF_GIT_RE.is_match(line) || DIFF_COMBINED_RE.is_match(line) || DIFF_CC_RE.is_match(line)
}

#[derive(Debug, Clone)]
struct DiffHunk {
    header: String,
    lines: Vec<String>,
    score: f64,
}

#[derive(Debug, Clone)]
struct DiffFile {
    header: String,
    old_file: String,
    new_file: String,
    hunks: Vec<DiffHunk>,
    is_binary: bool,
    is_new_file: bool,
    is_deleted_file: bool,
    rename_lines: Vec<String>,
}

impl DiffFile {
    fn total_changes(&self) -> usize {
        self.hunks
            .iter()
            .map(|h| {
                h.lines
                    .iter()
                    .filter(|l| l.starts_with('+') || l.starts_with('-'))
                    .count()
            })
            .sum()
    }
}

struct ParsedDiff {
    pre_diff_lines: Vec<String>,
    files: Vec<DiffFile>,
}

fn parse_diff(lines: &[&str]) -> ParsedDiff {
    let mut files: Vec<DiffFile> = Vec::new();
    let mut current_file: Option<DiffFile> = None;
    let mut current_hunk: Option<DiffHunk> = None;
    let mut pre_diff_lines: Vec<String> = Vec::new();

    for &line in lines {
        if is_diff_header(line) {
            if let Some(h) = current_hunk.take()
                && let Some(f) = current_file.as_mut()
            {
                f.hunks.push(h);
            }
            if let Some(f) = current_file.take() {
                files.push(f);
            }
            current_file = Some(DiffFile {
                header: line.to_string(),
                old_file: String::new(),
                new_file: String::new(),
                hunks: Vec::new(),
                is_binary: false,
                is_new_file: false,
                is_deleted_file: false,
                rename_lines: Vec::new(),
            });
            continue;
        }

        if current_file.is_none() {
            pre_diff_lines.push(line.to_string());
            continue;
        }

        if let Some(f) = current_file.as_mut() {
            if line.starts_with("new file mode") {
                f.is_new_file = true;
            } else if line.starts_with("deleted file mode") {
                f.is_deleted_file = true;
            } else if line.starts_with("rename ")
                || line.starts_with("similarity ")
                || line.starts_with("copy ")
                || line.starts_with("dissimilarity ")
            {
                f.rename_lines.push(line.to_string());
            } else if BINARY_RE.is_match(line) {
                f.is_binary = true;
            }
        }

        if OLD_FILE_RE.is_match(line) {
            if let Some(f) = current_file.as_mut() {
                f.old_file = line.to_string();
            }
            continue;
        }
        if NEW_FILE_RE.is_match(line) {
            if let Some(f) = current_file.as_mut() {
                f.new_file = line.to_string();
            }
            continue;
        }

        if HUNK_HEADER_RE.is_match(line) {
            if let Some(h) = current_hunk.take()
                && let Some(f) = current_file.as_mut()
            {
                f.hunks.push(h);
            }
            current_hunk = Some(DiffHunk {
                header: line.to_string(),
                lines: Vec::new(),
                score: 0.0,
            });
            continue;
        }

        if let Some(h) = current_hunk.as_mut() {
            h.lines.push(line.to_string());
        }
    }

    if let Some(h) = current_hunk.take()
        && let Some(f) = current_file.as_mut()
    {
        f.hunks.push(h);
    }
    if let Some(f) = current_file.take() {
        files.push(f);
    }

    ParsedDiff {
        pre_diff_lines,
        files,
    }
}

fn score_hunks(files: &mut [DiffFile], context: &str) {
    let context_lower = context.to_lowercase();
    let context_words: Vec<&str> = context_lower.split_whitespace().collect();

    for file in files.iter_mut() {
        for hunk in file.hunks.iter_mut() {
            let change_count = hunk
                .lines
                .iter()
                .filter(|l| l.starts_with('+') || l.starts_with('-'))
                .count();
            let mut score =
                (change_count as f64 * SCORE_CHANGE_DENSITY_WEIGHT).min(SCORE_CHANGE_DENSITY_CAP);

            let hunk_content_lower = hunk.lines.join("\n").to_lowercase();
            for word in &context_words {
                if word.len() > SCORE_CONTEXT_MIN_WORD_LEN && hunk_content_lower.contains(word) {
                    score += SCORE_CONTEXT_WORD_WEIGHT;
                }
            }
            for pat in PRIORITY_PATTERNS.iter() {
                if pat.is_match(&hunk_content_lower) {
                    score += SCORE_PRIORITY_PATTERN_BOOST;
                    break;
                }
            }
            hunk.score = score.min(SCORE_TOTAL_CAP);
        }
    }
}

fn extract_line_number(header: &str) -> usize {
    HUNK_NEW_RANGE_RE
        .captures(header)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<usize>().ok())
        .unwrap_or(0)
}

/// Keep first + last + top-scored middle hunks, re-sorted back into
/// original appearance order. Returns (kept, dropped_count).
fn select_hunks(hunks: Vec<DiffHunk>, max_per_file: usize) -> (Vec<DiffHunk>, usize) {
    if hunks.len() <= max_per_file || hunks.is_empty() {
        return (hunks, 0);
    }

    let mut indexed: Vec<DiffHunk> = hunks;
    let first = indexed.remove(0);
    let last = if !indexed.is_empty() {
        Some(indexed.remove(indexed.len() - 1))
    } else {
        None
    };
    let middle = indexed;
    let dropped_count_base = if last.is_some() { 2 } else { 1 };
    let remaining_slots = max_per_file.saturating_sub(dropped_count_base);

    let mut middle_sorted = middle;
    middle_sorted.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let dropped = middle_sorted.len().saturating_sub(remaining_slots);
    let kept_middle: Vec<DiffHunk> = middle_sorted.into_iter().take(remaining_slots).collect();

    let mut selected = vec![first];
    selected.extend(kept_middle);
    if let Some(l) = last {
        selected.push(l);
    }
    selected.sort_by_key(|h| extract_line_number(&h.header));

    (selected, dropped)
}

fn reduce_context(hunk: &DiffHunk, max_context: usize) -> DiffHunk {
    let change_positions: Vec<usize> = hunk
        .lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| (l.starts_with('+') || l.starts_with('-')).then_some(i))
        .collect();

    if change_positions.is_empty() {
        let take = max_context.min(hunk.lines.len());
        return DiffHunk {
            header: hunk.header.clone(),
            lines: hunk.lines.iter().take(take).cloned().collect(),
            score: hunk.score,
        };
    }

    let mut keep = BTreeSet::new();
    for &pos in &change_positions {
        keep.insert(pos);
        let lo = pos.saturating_sub(max_context);
        keep.extend(lo..pos);
        let hi = (pos + max_context + 1).min(hunk.lines.len());
        keep.extend((pos + 1)..hi);
    }
    // Always keep structural markers like `\ No newline at end of file` regardless of distance from a change.
    for (i, line) in hunk.lines.iter().enumerate() {
        if line.starts_with('\\') {
            keep.insert(i);
        }
    }

    DiffHunk {
        header: hunk.header.clone(),
        lines: keep.into_iter().map(|i| hunk.lines[i].clone()).collect(),
        score: hunk.score,
    }
}

fn format_output(
    pre_diff_lines: &[String],
    files: &[DiffFile],
    files_affected: usize,
    hunks_removed: usize,
) -> String {
    let mut out: Vec<String> = pre_diff_lines.to_vec();

    for f in files {
        out.push(f.header.clone());
        out.extend(f.rename_lines.iter().cloned());

        if f.is_new_file {
            out.push("new file mode 100644".into());
        } else if f.is_deleted_file {
            out.push("deleted file mode 100644".into());
        }

        if f.is_binary {
            out.push("Binary files differ".into());
            continue;
        }

        if !f.old_file.is_empty() {
            out.push(f.old_file.clone());
        }
        if !f.new_file.is_empty() {
            out.push(f.new_file.clone());
        }

        for h in &f.hunks {
            out.push(h.header.clone());
            out.extend(h.lines.iter().cloned());
        }
    }

    if files_affected > 0 {
        let mut parts = vec![format!("{files_affected} files changed")];
        if hunks_removed > 0 {
            parts.push(format!("{hunks_removed} hunks omitted"));
        }
        out.push(format!("[{}]", parts.join(", ")));
    }

    out.join("\n")
}

pub struct DiffCompressConfig {
    pub max_context_lines: usize,
    pub max_hunks_per_file: usize,
    pub max_files: usize,
}

impl Default for DiffCompressConfig {
    fn default() -> Self {
        Self {
            max_context_lines: 2,
            max_hunks_per_file: 10,
            max_files: 20,
        }
    }
}

pub struct DiffCompressResult {
    pub content: String,
    pub original_line_count: usize,
    pub compressed_line_count: usize,
    pub files_affected: usize,
    pub hunks_removed: usize,
}

pub fn compress_diff(
    content: &str,
    context: &str,
    config: &DiffCompressConfig,
) -> DiffCompressResult {
    let lines: Vec<&str> = content.lines().collect();
    let original_line_count = lines.len();

    let parsed = parse_diff(&lines);
    let mut files = parsed.files;

    // Nothing recognizable as a diff -- pass through unchanged rather than emit an empty result.
    if files.is_empty() {
        return DiffCompressResult {
            content: content.to_string(),
            original_line_count,
            compressed_line_count: original_line_count,
            files_affected: 0,
            hunks_removed: 0,
        };
    }

    score_hunks(&mut files, context);

    if files.len() > config.max_files {
        files.sort_by_key(|f| std::cmp::Reverse(f.total_changes()));
        files.truncate(config.max_files);
    }

    let mut hunks_removed_total = 0usize;
    let compressed_files: Vec<DiffFile> = files
        .into_iter()
        .map(|file| {
            let (selected, dropped) = select_hunks(file.hunks, config.max_hunks_per_file);
            hunks_removed_total += dropped;
            let trimmed_hunks: Vec<DiffHunk> = selected
                .iter()
                .map(|h| reduce_context(h, config.max_context_lines))
                .collect();
            DiffFile {
                hunks: trimmed_hunks,
                ..file
            }
        })
        .collect();

    let files_affected = compressed_files.len();
    let compressed = format_output(
        &parsed.pre_diff_lines,
        &compressed_files,
        files_affected,
        hunks_removed_total,
    );
    let compressed_line_count = compressed.lines().count();

    DiffCompressResult {
        content: compressed,
        original_line_count,
        compressed_line_count,
        files_affected,
        hunks_removed: hunks_removed_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_n_hunk_diff(n: usize) -> String {
        let mut s = String::from("diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n");
        for i in 0..n {
            let start = i * 100 + 1;
            s.push_str(&format!("@@ -{0},6 +{0},6 @@\n", start));
            s.push_str(&format!(
                " ctx_a_{i}\n ctx_b_{i}\n-old_{i}\n+new_{i}\n ctx_c_{i}\n ctx_d_{i}\n"
            ));
        }
        s
    }

    fn build_synthetic_diff(n_files: usize) -> String {
        let mut s = String::new();
        for i in 0..n_files {
            s.push_str(&format!(
                "diff --git a/file_{i}.py b/file_{i}.py\n--- a/file_{i}.py\n+++ b/file_{i}.py\n@@ -1,10 +1,12 @@\n"
            ));
            for k in 0..5 {
                s.push_str(&format!(" context_{k}_{i}\n"));
            }
            for k in 0..3 {
                s.push_str(&format!("-removed_{k}_{i}\n"));
            }
            for k in 0..5 {
                s.push_str(&format!("+added_{k}_{i}\n"));
            }
        }
        s
    }

    #[test]
    fn non_diff_input_passes_through() {
        let input = "this is not a diff\njust prose\n".repeat(20);
        let r = compress_diff(&input, "", &DiffCompressConfig::default());
        assert_eq!(r.content, input);
        assert_eq!(r.files_affected, 0);
    }

    #[test]
    fn max_hunks_per_file_cap_drops_excess() {
        let cfg = DiffCompressConfig {
            max_hunks_per_file: 10,
            ..Default::default()
        };
        let input = build_n_hunk_diff(15);
        let r = compress_diff(&input, "", &cfg);
        assert_eq!(r.hunks_removed, 5);
    }

    #[test]
    fn max_files_cap_keeps_heaviest_files() {
        let cfg = DiffCompressConfig {
            max_files: 20,
            ..Default::default()
        };
        let input = build_synthetic_diff(25);
        let r = compress_diff(&input, "", &cfg);
        assert_eq!(r.files_affected, 20);
    }

    #[test]
    fn rename_markers_are_preserved_in_output() {
        let input = "diff --git a/old.py b/new.py\n\
                     similarity index 92%\n\
                     rename from old.py\n\
                     rename to new.py\n\
                     --- a/old.py\n\
                     +++ b/new.py\n\
                     @@ -1,3 +1,3 @@\n\
                      ctx_a\n\
                     -old_line\n\
                     +new_line\n\
                      ctx_b\n";
        let r = compress_diff(input, "", &DiffCompressConfig::default());
        assert!(r.content.contains("similarity index 92%"));
        assert!(r.content.contains("rename from old.py"));
        assert!(r.content.contains("rename to new.py"));
    }

    #[test]
    fn combined_diff_3way_content_is_parsed_and_emitted() {
        let input = "diff --git a/merge.py b/merge.py\n\
                     --- a/merge.py\n\
                     +++ b/merge.py\n\
                     @@@ -1,3 -1,3 +1,4 @@@\n\
                       unchanged_a\n\
                      -old_branch_1\n\
                     - old_branch_2\n\
                     ++new_in_merge\n\
                       unchanged_b\n";
        let r = compress_diff(input, "", &DiffCompressConfig::default());
        assert!(r.content.contains("@@@ -1,3 -1,3 +1,4 @@@"));
        assert!(r.content.contains("++new_in_merge"));
        assert_eq!(r.files_affected, 1);
    }

    #[test]
    fn no_newline_marker_preserved_despite_distance() {
        let input = "diff --git a/last.txt b/last.txt\n\
                     --- a/last.txt\n\
                     +++ b/last.txt\n\
                     @@ -1,8 +1,8 @@\n\
                     -old_first\n\
                     +new_first\n\
                      ctx_a\n ctx_b\n ctx_c\n ctx_d\n ctx_e\n ctx_f\n\
                     \\ No newline at end of file\n";
        let r = compress_diff(input, "", &DiffCompressConfig::default());
        assert!(r.content.contains("\\ No newline at end of file"));
    }

    #[test]
    fn diff_combined_header_starts_a_file() {
        let input = "diff --combined merge.py\n\
                     --- a/merge.py\n\
                     +++ b/merge.py\n\
                     @@@ -1,3 -1,3 +1,4 @@@\n\
                       ctx_a\n\
                     - removed_p1\n\
                      -removed_p2\n\
                     ++added_in_merge\n\
                       ctx_b\n";
        let r = compress_diff(input, "", &DiffCompressConfig::default());
        assert_eq!(r.files_affected, 1);
        assert!(r.content.contains("diff --combined merge.py"));
    }

    #[test]
    fn pre_diff_content_is_preserved() {
        let input = "commit abc1234567890\n\
                     Author: Tester <t@example.com>\n\
                     \n    Refactor: rename and modify\n\n\
                     diff --git a/x.py b/x.py\n\
                     --- a/x.py\n\
                     +++ b/x.py\n\
                     @@ -1 +1 @@\n\
                     -a\n\
                     +b\n";
        let r = compress_diff(input, "", &DiffCompressConfig::default());
        assert!(r.content.starts_with("commit abc1234567890"));
        assert!(r.content.contains("Author: Tester"));
        assert!(r.content.contains("diff --git a/x.py b/x.py"));
    }

    #[test]
    fn large_diff_compresses_meaningfully() {
        let input = build_n_hunk_diff(20);
        let r = compress_diff(&input, "", &DiffCompressConfig::default());
        assert!(r.compressed_line_count < r.original_line_count);
        assert!(r.hunks_removed > 0);
    }
}
