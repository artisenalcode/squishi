//! Search-result compression: grep/ripgrep-style `path:line:text` output.
//! Group by file, cap matches per file, keep everything else (files with
//! few matches are already cheap — capping only kicks in where it helps).

use regex::Regex;
use std::collections::BTreeMap;
use std::sync::LazyLock;

const MAX_MATCHES_PER_FILE: usize = 5;

static SEARCH_RESULT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([^:]+):(\d+):").unwrap());

pub struct SearchCompressResult {
    pub original_lines: usize,
    pub compressed_lines: usize,
    pub content: String,
}

pub fn compress_search_results(content: &str) -> SearchCompressResult {
    let lines: Vec<&str> = content.lines().collect();
    let original_lines = lines.len();

    // Group line indices by file, preserving each file's first-seen order.
    let mut by_file: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut file_order: Vec<&str> = Vec::new();
    let mut unmatched: Vec<&str> = Vec::new();

    for line in &lines {
        if let Some(caps) = SEARCH_RESULT_RE.captures(line) {
            let file = caps.get(1).unwrap().as_str();
            if !by_file.contains_key(file) {
                file_order.push(file);
            }
            by_file.entry(file).or_default().push(line);
        } else {
            unmatched.push(line);
        }
    }

    let mut output_lines: Vec<String> = Vec::new();
    for file in &file_order {
        let matches = &by_file[file];
        if matches.len() > MAX_MATCHES_PER_FILE {
            output_lines.extend(
                matches[..MAX_MATCHES_PER_FILE]
                    .iter()
                    .map(|l| l.to_string()),
            );
            output_lines.push(format!(
                "{}: {} more matches omitted",
                file,
                matches.len() - MAX_MATCHES_PER_FILE
            ));
        } else {
            output_lines.extend(matches.iter().map(|l| l.to_string()));
        }
    }
    output_lines.extend(unmatched.iter().map(|l| l.to_string()));

    let compressed_lines = output_lines.len();
    SearchCompressResult {
        original_lines,
        compressed_lines,
        content: output_lines.join("\n"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn few_matches_per_file_are_kept_as_is() {
        let content = "a.rs:1:fn foo() {}\nb.rs:2:fn bar() {}\n";
        let result = compress_search_results(content);
        assert_eq!(result.original_lines, 2);
        assert_eq!(result.compressed_lines, 2);
        assert!(!result.content.contains("omitted"));
    }

    #[test]
    fn caps_many_matches_in_one_file() {
        let mut content = String::new();
        for i in 0..20 {
            content.push_str(&format!("a.rs:{i}:match number {i}\n"));
        }
        let result = compress_search_results(&content);
        assert_eq!(result.original_lines, 20);
        // MAX_MATCHES_PER_FILE kept + 1 omission marker
        assert_eq!(result.compressed_lines, MAX_MATCHES_PER_FILE + 1);
        assert!(result.content.contains("15 more matches omitted"));
    }

    #[test]
    fn preserves_per_file_grouping_across_interleaved_input() {
        let content = "a.rs:1:x\nb.rs:1:y\na.rs:2:x\nb.rs:2:y\n";
        let result = compress_search_results(content);
        // a.rs matches should be grouped together, then b.rs's.
        let a_pos = result.content.find("a.rs:1").unwrap();
        let a2_pos = result.content.find("a.rs:2").unwrap();
        let b_pos = result.content.find("b.rs:1").unwrap();
        assert!(a_pos < a2_pos);
        assert!(a2_pos < b_pos);
    }
}
