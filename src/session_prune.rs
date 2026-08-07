//! Structural pruning for Claude Code session transcripts — a different
//! problem from squishi's existing shape-based compressors
//! (`content_detect` + friends), which operate on a single blob's *text
//! shape*. This operates on a transcript's *structure* (which tool ran,
//! on what path, superseded by what) — real measurement earlier this
//! session found squishi's shape compressors barely touch real session
//! transcripts (the one path that loads the expensive semantic model,
//! `PlainText`, had zero qualifying blocks in a real 4713-line coding
//! session — everything reads as code/diff/log-shaped).
//!
//! Five rules, previously speced in README.md's "Remembered" section,
//! pressure-tested by the user's technical advisory board (2026-08-07,
//! see docs/ideation/agent-stack-architecture/) before being built here.
//! Rule 2 ("supersede write by read") ships but is **off by default** at
//! the CLI layer — board consensus (Fowler): real false-positive risk,
//! ship behind a flag until real usage data exists.
//!
//! Real transcript shape, confirmed by reading a live Claude Code
//! transcript directly, not assumed: one JSON record per line;
//! `message.content` is a list of blocks when the message carries tool
//! activity — `{"type":"tool_use","id","name","input"}` (input.file_path
//! present for Read/Write/Edit) and, in a later message,
//! `{"type":"tool_result","tool_use_id","content","is_error"}`.
//! Transcript JSONL isn't a versioned contract — parsing is defensive
//! throughout: an unparseable or unrecognized line is skipped, never a
//! hard error.
//!
//! No live transcript mutation happens here or anywhere in this crate —
//! confirmed via the real Claude Code hooks reference that no hook can
//! rewrite a past transcript entry. `apply_pruning` only ever produces a
//! new copy; callers are responsible for never overwriting the original.

use serde_json::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct ToolUseItem {
    pub line_index: usize,
    pub id: String,
    pub name: String,
    pub path: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResultItem {
    pub line_index: usize,
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SessionItem {
    ToolUse(ToolUseItem),
    ToolResult(ToolResultItem),
}

/// One prunable tool-result, anchored at the transcript line it lives on.
#[derive(Debug, Clone, PartialEq)]
pub struct PruneCandidate {
    pub line_index: usize,
    pub tool_use_id: String,
    pub rule: &'static str,
    pub reason: String,
    pub bytes: usize,
}

/// Parse a transcript into a flat, ordered list of tool activity.
/// Anything that isn't a recognizable tool_use/tool_result block —
/// plain text messages, unknown record shapes, malformed JSON lines —
/// is silently skipped, not an error. Transcript shape isn't a
/// versioned contract this crate controls.
pub fn parse(jsonl: &str) -> Vec<SessionItem> {
    let mut items = Vec::new();
    for (line_index, line) in jsonl.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let Some(content) = record
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_array())
        else {
            continue;
        };
        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("tool_use") => {
                    let (Some(id), Some(name)) = (
                        block.get("id").and_then(|v| v.as_str()),
                        block.get("name").and_then(|v| v.as_str()),
                    ) else {
                        continue;
                    };
                    let path = block
                        .get("input")
                        .and_then(|i| i.get("file_path"))
                        .and_then(|v| v.as_str())
                        .map(str::to_string);
                    items.push(SessionItem::ToolUse(ToolUseItem {
                        line_index,
                        id: id.to_string(),
                        name: name.to_string(),
                        path,
                    }));
                }
                Some("tool_result") => {
                    let Some(tool_use_id) = block.get("tool_use_id").and_then(|v| v.as_str())
                    else {
                        continue;
                    };
                    let content_str = block
                        .get("content")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let is_error = block
                        .get("is_error")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    items.push(SessionItem::ToolResult(ToolResultItem {
                        line_index,
                        tool_use_id: tool_use_id.to_string(),
                        content: content_str,
                        is_error,
                    }));
                }
                _ => {}
            }
        }
    }
    items
}

fn tool_use_index(items: &[SessionItem]) -> HashMap<&str, &ToolUseItem> {
    items
        .iter()
        .filter_map(|i| match i {
            SessionItem::ToolUse(t) => Some((t.id.as_str(), t)),
            SessionItem::ToolResult(_) => None,
        })
        .collect()
}

fn tool_result_index(items: &[SessionItem]) -> HashMap<&str, &ToolResultItem> {
    items
        .iter()
        .filter_map(|i| match i {
            SessionItem::ToolResult(r) => Some((r.tool_use_id.as_str(), r)),
            SessionItem::ToolUse(_) => None,
        })
        .collect()
}

/// Rule 1: an older `Read` of a path is prunable once a newer `Read` of
/// the same path exists later in the session.
pub fn dedupe_latest_read(items: &[SessionItem]) -> Vec<PruneCandidate> {
    let results = tool_result_index(items);
    let mut by_path: HashMap<&str, Vec<&ToolUseItem>> = HashMap::new();
    for item in items {
        if let SessionItem::ToolUse(t) = item
            && t.name == "Read"
            && let Some(path) = &t.path
        {
            by_path.entry(path.as_str()).or_default().push(t);
        }
    }

    let mut candidates = Vec::new();
    for (path, mut reads) in by_path {
        if reads.len() < 2 {
            continue;
        }
        reads.sort_by_key(|r| r.line_index);
        for older in &reads[..reads.len() - 1] {
            if let Some(result) = results.get(older.id.as_str()) {
                candidates.push(PruneCandidate {
                    line_index: result.line_index,
                    tool_use_id: older.id.clone(),
                    rule: "dedupe_latest_read",
                    reason: format!(
                        "older Read of {path} superseded by a later Read of the same path"
                    ),
                    bytes: result.content.len(),
                });
            }
        }
    }
    candidates
}

/// Rule 2 (off by default at the CLI layer — see module doc): a
/// `Write`/`Edit`'s own tool result is prunable once a later `Read` of
/// the same path exists.
pub fn supersede_write_by_read(items: &[SessionItem]) -> Vec<PruneCandidate> {
    let results = tool_result_index(items);
    let mut writes: Vec<&ToolUseItem> = Vec::new();
    let mut read_lines_by_path: HashMap<&str, Vec<usize>> = HashMap::new();
    for item in items {
        if let SessionItem::ToolUse(t) = item {
            match t.name.as_str() {
                "Write" | "Edit" if t.path.is_some() => writes.push(t),
                "Read" => {
                    if let Some(p) = &t.path {
                        read_lines_by_path
                            .entry(p.as_str())
                            .or_default()
                            .push(t.line_index);
                    }
                }
                _ => {}
            }
        }
    }

    let mut candidates = Vec::new();
    for w in writes {
        let path = w.path.as_ref().expect("filtered to Some above");
        let superseded = read_lines_by_path
            .get(path.as_str())
            .is_some_and(|lines| lines.iter().any(|&l| l > w.line_index));
        if superseded && let Some(result) = results.get(w.id.as_str()) {
            candidates.push(PruneCandidate {
                line_index: result.line_index,
                tool_use_id: w.id.clone(),
                rule: "supersede_write_by_read",
                reason: format!("{} of {path} verified by a later Read", w.name),
                bytes: result.content.len(),
            });
        }
    }
    candidates
}

/// Rule 3: an error tool-result is prunable if an identical `(tool
/// name, content)` error already appeared earlier in the session.
pub fn drop_redundant_errors(items: &[SessionItem]) -> Vec<PruneCandidate> {
    let uses = tool_use_index(items);
    let mut seen: HashMap<(String, String), usize> = HashMap::new();
    let mut errors: Vec<&ToolResultItem> = items
        .iter()
        .filter_map(|i| match i {
            SessionItem::ToolResult(r) if r.is_error => Some(r),
            _ => None,
        })
        .collect();
    errors.sort_by_key(|r| r.line_index);

    let mut candidates = Vec::new();
    for r in errors {
        let tool_name = uses
            .get(r.tool_use_id.as_str())
            .map(|t| t.name.clone())
            .unwrap_or_default();
        let key = (tool_name.clone(), r.content.clone());
        if let std::collections::hash_map::Entry::Vacant(e) = seen.entry(key) {
            e.insert(r.line_index);
        } else {
            candidates.push(PruneCandidate {
                line_index: r.line_index,
                tool_use_id: r.tool_use_id.clone(),
                rule: "drop_redundant_errors",
                reason: format!("identical {tool_name} error already seen earlier in the session"),
                bytes: r.content.len(),
            });
        }
    }
    candidates
}

/// Rule 4: tool-result content over `min_bytes` is prunable once it's no
/// longer within the last `window` session items — a recency window
/// over item count, not a wall-clock age cutoff.
pub fn prune_old_large_outputs(
    items: &[SessionItem],
    min_bytes: usize,
    window: usize,
) -> Vec<PruneCandidate> {
    let total = items.len();
    let mut candidates = Vec::new();
    for (idx, item) in items.iter().enumerate() {
        if let SessionItem::ToolResult(r) = item
            && r.content.len() > min_bytes
            && idx + window < total
        {
            candidates.push(PruneCandidate {
                line_index: r.line_index,
                tool_use_id: r.tool_use_id.clone(),
                rule: "prune_old_large_outputs",
                reason: format!("{} bytes, outside the last {window} items", r.content.len()),
                bytes: r.content.len(),
            });
        }
    }
    candidates
}

const BACKGROUND_TASK_MARKER: &str = "running in background with ID:";

/// Rule 5: repeated "background task launched" tool-results collapse to
/// the latest one.
pub fn collapse_task_launches(items: &[SessionItem]) -> Vec<PruneCandidate> {
    let mut launches: Vec<&ToolResultItem> = items
        .iter()
        .filter_map(|i| match i {
            SessionItem::ToolResult(r) if r.content.contains(BACKGROUND_TASK_MARKER) => Some(r),
            _ => None,
        })
        .collect();
    launches.sort_by_key(|r| r.line_index);
    if launches.len() < 2 {
        return Vec::new();
    }
    launches[..launches.len() - 1]
        .iter()
        .map(|r| PruneCandidate {
            line_index: r.line_index,
            tool_use_id: r.tool_use_id.clone(),
            rule: "collapse_task_launches",
            reason: "an earlier background-task-launch notice superseded by a later one"
                .to_string(),
            bytes: r.content.len(),
        })
        .collect()
}

/// Run every rule that's on. `include_rule_2` gates
/// `supersede_write_by_read` — off by default at the CLI layer per board
/// consensus, always available as a pure function above regardless.
pub fn run(
    items: &[SessionItem],
    include_rule_2: bool,
    min_bytes: usize,
    window: usize,
) -> Vec<PruneCandidate> {
    let mut candidates = dedupe_latest_read(items);
    if include_rule_2 {
        candidates.extend(supersede_write_by_read(items));
    }
    candidates.extend(drop_redundant_errors(items));
    candidates.extend(prune_old_large_outputs(items, min_bytes, window));
    candidates.extend(collapse_task_launches(items));
    candidates
}

/// Produce a pruned *copy* of `jsonl` — never mutates, never touches the
/// input. Only lines containing a pruned tool_result are rewritten (the
/// flagged block's `content` replaced with a short marker); every other
/// line is passed through byte-for-byte, so a diff against the original
/// only ever shows the intentional changes.
pub fn apply_pruning(jsonl: &str, candidates: &[PruneCandidate]) -> String {
    let prune_ids: HashMap<&str, &PruneCandidate> = candidates
        .iter()
        .map(|c| (c.tool_use_id.as_str(), c))
        .collect();

    let mut out = String::with_capacity(jsonl.len());
    for line in jsonl.lines() {
        if line.trim().is_empty() || !prune_ids.keys().any(|id| line.contains(id)) {
            out.push_str(line);
            out.push('\n');
            continue;
        }

        let Ok(mut record) = serde_json::from_str::<Value>(line) else {
            out.push_str(line);
            out.push('\n');
            continue;
        };

        let mut changed = false;
        if let Some(content) = record
            .get_mut("message")
            .and_then(|m| m.get_mut("content"))
            .and_then(|c| c.as_array_mut())
        {
            for block in content.iter_mut() {
                if block.get("type").and_then(|t| t.as_str()) != Some("tool_result") {
                    continue;
                }
                let Some(id) = block.get("tool_use_id").and_then(|v| v.as_str()) else {
                    continue;
                };
                let Some(candidate) = prune_ids.get(id) else {
                    continue;
                };
                let marker = format!(
                    "[... squishi session_prune: {} removed, {} bytes ...]",
                    candidate.rule, candidate.bytes
                );
                if let Some(obj) = block.as_object_mut() {
                    obj.insert("content".to_string(), Value::String(marker));
                    changed = true;
                }
            }
        }

        out.push_str(&if changed {
            record.to_string()
        } else {
            line.to_string()
        });
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-shape fixture lines — field names, block structure and value
    /// shapes match a live Claude Code transcript, confirmed by direct
    /// inspection (see module doc), not invented.
    fn tool_use_line(id: &str, name: &str, path: Option<&str>) -> String {
        let input = match path {
            Some(p) => format!(r#"{{"file_path":"{p}"}}"#),
            None => "{}".to_string(),
        };
        format!(
            r#"{{"type":"assistant","message":{{"role":"assistant","content":[{{"type":"tool_use","id":"{id}","name":"{name}","input":{input}}}]}}}}"#
        )
    }

    fn tool_result_line(tool_use_id: &str, content: &str, is_error: bool) -> String {
        format!(
            r#"{{"type":"user","message":{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"{tool_use_id}","content":{},"is_error":{is_error}}}]}}}}"#,
            serde_json::to_string(content).unwrap()
        )
    }

    #[test]
    fn parse_extracts_real_tool_use_and_tool_result_shapes() {
        let jsonl = format!(
            "{}\n{}\n",
            tool_use_line("toolu_1", "Read", Some("/a.rs")),
            tool_result_line("toolu_1", "file contents", false)
        );
        let items = parse(&jsonl);
        assert_eq!(items.len(), 2);
        assert!(matches!(
            &items[0],
            SessionItem::ToolUse(t) if t.id == "toolu_1" && t.name == "Read" && t.path.as_deref() == Some("/a.rs")
        ));
        assert!(matches!(
            &items[1],
            SessionItem::ToolResult(r) if r.tool_use_id == "toolu_1" && r.content == "file contents" && !r.is_error
        ));
    }

    #[test]
    fn parse_skips_malformed_lines_without_aborting() {
        let jsonl = format!(
            "not json at all\n{}\n{{\"malformed\n{}\n",
            tool_use_line("toolu_1", "Read", Some("/a.rs")),
            tool_result_line("toolu_1", "ok", false)
        );
        let items = parse(&jsonl);
        assert_eq!(
            items.len(),
            2,
            "real lines around malformed ones must still parse"
        );
    }

    #[test]
    fn dedupe_latest_read_flags_all_but_the_last_read_of_a_path() {
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n",
            tool_use_line("toolu_1", "Read", Some("/a.rs")),
            tool_result_line("toolu_1", "old contents", false),
            tool_use_line("toolu_2", "Read", Some("/a.rs")),
            tool_result_line("toolu_2", "new contents", false),
        );
        let items = parse(&jsonl);
        let candidates = dedupe_latest_read(&items);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].tool_use_id, "toolu_1");
        assert_eq!(candidates[0].rule, "dedupe_latest_read");
    }

    #[test]
    fn dedupe_latest_read_does_not_flag_reads_of_different_paths() {
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n",
            tool_use_line("toolu_1", "Read", Some("/a.rs")),
            tool_result_line("toolu_1", "a contents", false),
            tool_use_line("toolu_2", "Read", Some("/b.rs")),
            tool_result_line("toolu_2", "b contents", false),
        );
        let items = parse(&jsonl);
        assert!(dedupe_latest_read(&items).is_empty());
    }

    #[test]
    fn supersede_write_by_read_flags_a_write_verified_by_a_later_read() {
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n",
            tool_use_line("toolu_1", "Write", Some("/a.rs")),
            tool_result_line("toolu_1", "File written", false),
            tool_use_line("toolu_2", "Read", Some("/a.rs")),
            tool_result_line("toolu_2", "new contents", false),
        );
        let items = parse(&jsonl);
        let candidates = supersede_write_by_read(&items);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].tool_use_id, "toolu_1");
    }

    #[test]
    fn supersede_write_by_read_does_not_flag_a_write_with_no_later_read() {
        let jsonl = format!(
            "{}\n{}\n",
            tool_use_line("toolu_1", "Write", Some("/a.rs")),
            tool_result_line("toolu_1", "File written", false),
        );
        let items = parse(&jsonl);
        assert!(supersede_write_by_read(&items).is_empty());
    }

    #[test]
    fn drop_redundant_errors_flags_a_repeated_identical_error() {
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n",
            tool_use_line("toolu_1", "Bash", None),
            tool_result_line("toolu_1", "command not found: foo", true),
            tool_use_line("toolu_2", "Bash", None),
            tool_result_line("toolu_2", "command not found: foo", true),
        );
        let items = parse(&jsonl);
        let candidates = drop_redundant_errors(&items);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].tool_use_id, "toolu_2");
    }

    #[test]
    fn drop_redundant_errors_does_not_flag_different_errors() {
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n",
            tool_use_line("toolu_1", "Bash", None),
            tool_result_line("toolu_1", "error one", true),
            tool_use_line("toolu_2", "Bash", None),
            tool_result_line("toolu_2", "error two", true),
        );
        let items = parse(&jsonl);
        assert!(drop_redundant_errors(&items).is_empty());
    }

    #[test]
    fn prune_old_large_outputs_flags_a_large_result_outside_the_recency_window() {
        let big = "x".repeat(200);
        let mut lines = vec![
            tool_use_line("toolu_1", "Bash", None),
            tool_result_line("toolu_1", &big, false),
        ];
        // Push enough small, unrelated items to push the big one outside a window of 2.
        for i in 2..6 {
            lines.push(tool_use_line(&format!("toolu_{i}"), "Bash", None));
            lines.push(tool_result_line(&format!("toolu_{i}"), "ok", false));
        }
        let jsonl = format!("{}\n", lines.join("\n"));
        let items = parse(&jsonl);
        let candidates = prune_old_large_outputs(&items, 100, 2);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].tool_use_id, "toolu_1");
    }

    #[test]
    fn prune_old_large_outputs_does_not_flag_content_within_the_window() {
        let big = "x".repeat(200);
        let jsonl = format!(
            "{}\n{}\n",
            tool_use_line("toolu_1", "Bash", None),
            tool_result_line("toolu_1", &big, false),
        );
        let items = parse(&jsonl);
        assert!(prune_old_large_outputs(&items, 100, 10).is_empty());
    }

    #[test]
    fn collapse_task_launches_flags_all_but_the_latest_launch_notice() {
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n",
            tool_use_line("toolu_1", "Bash", None),
            tool_result_line(
                "toolu_1",
                "Command running in background with ID: abc",
                false
            ),
            tool_use_line("toolu_2", "Bash", None),
            tool_result_line(
                "toolu_2",
                "Command running in background with ID: def",
                false
            ),
        );
        let items = parse(&jsonl);
        let candidates = collapse_task_launches(&items);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].tool_use_id, "toolu_1");
    }

    #[test]
    fn collapse_task_launches_ignores_unrelated_content() {
        let jsonl = format!(
            "{}\n{}\n",
            tool_use_line("toolu_1", "Bash", None),
            tool_result_line("toolu_1", "regular output", false),
        );
        let items = parse(&jsonl);
        assert!(collapse_task_launches(&items).is_empty());
    }

    #[test]
    fn run_excludes_rule_2_by_default() {
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n",
            tool_use_line("toolu_1", "Write", Some("/a.rs")),
            tool_result_line("toolu_1", "File written", false),
            tool_use_line("toolu_2", "Read", Some("/a.rs")),
            tool_result_line("toolu_2", "new contents", false),
        );
        let items = parse(&jsonl);
        let without_rule_2 = run(&items, false, 10_000, 1000);
        assert!(
            !without_rule_2
                .iter()
                .any(|c| c.rule == "supersede_write_by_read")
        );

        let with_rule_2 = run(&items, true, 10_000, 1000);
        assert!(
            with_rule_2
                .iter()
                .any(|c| c.rule == "supersede_write_by_read")
        );
    }

    #[test]
    fn apply_pruning_replaces_only_flagged_content_and_preserves_line_count() {
        let jsonl = format!(
            "{}\n{}\n{}\n{}\n",
            tool_use_line("toolu_1", "Read", Some("/a.rs")),
            tool_result_line("toolu_1", "old contents", false),
            tool_use_line("toolu_2", "Read", Some("/a.rs")),
            tool_result_line("toolu_2", "new contents", false),
        );
        let items = parse(&jsonl);
        let candidates = dedupe_latest_read(&items);
        let pruned = apply_pruning(&jsonl, &candidates);

        assert_eq!(pruned.lines().count(), jsonl.lines().count());
        assert!(pruned.contains("squishi session_prune: dedupe_latest_read removed"));
        assert!(
            pruned.contains("new contents"),
            "the kept read must be untouched"
        );
        assert!(
            !pruned.contains("old contents"),
            "the pruned read's content must be gone"
        );
    }

    #[test]
    fn apply_pruning_never_mutates_the_original_string() {
        let jsonl = format!(
            "{}\n{}\n",
            tool_use_line("toolu_1", "Bash", None),
            tool_result_line("toolu_1", "some output", false),
        );
        let original = jsonl.clone();
        let candidates = vec![PruneCandidate {
            line_index: 1,
            tool_use_id: "toolu_1".to_string(),
            rule: "dedupe_latest_read",
            reason: "test".to_string(),
            bytes: 11,
        }];
        let _ = apply_pruning(&jsonl, &candidates);
        assert_eq!(jsonl, original);
    }
}
