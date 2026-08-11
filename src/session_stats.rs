//! Session-level stats aggregator: scans a Claude Code transcript for
//! squishi's own `--json` invocations and reports cumulative real savings
//! (chars_before/chars_after), broken down per content `kind`.
//!
//! `session_prune::parse`'s `ToolUseItem.path` only captures Read/Write/
//! Edit's `input.file_path` — a squishi call made via Bash carries no
//! structured path/command field there, so identifying "which tool calls
//! were squishi" by tool name or command text isn't reliable. Instead
//! this matches `ToolResultItem.content` against squishi's own known
//! `--json` output contract (`compressed`/`kind`/`source`/`chars_before`/
//! `chars_after` — see `main.rs`'s `build_output`): invocation-method-
//! agnostic, and a false-positive match would need another tool to
//! independently produce all five keys with those exact names, which
//! nothing else in a real transcript does.

use crate::session_prune::{self, SessionItem};
use serde_json::Value;
use std::collections::BTreeMap;

const REQUIRED_KEYS: [&str; 5] = [
    "compressed",
    "kind",
    "source",
    "chars_before",
    "chars_after",
];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KindStats {
    pub calls: usize,
    pub chars_before: u64,
    pub chars_after: u64,
}

impl KindStats {
    pub fn chars_saved(&self) -> u64 {
        self.chars_before.saturating_sub(self.chars_after)
    }

    /// `0.0` when `chars_before` is `0` (nothing to save a percentage of),
    /// not a division-by-zero panic or `NaN`.
    pub fn pct_saved(&self) -> f64 {
        if self.chars_before == 0 {
            0.0
        } else {
            self.chars_saved() as f64 / self.chars_before as f64 * 100.0
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionStats {
    pub by_kind: BTreeMap<String, KindStats>,
    pub total: KindStats,
}

/// Scans `jsonl` (a full or partial Claude Code transcript) and
/// aggregates every squishi `--json` call found in it. Anything that
/// doesn't parse as JSON, or parses but is missing one of the five
/// contract keys, is silently skipped — same "transcript shape isn't a
/// versioned contract, be defensive" posture `session_prune::parse` uses.
pub fn scan(jsonl: &str) -> SessionStats {
    let items = session_prune::parse(jsonl);
    let mut stats = SessionStats::default();

    for item in &items {
        let SessionItem::ToolResult(result) = item else {
            continue;
        };
        let Some(call) = parse_squishi_output(&result.content) else {
            continue;
        };

        stats.total.calls += 1;
        stats.total.chars_before += call.chars_before;
        stats.total.chars_after += call.chars_after;

        let kind_stats = stats.by_kind.entry(call.kind).or_default();
        kind_stats.calls += 1;
        kind_stats.chars_before += call.chars_before;
        kind_stats.chars_after += call.chars_after;
    }

    stats
}

struct SquishiCall {
    kind: String,
    chars_before: u64,
    chars_after: u64,
}

fn parse_squishi_output(content: &str) -> Option<SquishiCall> {
    let value: Value = serde_json::from_str(content.trim()).ok()?;
    let obj = value.as_object()?;
    if !REQUIRED_KEYS.iter().all(|k| obj.contains_key(*k)) {
        return None;
    }
    Some(SquishiCall {
        kind: obj.get("kind")?.as_str()?.to_string(),
        chars_before: obj.get("chars_before")?.as_u64()?,
        chars_after: obj.get("chars_after")?.as_u64()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_result_line(tool_use_id: &str, content: &str) -> String {
        let record = serde_json::json!({
            "message": {
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                    "is_error": false,
                }]
            }
        });
        record.to_string()
    }

    fn squishi_json(kind: &str, chars_before: u64, chars_after: u64) -> String {
        serde_json::json!({
            "compressed": "...",
            "kind": kind,
            "source": "dedup",
            "chars_before": chars_before,
            "chars_after": chars_after,
        })
        .to_string()
    }

    #[test]
    fn aggregates_real_squishi_calls_by_kind() {
        let jsonl = [
            tool_result_line("t1", &squishi_json("Json", 1000, 100)),
            tool_result_line("t2", &squishi_json("Json", 500, 50)),
            tool_result_line("t3", &squishi_json("Log", 2000, 200)),
        ]
        .join("\n");

        let stats = scan(&jsonl);

        assert_eq!(stats.total.calls, 3);
        assert_eq!(stats.total.chars_before, 3500);
        assert_eq!(stats.total.chars_after, 350);

        let json_stats = &stats.by_kind["Json"];
        assert_eq!(json_stats.calls, 2);
        assert_eq!(json_stats.chars_before, 1500);
        assert_eq!(json_stats.chars_after, 150);

        let log_stats = &stats.by_kind["Log"];
        assert_eq!(log_stats.calls, 1);
        assert_eq!(log_stats.chars_saved(), 1800);
    }

    #[test]
    fn skips_a_non_squishi_tool_result() {
        // A real Read tool result: plain file content, not squishi's
        // 5-key JSON contract — must not be miscounted.
        let jsonl = [
            tool_result_line("t1", "fn main() {\n    println!(\"hi\");\n}\n"),
            tool_result_line("t2", &squishi_json("Diff", 800, 400)),
        ]
        .join("\n");

        let stats = scan(&jsonl);

        assert_eq!(stats.total.calls, 1);
        assert_eq!(stats.by_kind.len(), 1);
        assert!(stats.by_kind.contains_key("Diff"));
    }

    #[test]
    fn skips_json_missing_a_required_key() {
        // Real squishi output missing chars_after would mean a
        // contract regression elsewhere — never silently half-count it.
        let partial = serde_json::json!({
            "compressed": "...",
            "kind": "Json",
            "source": "dedup",
            "chars_before": 100,
        })
        .to_string();
        let jsonl = tool_result_line("t1", &partial);

        let stats = scan(&jsonl);

        assert_eq!(stats.total.calls, 0);
    }

    #[test]
    fn empty_transcript_yields_zeroed_stats() {
        let stats = scan("");
        assert_eq!(stats.total.calls, 0);
        assert!(stats.by_kind.is_empty());
    }

    #[test]
    fn pct_saved_is_zero_not_nan_when_chars_before_is_zero() {
        let stats = KindStats::default();
        assert_eq!(stats.pct_saved(), 0.0);
    }
}
