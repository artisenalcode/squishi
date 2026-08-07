//! Extracts human/assistant prose from a Claude Code session transcript
//! and builds a ready-to-stage digest — the Rust port of
//! `mindforge/tools/session_to_trm.py` + the extraction half of
//! `extract_claude_sessions.py` (both Python, read in full before
//! porting; see docs/plan-2026-08-07-session-digest.md).
//!
//! Deliberately **not** layered on `session_prune`: that module prunes
//! `tool_result` content; this one discards all `tool_use`/`tool_result`
//! blocks outright, unconditionally, keeping only `type: "text"` blocks
//! from `user`/`assistant` messages. The two operate on disjoint fields
//! of the same transcript — running one before the other has no effect.
//! Real finding from reading both real implementations side by side
//! before assuming they'd compose, not assumed.
//!
//! Boundary held: this module only extracts and formats text. It never
//! calls `trm` or anything storage-shaped — that stays total-recall's
//! job (`trm ingest-session`), consistent with squishi's boundary rule
//! stated everywhere else in this crate.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SessionMeta {
    pub session_id: Option<String>,
    pub cwd: Option<String>,
    pub first_ts: Option<String>,
    pub last_ts: Option<String>,
    pub turn_count: usize,
    pub truncated: bool,
    pub raw_bytes: usize,
}

/// Strip a trailing `<system-reminder>...</...>` block — everything
/// from the first occurrence of the tag to the end of the string, same
/// as the Python version's `re.DOTALL` substitution (not just the same
/// line: a system-reminder block is typically multi-line).
fn strip_system_reminder(text: &str) -> String {
    match text.find("<system-reminder>") {
        Some(idx) => text[..idx].trim().to_string(),
        None => text.trim().to_string(),
    }
}

/// Truncate `text` to `max_chars` *characters* (not bytes — Rust byte
/// slicing panics mid-UTF-8-char; Python's `len()`/slicing is
/// codepoint-based, so this must be too), keeping the head and tail and
/// dropping the middle, matching the Python version's shape exactly.
fn truncate_middle(text: &str, max_chars: usize) -> (String, bool) {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return (text.to_string(), false);
    }
    let half = max_chars / 2;
    let head: String = chars[..half].iter().collect();
    let tail: String = chars[chars.len() - half..].iter().collect();
    (
        format!("{head}\n\n...[truncated — session exceeded per-item cap]...\n\n{tail}"),
        true,
    )
}

/// Pull human + assistant text turns only from a transcript. Returns
/// `(text, meta)`. Defensive throughout: an unparseable or unrecognized
/// line is skipped, never a hard error — same discipline as
/// `session_prune::parse`, same reason (transcript JSONL isn't a
/// versioned contract).
pub fn extract_session_text(jsonl: &str, max_chars: usize) -> (String, SessionMeta) {
    let mut lines_out: Vec<String> = Vec::new();
    let mut turn_count = 0usize;
    let mut session_id = None;
    let mut cwd = None;
    let mut first_ts = None;
    let mut last_ts = None;

    for line in jsonl.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(record) = serde_json::from_str::<Value>(line) else {
            continue;
        };

        if let Some(id) = record.get("sessionId").and_then(|v| v.as_str()) {
            session_id = Some(id.to_string());
        }
        if let Some(c) = record.get("cwd").and_then(|v| v.as_str()) {
            cwd = Some(c.to_string());
        }
        if let Some(ts) = record.get("timestamp").and_then(|v| v.as_str()) {
            if first_ts.is_none() {
                first_ts = Some(ts.to_string());
            }
            last_ts = Some(ts.to_string());
        }

        let role = record.get("type").and_then(|v| v.as_str());
        if !matches!(role, Some("user") | Some("assistant")) {
            continue;
        }

        let content = record.get("message").and_then(|m| m.get("content"));
        let mut texts: Vec<String> = Vec::new();
        match content {
            Some(Value::String(s)) => texts.push(s.clone()),
            Some(Value::Array(blocks)) => {
                for block in blocks {
                    if block.get("type").and_then(|t| t.as_str()) == Some("text")
                        && let Some(t) = block.get("text").and_then(|v| v.as_str())
                        && !t.is_empty()
                    {
                        texts.push(t.to_string());
                    }
                }
            }
            _ => {}
        }
        if texts.is_empty() {
            continue;
        }

        let label = if role == Some("user") {
            "USER"
        } else {
            "ASSISTANT"
        };
        for t in texts {
            let t = strip_system_reminder(&t);
            if t.is_empty() {
                continue;
            }
            turn_count += 1;
            lines_out.push(format!("{label}: {t}"));
        }
    }

    let text = lines_out.join("\n\n");
    let (text, truncated) = truncate_middle(&text, max_chars);

    let meta = SessionMeta {
        session_id,
        cwd,
        first_ts,
        last_ts,
        turn_count,
        truncated,
        raw_bytes: jsonl.len(),
    };
    (text, meta)
}

/// Same fixed header format as the Python version's
/// `build_digest_content`.
pub fn build_digest_content(compressed: &str, meta: &SessionMeta) -> String {
    format!(
        "SESSION DIGEST {}\n\n---\ntype: session-digest\nsession_id: {}\ncwd: {}\nfirst_ts: {}\nlast_ts: {}\nturn_count: {}\n---\n\n{compressed}",
        meta.session_id.as_deref().unwrap_or(""),
        meta.session_id.as_deref().unwrap_or(""),
        meta.cwd.as_deref().unwrap_or(""),
        meta.first_ts.as_deref().unwrap_or(""),
        meta.last_ts.as_deref().unwrap_or(""),
        meta.turn_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real-shape fixture lines — same field names/structure as a live
    /// Claude Code transcript (confirmed by direct inspection, see
    /// session_prune's own module doc for the same discipline).
    fn text_line(role: &str, text: &str, session_id: &str, cwd: &str, ts: &str) -> String {
        format!(
            r#"{{"type":"{role}","sessionId":"{session_id}","cwd":"{cwd}","timestamp":"{ts}","message":{{"role":"{role}","content":[{{"type":"text","text":{}}}]}}}}"#,
            serde_json::to_string(text).unwrap()
        )
    }

    #[test]
    fn extracts_user_and_assistant_text_turns_in_order() {
        let jsonl = format!(
            "{}\n{}\n",
            text_line(
                "user",
                "hello there",
                "sess-1",
                "/repo",
                "2026-08-07T00:00:00Z"
            ),
            text_line(
                "assistant",
                "hi back",
                "sess-1",
                "/repo",
                "2026-08-07T00:00:01Z"
            ),
        );
        let (text, meta) = extract_session_text(&jsonl, 100_000);
        assert_eq!(text, "USER: hello there\n\nASSISTANT: hi back");
        assert_eq!(meta.turn_count, 2);
        assert_eq!(meta.session_id.as_deref(), Some("sess-1"));
        assert_eq!(meta.cwd.as_deref(), Some("/repo"));
        assert_eq!(meta.first_ts.as_deref(), Some("2026-08-07T00:00:00Z"));
        assert_eq!(meta.last_ts.as_deref(), Some("2026-08-07T00:00:01Z"));
        assert!(!meta.truncated);
    }

    #[test]
    fn strips_a_trailing_system_reminder_block() {
        let jsonl = text_line(
            "user",
            "real question<system-reminder>\nmultiline\ninjected context\n</system-reminder>",
            "sess-1",
            "/repo",
            "2026-08-07T00:00:00Z",
        ) + "\n";
        let (text, _) = extract_session_text(&jsonl, 100_000);
        assert_eq!(text, "USER: real question");
    }

    #[test]
    fn skips_tool_use_and_tool_result_blocks_entirely() {
        let jsonl = r#"{"type":"assistant","sessionId":"s","cwd":"/r","timestamp":"t","message":{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{}}]}}"#
            .to_string()
            + "\n"
            + r#"{"type":"user","sessionId":"s","cwd":"/r","timestamp":"t","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":"output"}]}}"#
            + "\n";
        let (text, meta) = extract_session_text(&jsonl, 100_000);
        assert_eq!(text, "");
        assert_eq!(meta.turn_count, 0);
    }

    #[test]
    fn skips_malformed_lines_without_aborting() {
        let jsonl = format!(
            "not json\n{}\n{{\"malformed\n{}\n",
            text_line("user", "first", "s", "/r", "t1"),
            text_line("assistant", "second", "s", "/r", "t2"),
        );
        let (text, meta) = extract_session_text(&jsonl, 100_000);
        assert_eq!(meta.turn_count, 2);
        assert!(text.contains("first") && text.contains("second"));
    }

    #[test]
    fn truncates_long_text_in_the_middle_and_flags_it() {
        let jsonl = text_line("user", &"word ".repeat(50), "s", "/r", "t") + "\n";
        let (text, meta) = extract_session_text(&jsonl, 40);
        assert!(meta.truncated);
        assert!(text.contains("...[truncated — session exceeded per-item cap]..."));
    }

    #[test]
    fn truncation_is_char_safe_with_multibyte_content() {
        // A string full of multi-byte UTF-8 characters — byte-index
        // slicing at an arbitrary offset would panic here if this
        // weren't char-safe.
        let content = "café ".repeat(30);
        let jsonl = text_line("user", &content, "s", "/r", "t") + "\n";
        let (text, meta) = extract_session_text(&jsonl, 20);
        assert!(meta.truncated);
        // Must not panic, and must still be valid UTF-8 (guaranteed by
        // type, but assert content is non-empty and sane).
        assert!(text.contains("café"));
    }

    #[test]
    fn build_digest_content_matches_the_expected_header_shape() {
        let meta = SessionMeta {
            session_id: Some("sess-1".to_string()),
            cwd: Some("/repo".to_string()),
            first_ts: Some("t1".to_string()),
            last_ts: Some("t2".to_string()),
            turn_count: 3,
            truncated: false,
            raw_bytes: 100,
        };
        let digest = build_digest_content("compressed body here", &meta);
        assert!(digest.starts_with("SESSION DIGEST sess-1\n\n---\n"));
        assert!(digest.contains("session_id: sess-1"));
        assert!(digest.contains("cwd: /repo"));
        assert!(digest.contains("turn_count: 3"));
        assert!(digest.ends_with("compressed body here"));
    }

    #[test]
    fn empty_transcript_produces_empty_text_and_zero_turns() {
        let (text, meta) = extract_session_text("", 100_000);
        assert_eq!(text, "");
        assert_eq!(meta.turn_count, 0);
        assert!(!meta.truncated);
    }
}
