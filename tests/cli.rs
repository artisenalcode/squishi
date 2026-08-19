//! Black-box tests against the actual compiled binary — the real
//! deployable artifact governator's `squishi.rs` wrapper shells out to,
//! and the same binary a harness skill invokes. Unit tests in
//! `src/main.rs::route` cover the routing logic; these cover the thing
//! an actual consumer depends on: argv/stdin in, stdout out, in both
//! output modes.

use std::io::Write;
use std::process::{Command, Stdio};

/// Runs with `--json`, for the governator-contract tests below.
fn run_json(text: &str) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .arg("--json")
        .arg(text)
        .output()
        .expect("failed to run squishi binary");

    assert!(
        output.status.success(),
        "squishi exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout was not valid JSON ({e}): {stdout:?}"))
}

fn run(text: &str) -> serde_json::Value {
    run_json(text)
}

#[test]
fn short_diff_passes_through_with_expected_fields() {
    let value = run("diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b");
    assert_eq!(value["kind"], "Diff");
    assert_eq!(value["source"], "passthrough");
    assert!(value["compressed"].is_string());
    assert!(value["chars_before"].is_u64());
    assert!(value["chars_after"].is_u64());
}

#[test]
fn json_array_is_detected_and_compressed() {
    let value = run(r#"[{"a":1},{"a":1},{"a":1}]"#);
    assert_eq!(value["kind"], "Json");
    assert_eq!(value["source"], "json");
    assert!(value["elements_before"].is_u64());
    assert!(value["elements_after"].is_u64());
}

#[test]
fn plain_prose_dedups_and_reports_char_counts() {
    let value = run("just a short paragraph of prose with no special structure.");
    assert_eq!(value["kind"], "PlainText");
    assert_eq!(value["source"], "dedup");
    // line_dedup normalizes a trailing newline onto short content, so
    // chars_after can be chars_before + 1 — not a compression regression,
    // just confirming the char counts are real and in the right ballpark.
    let before = value["chars_before"].as_u64().unwrap();
    let after = value["chars_after"].as_u64().unwrap();
    assert!(after <= before + 1);
}

#[test]
fn adversarial_content_round_trips_as_valid_json() {
    // Embedded quotes, backslashes, newlines — the exact class of input
    // hand-rolled string formatting is easy to get subtly wrong on.
    let value = run("a line\n\"quoted\"\tand a \\backslash\\ in it");
    assert!(value["compressed"].as_str().unwrap().contains("quoted"));
}

#[test]
fn rust_source_is_classified_as_other() {
    let value = run("fn main() {\n    let x = 5;\n    println!(\"{}\", x);\n}\n");
    let kind = value["kind"].as_str().unwrap();
    assert!(
        kind.starts_with("Other("),
        "expected Other(_) classification, got {kind}"
    );
}

#[test]
fn default_output_is_bare_compressed_text_not_json() {
    // The harness-facing default: no --json, positional arg. stdout
    // should be exactly the compressed text, nothing else — not a JSON
    // object a caller has to parse just to get the content back out.
    let output = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .arg("just a short paragraph of prose with no special structure.")
        .output()
        .expect("failed to run squishi binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        serde_json::from_str::<serde_json::Value>(stdout.trim()).is_err(),
        "default output should not be JSON, got: {stdout:?}"
    );
    assert!(stdout.contains("just a short paragraph of prose"));
}

#[test]
fn reads_from_stdin_when_no_argument_given() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn squishi binary");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"content piped in over stdin, no positional argument at all")
        .expect("failed to write to stdin");

    let output = child.wait_with_output().expect("failed to wait on child");
    assert!(
        output.status.success(),
        "squishi exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("content piped in over stdin"));
}

#[test]
fn json_flag_still_works_when_content_comes_from_stdin() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .arg("--json")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn squishi binary");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(br#"[{"a":1},{"a":1}]"#)
        .expect("failed to write to stdin");

    let output = child.wait_with_output().expect("failed to wait on child");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("stdout should be valid JSON");
    assert_eq!(value["kind"], "Json");
}

#[test]
fn batch_mode_processes_multiple_items_and_returns_a_json_array_with_ids() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .args(["--batch", "--force-kind", "plain-text"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn squishi binary");

    let batch_input = r#"[
        {"id": "one", "text": "first item, short enough to skip semantic dedup entirely."},
        {"id": "two", "text": "second item, also short, a completely different id."}
    ]"#;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(batch_input.as_bytes())
        .expect("failed to write to stdin");

    let output = child.wait_with_output().expect("failed to wait on child");
    assert!(
        output.status.success(),
        "squishi --batch exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("stdout should be a valid JSON array");
    let array = value.as_array().expect("expected a JSON array");
    assert_eq!(array.len(), 2);
    assert_eq!(array[0]["id"], "one");
    assert_eq!(array[0]["kind"], "PlainText");
    assert_eq!(array[1]["id"], "two");
    assert_eq!(array[1]["kind"], "PlainText");
}

#[test]
#[ignore] // real model load (network/cache) — proves restore_punctuation:false actually blocks it end to end
fn batch_mode_restore_punctuation_false_blocks_restoration_on_real_unpunctuated_text() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .args(["--batch", "--force-kind", "plain-text"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn squishi binary");

    // Real shape: long enough (over SKIP_SEMANTIC_DEDUP_UNDER_CHARS) and
    // genuinely unpunctuated (would trigger restoration if allowed) —
    // the exact shape a YouTube-caption transcript has, but here marked
    // as ineligible, same as total-recall would mark a Wikipedia/git
    // source.
    let sentence = "hello and welcome to this transcript we are going to talk about many things today including neuroscience and psychology and how the brain actually works when it comes to emotion ";
    let long_unpunctuated: String = std::iter::repeat_n(sentence, 20).collect();
    let batch_input = serde_json::json!([
        {"id": "should-be-blocked", "text": long_unpunctuated, "restore_punctuation": false}
    ]);
    child
        .stdin
        .take()
        .unwrap()
        .write_all(batch_input.to_string().as_bytes())
        .expect("failed to write to stdin");

    let output = child.wait_with_output().expect("failed to wait on child");
    assert!(
        output.status.success(),
        "squishi --batch exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("stdout should be a valid JSON array");
    assert_eq!(value[0]["punctuation_restored"], false);
}

#[test]
#[ignore] // real model load (network/cache) — proves the `kept` array exposes every survivor, not just narrative-shaped ones
fn json_semantic_dedup_exposes_full_kept_array_with_index_and_shape() {
    let sentence = "hello and welcome to this transcript we are going to talk about many things today including neuroscience and psychology and how the brain actually works when it comes to emotion. ";
    let long_text: String = std::iter::repeat_n(sentence, 20).collect();
    let value = run(&long_text);

    let kept = value["kept"]
        .as_array()
        .expect("kept should be a JSON array");
    let after = value["sentences_after"]
        .as_u64()
        .expect("sentences_after should be present") as usize;
    assert_eq!(
        kept.len(),
        after,
        "kept array length should match sentences_after — it must include every survivor, not just narrative-shaped ones"
    );
    for entry in kept {
        assert!(entry["index"].is_u64());
        assert!(entry["text"].is_string());
        let shape = entry["shape"].as_str().expect("shape should be a string");
        assert!(shape == "narrative" || shape == "concept");
    }
}

#[test]
#[ignore] // real model load (network/cache) — proves include_embedding is opt-in per item, both directions
fn batch_mode_include_embedding_is_opt_in_per_item() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .args(["--batch", "--force-kind", "plain-text"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to spawn squishi binary");

    let sentence = "The database connection failed after three consecutive retry attempts today. ";
    let long_text: String = std::iter::repeat_n(sentence, 30).collect();
    let batch_input = serde_json::json!([
        {"id": "with-embedding", "text": long_text, "include_embedding": true},
        {"id": "without-embedding", "text": long_text}
    ]);
    child
        .stdin
        .take()
        .unwrap()
        .write_all(batch_input.to_string().as_bytes())
        .expect("failed to write to stdin");

    let output = child.wait_with_output().expect("failed to wait on child");
    assert!(
        output.status.success(),
        "squishi --batch exited non-zero: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&output.stdout).trim())
            .expect("stdout should be a valid JSON array");

    let with_embedding = value[0]["kept"]
        .as_array()
        .expect("kept should be an array");
    assert!(
        with_embedding.iter().any(|k| k.get("embedding").is_some()),
        "at least one kept sentence should carry an embedding when include_embedding is true: {with_embedding:?}"
    );
    let embedding_len = with_embedding
        .iter()
        .find_map(|k| k.get("embedding").and_then(|e| e.as_array()))
        .expect("at least one kept sentence should have an embedding array")
        .len();
    assert_eq!(embedding_len, 384, "all-MiniLM-L6-v2's real dimension");

    let without_embedding = value[1]["kept"]
        .as_array()
        .expect("kept should be an array");
    assert!(
        without_embedding
            .iter()
            .all(|k| k.get("embedding").is_none()),
        "no kept sentence should carry an embedding when include_embedding is omitted: {without_embedding:?}"
    );
}

#[test]
fn batch_mode_rejects_malformed_stdin_with_a_clear_error() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .arg("--batch")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn squishi binary");

    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"not valid json at all")
        .expect("failed to write to stdin");

    let output = child.wait_with_output().expect("failed to wait on child");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--batch"));
}

// --- --session-digest --start-line (ADR-0006 Phase 2) ---

fn run_session_digest(transcript_path: &std::path::Path, start_line: usize) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .arg("--session-digest")
        .arg(transcript_path)
        .arg("--start-line")
        .arg(start_line.to_string())
        .arg("--json")
        .output()
        .expect("failed to run squishi binary");

    assert!(
        output.status.success(),
        "squishi --session-digest exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout was not valid JSON ({e}): {stdout:?}"))
}

/// The real incremental contract: a second call using the first call's
/// own `total_lines` as `--start-line` returns only the delta, and that
/// delta is a strict suffix of what a whole-file (`--start-line 0`) call
/// against the same, now-longer transcript would produce.
#[test]
fn start_line_from_a_prior_calls_total_lines_yields_a_strict_suffix() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sess.jsonl");
    let first_turn = r#"{"type":"user","sessionId":"s","cwd":"/repo","timestamp":"t1","message":{"role":"user","content":[{"type":"text","text":"first real question"}]}}"#;
    std::fs::write(&path, format!("{first_turn}\n")).unwrap();

    let first = run_session_digest(&path, 0);
    let total_lines_after_first = first["total_lines"].as_u64().unwrap();
    assert!(total_lines_after_first >= 1);

    let second_turn = r#"{"type":"assistant","sessionId":"s","cwd":"/repo","timestamp":"t2","message":{"role":"assistant","content":[{"type":"text","text":"second real answer"}]}}"#;
    std::fs::write(&path, format!("{first_turn}\n{second_turn}\n")).unwrap();

    let incremental = run_session_digest(&path, total_lines_after_first as usize);
    let whole_file = run_session_digest(&path, 0);

    let incremental_content = incremental["content"].as_str().unwrap();
    let whole_file_content = whole_file["content"].as_str().unwrap();
    assert!(
        whole_file_content.ends_with(incremental_content.rsplit("\n\n").next().unwrap()),
        "incremental digest's real content should appear as a suffix of the whole-file digest"
    );
    assert!(incremental_content.contains("second real answer"));
    assert!(!incremental_content.contains("first real question"));
}

/// An incremental call that lands on a transcript with nothing new since
/// the checkpoint must succeed (exit 0, valid JSON with total_lines) --
/// NOT the "empty session" hard failure a whole-file call still gets.
#[test]
fn start_line_with_nothing_new_succeeds_instead_of_failing() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("sess.jsonl");
    let only_turn = r#"{"type":"user","sessionId":"s","cwd":"/repo","timestamp":"t1","message":{"role":"user","content":[{"type":"text","text":"only turn"}]}}"#;
    std::fs::write(&path, format!("{only_turn}\n")).unwrap();

    let first = run_session_digest(&path, 0);
    let total_lines = first["total_lines"].as_u64().unwrap();

    // Same file, no new lines appended -- start_line == total_lines.
    let second = run_session_digest(&path, total_lines as usize);
    assert_eq!(second["turn_count"], 0);
    assert_eq!(second["total_lines"], total_lines);
}

/// Regression: a truly empty whole-file digest (start_line 0, no real
/// content anywhere) must still fail loudly -- Phase 2 only relaxed the
/// empty-result check for incremental (start_line > 0) calls.
#[test]
fn start_line_zero_on_a_genuinely_empty_session_still_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("empty.jsonl");
    std::fs::write(&path, "").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .arg("--session-digest")
        .arg(&path)
        .output()
        .expect("failed to run squishi binary");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("nothing to digest"));
}

fn run_toon(text: &str, json: bool) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_squishi"));
    cmd.arg("--toon");
    if json {
        cmd.arg("--json");
    }
    cmd.arg(text)
        .output()
        .expect("failed to run squishi binary")
}

#[test]
fn toon_encodes_a_uniform_array_through_the_real_binary() {
    let output = run_toon(r#"[{"id":1,"name":"a"},{"id":2,"name":"b"}]"#, false);
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "[2]{id,name}:\n  1,a\n  2,b");
}

#[test]
fn toon_json_mode_reports_the_governator_contract() {
    let output = run_toon(r#"[{"id":1,"name":"a"},{"id":2,"name":"b"}]"#, true);
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(value["source"], "toon");
    assert!(value["chars_after"].as_u64().unwrap() < value["chars_before"].as_u64().unwrap());
}

#[test]
fn toon_falls_back_to_original_json_when_not_smaller() {
    let output = run_toon("5", true);
    assert!(output.status.success(), "{:?}", output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    assert_eq!(value["source"], "toon-not-smaller");
    assert_eq!(value["compressed"], "5");
}

#[test]
fn toon_rejects_non_json_input_with_a_clear_error() {
    let output = run_toon("not json at all", false);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("requires valid JSON"));
}

/// The real end-to-end proof: whatever `--toon` actually decided for
/// genuinely messy real production data (graphify's own graph.json,
/// confirmed non-uniform -- 3 distinct field shapes across 164 nodes,
/// one node is even missing `_origin`), the result recovers the exact
/// original value. Doesn't assume TOON wins here -- it may honestly
/// decline (`toon-not-smaller`) for data this irregular, and the test
/// checks whichever path was actually taken, not a guessed one.
#[test]
fn toon_round_trips_real_graph_json_through_the_binary() {
    let nodes_json = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/graph_nodes.json"
    ))
    .unwrap();

    let output = run_toon(&nodes_json, true);
    assert!(output.status.success(), "{:?}", output);
    let value: serde_json::Value =
        serde_json::from_str(String::from_utf8_lossy(&output.stdout).trim()).unwrap();
    let compressed = value["compressed"].as_str().unwrap();

    let recovered = if value["source"] == "toon" {
        squishi::toon::decode(compressed).unwrap()
    } else {
        // toon-not-smaller: `compressed` is the original JSON verbatim.
        serde_json::from_str(compressed).unwrap()
    };
    let nodes: serde_json::Value = serde_json::from_str(&nodes_json).unwrap();
    assert_eq!(recovered, nodes);
}

#[test]
fn generate_man_writes_a_real_man_page_and_leaves_no_other_output() {
    let out_dir = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_squishi"))
        .arg("--generate-man")
        .arg(out_dir.path())
        .output()
        .unwrap();

    assert!(output.status.success(), "{:?}", output);

    let man_path = out_dir.path().join("squishi.1");
    let contents = std::fs::read_to_string(&man_path).unwrap();
    // Real roff(7) man page markers, not just "some file got written" --
    // roff escapes hyphens as `\-`, so a documented flag reads `\-\-toon`.
    assert!(
        contents.contains(".TH squishi"),
        "missing .TH header: {contents}"
    );
    assert!(
        contents.contains("\\-\\-toon"),
        "missing a real documented flag: {contents}"
    );
    // --generate-man itself is a hidden dev-only flag -- must not appear
    // in the user-facing man page it produces.
    assert!(
        !contents.contains("generate\\-man"),
        "hidden flag leaked into its own output"
    );
}
