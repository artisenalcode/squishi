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
