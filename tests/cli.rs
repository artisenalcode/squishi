//! Black-box tests against the actual compiled binary — the real
//! deployable artifact governator's `squishi.rs` wrapper shells out to.
//! Unit tests in `src/main.rs::route` cover the routing logic; these
//! cover the thing an actual consumer depends on: argv in, stdout out,
//! and that stdout is valid, parseable JSON with the expected shape.

use std::process::Command;

fn run(text: &str) -> serde_json::Value {
    let output = Command::new(env!("CARGO_BIN_EXE_squishi"))
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
