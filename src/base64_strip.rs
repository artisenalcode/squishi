//! Deterministic base64-blob stripping — a zero-model pre-pass, not a new
//! `ContentKind`. A base64 blob (an embedded screenshot, a data-URI in
//! HTML/JSON) can appear inside any shape squishi detects — JSON, logs,
//! diffs, plain text — so it's stripped unconditionally before `detect()`
//! runs, the same way MCE's `Layer1Pruner` runs before its shape-aware
//! Layer 2 (found auditing `~/Code/_labs/audit-repos/MCE`, 2026-08-07).
//!
//! No dependency on lookaround (Rust's `regex` crate doesn't support it,
//! unlike the Python `re` MCE's own pattern uses) — not needed here:
//! requiring a minimum length via `{100,}` already makes greedy matching
//! consume the maximal contiguous base64-alphabet run on its own.

use regex::Regex;
use std::sync::LazyLock;

/// Minimum run length inside an already-identified `data:...;base64,`
/// blob — the `data:` prefix itself is the high-confidence signal here,
/// so this only needs to rule out a degenerate near-empty payload, not
/// guard against false positives the way the standalone threshold does.
const MIN_DATA_URI_INNER_CHARS: usize = 20;

/// Minimum run length for a *standalone* (unprefixed) base64-alphabet
/// run — much higher than the data-URI inner minimum, and higher than
/// the original 100 this started at. Real measurement (calibration
/// probe, 2026-08-07): a real JWT's base64url payload segment (203
/// chars, no `data:` prefix to disambiguate it) matched and got wrongly
/// stripped at 100. Real image blobs run into the thousands of chars;
/// real JWTs are typically well under 500. 500 separates the two in
/// practice without being able to guarantee it in every case — a real,
/// reasoned tradeoff, not a proof.
const MIN_STANDALONE_BLOB_CHARS: usize = 500;

static DATA_URI_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"data:[a-zA-Z0-9/+.-]+;base64,[A-Za-z0-9+/\n]{{{MIN_DATA_URI_INNER_CHARS},}}={{0,2}}"
    ))
    .unwrap()
});
static STANDALONE_B64_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"[A-Za-z0-9+/]{{{MIN_STANDALONE_BLOB_CHARS},}}={{0,2}}"
    ))
    .unwrap()
});

fn marker(matched_len: usize) -> String {
    format!("[... squishi pruned: base64 blob removed, {matched_len} chars ...]")
}

/// Strip base64 blobs from `text`, returning `(stripped_text,
/// blobs_removed)`. Data-URI blobs are matched first (more specific,
/// higher-confidence), then any remaining standalone base64 runs.
pub fn strip_base64_blobs(text: &str) -> (String, usize) {
    let mut removed = 0;

    let after_data_uri = DATA_URI_RE.replace_all(text, |caps: &regex::Captures| {
        removed += 1;
        marker(caps[0].len())
    });

    let after_standalone =
        STANDALONE_B64_RE.replace_all(&after_data_uri, |caps: &regex::Captures| {
            removed += 1;
            marker(caps[0].len())
        });

    (after_standalone.into_owned(), removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_with_no_base64_is_returned_unchanged() {
        let content = "just a normal paragraph of prose with no special structure.";
        let (result, removed) = strip_base64_blobs(content);
        assert_eq!(result, content);
        assert_eq!(removed, 0);
    }

    #[test]
    fn a_real_data_uri_gets_replaced_with_one_marker() {
        // A real (small, valid) base64-encoded 1x1 PNG data URI.
        let data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
        let content = format!("here is an image: {data_uri} end of message");
        let (result, removed) = strip_base64_blobs(&content);
        assert_eq!(removed, 1);
        assert!(!result.contains("iVBORw0KGgo"));
        assert!(result.contains("squishi pruned: base64 blob removed"));
        assert!(result.starts_with("here is an image:"));
        assert!(result.ends_with("end of message"));
    }

    #[test]
    fn a_standalone_long_base64_run_gets_replaced() {
        let blob = "A".repeat(500); // at the real MIN_STANDALONE_BLOB_CHARS threshold
        let content = format!("payload: {blob} done");
        let (result, removed) = strip_base64_blobs(&content);
        assert_eq!(removed, 1);
        assert!(!result.contains(&blob));
        assert!(result.contains("squishi pruned: base64 blob removed"));
    }

    #[test]
    fn short_base64_looking_strings_are_left_alone() {
        let content = "token: dGVzdA== is short and should stay";
        let (result, removed) = strip_base64_blobs(content);
        assert_eq!(removed, 0);
        assert_eq!(result, content);
    }

    /// Real regression (calibration probe, 2026-08-07): a real JWT's
    /// base64url payload segment (203 chars, no `data:` prefix) matched
    /// and got wrongly stripped at the original 100-char threshold. This
    /// pins that specific real fixture as a permanent regression test,
    /// not just a one-off probe run.
    #[test]
    fn a_real_jwt_payload_segment_is_not_mistaken_for_a_blob() {
        let jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.\
                    eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyLCJyb2xlIjoiYWRtaW4iLCJhdWQiOiJodHRwczovL2FwaS5leGFtcGxlLmNvbSIsImV4cCI6MTcwMDAwMDAwMCwiaXNzIjoiaHR0cHM6Ly9hdXRoLmV4YW1wbGUuY29tIn0.\
                    dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let (result, removed) = strip_base64_blobs(jwt);
        assert_eq!(
            removed, 0,
            "a real JWT should not be treated as a base64 blob"
        );
        assert_eq!(result, jwt);
    }

    #[test]
    fn a_base64_blob_inside_a_json_string_value_stays_valid_json() {
        let blob = "A".repeat(500);
        let content = format!(r#"{{"image": "{blob}", "name": "test"}}"#);
        let (result, removed) = strip_base64_blobs(&content);
        assert_eq!(removed, 1);
        let parsed: serde_json::Value =
            serde_json::from_str(&result).expect("result should still be valid JSON");
        assert_eq!(parsed["name"], "test");
        assert!(parsed["image"].as_str().unwrap().contains("squishi pruned"));
    }

    #[test]
    fn multiple_blobs_in_one_document_are_all_counted() {
        let blob = "A".repeat(500);
        let content = format!("{blob}\nmiddle text\n{blob}");
        let (_result, removed) = strip_base64_blobs(&content);
        assert_eq!(removed, 2);
    }
}
