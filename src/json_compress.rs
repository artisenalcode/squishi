//! JSON array compression: dedupe exact-duplicate elements, then cap to
//! first + last N with a dropped-count marker. Scoped to arrays — a
//! single object has nothing repeatable to compress.
//!
//! The dropped-count marker is enriched with verified facts about the
//! dropped elements when they're objects (see `crate::invariants`) —
//! a constant field, a numeric range, identifier coverage, or a small
//! safe enumeration, instead of just a count. Field values for this pass
//! come from a second, `raw_value`-based re-parse of the original input,
//! isolated from the `Value`-based dedup/selection pipeline below: a
//! normal `serde_json::Value` renormalizes number formatting (`5.00`
//! becomes `5.0`), which would make a range marker lie about the source.
//! `raw_value` was chosen over the alternative, `arbitrary_precision`,
//! because `arbitrary_precision` is crate-wide and was confirmed (in a
//! throwaway build, not assumed) to change `Value` equality everywhere —
//! `5.00 == 5.0` flips from `true` to `false`, which would silently
//! weaken the exact-duplicate dedup below. `raw_value` has zero such
//! effect, confirmed the same way.

use crate::invariants::{self, InvariantConfig, Unit};
use serde_json::Value;
use serde_json::value::RawValue;
use std::collections::BTreeMap;

/// Tunable knobs for `compress_json_array` — `--level` varies `keep_edge`.
pub struct JsonCompressConfig {
    /// First N and last N elements kept when capping.
    pub keep_edge: usize,
    /// Thresholds for the invariant-disclosure marker text.
    pub invariants: InvariantConfig,
}

impl Default for JsonCompressConfig {
    fn default() -> Self {
        Self {
            keep_edge: 5,
            invariants: InvariantConfig::default(),
        }
    }
}

pub struct JsonCompressResult {
    pub original_elements: usize,
    pub compressed_elements: usize,
    pub content: String,
}

/// Returns `None` if `content` isn't a JSON array — caller should fall
/// back to another compressor for objects/non-JSON.
pub fn compress_json_array(
    content: &str,
    config: &JsonCompressConfig,
) -> Option<JsonCompressResult> {
    let value: Value = serde_json::from_str(content.trim()).ok()?;
    let Value::Array(elements) = value else {
        return None;
    };

    let original_elements = elements.len();

    // Dedupe exact-duplicate elements, preserving first-seen order --
    // also track each kept element's ORIGINAL index in `elements`, so
    // the invariant-disclosure pass below can recover its exact
    // original-formatted field text via the isolated raw_value re-parse,
    // without this loop's own dedup semantics changing at all.
    let mut seen: Vec<Value> = Vec::new();
    let mut seen_original_index: Vec<usize> = Vec::new();
    for (i, el) in elements.iter().enumerate() {
        if !seen.contains(el) {
            seen.push(el.clone());
            seen_original_index.push(i);
        }
    }

    let keep_edge = config.keep_edge;
    let final_elements: Vec<Value> = if seen.len() > keep_edge * 2 {
        let dropped_indices = &seen_original_index[keep_edge..seen.len() - keep_edge];
        let marker = build_marker(content, dropped_indices, &config.invariants);

        let mut kept: Vec<Value> = seen[..keep_edge].to_vec();
        kept.push(Value::String(marker));
        kept.extend_from_slice(&seen[seen.len() - keep_edge..]);
        kept
    } else {
        seen
    };

    let compressed_elements = final_elements.len();
    let content = serde_json::to_string(&Value::Array(final_elements)).ok()?;

    Some(JsonCompressResult {
        original_elements,
        compressed_elements,
        content,
    })
}

/// Builds the dropped-elements marker: the bare count-only form, enriched
/// with `crate::invariants::describe`'s verified-fact suffix when the
/// dropped elements are objects and something safe survives disclosure.
/// Falls back to the bare form on any parse failure or when nothing
/// survives -- never a marker that looks enriched but claims nothing.
fn build_marker(content: &str, dropped_indices: &[usize], config: &InvariantConfig) -> String {
    let dropped = dropped_indices.len();
    let bare = format!("...{dropped} more elements omitted...");

    let Ok(raw_elements) = serde_json::from_str::<Vec<Box<RawValue>>>(content.trim()) else {
        return bare;
    };

    let units: Vec<Unit> = dropped_indices
        .iter()
        .filter_map(|&i| raw_elements.get(i))
        .filter_map(|raw| raw_object_fields(raw))
        .map(Unit::new)
        .collect();

    match invariants::describe(&units, config) {
        Some(facts) => format!("...{dropped} more elements omitted ({facts})..."),
        None => bare,
    }
}

/// Field name -> exact original-formatted value text for one raw array
/// element, if it's a JSON object. `None` for scalars/arrays/anything
/// that isn't an object -- v1's documented scope limit, see the spec.
fn raw_object_fields(raw: &RawValue) -> Option<Vec<(String, String)>> {
    let map: BTreeMap<String, Box<RawValue>> = serde_json::from_str(raw.get()).ok()?;
    Some(
        map.into_iter()
            .map(|(k, v)| (k, display_raw(v.get())))
            .collect(),
    )
}

/// A raw JSON value's text as it should appear in a marker: a quoted
/// JSON string is unescaped to its real content (`"widget"` -> `widget`);
/// anything else (number/bool/null/nested array or object) is already in
/// its correct display form as raw JSON text.
fn display_raw(raw: &str) -> String {
    serde_json::from_str::<String>(raw).unwrap_or_else(|_| raw.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_array_json_returns_none() {
        assert!(
            compress_json_array(r#"{"key": "value"}"#, &JsonCompressConfig::default()).is_none()
        );
    }

    #[test]
    fn non_json_returns_none() {
        assert!(compress_json_array("not json at all", &JsonCompressConfig::default()).is_none());
    }

    #[test]
    fn dedupes_exact_duplicate_elements() {
        let content = r#"[{"a":1},{"a":1},{"a":1},{"a":2}]"#;
        let result = compress_json_array(content, &JsonCompressConfig::default()).unwrap();
        assert_eq!(result.original_elements, 4);
        assert_eq!(result.compressed_elements, 2);
    }

    #[test]
    fn small_array_is_unchanged_besides_formatting() {
        let content = r#"[1,2,3]"#;
        let result = compress_json_array(content, &JsonCompressConfig::default()).unwrap();
        assert_eq!(result.original_elements, 3);
        assert_eq!(result.compressed_elements, 3);
    }

    #[test]
    fn large_array_caps_to_first_and_last_edge() {
        let elements: Vec<String> = (0..50).map(|i| format!(r#"{{"id":{i}}}"#)).collect();
        let content = format!("[{}]", elements.join(","));
        let config = JsonCompressConfig::default();
        let result = compress_json_array(&content, &config).unwrap();
        assert_eq!(result.original_elements, 50);
        assert_eq!(result.compressed_elements, config.keep_edge * 2 + 1);
        assert!(result.content.contains("more elements omitted"));
        assert!(result.content.contains(r#""id":0"#));
        assert!(result.content.contains(r#""id":49"#));
    }

    #[test]
    fn keep_edge_is_configurable() {
        let elements: Vec<String> = (0..10).map(|i| format!(r#"{{"id":{i}}}"#)).collect();
        let content = format!("[{}]", elements.join(","));
        let config = JsonCompressConfig {
            keep_edge: 2,
            invariants: InvariantConfig::default(),
        };
        let result = compress_json_array(&content, &config).unwrap();
        assert_eq!(result.compressed_elements, 2 * 2 + 1);
        assert!(result.content.contains(r#""id":0"#));
        assert!(result.content.contains(r#""id":1"#));
        assert!(!result.content.contains(r#""id":2"#));
    }

    /// End-to-end proof the enriched marker actually fires through the
    /// real `compress_json_array` entry point, not just the isolated
    /// `invariants` module: a constant field, a small enumeration, and a
    /// numeric range should all show up in one marker.
    #[test]
    fn dropped_elements_marker_states_constant_enum_and_range_facts() {
        let statuses = ["active", "pending", "closed"];
        let elements: Vec<String> = (0..20)
            .map(|i| {
                let status = statuses[i % statuses.len()];
                let amount = 5.00 + i as f64;
                format!(r#"{{"id":{i},"kind":"widget","status":"{status}","amount":{amount:.2}}}"#)
            })
            .collect();
        let content = format!("[{}]", elements.join(","));
        let result = compress_json_array(&content, &JsonCompressConfig::default()).unwrap();

        assert!(
            result.content.contains("kind=widget"),
            "expected constant field, got: {}",
            result.content
        );
        assert!(
            result.content.contains("status:"),
            "expected status enumeration, got: {}",
            result.content
        );
        assert!(
            result.content.contains("range amount="),
            "expected numeric range, got: {}",
            result.content
        );
        // The bare count must still be present -- facts are additive to
        // it, never a replacement for it.
        assert!(result.content.contains("more elements omitted"));
    }

    /// An array of objects with a unique identifier per element gets a
    /// coverage statement, not a huge enumeration -- the actual case
    /// caveman's own `order_id` example targets.
    #[test]
    fn dropped_elements_with_a_unique_id_each_render_coverage_not_enumeration() {
        let elements: Vec<String> = (0..20)
            .map(|i| format!(r#"{{"order_id":"ord-{i:04}"}}"#))
            .collect();
        let content = format!("[{}]", elements.join(","));
        let result = compress_json_array(&content, &JsonCompressConfig::default()).unwrap();

        // Zero-padded, contiguous order_ids -- the fixture happens to be
        // exactly the dense case, so this is the strongest possible
        // coverage form: verified full membership, not just a count.
        assert!(
            result
                .content
                .contains("order_id: ord-0005..ord-0014 all 10 present"),
            "expected dense coverage statement, got: {}",
            result.content
        );
        // Must not enumerate all 10 dropped order_ids individually.
        assert!(!result.content.contains("ord-0005×1"));
    }

    /// A field that fails the safety bar (too many distinct values, with
    /// real repeats so it doesn't qualify as identifier coverage either)
    /// is withheld entirely -- even while a DIFFERENT field on the same
    /// units (here, a unique `id`) does get disclosed. Proves withholding
    /// is per-field, not all-or-nothing for the whole marker.
    #[test]
    fn a_field_failing_the_safety_bar_is_withheld_even_when_another_field_is_disclosed() {
        let notes = [
            "alpha state",
            "beta state",
            "gamma state",
            "delta state",
            "epsilon state",
            "zeta state",
            "eta state",
        ];
        let elements: Vec<String> = (0..20)
            .map(|i| format!(r#"{{"id":{i},"note":"{}"}}"#, notes[i % notes.len()]))
            .collect();
        let content = format!("[{}]", elements.join(","));
        let result = compress_json_array(&content, &JsonCompressConfig::default()).unwrap();

        // Isolate just the marker text -- the surrounding KEPT elements
        // (first/last `keep_edge`) legitimately still contain "note" and
        // its real values verbatim; this test is only about what the
        // marker itself discloses for the DROPPED elements. The marker
        // is a JSON string starting with `"...`; find its own closing
        // quote (safe -- the marker text itself never contains `"`).
        let quote_start = result.content.find("\"...").unwrap() + 1;
        let after_start = &result.content[quote_start..];
        let quote_end = after_start.find('"').unwrap();
        let marker = &after_start[..quote_end];

        assert!(
            !marker.contains("note"),
            "note field must not appear in the marker: {marker}"
        );
        for note in notes {
            assert!(
                !marker.contains(note),
                "note value {note:?} must not leak into the marker: {marker}"
            );
        }
        assert!(
            marker.contains("id="),
            "id's own range should still be disclosed: {marker}"
        );
    }

    /// When NOTHING in the dropped elements clears the safety bar at
    /// all, the marker falls all the way back to the original bare
    /// count-only form -- never a marker that looks enriched (extra
    /// parens, an empty fact list) but says nothing real. A single
    /// credential-shaped field, varying and numeric-looking (would
    /// normally qualify for a range) but withheld regardless of shape --
    /// distinct per element too, so exact-duplicate dedup can't collapse
    /// the fixture out from under the test.
    #[test]
    fn dropped_elements_with_no_safe_facts_at_all_keep_the_exact_bare_marker() {
        let elements: Vec<String> = (0..20)
            .map(|i| format!(r#"{{"session_id":{i}}}"#))
            .collect();
        let content = format!("[{}]", elements.join(","));
        let result = compress_json_array(&content, &JsonCompressConfig::default()).unwrap();

        assert!(
            result.content.contains("...10 more elements omitted..."),
            "expected the exact bare marker with no facts, got: {}",
            result.content
        );
    }
}
