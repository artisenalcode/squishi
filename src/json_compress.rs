//! JSON array compression: dedupe exact-duplicate elements, then cap to first + last N plus any "interesting" rows (error-keyword hits, structural/status outliers) a drop would otherwise bury. Scoped to arrays -- a single object has nothing repeatable to compress.
//!
//! Interest survivors keep original relative order, so the dropped set can split into several runs instead of one contiguous middle slice; each run gets its own independently-computed marker.
//!
//! Dropped-element markers are enriched with verified facts (see `crate::invariants`) from a second `raw_value`-based re-parse, isolated from the `Value`-based pipeline below -- `Value` renormalizes number formatting (`5.00` -> `5.0`), which would make a range marker lie about the source; `raw_value` doesn't, unlike crate-wide `arbitrary_precision` (confirmed in a throwaway build to also change `Value` equality everywhere, which would weaken the exact-duplicate dedup).

use crate::compaction;
use crate::invariants::{self, InvariantConfig, Unit};
use crate::outliers;
use serde_json::Value;
use serde_json::value::RawValue;
use std::collections::{BTreeMap, BTreeSet};

/// Tunable knobs for `compress_json_array` — `--level` varies `keep_edge`.
pub struct JsonCompressConfig {
    /// First N and last N elements kept when capping.
    pub keep_edge: usize,
    /// Thresholds for the invariant-disclosure marker text.
    pub invariants: InvariantConfig,
    /// Minimum byte-savings ratio (0.0..1.0) CSV-schema rendering must beat the deduped array's minified JSON by, to be used instead of the lossy row-selection path.
    pub lossless_min_savings_ratio: f64,
}

impl Default for JsonCompressConfig {
    fn default() -> Self {
        Self {
            keep_edge: 5,
            invariants: InvariantConfig::default(),
            lossless_min_savings_ratio: 0.30,
        }
    }
}

/// Which path produced `JsonCompressResult::content`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonRendering {
    /// The usual dedup + edge/interest row selection; `content` is a JSON array, possibly with string markers standing in for dropped runs.
    RowSelected,
    /// A cleanly tabular array rendered losslessly as `[N]{cols}` CSV -- no rows dropped, `content` is CSV text, not JSON.
    CsvSchema,
}

pub struct JsonCompressResult {
    pub original_elements: usize,
    pub compressed_elements: usize,
    pub content: String,
    pub rendering: JsonRendering,
}

/// Returns `None` if `content` isn't a JSON array -- caller should fall back to another compressor for objects/non-JSON.
pub fn compress_json_array(
    content: &str,
    config: &JsonCompressConfig,
) -> Option<JsonCompressResult> {
    let value: Value = serde_json::from_str(content.trim()).ok()?;
    let Value::Array(elements) = value else {
        return None;
    };

    let original_elements = elements.len();

    // Dedupe exact-duplicate elements, preserving first-seen order and each kept element's ORIGINAL index, so the invariant-disclosure pass can recover its raw_value text.
    let mut seen: Vec<Value> = Vec::new();
    let mut seen_original_index: Vec<usize> = Vec::new();
    for (i, el) in elements.iter().enumerate() {
        if !seen.contains(el) {
            seen.push(el.clone());
            seen_original_index.push(i);
        }
    }

    // `render_csv_schema` itself declines arrays under 2 items.
    if let Some(csv) = compaction::render_csv_schema(&seen) {
        let minified_len = serde_json::to_string(&seen).map(|s| s.len()).unwrap_or(0);
        let savings = if minified_len == 0 {
            0.0
        } else {
            1.0 - (csv.len() as f64 / minified_len as f64)
        };
        if savings >= config.lossless_min_savings_ratio {
            return Some(JsonCompressResult {
                original_elements,
                compressed_elements: seen.len(),
                content: csv,
                rendering: JsonRendering::CsvSchema,
            });
        }
    }

    let keep_edge = config.keep_edge;
    let final_elements: Vec<Value> = if seen.len() > keep_edge * 2 {
        let kept_positions = select_kept_positions(&seen, keep_edge);
        build_output_with_markers(
            content,
            &seen,
            &seen_original_index,
            &kept_positions,
            &config.invariants,
        )
    } else {
        seen
    };

    let compressed_elements = final_elements.len();
    let content = serde_json::to_string(&Value::Array(final_elements)).ok()?;

    Some(JsonCompressResult {
        original_elements,
        compressed_elements,
        content,
        rendering: JsonRendering::RowSelected,
    })
}

/// Positions into `seen` to keep verbatim: first/last `keep_edge` plus up to `keep_edge * 2` "interesting" survivors (error-keyword hits first, then structural/status outliers). Interest positions already inside the edge budget don't cost anything extra.
fn select_kept_positions(seen: &[Value], keep_edge: usize) -> BTreeSet<usize> {
    let is_edge = |pos: usize| pos < keep_edge || pos >= seen.len() - keep_edge;
    let interest_budget = keep_edge * 2;

    let mut kept: BTreeSet<usize> = (0..keep_edge)
        .chain(seen.len() - keep_edge..seen.len())
        .collect();

    let mut interest_spent = 0usize;
    let error_positions = outliers::detect_error_items_for_preservation(seen, None);
    let structural_positions = outliers::detect_structural_outliers(seen);
    for pos in error_positions.into_iter().chain(structural_positions) {
        if is_edge(pos) || kept.contains(&pos) {
            continue;
        }
        if interest_spent >= interest_budget {
            break;
        }
        kept.insert(pos);
        interest_spent += 1;
    }

    kept
}

/// Walks `seen` in order, keeping every `kept_positions` entry verbatim and collapsing each run of consecutive dropped positions into its own independently-computed marker.
fn build_output_with_markers(
    content: &str,
    seen: &[Value],
    seen_original_index: &[usize],
    kept_positions: &BTreeSet<usize>,
    invariant_config: &InvariantConfig,
) -> Vec<Value> {
    let mut final_elements: Vec<Value> = Vec::new();
    let mut pending_run: Vec<usize> = Vec::new();

    let flush = |run: &mut Vec<usize>, out: &mut Vec<Value>| {
        if run.is_empty() {
            return;
        }
        let dropped_original: Vec<usize> =
            run.iter().map(|&pos| seen_original_index[pos]).collect();
        out.push(Value::String(build_marker(
            content,
            &dropped_original,
            invariant_config,
        )));
        run.clear();
    };

    for (pos, element) in seen.iter().enumerate() {
        if kept_positions.contains(&pos) {
            flush(&mut pending_run, &mut final_elements);
            final_elements.push(element.clone());
        } else {
            pending_run.push(pos);
        }
    }
    flush(&mut pending_run, &mut final_elements);

    final_elements
}

/// Builds the dropped-elements marker: bare count-only, enriched with `invariants::describe`'s verified-fact suffix when something safe survives disclosure. Falls back to bare on any parse failure.
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

/// Field name -> exact original-formatted value text for one raw array element, if it's a JSON object. `None` for scalars/arrays.
fn raw_object_fields(raw: &RawValue) -> Option<Vec<(String, String)>> {
    let map: BTreeMap<String, Box<RawValue>> = serde_json::from_str(raw.get()).ok()?;
    Some(
        map.into_iter()
            .map(|(k, v)| (k, display_raw(v.get())))
            .collect(),
    )
}

/// A raw JSON value's text as it should appear in a marker: a quoted string is unescaped (`"widget"` -> `widget`); anything else is already correct as raw JSON text.
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
    fn cleanly_tabular_array_renders_as_csv_schema_not_row_selection() {
        // Every element is a small, uniform, all-scalar object -- the
        // exact shape step 4/5's lossless path exists for. No
        // session_token/nested field here to disqualify it.
        let elements: Vec<String> = (0..50)
            .map(|i| format!(r#"{{"id":{i},"status":"ok"}}"#))
            .collect();
        let content = format!("[{}]", elements.join(","));
        let result = compress_json_array(&content, &JsonCompressConfig::default()).unwrap();

        assert_eq!(result.rendering, JsonRendering::CsvSchema);
        assert_eq!(result.original_elements, 50);
        // Lossless: every deduped row survives, nothing dropped.
        assert_eq!(result.compressed_elements, 50);
        assert!(result.content.starts_with("[50]{"));
        assert!(result.content.contains("0,ok"));
        assert!(result.content.contains("49,ok"));
        // Not a JSON array -- CSV text.
        assert!(serde_json::from_str::<Value>(&result.content).is_err());
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
        // `session_token` is a credential-named array cell: it disqualifies
        // the CSV-schema path (an array cell isn't scalar) and is
        // unconditionally withheld by invariants::describe's credential-name
        // check, so it can't add a stray fact either. That keeps this test on
        // the row-selection path it's actually about -- CSV-schema has its
        // own tests in `compaction.rs`.
        let elements: Vec<String> = (0..50)
            .map(|i| format!(r#"{{"id":{i},"session_token":[{i}]}}"#))
            .collect();
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
        // `session_token` disqualifies CSV-schema and is credential-named, so it can't add a stray invariants fact.
        let elements: Vec<String> = (0..10)
            .map(|i| format!(r#"{{"id":{i},"session_token":[{i}]}}"#))
            .collect();
        let content = format!("[{}]", elements.join(","));
        let config = JsonCompressConfig {
            keep_edge: 2,
            ..JsonCompressConfig::default()
        };
        let result = compress_json_array(&content, &config).unwrap();
        assert_eq!(result.compressed_elements, 2 * 2 + 1);
        assert!(result.content.contains(r#""id":0"#));
        assert!(result.content.contains(r#""id":1"#));
        assert!(!result.content.contains(r#""id":2"#));
    }

    /// End-to-end proof the enriched marker fires through `compress_json_array`, not just the isolated `invariants` module.
    #[test]
    fn dropped_elements_marker_states_constant_enum_and_range_facts() {
        // `session_token` disqualifies CSV-schema and is credential-named, so it can't add a stray invariants fact.
        let statuses = ["active", "pending", "closed"];
        let elements: Vec<String> = (0..20)
            .map(|i| {
                let status = statuses[i % statuses.len()];
                let amount = 5.00 + i as f64;
                format!(
                    r#"{{"id":{i},"kind":"widget","status":"{status}","amount":{amount:.2},"session_token":[{i}]}}"#
                )
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
        // Facts are additive to the bare count, never a replacement for it.
        assert!(result.content.contains("more elements omitted"));
    }

    /// An array of objects with a unique identifier per element gets a coverage statement, not a huge enumeration.
    #[test]
    fn dropped_elements_with_a_unique_id_each_render_coverage_not_enumeration() {
        // `session_token` disqualifies CSV-schema and is credential-named, so it can't add a stray invariants fact.
        let elements: Vec<String> = (0..20)
            .map(|i| format!(r#"{{"order_id":"ord-{i:04}","session_token":[{i}]}}"#))
            .collect();
        let content = format!("[{}]", elements.join(","));
        let result = compress_json_array(&content, &JsonCompressConfig::default()).unwrap();

        // Dense, contiguous order_ids -- the strongest coverage form: verified full membership, not just a count.
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

    /// A field failing the safety bar is withheld entirely even while a different field on the same units gets disclosed -- withholding is per-field, not all-or-nothing.
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
        // `session_token` disqualifies CSV-schema and is credential-named, so it can't add a stray invariants fact.
        let elements: Vec<String> = (0..20)
            .map(|i| {
                format!(
                    r#"{{"id":{i},"note":"{}","session_token":[{i}]}}"#,
                    notes[i % notes.len()]
                )
            })
            .collect();
        let content = format!("[{}]", elements.join(","));
        let result = compress_json_array(&content, &JsonCompressConfig::default()).unwrap();

        // Isolate the marker text, since kept edge elements legitimately still contain "note" verbatim -- the marker is a JSON string starting with `"...`, and never contains `"` itself.
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

    /// An error item buried mid-array survives the edge cap and splits the drop into two separately-computed markers instead of one assumed-contiguous marker.
    #[test]
    fn error_item_buried_in_the_middle_survives_and_splits_the_marker() {
        // `session_token` disqualifies CSV-schema; without it this fixture is cleanly tabular and the lossless path would fire instead of row-selection.
        let mut elements: Vec<String> = (0..30)
            .map(|i| format!(r#"{{"id":{i},"status":"ok","session_token":[{i}]}}"#))
            .collect();
        elements[15] =
            r#"{"id":15,"status":"error","msg":"request failed","session_token":[15]}"#.to_string();
        let content = format!("[{}]", elements.join(","));
        let result = compress_json_array(&content, &JsonCompressConfig::default()).unwrap();

        assert!(
            result.content.contains(r#""id":15"#),
            "buried error item must survive: {}",
            result.content
        );
        // Edge keeps (5 + 5) + the survivor + two markers, one per gap.
        assert_eq!(result.compressed_elements, 5 + 5 + 1 + 2);
        assert!(result.content.contains("...10 more elements omitted"));
        assert!(result.content.contains("...9 more elements omitted"));
    }

    /// When nothing in the dropped elements clears the safety bar, the marker falls back to the bare count-only form -- never one that looks enriched but says nothing real.
    #[test]
    fn dropped_elements_with_no_safe_facts_at_all_keep_the_exact_bare_marker() {
        // `session_token` disqualifies CSV-schema and is credential-named, so it can't add a stray invariants fact.
        let elements: Vec<String> = (0..20)
            .map(|i| format!(r#"{{"session_id":{i},"session_token":[{i}]}}"#))
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
