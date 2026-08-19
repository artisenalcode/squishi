//! Verified-fact elision markers: when a compressor drops units, state only what's actually true about the dropped set instead of a bare count. Withholds entirely, never partially, when nothing meets the bar -- a partial disclosure reads as complete and is worse than none.
//!
//! Decision order per field (first match wins): constant across every unit that has it -> state once. Else all present values parse as numbers -> a range, printed from the real ORIGINAL value strings, never a reparsed/renormalized number. Else every present value is distinct (identifier-shaped) -> coverage (distinct count + byte-wise min/max), or verified full membership if counted to be exactly dense. Else, if distinct count is small and every value is safe to print -> a full enumeration with an `absent×N` bucket. Otherwise withheld entirely.

use regex::Regex;
use std::sync::LazyLock;

/// One dropped unit's fields, as (name, original-formatted value text) pairs. Callers must supply real source text, never a reparsed/renormalized value -- `5.00` silently becoming `5.0` would make a range marker lie about the source.
pub struct Unit {
    pub fields: Vec<(String, String)>,
}

impl Unit {
    pub fn new(fields: Vec<(String, String)>) -> Self {
        Self { fields }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

pub struct InvariantConfig {
    /// A varying, non-identifier field with more distinct values than this is withheld entirely rather than partially enumerated.
    pub max_enum_values: usize,
    /// A value at or over this many bytes withholds its whole field -- a long value reads as content, not a safe label.
    pub max_value_len: usize,
    /// Flat cap on the disclosure suffix appended to the bare marker.
    pub max_marker_bytes: usize,
}

impl Default for InvariantConfig {
    fn default() -> Self {
        Self {
            max_enum_values: 5,
            max_value_len: 24,
            max_marker_bytes: 160,
        }
    }
}

static CREDENTIAL_FIELD_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)token|secret|password|passwd|api[_-]?key|auth|credential|session[_-]?id")
        .unwrap()
});

enum Fact {
    Constant {
        field: String,
        value: String,
    },
    Range {
        field: String,
        min: String,
        max: String,
    },
    Coverage {
        field: String,
        distinct: usize,
        min: String,
        max: String,
        dense: bool,
    },
    Enumeration {
        field: String,
        counts: Vec<(String, usize)>,
        absent: usize,
    },
}

/// `None` means nothing survived disclosure -- the caller keeps its bare "N omitted" marker unchanged.
pub fn describe(units: &[Unit], config: &InvariantConfig) -> Option<String> {
    if units.is_empty() {
        return None;
    }

    // Union of field names across all units, first-seen order, for deterministic output.
    let mut field_names: Vec<&str> = Vec::new();
    for u in units {
        for (k, _) in &u.fields {
            if !field_names.contains(&k.as_str()) {
                field_names.push(k.as_str());
            }
        }
    }

    let mut facts: Vec<Fact> = field_names
        .iter()
        .filter_map(|field| analyze_field(field, units, config))
        .collect();

    if facts.is_empty() {
        return None;
    }

    // Priority order: constant, range, coverage, enumeration.
    facts.sort_by_key(|f| match f {
        Fact::Constant { .. } => 0,
        Fact::Range { .. } => 1,
        Fact::Coverage { .. } => 2,
        Fact::Enumeration { .. } => 3,
    });

    let mut rendered: Vec<String> = Vec::new();
    let mut used = 0usize;
    for fact in &facts {
        let text = render_fact(fact);
        let cost = text.len() + if rendered.is_empty() { 0 } else { 1 };
        if used + cost > config.max_marker_bytes {
            break;
        }
        used += cost;
        rendered.push(text);
    }

    if rendered.is_empty() {
        return None;
    }
    Some(rendered.join(" "))
}

fn analyze_field(field: &str, units: &[Unit], config: &InvariantConfig) -> Option<Fact> {
    // Credential-shaped field NAMES are withheld regardless of value shape, checked before any classification, so a numeric `session_id` can't sneak out as a range.
    if CREDENTIAL_FIELD_RE.is_match(field) {
        return None;
    }

    let present: Vec<&str> = units.iter().filter_map(|u| u.get(field)).collect();
    if present.is_empty() {
        return None;
    }
    let absent = units.len() - present.len();

    let mut distinct_vals: Vec<&str> = Vec::new();
    for v in &present {
        if !distinct_vals.contains(v) {
            distinct_vals.push(v);
        }
    }

    // Constant, but ONLY when every unit has the field -- a value seen in 1 of 3 units would read as a universal claim about all dropped units. With any absences, this falls through to enumeration instead, which states the absent count explicitly.
    if absent == 0 && distinct_vals.len() == 1 {
        return Some(Fact::Constant {
            field: field.to_string(),
            value: distinct_vals[0].to_string(),
        });
    }

    // All present values numeric? -> range, printed from the real original strings, not the parsed f64. A range only claims about values it actually saw, so absences don't create the same risk a bare constant would.
    if let Some(nums) = present
        .iter()
        .map(|v| v.parse::<f64>().ok())
        .collect::<Option<Vec<f64>>>()
    {
        let mut idx_min = 0;
        let mut idx_max = 0;
        for (i, &n) in nums.iter().enumerate() {
            if n < nums[idx_min] {
                idx_min = i;
            }
            if n > nums[idx_max] {
                idx_max = i;
            }
        }
        return Some(Fact::Range {
            field: field.to_string(),
            min: present[idx_min].to_string(),
            max: present[idx_max].to_string(),
        });
    }

    // Distinct-per-unit (identifier-shaped)? Requires MORE present units than the enumeration cap allows -- with few units, every varying field trivially has "every value distinct", which would wrongly steal cases enumeration handles more precisely. Coverage is for when enumeration isn't safe: too many identifiers to list.
    if distinct_vals.len() == present.len() && present.len() > config.max_enum_values {
        let mut sorted = distinct_vals.clone();
        sorted.sort_unstable();
        let min = *sorted.first().unwrap();
        let max = *sorted.last().unwrap();
        // Coverage shows these two values verbatim -- if either reads more like content than a short identifier, don't disclose this field at all.
        let unsafe_extreme = [min, max]
            .iter()
            .any(|v| v.len() >= config.max_value_len || v.contains(' '));
        if !unsafe_extreme {
            let dense = is_dense(&distinct_vals);
            return Some(Fact::Coverage {
                field: field.to_string(),
                distinct: distinct_vals.len(),
                min: min.to_string(),
                max: max.to_string(),
                dense,
            });
        }
    }

    // Small, safe-to-print enumeration? Covers the single-distinct-value-with-absences case the constant branch declined to claim.
    if distinct_vals.len() <= config.max_enum_values {
        let unsafe_value = distinct_vals
            .iter()
            .any(|v| v.len() >= config.max_value_len || v.contains(' '));
        if !unsafe_value {
            let counts: Vec<(String, usize)> = distinct_vals
                .iter()
                .map(|v| {
                    let c = present.iter().filter(|p| *p == v).count();
                    (v.to_string(), c)
                })
                .collect();
            return Some(Fact::Enumeration {
                field: field.to_string(),
                counts,
                absent,
            });
        }
    }

    // Doesn't meet the bar for anything safe to say -- withhold entirely.
    None
}

/// Best-effort: only detects the "shared prefix + fixed-width numeric suffix + contiguous" shape (e.g. `wh-5000`..`wh-5059`). Density is counted from real parsed values, never inferred from endpoints alone.
fn is_dense(values: &[&str]) -> bool {
    let mut parsed: Vec<(&str, u64, usize)> = Vec::with_capacity(values.len());
    for v in values {
        let digit_count = v.chars().rev().take_while(|c| c.is_ascii_digit()).count();
        if digit_count == 0 || digit_count == v.len() {
            return false;
        }
        let split_at = v.len() - digit_count;
        let prefix = &v[..split_at];
        let suffix = &v[split_at..];
        let Ok(n) = suffix.parse::<u64>() else {
            return false;
        };
        parsed.push((prefix, n, suffix.len()));
    }
    let first_prefix = parsed[0].0;
    let first_width = parsed[0].2;
    if !parsed
        .iter()
        .all(|(p, _, w)| *p == first_prefix && *w == first_width)
    {
        return false;
    }
    let mut nums: Vec<u64> = parsed.iter().map(|(_, n, _)| *n).collect();
    nums.sort_unstable();
    let min = *nums.first().unwrap();
    let max = *nums.last().unwrap();
    (max - min + 1) as usize == nums.len()
}

fn render_fact(fact: &Fact) -> String {
    match fact {
        Fact::Constant { field, value } => format!("{field}={value}"),
        Fact::Range { field, min, max } => format!("range {field}={min}..{max}"),
        Fact::Coverage {
            field,
            distinct,
            min,
            max,
            dense,
        } => {
            if *dense {
                format!("{field}: {min}..{max} all {distinct} present")
            } else {
                format!("{field}: {distinct} distinct, {min}..{max}")
            }
        }
        Fact::Enumeration {
            field,
            counts,
            absent,
        } => {
            let mut parts: Vec<String> = counts.iter().map(|(v, c)| format!("{v}×{c}")).collect();
            if *absent > 0 {
                parts.push(format!("absent×{absent}"));
            }
            format!("{field}: {}", parts.join(" "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(pairs: &[(&str, &str)]) -> Unit {
        Unit::new(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    #[test]
    fn constant_field_detected_and_rendered() {
        let units = vec![
            unit(&[("state", "charged")]),
            unit(&[("state", "charged")]),
            unit(&[("state", "charged")]),
        ];
        let marker = describe(&units, &InvariantConfig::default()).unwrap();
        assert_eq!(marker, "state=charged");
    }

    #[test]
    fn numeric_varying_field_renders_a_range_using_original_strings() {
        let units = vec![
            unit(&[("amount", "5.00")]),
            unit(&[("amount", "199.99")]),
            unit(&[("amount", "42.50")]),
        ];
        let marker = describe(&units, &InvariantConfig::default()).unwrap();
        assert_eq!(marker, "range amount=5.00..199.99");
    }

    #[test]
    fn identifier_shaped_field_renders_coverage_not_enumeration() {
        // 8 distinct order_ids, above max_enum_values (5) and every value unique -- exactly the case coverage exists for.
        let ids = [
            "1003", "1000", "1047", "1012", "1099", "1055", "1071", "1030",
        ];
        let units: Vec<Unit> = ids
            .iter()
            .map(|n| unit(&[("order_id", &format!("ord-{n}"))]))
            .collect();
        let marker = describe(&units, &InvariantConfig::default()).unwrap();
        assert_eq!(marker, "order_id: 8 distinct, ord-1000..ord-1099");
    }

    #[test]
    fn dense_identifier_run_renders_verified_full_membership() {
        let units: Vec<Unit> = (5000..=5059)
            .map(|n| unit(&[("bin", &format!("wh-{n}"))]))
            .collect();
        let marker = describe(&units, &InvariantConfig::default()).unwrap();
        assert_eq!(marker, "bin: wh-5000..wh-5059 all 60 present");
    }

    #[test]
    fn more_than_max_distinct_non_identifier_field_is_withheld_entirely() {
        // 7 distinct `status` values across 10 units (some repeat, so not identifier-shaped), above max_enum_values -- too many to enumerate, not uniform enough for coverage.
        let statuses = [
            "queued", "queued", "running", "running", "done", "failed", "retrying", "queued",
            "done", "paused",
        ];
        let units: Vec<Unit> = statuses
            .iter()
            .map(|s| unit(&[("status", s), ("kind", "widget")]))
            .collect();
        let marker = describe(&units, &InvariantConfig::default()).unwrap();
        assert!(!marker.contains("status"), "got: {marker}");
        assert!(marker.contains("kind=widget"), "got: {marker}");
    }

    #[test]
    fn a_long_or_spaced_value_withholds_its_whole_field() {
        let units = vec![
            unit(&[("note", "a value with spaces in it")]),
            unit(&[("note", "another distinct note")]),
        ];
        let marker = describe(&units, &InvariantConfig::default());
        assert!(marker.is_none(), "got: {marker:?}");
    }

    #[test]
    fn a_credential_shaped_field_name_is_withheld_regardless_of_value_shape() {
        let units = vec![
            unit(&[("api_key", "abc123")]),
            unit(&[("api_key", "def456")]),
        ];
        let marker = describe(&units, &InvariantConfig::default());
        assert!(marker.is_none(), "got: {marker:?}");
    }

    #[test]
    fn absent_bucket_is_accurate() {
        let units = vec![
            unit(&[("status", "fulfilled")]),
            unit(&[("status", "shipped")]),
            unit(&[]), // no status field at all
        ];
        let marker = describe(&units, &InvariantConfig::default()).unwrap();
        assert!(marker.contains("absent×1"), "got: {marker}");
        assert!(marker.contains("fulfilled×1"), "got: {marker}");
        assert!(marker.contains("shipped×1"), "got: {marker}");
    }

    #[test]
    fn a_credential_shaped_field_name_is_withheld_even_when_numeric() {
        // Regression: the name check originally ran only inside the enumeration branch, so a numeric `session_id` slipped out as a range.
        let units = vec![
            unit(&[("session_id", "1000")]),
            unit(&[("session_id", "2000")]),
            unit(&[("session_id", "3000")]),
        ];
        let marker = describe(&units, &InvariantConfig::default());
        assert!(marker.is_none(), "got: {marker:?}");
    }

    #[test]
    fn coverage_withholds_a_field_whose_extreme_values_are_long_or_spaced() {
        // Coverage shows two real values (min/max) verbatim -- if either looks like content, don't disclose.
        let units: Vec<Unit> = (0..10)
            .map(|i| {
                let value = format!("a fairly long unique blob value number {i}");
                unit(&[("blob", &value)])
            })
            .collect();
        let marker = describe(&units, &InvariantConfig::default());
        assert!(marker.is_none(), "got: {marker:?}");
    }

    #[test]
    fn zero_units_returns_none() {
        assert!(describe(&[], &InvariantConfig::default()).is_none());
    }

    #[test]
    fn a_field_present_in_only_some_units_is_never_stated_as_a_bare_constant() {
        // A value seen in 1 of 3 units must not render as "region=us-east" -- that reads as a universal claim; must fall through to enumeration instead.
        let units = vec![unit(&[("region", "us-east")]), unit(&[]), unit(&[])];
        let marker = describe(&units, &InvariantConfig::default()).unwrap();
        assert!(
            !marker.contains("region=us-east"),
            "must not render as a bare constant: {marker}"
        );
        assert!(marker.contains("us-east×1"), "got: {marker}");
        assert!(marker.contains("absent×2"), "got: {marker}");
    }
}
