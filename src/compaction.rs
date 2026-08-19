//! Lossless CSV-schema rendering for cleanly tabular JSON arrays. Scoped to a flat single-level path only: no bucketing for heterogeneous arrays, no recursion into nested arrays/stringified-JSON, no nested-uniform column flattening, no substitution for long/opaque strings. Any array needing one of those loses out -- `render_csv_schema` returns `None` and the caller falls back to lossy row-selection.
//!
//! A heterogeneous array with no clean discriminator still renders as a sparse table (missing fields blank) rather than declining outright.
//!
//! Every cell that renders goes out as a real JSON scalar in full, keeping this path genuinely lossless -- at the cost of not shrinking arrays that lean on long opaque blobs, which the caller's own savings-ratio gate filters out instead.

use serde_json::Value;
use std::collections::BTreeMap;

struct FieldSpec {
    name: String,
    type_tag: &'static str,
    nullable: bool,
}

/// Renders `items` as a `[N]{col:type,...}` header followed by one CSV row per item, or `None` if the array isn't cleanly tabular: fewer than 2 items, a non-object item, or a nested/array/stringified-JSON cell.
pub fn render_csv_schema(items: &[Value]) -> Option<String> {
    if items.len() < 2 {
        return None;
    }
    let objects: Vec<&serde_json::Map<String, Value>> =
        items.iter().map(|v| v.as_object()).collect::<Option<_>>()?;
    if objects
        .iter()
        .any(|obj| obj.values().any(|v| !is_scalar(v)))
    {
        return None;
    }

    let key_freqs = compute_key_freqs(&objects);
    let total = items.len();

    let mut ordered: Vec<(&String, &usize)> = key_freqs.iter().collect();
    ordered.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    let ordered_keys: Vec<&str> = ordered.into_iter().map(|(k, _)| k.as_str()).collect();

    let fields: Vec<FieldSpec> = ordered_keys
        .iter()
        .map(|&k| FieldSpec {
            name: k.to_string(),
            type_tag: infer_type_tag(&objects, k),
            nullable: key_freqs[k] < total
                || objects
                    .iter()
                    .any(|o| matches!(o.get(k), Some(Value::Null))),
        })
        .collect();

    let mut out = String::new();
    out.push('[');
    out.push_str(&items.len().to_string());
    out.push_str("]{");
    let col_decl: Vec<String> = fields
        .iter()
        .map(|f| {
            if f.nullable {
                format!("{}:{}?", f.name, f.type_tag)
            } else {
                format!("{}:{}", f.name, f.type_tag)
            }
        })
        .collect();
    out.push_str(&col_decl.join(","));
    out.push_str("}\n");

    for obj in &objects {
        let cells: Vec<String> = ordered_keys
            .iter()
            .map(|&k| match obj.get(k) {
                None => String::new(),
                Some(v) => json_scalar_to_csv(v),
            })
            .collect();
        out.push_str(&cells.join(","));
        out.push('\n');
    }

    Some(out)
}

/// Everything that isn't an object or array. `Value::Null` is scalar (an absent/nullable field).
fn is_scalar(v: &Value) -> bool {
    !matches!(v, Value::Object(_) | Value::Array(_))
}

fn compute_key_freqs(objects: &[&serde_json::Map<String, Value>]) -> BTreeMap<String, usize> {
    let mut freqs: BTreeMap<String, usize> = BTreeMap::new();
    for obj in objects {
        for k in obj.keys() {
            *freqs.entry(k.clone()).or_insert(0) += 1;
        }
    }
    freqs
}

fn infer_type_tag(objects: &[&serde_json::Map<String, Value>], key: &str) -> &'static str {
    let mut tag: Option<&'static str> = None;
    for obj in objects {
        let Some(v) = obj.get(key) else { continue };
        if matches!(v, Value::Null) {
            continue;
        }
        let t = type_tag_for(v);
        match tag {
            None => tag = Some(t),
            Some(existing) if existing != t => return "json",
            _ => {}
        }
    }
    tag.unwrap_or("string")
}

fn type_tag_for(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_i64() || n.is_u64() => "int",
        Value::Number(_) => "float",
        Value::String(_) => "string",
        Value::Object(_) | Value::Array(_) => "json",
    }
}

fn json_scalar_to_csv(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            if needs_csv_quote(s) {
                csv_quote(s)
            } else {
                s.clone()
            }
        }
        // Object/array is unreachable -- is_scalar already declined the whole array.
        _ => csv_quote(&serde_json::to_string(v).unwrap_or_default()),
    }
}

fn needs_csv_quote(s: &str) -> bool {
    s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r')
}

fn csv_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        if c == '"' {
            out.push('"');
            out.push('"');
        } else {
            out.push(c);
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn too_few_items_declines() {
        let items = vec![json!({"a": 1})];
        assert!(render_csv_schema(&items).is_none());
    }

    #[test]
    fn non_object_array_declines() {
        let items = vec![json!(1), json!(2), json!(3)];
        assert!(render_csv_schema(&items).is_none());
    }

    #[test]
    fn nested_object_cell_declines() {
        let items = vec![
            json!({"id": 1, "meta": {"region": "us"}}),
            json!({"id": 2, "meta": {"region": "eu"}}),
        ];
        assert!(render_csv_schema(&items).is_none());
    }

    #[test]
    fn nested_array_cell_declines() {
        let items = vec![
            json!({"id": 1, "tags": ["a", "b"]}),
            json!({"id": 2, "tags": ["c"]}),
        ];
        assert!(render_csv_schema(&items).is_none());
    }

    #[test]
    fn pure_tabular_renders_header_and_rows() {
        let items = vec![
            json!({"id": 1, "name": "alice", "status": "ok"}),
            json!({"id": 2, "name": "bob", "status": "ok"}),
            json!({"id": 3, "name": "carol", "status": "fail"}),
        ];
        let out = render_csv_schema(&items).unwrap();
        let mut lines = out.lines();
        let header = lines.next().unwrap();
        assert!(header.starts_with("[3]{"));
        assert!(header.contains("id:int"));
        assert!(header.contains("name:string"));
        assert!(header.contains("status:string"));
        assert_eq!(lines.clone().count(), 3);
        assert!(out.contains("1,alice,ok"));
        assert!(out.contains("3,carol,fail"));
    }

    #[test]
    fn nullable_field_marked_with_question_mark() {
        let items = vec![json!({"id": 1, "tag": "a"}), json!({"id": 2})];
        let out = render_csv_schema(&items).unwrap();
        let header = out.lines().next().unwrap();
        assert!(header.contains("tag:string?"), "got: {header}");
        assert!(!header.contains("id:int?"), "got: {header}");
    }

    #[test]
    fn mixed_type_column_falls_back_to_json_tag() {
        let items = vec![json!({"v": 1}), json!({"v": "text"})];
        let out = render_csv_schema(&items).unwrap();
        let header = out.lines().next().unwrap();
        assert!(header.contains("v:json"), "got: {header}");
    }

    #[test]
    fn stable_field_ordering_is_frequency_then_alphabetical() {
        let items = vec![
            json!({"common": 1, "z_rare": 1}),
            json!({"common": 2, "a_rare": 1}),
            json!({"common": 3}),
        ];
        let out = render_csv_schema(&items).unwrap();
        let header = out.lines().next().unwrap();
        let common_pos = header.find("common").unwrap();
        let a_rare_pos = header.find("a_rare").unwrap();
        let z_rare_pos = header.find("z_rare").unwrap();
        assert!(common_pos < a_rare_pos);
        assert!(a_rare_pos < z_rare_pos);
    }

    #[test]
    fn comma_and_quote_values_are_csv_quoted() {
        let items = vec![
            json!({"note": "has, a comma"}),
            json!({"note": "has \"quotes\""}),
        ];
        let out = render_csv_schema(&items).unwrap();
        assert!(out.contains("\"has, a comma\""));
        assert!(out.contains("\"has \"\"quotes\"\"\""));
    }

    #[test]
    fn long_string_cell_renders_verbatim_not_substituted() {
        // No opaque substitution -- a long string renders in full. Losslessness over token savings.
        let blob = "x".repeat(500);
        let items = vec![
            json!({"id": 1, "blob": blob.clone()}),
            json!({"id": 2, "blob": "short"}),
        ];
        let out = render_csv_schema(&items).unwrap();
        assert!(out.contains(&blob));
    }

    /// Fields barely overlap across rows -- no bucketing here, so it goes straight to a sparse table.
    #[test]
    fn heterogeneous_array_renders_as_a_sparse_table() {
        let items = vec![
            json!({"a": 1}),
            json!({"b": 1}),
            json!({"c": 1}),
            json!({"d": 1}),
            json!({"e": 1}),
        ];
        let out = render_csv_schema(&items).unwrap();
        let header = out.lines().next().unwrap();
        for col in ["a", "b", "c", "d", "e"] {
            assert!(header.contains(col), "got: {header}");
        }
        assert!(out.contains("1,,,,"), "expected a sparse row, got: {out}");
    }
}
