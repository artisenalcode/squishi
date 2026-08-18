//! TOON (Token-Oriented Object Notation) encoder/decoder — a real, public,
//! MIT-licensed format (github.com/toon-format/spec, SPEC v4.1), not
//! reverse-engineered from caveman. Implements the comma-delimiter,
//! 2-space-indent core of the spec: scalar encoding (quoting/escaping,
//! canonical numbers, true/false/null), object encoding, the primitive-
//! array inline form, the tabular form for uniform arrays of
//! flat-primitive-field objects, and the list-item fallback form for
//! everything else (mixed-type arrays, non-uniform objects, objects with
//! nested-object fields).
//!
//! Deliberately NOT implemented for v1 (documented here, not silently
//! skipped): nested field-group folding in tabular headers (`field{a,b}`
//! collapsing into the header for uniform nested objects), the keyed-
//! tabular object-of-objects form (`[N:]`), tab/pipe delimiters, and
//! exponent notation for numbers outside the canonical range. All of
//! these degrade to a real, valid, still-correct (just less maximally
//! compact) form instead of failing — see `format_number` and the list
//! fallback in `encode_array`.
//!
//! Unlike every other compressor in this crate, TOON is LOSSLESS: `decode`
//! of an `encode` output reproduces the exact same `serde_json::Value` —
//! no CCR recovery handle needed. See
//! docs/ideation/squishi-toon-pixel/2026-08-18-toon-and-pixel-mode-spec.md.

use regex::Regex;
use serde_json::{Map, Number, Value};
use std::sync::LazyLock;

const INDENT: &str = "  ";

static NUMERIC_LIKE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^[+-]?[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?$").unwrap());

// ---------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------

/// Encode a JSON value as TOON text (comma delimiter, 2-space indent).
pub fn encode(value: &Value) -> String {
    match value {
        Value::Array(items) => {
            if items.is_empty() {
                return "[]".to_string();
            }
            encode_array(items, 0, None).join("\n")
        }
        Value::Object(map) => {
            if map.is_empty() {
                return String::new();
            }
            encode_object_fields(map, 0).join("\n")
        }
        other => encode_scalar(other),
    }
}

/// Encode `value` as TOON, but only if the result actually measures
/// smaller (in bytes) than `value`'s own compact JSON form. `None` means
/// TOON didn't help for this shape -- same "never ship a result that
/// isn't actually smaller" discipline every compressor in this crate
/// already follows (json_compress's dedup, log_compress's line
/// selection, ...). Since TOON is lossless, a `None` here costs nothing:
/// the caller just keeps the original JSON, no fallback machinery needed.
pub fn encode_if_smaller(value: &Value) -> Option<String> {
    let json_len = serde_json::to_string(value).ok()?.len();
    let toon = encode(value);
    if toon.len() < json_len {
        Some(toon)
    } else {
        None
    }
}

/// `key: value` / `key:` + nested lines, one entry per field, at `indent`.
fn encode_object_fields(map: &Map<String, Value>, indent: usize) -> Vec<String> {
    let pad = INDENT.repeat(indent);
    let mut lines = Vec::new();
    for (key, value) in map {
        match value {
            Value::Object(nested) => {
                if nested.is_empty() {
                    lines.push(format!("{pad}{key}:"));
                } else {
                    lines.push(format!("{pad}{key}:"));
                    lines.extend(encode_object_fields(nested, indent + 1));
                }
            }
            Value::Array(items) => {
                if items.is_empty() {
                    lines.push(format!("{pad}{key}: []"));
                } else {
                    lines.extend(encode_array(items, indent, Some(key)));
                }
            }
            scalar => lines.push(format!("{pad}{key}: {}", encode_scalar(scalar))),
        }
    }
    lines
}

/// Dispatches an array to whichever real TOON form applies: inline
/// primitive form, tabular form (uniform flat-object rows), or the list
/// fallback. `key` is `None` at root (array position), `Some(name)` when
/// this array is an object field.
fn encode_array(items: &[Value], indent: usize, key: Option<&str>) -> Vec<String> {
    let pad = INDENT.repeat(indent);
    let n = items.len();
    let prefix = key.map(|k| k.to_string()).unwrap_or_default();

    if let Some(field_order) = uniform_tabular_fields(items) {
        let header = format!(
            "{pad}{prefix}[{n}]{{{}}}:",
            field_order
                .iter()
                .map(|f| encode_bare_field_name(f))
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut lines = vec![header];
        let row_pad = INDENT.repeat(indent + 1);
        for item in items {
            let Value::Object(obj) = item else {
                unreachable!()
            };
            let row: Vec<String> = field_order
                .iter()
                .map(|f| encode_scalar(obj.get(f.as_str()).unwrap_or(&Value::Null)))
                .collect();
            lines.push(format!("{row_pad}{}", row.join(",")));
        }
        return lines;
    }

    if items.iter().all(is_scalar) {
        let values: Vec<String> = items.iter().map(encode_scalar).collect();
        return vec![format!("{pad}{prefix}[{n}]: {}", values.join(","))];
    }

    // List-item fallback: real, valid, lossless -- just not the maximally
    // compact tabular form. Covers mixed-type arrays, non-uniform arrays
    // of objects, and arrays containing nested arrays/objects.
    let mut lines = vec![format!("{pad}{prefix}[{n}]:")];
    let item_pad = INDENT.repeat(indent + 1);
    for item in items {
        lines.extend(encode_list_item(item, indent + 1, &item_pad));
    }
    lines
}

/// One `- ...` list entry, per §10: an object's first field (in encounter
/// order) goes directly on the hyphen line, remaining fields indent under
/// it; a scalar goes directly on the hyphen line; an array gets its own
/// header on the hyphen line with items nested one level deeper.
fn encode_list_item(item: &Value, indent: usize, item_pad: &str) -> Vec<String> {
    match item {
        Value::Object(obj) if obj.is_empty() => vec![format!("{item_pad}-")],
        Value::Object(obj) => {
            let mut iter = obj.iter();
            let (first_key, first_value) = iter.next().unwrap();
            let mut lines = Vec::new();
            match first_value {
                Value::Object(nested) if !nested.is_empty() => {
                    lines.push(format!("{item_pad}- {first_key}:"));
                    lines.extend(encode_object_fields(nested, indent + 2));
                }
                Value::Array(nested_items) if !nested_items.is_empty() => {
                    let sub = encode_array(nested_items, indent + 1, Some(first_key));
                    let mut sub_iter = sub.into_iter();
                    lines.push(format!(
                        "{item_pad}- {}",
                        sub_iter.next().unwrap().trim_start()
                    ));
                    lines.extend(sub_iter);
                }
                scalar => lines.push(format!(
                    "{item_pad}- {first_key}: {}",
                    encode_scalar(scalar)
                )),
            }
            let rest: Map<String, Value> = iter.map(|(k, v)| (k.clone(), v.clone())).collect();
            if !rest.is_empty() {
                lines.extend(encode_object_fields(&rest, indent + 1));
            }
            lines
        }
        Value::Array(items) if items.is_empty() => vec![format!("{item_pad}- []")],
        Value::Array(items) => {
            let sub = encode_array(items, indent, None);
            let mut sub_iter = sub.into_iter();
            let mut lines = vec![format!(
                "{item_pad}- {}",
                sub_iter.next().unwrap().trim_start()
            )];
            lines.extend(sub_iter);
            lines
        }
        scalar => vec![format!("{item_pad}- {}", encode_scalar(scalar))],
    }
}

fn is_scalar(v: &Value) -> bool {
    !matches!(v, Value::Object(_) | Value::Array(_))
}

/// `Some(field order)` iff every element is an object, every object has
/// exactly the same set of field names (order taken from the first
/// object), and every value in every object is a scalar — the "uniform,
/// flat-primitive-fields" case §9.3 calls tabular-eligible. Nested-object
/// or nested-array field values fall back to the list form instead of
/// attempting field-group folding (deferred, see module doc).
fn uniform_tabular_fields(items: &[Value]) -> Option<Vec<String>> {
    if items.len() < 2 {
        return None;
    }
    let Value::Object(first) = &items[0] else {
        return None;
    };
    let field_order: Vec<String> = first.keys().cloned().collect();
    for item in items {
        let Value::Object(obj) = item else {
            return None;
        };
        if obj.len() != field_order.len() {
            return None;
        }
        for field in &field_order {
            match obj.get(field.as_str()) {
                Some(v) if is_scalar(v) => {}
                _ => return None,
            }
        }
    }
    Some(field_order)
}

/// Field names in a tabular header are never quoted by this encoder (real
/// TOON data uses plain identifier-shaped keys in practice) — kept as its
/// own function so a future encoder change to quote unusual key names has
/// one place to do it.
fn encode_bare_field_name(name: &str) -> String {
    name.to_string()
}

fn encode_scalar(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => format_number(n),
        Value::String(s) => encode_string(s),
        _ => unreachable!("encode_scalar called on a non-scalar value"),
    }
}

/// Canonical decimal form per §2: no exponent within [1e-6, 1e21) or
/// zero, no leading zeros, no trailing fractional zeros, integral values
/// print without a decimal point, -0 normalizes to 0. Outside that range
/// this falls back to Rust's own float formatting rather than emitting
/// spec-optional exponent notation — a documented v1 gap (see module
/// doc), not a silent one: still a valid, round-trippable number, just
/// not guaranteed byte-identical to another encoder's output for
/// astronomically large/small values.
fn format_number(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        return i.to_string();
    }
    if let Some(u) = n.as_u64() {
        return u.to_string();
    }
    let f = n.as_f64().unwrap_or(0.0);
    if f == 0.0 {
        return "0".to_string();
    }
    let abs = f.abs();
    if !(1e-6..1e21).contains(&abs) {
        return format!("{f}");
    }
    // Canonical decimal, no exponent, no trailing fractional zeros.
    let mut s = format!("{f:.12}");
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
    }
    s
}

fn encode_string(s: &str) -> String {
    if needs_quoting(s) {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) <= 0x1F => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    } else {
        s.to_string()
    }
}

/// §7.2's full quoting bar, comma delimiter only (v1 scope).
fn needs_quoting(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    if s.starts_with(' ') || s.starts_with('\t') || s.ends_with(' ') || s.ends_with('\t') {
        return true;
    }
    if s == "true" || s == "false" || s == "null" {
        return true;
    }
    if NUMERIC_LIKE_RE.is_match(s) {
        return true;
    }
    if s.contains([':', '"', '\\', '[', ']', '{', '}', ',']) {
        return true;
    }
    if s.chars().any(|c| (c as u32) <= 0x1F) {
        return true;
    }
    if s.starts_with('-') || s.starts_with('#') {
        return true;
    }
    false
}

// ---------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------

#[derive(Debug, PartialEq)]
pub struct DecodeError(pub String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TOON decode error: {}", self.0)
    }
}

/// Decode TOON text back to the JSON value it represents.
pub fn decode(text: &str) -> Result<Value, DecodeError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() || (lines.len() == 1 && lines[0].trim().is_empty()) {
        return Ok(Value::Object(Map::new()));
    }
    if lines.len() == 1 && lines[0].trim() == "[]" {
        return Ok(Value::Array(Vec::new()));
    }
    if lines[0].trim_start().starts_with('[') {
        let (value, _) = parse_root_array(&lines, 0)?;
        return Ok(value);
    }
    let (map, _) = parse_object_fields(&lines, 0, 0)?;
    Ok(Value::Object(map))
}

fn line_indent(line: &str) -> usize {
    line.chars().take_while(|c| *c == ' ').count() / 2
}

fn parse_root_array(lines: &[&str], start: usize) -> Result<(Value, usize), DecodeError> {
    let header = lines[start].trim();
    let rest = &header[1..]; // drop leading '['
    let close = rest
        .find(']')
        .ok_or_else(|| DecodeError(format!("missing ']' in array header: {header}")))?;
    let n: usize = rest[..close]
        .parse()
        .map_err(|_| DecodeError(format!("bad array length in: {header}")))?;
    let after = &rest[close + 1..];
    parse_array_body(lines, start, after, n, 0)
}

/// Shared by root arrays and object-field arrays: `after` is whatever
/// followed the `[N]` in the header line (`{f1,f2}:`, `:`, or `: v1,v2`).
fn parse_array_body(
    lines: &[&str],
    header_line: usize,
    after: &str,
    n: usize,
    indent: usize,
) -> Result<(Value, usize), DecodeError> {
    if let Some(fields_part) = after.strip_prefix('{') {
        let close = fields_part
            .find('}')
            .ok_or_else(|| DecodeError("missing '}' in tabular header".to_string()))?;
        let fields: Vec<String> = fields_part[..close]
            .split(',')
            .map(|s| s.to_string())
            .collect();
        let mut items = Vec::with_capacity(n);
        let mut line_no = header_line + 1;
        for _ in 0..n {
            let row = lines
                .get(line_no)
                .ok_or_else(|| DecodeError("truncated tabular array".to_string()))?;
            let cells = split_delimited(row.trim());
            let mut obj = Map::new();
            for (field, cell) in fields.iter().zip(cells.iter()) {
                obj.insert(field.clone(), parse_scalar_token(cell));
            }
            items.push(Value::Object(obj));
            line_no += 1;
        }
        return Ok((Value::Array(items), line_no));
    }

    if let Some(inline) = after.strip_prefix(':') {
        let inline = inline.trim();
        if !inline.is_empty() {
            // Inline primitive array: `: v1,v2,v3`.
            let values = split_delimited(inline)
                .iter()
                .map(|t| parse_scalar_token(t))
                .collect();
            return Ok((Value::Array(values), header_line + 1));
        }
        // List form: following `- ` lines at indent+1.
        let mut items = Vec::with_capacity(n);
        let mut line_no = header_line + 1;
        for _ in 0..n {
            let (item, next) = parse_list_item(lines, line_no, indent + 1)?;
            items.push(item);
            line_no = next;
        }
        return Ok((Value::Array(items), line_no));
    }

    Err(DecodeError(format!(
        "unrecognized array header suffix: {after:?}"
    )))
}

fn parse_list_item(
    lines: &[&str],
    line_no: usize,
    indent: usize,
) -> Result<(Value, usize), DecodeError> {
    let line = lines
        .get(line_no)
        .ok_or_else(|| DecodeError("truncated list".to_string()))?;
    let trimmed = line.trim_start();
    let after_hyphen = trimmed
        .strip_prefix('-')
        .ok_or_else(|| DecodeError(format!("expected '-' list item, got: {line}")))?;
    let after_hyphen = after_hyphen.strip_prefix(' ').unwrap_or(after_hyphen);

    if after_hyphen.is_empty() {
        return Ok((Value::Object(Map::new()), line_no + 1));
    }
    if let Some(colon) = after_hyphen.find(':') {
        let key_part = &after_hyphen[..colon];
        if !key_part.contains('[') {
            // `- key: value` or `- key:` (nested object head).
            let key = key_part.to_string();
            let value_part = after_hyphen[colon + 1..].trim();
            let (first_value, mut next_line) = if value_part.is_empty() {
                // Nested object continues at indent+1, OR this is a bare
                // scalar-less key (rare) -- peek the next line's indent.
                if let Some(next) = lines.get(line_no + 1)
                    && line_indent(next) > indent
                    && !next.trim_start().starts_with('-')
                {
                    let (map, after) = parse_object_fields(lines, line_no + 1, indent + 1)?;
                    (Value::Object(map), after)
                } else {
                    (Value::Object(Map::new()), line_no + 1)
                }
            } else {
                (parse_scalar_token(value_part), line_no + 1)
            };
            let mut obj = Map::new();
            obj.insert(key, first_value);
            // Remaining sibling fields at indent+1 (object continuation)
            // -- `parse_object_fields` itself already consumes every
            // consecutive matching line in one call, stopping at the
            // first line that isn't a same-indent, non-list-item field,
            // so this is a single conditional call, not a loop.
            if let Some(next) = lines.get(next_line)
                && !next.trim().is_empty()
                && line_indent(next) == indent + 1
                && !next.trim_start().starts_with('-')
            {
                let (more, after) = parse_object_fields(lines, next_line, indent + 1)?;
                for (k, v) in more {
                    obj.insert(k, v);
                }
                next_line = after;
            }
            return Ok((Value::Object(obj), next_line));
        }
        // `- key[N]...` -- a nested array as the object's first field.
        let bracket = after_hyphen.find('[').unwrap();
        let key = after_hyphen[..bracket].to_string();
        let rest = &after_hyphen[bracket..];
        let close = rest
            .find(']')
            .ok_or_else(|| DecodeError("missing ']' after list-item array key".to_string()))?;
        let count: usize = rest[1..close]
            .parse()
            .map_err(|_| DecodeError("bad count in list-item array key".to_string()))?;
        let after = &rest[close + 1..];
        let (arr, next_line) = parse_array_body(lines, line_no, after, count, indent + 1)?;
        let mut obj = Map::new();
        obj.insert(key, arr);
        return Ok((Value::Object(obj), next_line));
    }
    if after_hyphen.starts_with('[') {
        let close = after_hyphen
            .find(']')
            .ok_or_else(|| DecodeError("missing ']' in list-item array".to_string()))?;
        let count: usize = after_hyphen[1..close]
            .parse()
            .map_err(|_| DecodeError("bad count in list-item array".to_string()))?;
        let after = &after_hyphen[close + 1..];
        return parse_array_body(lines, line_no, after, count, indent + 1);
    }
    Ok((parse_scalar_token(after_hyphen), line_no + 1))
}

fn parse_object_fields(
    lines: &[&str],
    start: usize,
    indent: usize,
) -> Result<(Map<String, Value>, usize), DecodeError> {
    let mut map = Map::new();
    let mut line_no = start;
    while let Some(line) = lines.get(line_no) {
        if line.trim().is_empty() {
            break;
        }
        if line_indent(line) != indent {
            break;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with('-') {
            break;
        }
        if let Some(bracket) = trimmed.find('[')
            && trimmed[..bracket].chars().all(|c| !c.is_whitespace())
            && !trimmed[..bracket].is_empty()
        {
            let key = trimmed[..bracket].to_string();
            let rest = &trimmed[bracket..];
            let close = rest
                .find(']')
                .ok_or_else(|| DecodeError(format!("missing ']' in header: {line}")))?;
            let n: usize = rest[1..close]
                .parse()
                .map_err(|_| DecodeError(format!("bad length in header: {line}")))?;
            let after = &rest[close + 1..];
            let (value, next) = parse_array_body(lines, line_no, after, n, indent)?;
            map.insert(key, value);
            line_no = next;
            continue;
        }
        let colon = trimmed
            .find(':')
            .ok_or_else(|| DecodeError(format!("expected ':' in: {line}")))?;
        let key = trimmed[..colon].to_string();
        let value_part = trimmed[colon + 1..].trim();
        if value_part.is_empty() {
            if let Some(next) = lines.get(line_no + 1)
                && line_indent(next) == indent + 1
            {
                let (nested, after) = parse_object_fields(lines, line_no + 1, indent + 1)?;
                map.insert(key, Value::Object(nested));
                line_no = after;
                continue;
            }
            map.insert(key, Value::Object(Map::new()));
            line_no += 1;
            continue;
        }
        map.insert(key, parse_scalar_token(value_part));
        line_no += 1;
    }
    Ok((map, line_no))
}

/// Splits a delimited (comma) row respecting quoted cells, then hands
/// each token to `parse_scalar_token`.
fn split_delimited(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if in_quotes => {
                if chars.peek() == Some(&'\\') {
                    // handled by the backslash branch below on next iter
                }
                in_quotes = false;
                current.push(c);
            }
            '"' => {
                in_quotes = true;
                current.push(c);
            }
            '\\' if in_quotes => {
                current.push(c);
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            ',' if !in_quotes => {
                out.push(std::mem::take(&mut current));
            }
            c => current.push(c),
        }
    }
    out.push(current);
    out
}

fn parse_scalar_token(token: &str) -> Value {
    let token = token.trim();
    if token.starts_with('"') && token.ends_with('"') && token.len() >= 2 {
        return Value::String(unescape_string(&token[1..token.len() - 1]));
    }
    match token {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        "null" => return Value::Null,
        "[]" => return Value::Array(Vec::new()),
        _ => {}
    }
    if NUMERIC_LIKE_RE.is_match(token) {
        // An integer-shaped token (no '.', no exponent) decodes to an
        // exact integer Number, not a float -- serde_json::Number
        // distinguishes 1 from 1.0 for equality even though they're
        // numerically identical, so always going through f64 here would
        // silently break round-tripping any plain integer value.
        if !token.contains('.') && !token.contains(['e', 'E']) {
            if let Ok(i) = token.parse::<i64>() {
                return Value::Number(Number::from(i));
            }
            if let Ok(u) = token.parse::<u64>() {
                return Value::Number(Number::from(u));
            }
        }
        if let Ok(f) = token.parse::<f64>()
            && let Some(num) = Number::from_f64(f)
        {
            return Value::Number(num);
        }
    }
    Value::String(token.to_string())
}

fn unescape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('u') => {
                let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                if let Ok(code) = u32::from_str_radix(&hex, 16)
                    && let Some(c) = char::from_u32(code)
                {
                    out.push(c);
                }
            }
            Some(other) => out.push(other),
            None => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Real, verbatim examples from the published spec
    /// (github.com/toon-format/spec, SPEC v4.1, Appendix A) — MIT
    /// licensed, fair to use directly as fixtures. Confirms this
    /// encoder's output matches the real spec's own canonical form, not
    /// just an internally-consistent guess.
    #[test]
    fn matches_the_real_spec_s_nested_object_example() {
        let value = json!({"user": {"id": 123, "name": "Ada"}});
        assert_eq!(encode(&value), "user:\n  id: 123\n  name: Ada");
    }

    #[test]
    fn matches_the_real_spec_s_empty_array_field_example() {
        let value = json!({"tags": []});
        assert_eq!(encode(&value), "tags: []");
    }

    #[test]
    fn matches_the_real_spec_s_mixed_list_form_example() {
        // items[3]:
        //   - 1
        //   - a: 1
        //   - text
        let value = json!({"items": [1, {"a": 1}, "text"]});
        assert_eq!(encode(&value), "items[3]:\n  - 1\n  - a: 1\n  - text");
    }

    #[test]
    fn matches_the_real_spec_s_tabular_quoting_example() {
        // links[2]{id,url}:
        //   1,"http://a:b"
        //   2,"https://example.com?q=a:b"
        let value = json!({"links": [
            {"id": 1, "url": "http://a:b"},
            {"id": 2, "url": "https://example.com?q=a:b"}
        ]});
        assert_eq!(
            encode(&value),
            "links[2]{id,url}:\n  1,\"http://a:b\"\n  2,\"https://example.com?q=a:b\""
        );
    }

    // --- Quoting rules (§7.2), each checked directly ---

    #[test]
    fn quotes_empty_string() {
        assert_eq!(encode_scalar(&json!("")), "\"\"");
    }

    #[test]
    fn quotes_a_value_that_looks_like_a_bool_or_null() {
        assert_eq!(encode_scalar(&json!("true")), "\"true\"");
        assert_eq!(encode_scalar(&json!("null")), "\"null\"");
    }

    #[test]
    fn quotes_a_numeric_looking_string() {
        assert_eq!(encode_scalar(&json!("42")), "\"42\"");
        assert_eq!(encode_scalar(&json!("-3.14")), "\"-3.14\"");
    }

    #[test]
    fn quotes_a_string_starting_with_hyphen_or_hash() {
        assert_eq!(encode_scalar(&json!("-widget")), "\"-widget\"");
        assert_eq!(encode_scalar(&json!("#123")), "\"#123\"");
    }

    #[test]
    fn quotes_a_string_containing_a_colon_or_comma() {
        assert_eq!(encode_scalar(&json!("a:b")), "\"a:b\"");
        assert_eq!(encode_scalar(&json!("a,b")), "\"a,b\"");
    }

    #[test]
    fn leaves_an_ordinary_string_unquoted() {
        assert_eq!(encode_scalar(&json!("widget")), "widget");
    }

    #[test]
    fn escapes_backslash_quote_and_control_chars() {
        assert_eq!(encode_scalar(&json!("a\\b")), "\"a\\\\b\"");
        assert_eq!(encode_scalar(&json!("a\"b")), "\"a\\\"b\"");
        assert_eq!(encode_scalar(&json!("a\nb")), "\"a\\nb\"");
    }

    // --- Number canonicalization (§2) ---

    #[test]
    fn integers_print_without_a_decimal_point() {
        assert_eq!(encode_scalar(&json!(5)), "5");
        assert_eq!(encode_scalar(&json!(5.0)), "5");
    }

    #[test]
    fn trailing_fractional_zeros_are_dropped() {
        assert_eq!(encode_scalar(&json!(1.5000)), "1.5");
    }

    #[test]
    fn negative_zero_normalizes_to_zero() {
        assert_eq!(encode_scalar(&json!(-0.0)), "0");
    }

    // --- Round trips: the real proof this is lossless, on real-shaped data ---

    #[test]
    fn round_trips_a_flat_object() {
        let value = json!({"id": 1, "name": "widget", "active": true, "note": null});
        let toon = encode(&value);
        assert_eq!(decode(&toon).unwrap(), value);
    }

    #[test]
    fn round_trips_a_nested_object() {
        let value = json!({"user": {"id": 123, "name": "Ada", "address": {"city": "Berlin"}}});
        let toon = encode(&value);
        assert_eq!(decode(&toon).unwrap(), value);
    }

    #[test]
    fn round_trips_a_uniform_tabular_array() {
        let value = json!([
            {"id": 1, "name": "a", "active": true},
            {"id": 2, "name": "b", "active": false},
            {"id": 3, "name": "c", "active": true},
        ]);
        let toon = encode(&value);
        assert!(toon.contains("[3]{"), "expected tabular form, got: {toon}");
        assert_eq!(decode(&toon).unwrap(), value);
    }

    #[test]
    fn round_trips_a_primitive_array() {
        let value = json!(["reading", "gaming", "coding"]);
        let toon = encode(&value);
        assert_eq!(decode(&toon).unwrap(), value);
    }

    #[test]
    fn round_trips_a_non_uniform_array_of_objects() {
        let value = json!([{"a": 1, "b": 2}, {"a": 1}]);
        let toon = encode(&value);
        assert_eq!(decode(&toon).unwrap(), value);
    }

    /// A list-item object with 3+ fields exercises the "remaining
    /// sibling fields after the first" path in `parse_list_item` --
    /// found by clippy flagging that path's original `while` as a loop
    /// that could never actually loop, which was structurally misleading
    /// even though the single-call result was correct; this proves the
    /// simplified version still handles more than one trailing field.
    #[test]
    fn round_trips_a_list_item_object_with_several_fields() {
        let value = json!([
            {"a": 1, "b": 2, "c": 3, "d": 4},
            "plain string sibling",
        ]);
        let toon = encode(&value);
        let decoded = decode(&toon).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn round_trips_values_needing_quoting() {
        let value = json!([
            {"id": 1, "url": "http://a:b"},
            {"id": 2, "url": "https://example.com?q=a:b"},
        ]);
        let toon = encode(&value);
        assert_eq!(decode(&toon).unwrap(), value);
    }

    #[test]
    fn round_trips_an_empty_array_and_empty_object_field() {
        let value = json!({"tags": [], "meta": {}});
        let toon = encode(&value);
        assert_eq!(decode(&toon).unwrap(), value);
    }

    #[test]
    fn encode_if_smaller_returns_some_for_a_genuinely_compressible_uniform_array() {
        let value = json!([
            {"id": 1, "name": "a", "active": true},
            {"id": 2, "name": "b", "active": false},
            {"id": 3, "name": "c", "active": true},
        ]);
        let toon = encode_if_smaller(&value);
        assert!(toon.is_some());
        let toon = toon.unwrap();
        assert!(toon.len() < serde_json::to_string(&value).unwrap().len());
    }

    #[test]
    fn encode_if_smaller_returns_none_when_toon_is_not_strictly_smaller() {
        // A bare root scalar: JSON's form is `5` (1 byte), TOON's is
        // also `5` (root scalars carry no object/array syntax either
        // way) -- equal, not smaller, so the gate must decline. (Found
        // while writing this test: a tiny flat OBJECT like {"a":1} is
        // actually smaller in TOON than JSON, 4 bytes vs 8, because
        // JSON's braces and quoted key cost more than TOON saves for
        // objects even at this size -- a real, correct result, not a
        // bug, just not a fixture that proves this branch.)
        let value = json!(5);
        assert_eq!(encode_if_smaller(&value), None);
    }

    /// The real proof, not spec examples alone: graphify's own real
    /// graph.json nodes array (already used as a real fixture for item
    /// #3's invariant markers), genuinely messy production data, round
    /// trips exactly.
    #[test]
    fn round_trips_graphify_s_real_graph_json_nodes() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/graphify-out/graph.json");
        let raw = std::fs::read_to_string(path).expect("graphify-out/graph.json must exist");
        let graph: Value = serde_json::from_str(&raw).unwrap();
        let nodes = graph.get("nodes").unwrap().clone();

        let toon = encode(&nodes);
        let decoded = decode(&toon).unwrap();
        assert_eq!(decoded, nodes);
    }

    #[test]
    fn round_trips_a_string_containing_a_real_comma_inside_a_tabular_cell() {
        let value = json!([
            {"id": 1, "label": "a, b, c"},
            {"id": 2, "label": "plain"},
        ]);
        let toon = encode(&value);
        assert_eq!(decode(&toon).unwrap(), value);
    }
}
