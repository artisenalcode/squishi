//! JSON array compression: dedupe exact-duplicate elements, then cap to
//! first + last N with a dropped-count marker. Scoped to arrays — a
//! single object has nothing repeatable to compress.

use serde_json::Value;

/// Tunable knobs for `compress_json_array` — `--level` varies `keep_edge`.
pub struct JsonCompressConfig {
    /// First N and last N elements kept when capping.
    pub keep_edge: usize,
}

impl Default for JsonCompressConfig {
    fn default() -> Self {
        Self { keep_edge: 5 }
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

    // Dedupe exact-duplicate elements, preserving first-seen order.
    let mut seen: Vec<Value> = Vec::new();
    for el in &elements {
        if !seen.contains(el) {
            seen.push(el.clone());
        }
    }

    let keep_edge = config.keep_edge;
    let final_elements: Vec<Value> = if seen.len() > keep_edge * 2 {
        let mut kept: Vec<Value> = seen[..keep_edge].to_vec();
        let dropped = seen.len() - keep_edge * 2;
        kept.push(Value::String(format!(
            "...{dropped} more elements omitted..."
        )));
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
        let config = JsonCompressConfig { keep_edge: 2 };
        let result = compress_json_array(&content, &config).unwrap();
        assert_eq!(result.compressed_elements, 2 * 2 + 1);
        assert!(result.content.contains(r#""id":0"#));
        assert!(result.content.contains(r#""id":1"#));
        assert!(!result.content.contains(r#""id":2"#));
    }
}
