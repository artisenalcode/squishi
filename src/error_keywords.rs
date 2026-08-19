//! Canonical error keyword set for item preservation.
//!
//! Ported from headroom's `smart_crusher/error_keywords.rs`. Intentionally
//! broad — better to over-preserve than to drop a real error item.

/// 12 error/failure keywords. Lowercase by construction; callers must
/// lowercase the haystack before substring-matching.
pub const ERROR_KEYWORDS: &[&str] = &[
    "error",
    "exception",
    "failed",
    "failure",
    "critical",
    "fatal",
    "crash",
    "panic",
    "abort",
    "timeout",
    "denied",
    "rejected",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_headroom_count() {
        assert_eq!(ERROR_KEYWORDS.len(), 12);
    }

    #[test]
    fn all_lowercase_invariant() {
        for &kw in ERROR_KEYWORDS {
            assert_eq!(
                kw,
                kw.to_lowercase(),
                "ERROR_KEYWORDS must all be lowercase"
            );
        }
    }

    #[test]
    fn pinned_membership() {
        // Pin the exact set so accidental edits surface in CI rather
        // than silently changing item-preservation behavior.
        let expected = [
            "error",
            "exception",
            "failed",
            "failure",
            "critical",
            "fatal",
            "crash",
            "panic",
            "abort",
            "timeout",
            "denied",
            "rejected",
        ];
        let actual: std::collections::BTreeSet<&str> = ERROR_KEYWORDS.iter().copied().collect();
        let expected: std::collections::BTreeSet<&str> = expected.iter().copied().collect();
        assert_eq!(actual, expected);
    }
}
