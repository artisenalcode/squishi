//! Self-diagnostics — checks this tool's own real failure surface (which
//! binary is actually running, whether the magika model classifies real
//! content correctly, where the semantic-dedup model cache resolves to
//! and whether it's already warm, and a best-effort proxy for whether
//! the PostToolUse hook has fired recently) and reports pass/warn/fail
//! per check.
//!
//! Modeled on `agent-browser doctor` (audited 2026-08-07,
//! docs/ideation/audit-repo-techniques/2026-08-07-enrichment-techniques.md)
//! and total-recall's own `doctor` (this session) — same `Check`/`Status`
//! shape for consistency of *pattern*, not a shared implementation (two
//! real implementations as of this plan; this project's own "no shared
//! abstraction before a third real need" convention).
//!
//! No `--fix` here (see `docs/plan-2026-08-07-doctor-commands.md`):
//! squishi has no repairable persistent state — no locks, no bank dirs,
//! nothing that gets stuck.

use crate::semantic_dedup::SemanticDedup;
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Warn,
    Fail,
    /// `--quick` skipped an expensive check — distinct from Pass so a
    /// caller can tell "verified working" from "not actually checked."
    Skipped,
}

impl Status {
    fn as_str(&self) -> &'static str {
        match self {
            Status::Pass => "pass",
            Status::Warn => "warn",
            Status::Fail => "fail",
            Status::Skipped => "skipped",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Check {
    pub name: String,
    pub status: Status,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct DoctorReport {
    pub checks: Vec<Check>,
}

impl DoctorReport {
    pub fn has_failures(&self) -> bool {
        self.checks.iter().any(|c| c.status == Status::Fail)
    }

    /// Reuses `serde_json` (already a squishi dependency, unlike
    /// total-recall) rather than hand-rolling escaping — `build_output`'s
    /// own reasoning for a plain `Map<String, Value>` over a derive macro
    /// applies here too.
    pub fn to_json(&self) -> Value {
        let checks: Vec<Value> = self
            .checks
            .iter()
            .map(|c| {
                let mut m = Map::new();
                m.insert("name".to_string(), Value::from(c.name.clone()));
                m.insert("status".to_string(), Value::from(c.status.as_str()));
                m.insert("message".to_string(), Value::from(c.message.clone()));
                Value::Object(m)
            })
            .collect();
        let mut out = Map::new();
        out.insert("checks".to_string(), Value::from(checks));
        Value::Object(out)
    }
}

/// Run every check. `quick` skips the two model-load checks (magika,
/// semantic-dedup) — mirrors total-recall's own `--quick`. Only
/// semantic-dedup is genuinely expensive since the 2026-08-18 candle-onnx
/// migration (real network/disk model fetch); magika's check is cheap
/// now but stays gated for output stability, see `check_magika_loads`.
pub fn run(quick: bool) -> DoctorReport {
    let checks = vec![
        check_binary_identity(),
        check_magika_loads(quick),
        check_model_cache_location(),
        check_semantic_dedup_loads(quick),
        check_hook_proxy_signal(),
    ];

    DoctorReport { checks }
}

fn check_binary_identity() -> Check {
    let name = "binary_identity".to_string();
    match std::env::current_exe() {
        Ok(path) => Check {
            name,
            status: Status::Pass,
            message: format!(
                "running {} (squishi v{})",
                path.display(),
                env!("CARGO_PKG_VERSION")
            ),
        },
        Err(e) => Check {
            name,
            status: Status::Warn,
            message: format!("could not resolve current_exe: {e}"),
        },
    }
}

/// 2026-08-18: `magika::Session::new()` (ort) replaced by a real
/// classification through `content_detect::detect()` (candle-onnx) —
/// see content_detect.rs's module doc comment for why. This check is no
/// longer expensive to run (the embedded model decodes once, lazily, in
/// well under a millisecond — no ~111ms `ort::Session` build) but stays
/// gated behind `--quick` for output stability with the rest of this
/// function's contract, and because a real classification is still a
/// stronger signal than "the model bytes decoded."
fn check_magika_loads(quick: bool) -> Check {
    let name = "magika_loads".to_string();
    if quick {
        return Check {
            name,
            status: Status::Skipped,
            message: "skipped (--quick)".to_string(),
        };
    }
    let sample = "fn main() {\n    let x = 5;\n    println!(\"{}\", x);\n}\n";
    match crate::content_detect::detect(sample) {
        crate::content_detect::ContentKind::Other(label) if label == "rust" => Check {
            name,
            status: Status::Pass,
            message: "candle-onnx magika model loads and classifies real Rust source correctly"
                .to_string(),
        },
        other => Check {
            name,
            status: Status::Fail,
            message: format!(
                "magika model loaded but misclassified a known-rust sample as {other:?}"
            ),
        },
    }
}

/// Reports where `semantic_dedup::SemanticDedup::load`'s `hf_hub::Api::new()`
/// actually resolves its cache directory to. Real finding from reading the
/// `hf-hub` source directly (not assumed): `Api::new()` uses
/// `Cache::default()` (always `~/.cache/huggingface/hub`), NOT
/// `Cache::from_env()` — `HF_HOME` is silently ignored by the call
/// squishi actually makes. Surfacing the real resolved path, and that
/// `HF_HOME` doesn't affect it, is exactly the kind of previously-invisible
/// information this check exists to give.
fn check_model_cache_location() -> Check {
    let name = "model_cache_location".to_string();
    let cache_dir = match dirs::home_dir() {
        Some(mut home) => {
            home.push(".cache");
            home.push("huggingface");
            home.push("hub");
            home
        }
        None => {
            return Check {
                name,
                status: Status::Warn,
                message: "could not resolve home directory to locate hf-hub cache".to_string(),
            };
        }
    };
    let hf_home_note = if std::env::var("HF_HOME").is_ok() {
        " (note: HF_HOME is set but squishi's hf_hub::Api::new() call does not honor it — \
          always resolves to ~/.cache/huggingface/hub)"
    } else {
        ""
    };
    if cache_dir.exists() {
        let model_present = cache_dir
            .join("models--sentence-transformers--all-MiniLM-L6-v2")
            .exists();
        Check {
            name,
            status: Status::Pass,
            message: format!(
                "{}{hf_home_note} — model files {}",
                cache_dir.display(),
                if model_present {
                    "already cached"
                } else {
                    "not yet cached (first semantic-dedup call will download)"
                }
            ),
        }
    } else {
        Check {
            name,
            status: Status::Warn,
            message: format!(
                "{}{hf_home_note} does not exist yet (created on first download)",
                cache_dir.display()
            ),
        }
    }
}

fn check_semantic_dedup_loads(quick: bool) -> Check {
    let name = "semantic_dedup_loads".to_string();
    if quick {
        return Check {
            name,
            status: Status::Skipped,
            message: "skipped (--quick)".to_string(),
        };
    }
    match SemanticDedup::load() {
        Ok(_) => Check {
            name,
            status: Status::Pass,
            message: "semantic dedup model loads".to_string(),
        },
        Err(e) => Check {
            name,
            status: Status::Fail,
            message: format!("semantic dedup failed to load: {e}"),
        },
    }
}

/// Best-effort proxy signal only — NOT a registration check. This
/// project's own filed bug (anthropics/claude-code#84439) established
/// there is no reliable programmatic way to confirm a plugin-registered
/// PostToolUse hook is actually live. This reports hook-file presence and
/// a recency proxy (`last-input.json`'s mtime) — explicitly labeled, never
/// upgraded to sound like a guarantee in either direction.
fn check_hook_proxy_signal() -> Check {
    let name = "hook_proxy_signal".to_string();
    let Some(home) = dirs::home_dir() else {
        return Check {
            name,
            status: Status::Warn,
            message: "could not resolve home directory to locate hook files".to_string(),
        };
    };
    let skill_dir = home.join(".claude/skills/squishi");
    let hooks_json = skill_dir.join("hooks/hooks.json");
    let handler = skill_dir.join("hooks-handlers/posttooluse.sh");
    let last_input = skill_dir.join("hooks-handlers/last-input.json");

    if !hooks_json.exists() || !handler.exists() {
        return Check {
            name,
            status: Status::Warn,
            message: format!(
                "hook files not found at {} — this is a proxy signal, not a \
                 registration check (see anthropics/claude-code#84439)",
                skill_dir.display()
            ),
        };
    }

    match last_input.metadata().and_then(|m| m.modified()) {
        Ok(modified) => {
            let age = std::time::SystemTime::now()
                .duration_since(modified)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            Check {
                name,
                status: Status::Pass,
                message: format!(
                    "hook files present; last observed firing {} ago \
                     (proxy signal only, not a registration guarantee — \
                     see anthropics/claude-code#84439)",
                    format_age(age)
                ),
            }
        }
        Err(_) => Check {
            name,
            status: Status::Warn,
            message: format!(
                "hook files present at {} but never observed firing yet \
                 (proxy signal only, not a registration guarantee)",
                skill_dir.display()
            ),
        },
    }
}

fn format_age(secs: u64) -> String {
    if secs < 60 {
        format!("{secs}s")
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86400)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_identity_check_passes_and_reports_a_real_path() {
        let check = check_binary_identity();
        assert_eq!(check.status, Status::Pass);
        assert!(check.message.contains("squishi v"));
    }

    #[test]
    fn magika_check_is_skipped_in_quick_mode() {
        let check = check_magika_loads(true);
        assert_eq!(check.status, Status::Skipped);
    }

    #[test]
    fn semantic_dedup_check_is_skipped_in_quick_mode() {
        let check = check_semantic_dedup_loads(true);
        assert_eq!(check.status, Status::Skipped);
    }

    #[test]
    fn model_cache_location_check_reports_a_real_path_under_home() {
        let check = check_model_cache_location();
        assert!(matches!(check.status, Status::Pass | Status::Warn));
        let home = dirs::home_dir().unwrap();
        assert!(check.message.starts_with(home.to_str().unwrap()));
    }

    #[test]
    fn report_has_failures_reflects_real_check_statuses() {
        let report = DoctorReport {
            checks: vec![
                Check {
                    name: "a".to_string(),
                    status: Status::Pass,
                    message: "ok".to_string(),
                },
                Check {
                    name: "b".to_string(),
                    status: Status::Warn,
                    message: "meh".to_string(),
                },
            ],
        };
        assert!(!report.has_failures());

        let mut with_failure = report.clone();
        with_failure.checks.push(Check {
            name: "c".to_string(),
            status: Status::Fail,
            message: "broken".to_string(),
        });
        assert!(with_failure.has_failures());
    }

    #[test]
    fn to_json_produces_the_expected_shape() {
        let report = DoctorReport {
            checks: vec![Check {
                name: "x".to_string(),
                status: Status::Fail,
                message: "quote \" and backslash \\".to_string(),
            }],
        };
        let json = report.to_json();
        let checks = json["checks"].as_array().unwrap();
        assert_eq!(checks.len(), 1);
        assert_eq!(checks[0]["name"], "x");
        assert_eq!(checks[0]["status"], "fail");
        assert_eq!(checks[0]["message"], "quote \" and backslash \\");
    }

    #[test]
    fn run_in_quick_mode_never_loads_either_model() {
        // Real end-to-end: run(true) must complete fast (no model I/O) and
        // report exactly two Skipped checks among the five.
        let report = run(true);
        assert_eq!(report.checks.len(), 5);
        let skipped = report
            .checks
            .iter()
            .filter(|c| c.status == Status::Skipped)
            .count();
        assert_eq!(skipped, 2);
    }
}
