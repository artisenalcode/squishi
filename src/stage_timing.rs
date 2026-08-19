//! Minimal per-stage duration accumulator, used only by the timing examples to split a model call's wall time into sub-stages (tokenize / build tensors / forward pass / postprocess) instead of one lumped number. Production callers never read this.

use std::time::Duration;

#[derive(Debug, Default, Clone, Copy)]
pub struct StageTimings {
    pub tokenize: Duration,
    pub build_tensors: Duration,
    pub forward: Duration,
    pub postprocess: Duration,
}

impl StageTimings {
    /// Adds another batch's stage durations into this total. Callers reset to `default()` once per top-level call, then accumulate each batch so a chunked input reports one summed total per stage.
    pub fn accumulate(&mut self, other: &StageTimings) {
        self.tokenize += other.tokenize;
        self.build_tensors += other.build_tensors;
        self.forward += other.forward;
        self.postprocess += other.postprocess;
    }

    pub fn total(&self) -> Duration {
        self.tokenize + self.build_tensors + self.forward + self.postprocess
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_sums_each_stage_independently() {
        let mut totals = StageTimings::default();
        totals.accumulate(&StageTimings {
            tokenize: Duration::from_millis(1),
            build_tensors: Duration::from_millis(2),
            forward: Duration::from_millis(3),
            postprocess: Duration::from_millis(4),
        });
        totals.accumulate(&StageTimings {
            tokenize: Duration::from_millis(10),
            build_tensors: Duration::from_millis(20),
            forward: Duration::from_millis(30),
            postprocess: Duration::from_millis(40),
        });
        assert_eq!(totals.tokenize, Duration::from_millis(11));
        assert_eq!(totals.build_tensors, Duration::from_millis(22));
        assert_eq!(totals.forward, Duration::from_millis(33));
        assert_eq!(totals.postprocess, Duration::from_millis(44));
    }

    #[test]
    fn total_sums_all_four_stages() {
        let totals = StageTimings {
            tokenize: Duration::from_millis(1),
            build_tensors: Duration::from_millis(2),
            forward: Duration::from_millis(3),
            postprocess: Duration::from_millis(4),
        };
        assert_eq!(totals.total(), Duration::from_millis(10));
    }

    #[test]
    fn default_is_all_zero() {
        let totals = StageTimings::default();
        assert_eq!(totals.total(), Duration::ZERO);
    }
}
