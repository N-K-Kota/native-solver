//! Game-independent regret and average-strategy update policies.
//!
//! DCFR follows Brown and Sandholm (AAAI 2019), "Solving
//! Imperfect-Information Games via Discounted Regret Minimization",
//! §Regret Discounting: <https://doi.org/10.1609/aaai.v33i01.33011829>.
//! After the one-based iteration `t`, add this iteration's contributions,
//! then discount accumulated positive regret by `t^1.5 / (t^1.5 + 1)`,
//! negative regret by `1/2`, and accumulated strategy by `(t/(t+1))^2`.
//! The regret sign is tested **after** adding the instantaneous regret.
//! This end-of-iteration convention gives strategy contribution `s` weight
//! `s^2 / (T+1)^2` after `T` iterations, hence relative weights `s^2`.

/// Update policy for the heads-up trainer; independent of the game tree.
///
/// Update order is independently selected by [`UpdateSchedule`]. Existing
/// constructors retain simultaneous updates; alternating updates are opt-in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SolverAlgorithm {
    /// Vanilla regret matching and uniformly weighted average strategies.
    Cfr,
    /// Legacy regret-matching+ with uniform averaging and no delay.
    ///
    /// This preserves the existing trainer exactly. It is not the usual
    /// alternating-update, linearly weighted CFR+ implementation, nor the
    /// quadratically weighted CFR+ baseline used in the DCFR paper.
    #[default]
    CfrPlus,
    /// Discounted CFR with fixed `(alpha, beta, gamma) = (1.5, 0, 2)`.
    /// Negative accumulated regrets are discounted, not clipped to zero.
    Dcfr,
}

/// Player update order, independent of regret discounting and averaging.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UpdateSchedule {
    /// Both players use the same pre-iteration strategy profile (legacy).
    #[default]
    Simultaneous,
    /// One iteration updates OOP, then IP against the updated OOP strategy.
    /// Each player's regrets and average strategy are updated exactly once.
    /// This uses two full traversals, with the existing own-reach averaging.
    Alternating,
}

/// Coefficients are computed once per global iteration, not per infoset.
#[derive(Debug, Clone, Copy)]
pub(crate) enum IterationUpdate {
    Cfr,
    CfrPlus,
    Dcfr { positive: f64, strategy: f64 },
}

impl SolverAlgorithm {
    pub(crate) fn iteration(self, t: u64) -> IterationUpdate {
        assert!(t > 0, "CFR iteration numbers are one-based");
        match self {
            Self::Cfr => IterationUpdate::Cfr,
            Self::CfrPlus => IterationUpdate::CfrPlus,
            Self::Dcfr => {
                let t = t as f64;
                let positive_power = t.powf(1.5);
                IterationUpdate::Dcfr {
                    positive: positive_power / (positive_power + 1.0),
                    strategy: (t / (t + 1.0)).powi(2),
                }
            }
        }
    }
}

impl IterationUpdate {
    /// Apply exactly one infoset/action update after its regret contributions
    /// have been reduced. The legacy modes preserve the original operation
    /// order, including clipping only after the whole delta is added.
    pub(crate) fn regret(self, previous: f64, delta: f64) -> f64 {
        let accumulated = previous + delta;
        match self {
            Self::Cfr => accumulated,
            Self::CfrPlus => accumulated.max(0.0),
            Self::Dcfr { positive, .. } => {
                accumulated * if accumulated > 0.0 { positive } else { 0.5 }
            }
        }
    }

    /// Called once after **all** this iteration's reach-weighted strategy
    /// contributions to a bucket/action have been added. Discounting per
    /// combo would incorrectly discount large buckets more often.
    pub(crate) fn strategy_sum(self, accumulated: f64) -> f64 {
        match self {
            Self::Dcfr { strategy, .. } => accumulated * strategy,
            Self::Cfr | Self::CfrPlus => accumulated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dcfr_first_iteration_adds_before_discounting() {
        let update = SolverAlgorithm::Dcfr.iteration(1);
        assert_eq!(update.regret(0.0, 8.0), 4.0);
        assert_eq!(update.regret(0.0, -8.0), -4.0);
        assert_eq!(update.strategy_sum(8.0), 2.0);
    }

    #[test]
    fn dcfr_uses_post_addition_sign_and_published_coefficients() {
        let update = SolverAlgorithm::Dcfr.iteration(4);
        // 4^1.5 = 8, beta=0 => 1/2, (4/5)^2 = 0.64.
        assert_eq!(update.regret(-6.0, 15.0), 8.0);
        assert_eq!(update.regret(6.0, -14.0), -4.0);
        assert_eq!(update.regret(3.0, -3.0), 0.0);
        assert!((update.strategy_sum(25.0) - 16.0).abs() < 1e-14);
    }

    #[test]
    fn dcfr_multiple_updates_match_independent_scalar_recurrence() {
        let mut regret = 0.0;
        let mut strategy_sum = 0.0;
        let mut expected_regret = 0.0;
        let mut weighted_strategy_sum = 0.0;
        for (i, (delta, contribution)) in [(8.0, 0.25), (-10.0, 0.0), (9.0, 0.75), (-2.0, 1.0)]
            .into_iter()
            .enumerate()
        {
            let t = (i + 1) as f64;
            let accumulated = expected_regret + delta;
            expected_regret = accumulated
                * if accumulated > 0.0 {
                    (t * t.sqrt()) / (t * t.sqrt() + 1.0)
                } else {
                    0.5
                };
            weighted_strategy_sum += t * t * contribution;
            let update = SolverAlgorithm::Dcfr.iteration(i as u64 + 1);
            regret = update.regret(regret, delta);
            strategy_sum = update.strategy_sum(strategy_sum + contribution);
            assert!((regret - expected_regret).abs() < 1e-14);
            assert!((strategy_sum - weighted_strategy_sum / (t + 1.0).powi(2)).abs() < 1e-14);
        }
    }

    #[test]
    fn legacy_updates_keep_undiscounted_uniform_sums() {
        assert_eq!(SolverAlgorithm::default(), SolverAlgorithm::CfrPlus);
        for t in [1, 2, 100, u64::MAX] {
            let vanilla = SolverAlgorithm::Cfr.iteration(t);
            let plus = SolverAlgorithm::CfrPlus.iteration(t);
            assert_eq!(vanilla.regret(2.0, -5.0), -3.0);
            assert_eq!(plus.regret(2.0, -5.0), 0.0);
            assert_eq!(vanilla.strategy_sum(7.0), 7.0);
            assert_eq!(plus.strategy_sum(7.0), 7.0);
        }
    }

    #[test]
    fn dcfr_late_iteration_coefficients_remain_finite() {
        let update = SolverAlgorithm::Dcfr.iteration(u64::MAX);
        assert!(update.regret(0.0, 1.0).is_finite());
        assert_eq!(update.regret(0.0, -1.0), -0.5);
        assert!(update.strategy_sum(1.0).is_finite());
    }

    #[test]
    #[should_panic(expected = "CFR iteration numbers are one-based")]
    fn zero_iteration_is_rejected() {
        SolverAlgorithm::Dcfr.iteration(0);
    }
}
