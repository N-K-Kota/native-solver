//! Kuhn poker: a 3-card, 2-player poker game with a single betting round.
//! It has a published closed-form equilibrium, which makes it the standard
//! toy problem for checking a CFR implementation is actually correct
//! before trusting it on a real (and much harder to verify) game tree.
//!
//! Cards are `0 = Jack`, `1 = Queen`, `2 = King`. Each player antes 1 chip;
//! the single allowed bet size is 1 chip. A game history is a string of
//! `'p'` (pass: check if no bet is pending, fold if one is) and `'b'`
//! (bet: open a bet if none is pending, call if one is).
//!
//! This module is intentionally decoupled from `crate::card`/`crate::eval`
//! — Kuhn poker's 3-card deck and depth-3 tree don't need any of that, and
//! its only job is to exercise the CFR recursion, regret matching, and
//! averaging in a place where the right answer is known in closed form.

use std::collections::HashMap;

const NUM_ACTIONS: usize = 2;
const PASS: usize = 0;
// Only indexed directly by tests (`strategy[BET]`); production code reaches
// the bet branch via `else` in the two-action loop below.
#[allow(dead_code)]
const BET: usize = 1;
const CARDS: [u8; 3] = [0, 1, 2];

#[derive(Debug, Clone)]
struct Node {
    regret_sum: [f64; NUM_ACTIONS],
    strategy_sum: [f64; NUM_ACTIONS],
}

impl Node {
    fn new() -> Self {
        Self {
            regret_sum: [0.0; NUM_ACTIONS],
            strategy_sum: [0.0; NUM_ACTIONS],
        }
    }

    /// Regret matching: positive regrets normalized to a distribution, or
    /// uniform if there is no positive regret yet.
    fn current_strategy(&self) -> [f64; NUM_ACTIONS] {
        let positive: [f64; NUM_ACTIONS] = std::array::from_fn(|a| self.regret_sum[a].max(0.0));
        let total: f64 = positive.iter().sum();
        if total > 0.0 {
            std::array::from_fn(|a| positive[a] / total)
        } else {
            [1.0 / NUM_ACTIONS as f64; NUM_ACTIONS]
        }
    }

    fn average_strategy(&self) -> [f64; NUM_ACTIONS] {
        let total: f64 = self.strategy_sum.iter().sum();
        if total > 0.0 {
            std::array::from_fn(|a| self.strategy_sum[a] / total)
        } else {
            [1.0 / NUM_ACTIONS as f64; NUM_ACTIONS]
        }
    }
}

/// Vanilla CFR (or CFR+, when `plus` is set) trainer for Kuhn poker.
pub struct KuhnTrainer {
    plus: bool,
    nodes: HashMap<String, Node>,
}

impl KuhnTrainer {
    /// `plus = true` applies the CFR+ regret floor (clip each action's
    /// cumulative regret to >= 0 after every update).
    pub fn new(plus: bool) -> Self {
        Self {
            plus,
            nodes: HashMap::new(),
        }
    }

    /// Run `iterations` training passes. Each pass enumerates all 6 card
    /// deals exactly (instead of sampling one at random), so training is
    /// deterministic and needs no RNG. Returns the average game value for
    /// player 0; the known equilibrium value is `-1/18`.
    pub fn train(&mut self, iterations: u32) -> f64 {
        let deals = all_deals();
        let mut total = 0.0;
        for _ in 0..iterations {
            for &cards in &deals {
                total += self.cfr(cards, "", 1.0, 1.0);
            }
        }
        total / (iterations as f64 * deals.len() as f64)
    }

    /// The trained average strategy `[P(pass), P(bet)]` for an information
    /// set, keyed as `"{card}{history}"` (e.g. `"0"` = Jack to act first,
    /// `"2pb"` = King facing a bet after checking).
    pub fn average_strategy(&self, info_set: &str) -> Option<[f64; NUM_ACTIONS]> {
        self.nodes.get(info_set).map(Node::average_strategy)
    }

    fn cfr(&mut self, cards: [u8; 2], history: &str, p0: f64, p1: f64) -> f64 {
        let plays = history.len();
        let player = plays % 2;
        let opponent = 1 - player;

        if plays > 1 {
            let terminal_pass = history.as_bytes()[plays - 1] == b'p';
            let double_bet = &history[plays - 2..plays] == "bb";
            let player_card_higher = cards[player] > cards[opponent];
            if terminal_pass {
                return if history == "pp" {
                    if player_card_higher {
                        1.0
                    } else {
                        -1.0
                    }
                } else {
                    // The opponent's pass just before this was a fold facing a bet.
                    1.0
                };
            } else if double_bet {
                return if player_card_higher { 2.0 } else { -2.0 };
            }
        }

        let info_set = format!("{}{}", cards[player], history);
        let strategy = self
            .nodes
            .entry(info_set.clone())
            .or_insert_with(Node::new)
            .current_strategy();

        let mut util = [0.0; NUM_ACTIONS];
        let mut node_util = 0.0;
        for a in 0..NUM_ACTIONS {
            let mut next_history = history.to_string();
            next_history.push(if a == PASS { 'p' } else { 'b' });
            util[a] = if player == 0 {
                -self.cfr(cards, &next_history, p0 * strategy[a], p1)
            } else {
                -self.cfr(cards, &next_history, p0, p1 * strategy[a])
            };
            node_util += strategy[a] * util[a];
        }

        let opponent_reach = if player == 0 { p1 } else { p0 };
        let own_reach = if player == 0 { p0 } else { p1 };
        let node = self.nodes.get_mut(&info_set).expect("just inserted above");
        for a in 0..NUM_ACTIONS {
            node.regret_sum[a] += opponent_reach * (util[a] - node_util);
            if self.plus {
                node.regret_sum[a] = node.regret_sum[a].max(0.0);
            }
            node.strategy_sum[a] += own_reach * strategy[a];
        }

        node_util
    }
}

fn all_deals() -> Vec<[u8; 2]> {
    let mut deals = Vec::with_capacity(6);
    for &c0 in &CARDS {
        for &c1 in &CARDS {
            if c0 != c1 {
                deals.push([c0, c1]);
            }
        }
    }
    deals
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 0.02;

    fn bet_freq(trainer: &KuhnTrainer, info_set: &str) -> f64 {
        trainer
            .average_strategy(info_set)
            .unwrap_or_else(|| panic!("info set {info_set:?} was never visited"))[BET]
    }

    #[test]
    fn converges_to_known_game_value() {
        let mut trainer = KuhnTrainer::new(true);
        let value = trainer.train(100_000);
        // Published Kuhn poker equilibrium value for player 0 is -1/18.
        assert!(
            (value - (-1.0 / 18.0)).abs() < TOLERANCE,
            "expected value close to -1/18, got {value}"
        );
    }

    #[test]
    fn converges_to_known_player0_frequencies() {
        let mut trainer = KuhnTrainer::new(true);
        trainer.train(100_000);

        // Player 0's opening action with a Queen is always check (fixed
        // regardless of the equilibrium's free bluffing parameter).
        assert!(bet_freq(&trainer, "1") < TOLERANCE);

        // Facing a bet after checking (history "pb"): fold Jack always,
        // call King always. These are pinned exactly by the equilibrium.
        assert!((bet_freq(&trainer, "0pb") - 0.0).abs() < TOLERANCE);
        assert!((bet_freq(&trainer, "2pb") - 1.0).abs() < TOLERANCE);

        // Everything else is governed by the equilibrium's one free
        // parameter, alpha = bet(J) (player 0's opening bluff frequency
        // with a Jack), which must lie in [0, 1/3]:
        //   bet(K)      == 3 * alpha
        //   call(Q,"pb") == alpha + 1/3
        let alpha = bet_freq(&trainer, "0");
        assert!(
            (0.0..=1.0 / 3.0 + TOLERANCE).contains(&alpha),
            "alpha={alpha} out of range"
        );
        let bet_k = bet_freq(&trainer, "2");
        assert!(
            (bet_k - 3.0 * alpha).abs() < TOLERANCE,
            "expected bet(K) == 3*alpha, got alpha={alpha}, bet(K)={bet_k}"
        );
        let call_q_pb = bet_freq(&trainer, "1pb");
        assert!(
            (call_q_pb - (alpha + 1.0 / 3.0)).abs() < TOLERANCE,
            "expected call(Q,\"pb\") == alpha + 1/3, got alpha={alpha}, call={call_q_pb}"
        );
    }

    #[test]
    fn converges_to_known_player1_frequencies() {
        let mut trainer = KuhnTrainer::new(true);
        trainer.train(100_000);

        // Player 1's strategy has no free parameter — every frequency is
        // pinned by the published equilibrium.
        assert!((bet_freq(&trainer, "0p") - 1.0 / 3.0).abs() < TOLERANCE); // J, facing check: bet 1/3
        assert!((bet_freq(&trainer, "1p") - 0.0).abs() < TOLERANCE); // Q, facing check: never bet
        assert!((bet_freq(&trainer, "2p") - 1.0).abs() < TOLERANCE); // K, facing check: always bet

        assert!((bet_freq(&trainer, "0b") - 0.0).abs() < TOLERANCE); // J, facing bet: always fold
        assert!((bet_freq(&trainer, "1b") - 1.0 / 3.0).abs() < TOLERANCE); // Q, facing bet: call 1/3
        assert!((bet_freq(&trainer, "2b") - 1.0).abs() < TOLERANCE); // K, facing bet: always call
    }

    #[test]
    fn vanilla_cfr_also_converges() {
        // Same checks with plus = false, to confirm the vanilla-CFR code
        // path (no regret flooring) converges too, just possibly slower.
        let mut trainer = KuhnTrainer::new(false);
        let value = trainer.train(200_000);
        assert!(
            (value - (-1.0 / 18.0)).abs() < TOLERANCE,
            "expected value close to -1/18, got {value}"
        );
    }
}
