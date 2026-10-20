//! Find near-equilibrium strategies for heads-up postflop No-Limit Hold'em
//! spots, and measure how exploitable they are.
//!
//! Give it a board, both players' ranges, the pot, the stacks, and the bet
//! sizes to consider; it trains a strategy with counterfactual regret
//! minimization (CFR, CFR+, or DCFR) and reports per-hand strategies, EVs,
//! and exploitability: how much a perfect opponent could still win against
//! the result. Chip amounts are exact integers, invalid inputs are reported
//! as errors, and an independent reference evaluator can check results on
//! small trees.
//!
//! - [`card`]: [`Card`]s and [`CardSet`]s.
//! - [`eval`]: five- and seven-card hand evaluation.
//! - [`range`]: weighted preflop ranges and their expansion into [`Combo`]s.
//! - [`solver`]: the betting tree ([`GameConfig`], [`build_tree`]) and the
//!   combo-vectorized CFR [`Trainer`], with exploitability and node EVs.
//!
//! Only heads-up postflop play is modeled: no preflop, no side pots, no
//! multiway pots. All values are in input chips unless a doc says otherwise.
//!
//! # Example
//!
//! Solve a river spot and check how close the result is to equilibrium:
//!
//! ```
//! use native_solver::{build_tree, Card, CardSet, Chips, GameConfig, Range, Trainer, TrainerOptions};
//!
//! let board = ["Ks", "Th", "7s", "4d", "2s"]
//!     .iter()
//!     .map(|card| card.parse())
//!     .collect::<Result<Vec<Card>, _>>()?;
//! let config = GameConfig::new(board.clone(), Chips::whole(100), Chips::whole(100))
//!     .with_bet_sizes([0.5, 1.0]);
//!
//! // Remove combos that share a card with the board.
//! let dead = CardSet::from_cards(&board);
//! let oop = "AA,KK,KQs,QJs,JTs:0.5".parse::<Range>()?.combos(dead);
//! let ip = "AKs,KJs,TT,99,A5s:0.5".parse::<Range>()?.combos(dead);
//!
//! let mut trainer = Trainer::new(build_tree(&config)?, [oop, ip], TrainerOptions::default())?;
//! trainer.train(500);
//!
//! // How much a best response could still gain, in chips.
//! assert!(trainer.exploitability().total < 1.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! See [the conventions](https://github.com/N-K-Kota/native-solver/blob/v0.1.0/docs/conventions.md) for units and value
//! normalization, and [the algorithms](https://github.com/N-K-Kota/native-solver/blob/v0.1.0/docs/algorithms.md) for the update
//! rules.

#![warn(missing_docs)]

// Compile and run the README example as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod card;
pub mod eval;
// Kuhn poker: a separate toy CFR with a known closed-form equilibrium, kept
// only to cross-check the update rules in tests.
#[cfg(test)]
mod games;
pub mod range;
pub mod solver;

pub use card::{Card, CardParseError, CardSet};
pub use eval::{evaluate_5, evaluate_7, HandCategory, HandStrength};
pub use range::{all_canonical_hands, Combo, Range, RangeError};
pub use solver::{
    build_tree, build_tree_bounded, Action, BoardError, CardAbstraction, Chips, Evaluation,
    Exploitability, GameConfig, Node, NodeEvaluation, Player, Policy, PolicyError, SolverAlgorithm,
    SolverGame, Terminal, Trainer, TrainerError, TrainerOptions, Tree, TreeError, UpdateSchedule,
    CHIP_SCALE,
};
