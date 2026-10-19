//! native-solver: a heads-up postflop No-Limit Hold'em solver.
//!
//! - [`card`]: cards as `u8` in `0..52` and card sets as `u64` bitmasks.
//! - [`eval`]: five- and seven-card hand evaluation.
//! - [`range`]: weighted preflop ranges and their expansion into [`Combo`]s.
//! - [`solver`]: the betting tree ([`GameConfig`], [`build_tree`]) and the
//!   combo-vectorized CFR [`Trainer`], with exploitability and node EVs.
//! - [`games`]: small toy games (Kuhn poker) that validate the CFR code.
//!
//! Only heads-up postflop play is modeled: no preflop, no side pots, no
//! multiway pots. All values are in input chips unless a doc says otherwise.

#![warn(missing_docs)]

// Compile and run the README example as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod card;
pub mod eval;
pub mod games;
pub mod range;
pub mod solver;

pub use card::{
    card_bit, card_from_str, card_to_string, rank, set_contains, set_from_cards, suit, Card,
    CardParseError, CardSet,
};
pub use eval::{category_of, evaluate_5, evaluate_7, HandCategory};
pub use range::{all_canonical_hands, Combo, Range, RangeError};
pub use solver::{
    build_tree, build_tree_bounded, Action, BoardError, CardAbstraction, Chips, Exploitability,
    GameConfig, Node, NodeEvaluation, Player, Policy, PolicyError, SolverAlgorithm, SolverGame,
    Terminal, Trainer, TrainerError, TrainerOptions, Tree, TreeError, UpdateSchedule, CHIP_SCALE,
};
