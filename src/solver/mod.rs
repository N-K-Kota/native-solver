//! Heads-up postflop NLHE: a multi-street betting tree and combo-vectorized
//! CFR trainer. No side pots or multiway games.

// Implementation modules stay private; their public types are re-exported
// below. `quality` is public: its independent reference evaluator is a
// feature in its own right.
mod cfr;
mod chips;
mod discount;
pub mod quality;
mod tree;

pub use cfr::{
    CardAbstraction, Evaluation, Exploitability, NodeEvaluation, Policy, PolicyError, SolverGame,
    Trainer, TrainerError, TrainerOptions,
};
pub use chips::{Chips, CHIP_SCALE};
pub use discount::{SolverAlgorithm, UpdateSchedule};
pub use tree::{
    build_tree, build_tree_bounded, Action, BoardError, GameConfig, Node, Player, Terminal, Tree,
    TreeError,
};

mod terminal;
