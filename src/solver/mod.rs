//! Heads-up postflop NLHE: a multi-street betting tree and combo-vectorized
//! CFR trainer. No side pots or multiway games.

// Implementation modules stay private; their public types are re-exported
// below. `quality` is public: its independent reference evaluator is a
// feature in its own right.
mod cfr;
mod chips;
mod discount;
mod preflop;
mod preflop_equity;
pub mod quality;
mod tree;

pub use cfr::{
    CardAbstraction, Evaluation, Exploitability, NodeEvaluation, Policy, PolicyError, SolverGame,
    Trainer, TrainerError, TrainerOptions,
};
pub use chips::{Chips, CHIP_SCALE};
pub use discount::{SolverAlgorithm, UpdateSchedule};
pub use preflop::{build_preflop_tree, build_preflop_tree_bounded, PreflopConfig};
#[doc(hidden)]
pub use preflop_equity::generate_preflop_equity_table;
pub use tree::{
    build_tree, build_tree_bounded, Action, BoardError, GameConfig, Node, Player, Terminal, Tree,
    TreeError,
};

mod terminal;
