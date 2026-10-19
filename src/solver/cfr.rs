//! Combo-vectorized CFR: instead of recursing per fixed hand pair (as the
//! Kuhn poker trainer does), each information set holds one regret/
//! strategy row **per combo** in the acting player's range
//! (`[action][combo]`), and terminal values are computed as a weighted
//! sum over the opponent's live (non-card-conflicting) combos. This is
//! the standard technique real postflop solvers use to solve a whole
//! range at once instead of one hand match-up at a time.
//!
//! `Chance` nodes (dealt community cards) are handled the same way: each
//! candidate card is equally likely conditional on a legal private deal.
//! Both players' hole cards are excluded: hero conflicts are masked while
//! aggregating branches and opponent conflicts are masked at terminals.
//! Every compatible private deal therefore has `N - 4` possible next
//! cards among the N cards not yet on the board, so raw counterfactual
//! values divide by `N - 4`, not by an own-hand-only conditional count.
//!
//! Sibling subtrees (a decision node's actions, a chance node's dealt
//! cards) are independent recursions given the strategy at this node, so
//! they're evaluated in parallel with `rayon`. Each decision node's
//! regret/strategy tables live behind a `Mutex` to make that safe — actual
//! contention is near zero, since a node is touched only by the one or two
//! threads whose branch passes through it.

use super::discount::{IterationUpdate, SolverAlgorithm, UpdateSchedule};
use super::terminal::{BoardOrder, TerminalIndex};
use super::tree::{validate_board, BoardError, Node, Player, Terminal};
#[cfg(test)]
use crate::card::CardSet;
use crate::card::{rank, set_contains, set_from_cards, suit, Card};
use crate::eval::evaluate_7;
use crate::range::Combo;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

mod reach_scratch;
use reach_scratch::ReachScratch;

struct NodeData {
    regret_sum: Vec<Vec<f64>>,
    strategy_sum: Vec<Vec<f64>>,
}

impl NodeData {
    fn empty() -> Self {
        Self {
            regret_sum: Vec::new(),
            strategy_sum: Vec::new(),
        }
    }

    fn sized(num_actions: usize, num_combos: usize) -> Self {
        Self {
            regret_sum: vec![vec![0.0; num_combos]; num_actions],
            strategy_sum: vec![vec![0.0; num_combos]; num_actions],
        }
    }
}

/// Which player learns, and which CFV vectors are needed by this traversal.
/// A missing vector stays empty; reaches remain available for own-reach
/// averaging and bucket weights even when only one player's CFV is returned.
#[derive(Clone, Copy)]
struct TrainingPass {
    update: IterationUpdate,
    updating: Option<usize>,
    values: u8,
}

impl TrainingPass {
    fn computes(self, player: usize) -> bool {
        self.values & (1 << player) != 0
    }
}

/// Learning state for one game: regret and strategy-sum tables plus the
/// update policy. The immutable game definition is shared as
/// `Arc<SolverGame>` (`Trainer` derefs to it), so a `Policy` snapshot can be
/// evaluated without touching, or waiting for, the training tables.
pub struct Trainer {
    game: Arc<SolverGame>,
    algorithm: SolverAlgorithm,
    update_schedule: UpdateSchedule,
    /// Full iterations completed across every call to `train`.
    iterations_completed: u64,
    node_data: Vec<Mutex<NodeData>>,
}

impl Deref for Trainer {
    type Target = SolverGame;
    fn deref(&self) -> &SolverGame {
        &self.game
    }
}

/// Normalized, suit-symmetrized average strategy for every decision node,
/// in bucket coordinates (`[node][action][bucket]`), taken at an iteration
/// boundary. Pass it to `SolverGame`'s read methods; it is independent of
/// later training.
#[derive(Debug, Clone)]
pub struct Policy {
    rows: Vec<Vec<Vec<f64>>>,
}

impl Policy {
    /// `[node][action][bucket]` probabilities, e.g. for persisting a snapshot.
    pub fn rows(&self) -> &[Vec<Vec<f64>>] {
        &self.rows
    }
}

/// Why persisted policy rows do not fit a game (see `SolverGame::policy_from_rows`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// The policy has a different number of decision nodes than the game.
    NodeCount {
        /// Decision nodes in the policy.
        policy: usize,
        /// Decision nodes in the game.
        game: usize,
    },
    /// A node's action or bucket count differs from the game's.
    Shape {
        /// The first mismatched decision node id.
        node: usize,
    },
    /// A bucket's probabilities are not finite, nonnegative, and summing to 1.
    NotDistribution {
        /// The first invalid decision node id.
        node: usize,
    },
    /// Suit-equivalent buckets have different probabilities.
    NotSymmetric {
        /// The first asymmetric decision node id.
        node: usize,
    },
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeCount { policy, game } => {
                write!(f, "policy has {policy} nodes, game has {game}")
            }
            Self::Shape { node } => write!(f, "policy shape does not match node {node}"),
            Self::NotDistribution { node } => {
                write!(f, "policy at node {node} is not a probability distribution")
            }
            Self::NotSymmetric { node } => write!(f, "policy at node {node} is not suit-symmetric"),
        }
    }
}

impl std::error::Error for PolicyError {}

/// Why [`Trainer::new`] rejected its inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainerError {
    /// The initial board is not a legal flop, turn, or river.
    Board(BoardError),
    /// The tree does not continue from the initial board: it reaches a
    /// showdown before the river, or deals a card past the river.
    TreeBoardMismatch,
    /// The tree is malformed: decision ids are not exactly
    /// `0..num_decision_nodes`, actions and children differ in length, a
    /// chance node does not deal every undealt card exactly once, or a
    /// terminal investment is negative or not finite.
    InvalidTree {
        /// What was wrong.
        reason: &'static str,
    },
    /// A player's range has no combo with positive weight.
    EmptyRange {
        /// The player whose range is empty.
        player: Player,
    },
    /// A combo has a card outside `0..52` or the same card twice.
    InvalidCombo {
        /// The player whose range holds the combo.
        player: Player,
        /// The combo's index in that player's range.
        index: usize,
    },
    /// A combo appears more than once in the same range.
    DuplicateCombo {
        /// The player whose range holds the combo.
        player: Player,
        /// The index of the second occurrence.
        index: usize,
    },
    /// A combo uses a card that is already on the initial board. Pass the
    /// board as dead cards to [`Range::combos`](crate::Range::combos).
    ComboConflictsWithBoard {
        /// The player whose range holds the combo.
        player: Player,
        /// The combo's index in that player's range.
        index: usize,
    },
    /// A combo weight is negative or not finite.
    InvalidWeight {
        /// The player whose range holds the combo.
        player: Player,
        /// The combo's index in that player's range.
        index: usize,
    },
    /// Every pair of positive-weight combos shares a card, so no deal is
    /// possible and every value would be undefined.
    NoCompatibleDeals,
    /// A [`CardAbstraction`] street asks for zero buckets.
    ZeroBuckets,
    /// A range is not closed under the initial board's free-suit swaps. This
    /// is required when the tree groups chance cards by suit isomorphism (with
    /// equal weights for swapped combos) or when flop or turn buckets are on.
    AsymmetricRange {
        /// The player whose range is asymmetric.
        player: Player,
    },
}

impl std::fmt::Display for TrainerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Board(error) => write!(f, "invalid initial board: {error}"),
            Self::TreeBoardMismatch => write!(f, "tree does not match the initial board"),
            Self::InvalidTree { reason } => write!(f, "invalid tree: {reason}"),
            Self::EmptyRange { player } => {
                write!(f, "{player:?} range has no combo with positive weight")
            }
            Self::InvalidCombo { player, index } => write!(
                f,
                "{player:?} combo {index} must hold two distinct cards in 0..52"
            ),
            Self::DuplicateCombo { player, index } => {
                write!(f, "{player:?} combo {index} repeats an earlier combo")
            }
            Self::ComboConflictsWithBoard { player, index } => write!(
                f,
                "{player:?} combo {index} uses a card on the initial board"
            ),
            Self::InvalidWeight { player, index } => write!(
                f,
                "{player:?} combo {index} weight must be finite and nonnegative"
            ),
            Self::NoCompatibleDeals => {
                write!(f, "no pair of positive-weight combos can be dealt together")
            }
            Self::ZeroBuckets => write!(f, "card abstraction bucket counts must be positive"),
            Self::AsymmetricRange { player } => write!(
                f,
                "{player:?} range must be suit-symmetric for suit isomorphism or flop/turn buckets"
            ),
        }
    }
}

impl std::error::Error for TrainerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Board(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BoardError> for TrainerError {
    fn from(error: BoardError) -> Self {
        Self::Board(error)
    }
}

/// Rejects inputs that `Trainer` cannot solve as a legal game.
fn validate_trainer_inputs(
    tree: &super::tree::Tree,
    initial_board: &[Card],
    combos: &[Vec<Combo>; 2],
    abstraction: CardAbstraction,
) -> Result<(), TrainerError> {
    validate_board(initial_board)?;
    let mut seen_ids = vec![false; tree.num_decision_nodes];
    validate_tree(&tree.root, &mut initial_board.to_vec(), &mut seen_ids)?;
    if seen_ids.iter().any(|&seen| !seen) {
        return Err(TrainerError::InvalidTree {
            reason: "decision ids must cover 0..num_decision_nodes",
        });
    }

    let board = set_from_cards(initial_board);
    let mut masks = [Vec::new(), Vec::new()];
    for (p, player) in [Player::Oop, Player::Ip].into_iter().enumerate() {
        let mut seen = std::collections::HashSet::new();
        for (index, combo) in combos[p].iter().enumerate() {
            let [a, b] = combo.cards;
            if a >= 52 || b >= 52 || a == b {
                return Err(TrainerError::InvalidCombo { player, index });
            }
            if !combo.weight.is_finite() || combo.weight < 0.0 {
                return Err(TrainerError::InvalidWeight { player, index });
            }
            if !seen.insert(normalize_pair(a, b)) {
                return Err(TrainerError::DuplicateCombo { player, index });
            }
            if combo.mask() & board != 0 {
                return Err(TrainerError::ComboConflictsWithBoard { player, index });
            }
            if combo.weight > 0.0 {
                masks[p].push(combo.mask());
            }
        }
        if masks[p].is_empty() {
            return Err(TrainerError::EmptyRange { player });
        }
    }
    if !masks[0]
        .iter()
        .any(|&oop| masks[1].iter().any(|&ip| oop & ip == 0))
    {
        return Err(TrainerError::NoCompatibleDeals);
    }

    if [abstraction.flop, abstraction.turn, abstraction.river].contains(&Some(0)) {
        return Err(TrainerError::ZeroBuckets);
    }
    // Suit-reduced trees derive non-representative runouts by relabeling, so
    // swapped combos must exist with equal weights. Flop and turn buckets
    // average scores over suit-grouped runouts (`avg_score`), which only needs
    // the swapped combos to exist. Later boards only remove free suits, so
    // the initial board's free-suit swaps cover every reachable board.
    let reduced = has_suit_reduction(&tree.root);
    let averaged_buckets = (abstraction.flop.is_some() && initial_board.len() == 3)
        || (abstraction.turn.is_some() && initial_board.len() <= 4);
    if reduced || averaged_buckets {
        let free: Vec<u8> = (0..4)
            .filter(|&s| !initial_board.iter().any(|&c| suit(c) == s))
            .collect();
        for (p, player) in [Player::Oop, Player::Ip].into_iter().enumerate() {
            let weights: HashMap<(Card, Card), f64> = combos[p]
                .iter()
                .map(|combo| (normalize_pair(combo.cards[0], combo.cards[1]), combo.weight))
                .collect();
            for (i, &sa) in free.iter().enumerate() {
                for &sb in &free[i + 1..] {
                    for combo in &combos[p] {
                        let [a, b] = combo.cards;
                        let swapped = normalize_pair(swap_suit(a, sa, sb), swap_suit(b, sa, sb));
                        let symmetric = match weights.get(&swapped) {
                            Some(&weight) => !reduced || weight == combo.weight,
                            None => false,
                        };
                        if !symmetric {
                            return Err(TrainerError::AsymmetricRange { player });
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Checks the structure `Trainer` relies on while walking `node` from `board`.
fn validate_tree(
    node: &Node,
    board: &mut Vec<Card>,
    seen_ids: &mut [bool],
) -> Result<(), TrainerError> {
    let invalid = |reason| Err(TrainerError::InvalidTree { reason });
    match node {
        Node::Decision {
            id,
            actions,
            children,
            ..
        } => {
            match seen_ids.get_mut(*id) {
                Some(seen) if !*seen => *seen = true,
                Some(_) => return invalid("decision ids must be unique"),
                None => return invalid("decision ids must be below num_decision_nodes"),
            }
            if actions.is_empty() || actions.len() != children.len() {
                return invalid("decisions need matching nonempty actions and children");
            }
            for child in children {
                validate_tree(child, board, seen_ids)?;
            }
        }
        Node::Chance { children } => {
            if board.len() >= 5 {
                return Err(TrainerError::TreeBoardMismatch);
            }
            let mut dealt = set_from_cards(board);
            let mut count = 0;
            for (members, _) in children {
                if members.is_empty() {
                    return invalid("chance classes must not be empty");
                }
                for &card in members {
                    if card >= 52 || set_contains(dealt, card) {
                        return invalid("chance nodes must deal each undealt card exactly once");
                    }
                    dealt |= 1u64 << card;
                    count += 1;
                }
            }
            if count != 52 - board.len() {
                return invalid("chance nodes must deal each undealt card exactly once");
            }
            for (members, child) in children {
                board.push(members[0]);
                let result = validate_tree(child, board, seen_ids);
                board.pop();
                result?;
            }
        }
        Node::Terminal(terminal) => {
            let invested = match terminal {
                Terminal::Fold { invested, .. } => invested,
                Terminal::Showdown { invested } => {
                    if board.len() != 5 {
                        return Err(TrainerError::TreeBoardMismatch);
                    }
                    invested
                }
            };
            if invested.iter().any(|x| !x.is_finite() || *x < 0.0) {
                return invalid("terminal investments must be finite and nonnegative");
            }
        }
    }
    Ok(())
}

/// Everything about a solve that training never changes: the tree, ranges,
/// board caches, card abstraction, and suit-symmetry groups. Read methods
/// (`evaluate`, `exploitability`, `average_strategy`) take a `Policy`.
pub struct SolverGame {
    tree: super::tree::Tree,
    initial_board: Vec<Card>,
    combos: [Vec<Combo>; 2],
    /// `evaluate_7` scores for every complete (5-card) board reachable via
    /// a `Chance` node, precomputed once so `train()` doesn't recompute
    /// hand strengths for the same board on every iteration.
    board_eval_cache: HashMap<Vec<Card>, [Vec<u32>; 2]>,
    /// Per player: `(min(card1,card2), max(card1,card2))` -> index into
    /// `combos[player]`. Used by `chance_value`'s suit-isomorphism
    /// relabeling to look up which combo a suit-swapped combo corresponds
    /// to, without a linear scan.
    combo_index: [HashMap<(Card, Card), usize>; 2],
    /// Linear-time card-removal kernels and immutable per-board rank orders.
    terminal_index: TerminalIndex,
    board_order_cache: HashMap<Vec<Card>, BoardOrder>,
    /// Per-street card abstraction setting this trainer was built with —
    /// see `CardAbstraction` and the module doc comment's "card
    /// abstraction" section. `CardAbstraction::default()` never buckets.
    abstraction: CardAbstraction,
    /// Per board (any length reachable by a Decision node — flop/turn use
    /// `avg_score`, river uses the real `evaluate_7` score), per player:
    /// combo index -> bucket index. Only populated for streets
    /// `abstraction` actually buckets.
    bucket_maps: HashMap<Vec<Card>, [Vec<usize>; 2]>,
    /// Per player: `0..combos[player].len()`, the fallback "every combo is
    /// its own bucket" mapping used at any node not covered by
    /// `bucket_maps` (a street `abstraction` leaves unbucketed).
    identity_buckets: [Vec<usize>; 2],
    /// Exact board-stabilizing suit orbits in bucket coordinates. Only reduced
    /// trees need this: roundoff must not turn interchangeable hands into
    /// different policies and invalidate representative-only chance traversal.
    suit_groups: HashMap<Vec<Card>, [Vec<Vec<usize>>; 2]>,
    /// Per decision node id: `(acting player index, board at that node)`,
    /// recorded once during `init_node_data` — `average_strategy` needs
    /// this to look up the node's bucket map and expand bucket-level
    /// probabilities back out to the per-combo output the API contract
    /// promises.
    node_meta: Vec<(usize, Vec<Card>)>,
    #[cfg(test)]
    use_pairwise_terminals: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    terminal_calls: [std::sync::atomic::AtomicUsize; 2],
}

/// How a [`Trainer`] learns: the regret/averaging rule, the player-update
/// order, and per-street card abstraction. `Default` is CFR+ with
/// simultaneous updates and every street solved exactly.
#[derive(Debug, Clone, Copy, Default)]
pub struct TrainerOptions {
    /// The regret and strategy-averaging rule.
    pub algorithm: SolverAlgorithm,
    /// Whether both players update together or in alternation.
    pub schedule: UpdateSchedule,
    /// Which streets are bucketed, and into how many buckets.
    pub abstraction: CardAbstraction,
}

/// Per-street opt-in card abstraction (bucketing). `Some(n)` groups each
/// player's combos on that street into at most `n` percentile buckets of
/// hand strength that share one strategy; tied strengths stay in one bucket,
/// so there can be fewer than `n`. `None` solves the street exactly, and
/// `Default` (all `None`) never buckets. Flop and turn buckets use the mean
/// river score over the remaining runouts; river buckets use the exact score.
#[derive(Debug, Clone, Copy, Default)]
pub struct CardAbstraction {
    /// Bucket count for decisions on a 3-card board.
    pub flop: Option<usize>,
    /// Bucket count for decisions on a 4-card board.
    pub turn: Option<usize>,
    /// Bucket count for decisions on a 5-card board.
    pub river: Option<usize>,
}

/// `SolverGame::evaluate`'s result for one decision node — per combo, for
/// the acting player, under the trained *average* strategy (not
/// `train()`'s informal last-iteration current-strategy value): `ev` is
/// the expected value from this point on (same chip units as
/// `pot`/`bet_sizes`), and `equity` is raw showdown win probability
/// against the opponent's reach-weighted range at this specific node —
/// only computed where it's cheap and exact (a complete, 5-card board),
/// since a true equity figure for an incomplete board needs a genuinely
/// separate future-runout win-probability average (not just
/// `avg_score`'s mean future score), which isn't implemented yet.
#[derive(Debug, Clone)]
pub struct NodeEvaluation {
    /// Per combo of the acting player: the node EV in input chips, i.e. the
    /// raw counterfactual value divided by the opponent's compatible reach
    /// (`0.0` when no opponent combo is compatible). Payoffs count half the
    /// initial pot as each player's own investment, as in [`Terminal`]. This
    /// is the value of the whole node under the average strategy, not the
    /// value of any single action.
    pub ev: Vec<f64>,
    /// Per combo of the acting player: showdown win probability (ties count
    /// half) against the opponent's compatible reach-weighted range, or
    /// `0.5` when no opponent combo is compatible. `Some` only on a complete
    /// 5-card board.
    pub equity: Option<Vec<f64>>,
}

/// `Trainer::exploitability`'s result: the standard CFR convergence metric.
/// `oop`/`ip` are each player's *best-response value* — the range-weighted
/// EV they'd get by deviating to a perfect best response against the
/// **other** player's fixed average strategy (everything else, including
/// the other player's own decisions, still plays the average strategy).
/// Since the game is zero-sum, both would equal the true game value (with
/// opposite sign) at an exact Nash equilibrium, so `total` — their average —
/// is the textbook exploitability figure: `0` at a perfect equilibrium,
/// growing with how much either side can still gain by deviating. Same chip
/// units as `pot`/`bet_sizes` (not normalized to e.g. mbb or % pot).
/// Values are expectations over compatible initial private-hand deals,
/// normalized by their joint range weight. All fields are NaN if that
/// weight is zero or nonfinite; callers must reject such an invalid metric.
#[derive(Debug, Clone, Copy)]
pub struct Exploitability {
    /// OOP's best-response value against IP's average strategy, in chips.
    pub oop: f64,
    /// IP's best-response value against OOP's average strategy, in chips.
    pub ip: f64,
    /// `(oop + ip) / 2`: the exploitability, in chips.
    pub total: f64,
}

impl Trainer {
    /// Builds a trainer for `tree`, starting on `initial_board`, with each
    /// player's combos (the order is kept and
    /// is the combo axis of every per-combo result). Takes ownership of
    /// `tree` so a `Trainer` can outlive the scope that built it.
    ///
    /// `options` selects the regret/averaging rule, the player-update order
    /// (fixed for this trainer's lifetime, including across training batches;
    /// an alternating iteration is an OOP pass followed by an IP pass) and
    /// per-street card abstraction. `TrainerOptions::default()` is CFR+ with
    /// simultaneous updates and no abstraction.
    ///
    /// Trees with grouped chance cards require ranges closed under every
    /// board-free suit swap, with equal weights for the corresponding combos.
    /// Canonical `Range` inputs satisfy this after initial-board removal.
    ///
    /// # Errors
    /// Returns a [`TrainerError`] if the board is illegal, the tree does not
    /// continue from it or is malformed, a range is empty or holds invalid,
    /// repeated, or board-blocked combos, no deal is possible, an abstraction
    /// asks for zero buckets, or a suit-reduced tree gets an asymmetric range.
    pub fn new(
        tree: super::tree::Tree,
        initial_board: Vec<Card>,
        combos: [Vec<Combo>; 2],
        options: TrainerOptions,
    ) -> Result<Self, TrainerError> {
        validate_trainer_inputs(&tree, &initial_board, &combos, options.abstraction)?;
        Ok(Self::new_unchecked(tree, initial_board, combos, options))
    }

    /// `new` without input validation, for tests that drive the terminal
    /// kernels with deliberately degenerate ranges.
    fn new_unchecked(
        tree: super::tree::Tree,
        initial_board: Vec<Card>,
        combos: [Vec<Combo>; 2],
        options: TrainerOptions,
    ) -> Self {
        let TrainerOptions {
            algorithm,
            schedule: update_schedule,
            abstraction,
        } = options;
        let mut board_eval_cache = HashMap::new();
        precompute_board_eval(&tree.root, &initial_board, &combos, &mut board_eval_cache);
        // `precompute_board_eval` only inserts entries for boards reached
        // through a `Chance` node, so a solve that *starts* already at a
        // complete river board (no chance nodes in the tree at all — the
        // common "solve this exact river spot" case) would otherwise never
        // get an entry, silently disabling bucketing for it.
        if initial_board.len() == 5 {
            board_eval_cache
                .entry(initial_board.clone())
                .or_insert_with(|| eval_combos_if_complete(&combos, &initial_board));
        }

        let combo_index = [build_combo_index(&combos[0]), build_combo_index(&combos[1])];
        let terminal_index = TerminalIndex::new(&combos);
        let board_order_cache = board_eval_cache
            .iter()
            .filter(|(board, _)| board.len() == 5)
            .map(|(board, scores)| {
                (
                    board.clone(),
                    BoardOrder::new(&combos, scores, set_from_cards(board)),
                )
            })
            .collect();

        let mut bucket_maps: HashMap<Vec<Card>, [Vec<usize>; 2]> = HashMap::new();
        if let Some(n) = abstraction.river {
            // The cache also holds turn boards (with empty scores); bucketing
            // those would give turn decisions zero buckets.
            for (board, scores) in board_eval_cache.iter().filter(|(b, _)| b.len() == 5) {
                let scores_f64 = [
                    scores[0].iter().map(|&s| s as f64).collect::<Vec<f64>>(),
                    scores[1].iter().map(|&s| s as f64).collect::<Vec<f64>>(),
                ];
                bucket_maps.insert(
                    board.clone(),
                    [
                        compute_percentile_buckets(&scores_f64[0], n),
                        compute_percentile_buckets(&scores_f64[1], n),
                    ],
                );
            }
        }
        if abstraction.flop.is_some() || abstraction.turn.is_some() {
            // Every board a Decision node could actually be built at: the
            // initial board itself (same "starts already at this street"
            // reasoning as the river fix above), plus every board reached
            // by dealing a class representative — matching precisely what
            // `cfr()` ever recurses into (`chance_value` never visits a
            // non-representative member's own board; it derives those via
            // relabeling), so this can't miss or over-collect a board.
            let mut reachable_boards = std::collections::HashSet::new();
            reachable_boards.insert(initial_board.clone());
            collect_reachable_boards(&tree.root, &initial_board, &mut reachable_boards);
            let mut avg_score_cache = HashMap::new();
            for board in &reachable_boards {
                let n = match board.len() {
                    3 => abstraction.flop,
                    4 => abstraction.turn,
                    _ => None,
                };
                if let Some(n) = n {
                    let scores = avg_score(
                        board,
                        &combos,
                        &combo_index,
                        &board_eval_cache,
                        &mut avg_score_cache,
                    );
                    bucket_maps.insert(
                        board.clone(),
                        [
                            compute_percentile_buckets(&scores[0], n),
                            compute_percentile_buckets(&scores[1], n),
                        ],
                    );
                }
            }
        }
        let identity_buckets = [
            (0..combos[0].len()).collect(),
            (0..combos[1].len()).collect(),
        ];

        let mut node_data: Vec<NodeData> = (0..tree.num_decision_nodes)
            .map(|_| NodeData::empty())
            .collect();
        let mut node_meta: Vec<(usize, Vec<Card>)> = vec![(0, Vec::new()); tree.num_decision_nodes];
        init_node_data(
            &tree.root,
            &initial_board,
            &bucket_maps,
            &identity_buckets,
            &mut node_data,
            &mut node_meta,
        );
        let node_data: Vec<Mutex<NodeData>> = node_data.into_iter().map(Mutex::new).collect();
        let mut suit_groups = HashMap::new();
        if has_suit_reduction(&tree.root) {
            for (_, board) in &node_meta {
                suit_groups.entry(board.clone()).or_insert_with(|| {
                    let maps = bucket_maps.get(board).unwrap_or(&identity_buckets);
                    suit_bucket_groups(board, &combos, &combo_index, maps)
                });
            }
        }

        let game = SolverGame {
            tree,
            initial_board,
            combos,
            board_eval_cache,
            combo_index,
            terminal_index,
            board_order_cache,
            abstraction,
            bucket_maps,
            identity_buckets,
            suit_groups,
            node_meta,
            #[cfg(test)]
            use_pairwise_terminals: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            terminal_calls: std::array::from_fn(|_| std::sync::atomic::AtomicUsize::new(0)),
        };
        Self {
            game: Arc::new(game),
            algorithm,
            update_schedule,
            iterations_completed: 0,
            node_data,
        }
    }

    /// The fixed update policy selected at construction.
    pub fn algorithm(&self) -> SolverAlgorithm {
        self.algorithm
    }

    /// The fixed player-update order selected at construction.
    pub fn update_schedule(&self) -> UpdateSchedule {
        self.update_schedule
    }

    /// Full iterations completed across all training batches (two passes per
    /// alternating iteration, one per simultaneous iteration). In DCFR this
    /// is the global clock for discounting; `train(0)` does not advance it.
    pub fn iterations_completed(&self) -> u64 {
        self.iterations_completed
    }

    /// Run `iterations` CFR iterations. Returns the root expected
    /// value for `[OOP, IP]` under the *final iteration's* current
    /// strategy (informal — use `average_strategy` for the converged
    /// policy, which is stable across iterations). For alternating updates,
    /// the return is from the final IP pass, after OOP updated and before IP updates.
    pub fn train(&mut self, iterations: u32) -> [f64; 2] {
        let mut last = [0.0, 0.0];
        // Borrow the shared game, not `self`, so the iteration clock can advance.
        let game = Arc::clone(&self.game);
        let board = game.initial_board.clone();
        let empty_eval = [Vec::new(), Vec::new()];
        let board_eval = game.board_eval_cache.get(&board).unwrap_or(&empty_eval);
        for step in 0..iterations {
            let iteration = self
                .iterations_completed
                .checked_add(1)
                .expect("CFR iteration counter overflow");
            let update = self.algorithm.iteration(iteration);
            let reach0: Vec<f64> = self.combos[0].iter().map(|c| c.weight).collect();
            let reach1: Vec<f64> = self.combos[1].iter().map(|c| c.weight).collect();
            let reach = [reach0.as_slice(), reach1.as_slice()];
            let pass = match self.update_schedule {
                UpdateSchedule::Simultaneous => TrainingPass {
                    update,
                    updating: None,
                    values: 3,
                },
                UpdateSchedule::Alternating => {
                    self.cfr(
                        &self.tree.root,
                        reach,
                        &board,
                        board_eval,
                        TrainingPass {
                            update,
                            updating: Some(0),
                            values: 1,
                        },
                    );
                    // Preserve the exact public train() return on the final IP
                    // pass. Interior iterations need only the learning player's
                    // CFVs; no zero-sum reconstruction or extra traversal.
                    TrainingPass {
                        update,
                        updating: Some(1),
                        values: if step + 1 == iterations { 3 } else { 2 },
                    }
                }
            };
            let [cfv0, cfv1] = self.cfr(&self.tree.root, reach, &board, board_eval, pass);
            let ev0: f64 = cfv0
                .iter()
                .zip(&self.combos[0])
                .map(|(v, c)| v * c.weight)
                .sum();
            let ev1: f64 = cfv1
                .iter()
                .zip(&self.combos[1])
                .map(|(v, c)| v * c.weight)
                .sum();
            last = [ev0, ev1];
            self.iterations_completed = iteration;
        }
        last
    }

    /// The shared, immutable game definition.
    pub fn game(&self) -> &Arc<SolverGame> {
        &self.game
    }

    /// Snapshot of the average strategy at this iteration boundary.
    pub fn policy(&self) -> Policy {
        Policy {
            rows: (0..self.node_data.len())
                .map(|id| self.average_bucket_strategy(id))
                .collect(),
        }
    }

    /// `SolverGame::evaluate` under the current average strategy.
    pub fn evaluate(&self) -> (HashMap<usize, NodeEvaluation>, [f64; 2]) {
        self.game.evaluate(&self.policy())
    }

    /// `SolverGame::exploitability` under the current average strategy.
    pub fn exploitability(&self) -> Exploitability {
        self.game.exploitability(&self.policy())
    }

    /// The trained average strategy for one decision node, as
    /// `[action][combo]` (see `SolverGame::average_strategy`).
    pub fn average_strategy(&self, node_id: usize) -> Vec<Vec<f64>> {
        self.game
            .expand_strategy(node_id, &self.average_bucket_strategy(node_id))
    }

    /// Normalized strategy sums in bucket coordinates, suit-symmetrized.
    fn average_bucket_strategy(&self, node_id: usize) -> Vec<Vec<f64>> {
        let (player, board) = &self.node_meta[node_id];
        let (num_buckets, _) = self.combo_buckets(*player, board);
        let data = self.node_data[node_id].lock().unwrap();
        let num_actions = data.regret_sum.len();
        let mut bucket_strategy: Vec<Vec<f64>> = (0..num_actions)
            .map(|a| {
                (0..num_buckets)
                    .map(|b| {
                        let total: f64 = (0..num_actions).map(|a2| data.strategy_sum[a2][b]).sum();
                        if total > 0.0 {
                            data.strategy_sum[a][b] / total
                        } else {
                            1.0 / num_actions as f64
                        }
                    })
                    .collect()
            })
            .collect();
        self.symmetrize_strategy(node_id, &mut bucket_strategy);
        bucket_strategy
    }

    fn current_strategy(&self, id: usize, num_actions: usize, num_combos: usize) -> Vec<Vec<f64>> {
        let data = self.node_data[id].lock().unwrap();
        let mut strategy = vec![vec![0.0; num_combos]; num_actions];
        for i in 0..num_combos {
            let mut total = 0.0;
            for (regret_row, strat_row) in data.regret_sum.iter().zip(strategy.iter_mut()) {
                let r = regret_row[i].max(0.0);
                strat_row[i] = r;
                total += r;
            }
            if total > 0.0 {
                for row in strategy.iter_mut().take(num_actions) {
                    row[i] /= total;
                }
            } else {
                for row in strategy.iter_mut().take(num_actions) {
                    row[i] = 1.0 / num_actions as f64;
                }
            }
        }
        self.symmetrize_strategy(id, &mut strategy);
        strategy
    }

    /// Recurse into `node`. Sibling subtrees (a decision's actions, a
    /// chance node's dealt cards) are independent given the strategy
    /// fixed at this node, so they run in parallel via rayon; only the
    /// regret/strategy-sum update at the end takes the node's `Mutex`.
    fn cfr(
        &self,
        node: &Node,
        reach: [&[f64]; 2],
        board: &[Card],
        board_eval: &[Vec<u32>; 2],
        pass: TrainingPass,
    ) -> [Vec<f64>; 2] {
        match node {
            Node::Terminal(t) => std::array::from_fn(|p| {
                if pass.computes(p) {
                    self.terminal_value_for_player(t, p, reach[1 - p], board, board_eval)
                } else {
                    Vec::new()
                }
            }),
            Node::Chance { children } => self.chance_value(children, reach, board, pass),
            Node::Decision {
                id,
                player,
                actions,
                children,
            } => {
                let p = player.index();
                let num_own_combos = self.combos[p].len();
                let num_other_combos = self.combos[1 - p].len();
                let (num_buckets, buckets) = self.combo_buckets(p, board);
                let strategy = self.current_strategy(*id, actions.len(), num_buckets);

                let child_cfvs: Vec<[Vec<f64>; 2]> = children
                    .par_iter()
                    .enumerate()
                    .map(|(a, child)| {
                        // This action changes only the acting player's reach.
                        // Each sibling owns that vector; the unchanged opponent
                        // slice is shared read-only through the recursive call.
                        let mut own_reach = ReachScratch::copy_from(reach[p]);
                        for i in 0..num_own_combos {
                            own_reach[i] *= strategy[a][buckets[i]];
                        }
                        let mut child_reach = reach;
                        child_reach[p] = &own_reach;
                        self.cfr(child, child_reach, board, board_eval, pass)
                    })
                    .collect();

                let mut own_cfv = vec![0.0; if pass.computes(p) { num_own_combos } else { 0 }];
                let mut other_cfv = vec![
                    0.0;
                    if pass.computes(1 - p) {
                        num_other_combos
                    } else {
                        0
                    }
                ];
                for (a, cfv) in child_cfvs.iter().enumerate() {
                    let own = &cfv[p];
                    let other = &cfv[1 - p];
                    for (i, v) in own_cfv.iter_mut().enumerate() {
                        *v += strategy[a][buckets[i]] * own[i];
                    }
                    for (j, v) in other_cfv.iter_mut().enumerate() {
                        *v += other[j];
                    }
                }

                // A non-updating player's strategy affects the requested CFVs,
                // but its regrets and strategy mass must remain untouched.
                if pass.updating.is_some_and(|target| target != p) {
                    return if p == 0 {
                        [own_cfv, other_cfv]
                    } else {
                        [other_cfv, own_cfv]
                    };
                }

                // With no abstraction maps every bucket is exactly combo i.
                // Its merge weight is one even for zero/board-dead own reach;
                // terminal CFVs already exclude impossible private deals.
                // Do not infer this from bucket counts: tied scores can leave
                // holes in a map while still sharing a bucket.
                if self.bucket_maps.is_empty() {
                    let mut data = self.node_data[*id].lock().unwrap();
                    for (a, cfv) in child_cfvs.iter().enumerate() {
                        for (i, &own) in cfv[p].iter().enumerate() {
                            // Keep the old zero-initialized delta accumulation,
                            // including its signed-zero behavior.
                            let delta = 0.0 + (own - own_cfv[i]);
                            data.regret_sum[a][i] =
                                pass.update.regret(data.regret_sum[a][i], delta);
                            data.strategy_sum[a][i] += reach[p][i] * strategy[a][i];
                            data.strategy_sum[a][i] =
                                pass.update.strategy_sum(data.strategy_sum[a][i]);
                        }
                    }
                    return if p == 0 {
                        [own_cfv, other_cfv]
                    } else {
                        [other_cfv, own_cfv]
                    };
                }

                // `effective_reach` treats a combo whose hole card physically
                // conflicts with `board` (impossible on this specific
                // runout — see `eval_combos_if_complete`'s doc comment) as
                // reach 0 *only* for the bucket-merge weighting below, so a
                // phantom combo sharing a bucket with a real one can never
                // skew the real combo's learned regret. When every combo is
                // its own bucket (unabstracted, or `river_buckets` capped to
                // a bijection) each bucket's total is either that one
                // combo's own reach or — if masked or genuinely zero — falls
                // through to the equal-weight branch below with a single
                // member, which reduces exactly to the plain per-combo
                // regret update this replaces (`own[i] - own_cfv[i]`, no
                // reach weighting at all), so this is an exact
                // generalization, not a behavior change, for that case.
                let board_set = set_from_cards(board);
                let effective_reach: Vec<f64> = (0..num_own_combos)
                    .map(|i| {
                        let [h1, h2] = self.combos[p][i].cards;
                        if set_contains(board_set, h1) || set_contains(board_set, h2) {
                            0.0
                        } else {
                            reach[p][i]
                        }
                    })
                    .collect();
                let mut bucket_reach_total = vec![0.0; num_buckets];
                let mut bucket_size = vec![0usize; num_buckets];
                for i in 0..num_own_combos {
                    bucket_reach_total[buckets[i]] += effective_reach[i];
                    bucket_size[buckets[i]] += 1;
                }

                {
                    let mut data = self.node_data[*id].lock().unwrap();
                    for (a, cfv) in child_cfvs.iter().enumerate() {
                        let own = &cfv[p];
                        let mut regret_delta = vec![0.0; num_buckets];
                        for i in 0..num_own_combos {
                            let b = buckets[i];
                            let delta = own[i] - own_cfv[i];
                            let weight = if bucket_reach_total[b] > 0.0 {
                                effective_reach[i] / bucket_reach_total[b]
                            } else {
                                1.0 / bucket_size[b] as f64
                            };
                            regret_delta[b] += weight * delta;
                            data.strategy_sum[a][b] += reach[p][i] * strategy[a][b];
                        }
                        for (b, delta) in regret_delta.iter().enumerate() {
                            data.regret_sum[a][b] =
                                pass.update.regret(data.regret_sum[a][b], *delta);
                            // Once per bucket, after every combo's contribution:
                            // never apply DCFR discounting inside the combo loop.
                            data.strategy_sum[a][b] =
                                pass.update.strategy_sum(data.strategy_sum[a][b]);
                        }
                    }
                }

                if p == 0 {
                    [own_cfv, other_cfv]
                } else {
                    [other_cfv, own_cfv]
                }
            }
        }
    }

    /// Average raw `cfv` over every legal next card of each compatible
    /// private deal. Hero-card conflicts are excluded here; opponent-card
    /// conflicts are excluded at terminals. Together they leave N - 4
    /// possible cards, even though the outer vector is hero-combo indexed.
    ///
    /// `children` is grouped into suit-isomorphism classes (see
    /// `tree::deal_classes`): only the class's representative
    /// (`members[0]`) is actually recursed into. Every other member's
    /// counterfactual values are derived from the representative's by
    /// `relabel_cfv` — a swap of the two suits involved is a symmetry of
    /// the game (it doesn't change tree structure or `evaluate_7`
    /// outcomes), so member `card`'s cfv for combo `j` equals the
    /// representative's cfv for the combo obtained by applying that same
    /// suit swap to combo `j`'s two cards. With isomorphism disabled every
    /// class is a singleton and this reduces to the original per-card loop.
    fn chance_value(
        &self,
        children: &[(Vec<Card>, Node)],
        reach: [&[f64]; 2],
        board: &[Card],
        pass: TrainingPass,
    ) -> [Vec<f64>; 2] {
        let n: usize = children.iter().map(|(members, _)| members.len()).sum();
        let divisor = (n as f64) - 4.0;

        let class_results: Vec<(&Vec<Card>, [Vec<f64>; 2])> = children
            .par_iter()
            .map(|(members, child)| {
                let rep = members[0];
                let mut child_board = board.to_vec();
                child_board.push(rep);
                let empty_eval = [Vec::new(), Vec::new()];
                let child_board_eval = self
                    .board_eval_cache
                    .get(&child_board)
                    .unwrap_or(&empty_eval);
                let child_cfv = self.cfr(child, reach, &child_board, child_board_eval, pass);
                (members, child_cfv)
            })
            .collect();

        let mut out: [Vec<f64>; 2] = std::array::from_fn(|p| {
            vec![
                0.0;
                if pass.computes(p) {
                    self.combos[p].len()
                } else {
                    0
                }
            ]
        });
        for (members, rep_cfv) in &class_results {
            let rep = members[0];
            for &card in members.iter() {
                for p in 0..2 {
                    if !pass.computes(p) {
                        continue;
                    }
                    let values = if card == rep {
                        rep_cfv[p].clone()
                    } else {
                        self.relabel_cfv(p, &rep_cfv[p], suit(rep), suit(card))
                    };
                    for (
                        i,
                        &Combo {
                            cards: [h1, h2], ..
                        },
                    ) in self.combos[p].iter().enumerate()
                    {
                        if h1 != card && h2 != card {
                            out[p][i] += values[i];
                        }
                    }
                }
            }
        }
        for values in &mut out {
            for v in values {
                *v /= divisor;
            }
        }
        out
    }
}

impl SolverGame {
    /// Number of regret/strategy rows (buckets, or plain combos if
    /// unabstracted) `player`'s combos collapse into at the given `board`.
    fn combo_buckets(&self, player: usize, board: &[Card]) -> (usize, &[usize]) {
        bucket_count_and_map(player, board, &self.bucket_maps, &self.identity_buckets)
    }

    /// One full-tree pass under the fixed, converged *average* strategy —
    /// as opposed to `cfr()`'s training pass, which uses the evolving
    /// regret-matching "current" strategy and mutates `node_data`. Reads only
    /// the immutable game and `policy`, so it runs concurrently with training
    /// and always describes exactly the iteration `policy` was taken at.
    ///
    /// Returns every visited decision node's `NodeEvaluation`, keyed by
    /// node id — computing this for a single node still requires walking
    /// its entire subtree (unlike `average_strategy`, which is a direct
    /// per-node lookup), so callers that need more than one node's
    /// evaluation should call this once and read from the map, not call it
    /// per node — plus the root's own range-weighted `[OOP, IP]` value
    /// pair (same shape and same zero-sum property as `train()`'s return,
    /// but under the average strategy rather than the last training
    /// iteration's current strategy).
    pub fn evaluate(&self, policy: &Policy) -> (HashMap<usize, NodeEvaluation>, [f64; 2]) {
        let mut out = HashMap::new();
        let board = self.initial_board.clone();
        let empty_eval = [Vec::new(), Vec::new()];
        let board_eval = self.board_eval_cache.get(&board).unwrap_or(&empty_eval);
        let reach0: Vec<f64> = self.combos[0].iter().map(|c| c.weight).collect();
        let reach1: Vec<f64> = self.combos[1].iter().map(|c| c.weight).collect();
        let [cfv0, cfv1] = self.evaluate_node(
            policy,
            &self.tree.root,
            &[reach0, reach1],
            &board,
            board_eval,
            &mut out,
        );
        let ev0: f64 = cfv0
            .iter()
            .zip(&self.combos[0])
            .map(|(v, c)| v * c.weight)
            .sum();
        let ev1: f64 = cfv1
            .iter()
            .zip(&self.combos[1])
            .map(|(v, c)| v * c.weight)
            .sum();
        (out, [ev0, ev1])
    }

    /// The standard CFR convergence metric: how much either player could
    /// still gain, in expectation over their full range, by deviating from
    /// the trained average strategy to a perfect best response against the
    /// *other* player's average strategy (see `Exploitability`'s doc
    /// comment for the exact definition). Two independent full-tree passes
    /// — one best-responding as OOP, one as IP — via `best_response_value`,
    /// which mirrors `evaluate_node`'s structure but takes a max over
    /// actions (instead of averaging by `average_strategy`) at the
    /// deviating player's own decision nodes.
    pub fn exploitability(&self, policy: &Policy) -> Exploitability {
        let board = self.initial_board.clone();
        let empty_eval = [Vec::new(), Vec::new()];
        let board_eval = self.board_eval_cache.get(&board).unwrap_or(&empty_eval);
        let reach0: Vec<f64> = self.combos[0].iter().map(|c| c.weight).collect();
        let reach1: Vec<f64> = self.combos[1].iter().map(|c| c.weight).collect();

        // CFVs remain unnormalized throughout recursion. Only the public
        // metric is an expectation over legal initial deals, rather than a
        // sum that changes with the size or common scale of either range.
        let board_set = set_from_cards(&board);
        let live = |c: &Combo| {
            !set_contains(board_set, c.cards[0]) && !set_contains(board_set, c.cards[1])
        };
        let compatible_mass =
            self.terminal_index
                .compatible_mass(&self.combos, 0, &reach1, board_set);
        let joint_mass: f64 = self.combos[0]
            .iter()
            .zip(&compatible_mass)
            .filter(|(c, _)| live(c))
            .map(|(c, mass)| c.weight * mass)
            .sum();
        if !joint_mass.is_finite() || joint_mass <= 0.0 {
            return Exploitability {
                oop: f64::NAN,
                ip: f64::NAN,
                total: f64::NAN,
            };
        }

        let br0 = self.best_response_value(
            policy,
            &self.tree.root,
            0,
            &[reach0.clone(), reach1.clone()],
            &board,
            board_eval,
        );
        let br1 = self.best_response_value(
            policy,
            &self.tree.root,
            1,
            &[reach0, reach1],
            &board,
            board_eval,
        );
        let oop = br0
            .iter()
            .zip(&self.combos[0])
            .filter(|(_, c)| live(c))
            .map(|(v, c)| v * c.weight)
            .sum::<f64>()
            / joint_mass;
        let ip = br1
            .iter()
            .zip(&self.combos[1])
            .filter(|(_, c)| live(c))
            .map(|(v, c)| v * c.weight)
            .sum::<f64>()
            / joint_mass;
        Exploitability {
            oop,
            ip,
            total: (oop + ip) / 2.0,
        }
    }

    /// The betting tree this trainer is solving, e.g. to look up node ids
    /// by walking its structure.
    pub fn tree(&self) -> &Node {
        &self.tree.root
    }

    /// The community cards dealt so far at decision node `node_id` (e.g.
    /// for display, or to label which card a `Chance` branch led through).
    pub fn node_board(&self, node_id: usize) -> &[Card] {
        &self.node_meta[node_id].1
    }

    /// The combos (`card1, card2, range_weight`) for `Player::Oop`/`Player::Ip`
    /// (index `0`/`1`), in the same order `average_strategy`'s combo axis uses.
    pub fn combos(&self, player_index: usize) -> &[Combo] {
        &self.combos[player_index]
    }

    /// The card abstraction setting this trainer was built with (see
    /// `Trainer::new`) — a `None` field means that street's decision nodes
    /// are exact (unabstracted).
    pub fn abstraction(&self) -> CardAbstraction {
        self.abstraction
    }

    /// The trained average strategy for a decision node, as `[action][combo]`
    /// probabilities (combos ordered as in the input range for that player)
    /// — always combo-indexed regardless of whether this node was
    /// river-bucketed internally: every combo in a bucket reports that
    /// bucket's shared average probability.
    pub fn average_strategy(&self, policy: &Policy, node_id: usize) -> Vec<Vec<f64>> {
        self.expand_strategy(node_id, &policy.rows[node_id])
    }

    /// Expands bucket rows to one probability per combo of the acting player.
    fn expand_strategy(&self, node_id: usize, bucket_strategy: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (player, board) = &self.node_meta[node_id];
        let (_, buckets) = self.combo_buckets(*player, board);
        bucket_strategy
            .iter()
            .map(|row| buckets.iter().map(|&b| row[b]).collect())
            .collect()
    }

    /// Rebuilds a policy persisted from `Policy::rows`, checking it fits this
    /// game: one row set per decision node, one row per action, one entry per
    /// bucket, each bucket a probability distribution, and suit-equivalent
    /// buckets identical. Rows are taken as given (a saved policy is already
    /// symmetrized), so reads match bit for bit; an asymmetric one is rejected
    /// because representative-only chance traversal would misreport it.
    pub fn policy_from_rows(&self, rows: Vec<Vec<Vec<f64>>>) -> Result<Policy, PolicyError> {
        fn action_counts(node: &Node, out: &mut [usize]) {
            match node {
                Node::Decision {
                    id,
                    actions,
                    children,
                    ..
                } => {
                    out[*id] = actions.len();
                    children.iter().for_each(|child| action_counts(child, out));
                }
                Node::Chance { children } => children
                    .iter()
                    .for_each(|(_, child)| action_counts(child, out)),
                Node::Terminal(_) => {}
            }
        }
        if rows.len() != self.num_decision_nodes() {
            return Err(PolicyError::NodeCount {
                policy: rows.len(),
                game: self.num_decision_nodes(),
            });
        }
        let mut actions = vec![0; rows.len()];
        action_counts(&self.tree.root, &mut actions);
        for (id, node_rows) in rows.iter().enumerate() {
            let (player, board) = &self.node_meta[id];
            let (num_buckets, _) = self.combo_buckets(*player, board);
            if node_rows.len() != actions[id]
                || node_rows.iter().any(|row| row.len() != num_buckets)
            {
                return Err(PolicyError::Shape { node: id });
            }
            for b in 0..num_buckets {
                let total: f64 = node_rows.iter().map(|row| row[b]).sum();
                if node_rows
                    .iter()
                    .any(|row| !row[b].is_finite() || row[b] < 0.0)
                    || (total - 1.0).abs() > 1e-9
                {
                    return Err(PolicyError::NotDistribution { node: id });
                }
            }
            if let Some(groups) = self.suit_groups.get(board) {
                let asymmetric = groups[*player].iter().any(|group| {
                    node_rows.iter().any(|row| {
                        group
                            .iter()
                            .any(|&b| row[b].to_bits() != row[group[0]].to_bits())
                    })
                });
                if asymmetric {
                    return Err(PolicyError::NotSymmetric { node: id });
                }
            }
        }
        Ok(Policy { rows })
    }

    /// Card-abstraction bucket of every combo, per board and player, sorted by
    /// board. Empty when no street is bucketed. Lets a persisted policy check
    /// that its rows still mean the same hands.
    pub fn bucket_assignments(&self) -> Vec<(&[Card], &[Vec<usize>; 2])> {
        let mut maps: Vec<_> = self
            .bucket_maps
            .iter()
            .map(|(board, maps)| (board.as_slice(), maps))
            .collect();
        maps.sort_unstable_by(|a, b| a.0.cmp(b.0));
        maps
    }

    /// Number of decision nodes (valid `node_id`s are `0..num_decision_nodes`).
    pub fn num_decision_nodes(&self) -> usize {
        self.node_meta.len()
    }

    fn symmetrize_strategy(&self, id: usize, strategy: &mut [Vec<f64>]) {
        let (player, board) = &self.node_meta[id];
        if let Some(groups) = self.suit_groups.get(board) {
            for group in &groups[*player] {
                for row in strategy.iter_mut() {
                    let probability =
                        group.iter().map(|&i| row[i]).sum::<f64>() / group.len() as f64;
                    for &i in group {
                        row[i] = probability;
                    }
                }
            }
        }
    }

    /// Re-expresses `rep_cfv` (computed with the class representative
    /// dealt, i.e. under a board whose newest card has suit `suit_a`) as
    /// the cfv for a board whose newest card is the same rank but suit
    /// `suit_b` instead: swapping `suit_a`/`suit_b` on both players' cards
    /// leaves the tree structure and every hand-strength comparison
    /// unchanged (both suits were free before this deal, so the swap fixes
    /// the rest of the board), so combo `j`'s cfv under the `suit_b`
    /// version equals the representative's cfv for whichever combo `j`
    /// becomes after swapping its two cards' suits.
    fn relabel_cfv(&self, player: usize, rep_cfv: &[f64], suit_a: u8, suit_b: u8) -> Vec<f64> {
        self.combos[player]
            .iter()
            .map(
                |&Combo {
                     cards: [h1, h2], ..
                 }| {
                    let s1 = swap_suit(h1, suit_a, suit_b);
                    let s2 = swap_suit(h2, suit_a, suit_b);
                    let idx = self.combo_index[player][&normalize_pair(s1, s2)];
                    rep_cfv[idx]
                },
            )
            .collect()
    }

    /// `evaluate()`'s recursion — mirrors `cfr()`'s structure (same
    /// `Terminal`/`Chance`/`Decision` handling), but reads the fixed
    /// `average_strategy` at each decision node instead of computing a
    /// "current" one via regret matching, and never touches `node_data`.
    /// `terminal_value` doesn't need a parallel version — it's already
    /// policy-agnostic, depending only on the `reach` and `board_eval`
    /// threaded down to it, not on how the strategy that produced `reach`
    /// was obtained.
    ///
    /// `average_strategy` already expands any river/flop/turn bucketing
    /// back out to one probability per real combo (see its own doc
    /// comment), so — unlike `cfr()`'s training branch — this never needs
    /// bucket-aware weighting: every combo's regret/EV here is already
    /// fully independent, matching the pre-abstraction per-combo design.
    fn evaluate_node(
        &self,
        policy: &Policy,
        node: &Node,
        reach: &[Vec<f64>; 2],
        board: &[Card],
        board_eval: &[Vec<u32>; 2],
        out: &mut HashMap<usize, NodeEvaluation>,
    ) -> [Vec<f64>; 2] {
        match node {
            Node::Terminal(t) => self.terminal_value(t, reach, board, board_eval),
            Node::Chance { children } => {
                self.evaluate_chance_value(policy, children, reach, board, out)
            }
            Node::Decision {
                id,
                player,
                children,
                ..
            } => {
                let p = player.index();
                let num_own_combos = self.combos[p].len();
                let num_other_combos = self.combos[1 - p].len();
                let strategy = self.average_strategy(policy, *id);

                let child_cfvs: Vec<[Vec<f64>; 2]> = children
                    .iter()
                    .enumerate()
                    .map(|(a, child)| {
                        let mut child_reach = reach.clone();
                        for i in 0..num_own_combos {
                            child_reach[p][i] *= strategy[a][i];
                        }
                        self.evaluate_node(policy, child, &child_reach, board, board_eval, out)
                    })
                    .collect();

                let mut own_ev = vec![0.0; num_own_combos];
                let mut other_ev = vec![0.0; num_other_combos];
                for (a, cfv) in child_cfvs.iter().enumerate() {
                    let own = &cfv[p];
                    let other = &cfv[1 - p];
                    for i in 0..num_own_combos {
                        own_ev[i] += strategy[a][i] * own[i];
                    }
                    for (j, v) in other_ev.iter_mut().enumerate() {
                        *v += other[j];
                    }
                }

                let equity = if board.len() == 5 {
                    Some(self.equity_at_node(p, reach, board, board_eval))
                } else {
                    None
                };

                // `own_ev` here is still a raw counterfactual value: a SUM
                // weighted by the opponent's *unnormalized* reach across all
                // their live combos (the convention used throughout this
                // module — see `showdown_cfv`/`terminal_value`/`chance_value`),
                // not a human-meaningful "expected value of holding this
                // exact combo". By CFR's `v(I) = pi_{-i}(I) * u_i(sigma|I)`
                // identity, dividing by the opponent's total live reach
                // (`non_conflicting_reach_sum`, the same board/hand-blocking
                // exclusion `equity_at_node` already applies) recovers the
                // true per-combo expected value for display. Only the
                // *stored* value is normalized this way — the *returned*
                // own_ev/other_ev must stay in the raw convention so the
                // parent's own weighted-sum aggregation and the chance-node
                // relabeling stay correct.
                let board_set = set_from_cards(board);
                let compatible_mass =
                    self.terminal_index
                        .compatible_mass(&self.combos, p, &reach[1 - p], board_set);
                let display_ev: Vec<f64> = (0..num_own_combos)
                    .map(|i| {
                        let denom = compatible_mass[i];
                        if denom > 0.0 {
                            own_ev[i] / denom
                        } else {
                            0.0
                        }
                    })
                    .collect();
                out.insert(
                    *id,
                    NodeEvaluation {
                        ev: display_ev,
                        equity,
                    },
                );

                if p == 0 {
                    [own_ev, other_ev]
                } else {
                    [other_ev, own_ev]
                }
            }
        }
    }

    /// `evaluate_node`'s counterpart to `chance_value`: same
    /// suit-isomorphism relabeling (see that function's doc comment for
    /// why a class's non-representative members can't just reuse the
    /// representative's own values unchanged), but recursing via
    /// `evaluate_node` instead of `cfr`.
    fn evaluate_chance_value(
        &self,
        policy: &Policy,
        children: &[(Vec<Card>, Node)],
        reach: &[Vec<f64>; 2],
        board: &[Card],
        out: &mut HashMap<usize, NodeEvaluation>,
    ) -> [Vec<f64>; 2] {
        let n: usize = children.iter().map(|(members, _)| members.len()).sum();
        let divisor = (n as f64) - 4.0;

        let class_results: Vec<(&Vec<Card>, [Vec<f64>; 2])> = children
            .iter()
            .map(|(members, child)| {
                let rep = members[0];
                let mut child_board = board.to_vec();
                child_board.push(rep);
                let empty_eval = [Vec::new(), Vec::new()];
                let child_board_eval = self
                    .board_eval_cache
                    .get(&child_board)
                    .unwrap_or(&empty_eval);
                let child_cfv =
                    self.evaluate_node(policy, child, reach, &child_board, child_board_eval, out);
                (members, child_cfv)
            })
            .collect();

        let mut cfv0 = vec![0.0; self.combos[0].len()];
        let mut cfv1 = vec![0.0; self.combos[1].len()];

        for (members, rep_cfv) in &class_results {
            let rep = members[0];
            for &card in members.iter() {
                let (cfv0_card, cfv1_card) = if card == rep {
                    (rep_cfv[0].clone(), rep_cfv[1].clone())
                } else {
                    (
                        self.relabel_cfv(0, &rep_cfv[0], suit(rep), suit(card)),
                        self.relabel_cfv(1, &rep_cfv[1], suit(rep), suit(card)),
                    )
                };
                for (
                    i,
                    &Combo {
                        cards: [h1, h2], ..
                    },
                ) in self.combos[0].iter().enumerate()
                {
                    if h1 != card && h2 != card {
                        cfv0[i] += cfv0_card[i];
                    }
                }
                for (
                    j,
                    &Combo {
                        cards: [h1, h2], ..
                    },
                ) in self.combos[1].iter().enumerate()
                {
                    if h1 != card && h2 != card {
                        cfv1[j] += cfv1_card[j];
                    }
                }
            }
        }

        for v in &mut cfv0 {
            *v /= divisor;
        }
        for v in &mut cfv1 {
            *v /= divisor;
        }
        [cfv0, cfv1]
    }

    /// One player (`x`)'s counterfactual value when `x` plays a perfect
    /// best response — for each of `x`'s own combos independently, the
    /// single action maximizing that combo's value, no mixing — against the
    /// *other* player's fixed `average_strategy` everywhere else in the
    /// tree. Structurally this is `evaluate_node` specialized to only ever
    /// report `x`'s own side: at a decision node belonging to the other
    /// player, it's exactly `evaluate_node`'s "other" (non-acting) output,
    /// since that's already an unweighted sum across actions (the
    /// opponent's `average_strategy` weighting is baked in via the
    /// `child_reach` scaling before recursing, same as there); at `x`'s own
    /// decision nodes, the analogous weighted-average is replaced with a
    /// per-combo `max` over the actions' returned values. `terminal_value`
    /// is reused unchanged (policy-agnostic, see `evaluate_node`'s doc
    /// comment) and `x`'s own reach is threaded down unmodified at `x`'s own
    /// nodes — `terminal_value` never reads a player's *own* reach for that
    /// player's own cfv (only the opponent's, to weight how likely the
    /// opponent was to actually play into this node — see
    /// `non_conflicting_reach_sum`'s callers), so unlike `evaluate_node`,
    /// which needs both sides' cfv and hence both sides' reach scaled
    /// correctly, this only ever needs the *non-deviating* player's reach
    /// threaded correctly.
    fn best_response_value(
        &self,
        policy: &Policy,
        node: &Node,
        x: usize,
        reach: &[Vec<f64>; 2],
        board: &[Card],
        board_eval: &[Vec<u32>; 2],
    ) -> Vec<f64> {
        match node {
            Node::Terminal(t) => {
                self.terminal_value_for_player(t, x, &reach[1 - x], board, board_eval)
            }
            Node::Chance { children } => {
                self.best_response_chance_value(policy, children, x, reach, board)
            }
            Node::Decision {
                id,
                player,
                children,
                ..
            } => {
                let p = player.index();
                if p == x {
                    let num_own_combos = self.combos[x].len();
                    let child_vals: Vec<Vec<f64>> = children
                        .par_iter()
                        .map(|child| {
                            self.best_response_value(policy, child, x, reach, board, board_eval)
                        })
                        .collect();
                    (0..num_own_combos)
                        .map(|i| {
                            child_vals
                                .iter()
                                .map(|v| v[i])
                                .fold(f64::NEG_INFINITY, f64::max)
                        })
                        .collect()
                } else {
                    let num_opp_combos = self.combos[p].len();
                    let strategy = self.average_strategy(policy, *id);
                    let child_vals: Vec<Vec<f64>> = children
                        .par_iter()
                        .enumerate()
                        .map(|(a, child)| {
                            let mut child_reach = reach.clone();
                            for i in 0..num_opp_combos {
                                child_reach[p][i] *= strategy[a][i];
                            }
                            self.best_response_value(
                                policy,
                                child,
                                x,
                                &child_reach,
                                board,
                                board_eval,
                            )
                        })
                        .collect();
                    let num_own_combos = self.combos[x].len();
                    let mut out = vec![0.0; num_own_combos];
                    for v in &child_vals {
                        for (i, val) in v.iter().enumerate() {
                            out[i] += val;
                        }
                    }
                    out
                }
            }
        }
    }

    /// `best_response_value`'s counterpart to `chance_value`/
    /// `evaluate_chance_value`: same suit-isomorphism relabeling (only the
    /// class representative is actually recursed into; other members reuse
    /// `relabel_cfv`), specialized to the single player `x` whose
    /// best-response value this whole pass tracks.
    fn best_response_chance_value(
        &self,
        policy: &Policy,
        children: &[(Vec<Card>, Node)],
        x: usize,
        reach: &[Vec<f64>; 2],
        board: &[Card],
    ) -> Vec<f64> {
        let n: usize = children.iter().map(|(members, _)| members.len()).sum();
        let divisor = (n as f64) - 4.0;

        let class_results: Vec<(&Vec<Card>, Vec<f64>)> = children
            .par_iter()
            .map(|(members, child)| {
                let rep = members[0];
                let mut child_board = board.to_vec();
                child_board.push(rep);
                let empty_eval = [Vec::new(), Vec::new()];
                let child_board_eval = self
                    .board_eval_cache
                    .get(&child_board)
                    .unwrap_or(&empty_eval);
                let child_val = self.best_response_value(
                    policy,
                    child,
                    x,
                    reach,
                    &child_board,
                    child_board_eval,
                );
                (members, child_val)
            })
            .collect();

        let mut out = vec![0.0; self.combos[x].len()];
        for (members, rep_val) in &class_results {
            let rep = members[0];
            for &card in members.iter() {
                let val_card = if card == rep {
                    rep_val.clone()
                } else {
                    self.relabel_cfv(x, rep_val, suit(rep), suit(card))
                };
                for (
                    i,
                    &Combo {
                        cards: [h1, h2], ..
                    },
                ) in self.combos[x].iter().enumerate()
                {
                    if h1 != card && h2 != card {
                        out[i] += val_card[i];
                    }
                }
            }
        }
        for v in &mut out {
            *v /= divisor;
        }
        out
    }

    /// Raw showdown win probability for each of `player`'s combos against
    /// the opponent's `reach`-weighted range at this node — only valid
    /// (and only ever called) once `board_eval` is a complete board's real
    /// `evaluate_7` scores. Excludes opponent combos blocked by the hero's
    /// own two cards *and* ones that physically conflict with the board
    /// (same exclusion `showdown_cfv` uses — without it, a phantom
    /// opponent combo could still be assigned nonzero weight here even
    /// though `reach` was never masked for it, since that masking only
    /// ever happens inside bucket-merge aggregation during training, not
    /// in this reach vector itself).
    fn equity_at_node(
        &self,
        player: usize,
        reach: &[Vec<f64>; 2],
        board: &[Card],
        board_eval: &[Vec<u32>; 2],
    ) -> Vec<f64> {
        let board_set = set_from_cards(board);
        let positive_reach: Vec<f64> = reach[1 - player]
            .iter()
            .map(|&r| if r <= 0.0 { 0.0 } else { r })
            .collect();
        let mass =
            self.terminal_index
                .compatible_mass(&self.combos, player, &positive_reach, board_set);
        let net = self.terminal_index.showdown(
            &self.combos,
            player,
            &positive_reach,
            board_eval,
            &self.board_order_cache[board],
            board_set,
            &[1.0, 1.0],
        );
        net.iter()
            .zip(mass)
            .map(|(net, mass)| {
                if mass > 0.0 {
                    0.5 + 0.5 * net / mass
                } else {
                    0.5
                }
            })
            .collect()
    }

    fn terminal_value(
        &self,
        t: &Terminal,
        reach: &[Vec<f64>; 2],
        board: &[Card],
        board_eval: &[Vec<u32>; 2],
    ) -> [Vec<f64>; 2] {
        [
            self.terminal_value_for_player(t, 0, &reach[1], board, board_eval),
            self.terminal_value_for_player(t, 1, &reach[0], board, board_eval),
        ]
    }

    fn terminal_value_for_player(
        &self,
        t: &Terminal,
        player: usize,
        reach_opp: &[f64],
        board: &[Card],
        board_eval: &[Vec<u32>; 2],
    ) -> Vec<f64> {
        #[cfg(test)]
        self.terminal_calls[player].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let board_set = set_from_cards(board);
        #[cfg(test)]
        if self
            .use_pairwise_terminals
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            return (0..self.combos[player].len())
                .map(|i| {
                    let [a, b] = self.combos[player][i].cards;
                    if set_contains(board_set, a) || set_contains(board_set, b) {
                        return 0.0;
                    }
                    match t {
                        Terminal::Fold { winner, invested } => {
                            let value = if winner.index() == player {
                                invested[1 - player]
                            } else {
                                -invested[player]
                            };
                            value * self.non_conflicting_reach_sum(player, i, reach_opp, board_set)
                        }
                        Terminal::Showdown { invested } => {
                            self.showdown_cfv(player, i, invested, reach_opp, board_eval, board_set)
                        }
                    }
                })
                .collect();
        }
        match t {
            Terminal::Fold { winner, invested } => {
                let value = if winner.index() == player {
                    invested[1 - player]
                } else {
                    -invested[player]
                };
                self.terminal_index
                    .compatible_mass(&self.combos, player, reach_opp, board_set)
                    .into_iter()
                    .map(|mass| value * mass)
                    .collect()
            }
            Terminal::Showdown { invested } => self.terminal_index.showdown(
                &self.combos,
                player,
                reach_opp,
                board_eval,
                &self.board_order_cache[board],
                board_set,
                invested,
            ),
        }
    }

    /// Sums `reach_opp` over the opponent's combos that are actually live
    /// against `player`'s own combo `combo_idx`: not blocked by `player`'s
    /// two hole cards, *and* not physically impossible given `board_set`
    /// (an opponent combo whose card was itself dealt onto the board this
    /// branch can never really be held here — see `chance_value`'s doc
    /// comment on the "N-4" private-deal normalization this complements. Without this
    /// check a board-conflicting opponent combo would still contribute its
    /// full reach weight to this terminal, breaking the zero-sum
    /// invariant).
    #[cfg(test)]
    fn non_conflicting_reach_sum(
        &self,
        player: usize,
        combo_idx: usize,
        reach_opp: &[f64],
        board_set: CardSet,
    ) -> f64 {
        let opp = 1 - player;
        let [h1, h2] = self.combos[player][combo_idx].cards;
        self.combos[opp]
            .iter()
            .zip(reach_opp)
            .filter(
                |(
                    Combo {
                        cards: [v1, v2], ..
                    },
                    _,
                )| {
                    *v1 != h1
                        && *v1 != h2
                        && *v2 != h1
                        && *v2 != h2
                        && !set_contains(board_set, *v1)
                        && !set_contains(board_set, *v2)
                },
            )
            .map(|(_, &r)| r)
            .sum()
    }

    /// Same board-conflict exclusion as `non_conflicting_reach_sum`,
    /// applied to the opponent combos aggregated over for a showdown value
    /// (see that function's doc comment).
    #[cfg(test)]
    fn showdown_cfv(
        &self,
        player: usize,
        combo_idx: usize,
        invested: &[f64; 2],
        reach_opp: &[f64],
        board_eval: &[Vec<u32>; 2],
        board_set: CardSet,
    ) -> f64 {
        let opp = 1 - player;
        let [h1, h2] = self.combos[player][combo_idx].cards;
        let hero_score = board_eval[player][combo_idx];
        let mut cfv = 0.0;
        for (
            j,
            &Combo {
                cards: [v1, v2], ..
            },
        ) in self.combos[opp].iter().enumerate()
        {
            if v1 == h1 || v1 == h2 || v2 == h1 || v2 == h2 {
                continue;
            }
            if set_contains(board_set, v1) || set_contains(board_set, v2) {
                continue;
            }
            let villain_score = board_eval[opp][j];
            let net = match hero_score.cmp(&villain_score) {
                std::cmp::Ordering::Greater => invested[opp],
                std::cmp::Ordering::Less => -invested[player],
                std::cmp::Ordering::Equal => (invested[opp] - invested[player]) / 2.0,
            };
            cfv += reach_opp[j] * net;
        }
        cfv
    }
}

/// `(min(c1,c2), max(c1,c2))`, the order-independent key `combo_index` and
/// `relabel_cfv` use to look up a combo regardless of the tuple order
/// `range::hand_combos` happened to store it in.
fn normalize_pair(c1: Card, c2: Card) -> (Card, Card) {
    if c1 <= c2 {
        (c1, c2)
    } else {
        (c2, c1)
    }
}

fn build_combo_index(combos: &[Combo]) -> HashMap<(Card, Card), usize> {
    combos
        .iter()
        .enumerate()
        .map(
            |(
                i,
                &Combo {
                    cards: [c1, c2], ..
                },
            )| (normalize_pair(c1, c2), i),
        )
        .collect()
}

/// `card` with its suit swapped to the other side of the `suit_a`/`suit_b`
/// transposition (identity if `card` uses neither suit).
fn swap_suit(card: Card, suit_a: u8, suit_b: u8) -> Card {
    let s = suit(card);
    let new_suit = if s == suit_a {
        suit_b
    } else if s == suit_b {
        suit_a
    } else {
        s
    };
    rank(card) * 4 + new_suit
}

/// `evaluate_7` scores for every combo, if `board` is a complete 5-card
/// board — otherwise empty (no `Terminal::Showdown` is ever reachable with
/// an incomplete board, so it's never read in that case).
///
/// A combo whose hole card was itself dealt as one of `board`'s later
/// (post-initial-deal) cards is physically impossible for this specific
/// board — `chance_value`'s per-node exclusion check always drops such a
/// combo from every aggregate sum before it reaches the root (that's the
/// whole point of the "N-4" private-deal normalization this module's doc comment
/// describes), so its score here is never actually read. Skipping
/// `evaluate_7` for it (sentinel `0`) avoids feeding it a 7-card hand that
/// can't occur in reality — e.g. a pocket-pair combo whose two ranks-match
/// a rank that also completes trips/quads on the board, which pushes one
/// rank to 5 "occurrences" among the 7 cards, a pattern `eval::evaluate_5`'s
/// lookup table correctly never populates (impossible with 4 suits/rank).
fn eval_combos_if_complete(combos: &[Vec<Combo>; 2], board: &[Card]) -> [Vec<u32>; 2] {
    if board.len() != 5 {
        return [Vec::new(), Vec::new()];
    }
    let board_set = set_from_cards(board);
    let score = |combo: &Combo| {
        if set_contains(board_set, combo.cards[0]) || set_contains(board_set, combo.cards[1]) {
            return 0;
        }
        evaluate_7([
            combo.cards[0],
            combo.cards[1],
            board[0],
            board[1],
            board[2],
            board[3],
            board[4],
        ])
    };
    [
        combos[0].iter().map(score).collect(),
        combos[1].iter().map(score).collect(),
    ]
}

/// Walks every `Chance` branch once, filling `cache` with the
/// `evaluate_7` scores for each complete board reachable from `board`.
fn precompute_board_eval(
    node: &Node,
    board: &[Card],
    combos: &[Vec<Combo>; 2],
    cache: &mut HashMap<Vec<Card>, [Vec<u32>; 2]>,
) {
    match node {
        Node::Decision { children, .. } => {
            for child in children {
                precompute_board_eval(child, board, combos, cache);
            }
        }
        Node::Chance { children } => {
            for (members, child) in children {
                // Only the class representative is actually recursed into
                // by `chance_value` — other members' values are derived by
                // relabeling, never by looking up their own board here.
                let rep = members[0];
                let mut child_board = board.to_vec();
                child_board.push(rep);
                cache
                    .entry(child_board.clone())
                    .or_insert_with(|| eval_combos_if_complete(combos, &child_board));
                precompute_board_eval(child, &child_board, combos, cache);
            }
        }
        Node::Terminal(_) => {}
    }
}

/// Every board a `Decision` node is ever actually built at, below `board`
/// (not including `board` itself — callers that need the starting board
/// too, e.g. a solve beginning already at the target street, add it
/// separately). Mirrors `precompute_board_eval`'s representative-only
/// descent through `Chance` nodes for the same reason: those are the only
/// boards `cfr()` ever recurses into.
fn collect_reachable_boards(
    node: &Node,
    board: &[Card],
    acc: &mut std::collections::HashSet<Vec<Card>>,
) {
    match node {
        Node::Decision { children, .. } => {
            for child in children {
                collect_reachable_boards(child, board, acc);
            }
        }
        Node::Chance { children } => {
            for (members, child) in children {
                let rep = members[0];
                let mut child_board = board.to_vec();
                child_board.push(rep);
                acc.insert(child_board.clone());
                collect_reachable_boards(child, &child_board, acc);
            }
        }
        Node::Terminal(_) => {}
    }
}

/// Splits combos into `num_buckets` contiguous, roughly-equal-sized groups
/// by score percentile (river: real `evaluate_7` scores; flop/turn:
/// `avg_score`'s future-score expectation) — sorting by score and cutting
/// every `n / num_buckets`-th combo keeps the assignment monotonic (a
/// higher-scoring combo never lands in a lower bucket than a
/// lower-scoring one). When `num_buckets >= scores.len()` this reduces to
/// an exact bijection (every combo its own bucket, in score order) — the
/// baseline `CardAbstraction`'s per-street `Some(n)` for large `n` needs
/// to behave identically to `None`.
fn compute_percentile_buckets(scores: &[f64], num_buckets: usize) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let num_buckets = num_buckets.clamp(1, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap());
    let mut bucket_of = vec![0usize; n];
    for (rank, &i) in order.iter().enumerate() {
        bucket_of[i] = rank * num_buckets / n;
        // Never split a run of exactly-equal scores across a bucket
        // boundary: two combos this function scores identically are often
        // suit-swap-equivalent under `avg_score`'s isomorphism-class
        // averaging (or the river's raw `evaluate_7` tie), and
        // `chance_value`'s relabeling assumes such combos always share
        // the exact same strategy — a boundary landing mid-tie would
        // silently break that invariant.
        if rank > 0 && scores[i] == scores[order[rank - 1]] {
            bucket_of[i] = bucket_of[order[rank - 1]];
        }
    }
    bucket_of
}

/// Each combo's expected `evaluate_7` score from `board` through to the
/// river, averaged over future streets — the proxy `CardAbstraction`'s
/// flop/turn bucketing ranks combos by, since (unlike the river) their
/// hand isn't final yet. This is coarser than true equity: it's a plain
/// average of the eventual showdown *score*, not a win probability against
/// an opponent range, so it doesn't capture *how* a hand realizes its
/// value (e.g. a made hand with no more equity to gain and a drawing hand
/// with the same average score play very differently in reality) — but it
/// reuses the exact `evaluate_7` machinery already computed for real
/// showdowns, needs no separate equity engine, and preserves the right
/// ordering for the common case (stronger hands now tend to stay stronger).
///
/// Always groups future deals by suit-isomorphism class (`deal_classes`
/// with `suit_isomorphism: true`) regardless of whether the *tree* itself
/// was built with isomorphism — grouping same-rank free-suit cards is
/// always a valid equivalence, so this one-time precomputation is cheaper
/// this way, and when the tree *was* isomorphism-reduced this also means
/// every board it visits at the base case is already in `board_eval_cache`
/// (zero redundant `evaluate_7` calls beyond what training needs anyway).
fn avg_score(
    board: &[Card],
    combos: &[Vec<Combo>; 2],
    combo_index: &[HashMap<(Card, Card), usize>; 2],
    board_eval_cache: &HashMap<Vec<Card>, [Vec<u32>; 2]>,
    cache: &mut HashMap<Vec<Card>, [Vec<f64>; 2]>,
) -> [Vec<f64>; 2] {
    if let Some(v) = cache.get(board) {
        return v.clone();
    }
    let result = if board.len() == 5 {
        let scores = match board_eval_cache.get(board) {
            Some(scores) => scores.clone(),
            None => eval_combos_if_complete(combos, board),
        };
        [
            scores[0].iter().map(|&s| s as f64).collect(),
            scores[1].iter().map(|&s| s as f64).collect(),
        ]
    } else {
        let mut sum = [vec![0.0; combos[0].len()], vec![0.0; combos[1].len()]];
        let mut weight = [vec![0.0; combos[0].len()], vec![0.0; combos[1].len()]];
        for members in super::tree::deal_classes(true, board) {
            let rep = members[0];
            let mut next_board = board.to_vec();
            next_board.push(rep);
            // `child[p][i]` is the real (or further-averaged) score for
            // combo `i` under board+`rep` specifically. For any *other*
            // class member `m`, a fixed combo's score under board+`m`
            // isn't `child[p][i]` (the two boards aren't the same!) — it's
            // `child`'s value for whichever combo `i` becomes after
            // swapping `rep`'s suit with `m`'s (the same relabeling
            // `chance_value`/`relabel_cfv` uses, for the same reason: that
            // suit swap is the symmetry that maps board+`m` to board+`rep`).
            let child = avg_score(&next_board, combos, combo_index, board_eval_cache, cache);
            for &m in &members {
                let (suit_a, suit_b) = (suit(rep), suit(m));
                for p in 0..2 {
                    for (
                        i,
                        &Combo {
                            cards: [h1, h2], ..
                        },
                    ) in combos[p].iter().enumerate()
                    {
                        if h1 == m || h2 == m {
                            continue; // this combo can't coexist with m being dealt
                        }
                        let idx = if m == rep {
                            i
                        } else {
                            let s1 = swap_suit(h1, suit_a, suit_b);
                            let s2 = swap_suit(h2, suit_a, suit_b);
                            combo_index[p][&normalize_pair(s1, s2)]
                        };
                        sum[p][i] += child[p][idx];
                        weight[p][i] += 1.0;
                    }
                }
            }
        }
        [
            sum[0]
                .iter()
                .zip(&weight[0])
                .map(|(&s, &w)| if w > 0.0 { s / w } else { 0.0 })
                .collect(),
            sum[1]
                .iter()
                .zip(&weight[1])
                .map(|(&s, &w)| if w > 0.0 { s / w } else { 0.0 })
                .collect(),
        ]
    };
    cache.insert(board.to_vec(), result.clone());
    result
}

/// `player`'s bucket count and combo->bucket map at `board`: the
/// precomputed river map if one exists there, else the identity fallback
/// (every combo its own bucket — non-river nodes, or `river_buckets: None`).
fn bucket_count_and_map<'a>(
    player: usize,
    board: &[Card],
    bucket_maps: &'a HashMap<Vec<Card>, [Vec<usize>; 2]>,
    identity_buckets: &'a [Vec<usize>; 2],
) -> (usize, &'a [usize]) {
    match bucket_maps.get(board) {
        Some(maps) => {
            let map = &maps[player];
            let num_buckets = map.iter().copied().max().map_or(0, |m| m + 1);
            (num_buckets, map)
        }
        None => (identity_buckets[player].len(), &identity_buckets[player]),
    }
}

/// Walks the tree threading `board` (mirroring `precompute_board_eval`'s
/// representative-only descent through `Chance` nodes) to size each
/// decision node's regret/strategy tables by its actual bucket count and
/// record `(player, board)` in `node_meta` for `average_strategy` to use.
fn init_node_data(
    node: &Node,
    board: &[Card],
    bucket_maps: &HashMap<Vec<Card>, [Vec<usize>; 2]>,
    identity_buckets: &[Vec<usize>; 2],
    out: &mut [NodeData],
    node_meta: &mut [(usize, Vec<Card>)],
) {
    match node {
        Node::Decision {
            id,
            player,
            actions,
            children,
        } => {
            let p = player.index();
            let (num_buckets, _) = bucket_count_and_map(p, board, bucket_maps, identity_buckets);
            out[*id] = NodeData::sized(actions.len(), num_buckets);
            node_meta[*id] = (p, board.to_vec());
            for child in children {
                init_node_data(child, board, bucket_maps, identity_buckets, out, node_meta);
            }
        }
        Node::Chance { children } => {
            for (members, child) in children {
                let rep = members[0];
                let mut child_board = board.to_vec();
                child_board.push(rep);
                init_node_data(
                    child,
                    &child_board,
                    bucket_maps,
                    identity_buckets,
                    out,
                    node_meta,
                );
            }
        }
        Node::Terminal(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card_from_str;
    use crate::range::Range;
    use crate::solver::tree::{build_tree, GameConfig};
    use crate::solver::Chips;

    #[test]
    fn dcfr_discounts_once_per_bucket_after_adding_the_full_iteration() {
        use crate::solver::tree::{Action, Player, Tree};
        let c = |s| card_from_str(s).unwrap();
        let board = ["2c", "3d", "7h", "8s", "9c"].map(c).to_vec();
        // Both AA combos have the same strength and merge into one bucket.
        // Unequal reaches expose mistakenly discounting after each combo.
        let combos = [
            vec![
                Combo::new(c("Ac"), c("Ad"), 1.0),
                Combo::new(c("Ah"), c("As"), 3.0),
            ],
            vec![Combo::new(c("Kh"), c("Kd"), 1.0)],
        ];
        let tree = Tree {
            root: Node::Decision {
                id: 0,
                player: Player::Oop,
                actions: vec![Action::Check, Action::Bet(Chips::whole(8))],
                children: vec![
                    Node::Terminal(Terminal::Fold {
                        winner: Player::Oop,
                        invested: [8.0, 8.0],
                    }),
                    Node::Terminal(Terminal::Fold {
                        winner: Player::Ip,
                        invested: [8.0, 8.0],
                    }),
                ],
            },
            num_decision_nodes: 1,
        };
        let mut trainer = Trainer::new(
            tree,
            board,
            combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::Dcfr,
                abstraction: CardAbstraction {
                    river: Some(1),
                    ..CardAbstraction::default()
                },
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        assert_eq!(trainer.train(1), [0.0, 0.0]);
        {
            let data = trainer.node_data[0].lock().unwrap();
            // Uniform strategy gives instantaneous regret +/-8. At t=1
            // both signs discount by 1/2, and total strategy mass 4 by 1/4.
            assert_eq!(data.regret_sum, vec![vec![4.0], vec![-4.0]]);
            assert_eq!(data.strategy_sum, vec![vec![0.5], vec![0.5]]);
        }
        assert_eq!(trainer.train(1), [32.0, -32.0]);
        {
            let data = trainer.node_data[0].lock().unwrap();
            let positive = 2.0_f64.powf(1.5);
            assert_eq!(data.regret_sum[0][0], 4.0 * positive / (positive + 1.0));
            // The action now always loses 16 relative to the current action;
            // both the old -4 and new -16 are discounted together.
            assert_eq!(data.regret_sum[1][0], -10.0);
            assert_eq!(data.strategy_sum[0][0], 2.0);
            assert_eq!(data.strategy_sum[1][0], 2.0 / 9.0);
        }
        // Iteration weights are 1^2:2^2, so (.5+4)/5 = .9.
        for probability in &trainer.average_strategy(0)[0] {
            assert!((probability - 0.9).abs() < 1e-15);
        }
    }

    #[test]
    fn chunking_preserves_all_update_state_through_chance_nodes() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            let c = |s| card_from_str(s).unwrap();
            let cfg = GameConfig {
                board: ["2c", "7d", "9h", "Js"].map(c).to_vec(),
                pot: Chips::whole(2),
                effective_stack: Chips::whole(1),
                bet_sizes: vec![0.5],
                max_raises: 0,
                suit_isomorphism: false,
            };
            let combos = [
                vec![
                    Combo::new(c("Jh"), c("Jd"), 0.7),
                    Combo::new(c("Ac"), c("Kd"), 1.0),
                ],
                vec![Combo::new(c("7h"), c("6s"), 1.0)],
            ];
            for (algorithm, schedule) in [
                SolverAlgorithm::Cfr,
                SolverAlgorithm::CfrPlus,
                SolverAlgorithm::Dcfr,
            ]
            .into_iter()
            .flat_map(|a| {
                [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating].map(|s| (a, s))
            }) {
                let make = || {
                    Trainer::new(
                        build_tree(&cfg).unwrap(),
                        cfg.board.clone(),
                        combos.clone(),
                        TrainerOptions {
                            algorithm,
                            schedule,
                            abstraction: CardAbstraction {
                                river: Some(2),
                                ..CardAbstraction::default()
                            },
                        },
                    )
                    .unwrap()
                };
                let mut whole = make();
                let mut chunks = make();
                let expected = whole.train(11);
                chunks.train(3);
                chunks.train(0);
                assert_eq!(chunks.train(8), expected, "{algorithm:?}");
                assert_eq!(whole.iterations_completed(), 11);
                assert_eq!(chunks.iterations_completed(), 11);
                for (a, b) in whole.node_data.iter().zip(&chunks.node_data) {
                    let a = a.lock().unwrap();
                    let b = b.lock().unwrap();
                    assert_eq!(a.regret_sum, b.regret_sum, "{algorithm:?}");
                    assert_eq!(a.strategy_sum, b.strategy_sum, "{algorithm:?}");
                }
            }
        });
    }

    fn river_board() -> Vec<Card> {
        ["2h", "7c", "9d", "Jc", "4s"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect()
    }

    struct Scenario {
        tree: super::super::tree::Tree,
        trainer_combos: [Vec<Combo>; 2],
        board: Vec<Card>,
        pot: f64,
        bet: f64,
    }

    fn build_scenario() -> Scenario {
        let board = river_board();
        let dead = crate::card::set_from_cards(&board);
        // OOP: JJ (value, always beats IP's 99) + KQo (air, always loses to IP's 99).
        let oop_range = Range::parse("JJ,KQo").unwrap();
        // IP: 99 (bluffcatcher — beats KQo, loses to JJ).
        let ip_range = Range::parse("99").unwrap();
        let oop_combos = oop_range.combos(dead);
        let ip_combos = ip_range.combos(dead);

        let pot = 100.0;
        let bet = 100.0; // pot-size bet
        let cfg = GameConfig {
            board: board.clone(),
            pot: Chips::from_f64(pot).unwrap(),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![bet / pot],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let tree = build_tree(&cfg).unwrap();
        Scenario {
            tree,
            trainer_combos: [oop_combos, ip_combos],
            board,
            pot,
            bet,
        }
    }

    fn is_jj(c1: Card, c2: Card) -> bool {
        crate::card::rank(c1) == 9 && crate::card::rank(c2) == 9 // J = rank index 9
    }

    #[test]
    fn zero_sum_at_root() {
        let scenario = build_scenario();
        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        let [ev0, ev1] = trainer.train(20_000);
        assert!(
            (ev0 + ev1).abs() < 1.0,
            "expected zero-sum root EV, got ev0={ev0}, ev1={ev1}"
        );
    }

    #[test]
    fn converges_to_known_bet_and_call_frequencies() {
        let scenario = build_scenario();
        let value_combo_count = scenario.trainer_combos[0]
            .iter()
            .filter(
                |&&Combo {
                     cards: [c1, c2], ..
                 }| is_jj(c1, c2),
            )
            .count();
        let bluff_combo_count = scenario.trainer_combos[0].len() - value_combo_count;
        let wv = value_combo_count as f64;
        let wb = bluff_combo_count as f64;
        let pot = scenario.pot;
        let bet = scenario.bet;

        let root_id = match &scenario.tree.root {
            Node::Decision { id, .. } => *id,
            _ => panic!("expected decision"),
        };
        let ip_facing_bet_id = match &scenario.tree.root {
            Node::Decision { children, .. } => match &children[1] {
                Node::Decision { id, .. } => *id,
                _ => panic!("expected IP decision facing the bet"),
            },
            _ => panic!("expected decision"),
        };

        let combos0 = scenario.trainer_combos[0].clone();
        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer.train(60_000);

        let root_strategy = trainer.average_strategy(root_id); // [action][combo], action 1 = Bet
        let combos0 = &combos0;
        let bet_freq_value: Vec<f64> = combos0
            .iter()
            .enumerate()
            .filter(
                |(
                    _,
                    &Combo {
                        cards: [c1, c2], ..
                    },
                )| is_jj(c1, c2),
            )
            .map(|(i, _)| root_strategy[1][i])
            .collect();
        let bet_freq_bluff: Vec<f64> = combos0
            .iter()
            .enumerate()
            .filter(
                |(
                    _,
                    &Combo {
                        cards: [c1, c2], ..
                    },
                )| !is_jj(c1, c2),
            )
            .map(|(i, _)| root_strategy[1][i])
            .collect();

        let tol = 0.05;
        for f in &bet_freq_value {
            assert!(
                (f - 1.0).abs() < tol,
                "value combo should always bet, got {f}"
            );
        }
        let expected_beta = (bet / (pot + bet)) * (wv / wb);
        for f in &bet_freq_bluff {
            assert!(
                (f - expected_beta).abs() < tol,
                "expected bluff bet freq {expected_beta}, got {f}"
            );
        }

        let ip_strategy = trainer.average_strategy(ip_facing_bet_id); // action 1 = Call
        let expected_call = pot / (pot + bet); // MDF
        for &f in &ip_strategy[1] {
            assert!(
                (f - expected_call).abs() < tol,
                "expected call freq (MDF) {expected_call}, got {f}"
            );
        }
    }

    // --- Multi-street (turn -> chance node deals the river -> river betting) ---
    //
    // Board starts on the turn (4 cards); a single chance node deals all 48
    // possible river cards. The hands below were checked to keep the same win/lose
    // ordering across *every* possible river card.
    //
    // Unlike the single-street scenario, reaching the river here still
    // leaves a full betting round before showdown, so the single-street closed-form
    // bluff-frequency formula (which assumed an immediate showdown) no
    // longer applies exactly — a genuine multi-street equilibrium can bluff
    // at a different rate once the value of retaining bluffs into a further
    // street of betting is accounted for. Empirically, two of the three
    // single-street relationships still hold almost
    // exactly even with a full river betting round: value bets ~100% of the
    // time, and IP's calling frequency still matches the pot-odds MDF
    // formula. Only the bluffing frequency itself shifts. So this test
    // checks those two robust relationships precisely, and checks the
    // bluffing frequency only qualitatively (genuine interior mixing, and
    // identical across combos of the same hand type).
    fn turn_board() -> Vec<Card> {
        ["2c", "7d", "9h", "Js"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect()
    }

    struct MultiStreetScenario {
        tree: super::super::tree::Tree,
        trainer_combos: [Vec<Combo>; 2],
        board: Vec<Card>,
        pot: f64,
        bet: f64,
    }

    /// `Range::combos()` iterates a `HashMap<String, f64>` internally, whose
    /// element order isn't guaranteed to match between two independently
    /// parsed `Range`s (see the suit-isomorphism equivalence test's doc
    /// comment for the same pitfall) — build these once and share/clone
    /// them across trainers being compared, rather than calling
    /// `build_multi_street_scenario()` (which re-parses fresh) more than once.
    fn multi_street_scenario_combos() -> [Vec<Combo>; 2] {
        let dead = crate::card::set_from_cards(&turn_board());
        // OOP: JJ (value, always beats IP's 76o on any river) + 54o (air, always loses).
        let oop_combos = Range::parse("JJ,54o").unwrap().combos(dead);
        // IP: 76o (bluffcatcher — always beats 54o, always loses to JJ, on any river).
        let ip_combos = Range::parse("76o").unwrap().combos(dead);
        [oop_combos, ip_combos]
    }

    fn build_multi_street_scenario() -> MultiStreetScenario {
        build_multi_street_scenario_with_combos(multi_street_scenario_combos())
    }

    fn build_multi_street_scenario_with_combos(
        trainer_combos: [Vec<Combo>; 2],
    ) -> MultiStreetScenario {
        let board = turn_board();
        let pot = 100.0;
        let bet = 100.0;
        let cfg = GameConfig {
            board: board.clone(),
            pot: Chips::from_f64(pot).unwrap(),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![bet / pot],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let tree = build_tree(&cfg).unwrap();
        MultiStreetScenario {
            tree,
            trainer_combos,
            board,
            pot,
            bet,
        }
    }

    fn is_pocket_jacks(c1: Card, c2: Card) -> bool {
        crate::card::rank(c1) == 9 && crate::card::rank(c2) == 9
    }

    #[test]
    fn multi_street_zero_sum_at_root() {
        let scenario = build_multi_street_scenario();
        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        let [ev0, ev1] = trainer.train(500);
        assert!(
            (ev0 + ev1).abs() < 1.0,
            "expected zero-sum root EV, got ev0={ev0}, ev1={ev1}"
        );
    }

    #[test]
    fn multi_street_value_bets_and_ip_defends_at_mdf() {
        let scenario = build_multi_street_scenario();
        let pot = scenario.pot;
        let bet = scenario.bet;

        let root_id = match &scenario.tree.root {
            Node::Decision { id, .. } => *id,
            _ => panic!("expected decision"),
        };
        let ip_facing_bet_id = match &scenario.tree.root {
            Node::Decision { children, .. } => match &children[1] {
                Node::Decision { id, .. } => *id,
                _ => panic!("expected IP decision facing the bet"),
            },
            _ => panic!("expected decision"),
        };

        let combos0 = scenario.trainer_combos[0].clone();
        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer.train(4_000);

        let root_strategy = trainer.average_strategy(root_id); // [action][combo], action 1 = Bet
        let bet_freq_value: Vec<f64> = combos0
            .iter()
            .enumerate()
            .filter(
                |(
                    _,
                    &Combo {
                        cards: [c1, c2], ..
                    },
                )| is_pocket_jacks(c1, c2),
            )
            .map(|(i, _)| root_strategy[1][i])
            .collect();
        let bet_freq_bluff: Vec<f64> = combos0
            .iter()
            .enumerate()
            .filter(
                |(
                    _,
                    &Combo {
                        cards: [c1, c2], ..
                    },
                )| !is_pocket_jacks(c1, c2),
            )
            .map(|(i, _)| root_strategy[1][i])
            .collect();

        let tol = 0.08;
        // Slightly looser: pure-value combos should approach betting 100%
        // of the time, but CFR+'s average strategy converges at O(1/sqrt(T))
        // and this scenario's tree got noticeably bigger once the
        // board-conflict exclusion fix (see the cfr.rs module doc)
        // stopped a phantom-combo bug from artificially accelerating early
        // convergence, so this specific check needs a bit more slack at a
        // fixed, debug-build-friendly iteration count.
        let value_bet_tol = 0.1;
        for f in &bet_freq_value {
            assert!(
                (f - 1.0).abs() < value_bet_tol,
                "value combo should always bet, got {f}"
            );
        }

        // Bluffing frequency: checked only qualitatively (see comment above)
        // — genuine interior mixing, not degenerate, and identical across
        // combos of the same hand type (verifies symmetric combos are
        // treated symmetrically by the chance-node averaging).
        let bluff_avg = bet_freq_bluff.iter().sum::<f64>() / bet_freq_bluff.len() as f64;
        assert!(
            (0.01..0.9).contains(&bluff_avg),
            "expected genuine interior bluff mixing, got {bluff_avg}"
        );
        for f in &bet_freq_bluff {
            assert!((f - bluff_avg).abs() < 0.02, "combos of the same hand type should bet at the same frequency, got {f} vs avg {bluff_avg}");
        }
        assert!(
            bluff_avg < bet_freq_value[0],
            "bluffs should bet less often than value, got bluff={bluff_avg}, value={}",
            bet_freq_value[0]
        );

        let ip_strategy = trainer.average_strategy(ip_facing_bet_id); // action 1 = Call
        let expected_call = pot / (pot + bet); // MDF
        for &f in &ip_strategy[1] {
            assert!(
                (f - expected_call).abs() < tol,
                "expected call freq (MDF) {expected_call}, got {f}"
            );
        }
    }

    // --- Suit isomorphism: reduced (chance-node classes + relabeling)
    // must converge to the same equilibrium as unreduced (one child per
    // card), since it's purely a computational grouping and shouldn't
    // change the underlying extensive-form game at all.

    fn two_suit_turn_board() -> Vec<Card> {
        // Only clubs and diamonds on board -> hearts and spades are both
        // free, so river deals actually get grouped into classes of size
        // up to 2 (unlike `turn_board()`, which is rainbow and gets no
        // isomorphism reduction at all).
        ["2c", "7d", "9c", "Jd"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect()
    }

    /// `Range::combos()` iterates a `HashMap<String, f64>` internally, whose
    /// element order isn't guaranteed to match between two independently
    /// parsed `Range`s (std `HashMap`'s per-instance hasher keys differ) —
    /// so building the reduced/unreduced scenarios from two separate
    /// `Range::parse` calls would silently compare unrelated combos
    /// position-wise. Build the combo lists once and share them so "combo
    /// index i" means the same physical combo in both trainers.
    fn build_two_suit_scenario(
        suit_isomorphism: bool,
        trainer_combos: [Vec<Combo>; 2],
    ) -> MultiStreetScenario {
        let board = two_suit_turn_board();
        let pot = 100.0;
        let bet = 100.0;
        let cfg = GameConfig {
            board: board.clone(),
            pot: Chips::from_f64(pot).unwrap(),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![bet / pot],
            max_raises: 0,
            suit_isomorphism,
        };
        let tree = build_tree(&cfg).unwrap();
        MultiStreetScenario {
            tree,
            trainer_combos,
            board,
            pot,
            bet,
        }
    }

    fn two_suit_scenario_combos() -> [Vec<Combo>; 2] {
        let dead = crate::card::set_from_cards(&two_suit_turn_board());
        let oop_combos = Range::parse("JJ,54o").unwrap().combos(dead);
        let ip_combos = Range::parse("76o").unwrap().combos(dead);
        [oop_combos, ip_combos]
    }

    #[test]
    fn suit_isomorphism_actually_reduces_the_tree() {
        let combos = two_suit_scenario_combos();
        let reduced = build_two_suit_scenario(true, combos.clone());
        let unreduced = build_two_suit_scenario(false, combos);
        assert!(
            reduced.tree.num_decision_nodes < unreduced.tree.num_decision_nodes,
            "expected fewer decision nodes with isomorphism enabled, got {} (reduced) vs {} (unreduced)",
            reduced.tree.num_decision_nodes,
            unreduced.tree.num_decision_nodes
        );
    }

    /// Node ids are allocated in preorder during `build()`, so the "check"
    /// branch (built before the "bet" branch) consumes a different number
    /// of ids depending on how much the chance node inside it is reduced —
    /// the same numeric id can point at unrelated nodes in the reduced vs.
    /// unreduced trees. So ids must be looked up per-tree, not shared.
    fn root_and_ip_facing_bet_ids(root: &Node) -> (usize, usize) {
        let Node::Decision {
            id: root_id,
            children,
            ..
        } = root
        else {
            panic!("expected decision")
        };
        let Node::Decision {
            id: ip_facing_bet_id,
            ..
        } = &children[1]
        else {
            panic!("expected IP decision facing the bet")
        };
        (*root_id, *ip_facing_bet_id)
    }

    #[test]
    fn suit_isomorphism_reduced_and_unreduced_converge_to_the_same_equilibrium() {
        let combos = two_suit_scenario_combos();
        let reduced = build_two_suit_scenario(true, combos.clone());
        let unreduced = build_two_suit_scenario(false, combos);

        let (reduced_root_id, reduced_ip_facing_bet_id) =
            root_and_ip_facing_bet_ids(&reduced.tree.root);
        let (unreduced_root_id, unreduced_ip_facing_bet_id) =
            root_and_ip_facing_bet_ids(&unreduced.tree.root);

        let iterations = 3_000;
        let mut reduced_trainer = Trainer::new(
            reduced.tree,
            reduced.board,
            reduced.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        reduced_trainer.train(iterations);
        let mut unreduced_trainer = Trainer::new(
            unreduced.tree,
            unreduced.board,
            unreduced.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        unreduced_trainer.train(iterations);

        let tol = 0.05;
        for (row_r, row_u) in reduced_trainer
            .average_strategy(reduced_root_id)
            .iter()
            .zip(unreduced_trainer.average_strategy(unreduced_root_id).iter())
        {
            for (&fr, &fu) in row_r.iter().zip(row_u.iter()) {
                assert!(
                    (fr - fu).abs() < tol,
                    "root strategy mismatch: reduced={fr}, unreduced={fu}"
                );
            }
        }
        for (row_r, row_u) in reduced_trainer
            .average_strategy(reduced_ip_facing_bet_id)
            .iter()
            .zip(
                unreduced_trainer
                    .average_strategy(unreduced_ip_facing_bet_id)
                    .iter(),
            )
        {
            for (&fr, &fu) in row_r.iter().zip(row_u.iter()) {
                assert!(
                    (fr - fu).abs() < tol,
                    "IP-facing-bet strategy mismatch: reduced={fr}, unreduced={fu}"
                );
            }
        }
    }

    // --- River card abstraction (bucketing) ---

    #[test]
    fn compute_percentile_buckets_is_monotonic_and_bijective_when_large_enough() {
        let scores = [30.0, 10.0, 20.0, 10.0, 40.0];
        let n = scores.len();

        // A bucket count >= the combo count must be an exact bijection
        // *over distinct scores*: every combo its own bucket, in score
        // order, except that combos 1 and 3 (tied at 10.0) must always
        // share a bucket (see `compute_percentile_buckets_never_splits_a_tie`)
        // — so 4 distinct buckets for 5 combos here, not 5.
        let bijective = compute_percentile_buckets(&scores, 100);
        assert_eq!(
            bijective[1], bijective[3],
            "tied scores must share a bucket, got {bijective:?}"
        );
        let mut seen = std::collections::HashSet::new();
        for (i, &b) in bijective.iter().enumerate() {
            if i == 3 {
                continue; // shares combo 1's bucket, already recorded
            }
            assert!(
                seen.insert(b),
                "expected every distinct-score combo to get a distinct bucket, got {bijective:?}"
            );
        }
        // Monotonic: a strictly higher score never lands in a lower bucket.
        for i in 0..n {
            for j in 0..n {
                if scores[i] < scores[j] {
                    assert!(
                        bijective[i] <= bijective[j],
                        "score {} (bucket {}) should not outrank score {} (bucket {})",
                        scores[i],
                        bijective[i],
                        scores[j],
                        bijective[j]
                    );
                }
            }
        }

        // A small bucket count must actually merge combos (fewer distinct
        // buckets used than combos) while staying monotonic and in range.
        let reduced = compute_percentile_buckets(&scores, 2);
        let distinct: std::collections::HashSet<usize> = reduced.iter().copied().collect();
        assert!(
            distinct.len() <= 2,
            "expected at most 2 distinct buckets, got {distinct:?}"
        );
        assert!(reduced.iter().all(|&b| b < 2));
        for i in 0..n {
            for j in 0..n {
                if scores[i] < scores[j] {
                    assert!(reduced[i] <= reduced[j]);
                }
            }
        }
    }

    /// `river_buckets: Some(n)` with `n` at least as large as either
    /// player's combo count must be a bijection (see
    /// `compute_percentile_buckets_is_monotonic_and_bijective_when_large_enough`),
    /// so it must converge to the same equilibrium as `None` — this is the
    /// same kind of equivalence check as suit isomorphism's, and also
    /// exercises the board_eval_cache fix for a solve that starts already
    /// at a complete river board (no `Chance` node at all, so
    /// `precompute_board_eval` alone would never seed a bucket map for it).
    #[test]
    fn river_buckets_bijective_matches_unbucketed() {
        let board = river_board();
        let dead = crate::card::set_from_cards(&board);
        let oop_combos = Range::parse("JJ,KQo").unwrap().combos(dead);
        let ip_combos = Range::parse("99").unwrap().combos(dead);
        let pot = 100.0;
        let bet = 100.0;
        let cfg = GameConfig {
            board: board.clone(),
            pot: Chips::from_f64(pot).unwrap(),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![bet / pot],
            max_raises: 0,
            suit_isomorphism: false,
        };

        let tree_none = build_tree(&cfg).unwrap();
        let tree_bucketed = build_tree(&cfg).unwrap();
        let (root_id, ip_facing_bet_id) = root_and_ip_facing_bet_ids(&tree_none.root);

        let iterations = 3_000;
        let mut trainer_none = Trainer::new(
            tree_none,
            board.clone(),
            [oop_combos.clone(), ip_combos.clone()],
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer_none.train(iterations);
        // 100 exceeds both players' combo counts here, so this is a bijection on both sides.
        let mut trainer_bucketed = Trainer::new(
            tree_bucketed,
            board,
            [oop_combos, ip_combos],
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                abstraction: CardAbstraction {
                    river: Some(100),
                    ..Default::default()
                },
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer_bucketed.train(iterations);

        let tol = 0.05;
        for (row_n, row_b) in trainer_none
            .average_strategy(root_id)
            .iter()
            .zip(trainer_bucketed.average_strategy(root_id).iter())
        {
            for (&fn_, &fb) in row_n.iter().zip(row_b.iter()) {
                assert!(
                    (fn_ - fb).abs() < tol,
                    "root strategy mismatch: none={fn_}, bucketed(bijective)={fb}"
                );
            }
        }
        for (row_n, row_b) in trainer_none
            .average_strategy(ip_facing_bet_id)
            .iter()
            .zip(trainer_bucketed.average_strategy(ip_facing_bet_id).iter())
        {
            for (&fn_, &fb) in row_n.iter().zip(row_b.iter()) {
                assert!(
                    (fn_ - fb).abs() < tol,
                    "IP-facing-bet strategy mismatch: none={fn_}, bucketed(bijective)={fb}"
                );
            }
        }
    }

    /// With a genuinely small bucket count, exact per-combo behavior isn't
    /// expected (that's the whole point of abstraction), but the trained
    /// strategy should still be sane: zero-sum at the root, and betting
    /// frequency should respect real hand-strength ordering across three
    /// clearly-separated tiers (trips > overpair > air) sharing only 3
    /// buckets between them.
    #[test]
    fn river_buckets_reduced_stays_zero_sum_and_respects_hand_strength_order() {
        let board = river_board(); // 2h 7c 9d Jc 4s
        let dead = crate::card::set_from_cards(&board);
        // OOP: JJ (trips jacks - top tier), QQ (overpair - middle tier),
        // KQo+72o (air, kicker-only - bottom tier).
        let oop_range = Range::parse("JJ,QQ,KQo,72o").unwrap();
        // IP: 99 (trips nines, beats everything but JJ) + 55 (a small
        // pocket pair unconnected to the board, beaten by QQ but ahead of
        // pure air). Giving IP a *range* instead of a single fixed hand
        // matters here: with IP holding only 99, QQ and air would both
        // simply always lose to IP's one possible hand, making them
        // strategically interchangeable "always-losing" bluff candidates
        // with no real reason for one to bet more than the other — 55
        // gives QQ genuine partial equity (it beats 55, loses to 99) so it
        // actually is a distinct middle tier.
        let ip_range = Range::parse("99,55").unwrap();
        let oop_combos = oop_range.combos(dead);
        let ip_combos = ip_range.combos(dead);

        let pot = 100.0;
        let bet = 100.0;
        let cfg = GameConfig {
            board: board.clone(),
            pot: Chips::from_f64(pot).unwrap(),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![bet / pot],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let tree = build_tree(&cfg).unwrap();
        let (root_id, _) = root_and_ip_facing_bet_ids(&tree.root);

        let combos0 = oop_combos.clone();
        let mut trainer = Trainer::new(
            tree,
            board,
            [oop_combos, ip_combos],
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                abstraction: CardAbstraction {
                    river: Some(3),
                    ..Default::default()
                },
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        let [ev0, ev1] = trainer.train(4_000);
        assert!(
            (ev0 + ev1).abs() < 1.0,
            "expected zero-sum root EV even under abstraction, got ev0={ev0}, ev1={ev1}"
        );

        let is_jj = |c1: Card, c2: Card| crate::card::rank(c1) == 9 && crate::card::rank(c2) == 9;
        let is_qq = |c1: Card, c2: Card| crate::card::rank(c1) == 10 && crate::card::rank(c2) == 10;
        let avg = |pred: &dyn Fn(Card, Card) -> bool, strategy: &[Vec<f64>]| -> f64 {
            let matching: Vec<f64> = combos0
                .iter()
                .enumerate()
                .filter(
                    |(
                        _,
                        &Combo {
                            cards: [c1, c2], ..
                        },
                    )| pred(c1, c2),
                )
                .map(|(i, _)| strategy[1][i]) // action 1 = Bet
                .collect();
            matching.iter().sum::<f64>() / matching.len() as f64
        };

        let root_strategy = trainer.average_strategy(root_id);
        let bet_freq_jj = avg(&is_jj, &root_strategy);
        let bet_freq_qq = avg(&is_qq, &root_strategy);
        let bet_freq_air = avg(&|c1, c2| !is_jj(c1, c2) && !is_qq(c1, c2), &root_strategy);

        // With only 3 buckets shared across 4 hand types, the top bucket
        // will typically end up merging JJ (only 3 combos, the very top of
        // the score range) with some of the next-highest-scoring QQ
        // combos — so an unbucketed-style ">90% bet" floor for JJ alone
        // isn't a fair expectation here; what abstraction should still
        // guarantee is that hand-strength order isn't inverted, and that
        // trips clearly bets more than pure air (not diluted down to it).
        let tol = 0.05;
        assert!(
            bet_freq_jj >= bet_freq_qq - tol,
            "trips should bet at least as often as the overpair tier, got jj={bet_freq_jj}, qq={bet_freq_qq}"
        );
        assert!(
            bet_freq_qq >= bet_freq_air - tol,
            "overpair should bet at least as often as pure air, got qq={bet_freq_qq}, air={bet_freq_air}"
        );
        assert!(
            bet_freq_jj > bet_freq_air + 0.1,
            "trips should still clearly out-bet pure air despite bucketing, got jj={bet_freq_jj}, air={bet_freq_air}"
        );
    }

    // --- Flop/turn card abstraction ---

    /// Cross-checks `avg_score`'s suit-isomorphism-class-weighted
    /// averaging against a naive full 1-card-at-a-time enumeration (no
    /// classes at all) — the same "fast path vs. reference" pattern
    /// `eval::evaluate_5` is validated with elsewhere in this codebase.
    /// Uses a two-suit board (like `two_suit_turn_board`) specifically so
    /// classes of size > 1 actually get exercised, not just trivial
    /// singletons.
    #[test]
    fn avg_score_matches_naive_full_enumeration_average() {
        let board = two_suit_turn_board(); // 2c 7d 9c Jd -- h/s free
        let dead = crate::card::set_from_cards(&board);
        let combos = Range::parse("JJ,54o").unwrap().combos(dead);
        let used = crate::card::set_from_cards(&board);

        let naive: Vec<f64> = combos
            .iter()
            .map(
                |&Combo {
                     cards: [h1, h2], ..
                 }| {
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    for c in 0..52u8 {
                        if crate::card::set_contains(used, c) || c == h1 || c == h2 {
                            continue;
                        }
                        let mut river = board.clone();
                        river.push(c);
                        let score =
                            evaluate_7([h1, h2, river[0], river[1], river[2], river[3], river[4]]);
                        sum += score as f64;
                        count += 1.0;
                    }
                    sum / count
                },
            )
            .collect();

        let board_eval_cache = HashMap::new();
        let mut memo = HashMap::new();
        let combo_index = [build_combo_index(&combos), HashMap::new()];
        let via_classes = avg_score(
            &board,
            &[combos.clone(), Vec::new()],
            &combo_index,
            &board_eval_cache,
            &mut memo,
        );

        for (i, (&n, &c)) in naive.iter().zip(via_classes[0].iter()).enumerate() {
            assert!((n - c).abs() < 1e-9, "combo {i}: naive={n}, avg_score={c}");
        }
    }

    fn flop_board() -> Vec<Card> {
        // Monotone (all clubs) -> maximal isomorphism benefit (3 free
        // suits), keeping the two-chance-node tree small enough to train
        // quickly even without bucketing, for the equivalence check below.
        ["2c", "7c", "9c"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect()
    }

    struct FlopScenario {
        tree: super::super::tree::Tree,
        trainer_combos: [Vec<Combo>; 2],
        board: Vec<Card>,
    }

    fn flop_scenario_combos() -> [Vec<Combo>; 2] {
        let dead = crate::card::set_from_cards(&flop_board());
        let oop_combos = Range::parse("JJ,54o").unwrap().combos(dead);
        let ip_combos = Range::parse("76o").unwrap().combos(dead);
        [oop_combos, ip_combos]
    }

    fn build_flop_scenario(trainer_combos: [Vec<Combo>; 2]) -> FlopScenario {
        let board = flop_board();
        let cfg = GameConfig {
            board: board.clone(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![1.0],
            max_raises: 0,
            suit_isomorphism: true,
        };
        let tree = build_tree(&cfg).unwrap();
        FlopScenario {
            tree,
            trainer_combos,
            board,
        }
    }

    /// `CardAbstraction { turn: Some(n), .. }` with `n` at least as large
    /// as either player's combo count must be a bijection on the turn
    /// street (same reasoning as `compute_percentile_buckets_...`), so it
    /// must converge to the same equilibrium as no abstraction at all.
    /// Uses `turn_board()`'s scenario directly (a solve that *starts*
    /// already at the turn — no chance node deals into it), which also
    /// exercises the "initial board" inclusion in `collect_reachable_boards`
    /// usage for turn/flop, mirroring the river case.
    #[test]
    fn turn_buckets_bijective_matches_unbucketed() {
        let combos = multi_street_scenario_combos();
        let scenario_none = build_multi_street_scenario_with_combos(combos.clone());
        let scenario_bucketed = build_multi_street_scenario_with_combos(combos);
        let (root_id, ip_facing_bet_id) = root_and_ip_facing_bet_ids(&scenario_none.tree.root);

        let iterations = 1_500;
        let mut trainer_none = Trainer::new(
            scenario_none.tree,
            scenario_none.board,
            scenario_none.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer_none.train(iterations);
        let mut trainer_bucketed = Trainer::new(
            scenario_bucketed.tree,
            scenario_bucketed.board,
            scenario_bucketed.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                abstraction: CardAbstraction {
                    turn: Some(100),
                    ..Default::default()
                },
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer_bucketed.train(iterations);

        let tol = 0.05;
        for (row_n, row_b) in trainer_none
            .average_strategy(root_id)
            .iter()
            .zip(trainer_bucketed.average_strategy(root_id).iter())
        {
            for (&fn_, &fb) in row_n.iter().zip(row_b.iter()) {
                assert!(
                    (fn_ - fb).abs() < tol,
                    "root strategy mismatch: none={fn_}, bucketed(bijective)={fb}"
                );
            }
        }
        for (row_n, row_b) in trainer_none
            .average_strategy(ip_facing_bet_id)
            .iter()
            .zip(trainer_bucketed.average_strategy(ip_facing_bet_id).iter())
        {
            for (&fn_, &fb) in row_n.iter().zip(row_b.iter()) {
                assert!(
                    (fn_ - fb).abs() < tol,
                    "IP-facing-bet strategy mismatch: none={fn_}, bucketed(bijective)={fb}"
                );
            }
        }
    }

    /// Same equivalence check, but on the flop: `CardAbstraction { flop:
    /// Some(n), .. }` bijective must match unabstracted, exercising a tree
    /// with *two* chance nodes (flop deals the turn, which deals the
    /// river) and `avg_score`'s two-level recursion.
    #[test]
    fn flop_buckets_bijective_matches_unbucketed() {
        let combos = flop_scenario_combos();
        let scenario_none = build_flop_scenario(combos.clone());
        let scenario_bucketed = build_flop_scenario(combos);
        let (root_id, ip_facing_bet_id) = root_and_ip_facing_bet_ids(&scenario_none.tree.root);

        let iterations = 300;
        let mut trainer_none = Trainer::new(
            scenario_none.tree,
            scenario_none.board,
            scenario_none.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer_none.train(iterations);
        let mut trainer_bucketed = Trainer::new(
            scenario_bucketed.tree,
            scenario_bucketed.board,
            scenario_bucketed.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                abstraction: CardAbstraction {
                    flop: Some(100),
                    ..Default::default()
                },
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer_bucketed.train(iterations);

        let tol = 0.08;
        for (row_n, row_b) in trainer_none
            .average_strategy(root_id)
            .iter()
            .zip(trainer_bucketed.average_strategy(root_id).iter())
        {
            for (&fn_, &fb) in row_n.iter().zip(row_b.iter()) {
                assert!(
                    (fn_ - fb).abs() < tol,
                    "root strategy mismatch: none={fn_}, bucketed(bijective)={fb}"
                );
            }
        }
        for (row_n, row_b) in trainer_none
            .average_strategy(ip_facing_bet_id)
            .iter()
            .zip(trainer_bucketed.average_strategy(ip_facing_bet_id).iter())
        {
            for (&fn_, &fb) in row_n.iter().zip(row_b.iter()) {
                assert!(
                    (fn_ - fb).abs() < tol,
                    "IP-facing-bet strategy mismatch: none={fn_}, bucketed(bijective)={fb}"
                );
            }
        }
    }

    /// With a small turn bucket count, exact per-combo behavior isn't
    /// expected, but zero-sum should still hold and JJ (the only real
    /// value hand, always beating IP's fixed 76o) should still bet
    /// noticeably more than pure air (54o, which always loses) — the same
    /// kind of sanity check as the river's reduced-bucket test, adapted to
    /// a street where the strength proxy is `avg_score`, not a final
    /// `evaluate_7` score.
    #[test]
    fn turn_buckets_reduced_stays_zero_sum_and_respects_hand_strength_order() {
        let scenario = build_multi_street_scenario();
        let (root_id, _) = root_and_ip_facing_bet_ids(&scenario.tree.root);
        let combos0 = scenario.trainer_combos[0].clone();

        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                abstraction: CardAbstraction {
                    turn: Some(2),
                    ..Default::default()
                },
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        let [ev0, ev1] = trainer.train(1_500);
        assert!(
            (ev0 + ev1).abs() < 1.0,
            "expected zero-sum root EV even under turn abstraction, got ev0={ev0}, ev1={ev1}"
        );

        let is_jj = |c1: Card, c2: Card| crate::card::rank(c1) == 9 && crate::card::rank(c2) == 9;
        let root_strategy = trainer.average_strategy(root_id);
        let bet_freq_jj: Vec<f64> = combos0
            .iter()
            .enumerate()
            .filter(
                |(
                    _,
                    &Combo {
                        cards: [c1, c2], ..
                    },
                )| is_jj(c1, c2),
            )
            .map(|(i, _)| root_strategy[1][i])
            .collect();
        let bet_freq_air: Vec<f64> = combos0
            .iter()
            .enumerate()
            .filter(
                |(
                    _,
                    &Combo {
                        cards: [c1, c2], ..
                    },
                )| !is_jj(c1, c2),
            )
            .map(|(i, _)| root_strategy[1][i])
            .collect();
        let avg_jj = bet_freq_jj.iter().sum::<f64>() / bet_freq_jj.len() as f64;
        let avg_air = bet_freq_air.iter().sum::<f64>() / bet_freq_air.len() as f64;
        assert!(avg_jj > avg_air + 0.1, "value should still clearly out-bet air despite turn bucketing, got jj={avg_jj}, air={avg_air}");
    }

    /// A naive rank-based percentile split can place two *exactly tied*
    /// scores into different buckets purely because their sort positions
    /// straddle a cut point. That's specifically dangerous under suit
    /// isomorphism: two combos related by a free-suit swap always get the
    /// identical `avg_score` (a provable symmetry — the whole future
    /// board/outcome distribution is suit-blind), and `chance_value`'s
    /// relabeling assumes such combos always share one strategy, so
    /// splitting their tie across a bucket boundary would silently break
    /// that invariant. Regression test for exactly that bug, found via
    /// `flop_and_isomorphism_interaction_stays_zero_sum` below.
    #[test]
    fn compute_percentile_buckets_never_splits_a_tie() {
        let scores = [10.0, 10.0, 10.0, 20.0, 20.0, 30.0, 40.0, 40.0];
        for num_buckets in 1..=scores.len() {
            let buckets = compute_percentile_buckets(&scores, num_buckets);
            for (a, b) in [(0, 1), (1, 2), (3, 4), (6, 7)] {
                assert_eq!(
                    buckets[a], buckets[b],
                    "num_buckets={num_buckets}: tied scores {}/{} got different buckets {:?}",
                    scores[a], scores[b], buckets
                );
            }
        }
    }

    /// Regression test for a real bug found while building flop/turn
    /// abstraction: with isomorphism enabled, `chance_value`'s relabeling
    /// requires every ancestor decision node (bucketed or not) to treat
    /// suit-swap-equivalent combos identically. A flop decision node
    /// bucketed by `avg_score` broke that when tied scores landed in
    /// different buckets (fixed by `compute_percentile_buckets_never_splits_a_tie`'s
    /// fix) — the visible symptom was the root EV silently losing its
    /// zero-sum property at some iteration counts (but not others,
    /// depending on when the bucket-shared regret happened to matter),
    /// entirely deterministically (not a floating-point/thread-ordering
    /// issue — reproduced bit-for-bit across repeated runs), which is why
    /// this checks several iteration counts, not just one.
    #[test]
    fn flop_and_isomorphism_interaction_stays_zero_sum() {
        for iters in [1u32, 5, 20, 100] {
            let combos = flop_scenario_combos();
            let scenario = build_flop_scenario(combos);
            let mut trainer = Trainer::new(
                scenario.tree,
                scenario.board,
                scenario.trainer_combos,
                TrainerOptions {
                    algorithm: SolverAlgorithm::CfrPlus,
                    abstraction: CardAbstraction {
                        flop: Some(2),
                        ..Default::default()
                    },
                    ..TrainerOptions::default()
                },
            )
            .unwrap();
            let [ev0, ev1] = trainer.train(iters);
            assert!(
                (ev0 + ev1).abs() < 1e-6,
                "iters={iters}: expected exact zero-sum, got ev0={ev0}, ev1={ev1}, sum={}",
                ev0 + ev1
            );
        }
    }

    // --- EV / equity (Trainer::evaluate) ---

    #[test]
    fn evaluate_average_strategy_root_ev_is_zero_sum() {
        let scenario = build_scenario();
        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer.train(20_000);
        let (_, [ev0, ev1]) = trainer.evaluate();
        assert!(
            (ev0 + ev1).abs() < 1.0,
            "expected zero-sum root EV under the average strategy, got ev0={ev0}, ev1={ev1}"
        );
    }

    /// `river_board()`'s scenario (`build_scenario`) has a known winner at
    /// showdown: OOP's JJ always beats IP's fixed 99, OOP's KQo always
    /// loses to it. Equity is exact and cheap at a river (complete-board)
    /// node, so this checks it directly against hand-derived values,
    /// unlike EV (which folds in betting and is only checked qualitatively
    /// elsewhere).
    #[test]
    fn river_equity_matches_known_showdown_outcomes() {
        let scenario = build_scenario();
        let root_id = match &scenario.tree.root {
            Node::Decision { id, .. } => *id,
            _ => panic!("expected decision"),
        };
        let combos0 = scenario.trainer_combos[0].clone();
        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer.train(5_000);

        let (eval, _) = trainer.evaluate();
        let root_eval = eval.get(&root_id).expect("root should have an evaluation");
        let equity = root_eval
            .equity
            .as_ref()
            .expect("river (complete board) node should have equity");

        for (
            i,
            &Combo {
                cards: [c1, c2], ..
            },
        ) in combos0.iter().enumerate()
        {
            if is_jj(c1, c2) {
                assert!(
                    (equity[i] - 1.0).abs() < 1e-9,
                    "JJ should have 100% equity vs IP's fixed 99, got {}",
                    equity[i]
                );
            } else {
                assert!(
                    equity[i].abs() < 1e-9,
                    "KQo should have 0% equity vs IP's fixed 99, got {}",
                    equity[i]
                );
            }
        }
    }

    /// Equity is only computed where it's cheap and exact (a complete
    /// board) — see `NodeEvaluation`'s doc comment for why flop/turn
    /// equity isn't implemented yet. A pre-river node's evaluation should
    /// still have a sensible EV, just no equity figure.
    #[test]
    fn equity_is_none_before_the_river() {
        let scenario = build_multi_street_scenario(); // starts at the turn, one chance node to the river
        let root_id = match &scenario.tree.root {
            Node::Decision { id, .. } => *id,
            _ => panic!("expected decision"),
        };
        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer.train(500);

        let (eval, _) = trainer.evaluate();
        let root_eval = eval.get(&root_id).expect("root should have an evaluation");
        assert!(
            root_eval.equity.is_none(),
            "expected no equity figure at a pre-river (incomplete board) node"
        );
        assert_eq!(
            root_eval.ev.len(),
            trainer.combos(0).len(),
            "EV should still be reported per combo"
        );
    }

    /// Regression test for a normalization bug: the per-node `NodeEvaluation.ev`
    /// stored for display must be a true per-combo expected value, not the
    /// raw counterfactual value `v(I) = sum_j reach_opp[j] * payoff_j` used
    /// internally elsewhere in this module (a sum over the opponent's whole
    /// live range, which scales with the *number* of opponent combos and can
    /// be many times larger than any real outcome). In `build_scenario`, the
    /// most any combo can win is `pot + bet` (called and won) and the most it
    /// can lose is `bet` (called and lost), so every displayed EV must stay
    /// within that range — the un-normalized bug produced values in the
    /// hundreds to thousands, far outside it.
    #[test]
    fn evaluate_ev_is_normalized_per_combo_not_raw_counterfactual_value() {
        let scenario = build_scenario();
        let pot = scenario.pot;
        let bet = scenario.bet;
        let root_id = match &scenario.tree.root {
            Node::Decision { id, .. } => *id,
            _ => panic!("expected decision"),
        };
        let ip_facing_bet_id = match &scenario.tree.root {
            Node::Decision { children, .. } => match &children[1] {
                Node::Decision { id, .. } => *id,
                _ => panic!("expected IP decision facing the bet"),
            },
            _ => panic!("expected decision"),
        };

        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer.train(20_000);
        let (eval, _) = trainer.evaluate();

        let bound = pot + bet;
        let root_eval = eval.get(&root_id).expect("root should have an evaluation");
        for &ev in &root_eval.ev {
            assert!(
                ev.abs() <= bound + 1e-6,
                "root EV {ev} exceeds the game's stake bound of {bound}"
            );
        }
        let ip_eval = eval
            .get(&ip_facing_bet_id)
            .expect("IP-facing-bet node should have an evaluation");
        for &ev in &ip_eval.ev {
            assert!(
                ev.abs() <= bound + 1e-6,
                "IP-facing-bet EV {ev} exceeds the game's stake bound of {bound}"
            );
        }

        // JJ always bets and always beats IP's fixed 99, so under the
        // converged average strategy its root EV should be close to
        // winning the pot every time it isn't check-through — strictly
        // positive, not just "in bounds".
        let combos0 = trainer.combos(0);
        for (
            i,
            &Combo {
                cards: [c1, c2], ..
            },
        ) in combos0.iter().enumerate()
        {
            if is_jj(c1, c2) {
                assert!(
                    root_eval.ev[i] > 0.0,
                    "JJ should have positive EV, got {}",
                    root_eval.ev[i]
                );
            }
        }
    }

    // --- Exploitability (Trainer::exploitability) ---

    /// `Exploitability::total` is the standard `(BR0 + BR1) / 2` figure — it
    /// can never be meaningfully negative (best-responding to a fixed
    /// opponent strategy can never do worse than the strategy actually
    /// played) and shrinks toward 0 as training converges. Check the
    /// monotonic trend rather than an iteration-specific value; separate
    /// analytical tests below pin the units and normalization.
    #[test]
    fn policy_snapshot_is_frozen_and_matches_trainer_reads() {
        let board: Vec<Card> = ["2c", "7c", "9d", "Jd"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect();
        let dead = set_from_cards(&board);
        let combos = [
            Range::parse("AA:0.5,KK,QQ").unwrap().combos(dead),
            Range::parse("JJ,TT").unwrap().combos(dead),
        ];
        let cfg = GameConfig {
            board: board.clone(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(150),
            bet_sizes: vec![0.5, 1.0],
            max_raises: 1,
            suit_isomorphism: true,
        };
        let abstraction = CardAbstraction {
            river: Some(2),
            ..CardAbstraction::default()
        };
        let make = || {
            Trainer::new(
                build_tree(&cfg).unwrap(),
                board.clone(),
                combos.clone(),
                TrainerOptions {
                    algorithm: SolverAlgorithm::Dcfr,
                    abstraction,
                    ..TrainerOptions::default()
                },
            )
            .unwrap()
        };
        let bits = |(nodes, root): (HashMap<usize, NodeEvaluation>, [f64; 2])| {
            let mut nodes: Vec<_> = nodes
                .into_iter()
                .map(|(id, e)| (id, e.ev.iter().map(|v| v.to_bits()).collect::<Vec<_>>()))
                .collect();
            nodes.sort();
            (nodes, root.map(f64::to_bits))
        };

        let mut trainer = make();
        trainer.train(30);
        let policy = trainer.policy();
        let game = Arc::clone(trainer.game());
        let evaluation = bits(game.evaluate(&policy));
        let exploitability = game.exploitability(&policy).total.to_bits();
        assert_eq!(evaluation, bits(trainer.evaluate()));
        assert_eq!(exploitability, trainer.exploitability().total.to_bits());
        for id in 0..game.num_decision_nodes() {
            assert_eq!(
                trainer.average_strategy(id),
                game.average_strategy(&policy, id)
            );
        }

        // Training on does not disturb the snapshot, which still equals a
        // fresh trainer stopped at the same iteration.
        trainer.train(30);
        assert_ne!(evaluation, bits(trainer.evaluate()));
        assert_eq!(evaluation, bits(game.evaluate(&policy)));
        assert_eq!(exploitability, game.exploitability(&policy).total.to_bits());
        let mut fresh = make();
        fresh.train(30);
        assert_eq!(evaluation, bits(fresh.evaluate()));

        // A policy rebuilt from its rows (as a save does) reads identically,
        // and rows that do not fit the game are rejected.
        let restored = game.policy_from_rows(policy.rows().to_vec()).unwrap();
        assert_eq!(evaluation, bits(game.evaluate(&restored)));
        let mut short = policy.rows().to_vec();
        short.pop();
        assert!(game.policy_from_rows(short).is_err());
        let mut skewed = policy.rows().to_vec();
        skewed[0][0][0] += 0.5;
        assert!(game.policy_from_rows(skewed).is_err());

        // A distribution that breaks suit symmetry is rejected: chance nodes
        // derive non-representative cards from the representative's values.
        let (id, b, partner) = game
            .node_meta
            .iter()
            .enumerate()
            .find_map(|(id, (player, board))| {
                let group = game.suit_groups.get(board)?[*player]
                    .iter()
                    .find(|g| g.len() > 1)?;
                Some((id, group[0], group[1]))
            })
            .expect("suit-reduced fixture has a nontrivial suit group");
        let mut asymmetric = policy.rows().to_vec();
        let rows = &mut asymmetric[id];
        rows.iter_mut().for_each(|row| row[b] = 0.0);
        rows[0][b] = 1.0;
        rows.iter_mut().for_each(|row| row[partner] = 0.0);
        let last = rows.len() - 1;
        rows[last][partner] = 1.0;
        assert!(last > 0);
        assert!(
            matches!(game.policy_from_rows(asymmetric).unwrap_err(), PolicyError::NotSymmetric { node } if node == id)
        );
    }

    #[test]
    fn exploitability_decreases_with_more_training() {
        let few = build_scenario();
        let mut few_trainer = Trainer::new(
            few.tree,
            few.board,
            few.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        few_trainer.train(100);
        let few_expl = few_trainer.exploitability();

        let many = build_scenario();
        let mut many_trainer = Trainer::new(
            many.tree,
            many.board,
            many.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        many_trainer.train(20_000);
        let many_expl = many_trainer.exploitability();

        assert!(
            few_expl.total > -1e-6,
            "exploitability should never be meaningfully negative, got {}",
            few_expl.total
        );
        assert!(
            many_expl.total > -1e-6,
            "exploitability should never be meaningfully negative, got {}",
            many_expl.total
        );
        assert!(
            many_expl.total < few_expl.total,
            "expected exploitability to shrink with more training: 100 iters -> {}, 20,000 iters -> {}",
            few_expl.total,
            many_expl.total
        );
    }

    #[test]
    fn exploitability_is_near_zero_after_convergence() {
        let scenario = build_scenario();
        let pot = scenario.pot;
        let mut trainer = Trainer::new(
            scenario.tree,
            scenario.board,
            scenario.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        trainer.train(20_000);
        let expl = trainer.exploitability();
        assert!(
            expl.total < pot * 0.001,
            "expected exploitability below 0.1% of pot after 20,000 iterations, got {}",
            expl.total
        );
    }

    // A tiny, independently solvable river game. OOP always wins showdown:
    // check wins 5; bet wins 10 when IP calls and 5 when IP folds. With the
    // untrained uniform strategy, BR(OOP) = 7.5, BR(IP) = -5, so exploitability
    // is 1.25 chips. This pins a NONZERO metric, unlike an equilibrium-only
    // assertion that would pass with any (or no) normalization.
    fn analytical_exploitability_trainer(combos: [Vec<Combo>; 2]) -> Trainer {
        use crate::solver::tree::{Action, Player, Tree};
        let board: Vec<Card> = ["2c", "7d", "9h", "Js", "Qd"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect();
        let scores = eval_combos_if_complete(&combos, &board);
        let dead = set_from_cards(&board);
        // Assert the fixture's actual hand ordering, including every legal
        // pair; otherwise the analytical payoff argument would be invalid.
        for (
            i,
            &Combo {
                cards: [h1, h2], ..
            },
        ) in combos[0].iter().enumerate()
        {
            for (
                j,
                &Combo {
                    cards: [v1, v2], ..
                },
            ) in combos[1].iter().enumerate()
            {
                if [h1, h2, v1, v2].iter().any(|&c| set_contains(dead, c))
                    || h1 == v1
                    || h1 == v2
                    || h2 == v1
                    || h2 == v2
                {
                    continue;
                }
                assert!(
                    scores[0][i] > scores[1][j],
                    "fixture requires OOP to win every legal showdown"
                );
            }
        }
        let tree = Tree {
            root: Node::Decision {
                id: 0,
                player: Player::Oop,
                actions: vec![Action::Check, Action::Bet(Chips::whole(5))],
                children: vec![
                    Node::Terminal(Terminal::Showdown {
                        invested: [5.0, 5.0],
                    }),
                    Node::Decision {
                        id: 1,
                        player: Player::Ip,
                        actions: vec![Action::Call, Action::Fold],
                        children: vec![
                            Node::Terminal(Terminal::Showdown {
                                invested: [10.0, 10.0],
                            }),
                            Node::Terminal(Terminal::Fold {
                                winner: Player::Oop,
                                invested: [10.0, 5.0],
                            }),
                        ],
                    },
                ],
            },
            num_decision_nodes: 2,
        };
        // Unchecked: the fixtures deliberately include board-blocked combos
        // and impossible deals, which the kernels must still ignore.
        Trainer::new_unchecked(
            tree,
            board,
            combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
    }

    fn weighted_combo(a: &str, b: &str, weight: f64) -> Combo {
        Combo::new(card_from_str(a).unwrap(), card_from_str(b).unwrap(), weight)
    }

    fn assert_analytical_exploitability(expl: Exploitability) {
        assert!(
            (expl.oop - 7.5).abs() < 1e-10,
            "OOP BR must be 7.5 chips, got {}",
            expl.oop
        );
        assert!(
            (expl.ip + 5.0).abs() < 1e-10,
            "IP BR must be -5 chips, got {}",
            expl.ip
        );
        assert!(
            (expl.total - 1.25).abs() < 1e-10,
            "exploitability must be 1.25 chips, got {}",
            expl.total
        );
    }

    #[test]
    fn exploitability_matches_analytical_best_responses_with_blockers() {
        // Three legal deals, not the four pairs in the Cartesian product:
        // AcAd cannot be dealt against AcKs. Board-blocked entries on BOTH
        // sides must affect neither numerator nor denominator.
        let combos = [
            vec![
                weighted_combo("Ac", "Ad", 1.0),
                weighted_combo("Ah", "As", 1.0),
                weighted_combo("Qd", "Qc", 1.0),
            ],
            vec![
                weighted_combo("Kc", "Kd", 1.0),
                weighted_combo("Ac", "Ks", 1.0),
                weighted_combo("2c", "3d", 1.0),
            ],
        ];
        assert_analytical_exploitability(
            analytical_exploitability_trainer(combos).exploitability(),
        );
    }

    #[test]
    fn exploitability_chip_units_ignore_range_weight_scale() {
        for (oop_scale, ip_scale) in [(1.0, 1.0), (0.25, 1.0), (1.0, 0.7), (0.25, 0.7)] {
            let combos = [
                vec![
                    weighted_combo("Ac", "Ad", oop_scale),
                    weighted_combo("Ah", "As", 0.3 * oop_scale),
                ],
                vec![
                    weighted_combo("Kc", "Kd", ip_scale),
                    weighted_combo("Ac", "Ks", 0.6 * ip_scale),
                ],
            ];
            assert_analytical_exploitability(
                analytical_exploitability_trainer(combos).exploitability(),
            );
        }
    }

    #[test]
    fn exploitability_rejects_zero_or_nonfinite_joint_mass() {
        for combos in [
            [
                vec![weighted_combo("Ac", "Ad", 1.0)],
                vec![weighted_combo("Ac", "Ks", 1.0)],
            ],
            [
                vec![weighted_combo("Ac", "Ad", 0.0)],
                vec![weighted_combo("Kc", "Kd", 1.0)],
            ],
            [
                vec![weighted_combo("Qd", "Qc", 1.0)],
                vec![weighted_combo("Kc", "Kd", 1.0)],
            ],
            [
                vec![weighted_combo("Ac", "Ad", f64::INFINITY)],
                vec![weighted_combo("Kc", "Kd", 1.0)],
            ],
            [
                vec![weighted_combo("Ac", "Ad", f64::NAN)],
                vec![weighted_combo("Kc", "Kd", 1.0)],
            ],
        ] {
            let expl = analytical_exploitability_trainer(combos).exploitability();
            assert!(
                expl.oop.is_nan() && expl.ip.is_nan() && expl.total.is_nan(),
                "undefined metric must not appear converged: {expl:?}"
            );
        }
    }

    /// `best_response_chance_value` (the counterpart to `chance_value`'s
    /// suit-isomorphism relabeling) is only exercised by a scenario with an
    /// actual `Chance` node whose classes have more than one member — this
    /// codebase has repeatedly found relabeling bugs that only manifest
    /// with isomorphism enabled, so mirror
    /// `suit_isomorphism_reduced_and_unreduced_converge_to_the_same_equilibrium`
    /// and check that isomorphism-reduced and unreduced trees converge to
    /// the same exploitability rather than merely not-crashing.
    #[test]
    fn exploitability_isomorphism_reduced_and_unreduced_match() {
        let combos = two_suit_scenario_combos();
        let reduced = build_two_suit_scenario(true, combos.clone());
        let unreduced = build_two_suit_scenario(false, combos);

        let iterations = 3_000;
        let mut reduced_trainer = Trainer::new(
            reduced.tree,
            reduced.board,
            reduced.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        reduced_trainer.train(iterations);
        let mut unreduced_trainer = Trainer::new(
            unreduced.tree,
            unreduced.board,
            unreduced.trainer_combos,
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        unreduced_trainer.train(iterations);

        let reduced_expl = reduced_trainer.exploitability();
        let unreduced_expl = unreduced_trainer.exploitability();
        // Reduced/unreduced training takes different floating-point paths
        // (different iteration order over the tree) to the same underlying
        // equilibrium, so — unlike the average-strategy comparison this test
        // mirrors, whose *probabilities* are naturally bounded to [0, 1] and
        // so can use a tight absolute tolerance — exploitability's chip
        // magnitude calls for a relative tolerance instead.
        let rel_diff =
            (reduced_expl.total - unreduced_expl.total).abs() / unreduced_expl.total.abs().max(1.0);
        assert!(
            rel_diff < 0.1,
            "expected matching exploitability with isomorphism enabled/disabled: reduced={}, unreduced={}",
            reduced_expl.total,
            unreduced_expl.total
        );
    }
}

#[cfg(test)]
mod terminal_regression_tests {
    use super::*;
    use crate::card::card_from_str;
    use crate::solver::tree::{Player, Tree};
    use crate::solver::Chips;

    fn close(actual: f64, expected: f64) {
        let tolerance = 2e-10 + expected.abs() * 2e-12;
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }

    #[test]
    fn linear_terminals_match_pairwise_oracle() {
        // Entire deck includes initial-board conflicts, overlaps, equal-rank
        // groups, empty support and every valid two-card blocker pattern.
        let full: Vec<_> = (0..52)
            .flat_map(|a| (a + 1..52).map(move |b| Combo::new(a, b, 1.0)))
            .collect();
        let mut asymmetric = full.iter().step_by(7).copied().collect::<Vec<_>>();
        asymmetric.reverse();
        let Combo {
            cards: [a, b],
            weight,
        } = asymmetric[3];
        asymmetric.push(Combo::new(b, a, weight)); // repeated, reversed exact pair
        for board in [
            ["2c", "7d", "9h", "Js", "Qd"],
            ["Tc", "Jc", "Qc", "Kc", "Ac"],
            ["Jh", "Js", "2c", "2d", "3h"],
        ] {
            let board: Vec<_> = board.iter().map(|s| card_from_str(s).unwrap()).collect();
            for combos in [
                [full.clone(), full.clone()],
                [asymmetric.clone(), full.clone()],
                [full.clone(), Vec::new()],
            ] {
                let tree = Tree {
                    root: Node::Terminal(Terminal::Showdown {
                        invested: [37.0, 121.0],
                    }),
                    num_decision_nodes: 0,
                };
                // Unchecked: blocked, repeated, and empty ranges exercise the
                // kernels' degenerate paths.
                let trainer = Trainer::new_unchecked(
                    tree,
                    board.clone(),
                    combos,
                    TrainerOptions {
                        algorithm: SolverAlgorithm::CfrPlus,
                        ..TrainerOptions::default()
                    },
                );
                for phase in 0..3 {
                    let reach: [Vec<f64>; 2] = std::array::from_fn(|p| {
                        (0..trainer.combos[p].len())
                            .map(|i| {
                                if phase == 2 || (i + phase) % 5 == 0 {
                                    0.0
                                } else {
                                    ((i * 37 + p * 19 + phase) % 103) as f64 / 103.0
                                }
                            })
                            .collect()
                    });
                    for terminal in [
                        Terminal::Showdown {
                            invested: [37.0, 121.0],
                        },
                        Terminal::Showdown {
                            invested: [50.0, 50.0],
                        },
                        Terminal::Fold {
                            winner: Player::Oop,
                            invested: [37.0, 121.0],
                        },
                        Terminal::Fold {
                            winner: Player::Ip,
                            invested: [37.0, 121.0],
                        },
                    ] {
                        let scores = eval_combos_if_complete(&trainer.combos, &board);
                        trainer
                            .use_pairwise_terminals
                            .store(false, std::sync::atomic::Ordering::Relaxed);
                        let actual = trainer.terminal_value(&terminal, &reach, &board, &scores);
                        if let Terminal::Showdown { invested } = &terminal {
                            for p in 0..2 {
                                let before = trainer.terminal_index.showdown_before_rank_cache(
                                    &trainer.combos,
                                    p,
                                    &reach[1 - p],
                                    &scores,
                                    &trainer.board_order_cache[&board],
                                    set_from_cards(&board),
                                    invested,
                                );
                                assert_eq!(
                                    actual[p].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                    before.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                                );
                            }
                        }
                        trainer
                            .use_pairwise_terminals
                            .store(true, std::sync::atomic::Ordering::Relaxed);
                        let reference = trainer.terminal_value(&terminal, &reach, &board, &scores);
                        for p in 0..2 {
                            for (a, b) in actual[p].iter().zip(&reference[p]) {
                                close(*a, *b);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn linear_terminals_preserve_training_and_best_response() {
        use crate::range::Range;
        use crate::solver::tree::{build_tree, GameConfig};
        for board in [
            vec!["2c", "7d", "9h", "Js", "Qd"],
            vec!["Jh", "Js", "2c", "2d"],
        ] {
            let board: Vec<_> = board.iter().map(|s| card_from_str(s).unwrap()).collect();
            let combos: [Vec<_>; 2] = ["AA,KK,JJ,54s:0.3", "QQ,TT,88,65s:0.7"].map(|s| {
                let mut v = Range::parse(s).unwrap().combos(set_from_cards(&board));
                v.sort_by_key(|c| (c.cards[0], c.cards[1]));
                v
            });
            let cfg = GameConfig {
                board: board.clone(),
                pot: Chips::whole(100),
                effective_stack: Chips::whole(100),
                bet_sizes: vec![0.5],
                max_raises: 0,
                suit_isomorphism: false,
            };
            let mut actual = Trainer::new(
                build_tree(&cfg).unwrap(),
                board.clone(),
                combos.clone(),
                TrainerOptions {
                    algorithm: SolverAlgorithm::CfrPlus,
                    ..TrainerOptions::default()
                },
            )
            .unwrap();
            let mut reference = Trainer::new(
                build_tree(&cfg).unwrap(),
                board,
                combos,
                TrainerOptions {
                    algorithm: SolverAlgorithm::CfrPlus,
                    ..TrainerOptions::default()
                },
            )
            .unwrap();
            reference
                .use_pairwise_terminals
                .store(true, std::sync::atomic::Ordering::Relaxed);
            let ev_a = actual.train(60);
            let ev_b = reference.train(60);
            for p in 0..2 {
                close(ev_a[p], ev_b[p]);
            }
            for id in 0..actual.tree.num_decision_nodes {
                for (a, b) in actual
                    .average_strategy(id)
                    .iter()
                    .flatten()
                    .zip(reference.average_strategy(id).iter().flatten())
                {
                    close(*a, *b);
                }
            }
            let a = actual.exploitability();
            let b = reference.exploitability();
            close(a.oop, b.oop);
            close(a.ip, b.ip);
            close(a.total, b.total);
            let (a, _) = actual.evaluate();
            let (b, _) = reference.evaluate();
            for (id, eval) in a {
                for (x, y) in eval.ev.iter().zip(&b[&id].ev) {
                    close(*x, *y);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "cfr/target_values_tests.rs"]
mod target_values_tests;

#[cfg(test)]
#[path = "suit_reduction_tests.rs"]
mod suit_reduction_tests;

fn has_suit_reduction(node: &Node) -> bool {
    match node {
        Node::Chance { children } => children
            .iter()
            .any(|(members, child)| members.len() > 1 || has_suit_reduction(child)),
        Node::Decision { children, .. } => children.iter().any(has_suit_reduction),
        Node::Terminal(_) => false,
    }
}

/// The free-suit transpositions generate the board's full stabilizer. Union
/// bucket indices so overlapping abstraction buckets still form a partition.
fn suit_bucket_groups(
    board: &[Card],
    combos: &[Vec<Combo>; 2],
    index: &[HashMap<(Card, Card), usize>; 2],
    maps: &[Vec<usize>; 2],
) -> [Vec<Vec<usize>>; 2] {
    let free: Vec<u8> = (0..4)
        .filter(|s| !board.iter().any(|&c| suit(c) == *s))
        .collect();
    std::array::from_fn(|p| {
        if free.len() < 2 {
            return Vec::new();
        }
        let n = maps[p].iter().max().map_or(0, |b| b + 1);
        let mut parent: Vec<usize> = (0..n).collect();
        fn root(parent: &[usize], mut i: usize) -> usize {
            while parent[i] != i {
                i = parent[i];
            }
            i
        }
        for (a, &sa) in free.iter().enumerate() {
            for &sb in &free[a + 1..] {
                for (
                    i,
                    &Combo {
                        cards: [h1, h2],
                        weight: w,
                    },
                ) in combos[p].iter().enumerate()
                {
                    let pair = normalize_pair(swap_suit(h1, sa, sb), swap_suit(h2, sa, sb));
                    let &j = index[p]
                        .get(&pair)
                        .expect("suit reduction requires suit-closed ranges");
                    assert_eq!(
                        w, combos[p][j].weight,
                        "suit reduction requires suit-symmetric range weights"
                    );
                    let (x, y) = (root(&parent, maps[p][i]), root(&parent, maps[p][j]));
                    parent[x.max(y)] = x.min(y);
                }
            }
        }
        let mut groups = vec![Vec::new(); n];
        for i in 0..n {
            groups[root(&parent, i)].push(i);
        }
        groups.into_iter().filter(|group| group.len() > 1).collect()
    })
}
