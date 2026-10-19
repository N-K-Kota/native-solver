//! Betting tree spanning one or more streets, heads-up. OOP (out of
//! position) acts first on every street. Once a betting round ends without
//! a fold, a `Chance` node deals the next community card (unless the board
//! is already complete) before a fresh betting round starts; on the river
//! it resolves straight to showdown. Bet/raise sizes are configured as
//! fractions of the pot and clipped to the smaller of the two stacks (no
//! side pots — both players start each street with equal remaining stacks,
//! so this is exact, not an approximation). Chip amounts are exact integers
//! (`Chips`), so an all-in leaves exactly zero behind at any input scale.

use super::chips::Chips;
use crate::card::{
    card_bit, card_to_string, rank, set_contains, set_from_cards, suit, Card, CardSet,
};

/// Setup for betting starting from a given (partial or complete) board.
#[derive(Debug, Clone)]
pub struct GameConfig {
    /// The already-dealt board this street starts on: 3 cards (flop), 4
    /// (turn), or 5 (river, no further chance nodes are built).
    pub board: Vec<Card>,
    /// Chips already in the pot at the start of the first street.
    pub pot: Chips,
    /// Each player's remaining stack behind, at the start of the first street.
    pub effective_stack: Chips,
    /// Bet/raise sizes, as a fraction of the relevant pot size, reused
    /// identically on every street. Each amount is rounded half-to-even to a
    /// whole `Chips` unit, clipped to the available stack, and deduplicated;
    /// not automatically extended with an all-in option — include a large
    /// enough fraction for that.
    pub bet_sizes: Vec<f64>,
    /// Maximum number of raises allowed per street (resets each street).
    pub max_raises: u32,
    /// Group `Chance` node deals by suit-isomorphism class (see
    /// `docs/conventions.md`): candidate cards of the same
    /// rank whose suits are both entirely absent from the current board
    /// are strategically interchangeable, so the tree only recurses once
    /// per class (via a representative card) and `cfr.rs` derives the
    /// other members' counterfactual values by relabeling combo indices
    /// under the corresponding suit swap. When `false`, every class is a
    /// singleton (one card each) — the old, unreduced behavior.
    pub suit_isomorphism: bool,
}

/// A seat in the heads-up hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Player {
    /// Out of position: acts first on every street. Index 0.
    Oop,
    /// In position: acts last on every street. Index 1.
    Ip,
}

impl Player {
    /// The opponent.
    pub fn other(self) -> Player {
        match self {
            Player::Oop => Player::Ip,
            Player::Ip => Player::Oop,
        }
    }

    /// `0` for OOP and `1` for IP: the index into every `[_; 2]` per-player array.
    pub fn index(self) -> usize {
        match self {
            Player::Oop => 0,
            Player::Ip => 1,
        }
    }
}

/// A betting action at a decision node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Pass with no bet to call.
    Check,
    /// Open the betting with this many chips.
    Bet(Chips),
    /// Match the outstanding bet.
    Call,
    /// Give up the pot.
    Fold,
    /// Call and add this many chips on top (not the raise-to total).
    Raise(Chips),
}

/// How a hand ends, with the payoffs the solver uses.
#[derive(Debug, Clone, Copy)]
pub enum Terminal {
    /// A player folded and `winner` takes the pot; `invested` is each player's total
    /// contribution to the pot at this point (including the pre-street
    /// `pot`, split evenly as each player's baseline), in input chips. Built
    /// trees derive these from exact `Chips`; they stay `f64` because they are
    /// the payoffs the solver kernels consume.
    Fold {
        /// The player who did not fold.
        winner: Player,
        /// `[OOP, IP]` total contribution, in input chips.
        invested: [f64; 2],
    },
    /// Both players reach showdown on a complete board; the better hand wins
    /// the opponent's `invested` and a tie splits the pot.
    Showdown {
        /// `[OOP, IP]` total contribution, in input chips.
        invested: [f64; 2],
    },
}

/// A node of the betting tree.
#[derive(Debug, Clone)]
pub enum Node {
    /// A player chooses one of `actions`.
    Decision {
        /// Dense id in `0..Tree::num_decision_nodes`, assigned in depth-first
        /// preorder.
        /// Strategies and evaluations are keyed by it.
        id: usize,
        /// The player to act.
        player: Player,
        /// The legal actions, in a fixed order.
        actions: Vec<Action>,
        /// One subtree per action, in the same order as `actions`.
        children: Vec<Node>,
    },
    /// Deals the next community card, grouped into suit-isomorphism
    /// classes (see `GameConfig::suit_isomorphism`): each entry is one
    /// class's member cards plus the subtree built from its representative
    /// (`members[0]`). With isomorphism disabled every class is a
    /// singleton, so this is equivalent to one child per undealt card (52
    /// minus however many board cards have been dealt so far along this
    /// path).
    Chance {
        /// `(class members, subtree)` per suit-isomorphism class.
        children: Vec<(Vec<Card>, Node)>,
    },
    /// The hand is over.
    Terminal(Terminal),
}

/// Why a board was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoardError {
    /// The board must have 3 (flop), 4 (turn), or 5 (river) cards.
    WrongSize {
        /// The number of cards supplied.
        len: usize,
    },
    /// A card is outside `0..52`.
    InvalidCard {
        /// The offending card value.
        card: Card,
    },
    /// The same card appears more than once.
    DuplicateCard {
        /// The repeated card.
        card: Card,
    },
}

impl std::fmt::Display for BoardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongSize { len } => write!(f, "board has {len} cards; expected 3, 4, or 5"),
            Self::InvalidCard { card } => write!(f, "board card {card} is outside 0..52"),
            Self::DuplicateCard { card } => {
                write!(
                    f,
                    "board card {} appears more than once",
                    card_to_string(*card)
                )
            }
        }
    }
}

impl std::error::Error for BoardError {}

/// Checks that `board` is a legal flop, turn, or river.
pub(crate) fn validate_board(board: &[Card]) -> Result<(), BoardError> {
    if !(3..=5).contains(&board.len()) {
        return Err(BoardError::WrongSize { len: board.len() });
    }
    let mut seen: CardSet = 0;
    for &card in board {
        if card >= 52 {
            return Err(BoardError::InvalidCard { card });
        }
        if set_contains(seen, card) {
            return Err(BoardError::DuplicateCard { card });
        }
        seen |= card_bit(card);
    }
    Ok(())
}

/// Why a betting tree was not built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TreeError {
    /// `GameConfig::board` is not a legal flop, turn, or river.
    Board(BoardError),
    /// `GameConfig::pot` is zero or negative.
    NonPositivePot,
    /// `GameConfig::effective_stack` is negative.
    NegativeStack,
    /// A bet size is zero, negative, or not finite.
    InvalidBetSize {
        /// Position of the size in `GameConfig::bet_sizes`.
        index: usize,
    },
    /// The tree would need more decision nodes than the supplied budget.
    TooManyDecisionNodes {
        /// The decision-node budget that was exceeded.
        limit: usize,
    },
}

impl std::fmt::Display for TreeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Board(error) => write!(f, "invalid board: {error}"),
            Self::NonPositivePot => write!(f, "pot must be positive"),
            Self::NegativeStack => write!(f, "effective stack must not be negative"),
            Self::InvalidBetSize { index } => write!(
                f,
                "bet size at index {index} must be a positive, finite pot fraction"
            ),
            Self::TooManyDecisionNodes { limit } => write!(
                f,
                "game tree exceeds {limit} decision nodes; reduce bet sizes, raises, or start on a later street"
            ),
        }
    }
}

impl std::error::Error for TreeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Board(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BoardError> for TreeError {
    fn from(error: BoardError) -> Self {
        Self::Board(error)
    }
}

/// Rejects a configuration that does not describe a legal game.
fn validate_config(cfg: &GameConfig) -> Result<(), TreeError> {
    validate_board(&cfg.board)?;
    if cfg.pot <= Chips::ZERO {
        return Err(TreeError::NonPositivePot);
    }
    if cfg.effective_stack < Chips::ZERO {
        return Err(TreeError::NegativeStack);
    }
    if let Some(index) = cfg
        .bet_sizes
        .iter()
        .position(|&size| !size.is_finite() || size <= 0.0)
    {
        return Err(TreeError::InvalidBetSize { index });
    }
    Ok(())
}

/// A built betting tree.
pub struct Tree {
    /// The first node: OOP's first decision, unless no stack is left to bet.
    pub root: Node,
    /// The number of decision nodes; their ids are `0..num_decision_nodes`.
    pub num_decision_nodes: usize,
}

/// Builds the full betting tree for `cfg`. Use [`build_tree_bounded`] for
/// untrusted input, since the tree can grow very large.
///
/// # Errors
/// Returns an error if the board is not a legal flop, turn, or river, the
/// pot is not positive, the stack is negative, or a bet size is not a
/// positive finite number.
pub fn build_tree(cfg: &GameConfig) -> Result<Tree, TreeError> {
    build_tree_bounded(cfg, usize::MAX)
}

/// Stops construction before allocating a decision beyond the supplied budget.
/// Callers must also bound input size and recursion depth (notably max_raises).
/// Successful trees preserve the same node IDs and actions as `build_tree`.
///
/// # Errors
/// Returns the same configuration errors as [`build_tree`], or
/// [`TreeError::TooManyDecisionNodes`] when the budget is exceeded.
pub fn build_tree_bounded(cfg: &GameConfig, max_decision_nodes: usize) -> Result<Tree, TreeError> {
    validate_config(cfg)?;
    let mut budget = BuildBudget {
        next_id: 0,
        max_decision_nodes,
    };
    let stack = [cfg.effective_stack, cfg.effective_stack];
    let root = build(
        cfg,
        &cfg.board,
        Player::Oop,
        [Chips::ZERO; 2],
        stack,
        None,
        false,
        0,
        &mut budget,
    )?;
    Ok(Tree {
        root,
        num_decision_nodes: budget.next_id,
    })
}

struct BuildBudget {
    next_id: usize,
    max_decision_nodes: usize,
}

impl BuildBudget {
    fn allocate_id(&mut self) -> Result<usize, TreeError> {
        if self.next_id >= self.max_decision_nodes {
            return Err(TreeError::TooManyDecisionNodes {
                limit: self.max_decision_nodes,
            });
        }
        let id = self.next_id;
        self.next_id += 1;
        Ok(id)
    }
}

/// Suits that appear nowhere in `board` — any permutation among them maps
/// the board to itself, so cards of the same rank in different free suits
/// are strategically interchangeable at this point in the tree.
fn free_suits(board: &[Card]) -> [bool; 4] {
    let mut used = [false; 4];
    for &c in board {
        used[suit(c) as usize] = true;
    }
    [!used[0], !used[1], !used[2], !used[3]]
}

/// Groups the undealt cards into suit-isomorphism classes: cards of the
/// same rank whose suits are both free are one class (any one of them can
/// be trained and the rest derived by relabeling — see `cfr.rs`); a card
/// in a suit already pinned by the board is always its own singleton
/// class. With `suit_isomorphism` off, every class is a singleton
/// (equivalent to one child per undealt card).
/// Not tied to `GameConfig` (only needs the flag, not the rest of the
/// setup) so `cfr.rs` can also call it directly for its own internal
/// purposes — see `cfr::avg_score`, which always groups this way
/// regardless of whether the *tree* was built with isomorphism, since
/// grouping same-rank free-suit cards is always a valid equivalence.
pub(crate) fn deal_classes(suit_isomorphism: bool, board: &[Card]) -> Vec<Vec<Card>> {
    let used = set_from_cards(board);
    let free = if suit_isomorphism {
        free_suits(board)
    } else {
        [false; 4]
    };

    let mut classes: Vec<Vec<Card>> = Vec::new();
    let mut free_rank_class: [Option<usize>; 13] = [None; 13];
    for c in 0..52u8 {
        if set_contains(used, c) {
            continue;
        }
        if free[suit(c) as usize] {
            let r = rank(c) as usize;
            match free_rank_class[r] {
                Some(idx) => classes[idx].push(c),
                None => {
                    free_rank_class[r] = Some(classes.len());
                    classes.push(vec![c]);
                }
            }
        } else {
            classes.push(vec![c]);
        }
    }
    classes
}

/// Ends the current betting round: showdown if the board is already
/// complete (5 cards), otherwise a `Chance` node dealing the next card
/// into a fresh betting round (OOP acts first again, raise count resets).
fn end_of_street(
    cfg: &GameConfig,
    board: &[Card],
    added: [Chips; 2],
    stack: [Chips; 2],
    budget: &mut BuildBudget,
) -> Result<Node, TreeError> {
    if board.len() >= 5 {
        return Ok(Node::Terminal(Terminal::Showdown {
            invested: invested(cfg, added),
        }));
    }
    let mut children = Vec::new();
    for members in deal_classes(cfg.suit_isomorphism, board) {
        let rep = members[0];
        let mut next_board = board.to_vec();
        next_board.push(rep);
        let child = build(
            cfg,
            &next_board,
            Player::Oop,
            added,
            stack,
            None,
            false,
            0,
            budget,
        )?;
        children.push((members, child));
    }
    Ok(Node::Chance { children })
}

/// Terminal payoffs: half the starting pot each plus everything added since.
/// The half may be a half-unit, so it is applied only after leaving `Chips`.
fn invested(cfg: &GameConfig, added: [Chips; 2]) -> [f64; 2] {
    let baseline = cfg.pot.to_f64() / 2.0;
    [baseline + added[0].to_f64(), baseline + added[1].to_f64()]
}

/// `added` is each player's contribution beyond their half of `cfg.pot`.
#[allow(clippy::too_many_arguments)]
fn build(
    cfg: &GameConfig,
    board: &[Card],
    player: Player,
    added: [Chips; 2],
    stack: [Chips; 2],
    facing: Option<Chips>,
    prev_checked: bool,
    raises_so_far: u32,
    budget: &mut BuildBudget,
) -> Result<Node, TreeError> {
    // Once an all-in is called, neither player has a betting decision left.
    // Keep every remaining public-card chance (and its terminal contribution),
    // but do not allocate two single-action check nodes on each later street.
    // In particular this must precede allocate_id: forced checks must not use
    // the admission budget or leave holes in the trainer's decision IDs.
    if facing.is_none() && stack == [Chips::ZERO; 2] {
        return end_of_street(cfg, board, added, stack, budget);
    }
    let id = budget.allocate_id()?;

    let mut actions = Vec::new();
    let mut children = Vec::new();

    if let Some(to_call) = facing {
        actions.push(Action::Fold);
        children.push(Node::Terminal(Terminal::Fold {
            winner: player.other(),
            invested: invested(cfg, added),
        }));

        let mut call_added = added;
        call_added[player.index()] += to_call;
        let mut call_stack = stack;
        call_stack[player.index()] -= to_call;
        actions.push(Action::Call);
        children.push(end_of_street(cfg, board, call_added, call_stack, budget)?);

        if raises_so_far < cfg.max_raises && stack[player.index()] > to_call {
            let opponent = player.other();
            let pot_after_call = cfg.pot + added[0] + added[1] + to_call;
            let own_room = stack[player.index()] - to_call;
            let opponent_room = stack[opponent.index()];
            let raise_extras =
                sized_amounts(&cfg.bet_sizes, pot_after_call, own_room.min(opponent_room));

            for extra in raise_extras {
                let mut new_added = added;
                new_added[player.index()] += to_call + extra;
                let mut new_stack = stack;
                new_stack[player.index()] -= to_call + extra;
                actions.push(Action::Raise(extra));
                children.push(build(
                    cfg,
                    board,
                    opponent,
                    new_added,
                    new_stack,
                    Some(extra),
                    false,
                    raises_so_far + 1,
                    budget,
                )?);
            }
        }
    } else {
        let check_child = if prev_checked {
            end_of_street(cfg, board, added, stack, budget)?
        } else {
            build(
                cfg,
                board,
                player.other(),
                added,
                stack,
                None,
                true,
                raises_so_far,
                budget,
            )?
        };
        actions.push(Action::Check);
        children.push(check_child);

        if stack[player.index()] > Chips::ZERO {
            let opponent = player.other();
            let pot_now = cfg.pot + added[0] + added[1];
            let room = stack[player.index()].min(stack[opponent.index()]);
            let bet_amounts = sized_amounts(&cfg.bet_sizes, pot_now, room);

            for amt in bet_amounts {
                let mut new_added = added;
                new_added[player.index()] += amt;
                let mut new_stack = stack;
                new_stack[player.index()] -= amt;
                actions.push(Action::Bet(amt));
                children.push(build(
                    cfg,
                    board,
                    opponent,
                    new_added,
                    new_stack,
                    Some(amt),
                    false,
                    raises_so_far,
                    budget,
                )?);
            }
        }
    }

    Ok(Node::Decision {
        id,
        player,
        actions,
        children,
    })
}

/// Pot-fraction amounts clipped to `room`, ascending, without duplicates or
/// non-positive amounts.
fn sized_amounts(fractions: &[f64], pot: Chips, room: Chips) -> Vec<Chips> {
    let mut amounts: Vec<Chips> = fractions
        .iter()
        .map(|&frac| pot.fraction(frac).min(room))
        .filter(|&amt| amt > Chips::ZERO)
        .collect();
    amounts.sort_unstable();
    amounts.dedup();
    amounts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card_from_str;

    fn river_board() -> Vec<Card> {
        ["2h", "7c", "9d", "Jc", "4s"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect()
    }

    fn turn_board() -> Vec<Card> {
        ["2h", "7c", "9d", "Jc"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect()
    }

    /// Public history as `(player, actions)` per decision plus terminal
    /// investments, with amounts divided by `scale` (which must divide them).
    fn shape(node: &Node, scale: i64, out: &mut Vec<String>) {
        let unscale = |amount: Chips| {
            assert_eq!(
                amount.0 % scale,
                0,
                "{amount:?} is not a multiple of {scale}"
            );
            Chips(amount.0 / scale)
        };
        match node {
            Node::Decision {
                player,
                actions,
                children,
                ..
            } => {
                let actions: Vec<Action> = actions
                    .iter()
                    .map(|action| match *action {
                        Action::Bet(amount) => Action::Bet(unscale(amount)),
                        Action::Raise(amount) => Action::Raise(unscale(amount)),
                        other => other,
                    })
                    .collect();
                out.push(format!("{player:?} {actions:?}"));
                children.iter().for_each(|child| shape(child, scale, out));
            }
            Node::Chance { children } => children
                .iter()
                .for_each(|(_, child)| shape(child, scale, out)),
            Node::Terminal(Terminal::Fold { invested, .. } | Terminal::Showdown { invested }) => {
                out.push(format!("{:?}", invested.map(|chips| chips / scale as f64)));
            }
        }
    }

    /// Replays every path's stacks; asserts amounts are positive, stacks never
    /// go negative, and nobody acts again once an all-in has been called.
    fn check_stacks(node: &Node, stack: [Chips; 2], facing: Option<Chips>, all_in: bool) {
        match node {
            Node::Decision {
                player,
                actions,
                children,
                ..
            } => {
                assert!(!all_in, "decision after a called all-in");
                let p = player.index();
                for (action, child) in actions.iter().zip(children) {
                    let mut next = stack;
                    let (paid, next_facing) = match *action {
                        Action::Check | Action::Fold => (Chips::ZERO, None),
                        Action::Call => (facing.unwrap(), None),
                        Action::Bet(amount) => (amount, Some(amount)),
                        Action::Raise(extra) => (facing.unwrap() + extra, Some(extra)),
                    };
                    if let Action::Bet(amount) | Action::Raise(amount) = *action {
                        assert!(amount > Chips::ZERO);
                    }
                    next[p] -= paid;
                    assert!(next[p] >= Chips::ZERO, "stack went negative");
                    let called_all_in = matches!(action, Action::Call) && next == [Chips::ZERO; 2];
                    check_stacks(child, next, next_facing, called_all_in);
                }
            }
            Node::Chance { children } => children
                .iter()
                .for_each(|(_, child)| check_stacks(child, stack, None, all_in)),
            Node::Terminal(_) => {}
        }
    }

    fn config(
        board: Vec<Card>,
        pot: Chips,
        stack: Chips,
        bet_sizes: Vec<f64>,
        max_raises: u32,
    ) -> GameConfig {
        GameConfig {
            board,
            pot,
            effective_stack: stack,
            bet_sizes,
            max_raises,
            suit_isomorphism: true,
        }
    }

    fn scaled(cfg: &GameConfig, scale: i64) -> GameConfig {
        GameConfig {
            pot: Chips(cfg.pot.0 * scale),
            effective_stack: Chips(cfg.effective_stack.0 * scale),
            ..cfg.clone()
        }
    }

    #[test]
    fn all_in_leaves_no_residual_stack_at_any_scale() {
        // With f64 chips, pot 5.5 / stack 9 left an 8.88e-16 stack after an
        // all-in and built `Raise(8.88e-16)`; scaling by 10 hid it.
        for (pot, stack) in [(5500, 9000), (1500, 3900)] {
            for board in [river_board(), turn_board()] {
                let cfg = config(board, Chips(pot), Chips(stack), vec![0.33, 0.75, 1.5], 3);
                let base = build_tree(&cfg).unwrap().num_decision_nodes;
                for scale in [1, 10, 100] {
                    let tree = build_tree(&scaled(&cfg, scale)).unwrap();
                    assert_eq!(
                        tree.num_decision_nodes, base,
                        "pot {pot} stack {stack} x{scale}"
                    );
                    check_stacks(
                        &tree.root,
                        [cfg.effective_stack; 2].map(|s| Chips(s.0 * scale)),
                        None,
                        false,
                    );
                }
            }
        }
        let river = config(
            river_board(),
            Chips(5500),
            Chips(9000),
            vec![0.33, 0.75, 1.5],
            3,
        );
        assert_eq!(build_tree(&river).unwrap().num_decision_nodes, 26);
    }

    #[test]
    fn tree_shape_is_scale_invariant_when_sizes_need_no_rounding() {
        // Whole-chip pots and stacks with half-pot multiples stay integral in
        // 0.001 units for three halvings (bet + two raises on one street), so
        // every scaled tree must be the base tree with amounts multiplied.
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        for _ in 0..200 {
            let pot = Chips::whole(1 + next(40) as i64);
            let stack = Chips::whole(1 + next(120) as i64);
            let sizes: Vec<f64> = (0..1 + next(3))
                .map(|_| [0.5, 1.0, 1.5, 2.0][next(4) as usize])
                .collect();
            let cfg = config(river_board(), pot, stack, sizes, next(3) as u32);
            let mut base = Vec::new();
            shape(&build_tree(&cfg).unwrap().root, 1, &mut base);
            for scale in [10, 100] {
                let mut other = Vec::new();
                shape(
                    &build_tree(&scaled(&cfg, scale)).unwrap().root,
                    scale,
                    &mut other,
                );
                assert_eq!(base, other, "{cfg:?} x{scale}");
            }
        }
    }

    #[test]
    fn arbitrary_thousandth_amounts_keep_stacks_exact() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |bound: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % bound
        };
        for _ in 0..300 {
            let pot = Chips(1 + next(20_000) as i64);
            let stack = Chips(1 + next(60_000) as i64);
            let sizes: Vec<f64> = (0..1 + next(4))
                .map(|_| (1 + next(300)) as f64 / 100.0)
                .collect();
            let cfg = config(river_board(), pot, stack, sizes, next(4) as u32);
            check_stacks(&build_tree(&cfg).unwrap().root, [stack; 2], None, false);
        }
    }

    #[test]
    fn bounded_tree_preserves_ids_actions_and_exact_boundary() {
        let cfg = GameConfig {
            board: turn_board(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![0.5, 1.0],
            max_raises: 1,
            suit_isomorphism: true,
        };
        let original = build_tree(&cfg).unwrap();
        let bounded = build_tree_bounded(&cfg, original.num_decision_nodes).unwrap();
        assert_eq!(
            format!("{:?}", original.root),
            format!("{:?}", bounded.root)
        );
        assert_eq!(original.num_decision_nodes, bounded.num_decision_nodes);
        assert!(build_tree_bounded(&cfg, original.num_decision_nodes - 1).is_err());
        assert!(build_tree_bounded(&cfg, 0).is_err());
    }

    #[test]
    fn bounded_tree_aborts_large_flop_construction() {
        let cfg = GameConfig {
            board: turn_board()[..3].to_vec(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(100000),
            bet_sizes: vec![0.25, 0.5, 1.0, 2.0],
            max_raises: 4,
            suit_isomorphism: false,
        };
        let error = build_tree_bounded(&cfg, 100)
            .err()
            .expect("must exceed construction budget");
        assert_eq!(error, TreeError::TooManyDecisionNodes { limit: 100 });
        assert!(error.to_string().contains("100 decision nodes"));
    }

    #[test]
    fn check_check_ends_in_showdown_on_the_river() {
        let cfg = GameConfig {
            board: river_board(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![1.0],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let tree = build_tree(&cfg).unwrap();
        let Node::Decision {
            actions, children, ..
        } = &tree.root
        else {
            panic!("expected decision")
        };
        assert_eq!(actions.len(), 2);
        let Node::Decision {
            player: ip_player,
            actions: ip_actions,
            children: ip_children,
            ..
        } = &children[0]
        else {
            panic!("expected IP decision after check")
        };
        assert_eq!(*ip_player, Player::Ip);
        assert_eq!(ip_actions.len(), 2);
        assert!(matches!(
            ip_children[0],
            Node::Terminal(Terminal::Showdown { .. })
        ));
    }

    #[test]
    fn bet_size_is_pot_fraction_and_call_ends_in_showdown_on_the_river() {
        let cfg = GameConfig {
            board: river_board(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![1.0],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let tree = build_tree(&cfg).unwrap();
        let Node::Decision {
            actions, children, ..
        } = &tree.root
        else {
            panic!("expected decision")
        };
        let Action::Bet(amount) = actions[1] else {
            panic!("expected a bet action")
        };
        assert_eq!(amount, Chips::whole(100));

        let Node::Decision {
            actions: ip_actions,
            children: ip_children,
            ..
        } = &children[1]
        else {
            panic!("expected IP decision facing the bet")
        };
        assert_eq!(ip_actions.len(), 2);
        assert!(matches!(
            ip_children[0],
            Node::Terminal(Terminal::Fold { .. })
        ));
        assert!(matches!(
            ip_children[1],
            Node::Terminal(Terminal::Showdown { .. })
        ));
    }

    #[test]
    fn raises_are_capped_by_max_raises_per_street() {
        let cfg = GameConfig {
            board: river_board(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(1000),
            bet_sizes: vec![1.0],
            max_raises: 1,
            suit_isomorphism: false,
        };
        let tree = build_tree(&cfg).unwrap();
        let Node::Decision { children, .. } = &tree.root else {
            panic!("expected decision")
        };
        let Node::Decision {
            actions,
            children: ip_children,
            ..
        } = &children[1]
        else {
            panic!("expected IP decision facing the bet")
        };
        assert_eq!(actions.len(), 3);
        let Node::Decision {
            actions: oop_actions,
            ..
        } = &ip_children[2]
        else {
            panic!("expected OOP decision facing the raise")
        };
        assert_eq!(oop_actions.len(), 2);
    }

    #[test]
    fn check_check_on_the_turn_deals_a_chance_node_with_48_river_cards() {
        let cfg = GameConfig {
            board: turn_board(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![1.0],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let tree = build_tree(&cfg).unwrap();
        let Node::Decision { children, .. } = &tree.root else {
            panic!("expected decision")
        };
        let Node::Decision {
            children: ip_children,
            ..
        } = &children[0]
        else {
            panic!("expected IP decision after check")
        };
        // Check-check on the turn (4-card board) should deal the river, not show down directly.
        let Node::Chance {
            children: river_deals,
        } = &ip_children[0]
        else {
            panic!("expected a chance node dealing the river")
        };
        // With isomorphism disabled, every class is a singleton: one per undealt card.
        assert_eq!(river_deals.len(), 48); // 52 - 4 board cards
        for (members, _) in river_deals {
            assert_eq!(members.len(), 1);
        }

        // Every river deal leads into a fresh betting round (OOP decision), not straight to showdown.
        for (_, node) in river_deals {
            assert!(matches!(
                node,
                Node::Decision {
                    player: Player::Oop,
                    ..
                }
            ));
        }
    }

    #[test]
    fn river_deals_never_repeat_an_existing_board_card() {
        let cfg = GameConfig {
            board: turn_board(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![1.0],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let tree = build_tree(&cfg).unwrap();
        let Node::Decision { children, .. } = &tree.root else {
            panic!("expected decision")
        };
        let Node::Decision {
            children: ip_children,
            ..
        } = &children[0]
        else {
            panic!("expected IP decision after check")
        };
        let Node::Chance {
            children: river_deals,
        } = &ip_children[0]
        else {
            panic!("expected a chance node dealing the river")
        };
        let board_set = set_from_cards(&turn_board());
        for (members, _) in river_deals {
            for &card in members {
                assert!(!set_contains(board_set, card));
            }
        }
    }

    #[test]
    fn suit_isomorphism_groups_free_suit_cards_of_the_same_rank() {
        // Board uses only clubs and diamonds -> hearts and spades are free.
        let board: Vec<Card> = ["2c", "7d", "9c"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect();
        let cfg = GameConfig {
            board: board.clone(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(200),
            bet_sizes: vec![1.0],
            max_raises: 0,
            suit_isomorphism: true,
        };
        let tree = build_tree(&cfg).unwrap();
        let Node::Decision { children, .. } = &tree.root else {
            panic!("expected decision")
        };
        let Node::Decision {
            children: ip_children,
            ..
        } = &children[0]
        else {
            panic!("expected IP decision after check")
        };
        let Node::Chance { children: deals } = &ip_children[0] else {
            panic!("expected a chance node dealing the turn")
        };

        // 49 undealt cards; hearts and spades are free. Rather than
        // hand-derive the exact class breakdown, check the structural
        // invariants directly: every class shares one rank, only free-suit
        // (h/s) cards are ever grouped into a class of size > 1, and every
        // undealt card appears in exactly one class.
        let used = set_from_cards(&board);
        let mut seen = std::collections::HashSet::new();
        let mut total_members = 0;
        for (members, _) in deals {
            assert!(!members.is_empty());
            let ranks: std::collections::HashSet<u8> =
                members.iter().map(|&c| crate::card::rank(c)).collect();
            assert_eq!(
                ranks.len(),
                1,
                "every class member must share the same rank"
            );
            if members.len() > 1 {
                for &c in members {
                    let s = crate::card::suit(c);
                    assert!(
                        s == 2 || s == 3,
                        "grouped members must use a free suit (h=2, s=3)"
                    );
                }
            }
            for &c in members {
                assert!(
                    !set_contains(used, c),
                    "class must not include an already-used card"
                );
                assert!(
                    seen.insert(c),
                    "each undealt card must appear in exactly one class"
                );
            }
            total_members += members.len();
        }
        assert_eq!(total_members, 52 - board.len());
        // At least one class should actually group 2 free-suit cards together
        // (that's the whole point of isomorphism reduction).
        assert!(deals.iter().any(|(members, _)| members.len() == 2));
        // No class can exceed the number of free suits (2 here).
        assert!(deals.iter().all(|(members, _)| members.len() <= 2));
    }
}
