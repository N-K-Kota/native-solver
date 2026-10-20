//! Independent, exact strategy-quality reference for small heads-up trees.
//!
//! This deliberately does not call [`Trainer`](crate::Trainer) traversal, chance, blocker, or
//! terminal kernels. It enumerates legal private deals, evaluates showdowns
//! with [`evaluate_7`](crate::evaluate_7), and solves a separate decision problem for each own hand.
//! Opponent states are summed **before** maximizing at an information set:
//! a best response observes its own cards and public history, never the other
//! player's cards. Public deals condition on both players' private cards.
//!
//! Only complete, singleton-card chance trees are supported. Suit-isomorphism
//! representatives and sampled/incomplete chance trees are rejected. Both
//! players may deviate independently per combo, even when the supplied policy
//! was expanded from buckets. Thus this measures exploitability in the full
//! private-card game on the supplied *betting tree*, not in a bucket-restricted
//! game, and does not measure omitted bet sizes or other action-tree error.

use super::chips::Chips;
use super::tree::{Action, Node, Terminal};
use crate::card::{bit, Card};
use crate::eval::score_7;
use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::range::Combo;
type Policy = HashMap<usize, Vec<Vec<f64>>>;

/// Normalized quality of a fixed behavioral strategy profile.
///
/// All values are in chips, including half the initial pot in each player's
/// invested baseline, as in [`Terminal`]. Expectations are over legal initial
/// private deals with probability `w_oop * w_ip / joint_mass` and uniform legal
/// runouts. `profile_ev` is the profile's value for each player;
/// `best_response_ev[p]` holds the other player's policy fixed while player p
/// deviates. NashConv is the sum of the two unilateral gains, and exploitability
/// is NashConv / 2. For this zero-sum game NashConv also equals the sum of BR EVs.
/// No value is clipped to conceal numerical error or a negative gain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StrategyQuality {
    /// `[OOP, IP]` value of the supplied profile.
    pub profile_ev: [f64; 2],
    /// `[OOP, IP]` value of each player's best response to the other's policy.
    pub best_response_ev: [f64; 2],
    /// Sum of both players' gains from deviating to a best response.
    pub nash_conv: f64,
    /// `nash_conv / 2`.
    pub exploitability: f64,
    /// Unnormalized sum of products of compatible initial range weights.
    pub joint_mass: f64,
}

/// Invalid or unsupported reference-evaluation input.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum QualityError {
    /// The initial board has duplicate or out-of-range cards, or an
    /// unsupported length.
    InvalidBoard(String),
    /// A combo has invalid or repeated cards, or a weight is negative or
    /// not finite.
    InvalidRange {
        /// The player index whose range was rejected (0 = OOP, 1 = IP).
        player: usize,
        /// What was wrong.
        reason: String,
    },
    /// No legal pair of private hands has positive finite joint weight.
    NoCompatibleDeals,
    /// The tree is malformed for the given board.
    InvalidTree(String),
    /// A chance node groups several cards under one representative; expand
    /// suit-isomorphism reduction first.
    UnsupportedChanceReduction,
    /// The supplied policy has the wrong shape or is not a distribution.
    InvalidStrategy {
        /// The decision node with the invalid policy.
        node_id: usize,
        /// What was wrong.
        reason: String,
    },
    /// Evaluation produced a NaN or infinite value.
    NonFiniteResult,
}

impl fmt::Display for QualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBoard(reason) => write!(f, "invalid initial board: {reason}"),
            Self::InvalidRange { player, reason } => write!(f, "invalid player {player} range: {reason}"),
            Self::NoCompatibleDeals => write!(f, "legal private deals must have positive finite joint mass"),
            Self::InvalidTree(reason) => write!(f, "invalid reference tree: {reason}"),
            Self::UnsupportedChanceReduction => write!(f, "reference evaluation requires singleton chance classes; expand suit-isomorphism reduction first"),
            Self::InvalidStrategy { node_id, reason } => write!(f, "invalid strategy at node {node_id}: {reason}"),
            Self::NonFiniteResult => write!(f, "reference evaluation produced a nonfinite value"),
        }
    }
}

impl std::error::Error for QualityError {}

/// An independent, exact evaluator of strategy quality for small trees.
///
/// It shares no traversal code with [`Trainer`](crate::Trainer), so it can check
/// the trainer's results. Showdown comparisons are cached once per evaluator.
/// Time and memory grow with the tree size times the number of private-hand
/// pairs, so use it for small trees: tests, fixtures, and quality benchmarks.
///
/// # Examples
///
/// ```
/// use native_solver::solver::quality::ReferenceEvaluator;
/// use native_solver::{build_tree, Card, CardSet, Chips, GameConfig, Player, Range, Trainer, TrainerOptions};
///
/// let board = ["Ks", "Th", "7s", "4d", "2s"]
///     .iter()
///     .map(|card| card.parse())
///     .collect::<Result<Vec<Card>, _>>()?;
/// let dead = CardSet::from_cards(&board);
/// let config = GameConfig::new(board.clone(), Chips::whole(100), Chips::whole(100)).with_bet_sizes([1.0]);
/// let ranges = ["AA,KQs", "QQ,JJ"].map(|r| r.parse::<Range>().unwrap().combos(dead));
/// let mut trainer = Trainer::new(build_tree(&config)?, ranges, TrainerOptions::default())?;
/// trainer.train(200);
///
/// // Check the trainer's exploitability against the independent reference.
/// let game = trainer.game();
/// let reference = ReferenceEvaluator::new(game.tree(), &board, Player::ALL.map(|p| game.combos(p)))?;
/// let quality = reference.evaluate(|node| trainer.average_strategy(node))?;
/// assert!((quality.exploitability - trainer.exploitability().total).abs() < 1e-9);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ReferenceEvaluator<'a> {
    root: &'a Node,
    initial_board: &'a [Card],
    combos: [&'a [Combo]; 2],
    masks: [Vec<u64>; 2],
    /// Row-major OOP x IP normalized joint prior; impossible pairs have zero.
    prior: Vec<f64>,
    joint_mass: f64,
    decisions: Vec<(usize, usize, usize)>,
    /// Row-major OOP x IP showdown ordering (-1, 0, 1) for each complete board.
    outcomes: HashMap<u64, Vec<i8>>,
}

impl std::fmt::Debug for ReferenceEvaluator<'_> {
    /// A summary: board, range sizes, decisions, and joint mass.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReferenceEvaluator")
            .field("initial_board", &self.initial_board)
            .field("num_combos", &[self.combos[0].len(), self.combos[1].len()])
            .field("num_decisions", &self.decisions.len())
            .field("joint_mass", &self.joint_mass)
            .finish_non_exhaustive()
    }
}

impl<'a> ReferenceEvaluator<'a> {
    /// Validate the board, ranges, and complete unreduced tree.
    ///
    /// Finite nonnegative weights need not sum to one or be at most one.
    /// Board-blocked combos are accepted but excluded, as are conflicting
    /// private pairs and zero-weight combos. Decision IDs must be unique
    /// public histories. Terminal payoffs and action labels define the
    /// supplied game; this checks structural integrity, not whether the
    /// betting sequence follows NLHE rules.
    ///
    /// # Errors
    /// Returns a [`QualityError`] for a board that is not 3–5 distinct cards,
    /// duplicate or reversed combos, a combo holding one card twice,
    /// nonfinite or negative weights, zero or nonfinite legal joint mass, a
    /// malformed tree, or a suit-reduced chance node.
    pub fn new(
        root: &'a Node,
        initial_board: &'a [Card],
        combos: [&'a [Combo]; 2],
    ) -> Result<Self, QualityError> {
        if !(3..=5).contains(&initial_board.len()) {
            return Err(QualityError::InvalidBoard("expected 3 to 5 cards".into()));
        }
        let mut board_mask = 0u64;
        for &card in initial_board {
            if board_mask & bit(card) != 0 {
                return Err(QualityError::InvalidBoard("cards must be distinct".into()));
            }
            board_mask |= bit(card);
        }
        let mut masks = [Vec::new(), Vec::new()];
        for player in 0..2 {
            let mut seen = HashSet::new();
            for &Combo {
                cards: [a, b],
                weight,
            } in combos[player]
            {
                let reason = if a == b {
                    Some("a combo must contain two distinct cards")
                } else if !weight.is_finite() || weight < 0.0 {
                    Some("range weights must be finite and nonnegative")
                } else if !seen.insert((a.min(b), a.max(b))) {
                    Some("duplicate private combo")
                } else {
                    None
                };
                if let Some(reason) = reason {
                    return Err(QualityError::InvalidRange {
                        player,
                        reason: reason.into(),
                    });
                }
                masks[player].push((bit(a)) | (bit(b)));
            }
        }
        let mut prior = Vec::new();
        for (i, a) in combos[0].iter().enumerate() {
            for (j, b) in combos[1].iter().enumerate() {
                let legal =
                    (masks[0][i] | masks[1][j]) & board_mask == 0 && masks[0][i] & masks[1][j] == 0;
                prior.push(if legal { a.weight * b.weight } else { 0.0 });
            }
        }
        let joint_mass: f64 = prior.iter().sum();
        if !joint_mass.is_finite() || joint_mass <= 0.0 {
            return Err(QualityError::NoCompatibleDeals);
        }
        for probability in &mut prior {
            *probability /= joint_mass;
        }
        let mut evaluator = Self {
            root,
            initial_board,
            combos,
            masks,
            prior,
            joint_mass,
            decisions: Vec::new(),
            outcomes: HashMap::new(),
        };
        evaluator.validate_tree(root, &mut initial_board.to_vec(), &mut HashSet::new(), 0)?;
        Ok(evaluator)
    }

    /// Measures the profile given by `strategy`: its EVs and exploitability.
    ///
    /// `strategy` is called once per decision node and returns
    /// `[action][combo]` probabilities in the input range order, for example
    /// `|id| trainer.average_strategy(id)`.
    ///
    /// Every column, including unreachable/blocked columns, must have finite
    /// probabilities in [0, 1] summing to one within 1e-10. Accepted roundoff is
    /// normalized to exactly one. The callback is not consulted during BR
    /// search, and no trainer state or regret tables are read or modified.
    ///
    /// # Errors
    /// Returns [`QualityError::InvalidStrategy`] for a strategy with the
    /// wrong shape or a column that is not a distribution, and
    /// [`QualityError::NonFiniteResult`] if evaluation overflows.
    pub fn evaluate<F>(&self, mut strategy: F) -> Result<StrategyQuality, QualityError>
    where
        F: FnMut(usize) -> Vec<Vec<f64>>,
    {
        let mut policies = HashMap::new();
        for &(id, player, actions) in &self.decisions {
            let mut policy = strategy(id);
            let invalid = |reason: &str| QualityError::InvalidStrategy {
                node_id: id,
                reason: reason.into(),
            };
            if policy.len() != actions
                || policy
                    .iter()
                    .any(|row| row.len() != self.combos[player].len())
            {
                return Err(invalid("expected [action][combo] dimensions"));
            }
            for combo in 0..self.combos[player].len() {
                let sum: f64 = policy.iter().map(|row| row[combo]).sum();
                if policy
                    .iter()
                    .any(|row| !row[combo].is_finite() || !(0.0..=1.0).contains(&row[combo]))
                    || !sum.is_finite()
                    || (sum - 1.0).abs() > 1e-10
                {
                    return Err(invalid(
                        "probability columns must be finite, nonnegative, and sum to one",
                    ));
                }
                for row in &mut policy {
                    row[combo] /= sum;
                }
            }
            policies.insert(id, policy);
        }
        let mut profile_ev = [0.0; 2];
        let mut best_response_ev = [0.0; 2];
        let board_mask = mask(self.initial_board);
        for player in 0..2 {
            for own in 0..self.combos[player].len() {
                let reach: Vec<f64> = (0..self.combos[1 - player].len())
                    .map(|opp| self.prior[self.pair_index(player, own, opp)])
                    .collect();
                if reach.iter().all(|&weight| weight == 0.0) {
                    continue;
                }
                let problem = PrivateProblem {
                    evaluator: self,
                    policies: &policies,
                    player,
                    own,
                };
                profile_ev[player] += problem.value(self.root, board_mask, &reach, false);
                best_response_ev[player] += problem.value(self.root, board_mask, &reach, true);
            }
        }
        let nash_conv =
            (best_response_ev[0] - profile_ev[0]) + (best_response_ev[1] - profile_ev[1]);
        let result = StrategyQuality {
            profile_ev,
            best_response_ev,
            nash_conv,
            exploitability: nash_conv / 2.0,
            joint_mass: self.joint_mass,
        };
        if profile_ev
            .iter()
            .chain(&best_response_ev)
            .chain([&nash_conv])
            .any(|x| !x.is_finite())
        {
            return Err(QualityError::NonFiniteResult);
        }
        Ok(result)
    }

    fn pair_index(&self, player: usize, own: usize, opp: usize) -> usize {
        if player == 0 {
            own * self.combos[1].len() + opp
        } else {
            opp * self.combos[1].len() + own
        }
    }

    fn validate_tree(
        &mut self,
        node: &Node,
        board: &mut Vec<Card>,
        ids: &mut HashSet<usize>,
        depth: usize,
    ) -> Result<(), QualityError> {
        let invalid = |reason: &str| QualityError::InvalidTree(reason.into());
        if depth > 256 {
            return Err(invalid("reference traversal depth exceeds 256"));
        }
        match node {
            Node::Decision {
                id,
                player,
                actions,
                children,
            } => {
                if actions.is_empty() || actions.len() != children.len() {
                    return Err(invalid(
                        "decisions need matching nonempty actions and children",
                    ));
                }
                if !ids.insert(*id) {
                    return Err(invalid("decision IDs must be unique public histories"));
                }
                if actions.iter().any(|action| matches!(action, Action::Bet(amount) | Action::Raise(amount) if *amount <= Chips::ZERO)) {
                    return Err(invalid("bet and raise amounts must be positive"));
                }
                self.decisions.push((*id, player.index(), actions.len()));
                for child in children {
                    self.validate_tree(child, board, ids, depth + 1)?;
                }
            }
            Node::Chance { children } => {
                if board.len() >= 5 {
                    return Err(invalid("cannot deal past a five-card board"));
                }
                if children.iter().any(|(members, _)| members.len() > 1) {
                    return Err(QualityError::UnsupportedChanceReduction);
                }
                if children.len() != 52 - board.len() {
                    return Err(invalid(
                        "chance must enumerate every card outside the public board",
                    ));
                }
                let mut seen = mask(board);
                for (members, child) in children {
                    if members.len() != 1 || seen & bit(members[0]) != 0 {
                        return Err(invalid(
                            "chance cards must be distinct singletons outside the public board",
                        ));
                    }
                    seen |= bit(members[0]);
                    board.push(members[0]);
                    self.validate_tree(child, board, ids, depth + 1)?;
                    board.pop();
                }
            }
            Node::Terminal(terminal) => {
                let invested = match terminal {
                    Terminal::Fold { invested, .. } | Terminal::Showdown { invested } => invested,
                };
                if invested
                    .iter()
                    .any(|amount| !amount.is_finite() || *amount < 0.0)
                {
                    return Err(invalid(
                        "terminal investments must be finite and nonnegative",
                    ));
                }
                if matches!(terminal, Terminal::Showdown { .. }) {
                    if board.len() != 5 {
                        return Err(invalid("showdown requires a complete five-card board"));
                    }
                    self.cache_outcomes(board);
                }
            }
        }
        Ok(())
    }

    fn cache_outcomes(&mut self, board: &[Card]) {
        let board_mask = mask(board);
        self.outcomes.entry(board_mask).or_insert_with(|| {
            let scores: [Vec<Option<u32>>; 2] = std::array::from_fn(|player| {
                self.combos[player]
                    .iter()
                    .enumerate()
                    .map(|(i, &Combo { cards: [a, b], .. })| {
                        (self.masks[player][i] & board_mask == 0).then(|| {
                            score_7([a, b, board[0], board[1], board[2], board[3], board[4]])
                        })
                    })
                    .collect()
            });
            let mut outcomes = Vec::new();
            for (i, a) in scores[0].iter().enumerate() {
                for (j, b) in scores[1].iter().enumerate() {
                    outcomes.push(match (a, b) {
                        (Some(a), Some(b)) if self.masks[0][i] & self.masks[1][j] == 0 => {
                            match a.cmp(b) {
                                std::cmp::Ordering::Less => -1,
                                std::cmp::Ordering::Equal => 0,
                                std::cmp::Ordering::Greater => 1,
                            }
                        }
                        _ => 0, // Impossible pairs have zero prior/reach.
                    });
                }
            }
            outcomes
        });
    }
}

struct PrivateProblem<'a, 'b> {
    evaluator: &'b ReferenceEvaluator<'a>,
    policies: &'b Policy,
    player: usize,
    own: usize,
}

impl PrivateProblem<'_, '_> {
    fn value(&self, node: &Node, board: u64, reach: &[f64], best_response: bool) -> f64 {
        if reach.iter().all(|&weight| weight == 0.0) {
            return 0.0;
        }
        match node {
            Node::Terminal(terminal) => reach
                .iter()
                .enumerate()
                .map(|(opp, &weight)| {
                    if weight == 0.0 {
                        return 0.0;
                    }
                    let payoff = match terminal {
                        Terminal::Fold { winner, invested } => {
                            if winner.index() == self.player {
                                invested[1 - self.player]
                            } else {
                                -invested[self.player]
                            }
                        }
                        Terminal::Showdown { invested } => {
                            let oop_order = self.evaluator.outcomes[&board]
                                [self.evaluator.pair_index(self.player, self.own, opp)];
                            let own_order = if self.player == 0 {
                                oop_order
                            } else {
                                -oop_order
                            };
                            match own_order {
                                1 => invested[1 - self.player],
                                -1 => -invested[self.player],
                                _ => (invested[1 - self.player] - invested[self.player]) / 2.0,
                            }
                        }
                    };
                    weight * payoff
                })
                .sum(),
            Node::Chance { children } => {
                // A fixed legal private pair removes four additional cards.
                // Carry the branch probability in each hidden opponent state,
                // preserving its posterior for every subsequent decision.
                let probability = 1.0 / (52 - board.count_ones() - 4) as f64;
                children
                    .iter()
                    .map(|(members, child)| {
                        let card = bit(members[0]);
                        if self.evaluator.masks[self.player][self.own] & card != 0 {
                            return 0.0;
                        }
                        let next: Vec<f64> = reach
                            .iter()
                            .enumerate()
                            .map(|(opp, &weight)| {
                                if self.evaluator.masks[1 - self.player][opp] & card == 0 {
                                    weight * probability
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        self.value(child, board | card, &next, best_response)
                    })
                    .sum()
            }
            Node::Decision {
                id,
                player,
                children,
                ..
            } => {
                let policy = &self.policies[id];
                if player.index() == self.player {
                    let values = children
                        .iter()
                        .map(|child| self.value(child, board, reach, best_response));
                    if best_response {
                        // One action per own hand/public history, after summing
                        // every opponent state. A per-pair max is clairvoyance.
                        values.fold(f64::NEG_INFINITY, f64::max)
                    } else {
                        values
                            .enumerate()
                            .map(|(action, value)| policy[action][self.own] * value)
                            .sum()
                    }
                } else {
                    children
                        .iter()
                        .enumerate()
                        .map(|(action, child)| {
                            let next: Vec<f64> = reach
                                .iter()
                                .enumerate()
                                .map(|(opp, &weight)| weight * policy[action][opp])
                                .collect();
                            self.value(child, board, &next, best_response)
                        })
                        .sum()
                }
            }
        }
    }
}

fn mask(cards: &[Card]) -> u64 {
    cards.iter().fold(0, |mask, &card| mask | (bit(card)))
}
