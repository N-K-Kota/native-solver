//! Heads-up preflop betting tree.
//!
//! The small blind is the button: it acts first before the flop and is
//! [`Player::Ip`], the player in position after it. The big blind is
//! [`Player::Oop`]. The tree holds the preflop betting round only. A fold
//! ends the hand as usual. Once a bet is called, or the big blind checks
//! behind a limp, no cards are dealt inside the tree: the hand ends in a
//! [`Terminal::Showdown`] on an empty board, which the solver settles over
//! every five-card runout. That is exact when the call puts a player all in,
//! and otherwise models a hand that is checked down from the flop on.

use super::chips::Chips;
use super::tree::{Action, BuildBudget, Node, Player, Terminal, Tree, TreeError};

/// Blinds, stacks, and raise options of a heads-up preflop tree.
///
/// By default nobody can raise, so the small blind only folds or completes.
/// Add raise sizes, the all-in, and a raise cap with the `with_*` methods.
///
/// # Examples
///
/// A push-or-fold game with 10 big blinds:
///
/// ```
/// use native_solver::{build_preflop_tree, Action, Chips, Node, Player, PreflopConfig};
///
/// let config = PreflopConfig::new(Chips::from_f64(0.5).unwrap(), Chips::whole(1), Chips::whole(10))
///     .with_all_in(true)
///     .with_max_raises(1)
///     .with_limp(false);
/// let tree = build_preflop_tree(&config)?;
/// assert!(tree.board.is_empty());
///
/// // The small blind acts first: fold, or raise 9 more to 10.
/// let Node::Decision { player, actions, .. } = &tree.root else { unreachable!() };
/// assert_eq!(*player, Player::Ip);
/// assert_eq!(actions, &[Action::Fold, Action::Raise(Chips::whole(9))]);
/// # Ok::<(), native_solver::TreeError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PreflopConfig {
    small_blind: Chips,
    big_blind: Chips,
    effective_stack: Chips,
    raise_sizes: Vec<f64>,
    all_in: bool,
    max_raises: u32,
    limp: bool,
}

impl PreflopConfig {
    /// A game with these blinds where each player starts the hand with
    /// `effective_stack` chips, blind included. No raises are allowed until
    /// [`with_max_raises`](Self::with_max_raises) says so.
    pub fn new(small_blind: Chips, big_blind: Chips, effective_stack: Chips) -> PreflopConfig {
        PreflopConfig {
            small_blind,
            big_blind,
            effective_stack,
            raise_sizes: Vec::new(),
            all_in: false,
            max_raises: 0,
            limp: true,
        }
    }

    /// Sets the raise sizes, each as a multiple of the bet being raised:
    /// `2.5` opens to 2.5 big blinds and re-raises a bet of 6 to 15.
    ///
    /// A size below the legal minimum raise is raised to it, and one beyond
    /// the stack becomes an all-in. Sizes that end up equal are offered once.
    pub fn with_raise_sizes(mut self, raise_sizes: impl Into<Vec<f64>>) -> PreflopConfig {
        self.raise_sizes = raise_sizes.into();
        self
    }

    /// Whether every raise decision also offers the all-in.
    pub fn with_all_in(mut self, all_in: bool) -> PreflopConfig {
        self.all_in = all_in;
        self
    }

    /// Caps the number of raises in the hand. The first raise over the big
    /// blind counts as one.
    pub fn with_max_raises(mut self, max_raises: u32) -> PreflopConfig {
        self.max_raises = max_raises;
        self
    }

    /// Whether the small blind may complete to the big blind (`true` by
    /// default). Turn it off for a raise-or-fold game.
    pub fn with_limp(mut self, limp: bool) -> PreflopConfig {
        self.limp = limp;
        self
    }

    /// The small blind, posted by [`Player::Ip`].
    pub fn small_blind(&self) -> Chips {
        self.small_blind
    }

    /// The big blind, posted by [`Player::Oop`].
    pub fn big_blind(&self) -> Chips {
        self.big_blind
    }

    /// Each player's chips at the start of the hand, blind included.
    pub fn effective_stack(&self) -> Chips {
        self.effective_stack
    }

    /// Raise sizes as multiples of the bet being raised.
    pub fn raise_sizes(&self) -> &[f64] {
        &self.raise_sizes
    }

    /// Whether every raise decision also offers the all-in.
    pub fn all_in(&self) -> bool {
        self.all_in
    }

    /// The cap on raises in the hand.
    pub fn max_raises(&self) -> u32 {
        self.max_raises
    }

    /// Whether the small blind may complete to the big blind.
    pub fn limp(&self) -> bool {
        self.limp
    }
}

/// Builds the heads-up preflop betting tree for `cfg`.
///
/// The returned [`Tree`] has an empty board and no chance nodes; its root is
/// the small blind's ([`Player::Ip`]'s) first decision. Pass it to
/// [`Trainer::new`](crate::Trainer::new) with ranges expanded against no
/// dead cards. See the module-level notes in [`PreflopConfig`] for how called
/// pots end.
///
/// # Errors
/// Returns [`TreeError::InvalidBlinds`] unless
/// `0 < small blind < big blind <= effective stack`, and
/// [`TreeError::InvalidRaiseSize`] for a raise size that is not a finite
/// number above `1`.
pub fn build_preflop_tree(cfg: &PreflopConfig) -> Result<Tree, TreeError> {
    build_preflop_tree_bounded(cfg, usize::MAX)
}

/// [`build_preflop_tree`], stopping once the tree would exceed
/// `max_decision_nodes`.
///
/// # Errors
/// Returns the same configuration errors as [`build_preflop_tree`], or
/// [`TreeError::TooManyDecisionNodes`] when the budget is exceeded.
pub fn build_preflop_tree_bounded(
    cfg: &PreflopConfig,
    max_decision_nodes: usize,
) -> Result<Tree, TreeError> {
    if cfg.small_blind <= Chips::ZERO
        || cfg.small_blind >= cfg.big_blind
        || cfg.big_blind > cfg.effective_stack
    {
        return Err(TreeError::InvalidBlinds);
    }
    if let Some(index) = cfg
        .raise_sizes
        .iter()
        .position(|&size| !size.is_finite() || size <= 1.0)
    {
        return Err(TreeError::InvalidRaiseSize { index });
    }
    let mut budget = BuildBudget::new(max_decision_nodes);
    let root = build(
        cfg,
        Player::Ip,
        [cfg.big_blind, cfg.small_blind],
        cfg.big_blind,
        0,
        &mut budget,
    )?;
    Ok(Tree {
        board: Vec::new(),
        root,
        num_decision_nodes: budget.next_id(),
    })
}

fn amounts(chips: [Chips; 2]) -> [f64; 2] {
    [chips[0].to_f64(), chips[1].to_f64()]
}

/// The totals a player may raise to, ascending, given the bet they face.
fn raise_totals(cfg: &PreflopConfig, bet: Chips, last_raise: Chips, raises: u32) -> Vec<Chips> {
    let stack = cfg.effective_stack;
    if raises >= cfg.max_raises || bet >= stack {
        return Vec::new();
    }
    let minimum = bet + last_raise;
    let mut totals: Vec<Chips> = cfg
        .raise_sizes
        .iter()
        .map(|&size| bet.fraction(size).max(minimum).min(stack))
        .collect();
    if cfg.all_in {
        totals.push(stack);
    }
    totals.sort_unstable();
    totals.dedup();
    totals
}

/// `player` to act having put in `put_in[player]`, indexed like
/// [`Player::index`]. `last_raise` is the size of the last raise (the big
/// blind before any), which the next raise must at least match.
fn build(
    cfg: &PreflopConfig,
    player: Player,
    put_in: [Chips; 2],
    last_raise: Chips,
    raises: u32,
    budget: &mut BuildBudget,
) -> Result<Node, TreeError> {
    let id = budget.allocate_id()?;
    let (me, other) = (player.index(), player.other().index());
    let bet = put_in[other].max(put_in[me]);
    let to_call = bet - put_in[me];
    let showdown = |put_in: [Chips; 2]| {
        Node::Terminal(Terminal::Showdown {
            invested: amounts(put_in),
        })
    };

    let mut actions = Vec::new();
    let mut children = Vec::new();
    if to_call > Chips::ZERO {
        actions.push(Action::Fold);
        children.push(Node::Terminal(Terminal::Fold {
            winner: player.other(),
            invested: amounts(put_in),
        }));

        // Completing the small blind is the only call that can leave the
        // other player a decision: the big blind's option.
        let completes = raises == 0 && player == Player::Ip;
        if !completes || cfg.limp {
            let mut called = put_in;
            called[me] = bet;
            actions.push(Action::Call);
            children.push(
                if completes && !raise_totals(cfg, bet, last_raise, raises).is_empty() {
                    build(cfg, player.other(), called, last_raise, raises, budget)?
                } else {
                    showdown(called)
                },
            );
        }
    } else {
        actions.push(Action::Check);
        children.push(showdown(put_in));
    }

    for total in raise_totals(cfg, bet, last_raise, raises) {
        let mut raised = put_in;
        raised[me] = total;
        actions.push(Action::Raise(total - bet));
        children.push(build(
            cfg,
            player.other(),
            raised,
            last_raise.max(total - bet),
            raises + 1,
            budget,
        )?);
    }

    Ok(Node::Decision {
        id,
        player,
        actions,
        children,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half() -> Chips {
        Chips::from_f64(0.5).unwrap()
    }

    fn decision(node: &Node) -> (Player, &[Action], &[Node]) {
        match node {
            Node::Decision {
                player,
                actions,
                children,
                ..
            } => (*player, actions, children),
            other => panic!("expected a decision, found {other:?}"),
        }
    }

    #[test]
    fn default_tree_is_fold_or_complete_with_no_option() {
        let tree = build_preflop_tree(&PreflopConfig::new(
            half(),
            Chips::whole(1),
            Chips::whole(20),
        ))
        .unwrap();
        assert!(tree.board.is_empty());
        assert_eq!(tree.num_decision_nodes, 1);
        let (player, actions, children) = decision(&tree.root);
        assert_eq!(player, Player::Ip);
        assert_eq!(actions, [Action::Fold, Action::Call]);
        assert_eq!(
            children[0],
            Node::Terminal(Terminal::Fold {
                winner: Player::Oop,
                invested: [1.0, 0.5],
            })
        );
        assert_eq!(
            children[1],
            Node::Terminal(Terminal::Showdown {
                invested: [1.0, 1.0],
            })
        );
    }

    #[test]
    fn limp_gives_the_big_blind_its_option() {
        let config = PreflopConfig::new(half(), Chips::whole(1), Chips::whole(20))
            .with_raise_sizes([3.0])
            .with_max_raises(2);
        let tree = build_preflop_tree(&config).unwrap();
        let (_, actions, children) = decision(&tree.root);
        assert_eq!(
            actions,
            [Action::Fold, Action::Call, Action::Raise(Chips::whole(2))]
        );

        // After the limp the big blind checks or raises to 3.
        let (player, actions, option) = decision(&children[1]);
        assert_eq!(player, Player::Oop);
        assert_eq!(actions, [Action::Check, Action::Raise(Chips::whole(2))]);
        assert_eq!(
            option[0],
            Node::Terminal(Terminal::Showdown {
                invested: [1.0, 1.0],
            })
        );
        // The small blind then folds, calls, or re-raises to 9.
        let (player, actions, facing) = decision(&option[1]);
        assert_eq!(player, Player::Ip);
        assert_eq!(
            actions,
            [Action::Fold, Action::Call, Action::Raise(Chips::whole(6))]
        );
        assert_eq!(
            facing[0],
            Node::Terminal(Terminal::Fold {
                winner: Player::Oop,
                invested: [3.0, 1.0],
            })
        );
        // The cap of two raises is spent: the big blind can only fold or call.
        let (_, actions, last) = decision(&facing[2]);
        assert_eq!(actions, [Action::Fold, Action::Call]);
        assert_eq!(
            last[1],
            Node::Terminal(Terminal::Showdown {
                invested: [9.0, 9.0],
            })
        );
    }

    #[test]
    fn sizes_clip_to_the_stack_and_merge_with_the_all_in() {
        let config = PreflopConfig::new(half(), Chips::whole(1), Chips::whole(8))
            .with_raise_sizes([2.5, 3.0, 20.0])
            .with_all_in(true)
            .with_max_raises(3);
        let tree = build_preflop_tree(&config).unwrap();
        let (_, actions, children) = decision(&tree.root);
        assert_eq!(
            actions,
            [
                Action::Fold,
                Action::Call,
                Action::Raise(Chips::from_f64(1.5).unwrap()),
                Action::Raise(Chips::whole(2)),
                Action::Raise(Chips::whole(7)),
            ]
        );
        // Facing the open to 3: 2.5x is 7.5, 3x and 20x are the all-in of 8.
        let (_, actions, _) = decision(&children[3]);
        assert_eq!(
            actions,
            [
                Action::Fold,
                Action::Call,
                Action::Raise(Chips::from_f64(4.5).unwrap()),
                Action::Raise(Chips::whole(5)),
            ]
        );
        // Facing an all-in there is nothing left to raise.
        let (_, actions, all_in) = decision(&children[4]);
        assert_eq!(actions, [Action::Fold, Action::Call]);
        assert_eq!(
            all_in[1],
            Node::Terminal(Terminal::Showdown {
                invested: [8.0, 8.0],
            })
        );
    }

    #[test]
    fn small_sizes_are_lifted_to_the_minimum_raise() {
        // 1.5x of a bet of 4 would be 6, but the last raise was 3.
        let config = PreflopConfig::new(half(), Chips::whole(1), Chips::whole(100))
            .with_raise_sizes([4.0, 1.5])
            .with_max_raises(2)
            .with_limp(false);
        let tree = build_preflop_tree(&config).unwrap();
        let (_, actions, children) = decision(&tree.root);
        // 1.5x of the big blind is lifted to the minimum raise to 2.
        assert_eq!(
            actions,
            [
                Action::Fold,
                Action::Raise(Chips::whole(1)),
                Action::Raise(Chips::whole(3)),
            ]
        );
        let (_, actions, _) = decision(&children[2]);
        assert_eq!(
            actions,
            [
                Action::Fold,
                Action::Call,
                Action::Raise(Chips::whole(3)),
                Action::Raise(Chips::whole(12)),
            ]
        );
    }

    #[test]
    fn a_stack_of_one_big_blind_leaves_only_fold_or_call() {
        let config = PreflopConfig::new(half(), Chips::whole(1), Chips::whole(1))
            .with_all_in(true)
            .with_max_raises(4);
        let tree = build_preflop_tree(&config).unwrap();
        let (_, actions, _) = decision(&tree.root);
        assert_eq!(actions, [Action::Fold, Action::Call]);
        assert_eq!(tree.num_decision_nodes, 1);
    }

    #[test]
    fn decision_ids_are_dense_and_the_budget_is_enforced() {
        let config = PreflopConfig::new(half(), Chips::whole(1), Chips::whole(100))
            .with_raise_sizes([2.5, 4.0])
            .with_all_in(true)
            .with_max_raises(4);
        let tree = build_preflop_tree(&config).unwrap();
        fn ids(node: &Node, out: &mut Vec<usize>) {
            if let Node::Decision { id, children, .. } = node {
                out.push(*id);
                children.iter().for_each(|child| ids(child, out));
            }
        }
        let mut seen = Vec::new();
        ids(&tree.root, &mut seen);
        seen.sort_unstable();
        assert!(seen.iter().copied().eq(0..tree.num_decision_nodes));
        assert_eq!(
            build_preflop_tree_bounded(&config, tree.num_decision_nodes - 1),
            Err(TreeError::TooManyDecisionNodes {
                limit: tree.num_decision_nodes - 1
            })
        );
        assert!(build_preflop_tree_bounded(&config, tree.num_decision_nodes).is_ok());
    }

    #[test]
    fn invalid_blinds_and_sizes_are_rejected() {
        let config = |sb: f64, bb: f64, stack: f64| {
            PreflopConfig::new(
                Chips::from_f64(sb).unwrap(),
                Chips::from_f64(bb).unwrap(),
                Chips::from_f64(stack).unwrap(),
            )
        };
        for (sb, bb, stack) in [(0.0, 1.0, 10.0), (1.0, 1.0, 10.0), (0.5, 1.0, 0.9)] {
            assert_eq!(
                build_preflop_tree(&config(sb, bb, stack)),
                Err(TreeError::InvalidBlinds)
            );
        }
        for size in [1.0, 0.5, f64::NAN, f64::INFINITY] {
            assert_eq!(
                build_preflop_tree(&config(0.5, 1.0, 10.0).with_raise_sizes([3.0, size])),
                Err(TreeError::InvalidRaiseSize { index: 1 })
            );
        }
    }
}
