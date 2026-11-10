//! The training traversal.
//!
//! [`Plan`] is the public tree compiled once per game: every node knows its
//! board's caches (scores, rank order, buckets, suit groups) by index, so the
//! traversal does no hashing, and every subtree knows how much scratch space
//! and roughly how much work it needs.
//!
//! Small subtrees run sequentially inside one preallocated scratch buffer,
//! with no allocation and no Rayon task per node. Larger ones fan out over
//! Rayon exactly as before. Both paths perform the same floating-point
//! operations in the same order, so the split point (and the thread count)
//! never changes a result.

use super::reach_scratch::ReachScratch;
use super::{
    bucket_count_and_map, normalize_pair, swap_suit, NodeData, NodeEvaluation, Policy, SolverGame,
    SuitGroups, Trainer, TrainingPass,
};
use crate::card::{bit, mask_of, Card};
use crate::range::Combo;
use crate::solver::discount::IterationUpdate;
use crate::solver::terminal::BoardOrder;
use crate::solver::tree::{Node, Terminal};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

/// Subtrees whose estimated work (vector elements touched per traversal) is
/// below this run sequentially: a Rayon task costs more than they do.
const PARALLEL_WORK: u64 = 32_768;

/// "This member is the class representative: no relabeling."
const NO_PERMUTATION: u32 = u32::MAX;

/// Everything the traversal needs to know about one board.
pub(super) struct PlanBoard {
    pub(super) cards: Vec<Card>,
    pub(super) mask: u64,
    /// Empty per-player vectors before the river.
    pub(super) scores: Arc<[Vec<u32>; 2]>,
    pub(super) order: Option<Arc<BoardOrder>>,
    /// `None` when this street is not bucketed (every combo is its own bucket).
    pub(super) buckets: Option<Arc<[Vec<usize>; 2]>>,
    pub(super) num_buckets: [usize; 2],
    pub(super) groups: Option<Arc<SuitGroups>>,
    /// Per player and combo: neither hole card is on this board.
    pub(super) live: [Vec<bool>; 2],
}

pub(super) struct ChanceChild {
    pub(super) node: u32,
    /// Every card of the class with the permutation that relabels the
    /// representative's values for it ([`NO_PERMUTATION`] for the
    /// representative itself).
    pub(super) members: Vec<(Card, u32)>,
}

pub(super) enum PlanNode {
    Terminal {
        board: u32,
        terminal: Terminal,
    },
    Chance {
        divisor: f64,
        children: Vec<ChanceChild>,
    },
    Decision {
        id: usize,
        player: usize,
        board: u32,
        children: Vec<u32>,
    },
}

pub(super) struct Plan {
    pub(super) nodes: Vec<PlanNode>,
    pub(super) root: u32,
    pub(super) boards: Vec<PlanBoard>,
    /// Per suit transposition used by a chance node, per player: combo `i`'s
    /// value under the swapped board is the representative's value for combo
    /// `permutation[i]`.
    pub(super) permutations: Vec<[Vec<u32>; 2]>,
    /// Hole-card masks, per player and combo.
    pub(super) bits: [Vec<u64>; 2],
    /// Scratch `f64`s a sequential traversal of each subtree needs.
    pub(super) scratch: Vec<usize>,
    /// Estimated vector elements touched by one traversal of each subtree.
    pub(super) work: Vec<u64>,
    /// Whether any street is bucketed (selects the bucket-merging update).
    pub(super) bucketed: bool,
}

struct Builder<'a> {
    game: &'a SolverGame,
    plan: Plan,
    board_ids: HashMap<Vec<Card>, u32>,
    permutation_ids: HashMap<(u8, u8), u32>,
    empty_scores: Arc<[Vec<u32>; 2]>,
}

impl Plan {
    pub(super) fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            root: 0,
            boards: Vec::new(),
            permutations: Vec::new(),
            bits: [Vec::new(), Vec::new()],
            scratch: Vec::new(),
            work: Vec::new(),
            bucketed: false,
        }
    }

    /// Compiles `game`'s tree. Every cache of `game` must already be filled.
    pub(super) fn build(game: &SolverGame) -> Self {
        let mut builder = Builder {
            game,
            plan: Plan {
                bits: std::array::from_fn(|p| game.combos[p].iter().map(Combo::bits).collect()),
                bucketed: !game.bucket_maps.is_empty(),
                ..Plan::empty()
            },
            board_ids: HashMap::new(),
            permutation_ids: HashMap::new(),
            empty_scores: Arc::new([Vec::new(), Vec::new()]),
        };
        let board = game.initial_board.clone();
        builder.plan.root = builder.node(&game.tree.root, &board);
        builder.plan
    }
}

impl Builder<'_> {
    fn board(&mut self, board: &[Card]) -> u32 {
        if let Some(&id) = self.board_ids.get(board) {
            return id;
        }
        let game = self.game;
        let mask = mask_of(board);
        let buckets = game.bucket_maps.get(board).cloned();
        let id = self.plan.boards.len() as u32;
        self.plan.boards.push(PlanBoard {
            cards: board.to_vec(),
            mask,
            scores: game
                .board_eval_cache
                .get(board)
                .cloned()
                .unwrap_or_else(|| Arc::clone(&self.empty_scores)),
            order: game.board_order_cache.get(board).cloned(),
            num_buckets: std::array::from_fn(|p| {
                bucket_count_and_map(p, board, &game.bucket_maps, &game.identity_buckets).0
            }),
            buckets,
            groups: game.suit_groups.get(board).cloned(),
            live: std::array::from_fn(|p| {
                game.combos[p]
                    .iter()
                    .map(|c| c.bits() & mask == 0)
                    .collect()
            }),
        });
        self.board_ids.insert(board.to_vec(), id);
        id
    }

    fn permutation(&mut self, suit_a: u8, suit_b: u8) -> u32 {
        let key = (suit_a.min(suit_b), suit_a.max(suit_b));
        if let Some(&id) = self.permutation_ids.get(&key) {
            return id;
        }
        let game = self.game;
        let id = self.plan.permutations.len() as u32;
        self.plan.permutations.push(std::array::from_fn(|p| {
            game.combos[p]
                .iter()
                .map(|combo| {
                    let [h1, h2] = combo.cards;
                    let pair = normalize_pair(
                        swap_suit(h1, suit_a, suit_b),
                        swap_suit(h2, suit_a, suit_b),
                    );
                    game.combo_index[p][&pair] as u32
                })
                .collect()
        }));
        self.permutation_ids.insert(key, id);
        id
    }

    fn push(&mut self, node: PlanNode, scratch: usize, work: u64) -> u32 {
        let id = self.plan.nodes.len() as u32;
        self.plan.nodes.push(node);
        self.plan.scratch.push(scratch);
        self.plan.work.push(work);
        id
    }

    fn node(&mut self, node: &Node, board: &[Card]) -> u32 {
        let n = [self.game.combos[0].len(), self.game.combos[1].len()];
        let both = n[0] + n[1];
        match node {
            Node::Terminal(terminal) => {
                let board = self.board(board);
                self.push(
                    PlanNode::Terminal {
                        board,
                        terminal: *terminal,
                    },
                    0,
                    both as u64,
                )
            }
            Node::Chance { children } => {
                let cards: usize = children.iter().map(|(members, _)| members.len()).sum();
                let mut scratch = 0;
                let mut work = (cards * both) as u64;
                let children = children
                    .iter()
                    .map(|(members, child)| {
                        let rep = members[0];
                        let mut child_board = board.to_vec();
                        child_board.push(rep);
                        let node = self.node(child, &child_board);
                        scratch = scratch.max(self.plan.scratch[node as usize]);
                        work += self.plan.work[node as usize];
                        let members = members
                            .iter()
                            .map(|&card| {
                                let permutation = if card == rep {
                                    NO_PERMUTATION
                                } else {
                                    self.permutation(rep.suit(), card.suit())
                                };
                                (card, permutation)
                            })
                            .collect();
                        ChanceChild { node, members }
                    })
                    .collect();
                self.push(
                    PlanNode::Chance {
                        divisor: (cards as f64) - 4.0,
                        children,
                    },
                    both + scratch,
                    work,
                )
            }
            Node::Decision {
                id,
                player,
                children,
                ..
            } => {
                let p = player.index();
                let board_id = self.board(board);
                let num_buckets = self.plan.boards[board_id as usize].num_buckets[p];
                let mut scratch = 0;
                let mut work = (children.len() * both) as u64;
                let children: Vec<u32> = children
                    .iter()
                    .map(|child| {
                        let node = self.node(child, board);
                        scratch = scratch.max(self.plan.scratch[node as usize]);
                        work += self.plan.work[node as usize];
                        node
                    })
                    .collect();
                // Strategy rows, their totals, the child's own reach, every
                // child's own values, one child's opponent values, and the
                // bucket-merging update's temporaries.
                let frame = children.len() * num_buckets
                    + num_buckets
                    + n[p]
                    + children.len() * n[p]
                    + n[1 - p]
                    + update_scratch(self.plan.bucketed, n[p], num_buckets);
                self.push(
                    PlanNode::Decision {
                        id: *id,
                        player: p,
                        board: board_id,
                        children,
                    },
                    frame + scratch,
                    work,
                )
            }
        }
    }
}

thread_local! {
    // Taken out while in use, so a nested use (there is none today: the
    // sequential traversal never yields to Rayon) would simply allocate.
    static SCRATCH: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
    let mut buffer = SCRATCH
        .try_with(|scratch| scratch.take())
        .unwrap_or_default();
    if buffer.len() < len {
        buffer.resize(len, 0.0);
    }
    let result = f(&mut buffer[..len]);
    let _ = SCRATCH.try_with(|scratch| *scratch.borrow_mut() = buffer);
    result
}

/// Scratch `f64`s [`Trainer::update_node`] needs: none without abstraction,
/// else the effective reach and three per-bucket rows.
fn update_scratch(bucketed: bool, num_combos: usize, num_buckets: usize) -> usize {
    if bucketed {
        num_combos + 3 * num_buckets
    } else {
        0
    }
}

/// Adds one dealt card's values to `out`, skipping combos that hold the card.
///
/// `out` starts at `+0.0` and so never becomes `-0.0`; adding `+0.0` for a
/// blocked combo therefore leaves it bit-for-bit unchanged, which lets the
/// loop run without a branch.
fn accumulate_card(
    plan: &Plan,
    p: usize,
    card: Card,
    permutation: u32,
    values: &[f64],
    out: &mut [f64],
) {
    if out.is_empty() {
        return;
    }
    let card = bit(card);
    let bits = &plan.bits[p];
    if permutation == NO_PERMUTATION {
        for ((out, &value), &hand) in out.iter_mut().zip(values).zip(bits) {
            *out += if hand & card == 0 { value } else { 0.0 };
        }
    } else {
        let permutation = &plan.permutations[permutation as usize][p];
        for ((out, &source), &hand) in out.iter_mut().zip(permutation).zip(bits) {
            *out += if hand & card == 0 {
                values[source as usize]
            } else {
                0.0
            };
        }
    }
}

/// The acting player's reach after taking the action with strategy row `row`.
fn scale_reach(child: &mut [f64], reach: &[f64], row: &[f64], buckets: Option<&[usize]>) {
    match buckets {
        None => {
            for ((child, &reach), &probability) in child.iter_mut().zip(reach).zip(row) {
                *child = reach * probability;
            }
        }
        Some(buckets) => {
            for ((child, &reach), &bucket) in child.iter_mut().zip(reach).zip(buckets) {
                *child = reach * row[bucket];
            }
        }
    }
}

/// `own[i] += row[bucket(i)] * child[i]`.
fn add_weighted(own: &mut [f64], row: &[f64], child: &[f64], buckets: Option<&[usize]>) {
    match buckets {
        None => {
            for ((own, &child), &probability) in own.iter_mut().zip(child).zip(row) {
                *own += probability * child;
            }
        }
        Some(buckets) => {
            for ((own, &child), &bucket) in own.iter_mut().zip(child).zip(buckets) {
                *own += row[bucket] * child;
            }
        }
    }
}

/// One action's unbucketed regret and strategy-sum rows.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn update_rows(
    regret: &mut [f64],
    strategy_sum: &mut [f64],
    child: &[f64],
    node: &[f64],
    reach: &[f64],
    row: &[f64],
    live: &[bool],
    new_regret: impl Fn(f64, f64) -> f64,
    new_strategy_sum: impl Fn(f64) -> f64,
) {
    let len = child.len();
    let (regret, strategy_sum) = (&mut regret[..len], &mut strategy_sum[..len]);
    let (node, reach, row, live) = (&node[..len], &reach[..len], &row[..len], &live[..len]);
    for i in 0..len {
        // Keep the old zero-initialized delta accumulation, including its
        // signed-zero behavior.
        let delta = 0.0 + (child[i] - node[i]);
        regret[i] = new_regret(regret[i], delta);
        let mut sum = strategy_sum[i];
        // Board-dead combos add no average-strategy mass.
        if live[i] {
            sum += reach[i] * row[i];
        }
        strategy_sum[i] = new_strategy_sum(sum);
    }
}

impl Trainer {
    /// One traversal from the root with the initial ranges as reach.
    pub(super) fn run_pass(&self, pass: TrainingPass) -> [Vec<f64>; 2] {
        let game = &*self.game;
        let reach: [Vec<f64>; 2] =
            std::array::from_fn(|p| game.combos[p].iter().map(|c| c.weight).collect());
        let reach = [reach[0].as_slice(), reach[1].as_slice()];
        if rayon::current_num_threads() <= 1 {
            self.sequential_owned(game.plan.root, reach, pass)
        } else {
            self.parallel(game.plan.root, reach, pass)
        }
    }

    fn empty_values(&self, pass: TrainingPass) -> [Vec<f64>; 2] {
        std::array::from_fn(|p| {
            vec![
                0.0;
                if pass.computes(p) {
                    self.game.combos[p].len()
                } else {
                    0
                }
            ]
        })
    }

    fn sequential_owned(&self, node: u32, reach: [&[f64]; 2], pass: TrainingPass) -> [Vec<f64>; 2] {
        let mut out = self.empty_values(pass);
        let [out0, out1] = &mut out;
        with_scratch(self.game.plan.scratch[node as usize], |scratch| {
            self.sequential(node, reach, pass, [out0, out1], scratch)
        });
        out
    }

    /// Regret-matching strategy as `num_actions` rows of one probability per
    /// bucket, suit-symmetrized. `totals` is scratch of one row.
    pub(super) fn fill_strategy(
        &self,
        id: usize,
        board: &PlanBoard,
        player: usize,
        num_actions: usize,
        strategy: &mut [f64],
        totals: &mut [f64],
    ) {
        let num_buckets = board.num_buckets[player];
        totals.fill(0.0);
        {
            let data = self.node_data[id].lock().unwrap();
            for (a, regret_row) in data.regret_sum.iter().enumerate() {
                let row = &mut strategy[a * num_buckets..(a + 1) * num_buckets];
                for ((probability, &regret), total) in
                    row.iter_mut().zip(regret_row).zip(totals.iter_mut())
                {
                    let regret = regret.max(0.0);
                    *probability = regret;
                    *total += regret;
                }
            }
        }
        let uniform = 1.0 / num_actions as f64;
        for a in 0..num_actions {
            let row = &mut strategy[a * num_buckets..(a + 1) * num_buckets];
            for (probability, &total) in row.iter_mut().zip(totals.iter()) {
                *probability = if total > 0.0 {
                    *probability / total
                } else {
                    uniform
                };
            }
        }
        if let Some(groups) = &board.groups {
            for group in &groups[player] {
                for a in 0..num_actions {
                    let row = &mut strategy[a * num_buckets..(a + 1) * num_buckets];
                    let probability =
                        group.iter().map(|&i| row[i]).sum::<f64>() / group.len() as f64;
                    for &i in group {
                        row[i] = probability;
                    }
                }
            }
        }
    }

    /// Adds this iteration's regrets and average-strategy mass at one node.
    /// `child(a)` is action `a`'s values for the acting player and `node`
    /// the node's own. `scratch` holds at least [`update_scratch`] values.
    #[allow(clippy::too_many_arguments)]
    fn update_node<'a>(
        &self,
        id: usize,
        player: usize,
        board: &PlanBoard,
        pass: TrainingPass,
        reach: &[f64],
        strategy: &[f64],
        num_actions: usize,
        child: impl Fn(usize) -> &'a [f64],
        node: &[f64],
        scratch: &mut [f64],
    ) {
        let num_buckets = board.num_buckets[player];
        let live = &board.live[player];
        let mut data = self.node_data[id].lock().unwrap();
        let NodeData {
            regret_sum,
            strategy_sum,
        } = &mut *data;

        // With no abstraction maps every bucket is exactly combo i. Its merge
        // weight is one even for zero/board-dead own reach; terminal CFVs
        // already exclude impossible private deals. Do not infer this from
        // bucket counts: tied scores can leave holes in a map while still
        // sharing a bucket.
        if !self.game.plan.bucketed {
            for a in 0..num_actions {
                let row = &strategy[a * num_buckets..(a + 1) * num_buckets];
                let (regret, sum, child) = (&mut regret_sum[a], &mut strategy_sum[a], child(a));
                match pass.update {
                    IterationUpdate::Cfr => update_rows(
                        regret,
                        sum,
                        child,
                        node,
                        reach,
                        row,
                        live,
                        |r, d| r + d,
                        |s| s,
                    ),
                    IterationUpdate::CfrPlus => update_rows(
                        regret,
                        sum,
                        child,
                        node,
                        reach,
                        row,
                        live,
                        |r, d| (r + d).max(0.0),
                        |s| s,
                    ),
                    IterationUpdate::Dcfr { positive, strategy } => update_rows(
                        regret,
                        sum,
                        child,
                        node,
                        reach,
                        row,
                        live,
                        |r, d| {
                            let accumulated = r + d;
                            accumulated * if accumulated > 0.0 { positive } else { 0.5 }
                        },
                        |s| s * strategy,
                    ),
                }
            }
            return;
        }

        // `effective_reach` treats a combo whose hole card conflicts with the
        // board as reach 0 for the bucket-merge weighting and the strategy
        // sum, so a phantom combo sharing a bucket with a real one can never
        // skew the real combo's regret or average. When every combo is its
        // own bucket each bucket's total is either that combo's own reach or,
        // if masked or genuinely zero, falls through to the equal-weight
        // branch with a single member, which reduces exactly to the plain
        // per-combo regret update above.
        let buckets: &[usize] = board
            .buckets
            .as_ref()
            .map_or(&self.game.identity_buckets[player], |maps| &maps[player]);
        let num_own_combos = reach.len();
        let (effective_reach, rest) = scratch.split_at_mut(num_own_combos);
        let (bucket_reach_total, rest) = rest.split_at_mut(num_buckets);
        let (bucket_size, rest) = rest.split_at_mut(num_buckets);
        let regret_delta = &mut rest[..num_buckets];
        for i in 0..num_own_combos {
            effective_reach[i] = if live[i] { reach[i] } else { 0.0 };
        }
        bucket_reach_total.fill(0.0);
        // Member counts, kept as (exact) floats to share the scratch buffer.
        bucket_size.fill(0.0);
        for i in 0..num_own_combos {
            bucket_reach_total[buckets[i]] += effective_reach[i];
            bucket_size[buckets[i]] += 1.0;
        }
        for a in 0..num_actions {
            let row = &strategy[a * num_buckets..(a + 1) * num_buckets];
            let own = child(a);
            let (regret_sum, strategy_sum) = (&mut regret_sum[a], &mut strategy_sum[a]);
            regret_delta.fill(0.0);
            for i in 0..num_own_combos {
                let b = buckets[i];
                let delta = own[i] - node[i];
                let weight = if bucket_reach_total[b] > 0.0 {
                    effective_reach[i] / bucket_reach_total[b]
                } else {
                    1.0 / bucket_size[b]
                };
                regret_delta[b] += weight * delta;
                strategy_sum[b] += effective_reach[i] * row[b];
            }
            for (b, delta) in regret_delta.iter().enumerate() {
                regret_sum[b] = pass.update.regret(regret_sum[b], *delta);
                // Once per bucket, after every combo's contribution: never
                // apply DCFR discounting inside the combo loop.
                strategy_sum[b] = pass.update.strategy_sum(strategy_sum[b]);
            }
        }
    }

    /// Writes the requested players' values at `node` over `out` (an empty
    /// slice for a player the pass does not compute) and updates every
    /// decision node below. `scratch` holds at least `plan.scratch[node]`.
    fn sequential(
        &self,
        node: u32,
        reach: [&[f64]; 2],
        pass: TrainingPass,
        out: [&mut [f64]; 2],
        scratch: &mut [f64],
    ) {
        let game = &*self.game;
        let plan = &game.plan;
        match &plan.nodes[node as usize] {
            PlanNode::Terminal { board, terminal } => {
                let board = &plan.boards[*board as usize];
                for (p, out) in out.into_iter().enumerate() {
                    if pass.computes(p) {
                        game.terminal_into(
                            terminal,
                            p,
                            reach[1 - p],
                            board.mask,
                            &board.scores,
                            board.order.as_deref(),
                            out,
                        );
                    }
                }
            }
            PlanNode::Chance { divisor, children } => {
                let [out0, out1] = out;
                let (child0, rest) = scratch.split_at_mut(out0.len());
                let (child1, rest) = rest.split_at_mut(out1.len());
                out0.fill(0.0);
                out1.fill(0.0);
                for child in children {
                    self.sequential(child.node, reach, pass, [&mut *child0, &mut *child1], rest);
                    for &(card, permutation) in &child.members {
                        accumulate_card(plan, 0, card, permutation, child0, out0);
                        accumulate_card(plan, 1, card, permutation, child1, out1);
                    }
                }
                for value in out0.iter_mut().chain(out1.iter_mut()) {
                    *value /= divisor;
                }
            }
            PlanNode::Decision {
                id,
                player,
                board,
                children,
            } => {
                let p = *player;
                let board = &plan.boards[*board as usize];
                let num_actions = children.len();
                let num_buckets = board.num_buckets[p];
                let buckets = board.buckets.as_ref().map(|maps| maps[p].as_slice());
                let [out0, out1] = out;
                let (own_out, other_out) = if p == 0 { (out0, out1) } else { (out1, out0) };
                let own_len = own_out.len();

                let (strategy, rest) = scratch.split_at_mut(num_actions * num_buckets);
                let (totals, rest) = rest.split_at_mut(num_buckets);
                let (child_reach, rest) = rest.split_at_mut(reach[p].len());
                let (child_own, rest) = rest.split_at_mut(num_actions * own_len);
                let (child_other, rest) = rest.split_at_mut(other_out.len());
                let (update, rest) =
                    rest.split_at_mut(update_scratch(plan.bucketed, reach[p].len(), num_buckets));
                self.fill_strategy(*id, board, p, num_actions, strategy, totals);

                other_out.fill(0.0);
                for (a, &child) in children.iter().enumerate() {
                    // This action changes only the acting player's reach.
                    let row = &strategy[a * num_buckets..(a + 1) * num_buckets];
                    scale_reach(child_reach, reach[p], row, buckets);
                    let mut reach = reach;
                    reach[p] = &*child_reach;
                    let own = &mut child_own[a * own_len..(a + 1) * own_len];
                    let out = if p == 0 {
                        [own, &mut *child_other]
                    } else {
                        [&mut *child_other, own]
                    };
                    self.sequential(child, reach, pass, out, rest);
                    for (value, &child) in other_out.iter_mut().zip(child_other.iter()) {
                        *value += child;
                    }
                }
                own_out.fill(0.0);
                for a in 0..num_actions {
                    add_weighted(
                        own_out,
                        &strategy[a * num_buckets..(a + 1) * num_buckets],
                        &child_own[a * own_len..(a + 1) * own_len],
                        buckets,
                    );
                }

                // A non-updating player's strategy affects the requested CFVs,
                // but its regrets and strategy mass must remain untouched.
                if pass.updating.is_some_and(|target| target != p) {
                    return;
                }
                let child_own = &*child_own;
                self.update_node(
                    *id,
                    p,
                    board,
                    pass,
                    reach[p],
                    strategy,
                    num_actions,
                    |a| &child_own[a * own_len..(a + 1) * own_len],
                    own_out,
                    update,
                );
            }
        }
    }

    /// [`sequential`](Self::sequential) with sibling subtrees on Rayon.
    fn parallel(&self, node: u32, reach: [&[f64]; 2], pass: TrainingPass) -> [Vec<f64>; 2] {
        let game = &*self.game;
        let plan = &game.plan;
        if plan.work[node as usize] < PARALLEL_WORK {
            return self.sequential_owned(node, reach, pass);
        }
        match &plan.nodes[node as usize] {
            PlanNode::Terminal { .. } => self.sequential_owned(node, reach, pass),
            PlanNode::Chance { divisor, children } => {
                let child_values: Vec<[Vec<f64>; 2]> = children
                    .par_iter()
                    .map(|child| self.parallel(child.node, reach, pass))
                    .collect();
                let mut out = self.empty_values(pass);
                for (child, values) in children.iter().zip(&child_values) {
                    for &(card, permutation) in &child.members {
                        for p in 0..2 {
                            accumulate_card(plan, p, card, permutation, &values[p], &mut out[p]);
                        }
                    }
                }
                for values in &mut out {
                    for value in values {
                        *value /= divisor;
                    }
                }
                out
            }
            PlanNode::Decision {
                id,
                player,
                board,
                children,
            } => {
                let p = *player;
                let board = &plan.boards[*board as usize];
                let num_actions = children.len();
                let num_buckets = board.num_buckets[p];
                let buckets = board.buckets.as_ref().map(|maps| maps[p].as_slice());
                let mut strategy = vec![0.0; num_actions * num_buckets];
                let mut totals = vec![0.0; num_buckets];
                self.fill_strategy(*id, board, p, num_actions, &mut strategy, &mut totals);
                drop(totals);
                let strategy = &strategy[..];

                let child_values: Vec<[Vec<f64>; 2]> = children
                    .par_iter()
                    .enumerate()
                    .map(|(a, &child)| {
                        // Each sibling owns its reach vector; the unchanged
                        // opponent slice is shared read-only.
                        let mut own_reach = ReachScratch::copy_from(reach[p]);
                        let row = &strategy[a * num_buckets..(a + 1) * num_buckets];
                        scale_reach(&mut own_reach, reach[p], row, buckets);
                        let mut reach = reach;
                        reach[p] = &own_reach;
                        self.parallel(child, reach, pass)
                    })
                    .collect();

                let mut out = self.empty_values(pass);
                {
                    let [out0, out1] = &mut out;
                    let (own_out, other_out) = if p == 0 { (out0, out1) } else { (out1, out0) };
                    for (a, values) in child_values.iter().enumerate() {
                        add_weighted(
                            own_out,
                            &strategy[a * num_buckets..(a + 1) * num_buckets],
                            &values[p],
                            buckets,
                        );
                        for (value, &child) in other_out.iter_mut().zip(&values[1 - p]) {
                            *value += child;
                        }
                    }
                }
                if pass.updating.is_some_and(|target| target != p) {
                    return out;
                }
                self.update_node(
                    *id,
                    p,
                    board,
                    pass,
                    reach[p],
                    strategy,
                    num_actions,
                    |a| child_values[a][p].as_slice(),
                    &out[p],
                    &mut vec![0.0; update_scratch(plan.bucketed, reach[p].len(), num_buckets)],
                );
                out
            }
        }
    }
}

/// Reads of a fixed [`Policy`]: best response and per-node evaluation. They
/// use the same plan, scratch discipline, and operation order as training.
impl SolverGame {
    fn policy_row(policy: &Policy, id: usize, action: usize) -> &[f64] {
        &policy.rows[id][action]
    }

    /// `x`'s best-response values against `policy` at the plan's root, given
    /// the opponent's initial `reach_opp`.
    pub(super) fn best_response(&self, policy: &Policy, x: usize, reach_opp: &[f64]) -> Vec<f64> {
        let root = self.plan.root;
        if rayon::current_num_threads() <= 1 {
            self.best_response_owned(policy, root, x, reach_opp)
        } else {
            self.best_response_parallel(policy, root, x, reach_opp)
        }
    }

    fn best_response_owned(
        &self,
        policy: &Policy,
        node: u32,
        x: usize,
        reach_opp: &[f64],
    ) -> Vec<f64> {
        let mut out = vec![0.0; self.combos[x].len()];
        with_scratch(self.plan.scratch[node as usize], |scratch| {
            self.best_response_sequential(policy, node, x, reach_opp, &mut out, scratch)
        });
        out
    }

    fn best_response_sequential(
        &self,
        policy: &Policy,
        node: u32,
        x: usize,
        reach_opp: &[f64],
        out: &mut [f64],
        scratch: &mut [f64],
    ) {
        let plan = &self.plan;
        match &plan.nodes[node as usize] {
            PlanNode::Terminal { board, terminal } => {
                let board = &plan.boards[*board as usize];
                self.terminal_into(
                    terminal,
                    x,
                    reach_opp,
                    board.mask,
                    &board.scores,
                    board.order.as_deref(),
                    out,
                );
            }
            PlanNode::Chance { divisor, children } => {
                let (child_values, rest) = scratch.split_at_mut(out.len());
                out.fill(0.0);
                for child in children {
                    self.best_response_sequential(
                        policy,
                        child.node,
                        x,
                        reach_opp,
                        child_values,
                        rest,
                    );
                    for &(card, permutation) in &child.members {
                        accumulate_card(plan, x, card, permutation, child_values, out);
                    }
                }
                for value in out.iter_mut() {
                    *value /= divisor;
                }
            }
            PlanNode::Decision {
                id,
                player,
                board,
                children,
            } => {
                let (child_values, rest) = scratch.split_at_mut(out.len());
                if *player == x {
                    out.fill(f64::NEG_INFINITY);
                    for &child in children {
                        self.best_response_sequential(
                            policy,
                            child,
                            x,
                            reach_opp,
                            child_values,
                            rest,
                        );
                        for (best, &value) in out.iter_mut().zip(child_values.iter()) {
                            *best = best.max(value);
                        }
                    }
                } else {
                    let board = &plan.boards[*board as usize];
                    let buckets = board.buckets.as_ref().map(|maps| maps[*player].as_slice());
                    let (child_reach, rest) = rest.split_at_mut(reach_opp.len());
                    out.fill(0.0);
                    for (a, &child) in children.iter().enumerate() {
                        scale_reach(
                            child_reach,
                            reach_opp,
                            Self::policy_row(policy, *id, a),
                            buckets,
                        );
                        self.best_response_sequential(
                            policy,
                            child,
                            x,
                            child_reach,
                            child_values,
                            rest,
                        );
                        for (sum, &value) in out.iter_mut().zip(child_values.iter()) {
                            *sum += value;
                        }
                    }
                }
            }
        }
    }

    fn best_response_parallel(
        &self,
        policy: &Policy,
        node: u32,
        x: usize,
        reach_opp: &[f64],
    ) -> Vec<f64> {
        let plan = &self.plan;
        if plan.work[node as usize] < PARALLEL_WORK {
            return self.best_response_owned(policy, node, x, reach_opp);
        }
        match &plan.nodes[node as usize] {
            PlanNode::Terminal { .. } => self.best_response_owned(policy, node, x, reach_opp),
            PlanNode::Chance { divisor, children } => {
                let child_values: Vec<Vec<f64>> = children
                    .par_iter()
                    .map(|child| self.best_response_parallel(policy, child.node, x, reach_opp))
                    .collect();
                let mut out = vec![0.0; self.combos[x].len()];
                for (child, values) in children.iter().zip(&child_values) {
                    for &(card, permutation) in &child.members {
                        accumulate_card(plan, x, card, permutation, values, &mut out);
                    }
                }
                for value in &mut out {
                    *value /= divisor;
                }
                out
            }
            PlanNode::Decision {
                id,
                player,
                board,
                children,
            } => {
                if *player == x {
                    let child_values: Vec<Vec<f64>> = children
                        .par_iter()
                        .map(|&child| self.best_response_parallel(policy, child, x, reach_opp))
                        .collect();
                    let mut out = vec![f64::NEG_INFINITY; self.combos[x].len()];
                    for values in &child_values {
                        for (best, &value) in out.iter_mut().zip(values) {
                            *best = best.max(value);
                        }
                    }
                    out
                } else {
                    let board = &plan.boards[*board as usize];
                    let buckets = board.buckets.as_ref().map(|maps| maps[*player].as_slice());
                    let child_values: Vec<Vec<f64>> = children
                        .par_iter()
                        .enumerate()
                        .map(|(a, &child)| {
                            let mut child_reach = ReachScratch::copy_from(reach_opp);
                            scale_reach(
                                &mut child_reach,
                                reach_opp,
                                Self::policy_row(policy, *id, a),
                                buckets,
                            );
                            self.best_response_parallel(policy, child, x, &child_reach)
                        })
                        .collect();
                    let mut out = vec![0.0; self.combos[x].len()];
                    for values in &child_values {
                        for (sum, &value) in out.iter_mut().zip(values) {
                            *sum += value;
                        }
                    }
                    out
                }
            }
        }
    }

    /// Fraction of the pot `player`'s combos win at a river showdown against
    /// the opponent's positive reach.
    fn equity(&self, player: usize, reach_opp: &[f64], board: &PlanBoard) -> Vec<f64> {
        let positive_reach: Vec<f64> = reach_opp
            .iter()
            .map(|&r| if r <= 0.0 { 0.0 } else { r })
            .collect();
        let mass =
            self.terminal_index
                .compatible_mass(&self.combos, player, &positive_reach, board.mask);
        let net = self.terminal_index.showdown(
            &self.combos,
            player,
            &positive_reach,
            &board.scores,
            board
                .order
                .as_deref()
                .expect("every river board has a rank order"),
            board.mask,
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

    /// Both players' raw values under `policy` at the plan's root; fills one
    /// [`NodeEvaluation`] per decision node.
    pub(super) fn evaluate_plan(
        &self,
        policy: &Policy,
        reach: [&[f64]; 2],
        nodes: &[OnceLock<NodeEvaluation>],
    ) -> [Vec<f64>; 2] {
        let root = self.plan.root;
        if rayon::current_num_threads() <= 1 {
            self.evaluate_owned(policy, root, reach, nodes)
        } else {
            self.evaluate_parallel(policy, root, reach, nodes)
        }
    }

    fn evaluate_owned(
        &self,
        policy: &Policy,
        node: u32,
        reach: [&[f64]; 2],
        nodes: &[OnceLock<NodeEvaluation>],
    ) -> [Vec<f64>; 2] {
        let mut out = [
            vec![0.0; self.combos[0].len()],
            vec![0.0; self.combos[1].len()],
        ];
        let [out0, out1] = &mut out;
        with_scratch(self.plan.scratch[node as usize], |scratch| {
            self.evaluate_sequential(policy, node, reach, [out0, out1], scratch, nodes)
        });
        out
    }

    /// Records the display values of decision node `id`, where `player` acts.
    ///
    /// `own` is still a raw counterfactual value: a sum weighted by the
    /// opponent's unnormalized reach. Dividing by the opponent's compatible
    /// reach recovers the per-combo expected value. Only the stored value is
    /// normalized; the value returned to the parent stays raw.
    fn record_evaluation(
        &self,
        id: usize,
        player: usize,
        board: &PlanBoard,
        reach_opp: &[f64],
        own: &[f64],
        nodes: &[OnceLock<NodeEvaluation>],
    ) {
        let equity = (board.cards.len() == 5).then(|| self.equity(player, reach_opp, board));
        let mut ev =
            self.terminal_index
                .compatible_mass(&self.combos, player, reach_opp, board.mask);
        for (ev, &own) in ev.iter_mut().zip(own) {
            let denom = *ev;
            *ev = if denom > 0.0 { own / denom } else { 0.0 };
        }
        let fresh = nodes[id].set(NodeEvaluation { ev, equity }).is_ok();
        debug_assert!(fresh, "every decision node is evaluated once");
    }

    fn evaluate_sequential(
        &self,
        policy: &Policy,
        node: u32,
        reach: [&[f64]; 2],
        out: [&mut [f64]; 2],
        scratch: &mut [f64],
        nodes: &[OnceLock<NodeEvaluation>],
    ) {
        let plan = &self.plan;
        match &plan.nodes[node as usize] {
            PlanNode::Terminal { board, terminal } => {
                let board = &plan.boards[*board as usize];
                for (p, out) in out.into_iter().enumerate() {
                    self.terminal_into(
                        terminal,
                        p,
                        reach[1 - p],
                        board.mask,
                        &board.scores,
                        board.order.as_deref(),
                        out,
                    );
                }
            }
            PlanNode::Chance { divisor, children } => {
                let [out0, out1] = out;
                let (child0, rest) = scratch.split_at_mut(out0.len());
                let (child1, rest) = rest.split_at_mut(out1.len());
                out0.fill(0.0);
                out1.fill(0.0);
                for child in children {
                    self.evaluate_sequential(
                        policy,
                        child.node,
                        reach,
                        [&mut *child0, &mut *child1],
                        rest,
                        nodes,
                    );
                    for &(card, permutation) in &child.members {
                        accumulate_card(plan, 0, card, permutation, child0, out0);
                        accumulate_card(plan, 1, card, permutation, child1, out1);
                    }
                }
                for value in out0.iter_mut().chain(out1.iter_mut()) {
                    *value /= divisor;
                }
            }
            PlanNode::Decision {
                id,
                player,
                board,
                children,
            } => {
                let p = *player;
                let board = &plan.boards[*board as usize];
                let buckets = board.buckets.as_ref().map(|maps| maps[p].as_slice());
                let [out0, out1] = out;
                let (own_out, other_out) = if p == 0 { (out0, out1) } else { (out1, out0) };
                let (child_reach, rest) = scratch.split_at_mut(reach[p].len());
                let (child_own, rest) = rest.split_at_mut(own_out.len());
                let (child_other, rest) = rest.split_at_mut(other_out.len());
                own_out.fill(0.0);
                other_out.fill(0.0);
                for (a, &child) in children.iter().enumerate() {
                    let row = Self::policy_row(policy, *id, a);
                    scale_reach(child_reach, reach[p], row, buckets);
                    let mut reach = reach;
                    reach[p] = &*child_reach;
                    let out = if p == 0 {
                        [&mut *child_own, &mut *child_other]
                    } else {
                        [&mut *child_other, &mut *child_own]
                    };
                    self.evaluate_sequential(policy, child, reach, out, rest, nodes);
                    add_weighted(own_out, row, child_own, buckets);
                    for (value, &child) in other_out.iter_mut().zip(child_other.iter()) {
                        *value += child;
                    }
                }
                self.record_evaluation(*id, p, board, reach[1 - p], own_out, nodes);
            }
        }
    }

    fn evaluate_parallel(
        &self,
        policy: &Policy,
        node: u32,
        reach: [&[f64]; 2],
        nodes: &[OnceLock<NodeEvaluation>],
    ) -> [Vec<f64>; 2] {
        let plan = &self.plan;
        if plan.work[node as usize] < PARALLEL_WORK {
            return self.evaluate_owned(policy, node, reach, nodes);
        }
        match &plan.nodes[node as usize] {
            PlanNode::Terminal { .. } => self.evaluate_owned(policy, node, reach, nodes),
            PlanNode::Chance { divisor, children } => {
                let child_values: Vec<[Vec<f64>; 2]> = children
                    .par_iter()
                    .map(|child| self.evaluate_parallel(policy, child.node, reach, nodes))
                    .collect();
                let mut out = [
                    vec![0.0; self.combos[0].len()],
                    vec![0.0; self.combos[1].len()],
                ];
                for (child, values) in children.iter().zip(&child_values) {
                    for &(card, permutation) in &child.members {
                        for p in 0..2 {
                            accumulate_card(plan, p, card, permutation, &values[p], &mut out[p]);
                        }
                    }
                }
                for values in &mut out {
                    for value in values {
                        *value /= divisor;
                    }
                }
                out
            }
            PlanNode::Decision {
                id,
                player,
                board,
                children,
            } => {
                let p = *player;
                let board = &plan.boards[*board as usize];
                let buckets = board.buckets.as_ref().map(|maps| maps[p].as_slice());
                let child_values: Vec<[Vec<f64>; 2]> = children
                    .par_iter()
                    .enumerate()
                    .map(|(a, &child)| {
                        let mut own_reach = ReachScratch::copy_from(reach[p]);
                        scale_reach(
                            &mut own_reach,
                            reach[p],
                            Self::policy_row(policy, *id, a),
                            buckets,
                        );
                        let mut reach = reach;
                        reach[p] = &own_reach;
                        self.evaluate_parallel(policy, child, reach, nodes)
                    })
                    .collect();
                let mut out = [
                    vec![0.0; self.combos[0].len()],
                    vec![0.0; self.combos[1].len()],
                ];
                {
                    let [out0, out1] = &mut out;
                    let (own_out, other_out) = if p == 0 { (out0, out1) } else { (out1, out0) };
                    for (a, values) in child_values.iter().enumerate() {
                        add_weighted(
                            own_out,
                            Self::policy_row(policy, *id, a),
                            &values[p],
                            buckets,
                        );
                        for (value, &child) in other_out.iter_mut().zip(&values[1 - p]) {
                            *value += child;
                        }
                    }
                }
                self.record_evaluation(*id, p, board, reach[1 - p], &out[p], nodes);
                out
            }
        }
    }
}
