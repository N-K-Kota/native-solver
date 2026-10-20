//! Exact range terminal values using rank sweeps and card-removal sums.
//!
//! A compatible mass is `total - card[h1] - card[h2] + pair[h1,h2]`.
//! Strictly weaker/stronger rank prefixes cannot contain the hero's exact
//! pair, so no add-back is needed during the two showdown sweeps. Sorting
//! is done once per board; each subsequent evaluation is linear in the
//! two range lengths, rather than their Cartesian product. A rare direct
//! sum fallback preserves relative precision for nearly all-blocked reach;
//! adversarial reaches can therefore still take quadratic time.

use crate::card::{mask_has, Card};
use std::collections::HashMap;

use crate::range::Combo;

pub(super) struct TerminalIndex {
    /// All matching opponent entries, including reversed or repeated pairs.
    same_pair: [Vec<Vec<usize>>; 2],
}

pub(super) struct BoardOrder {
    order: [Vec<usize>; 2],
    // Indexed in the hero's sorted order. Each pair is the opponent prefix
    // end for strictly weaker hands and suffix start for strictly stronger.
    // Board-dead combos never enter either order; ties lie between bounds.
    rank_bounds: [Vec<(usize, usize)>; 2],
}

impl BoardOrder {
    pub(super) fn new(combos: &[Vec<Combo>; 2], scores: &[Vec<u32>; 2], board: u64) -> Self {
        let order: [Vec<usize>; 2] = std::array::from_fn(|p| {
            let mut order: Vec<_> = (0..combos[p].len())
                .filter(|&i| live(combos[p][i], board))
                .collect();
            order.sort_unstable_by_key(|&i| scores[p][i]);
            order
        });
        let rank_bounds = std::array::from_fn(|p| {
            let opponent = &order[1 - p];
            let (mut lower, mut upper) = (0, 0);
            order[p]
                .iter()
                .map(|&i| {
                    let score = scores[p][i];
                    while lower < opponent.len() && scores[1 - p][opponent[lower]] < score {
                        lower += 1;
                    }
                    upper = upper.max(lower);
                    while upper < opponent.len() && scores[1 - p][opponent[upper]] <= score {
                        upper += 1;
                    }
                    (lower, upper)
                })
                .collect()
        });
        Self { order, rank_bounds }
    }
}

#[derive(Clone)]
struct ReachMass {
    total: f64,
    card: [f64; 52],
    support: i32,
    card_support: [i32; 52],
}

impl ReachMass {
    fn new() -> Self {
        Self {
            total: 0.0,
            card: [0.0; 52],
            support: 0,
            card_support: [0; 52],
        }
    }

    fn add(&mut self, combo: Combo, weight: f64) {
        // Strategy zeros contribute neither mass nor support. Avoid touching
        // all three floating-point sums for these entries in each rank sweep.
        if weight == 0.0 {
            return;
        }
        self.total += weight;
        self.card[combo.cards[0].0 as usize] += weight;
        self.card[combo.cards[1].0 as usize] += weight;
        self.support += 1;
        self.card_support[combo.cards[0].0 as usize] += 1;
        self.card_support[combo.cards[1].0 as usize] += 1;
    }

    fn support_excluding(&self, combo: Combo) -> i32 {
        self.support
            - self.card_support[combo.cards[0].0 as usize]
            - self.card_support[combo.cards[1].0 as usize]
    }

    fn cancellation_prone(&self, value: f64) -> bool {
        // A subtraction can lose a tiny genuine compatible reach next to
        // larger blocked weights. Budget roundoff relative to the result,
        // not only the total: exact-zero support is handled separately.
        let error_bound = 8.0 * f64::EPSILON * self.support as f64 * self.total.abs();
        value <= error_bound / 1e-10
    }

    fn excluding(&self, combo: Combo) -> f64 {
        if self.support_excluding(combo) == 0 {
            // Preserve exact zero when every nonzero entry is blocked;
            // rounding residue must not invent reachable private deals.
            0.0
        } else {
            self.total - self.card[combo.cards[0].0 as usize] - self.card[combo.cards[1].0 as usize]
        }
    }
}

fn disjoint(a: Combo, b: Combo) -> bool {
    let ([a0, a1], [b0, b1]) = (a.cards, b.cards);
    a0 != b0 && a0 != b1 && a1 != b0 && a1 != b1
}

fn live(combo: Combo, board: u64) -> bool {
    !mask_has(board, combo.cards[0]) && !mask_has(board, combo.cards[1])
}

impl TerminalIndex {
    pub(super) fn new(combos: &[Vec<Combo>; 2]) -> Self {
        let lookup: [HashMap<(Card, Card), Vec<usize>>; 2] = std::array::from_fn(|p| {
            let mut lookup: HashMap<_, Vec<_>> = HashMap::new();
            for (i, combo) in combos[p].iter().enumerate() {
                lookup.entry(combo.sorted_cards()).or_default().push(i);
            }
            lookup
        });
        Self {
            same_pair: std::array::from_fn(|p| {
                combos[p]
                    .iter()
                    .map(|combo| {
                        lookup[1 - p]
                            .get(&combo.sorted_cards())
                            .cloned()
                            .unwrap_or_default()
                    })
                    .collect()
            }),
        }
    }

    pub(super) fn compatible_mass(
        &self,
        combos: &[Vec<Combo>; 2],
        player: usize,
        reach: &[f64],
        board: u64,
    ) -> Vec<f64> {
        let mut mass = ReachMass::new();
        for (&combo, &weight) in combos[1 - player].iter().zip(reach) {
            if live(combo, board) {
                mass.add(combo, weight);
            }
        }
        combos[player]
            .iter()
            .enumerate()
            .map(|(i, &combo)| {
                if live(combo, board) {
                    // If hero is live, every exact matching opponent pair is too.
                    let same = &self.same_pair[player][i];
                    let support = mass.support_excluding(combo)
                        + same.iter().filter(|&&j| reach[j] != 0.0).count() as i32;
                    if support == 0 {
                        0.0
                    } else {
                        let value = mass.total
                            - mass.card[combo.cards[0].0 as usize]
                            - mass.card[combo.cards[1].0 as usize]
                            + same.iter().map(|&j| reach[j]).sum::<f64>();
                        if mass.cancellation_prone(value) {
                            combos[1 - player]
                                .iter()
                                .zip(reach)
                                .filter(|(other, _)| {
                                    live(**other, board) && disjoint(combo, **other)
                                })
                                .map(|(_, &weight)| weight)
                                .sum()
                        } else {
                            value
                        }
                    }
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Returns win-minus-loss reach weighted by `half_pot`. With unequal
    /// investments, every compatible deal also contributes `tie_value`.
    /// Invalid hero combos stay zero and never enter a rank sweep.
    #[allow(clippy::too_many_arguments)] // Explicit immutable inputs keep the hot kernel allocation-free.
    pub(super) fn showdown(
        &self,
        combos: &[Vec<Combo>; 2],
        player: usize,
        reach: &[f64],
        scores: &[Vec<u32>; 2],
        order: &BoardOrder,
        board: u64,
        invested: &[f64; 2],
    ) -> Vec<f64> {
        let opp = 1 - player;
        let small = invested[0].abs().min(invested[1].abs());
        let large = invested[0].abs().max(invested[1].abs());
        if invested[0] != invested[1] && small < large * 1e-8 {
            // Constructed poker showdowns always have matched investments.
            // Public custom trees may not: avoid cancellation between the
            // affine tie offset and half-pot when their scales differ greatly.
            return combos[player]
                .iter()
                .enumerate()
                .map(|(i, &hero)| {
                    if !live(hero, board) {
                        return 0.0;
                    }
                    order.order[opp]
                        .iter()
                        .copied()
                        .filter(|&j| disjoint(hero, combos[opp][j]))
                        .map(|j| {
                            let net = match scores[player][i].cmp(&scores[opp][j]) {
                                std::cmp::Ordering::Greater => invested[opp],
                                std::cmp::Ordering::Less => -invested[player],
                                std::cmp::Ordering::Equal => {
                                    (invested[opp] - invested[player]) / 2.0
                                }
                            };
                            reach[j] * net
                        })
                        .sum()
                })
                .collect();
        }
        let half_pot = (invested[0] + invested[1]) / 2.0;
        let tie_value = (invested[opp] - invested[player]) / 2.0;
        let mut values = if tie_value == 0.0 {
            vec![0.0; combos[player].len()]
        } else {
            self.compatible_mass(combos, player, reach, board)
                .into_iter()
                .map(|mass| mass * tie_value)
                .collect()
        };
        let mut weaker = ReachMass::new();
        let mut j = 0;
        for (&i, &(end, _)) in order.order[player].iter().zip(&order.rank_bounds[player]) {
            while j < end {
                let k = order.order[opp][j];
                weaker.add(combos[opp][k], reach[k]);
                j += 1;
            }
            let mut mass = weaker.excluding(combos[player][i]);
            if weaker.support_excluding(combos[player][i]) > 0 && weaker.cancellation_prone(mass) {
                mass = order.order[opp][..j]
                    .iter()
                    .copied()
                    .filter(|&k| disjoint(combos[player][i], combos[opp][k]))
                    .map(|k| reach[k])
                    .sum();
            }
            values[i] += half_pot * mass;
        }
        let mut stronger = ReachMass::new();
        let mut j = order.order[opp].len();
        for (&i, &(_, start)) in order.order[player]
            .iter()
            .zip(&order.rank_bounds[player])
            .rev()
        {
            while j > start {
                j -= 1;
                let k = order.order[opp][j];
                stronger.add(combos[opp][k], reach[k]);
            }
            let mut mass = stronger.excluding(combos[player][i]);
            if stronger.support_excluding(combos[player][i]) > 0
                && stronger.cancellation_prone(mass)
            {
                mass = order.order[opp][j..]
                    .iter()
                    .copied()
                    .filter(|&k| disjoint(combos[player][i], combos[opp][k]))
                    .map(|k| reach[k])
                    .sum();
            }
            values[i] -= half_pot * mass;
        }
        values
    }
}

#[cfg(test)]
#[path = "terminal_rank_reference.rs"]
mod rank_reference;
