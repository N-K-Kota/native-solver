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
    /// Whether any of a player's combos has a matching opponent entry.
    any_same_pair: [bool; 2],
    /// Each combo's two card indices, per player.
    cards: [Vec<[u8; 2]>; 2],
}

pub(super) struct BoardOrder {
    order: [Vec<usize>; 2],
    // Indexed in the hero's sorted order. Each pair is the opponent prefix
    // end for strictly weaker hands and suffix start for strictly stronger.
    // Board-dead combos never enter either order; ties lie between bounds.
    rank_bounds: [Vec<(usize, usize)>; 2],
    /// The card indices of each `order` entry, so a sweep reads them in
    /// sequence.
    cards: [Vec<[u8; 2]>; 2],
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
        let cards = std::array::from_fn(|p| {
            order[p]
                .iter()
                .map(|&i| combos[p][i].cards.map(|card| card.0))
                .collect()
        });
        Self {
            order,
            rank_bounds,
            cards,
        }
    }
}

/// Card indices are below 52; masking with 63 lets the compiler drop the
/// bounds check on the 64-entry tables.
const CARD_SLOTS: usize = 64;

fn slot(card: u8) -> usize {
    card as usize & (CARD_SLOTS - 1)
}

#[derive(Clone)]
struct ReachMass {
    total: f64,
    card: [f64; CARD_SLOTS],
    support: i32,
    card_support: [i32; CARD_SLOTS],
}

impl ReachMass {
    fn new() -> Self {
        Self {
            total: 0.0,
            card: [0.0; CARD_SLOTS],
            support: 0,
            card_support: [0; CARD_SLOTS],
        }
    }

    #[cfg(test)]
    fn add(&mut self, combo: Combo, weight: f64) {
        self.add_cards(combo.cards.map(|card| card.0), weight);
    }

    #[inline(always)]
    fn add_cards(&mut self, cards: [u8; 2], weight: f64) {
        // Strategy zeros contribute neither mass nor support. Avoid touching
        // all three floating-point sums for these entries in each rank sweep.
        if weight == 0.0 {
            return;
        }
        self.total += weight;
        self.card[slot(cards[0])] += weight;
        self.card[slot(cards[1])] += weight;
        self.support += 1;
        self.card_support[slot(cards[0])] += 1;
        self.card_support[slot(cards[1])] += 1;
    }

    #[cfg(test)]
    fn support_excluding(&self, combo: Combo) -> i32 {
        self.support_excluding_cards(combo.cards.map(|card| card.0))
    }

    #[inline(always)]
    fn support_excluding_cards(&self, cards: [u8; 2]) -> i32 {
        self.support - self.card_support[slot(cards[0])] - self.card_support[slot(cards[1])]
    }

    /// The largest value [`cancellation_prone`](Self::cancellation_prone)
    /// accepts.
    #[inline(always)]
    fn cancellation_limit(&self) -> f64 {
        // A subtraction can lose a tiny genuine compatible reach next to
        // larger blocked weights. Budget roundoff relative to the result,
        // not only the total: exact-zero support is handled separately.
        let error_bound = 8.0 * f64::EPSILON * self.support as f64 * self.total.abs();
        error_bound / 1e-10
    }

    /// Whether `value`, a compatible mass of this accumulator, may have lost
    /// its significant bits to cancellation.
    #[inline(always)]
    fn cancellation_prone(&self, value: f64) -> bool {
        // With at most 1024 entries the limit is below `|total| / 32`
        // (8 * EPSILON * 1024 / 1e-10 < 0.02), so a value well above that is
        // never prone and the division can be skipped.
        if self.support <= 1024 && value * 32.0 > self.total.abs() {
            return false;
        }
        value <= self.cancellation_limit()
    }

    #[cfg(test)]
    fn excluding(&self, combo: Combo) -> f64 {
        self.excluding_cards(combo.cards.map(|card| card.0))
    }

    /// Reach compatible with a hero holding `cards`, given `support`
    /// ([`support_excluding_cards`](Self::support_excluding_cards)).
    #[inline(always)]
    fn excluding_with_support(&self, cards: [u8; 2], support: i32) -> f64 {
        if support == 0 {
            // Preserve exact zero when every nonzero entry is blocked;
            // rounding residue must not invent reachable private deals.
            0.0
        } else {
            self.total - self.card[slot(cards[0])] - self.card[slot(cards[1])]
        }
    }

    #[cfg(test)]
    fn excluding_cards(&self, cards: [u8; 2]) -> f64 {
        self.excluding_with_support(cards, self.support_excluding_cards(cards))
    }
}

fn disjoint_cards(a: [u8; 2], b: [u8; 2]) -> bool {
    a[0] != b[0] && a[0] != b[1] && a[1] != b[0] && a[1] != b[1]
}

fn live_cards(cards: [u8; 2], board: u64) -> bool {
    (board >> cards[0] | board >> cards[1]) & 1 == 0
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
        let same_pair: [Vec<Vec<usize>>; 2] = std::array::from_fn(|p| {
            combos[p]
                .iter()
                .map(|combo| {
                    lookup[1 - p]
                        .get(&combo.sorted_cards())
                        .cloned()
                        .unwrap_or_default()
                })
                .collect()
        });
        Self {
            any_same_pair: std::array::from_fn(|p| {
                same_pair[p].iter().any(|same| !same.is_empty())
            }),
            same_pair,
            cards: std::array::from_fn(|p| {
                combos[p]
                    .iter()
                    .map(|combo| combo.cards.map(|card| card.0))
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
        let mut out = vec![0.0; combos[player].len()];
        self.compatible_mass_into(combos, player, reach, board, &mut out);
        out
    }

    /// [`compatible_mass`](Self::compatible_mass) written over `out`, which
    /// must hold one entry per `player` combo.
    pub(super) fn compatible_mass_into(
        &self,
        combos: &[Vec<Combo>; 2],
        player: usize,
        reach: &[f64],
        board: u64,
        out: &mut [f64],
    ) {
        debug_assert_eq!(out.len(), combos[player].len());
        let opp_cards = &self.cards[1 - player];
        let mut mass = ReachMass::new();
        for (&cards, &weight) in opp_cards.iter().zip(reach) {
            if live_cards(cards, board) {
                mass.add_cards(cards, weight);
            }
        }
        let limit = mass.cancellation_limit();
        let direct = |hero: [u8; 2]| -> f64 {
            opp_cards
                .iter()
                .zip(reach)
                .filter(|(other, _)| live_cards(**other, board) && disjoint_cards(hero, **other))
                .map(|(_, &weight)| weight)
                .sum()
        };
        if !self.any_same_pair[player] {
            // No opponent entry repeats a hero pair: nothing to add back.
            // (Adding the empty sum would not change a bit: the subtraction
            // below never yields a negative zero.)
            for (&cards, out) in self.cards[player].iter().zip(out.iter_mut()) {
                *out = if live_cards(cards, board) {
                    let support = mass.support_excluding_cards(cards);
                    if support == 0 {
                        0.0
                    } else {
                        let value =
                            mass.total - mass.card[slot(cards[0])] - mass.card[slot(cards[1])];
                        if value <= limit {
                            direct(cards)
                        } else {
                            value
                        }
                    }
                } else {
                    0.0
                };
            }
            return;
        }
        let same_pair = &self.same_pair[player];
        for ((&cards, same), out) in self.cards[player].iter().zip(same_pair).zip(out.iter_mut()) {
            *out = if live_cards(cards, board) {
                // If hero is live, every exact matching opponent pair is too.
                let support = mass.support_excluding_cards(cards)
                    + same.iter().filter(|&&j| reach[j] != 0.0).count() as i32;
                if support == 0 {
                    0.0
                } else {
                    let value = mass.total - mass.card[slot(cards[0])] - mass.card[slot(cards[1])]
                        + same.iter().map(|&j| reach[j]).sum::<f64>();
                    if value <= limit {
                        direct(cards)
                    } else {
                        value
                    }
                }
            } else {
                0.0
            };
        }
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
        let mut values = vec![0.0; combos[player].len()];
        self.showdown_into(
            combos,
            player,
            reach,
            scores,
            order,
            board,
            invested,
            &mut values,
        );
        values
    }

    /// [`showdown`](Self::showdown) written over `values`, which must hold
    /// one entry per `player` combo.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn showdown_into(
        &self,
        combos: &[Vec<Combo>; 2],
        player: usize,
        reach: &[f64],
        scores: &[Vec<u32>; 2],
        order: &BoardOrder,
        board: u64,
        invested: &[f64; 2],
        values: &mut [f64],
    ) {
        debug_assert_eq!(values.len(), combos[player].len());
        let opp = 1 - player;
        let small = invested[0].abs().min(invested[1].abs());
        let large = invested[0].abs().max(invested[1].abs());
        if invested[0] != invested[1] && small < large * 1e-8 {
            // Constructed poker showdowns always have matched investments.
            // Public custom trees may not: avoid cancellation between the
            // affine tie offset and half-pot when their scales differ greatly.
            for (i, (&hero, value)) in combos[player].iter().zip(values.iter_mut()).enumerate() {
                *value = if !live(hero, board) {
                    0.0
                } else {
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
                };
            }
            return;
        }
        let half_pot = (invested[0] + invested[1]) / 2.0;
        let tie_value = (invested[opp] - invested[player]) / 2.0;
        if tie_value == 0.0 {
            values.fill(0.0);
        } else {
            self.compatible_mass_into(combos, player, reach, board, values);
            for value in values.iter_mut() {
                *value *= tie_value;
            }
        }
        let hero_order = &order.order[player];
        let hero_cards = &order.cards[player];
        let bounds = &order.rank_bounds[player];
        let opp_order = &order.order[opp];
        let opp_cards = &order.cards[opp];
        debug_assert!(hero_order.len() == hero_cards.len() && hero_order.len() == bounds.len());

        let mut weaker = ReachMass::new();
        let mut j = 0;
        for ((&i, &hero), &(end, _)) in hero_order.iter().zip(hero_cards).zip(bounds) {
            if j < end {
                for (&k, &cards) in opp_order[j..end].iter().zip(&opp_cards[j..end]) {
                    weaker.add_cards(cards, reach[k]);
                }
                j = end;
            }
            let support = weaker.support_excluding_cards(hero);
            let mut mass = weaker.excluding_with_support(hero, support);
            if support > 0 && weaker.cancellation_prone(mass) {
                mass = opp_order[..j]
                    .iter()
                    .zip(&opp_cards[..j])
                    .filter(|(_, &cards)| disjoint_cards(hero, cards))
                    .map(|(&k, _)| reach[k])
                    .sum();
            }
            values[i] += half_pot * mass;
        }
        let mut stronger = ReachMass::new();
        let mut j = opp_order.len();
        for ((&i, &hero), &(_, start)) in hero_order.iter().zip(hero_cards).zip(bounds).rev() {
            if j > start {
                for (&k, &cards) in opp_order[start..j].iter().zip(&opp_cards[start..j]).rev() {
                    stronger.add_cards(cards, reach[k]);
                }
                j = start;
            }
            let support = stronger.support_excluding_cards(hero);
            let mut mass = stronger.excluding_with_support(hero, support);
            if support > 0 && stronger.cancellation_prone(mass) {
                mass = opp_order[j..]
                    .iter()
                    .zip(&opp_cards[j..])
                    .filter(|(_, &cards)| disjoint_cards(hero, cards))
                    .map(|(&k, _)| reach[k])
                    .sum();
            }
            values[i] -= half_pot * mass;
        }
    }
}

#[cfg(test)]
#[path = "terminal_rank_reference.rs"]
mod rank_reference;
