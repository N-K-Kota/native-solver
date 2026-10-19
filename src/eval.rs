//! Hand evaluation. `evaluate_5` scores an exact 5-card hand; `evaluate_7`
//! scores the best 5-card hand out of 7 cards (hole cards + board).
//!
//! `evaluate_5` is table-driven for speed, since CFR calls it an enormous
//! number of times. Every 5-card hand falls into one of two cases:
//!
//! - **All 5 ranks distinct**: straight/flush/high-card only (a repeated
//!   rank is impossible here, so no pair-or-better). Which of the 13
//!   ranks are present is a 13-bit bitmask (0..8192); `TABLES.straight_high`
//!   and `TABLES.high_card_tiebreak`, precomputed once and indexed directly
//!   by that bitmask, give the answer with no branching or sorting.
//! - **A rank repeats**: pair or better. Two cards can never share both
//!   rank and suit, so a repeated rank makes a flush structurally
//!   impossible — the hand's category is fully determined by the
//!   *multiset* of ranks alone. Assigning each rank a distinct prime and
//!   multiplying the 5 cards' primes gives a product that's unique per
//!   multiset (unique prime factorization), so `TABLES.paired` (keyed by
//!   that product) looks the score up directly.
//!
//! Both tables are built once, lazily, from `evaluate_5_reference` — the
//! original correctness-first implementation (grouping + sorting, checked
//! all 21 5-card subsets' worth of hand types by hand in earlier
//! milestones) — kept around purely as the oracle for table generation and
//! for the exhaustive cross-check test below. `evaluate_5` and
//! `evaluate_5_reference` are checked equal across all C(52,5) = 2,598,960
//! possible 5-card hands, not just a handful of examples.

use crate::card::{rank, suit, Card};
use std::collections::HashMap;
use std::sync::OnceLock;

const CAT_HIGH_CARD: u32 = 0;
const CAT_PAIR: u32 = 1;
const CAT_TWO_PAIR: u32 = 2;
const CAT_THREE_KIND: u32 = 3;
const CAT_STRAIGHT: u32 = 4;
const CAT_FLUSH: u32 = 5;
const CAT_FULL_HOUSE: u32 = 6;
const CAT_FOUR_KIND: u32 = 7;
const CAT_STRAIGHT_FLUSH: u32 = 8;

/// Hand categories, ordered from weakest to strongest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HandCategory {
    /// No pair or better.
    HighCard,
    /// One pair.
    Pair,
    /// Two pairs.
    TwoPair,
    /// Three cards of one rank.
    ThreeOfAKind,
    /// Five consecutive ranks, including the ace-low wheel.
    Straight,
    /// Five cards of one suit.
    Flush,
    /// Three of one rank and a pair of another.
    FullHouse,
    /// Four cards of one rank.
    FourOfAKind,
    /// A straight in one suit, including the royal flush.
    StraightFlush,
}

/// Extract the hand category from a score returned by `evaluate_5`/`evaluate_7`.
pub fn category_of(score: u32) -> HandCategory {
    match score >> 20 {
        CAT_HIGH_CARD => HandCategory::HighCard,
        CAT_PAIR => HandCategory::Pair,
        CAT_TWO_PAIR => HandCategory::TwoPair,
        CAT_THREE_KIND => HandCategory::ThreeOfAKind,
        CAT_STRAIGHT => HandCategory::Straight,
        CAT_FLUSH => HandCategory::Flush,
        CAT_FULL_HOUSE => HandCategory::FullHouse,
        CAT_FOUR_KIND => HandCategory::FourOfAKind,
        CAT_STRAIGHT_FLUSH => HandCategory::StraightFlush,
        other => unreachable!("invalid hand category bits: {other}"),
    }
}

/// Pack up to 5 tiebreak ranks (most significant first) into the low 20
/// bits of a score, in the same bit positions `pack` uses.
fn pack_ranks(ranks: &[u8]) -> u32 {
    let mut bits = 0u32;
    for (i, &r) in ranks.iter().enumerate() {
        bits |= (r as u32) << (16 - i * 4);
    }
    bits
}

/// Pack a category and up to 5 tiebreak ranks (most significant first) into
/// a single comparable score: higher score always beats lower score.
fn pack(category: u32, ranks: &[u8]) -> u32 {
    (category << 20) | pack_ranks(ranks)
}

/// Ranks (descending, unique) that make up a straight, if any. `ranks_desc`
/// must already be sorted descending and deduplicated.
fn straight_high(ranks_desc: &[u8]) -> Option<u8> {
    if ranks_desc.len() != 5 {
        return None;
    }
    // Wheel: A-2-3-4-5, i.e. rank indices 12,3,2,1,0. High card is the "5" (index 3).
    if ranks_desc == [12, 3, 2, 1, 0] {
        return Some(3);
    }
    for w in ranks_desc.windows(2) {
        if w[0] != w[1] + 1 {
            return None;
        }
    }
    Some(ranks_desc[0])
}

/// Original correctness-first `evaluate_5`: groups ranks by count and sorts,
/// checking every case explicitly. Kept (and `pub`, for `benches/`) as the
/// oracle table generation and the exhaustive test below cross-check the
/// fast path against — not used on the hot path, not the function to call.
/// Hidden from the documented API; it may change without notice.
#[doc(hidden)]
pub fn evaluate_5_reference(cards: [Card; 5]) -> u32 {
    let mut rank_counts = [0u8; 13];
    let mut suit_counts = [0u8; 4];
    for &c in &cards {
        rank_counts[rank(c) as usize] += 1;
        suit_counts[suit(c) as usize] += 1;
    }
    let is_flush = suit_counts.contains(&5);

    let mut ranks_desc: Vec<u8> = (0..13u8).filter(|&r| rank_counts[r as usize] > 0).collect();
    ranks_desc.sort_unstable_by(|a, b| b.cmp(a));
    let straight = straight_high(&ranks_desc);

    if is_flush {
        if let Some(high) = straight {
            return pack(CAT_STRAIGHT_FLUSH, &[high]);
        }
    }

    let mut groups: Vec<(u8, u8)> = (0..13u8)
        .filter(|&r| rank_counts[r as usize] > 0)
        .map(|r| (rank_counts[r as usize], r))
        .collect();
    groups.sort_unstable_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));

    match groups[0].0 {
        4 => pack(CAT_FOUR_KIND, &[groups[0].1, groups[1].1]),
        3 if groups[1].0 >= 2 => pack(CAT_FULL_HOUSE, &[groups[0].1, groups[1].1]),
        3 => pack(CAT_THREE_KIND, &[groups[0].1, groups[1].1, groups[2].1]),
        2 if groups[1].0 == 2 => pack(CAT_TWO_PAIR, &[groups[0].1, groups[1].1, groups[2].1]),
        2 => pack(
            CAT_PAIR,
            &[groups[0].1, groups[1].1, groups[2].1, groups[3].1],
        ),
        _ if is_flush => {
            let ranks: Vec<u8> = groups.iter().map(|g| g.1).collect();
            pack(CAT_FLUSH, &ranks)
        }
        _ if straight.is_some() => pack(CAT_STRAIGHT, &[straight.unwrap()]),
        _ => {
            let ranks: Vec<u8> = groups.iter().map(|g| g.1).collect();
            pack(CAT_HIGH_CARD, &ranks)
        }
    }
}

/// One prime per rank (2..=A); the product of a 5-card hand's rank primes
/// is unique per rank *multiset* (unique prime factorization), so it's a
/// perfect key for the paired-hand lookup table.
const RANK_PRIMES: [u64; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

struct Tables {
    /// Indexed by the 13-bit "which ranks are present" bitmask (only
    /// entries with exactly 5 bits set are meaningful): `Some(high)` if
    /// those 5 ranks form a straight.
    straight_high: Vec<Option<u8>>,
    /// Same indexing; packed tiebreak ranks for the non-straight case
    /// (used for both flush and high-card, which share the same ordering).
    high_card_tiebreak: Vec<u32>,
    /// Rank-prime product -> full packed score, for every 5-rank multiset
    /// that has at least one repeat (pair or better). A repeated rank
    /// makes a flush structurally impossible, so the category is fully
    /// determined by the multiset alone.
    paired: HashMap<u64, u32>,
}

static TABLES: OnceLock<Tables> = OnceLock::new();

fn tables() -> &'static Tables {
    TABLES.get_or_init(build_tables)
}

fn build_tables() -> Tables {
    let mut straight_high_table = vec![None; 8192];
    let mut high_card_tiebreak = vec![0u32; 8192];
    for mask in 0u16..8192 {
        if mask.count_ones() != 5 {
            continue;
        }
        let ranks_desc: Vec<u8> = (0..13u8).rev().filter(|&r| mask & (1 << r) != 0).collect();
        straight_high_table[mask as usize] = straight_high(&ranks_desc);
        high_card_tiebreak[mask as usize] = pack_ranks(&ranks_desc);
    }

    let mut paired = HashMap::new();
    for ranks in rank_multisets_with_repeat() {
        let hand = representative_hand(&ranks);
        let score = evaluate_5_reference(hand);
        let product: u64 = ranks.iter().map(|&r| RANK_PRIMES[r as usize]).product();
        paired.insert(product, score);
    }

    Tables {
        straight_high: straight_high_table,
        high_card_tiebreak,
        paired,
    }
}

/// Every non-increasing sequence of 5 ranks (0..13) with at least one
/// repeated value — i.e. every possible 5-card rank multiset that isn't
/// all-distinct (and excluding 5-of-a-kind, which is impossible with only
/// 4 suits per rank). There are C(17,5) - C(13,5) - 13 = 6188 - 1287 - 13
/// = 4888 of them.
fn rank_multisets_with_repeat() -> Vec<[u8; 5]> {
    let mut out = Vec::new();
    for r0 in 0..13u8 {
        for r1 in 0..=r0 {
            for r2 in 0..=r1 {
                for r3 in 0..=r2 {
                    for r4 in 0..=r3 {
                        let ranks = [r0, r1, r2, r3, r4];
                        let mut counts = [0u8; 13];
                        for &r in &ranks {
                            counts[r as usize] += 1;
                        }
                        let max_count = counts.iter().copied().max().unwrap();
                        if (2..=4).contains(&max_count) {
                            out.push(ranks);
                        }
                    }
                }
            }
        }
    }
    out
}

/// A concrete 5-card hand with the given ranks (assigning increasing
/// suits to repeats of the same rank, which never collides since a rank
/// repeats at most 4 times).
fn representative_hand(ranks: &[u8; 5]) -> [Card; 5] {
    let mut next_suit = [0u8; 13];
    let mut cards = [0u8; 5];
    for (i, &r) in ranks.iter().enumerate() {
        let s = next_suit[r as usize];
        next_suit[r as usize] += 1;
        cards[i] = r * 4 + s;
    }
    cards
}

/// Score an exact 5-card hand. Higher is better.
pub fn evaluate_5(cards: [Card; 5]) -> u32 {
    let t = tables();

    let mut suit_counts = [0u8; 4];
    let mut rank_mask: u16 = 0;
    let mut has_repeat = false;
    for &c in &cards {
        let r = rank(c);
        suit_counts[suit(c) as usize] += 1;
        let bit = 1u16 << r;
        if rank_mask & bit != 0 {
            has_repeat = true;
        }
        rank_mask |= bit;
    }
    let is_flush = suit_counts.contains(&5);

    if !has_repeat {
        if let Some(high) = t.straight_high[rank_mask as usize] {
            let cat = if is_flush {
                CAT_STRAIGHT_FLUSH
            } else {
                CAT_STRAIGHT
            };
            return pack(cat, &[high]);
        }
        let cat = if is_flush { CAT_FLUSH } else { CAT_HIGH_CARD };
        return (cat << 20) | t.high_card_tiebreak[rank_mask as usize];
    }

    let product: u64 = cards
        .iter()
        .map(|&c| RANK_PRIMES[rank(c) as usize])
        .product();
    *t.paired
        .get(&product)
        .expect("every 5-card rank multiset is precomputed in `paired`")
}

/// All 21 ways to choose 5 indices out of 7.
const COMBOS_7_CHOOSE_5: [[usize; 5]; 21] = [
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3, 5],
    [0, 1, 2, 3, 6],
    [0, 1, 2, 4, 5],
    [0, 1, 2, 4, 6],
    [0, 1, 2, 5, 6],
    [0, 1, 3, 4, 5],
    [0, 1, 3, 4, 6],
    [0, 1, 3, 5, 6],
    [0, 1, 4, 5, 6],
    [0, 2, 3, 4, 5],
    [0, 2, 3, 4, 6],
    [0, 2, 3, 5, 6],
    [0, 2, 4, 5, 6],
    [0, 3, 4, 5, 6],
    [1, 2, 3, 4, 5],
    [1, 2, 3, 4, 6],
    [1, 2, 3, 5, 6],
    [1, 2, 4, 5, 6],
    [1, 3, 4, 5, 6],
    [2, 3, 4, 5, 6],
];

/// Score the best 5-card hand out of 7 cards (e.g. 2 hole cards + 5 board cards).
pub fn evaluate_7(cards: [Card; 7]) -> u32 {
    COMBOS_7_CHOOSE_5
        .iter()
        .map(|combo| {
            let five = [
                cards[combo[0]],
                cards[combo[1]],
                cards[combo[2]],
                cards[combo[3]],
                cards[combo[4]],
            ];
            evaluate_5(five)
        })
        .max()
        .expect("21 combos is never empty")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::card_from_str;

    fn hand5(cards: &str) -> [Card; 5] {
        let parsed: Vec<Card> = cards
            .split_whitespace()
            .map(|c| card_from_str(c).unwrap())
            .collect();
        parsed.try_into().unwrap()
    }

    fn hand7(cards: &str) -> [Card; 7] {
        let parsed: Vec<Card> = cards
            .split_whitespace()
            .map(|c| card_from_str(c).unwrap())
            .collect();
        parsed.try_into().unwrap()
    }

    #[test]
    fn categorizes_each_hand_type() {
        assert_eq!(
            category_of(evaluate_5(hand5("As Ks Qs Js Ts"))),
            HandCategory::StraightFlush
        );
        assert_eq!(
            category_of(evaluate_5(hand5("5s 4s 3s 2s As"))),
            HandCategory::StraightFlush
        ); // wheel
        assert_eq!(
            category_of(evaluate_5(hand5("Ah Ad As Ac Kh"))),
            HandCategory::FourOfAKind
        );
        assert_eq!(
            category_of(evaluate_5(hand5("Ah Ad As Kh Kd"))),
            HandCategory::FullHouse
        );
        assert_eq!(
            category_of(evaluate_5(hand5("Ah 9h 7h 4h 2h"))),
            HandCategory::Flush
        );
        assert_eq!(
            category_of(evaluate_5(hand5("9h 8d 7s 6h 5c"))),
            HandCategory::Straight
        );
        assert_eq!(
            category_of(evaluate_5(hand5("Ah Ad As Kh Qd"))),
            HandCategory::ThreeOfAKind
        );
        assert_eq!(
            category_of(evaluate_5(hand5("Ah Ad Kh Kd Qc"))),
            HandCategory::TwoPair
        );
        assert_eq!(
            category_of(evaluate_5(hand5("Ah Ad Kh Qd Jc"))),
            HandCategory::Pair
        );
        assert_eq!(
            category_of(evaluate_5(hand5("Ah Kd Qh Jd 9c"))),
            HandCategory::HighCard
        );
    }

    #[test]
    fn wheel_straight_loses_to_six_high_straight() {
        let wheel = evaluate_5(hand5("5s 4s 3d 2c As"));
        let six_high = evaluate_5(hand5("6s 5d 4d 3c 2h"));
        assert!(six_high > wheel);
    }

    #[test]
    fn higher_trips_beats_lower_trips_in_full_house() {
        let aces_full = evaluate_5(hand5("Ah Ad As Kh Kd"));
        let kings_full = evaluate_5(hand5("Kh Kd Ks Ah Ad"));
        assert!(aces_full > kings_full);
    }

    #[test]
    fn flush_beats_straight_beats_trips() {
        let flush = evaluate_5(hand5("Ah 9h 7h 4h 2h"));
        let straight = evaluate_5(hand5("9h 8d 7s 6h 5c"));
        let trips = evaluate_5(hand5("Ah Ad As Kh Qd"));
        assert!(flush > straight);
        assert!(straight > trips);
    }

    #[test]
    fn evaluate_7_picks_best_two_pair_from_three_pairs() {
        // Board AA KK QQ, hole cards 2 3: best hand should be AAKKQ (top two pair), not AAQQ or KKQQ.
        let best = evaluate_7(hand7("2c 3d Ah Ad Kh Kd Qh"));
        let expected = evaluate_5(hand5("Ah Ad Kh Kd Qh"));
        assert_eq!(best, expected);
        assert_eq!(category_of(best), HandCategory::TwoPair);
    }

    #[test]
    fn evaluate_7_finds_straight_using_board_and_hole_cards() {
        // Hole: 9h 8d, board: 7s 6h 5c 2d 2s -> best is 9-high straight.
        let best = evaluate_7(hand7("9h 8d 7s 6h 5c 2d 2s"));
        assert_eq!(category_of(best), HandCategory::Straight);
        let expected = evaluate_5(hand5("9h 8d 7s 6h 5c"));
        assert_eq!(best, expected);
    }

    /// Cross-checks the table-driven `evaluate_5` against the original
    /// grouping-based `evaluate_5_reference` for every one of the
    /// C(52,5) = 2,598,960 possible 5-card hands — not just a handful of
    /// hand-picked examples.
    #[test]
    fn evaluate_5_matches_reference_for_every_possible_hand() {
        let mut checked = 0u64;
        for a in 0u8..52 {
            for b in (a + 1)..52 {
                for c in (b + 1)..52 {
                    for d in (c + 1)..52 {
                        for e in (d + 1)..52 {
                            let hand = [a, b, c, d, e];
                            let fast = evaluate_5(hand);
                            let reference = evaluate_5_reference(hand);
                            assert_eq!(fast, reference, "mismatch for hand {hand:?}");
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 2_598_960);
    }
}
