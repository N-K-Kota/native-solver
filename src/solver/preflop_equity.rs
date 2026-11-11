//! Exact all-in equities between two hold'em hands before any board card.
//!
//! A preflop tree ends a called pot by running out all five board cards. For
//! two disjoint hands that is a count over the `C(48, 5) = 1,712,304` boards
//! left in the deck: how often the first hand wins and how often they tie.
//!
//! Renaming suits does not change a matchup, so the 812,175 unordered pairs
//! of disjoint hands fall into [`CLASS_COUNT`] classes. The crate ships one
//! exact `(wins, ties)` pair of integers per class in `preflop_equity.bin`,
//! produced by [`generate_preflop_equity_table`]; nothing here is sampled or
//! rounded.

use crate::card::Card;
use crate::eval::HandState;
use crate::range::Combo;
use rayon::prelude::*;
use std::sync::OnceLock;

/// Five-card boards that avoid four given hole cards.
pub(crate) const BOARDS: u32 = 1_712_304;

/// Suit-isomorphism classes of unordered pairs of disjoint hands.
pub(crate) const CLASS_COUNT: usize = 47_008;

const MAGIC: [u8; 4] = *b"NSPE";
const FORMAT_VERSION: u32 = 1;
const HEADER_BYTES: usize = 12;
const ENTRY_BYTES: usize = 12;

static TABLE_BYTES: &[u8] = include_bytes!("preflop_equity.bin");

/// All 24 ways to rename the four suits.
const SUIT_PERMUTATIONS: [[u8; 4]; 24] = suit_permutations();

const fn suit_permutations() -> [[u8; 4]; 24] {
    let mut out = [[0u8; 4]; 24];
    let mut count = 0;
    let mut a = 0u8;
    while a < 4 {
        let mut b = 0u8;
        while b < 4 {
            let mut c = 0u8;
            while c < 4 {
                if a != b && a != c && b != c {
                    // The suits are 0..4, so the fourth is what is left of 6.
                    out[count] = [a, b, c, 6 - a - b - c];
                    count += 1;
                }
                c += 1;
            }
            b += 1;
        }
        a += 1;
    }
    assert!(count == 24);
    out
}

/// A hand as 12 bits: its lower card index, then its higher.
fn hand_code(cards: [u8; 2], suits: &[u8; 4]) -> u32 {
    let rename = |card: u8| (card & !3) | suits[(card & 3) as usize];
    let (a, b) = (rename(cards[0]) as u32, rename(cards[1]) as u32);
    (a.min(b) << 6) | a.max(b)
}

/// The class of the matchup `first` against `second`: the smallest key any
/// suit renaming gives, and whether `first` is the second hand of that key.
///
/// When both orders give the same smallest key the matchup is symmetric, so
/// the orientation does not matter.
fn canonical(first: [u8; 2], second: [u8; 2]) -> (u32, bool) {
    let mut best = (u32::MAX, false);
    for suits in &SUIT_PERMUTATIONS {
        let (a, b) = (hand_code(first, suits), hand_code(second, suits));
        let candidate = if a <= b {
            ((a << 12) | b, false)
        } else {
            ((b << 12) | a, true)
        };
        if candidate.0 < best.0 {
            best = candidate;
        }
    }
    best
}

fn key_hands(key: u32) -> [[u8; 2]; 2] {
    let hand = |code: u32| [(code >> 6) as u8, (code & 63) as u8];
    [hand(key >> 12), hand(key & 0xfff)]
}

/// Every class key, ascending.
fn class_keys() -> Vec<u32> {
    let mut keys = Vec::new();
    for a in 0..52u8 {
        for b in a + 1..52 {
            for c in a + 1..52 {
                for d in c + 1..52 {
                    if c != b && d != b {
                        keys.push(canonical([a, b], [c, d]).0);
                    }
                }
            }
        }
    }
    keys.sort_unstable();
    keys.dedup();
    keys
}

/// How often `hands[0]` beats and ties `hands[1]` over every board.
fn count_outcomes(hands: [[u8; 2]; 2]) -> (u32, u32) {
    let used = hands
        .iter()
        .flatten()
        .fold(0u64, |mask, &card| mask | 1 << card);
    let deck: Vec<Card> = (0..52u8)
        .filter(|&card| used >> card & 1 == 0)
        .map(Card)
        .collect();
    let (mut wins, mut ties) = (0u32, 0u32);
    let n = deck.len();
    let empty = HandState::default();
    for a in 0..n {
        let board1 = empty.with(deck[a]);
        for b in a + 1..n {
            let board2 = board1.with(deck[b]);
            for c in b + 1..n {
                let board3 = board2.with(deck[c]);
                for d in c + 1..n {
                    let board4 = board3.with(deck[d]);
                    for &river in &deck[d + 1..] {
                        let board = board4.with(river);
                        let score =
                            |hand: [u8; 2]| board.with(Card(hand[0])).with(Card(hand[1])).score();
                        match score(hands[0]).cmp(&score(hands[1])) {
                            std::cmp::Ordering::Greater => wins += 1,
                            std::cmp::Ordering::Equal => ties += 1,
                            std::cmp::Ordering::Less => {}
                        }
                    }
                }
            }
        }
    }
    (wins, ties)
}

/// Computes the table shipped as `src/solver/preflop_equity.bin`.
///
/// Enumerates every board for every class: tens of CPU-minutes. Only the
/// generator example and an ignored test call it.
#[doc(hidden)]
pub fn generate_preflop_equity_table() -> Vec<u8> {
    let keys = class_keys();
    assert_eq!(keys.len(), CLASS_COUNT);
    let counts: Vec<(u32, u32)> = keys
        .par_iter()
        .map(|&key| count_outcomes(key_hands(key)))
        .collect();
    let mut bytes = Vec::with_capacity(HEADER_BYTES + ENTRY_BYTES * keys.len());
    bytes.extend_from_slice(&MAGIC);
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for (key, (wins, ties)) in keys.iter().zip(counts) {
        for value in [*key, wins, ties] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// The shipped table: class key, wins and ties of the key's first hand,
/// ascending by key.
fn table() -> &'static [(u32, u32, u32)] {
    static TABLE: OnceLock<Vec<(u32, u32, u32)>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let word =
            |offset: usize| u32::from_le_bytes(TABLE_BYTES[offset..offset + 4].try_into().unwrap());
        assert!(
            TABLE_BYTES.len() == HEADER_BYTES + ENTRY_BYTES * CLASS_COUNT
                && TABLE_BYTES[..4] == MAGIC
                && word(4) == FORMAT_VERSION
                && word(8) as usize == CLASS_COUNT,
            "preflop equity table is damaged"
        );
        (0..CLASS_COUNT)
            .map(|i| {
                let at = HEADER_BYTES + ENTRY_BYTES * i;
                (word(at), word(at + 4), word(at + 8))
            })
            .collect()
    })
}

/// Boards on which `first` wins, ties, and loses against `second`. The hands
/// must not share a card.
pub(crate) fn outcomes(first: [Card; 2], second: [Card; 2]) -> [u32; 3] {
    let (key, swapped) = canonical(first.map(|card| card.0), second.map(|card| card.0));
    let table = table();
    let entry = table
        .binary_search_by_key(&key, |entry| entry.0)
        .expect("every matchup of disjoint hands has a class");
    let (_, wins, ties) = table[entry];
    let losses = BOARDS - wins - ties;
    if swapped {
        [losses, ties, wins]
    } else {
        [wins, ties, losses]
    }
}

/// Per pair of combos, the first player's win probability minus its loss
/// probability over all runouts, for both seats.
pub(crate) struct RunoutEdge {
    /// `[player]`: row-major, one row per own combo and one column per
    /// opponent combo. Zero where the two combos share a card.
    edge: [Vec<f64>; 2],
    widths: [usize; 2],
}

impl RunoutEdge {
    pub(crate) fn new(combos: &[Vec<Combo>; 2]) -> Self {
        let widths = [combos[1].len(), combos[0].len()];
        let rows: Vec<Vec<f64>> = combos[0]
            .par_iter()
            .map(|own| {
                combos[1]
                    .iter()
                    .map(|other| {
                        if own.bits() & other.bits() != 0 || own.cards[0] == own.cards[1] {
                            return 0.0;
                        }
                        let [wins, _, losses] = outcomes(own.cards, other.cards);
                        (wins as f64 - losses as f64) / BOARDS as f64
                    })
                    .collect()
            })
            .collect();
        let mut transposed = vec![0.0; combos[0].len() * combos[1].len()];
        for (i, row) in rows.iter().enumerate() {
            for (j, &edge) in row.iter().enumerate() {
                // The opponent's edge is the exact negation.
                transposed[j * widths[1] + i] = -edge;
            }
        }
        Self {
            edge: [rows.concat(), transposed],
            widths,
        }
    }

    /// `out[i] = sum over opponent combos j of edge[i][j] * reach[j]`.
    ///
    /// The sum runs in four interleaved lanes that are added at the end, in
    /// an order fixed here, so it vectorizes without depending on the CPU.
    pub(crate) fn apply(&self, player: usize, reach: &[f64], out: &mut [f64]) {
        let width = self.widths[player];
        debug_assert_eq!(reach.len(), width);
        // A row of zero reach sums to +0.0 in every lane.
        if width == 0 || reach.iter().all(|&weight| weight == 0.0) {
            out.fill(0.0);
            return;
        }
        for (out, row) in out.iter_mut().zip(self.edge[player].chunks_exact(width)) {
            let mut lanes = [0.0f64; 4];
            let (row_chunks, reach_chunks) = (row.chunks_exact(4), reach.chunks_exact(4));
            let (row_tail, reach_tail) = (row_chunks.remainder(), reach_chunks.remainder());
            for (edge, reach) in row_chunks.zip(reach_chunks) {
                for lane in 0..4 {
                    lanes[lane] += edge[lane] * reach[lane];
                }
            }
            let mut sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            for (edge, reach) in row_tail.iter().zip(reach_tail) {
                sum += edge * reach;
            }
            *out = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hand(text: &str) -> [Card; 2] {
        [text[..2].parse().unwrap(), text[2..].parse().unwrap()]
    }

    #[test]
    fn the_shipped_table_lists_every_class_once_in_order() {
        let table = table();
        assert_eq!(table.len(), CLASS_COUNT);
        assert!(table.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(table.iter().all(|&(_, wins, ties)| wins + ties <= BOARDS));
        let keys = class_keys();
        assert!(table.iter().map(|entry| entry.0).eq(keys));
    }

    #[test]
    fn sampled_classes_match_a_fresh_enumeration() {
        // Every 2,611th class: 19 full enumerations.
        for &(key, wins, ties) in table().iter().step_by(2611) {
            assert_eq!(count_outcomes(key_hands(key)), (wins, ties), "key {key:#x}");
        }
    }

    #[test]
    fn known_matchups() {
        let equity = |first: &str, second: &str| {
            let [wins, ties, losses] = outcomes(hand(first), hand(second));
            assert_eq!(wins + ties + losses, BOARDS);
            (wins as f64 + ties as f64 / 2.0) / BOARDS as f64
        };
        // Widely published figures: aces are a bit better than a 4-to-1
        // favorite over kings, a small pair against two overcards is close
        // to a coin flip, and a dominated ace wins about a quarter.
        assert!((0.81..0.83).contains(&equity("AsAh", "KdKc")));
        assert!((0.52..0.56).contains(&equity("2s2h", "AdKc")));
        assert!((0.45..0.48).contains(&equity("AsKs", "QdQc")));
        assert!((0.23..0.30).contains(&equity("AdQc", "AsKh")));
        // Sharing suits with the kings takes away some of their flushes.
        assert!(equity("AsAh", "KsKh") > equity("AsAh", "KdKc"));

        // The same hand in two suits can only tie or win by a flush.
        let [wins, ties, losses] = outcomes(hand("AsKs"), hand("AhKh"));
        assert_eq!(wins, losses);
        assert!(ties > 9 * wins);
    }

    #[test]
    fn outcomes_reverse_with_the_hands_and_ignore_suit_names() {
        let (first, second) = (hand("Qh9h"), hand("Qd8s"));
        let forward = outcomes(first, second);
        let backward = outcomes(second, first);
        assert_eq!(forward, [backward[2], backward[1], backward[0]]);
        // Hearts become clubs, diamonds stay, spades become hearts.
        assert_eq!(outcomes(hand("Qc9c"), hand("Qd8h")), forward);
        // Card order inside a hand does not matter.
        assert_eq!(outcomes(hand("9hQh"), hand("8sQd")), forward);
        assert_eq!(forward, {
            let (wins, ties) = count_outcomes([first, second].map(|h| h.map(|c| c.0)));
            [wins, ties, BOARDS - wins - ties]
        });
    }

    #[test]
    fn edge_rows_are_antisymmetric_and_skip_shared_cards() {
        let combo = |text: &str| Combo::new(hand(text)[0], hand(text)[1], 1.0);
        let combos = [
            vec![combo("AsAh"), combo("7d2c"), combo("KsQs")],
            vec![combo("AsKd"), combo("JhTh")],
        ];
        let edge = RunoutEdge::new(&combos);
        let mut oop = [0.0; 3];
        edge.apply(0, &[1.0, 0.0], &mut oop);
        assert_eq!(oop[0], 0.0, "AsAh shares the ace of spades with AsKd");
        assert!(oop[1] < 0.0 && oop[2] < 0.0);
        let mut ip = [0.0; 2];
        edge.apply(1, &[0.0, 1.0, 0.0], &mut ip);
        assert_eq!(ip[0], -oop[1]);
    }

    /// Regenerates the whole table; tens of CPU-minutes. Run with
    /// `cargo test --release the_shipped_table_matches_the_generator -- --ignored`.
    #[test]
    #[ignore = "exhaustive: tens of CPU-minutes"]
    fn the_shipped_table_matches_the_generator() {
        assert!(generate_preflop_equity_table() == TABLE_BYTES);
    }
}
