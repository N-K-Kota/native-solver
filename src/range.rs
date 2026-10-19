//! Hand ranges: the standard 169 canonical starting hands (e.g. "AKs",
//! "AKo", "AA"), each with a weight in `0.0..=1.0`, expandable into
//! concrete card-pair combos with dead-card removal applied.

use crate::card::{card_bit, set_contains, Card, CardSet, RANKS};
use std::collections::HashMap;

/// Why a range spec or hand weight was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum RangeError {
    /// A `hand:weight` token whose weight is not a number.
    InvalidWeight {
        /// The whole `hand:weight` token as written.
        token: String,
    },
    /// A weight that is not finite or lies outside `0.0..=1.0`.
    WeightOutOfRange(f64),
    /// Two different ranks without an `s`/`o` marker, e.g. `AK`.
    NotAPair(String),
    /// A pair with an `s`/`o` marker, e.g. `AAs`.
    PairWithMarker(String),
    /// A third character other than `s` or `o`.
    InvalidMarker(String),
    /// A rank character other than `23456789TJQKA`.
    InvalidRank(char),
    /// Anything else that is not canonical hand notation.
    InvalidNotation(String),
}

impl std::fmt::Display for RangeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidWeight { token } => write!(f, "invalid weight in '{token}'"),
            Self::WeightOutOfRange(_) => {
                write!(f, "range weight must be finite and between 0 and 1")
            }
            Self::NotAPair(hand) => {
                write!(f, "'{hand}' is not a pair; did you mean to add 's' or 'o'?")
            }
            Self::PairWithMarker(hand) => {
                write!(f, "pair '{hand}' cannot have a suited/offsuit marker")
            }
            Self::InvalidMarker(hand) => write!(f, "invalid suited/offsuit marker in '{hand}'"),
            Self::InvalidRank(c) => write!(f, "invalid rank char: {c}"),
            Self::InvalidNotation(hand) => write!(f, "invalid hand notation: '{hand}'"),
        }
    }
}

impl std::error::Error for RangeError {}

/// One concrete two-card starting hand and its weight in a range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Combo {
    /// The two hole cards, in no guaranteed order.
    pub cards: [Card; 2],
    /// The combo's range weight, in `0.0..=1.0`.
    pub weight: f64,
}

impl Combo {
    /// A combo of two distinct cards; their order is kept as given.
    pub fn new(first: Card, second: Card, weight: f64) -> Self {
        Self {
            cards: [first, second],
            weight,
        }
    }

    /// The two cards as a set, for dead-card and blocker checks.
    pub fn mask(&self) -> CardSet {
        card_bit(self.cards[0]) | card_bit(self.cards[1])
    }

    /// Whether the combo shares a card with `cards`.
    pub fn conflicts_with(&self, cards: CardSet) -> bool {
        self.mask() & cards != 0
    }

    /// The cards ordered low then high, so equal hands compare equal
    /// regardless of the order their cards were given in.
    pub fn sorted_cards(&self) -> (Card, Card) {
        let [a, b] = self.cards;
        (a.min(b), a.max(b))
    }
}

/// A weighted preflop range, keyed by canonical hand notation.
#[derive(Debug, Clone, Default)]
pub struct Range {
    weights: HashMap<String, f64>,
}

impl Range {
    /// An empty range: every hand has weight `0.0`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a comma-separated range spec, e.g. `"AA,AKs,AQo:0.5,72o"`.
    /// A hand without an explicit `:weight` suffix gets weight `1.0`.
    pub fn parse(spec: &str) -> Result<Self, RangeError> {
        let mut range = Range::new();
        for token in spec.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let (hand, weight) = match token.split_once(':') {
                Some((h, w)) => (
                    h,
                    w.trim()
                        .parse::<f64>()
                        .map_err(|_| RangeError::InvalidWeight {
                            token: token.into(),
                        })?,
                ),
                None => (token, 1.0),
            };
            range.set_weight(hand, weight)?;
        }
        Ok(range)
    }

    /// Set the weight for a canonical hand (e.g. "AA", "AKs", "72o").
    pub fn set_weight(&mut self, hand: &str, weight: f64) -> Result<(), RangeError> {
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err(RangeError::WeightOutOfRange(weight));
        }
        let canon = canonicalize(hand)?;
        self.weights.insert(canon, weight);
        Ok(())
    }

    /// Get the weight for a canonical hand, defaulting to 0.0 if unset.
    pub fn get_weight(&self, hand: &str) -> f64 {
        canonicalize(hand)
            .ok()
            .and_then(|c| self.weights.get(&c).copied())
            .unwrap_or(0.0)
    }

    /// Expand into concrete combos, dropping any
    /// combo that intersects `dead` (e.g. board cards or the opponent's hand).
    /// Combos are ordered by their lower card, then their higher card, so the
    /// order — and every solver result that sums over it — is the same on
    /// every run regardless of how the range was built.
    pub fn combos(&self, dead: CardSet) -> Vec<Combo> {
        let mut out = Vec::new();
        for (hand, &weight) in &self.weights {
            if weight <= 0.0 {
                continue;
            }
            hand_combos(hand, weight, dead, &mut out);
        }
        out.sort_unstable_by_key(Combo::sorted_cards);
        out
    }
}

/// All 169 canonical hand strings (13 pairs, 78 suited, 78 offsuit).
pub fn all_canonical_hands() -> Vec<String> {
    let ranks = RANKS.as_bytes();
    let mut out = Vec::with_capacity(169);
    for i in (0..13u8).rev() {
        for j in (0..13u8).rev() {
            if i == j {
                let c = ranks[i as usize] as char;
                out.push(format!("{c}{c}"));
            } else if i > j {
                let hi = ranks[i as usize] as char;
                let lo = ranks[j as usize] as char;
                out.push(format!("{hi}{lo}s"));
                out.push(format!("{hi}{lo}o"));
            }
        }
    }
    out
}

fn rank_index(c: char) -> Result<u8, RangeError> {
    RANKS
        .find(c.to_ascii_uppercase())
        .map(|i| i as u8)
        .ok_or(RangeError::InvalidRank(c))
}

/// Normalize hand notation: uppercase ranks, high rank first, lowercase s/o suffix.
fn canonicalize(hand: &str) -> Result<String, RangeError> {
    let chars: Vec<char> = hand.chars().collect();
    match chars.len() {
        2 => {
            let r1 = rank_index(chars[0])?;
            let r2 = rank_index(chars[1])?;
            if r1 != r2 {
                return Err(RangeError::NotAPair(hand.into()));
            }
            let c = chars[0].to_ascii_uppercase();
            Ok(format!("{c}{c}"))
        }
        3 => {
            let r1 = rank_index(chars[0])?;
            let r2 = rank_index(chars[1])?;
            if r1 == r2 {
                return Err(RangeError::PairWithMarker(hand.into()));
            }
            let marker = chars[2].to_ascii_lowercase();
            if marker != 's' && marker != 'o' {
                return Err(RangeError::InvalidMarker(hand.into()));
            }
            let (hi, lo) = if r1 >= r2 {
                (chars[0], chars[1])
            } else {
                (chars[1], chars[0])
            };
            Ok(format!(
                "{}{}{}",
                hi.to_ascii_uppercase(),
                lo.to_ascii_uppercase(),
                marker
            ))
        }
        _ => Err(RangeError::InvalidNotation(hand.into())),
    }
}

fn hand_combos(canon: &str, weight: f64, dead: CardSet, out: &mut Vec<Combo>) {
    let chars: Vec<char> = canon.chars().collect();
    let push_if_alive = |c1: Card, c2: Card, out: &mut Vec<Combo>| {
        if !set_contains(dead, c1) && !set_contains(dead, c2) {
            out.push(Combo::new(c1, c2, weight));
        }
    };

    if chars.len() == 2 {
        let r = rank_index(chars[0]).expect("canonical hand has valid rank");
        for s1 in 0..4u8 {
            for s2 in (s1 + 1)..4u8 {
                push_if_alive(r * 4 + s1, r * 4 + s2, out);
            }
        }
        return;
    }

    let r1 = rank_index(chars[0]).expect("canonical hand has valid rank");
    let r2 = rank_index(chars[1]).expect("canonical hand has valid rank");
    let suited = chars[2] == 's';
    if suited {
        for s in 0..4u8 {
            push_if_alive(r1 * 4 + s, r2 * 4 + s, out);
        }
    } else {
        for s1 in 0..4u8 {
            for s2 in 0..4u8 {
                if s1 != s2 {
                    push_if_alive(r1 * 4 + s1, r2 * 4 + s2, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::{card_from_str, set_from_cards};

    /// `HashMap` iteration order differs between instances, so without the
    /// sort two parses of the same spec could expand in different orders.
    #[test]
    fn combo_order_is_deterministic_and_sorted() {
        let spec = "AA,KK:0.5,AKs,AQo,T9s,72o,JJ,QJs:0.25";
        let dead = set_from_cards(&[card_from_str("Ah").unwrap(), card_from_str("9s").unwrap()]);
        let first = Range::parse(spec).unwrap().combos(dead);
        for _ in 0..50 {
            assert_eq!(Range::parse(spec).unwrap().combos(dead), first);
        }
        let keys: Vec<_> = first.iter().map(Combo::sorted_cards).collect();
        assert!(
            keys.windows(2).all(|pair| pair[0] < pair[1]),
            "combos must be strictly ascending"
        );
    }

    #[test]
    fn pair_expands_to_six_combos() {
        let range = Range::parse("AA").unwrap();
        assert_eq!(range.combos(0).len(), 6);
    }

    #[test]
    fn suited_expands_to_four_combos() {
        let range = Range::parse("AKs").unwrap();
        assert_eq!(range.combos(0).len(), 4);
    }

    #[test]
    fn offsuit_expands_to_twelve_combos() {
        let range = Range::parse("AKo").unwrap();
        assert_eq!(range.combos(0).len(), 12);
    }

    #[test]
    fn full_range_has_1326_combos() {
        let spec = all_canonical_hands().join(",");
        let range = Range::parse(&spec).unwrap();
        assert_eq!(all_canonical_hands().len(), 169);
        assert_eq!(range.combos(0).len(), 1326);
    }

    #[test]
    fn dead_cards_remove_conflicting_combos() {
        let range = Range::parse("AA").unwrap();
        let dead = set_from_cards(&[card_from_str("Ah").unwrap()]);
        // One ace is dead, so only C(3,2) = 3 combos survive.
        assert_eq!(range.combos(dead).len(), 3);
    }

    #[test]
    fn canonicalize_ignores_input_order_and_case() {
        let mut a = Range::new();
        a.set_weight("KAs", 1.0).unwrap();
        let mut b = Range::new();
        b.set_weight("aks", 1.0).unwrap();
        assert_eq!(a.get_weight("AKs"), 1.0);
        assert_eq!(b.get_weight("AKs"), 1.0);
    }

    #[test]
    fn parses_per_hand_weights() {
        let range = Range::parse("AA, AKo:0.5, 72o:0.25").unwrap();
        assert_eq!(range.get_weight("AA"), 1.0);
        assert_eq!(range.get_weight("AKo"), 0.5);
        assert_eq!(range.get_weight("72o"), 0.25);
        assert_eq!(range.get_weight("KK"), 0.0);
    }

    #[test]
    fn rejects_invalid_hand_notation() {
        assert_eq!(
            Range::parse("AKq").unwrap_err(),
            RangeError::InvalidMarker("AKq".into())
        );
        assert_eq!(
            Range::parse("AAs").unwrap_err(),
            RangeError::PairWithMarker("AAs".into())
        );
        assert_eq!(
            Range::parse("AK").unwrap_err(),
            RangeError::NotAPair("AK".into())
        );
        assert_eq!(
            Range::parse("A").unwrap_err(),
            RangeError::InvalidNotation("A".into())
        );
        assert_eq!(
            Range::parse("AXs").unwrap_err(),
            RangeError::InvalidRank('X')
        );
        assert_eq!(
            Range::parse("AA:x").unwrap_err(),
            RangeError::InvalidWeight {
                token: "AA:x".into()
            }
        );
    }

    #[test]
    fn rejects_invalid_weights_including_non_finite_values() {
        for spec in ["AA:NaN", "AA:inf", "AA:-inf", "AA:-0.1", "AA:1.1"] {
            assert!(Range::parse(spec).is_err(), "accepted {spec}");
        }
        assert!(Range::parse("AA:0,KK:1").is_ok());
    }
}
