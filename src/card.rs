//! Cards and card sets. A [`Card`] is one of the 52 cards of a standard
//! deck; a [`CardSet`] is a set of cards packed into a bitmask for fast
//! membership and removal checks.

use std::fmt;
use std::str::FromStr;

/// Rank characters, low to high: 2..=9, T, J, Q, K, A.
pub const RANKS: &str = "23456789TJQKA";
/// Suit characters: clubs, diamonds, hearts, spades.
pub const SUITS: &str = "cdhs";

/// One of the 52 cards, stored as its index `rank * 4 + suit`.
///
/// Every [`Card`] value is valid: it can only be built from an index in
/// `0..52`, a rank and suit in range, or parsed from text.
///
/// ```
/// use native_solver::Card;
///
/// let card: Card = "As".parse()?;
/// assert_eq!((card.rank(), card.suit()), (12, 3));
/// assert_eq!(card.to_string(), "As");
/// assert_eq!(Card::new(12, 3), Some(card));
/// # Ok::<(), native_solver::CardParseError>(())
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Card(pub(crate) u8);

impl Card {
    /// The card with rank index `rank` in `0..13` (0 = "2", 12 = "A") and suit
    /// index `suit` in `0..4` (clubs, diamonds, hearts, spades), or `None`
    /// if either is out of range.
    pub const fn new(rank: u8, suit: u8) -> Option<Card> {
        if rank < 13 && suit < 4 {
            Some(Card(rank * 4 + suit))
        } else {
            None
        }
    }

    /// The card with index `index` in `0..52` (`rank * 4 + suit`), or `None`
    /// if the index is out of range.
    pub const fn from_index(index: u8) -> Option<Card> {
        if index < 52 {
            Some(Card(index))
        } else {
            None
        }
    }

    /// The index `rank * 4 + suit`, in `0..52`.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Rank index in `0..13` (0 = "2", 12 = "A").
    pub const fn rank(self) -> u8 {
        self.0 / 4
    }

    /// Suit index in `0..4`: clubs, diamonds, hearts, spades.
    pub const fn suit(self) -> u8 {
        self.0 % 4
    }

    /// All 52 cards in index order.
    pub fn all() -> impl Iterator<Item = Card> {
        (0..52).map(Card)
    }
}

impl fmt::Display for Card {
    /// Two characters: rank then suit, e.g. `As`, `Td`, `2c`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = RANKS.as_bytes()[self.rank() as usize] as char;
        let s = SUITS.as_bytes()[self.suit() as usize] as char;
        write!(f, "{r}{s}")
    }
}

impl fmt::Debug for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Card({self})")
    }
}

/// Why a card string could not be parsed; each variant carries the input.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CardParseError {
    /// Not exactly two characters.
    Length(String),
    /// The first character is not one of `23456789TJQKA`.
    Rank(String),
    /// The second character is not one of `cdhs`.
    Suit(String),
}

impl fmt::Display for CardParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length(s) => write!(f, "invalid card string: {s:?}"),
            Self::Rank(s) => write!(f, "invalid rank in card {s:?}"),
            Self::Suit(s) => write!(f, "invalid suit in card {s:?}"),
        }
    }
}

impl std::error::Error for CardParseError {}

impl FromStr for Card {
    type Err = CardParseError;

    /// Parses the two-character notation, e.g. `"As"`, `"Td"`, `"2c"`. Rank
    /// and suit are case-insensitive.
    ///
    /// # Errors
    /// Returns a [`CardParseError`] naming the part that is wrong: the
    /// length, the rank, or the suit.
    fn from_str(s: &str) -> Result<Card, CardParseError> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != 2 {
            return Err(CardParseError::Length(s.into()));
        }
        let rank = RANKS
            .find(chars[0].to_ascii_uppercase())
            .ok_or_else(|| CardParseError::Rank(s.into()))? as u8;
        let suit = SUITS
            .find(chars[1].to_ascii_lowercase())
            .ok_or_else(|| CardParseError::Suit(s.into()))? as u8;
        Ok(Card(rank * 4 + suit))
    }
}

/// A set of cards, packed as a 52-bit mask (bit `card.index()` per card).
///
/// ```
/// use native_solver::{Card, CardSet};
///
/// let board: Vec<Card> = ["Ks", "Th", "7s"].iter().map(|s| s.parse()).collect::<Result<_, _>>()?;
/// let dead = CardSet::from_cards(&board);
/// assert!(dead.contains("Ks".parse()?));
/// assert_eq!(dead.len(), 3);
/// # Ok::<(), native_solver::CardParseError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct CardSet(pub(crate) u64);

impl CardSet {
    /// The empty set.
    pub const EMPTY: CardSet = CardSet(0);

    /// The set holding exactly `cards`.
    pub fn from_cards(cards: &[Card]) -> CardSet {
        cards.iter().copied().collect()
    }

    /// Whether `card` is in the set.
    pub const fn contains(self, card: Card) -> bool {
        self.0 & (1 << card.0) != 0
    }

    /// Adds `card`; returns whether it was newly added.
    pub fn insert(&mut self, card: Card) -> bool {
        let added = !self.contains(card);
        self.0 |= 1 << card.0;
        added
    }

    /// Whether the two sets share no card.
    pub const fn is_disjoint(self, other: CardSet) -> bool {
        self.0 & other.0 == 0
    }

    /// The cards in either set.
    pub const fn union(self, other: CardSet) -> CardSet {
        CardSet(self.0 | other.0)
    }

    /// The number of cards in the set.
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    /// Whether the set is empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The cards in the set, in index order.
    pub fn iter(self) -> impl Iterator<Item = Card> {
        Card::all().filter(move |&card| self.contains(card))
    }
}

impl From<Card> for CardSet {
    fn from(card: Card) -> CardSet {
        CardSet(1 << card.0)
    }
}

impl FromIterator<Card> for CardSet {
    fn from_iter<I: IntoIterator<Item = Card>>(cards: I) -> CardSet {
        let mut set = CardSet::EMPTY;
        set.extend(cards);
        set
    }
}

impl Extend<Card> for CardSet {
    fn extend<I: IntoIterator<Item = Card>>(&mut self, cards: I) {
        for card in cards {
            self.insert(card);
        }
    }
}

impl std::ops::BitOr for CardSet {
    type Output = CardSet;

    fn bitor(self, other: CardSet) -> CardSet {
        self.union(other)
    }
}

/// Raw `u64` mask helpers for the solver's hot paths, which keep card sets
/// as plain bitmasks.
pub(crate) fn bit(card: Card) -> u64 {
    1 << card.0
}

/// The raw mask of `cards`.
pub(crate) fn mask_of(cards: &[Card]) -> u64 {
    cards.iter().fold(0, |mask, &card| mask | bit(card))
}

/// Whether the raw `mask` contains `card`.
pub(crate) fn mask_has(mask: u64, card: Card) -> bool {
    mask & bit(card) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(s: &str) -> Card {
        s.parse().unwrap()
    }

    #[test]
    fn parse_errors_say_which_part_is_wrong() {
        assert_eq!("A".parse::<Card>(), Err(CardParseError::Length("A".into())));
        assert_eq!("Xs".parse::<Card>(), Err(CardParseError::Rank("Xs".into())));
        assert_eq!("Ax".parse::<Card>(), Err(CardParseError::Suit("Ax".into())));
        assert_eq!(
            CardParseError::Rank("Xs".into()).to_string(),
            "invalid rank in card \"Xs\""
        );
        assert_eq!("aS".parse::<Card>(), "As".parse::<Card>());
    }

    #[test]
    fn parses_and_formats_round_trip() {
        for s in ["As", "Kh", "Td", "2c", "9s"] {
            assert_eq!(card(s).to_string(), s);
        }
        assert_eq!(format!("{:?}", card("Td")), "Card(Td)");
    }

    #[test]
    fn rank_and_suit_indices() {
        assert_eq!((card("As").rank(), card("As").suit()), (12, 3));
        assert_eq!((card("2c").rank(), card("2c").suit()), (0, 0));
        assert_eq!(card("As").index(), 51);
    }

    #[test]
    fn constructors_reject_out_of_range_values() {
        assert_eq!(Card::new(12, 3), Some(card("As")));
        assert_eq!(Card::new(13, 0), None);
        assert_eq!(Card::new(0, 4), None);
        assert_eq!(Card::from_index(51), Some(card("As")));
        assert_eq!(Card::from_index(52), None);
        assert_eq!(Card::all().count(), 52);
        assert!(Card::all().all(|c| Card::from_index(c.index()) == Some(c)));
    }

    #[test]
    fn rejects_invalid_cards() {
        assert!("Xs".parse::<Card>().is_err());
        assert!("A".parse::<Card>().is_err());
        assert!("Azz".parse::<Card>().is_err());
    }

    #[test]
    fn card_set_operations() {
        let mut set = CardSet::from_cards(&[card("As"), card("Kh")]);
        assert!(set.contains(card("As")) && set.contains(card("Kh")));
        assert!(!set.contains(card("Td")));
        assert_eq!(set.len(), 2);
        assert!(set.insert(card("Td")));
        assert!(!set.insert(card("Td")));
        assert_eq!(
            set.iter().collect::<Vec<_>>(),
            [card("Td"), card("Kh"), card("As")]
        );
        assert!(set.is_disjoint(CardSet::from(card("2c"))));
        assert!(!set.is_disjoint(CardSet::from(card("As")) | CardSet::from(card("2c"))));
        assert!(CardSet::EMPTY.is_empty());
    }
}
