//! Card representation: a card is a `u8` in `0..52` (`rank * 4 + suit`).
//! Hands/boards are represented as `u64` bitmasks (`CardSet`) for fast
//! membership and removal checks.

/// A single card, encoded as `rank * 4 + suit`.
pub type Card = u8;

/// A set of cards packed as a bitmask, one bit per card index.
pub type CardSet = u64;

/// Rank characters, low to high: 2..=9, T, J, Q, K, A.
pub const RANKS: &str = "23456789TJQKA";
/// Suit characters: clubs, diamonds, hearts, spades.
pub const SUITS: &str = "cdhs";

/// Rank index in `0..13` (0 = "2", 12 = "A").
pub fn rank(card: Card) -> u8 {
    card / 4
}

/// Suit index in `0..4`.
pub fn suit(card: Card) -> u8 {
    card % 4
}

/// Why a card string could not be parsed; each variant carries the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CardParseError {
    /// Not exactly two characters.
    Length(String),
    /// The first character is not one of `23456789TJQKA`.
    Rank(String),
    /// The second character is not one of `cdhs`.
    Suit(String),
}

impl std::fmt::Display for CardParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Length(s) => write!(f, "invalid card string: {s:?}"),
            Self::Rank(s) => write!(f, "invalid rank in card {s:?}"),
            Self::Suit(s) => write!(f, "invalid suit in card {s:?}"),
        }
    }
}

impl std::error::Error for CardParseError {}

/// Parse a card from its two-character notation, e.g. `"As"`, `"Td"`, `"2c"`.
/// Rank and suit are case-insensitive.
pub fn card_from_str(s: &str) -> Result<Card, CardParseError> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() != 2 {
        return Err(CardParseError::Length(s.into()));
    }
    let rank_char = chars[0].to_ascii_uppercase();
    let suit_char = chars[1].to_ascii_lowercase();
    let rank = RANKS
        .find(rank_char)
        .ok_or_else(|| CardParseError::Rank(s.into()))? as u8;
    let suit = SUITS
        .find(suit_char)
        .ok_or_else(|| CardParseError::Suit(s.into()))? as u8;
    Ok(rank * 4 + suit)
}

/// Format a card back to its two-character notation.
pub fn card_to_string(card: Card) -> String {
    let r = RANKS.as_bytes()[rank(card) as usize] as char;
    let s = SUITS.as_bytes()[suit(card) as usize] as char;
    format!("{r}{s}")
}

/// Bitmask for a single card.
pub fn card_bit(card: Card) -> CardSet {
    1u64 << card
}

/// Build a `CardSet` from a slice of cards.
pub fn set_from_cards(cards: &[Card]) -> CardSet {
    cards.iter().fold(0u64, |acc, &c| acc | card_bit(c))
}

/// Whether `set` contains `card`.
pub fn set_contains(set: CardSet, card: Card) -> bool {
    set & card_bit(card) != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_errors_say_which_part_is_wrong() {
        assert_eq!(card_from_str("A"), Err(CardParseError::Length("A".into())));
        assert_eq!(card_from_str("Xs"), Err(CardParseError::Rank("Xs".into())));
        assert_eq!(card_from_str("Ax"), Err(CardParseError::Suit("Ax".into())));
        assert_eq!(
            CardParseError::Rank("Xs".into()).to_string(),
            "invalid rank in card \"Xs\""
        );
        assert_eq!(card_from_str("aS"), card_from_str("As"));
    }

    #[test]
    fn parses_and_formats_round_trip() {
        for s in ["As", "Kh", "Td", "2c", "9s"] {
            let card = card_from_str(s).unwrap();
            assert_eq!(card_to_string(card), s);
        }
    }

    #[test]
    fn accepts_lowercase_rank_and_uppercase_suit() {
        assert_eq!(card_from_str("as").unwrap(), card_from_str("As").unwrap());
    }

    #[test]
    fn rank_and_suit_indices() {
        let ace_spades = card_from_str("As").unwrap();
        assert_eq!(rank(ace_spades), 12);
        assert_eq!(suit(ace_spades), 3);

        let two_clubs = card_from_str("2c").unwrap();
        assert_eq!(rank(two_clubs), 0);
        assert_eq!(suit(two_clubs), 0);
    }

    #[test]
    fn rejects_invalid_cards() {
        assert!(card_from_str("Xs").is_err());
        assert!(card_from_str("A").is_err());
        assert!(card_from_str("Azz").is_err());
    }

    #[test]
    fn card_set_membership() {
        let as_ = card_from_str("As").unwrap();
        let kh = card_from_str("Kh").unwrap();
        let td = card_from_str("Td").unwrap();
        let set = set_from_cards(&[as_, kh]);
        assert!(set_contains(set, as_));
        assert!(set_contains(set, kh));
        assert!(!set_contains(set, td));
    }
}
