//! Exact chip amounts for tree construction. Pots, stacks, and bet sizes are
//! integers in units of `1 / CHIP_SCALE` input chips, so all-in detection,
//! raise eligibility, and bet deduplication are exact comparisons and the
//! tree's shape does not depend on the scale of the inputs. Payoffs are
//! converted back to `f64` input chips only at terminals.

use std::fmt;
use std::ops::{Add, AddAssign, Sub, SubAssign};

/// Internal units per input chip (amounts are quantized to 0.001 chips).
pub const CHIP_SCALE: i64 = 1000;

/// Largest unit count accepted from `f64` input: every value up to 2^53 is
/// exact in `f64`, so the conversion back for payoffs is lossless.
const MAX_UNITS: f64 = 9_007_199_254_740_992.0;

/// An exact chip amount, stored as an integer count of `1 / CHIP_SCALE`
/// input chips. Build one with [`Chips::whole`] or [`Chips::from_f64`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Chips(pub(crate) i64);

impl Chips {
    /// No chips.
    pub const ZERO: Chips = Chips(0);

    /// A whole number of input chips.
    pub const fn whole(chips: i64) -> Chips {
        Chips(chips * CHIP_SCALE)
    }

    /// An amount in internal units of `1 / CHIP_SCALE` input chips, so
    /// `Chips::from_units(CHIP_SCALE)` is one chip.
    pub const fn from_units(units: i64) -> Chips {
        Chips(units)
    }

    /// The amount in internal units of `1 / CHIP_SCALE` input chips.
    pub const fn units(self) -> i64 {
        self.0
    }

    /// Converts an input-chip amount that is a multiple of `1 / CHIP_SCALE`
    /// (allowing only `f64` representation error). Returns `None` for
    /// non-finite, out-of-range, or finer-grained amounts instead of silently
    /// solving a different game than the caller described.
    pub fn from_f64(amount: f64) -> Option<Chips> {
        let scaled = amount * CHIP_SCALE as f64;
        if !scaled.is_finite() || scaled.abs() > MAX_UNITS {
            return None;
        }
        let units = scaled.round();
        let error = (units / CHIP_SCALE as f64 - amount).abs();
        (error <= 1e-9 || error <= 4.0 * f64::EPSILON * amount.abs()).then_some(Chips(units as i64))
    }

    /// The amount in input chips.
    pub fn to_f64(self) -> f64 {
        self.0 as f64 / CHIP_SCALE as f64
    }

    /// `fraction` of this amount, rounded half-to-even to a whole unit. The
    /// only floating-point step in sizing; non-finite or negative results
    /// become zero or negative and are discarded by the tree builder.
    pub fn fraction(self, fraction: f64) -> Chips {
        // `as` saturates out-of-range values and maps NaN to 0.
        Chips((self.0 as f64 * fraction).round_ties_even() as i64)
    }
}

impl Add for Chips {
    type Output = Chips;
    fn add(self, rhs: Chips) -> Chips {
        Chips(self.0 + rhs.0)
    }
}

impl AddAssign for Chips {
    fn add_assign(&mut self, rhs: Chips) {
        self.0 += rhs.0;
    }
}

impl Sub for Chips {
    type Output = Chips;
    fn sub(self, rhs: Chips) -> Chips {
        Chips(self.0 - rhs.0)
    }
}

impl SubAssign for Chips {
    fn sub_assign(&mut self, rhs: Chips) {
        self.0 -= rhs.0;
    }
}

/// Decimal input chips with at most three fractional digits and no trailing
/// zeros (`1.815`, `0.5`, `100`), so distinct amounts never print alike.
impl fmt::Display for Chips {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.0 < 0 { "-" } else { "" };
        let units = self.0.unsigned_abs();
        let scale = CHIP_SCALE as u64;
        let (whole, frac) = (units / scale, units % scale);
        if frac == 0 {
            return write!(f, "{sign}{whole}");
        }
        let digits = format!("{frac:03}");
        write!(f, "{sign}{whole}.{}", digits.trim_end_matches('0'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_f64_accepts_thousandths_and_rejects_finer_amounts() {
        assert_eq!(Chips::from_f64(5.5), Some(Chips(5500)));
        assert_eq!(Chips::from_f64(0.001), Some(Chips(1)));
        assert_eq!(Chips::from_f64(0.1 + 0.2), Some(Chips(300)));
        assert_eq!(Chips::from_f64(1e9), Some(Chips::whole(1_000_000_000)));
        assert_eq!(
            Chips::from_f64(123_456_789.123),
            Some(Chips(123_456_789_123))
        );
        assert_eq!(Chips::from_f64(0.0005), None);
        assert_eq!(Chips::from_f64(1e-6), None);
        assert_eq!(Chips::from_f64(1_000_000.000_1), None);
        assert_eq!(Chips::from_f64(f64::NAN), None);
        assert_eq!(Chips::from_f64(f64::INFINITY), None);
        assert_eq!(Chips::from_f64(f64::MAX), None);
    }

    #[test]
    fn display_is_exact_and_trims_trailing_zeros() {
        assert_eq!(Chips(1815).to_string(), "1.815");
        assert_eq!(Chips(500).to_string(), "0.5");
        assert_eq!(Chips(1).to_string(), "0.001");
        assert_eq!(Chips::whole(100).to_string(), "100");
        assert_eq!(Chips(-2050).to_string(), "-2.05");
        assert_ne!(Chips(1814).to_string(), Chips(1815).to_string());
    }

    #[test]
    fn fraction_rounds_half_to_even() {
        assert_eq!(Chips(5500).fraction(0.33), Chips(1815));
        assert_eq!(Chips(5).fraction(0.5), Chips(2));
        assert_eq!(Chips(7).fraction(0.5), Chips(4));
        assert_eq!(Chips(100).fraction(f64::NAN), Chips::ZERO);
    }
}
