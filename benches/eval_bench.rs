//! Benchmarks the table-driven `evaluate_5`/`evaluate_7` against the
//! original grouping-and-sorting `evaluate_5_reference` they replaced (see
//! src/eval.rs).
//!
//! Run with `cargo bench`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use native_solver::eval::{evaluate_5, evaluate_5_reference, evaluate_7};
use native_solver::Card;

/// Tiny xorshift PRNG so the sample is deterministic without adding a
/// `rand` dependency just for benchmark inputs.
struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn random_hand<const N: usize>(rng: &mut Xorshift) -> [Card; N] {
    let mut cards = [Card::from_index(0).unwrap(); N];
    let mut used = 0u64;
    let mut i = 0;
    while i < N {
        let c = (rng.next() % 52) as u8;
        if used & (1 << c) == 0 {
            used |= 1 << c;
            cards[i] = Card::from_index(c).unwrap();
            i += 1;
        }
    }
    cards
}

fn sample_hands<const N: usize>(count: usize) -> Vec<[Card; N]> {
    let mut rng = Xorshift(88172645463325252);
    (0..count).map(|_| random_hand::<N>(&mut rng)).collect()
}

fn bench_evaluate_5(c: &mut Criterion) {
    let hands: Vec<[Card; 5]> = sample_hands(2000);

    c.bench_function("evaluate_5 (table-driven)", |b| {
        b.iter(|| {
            for &h in &hands {
                black_box(evaluate_5(black_box(h)));
            }
        })
    });

    c.bench_function("evaluate_5_reference (old: group + sort)", |b| {
        b.iter(|| {
            for &h in &hands {
                black_box(evaluate_5_reference(black_box(h)));
            }
        })
    });
}

fn bench_evaluate_7(c: &mut Criterion) {
    let hands: Vec<[Card; 7]> = sample_hands(2000);

    c.bench_function("evaluate_7 (best-of-21, table-driven evaluate_5)", |b| {
        b.iter(|| {
            for &h in &hands {
                black_box(evaluate_7(black_box(h)));
            }
        })
    });
}

criterion_group!(benches, bench_evaluate_5, bench_evaluate_7);
criterion_main!(benches);
