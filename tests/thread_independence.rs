//! Results must not depend on the thread count.
//!
//! With one thread the whole tree is walked sequentially in a scratch
//! buffer. With several, subtrees above a work threshold fan out over Rayon
//! and only the small ones below them run sequentially. Both must perform
//! the same floating-point operations in the same order, so every output is
//! compared bit for bit. The trees here are large enough to take the
//! parallel path.
use native_solver::{
    build_tree, Card, CardAbstraction, CardSet, Chips, GameConfig, Range, SolverAlgorithm, Trainer,
    TrainerOptions, UpdateSchedule,
};

/// Every number a solve reports, as raw bits.
fn solve(threads: usize, suit_isomorphism: bool, options: TrainerOptions) -> Vec<u64> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let board: Vec<Card> = ["2c", "7c", "9c", "Kc"]
            .iter()
            .map(|card| card.parse().unwrap())
            .collect();
        let dead = CardSet::from_cards(&board);
        let ranges = [
            "AA,KK,QQ,AKs,AKo,AQs,AQo,KQs,KQo,JTs",
            "TT,99,88,77,66,KJs,KJo,KTs,KTo,QTs,QTo,98s",
        ]
        .map(|range| range.parse::<Range>().unwrap().combos(dead));
        let config = GameConfig::new(board, Chips::whole(100), Chips::whole(200))
            .with_bet_sizes([0.5, 1.0])
            .with_max_raises(1)
            .with_suit_isomorphism(suit_isomorphism);
        let mut trainer = Trainer::new(build_tree(&config).unwrap(), ranges, options).unwrap();

        let mut bits = Vec::new();
        for iterations in [1, 4] {
            bits.extend(trainer.train(iterations).map(f64::to_bits));
            let policy = trainer.policy();
            bits.extend(
                policy
                    .rows()
                    .iter()
                    .flatten()
                    .flatten()
                    .map(|p| p.to_bits()),
            );
            let exploitability = trainer.game().exploitability(&policy);
            bits.extend([exploitability.oop, exploitability.ip].map(f64::to_bits));
            let evaluation = trainer.game().evaluate(&policy);
            bits.extend(evaluation.root_values.map(f64::to_bits));
            for node in &evaluation.nodes {
                bits.extend(node.ev.iter().map(|ev| ev.to_bits()));
                bits.extend(node.equity.iter().flatten().map(|equity| equity.to_bits()));
            }
        }
        bits
    })
}

fn assert_thread_independent(suit_isomorphism: bool, options: TrainerOptions) {
    let sequential = solve(1, suit_isomorphism, options);
    assert!(sequential
        .iter()
        .all(|&bits| f64::from_bits(bits).is_finite()));
    for threads in [2, 5] {
        assert!(
            solve(threads, suit_isomorphism, options) == sequential,
            "{threads} threads changed a result: iso={suit_isomorphism} {options:?}"
        );
    }
}

#[test]
fn every_algorithm_and_schedule_is_thread_independent() {
    for algorithm in [
        SolverAlgorithm::Cfr,
        SolverAlgorithm::CfrPlus,
        SolverAlgorithm::Dcfr,
    ] {
        for schedule in [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating] {
            assert_thread_independent(
                false,
                TrainerOptions {
                    algorithm,
                    schedule,
                    ..TrainerOptions::default()
                },
            );
        }
    }
}

#[test]
fn suit_reduction_and_buckets_are_thread_independent() {
    assert_thread_independent(true, TrainerOptions::default());
    let abstraction = CardAbstraction {
        turn: Some(7),
        river: Some(9),
        ..CardAbstraction::default()
    };
    for suit_isomorphism in [false, true] {
        assert_thread_independent(
            suit_isomorphism,
            TrainerOptions {
                abstraction,
                ..TrainerOptions::default()
            },
        );
    }
}
