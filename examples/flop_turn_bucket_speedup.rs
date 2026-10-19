//! Ad-hoc timing/memory comparison for flop/turn card abstraction
//! (bucketing) — not a criterion bench, just a measurement script run
//! manually during development. Run with:
//!   cargo run --release --example flop_turn_bucket_speedup

use native_solver::card::{card_from_str, set_from_cards};
use native_solver::range::Range;
use native_solver::solver::Chips;
use native_solver::solver::{build_tree, CardAbstraction, GameConfig, Trainer};
use native_solver::solver::{SolverAlgorithm, TrainerOptions};
use std::time::Instant;

fn run(abstraction: CardAbstraction, iterations: u32) -> (std::time::Duration, [f64; 2]) {
    // Monotone flop -> maximal isomorphism benefit, so the tree (two
    // chance nodes: flop deals the turn, which deals the river) stays
    // tractable even for the unabstracted baseline.
    let board: Vec<_> = ["2c", "7c", "9c"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let dead = set_from_cards(&board);
    let oop_combos = Range::parse("JJ,TT,99,AJs,KJs,AJo,KJo")
        .unwrap()
        .combos(dead);
    let ip_combos = Range::parse("88,77,ATs,KTs,ATo,KTo").unwrap().combos(dead);

    let cfg = GameConfig {
        board: board.clone(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(200),
        bet_sizes: vec![1.0],
        max_raises: 0,
        suit_isomorphism: true,
    };
    let tree = build_tree(&cfg).unwrap();
    let mut trainer = Trainer::new(
        tree,
        board,
        [oop_combos, ip_combos],
        TrainerOptions {
            algorithm: SolverAlgorithm::CfrPlus,
            abstraction,
            ..TrainerOptions::default()
        },
    )
    .unwrap();
    let start = Instant::now();
    let ev = trainer.train(iterations);
    (start.elapsed(), ev)
}

fn main() {
    let iterations = 100;
    println!("iterations={iterations}");
    // ev is train()'s *last-iteration current-strategy* value (informal,
    // see its doc comment) -- only useful here as a zero-sum sanity check.

    let (dur_none, ev_none) = run(CardAbstraction::default(), iterations);
    println!("no abstraction:        {dur_none:?}, ev={ev_none:?}");

    let (dur_ft, ev_ft) = run(
        CardAbstraction {
            flop: Some(6),
            turn: Some(6),
            river: None,
        },
        iterations,
    );
    println!("flop=6, turn=6 buckets: {dur_ft:?}, ev={ev_ft:?}");

    println!(
        "time ratio (none/bucketed): {:.2}x",
        dur_none.as_secs_f64() / dur_ft.as_secs_f64()
    );

    let dead = set_from_cards(&["2c", "7c", "9c"].map(|s| card_from_str(s).unwrap()));
    let oop_n = Range::parse("JJ,TT,99,AJs,KJs,AJo,KJo")
        .unwrap()
        .combos(dead)
        .len();
    let ip_n = Range::parse("88,77,ATs,KTs,ATo,KTo")
        .unwrap()
        .combos(dead)
        .len();
    println!(
        "OOP combos={oop_n} -> 6 buckets ({:.2}x fewer rows per flop/turn node)",
        oop_n as f64 / 6.0
    );
    println!(
        "IP combos={ip_n} -> 6 buckets ({:.2}x fewer rows per flop/turn node)",
        ip_n as f64 / 6.0
    );
}
