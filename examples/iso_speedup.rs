//! Ad-hoc timing comparison for suit isomorphism (not a criterion bench,
//! just a measurement script run manually during development). Run with:
//!   cargo run --release --example iso_speedup

use native_solver::card::{card_from_str, set_from_cards};
use native_solver::range::Range;
use native_solver::solver::Chips;
use native_solver::solver::{build_tree, GameConfig, Trainer};
use native_solver::solver::{SolverAlgorithm, TrainerOptions};
use std::time::Instant;

fn run(suit_isomorphism: bool, iterations: u32) -> (usize, std::time::Duration, [f64; 2]) {
    // Monotone turn (all clubs) -> 3 free suits, the best case for
    // isomorphism reduction, at a single chance node (river only, so the
    // unreduced tree stays small enough to actually finish).
    let board: Vec<_> = ["2c", "7c", "9c", "Kc"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let dead = set_from_cards(&board);
    let oop_combos = Range::parse("QQ,JJ,TT,99,AQo,KQo").unwrap().combos(dead);
    let ip_combos = Range::parse("88,77,66,KTo,QTo").unwrap().combos(dead);

    let cfg = GameConfig {
        board: board.clone(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(200),
        bet_sizes: vec![0.5, 1.0],
        max_raises: 1,
        suit_isomorphism,
    };
    let tree = build_tree(&cfg).unwrap();
    let num_decision_nodes = tree.num_decision_nodes;
    let mut trainer = Trainer::new(
        tree,
        [oop_combos, ip_combos],
        TrainerOptions {
            algorithm: SolverAlgorithm::CfrPlus,
            ..TrainerOptions::default()
        },
    )
    .unwrap();
    let start = Instant::now();
    let ev = trainer.train(iterations);
    (num_decision_nodes, start.elapsed(), ev)
}

fn main() {
    let iterations = 500;
    println!("iterations={iterations}");

    let (nodes_off, dur_off, ev_off) = run(false, iterations);
    println!("suit_isomorphism=false: {nodes_off} decision nodes, {dur_off:?}, ev={ev_off:?}");

    let (nodes_on, dur_on, ev_on) = run(true, iterations);
    println!("suit_isomorphism=true:  {nodes_on} decision nodes, {dur_on:?}, ev={ev_on:?}");

    println!(
        "node reduction: {:.2}x, time speedup: {:.2}x",
        nodes_off as f64 / nodes_on as f64,
        dur_off.as_secs_f64() / dur_on.as_secs_f64()
    );
}
