//! Ad-hoc timing/memory comparison for river card abstraction (bucketing)
//! — not a criterion bench, just a measurement script run manually during
//! development. Run with:
//!   cargo run --release --example bucket_speedup

use native_solver::card::{Card, CardSet};
use native_solver::range::Range;
use native_solver::solver::Chips;
use native_solver::solver::{build_tree, CardAbstraction, GameConfig, Trainer};
use native_solver::solver::{SolverAlgorithm, TrainerOptions};
use std::time::Instant;

fn run(river_buckets: Option<usize>, iterations: u32) -> (std::time::Duration, [f64; 2]) {
    let abstraction = CardAbstraction {
        river: river_buckets,
        ..Default::default()
    };
    // A single street directly at the river with wide-ish ranges, so most
    // of the tree's decision nodes are river nodes eligible for bucketing.
    let board: Vec<_> = ["2h", "7c", "9d", "Jc", "4s"]
        .iter()
        .map(|s| s.parse::<Card>().unwrap())
        .collect();
    let dead = CardSet::from_cards(&board);
    let oop_combos = Range::parse("JJ,TT,99,88,77,AJs,KJs,QJs,JTs,AJo,KJo,QJo,JTo")
        .unwrap()
        .combos(dead);
    let ip_combos = Range::parse("99,88,77,66,55,ATs,KTs,QTs,JTs,ATo,KTo,QTo,JTo")
        .unwrap()
        .combos(dead);

    let cfg = GameConfig::new(board.clone(), Chips::whole(100), Chips::whole(200))
        .with_bet_sizes([0.5, 1.0])
        .with_max_raises(1)
        .with_suit_isomorphism(true);
    let tree = build_tree(&cfg).unwrap();
    let mut trainer = Trainer::new(
        tree,
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
    let iterations = 300;
    println!("iterations={iterations}");
    // ev is train()'s *last-iteration current-strategy* value (see its doc
    // comment) -- informal and can swing a lot before convergence; only
    // useful here as a rough zero-sum sanity check (ev0 ~= -ev1), not as a
    // measure of solve quality.

    let (dur_none, ev_none) = run(None, iterations);
    println!("river_buckets=None:    {dur_none:?}, ev={ev_none:?}");

    let (dur_8, ev_8) = run(Some(8), iterations);
    println!("river_buckets=Some(8): {dur_8:?}, ev={ev_8:?}");

    println!(
        "time ratio (none/bucketed): {:.2}x",
        dur_none.as_secs_f64() / dur_8.as_secs_f64()
    );

    // Memory: regret_sum + strategy_sum table size at a river decision
    // node scales with (num_actions * num_combos) unbucketed vs
    // (num_actions * num_buckets) bucketed -- for a player's combo count
    // n and bucket count b, that's an n/b reduction per river node.
    let dead =
        CardSet::from_cards(&["2h", "7c", "9d", "Jc", "4s"].map(|s| s.parse::<Card>().unwrap()));
    let oop_n = Range::parse("JJ,TT,99,88,77,AJs,KJs,QJs,JTs,AJo,KJo,QJo,JTo")
        .unwrap()
        .combos(dead)
        .len();
    let ip_n = Range::parse("99,88,77,66,55,ATs,KTs,QTs,JTs,ATo,KTo,QTo,JTo")
        .unwrap()
        .combos(dead)
        .len();
    println!(
        "OOP combos={oop_n} -> 8 buckets ({:.2}x fewer rows per river node)",
        oop_n as f64 / 8.0
    );
    println!(
        "IP combos={ip_n} -> 8 buckets ({:.2}x fewer rows per river node)",
        ip_n as f64 / 8.0
    );
}
