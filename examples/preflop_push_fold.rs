//! Heads-up push-or-fold equilibrium for a given stack, in big blinds:
//!   cargo run --release --example preflop_push_fold -- 10
//!
//! Prints how often each of the 169 hands shoves from the small blind and
//! calls from the big blind, as two 13x13 grids (suited hands above the
//! diagonal, offsuit below).

use native_solver::card::RANKS;
use native_solver::{
    all_canonical_hands, average_by_hand, build_preflop_tree, CardSet, Chips, Combo, Player,
    PreflopConfig, Range, Trainer, TrainerOptions,
};
use std::collections::HashMap;

fn grid(title: &str, combos: &[Combo], strategy: &[Vec<f64>], action: usize) {
    let frequency: HashMap<String, f64> = average_by_hand(combos, strategy)
        .into_iter()
        .map(|(hand, row)| (hand, row[action]))
        .collect();
    let overall = strategy[action].iter().sum::<f64>() / combos.len() as f64;
    println!("\n{title}: {:.1}% of hands", overall * 100.0);
    let ranks: Vec<char> = RANKS.chars().rev().collect();
    for (i, &row) in ranks.iter().enumerate() {
        let mut line = String::new();
        for (j, &column) in ranks.iter().enumerate() {
            let hand = match i.cmp(&j) {
                std::cmp::Ordering::Equal => format!("{row}{row}"),
                std::cmp::Ordering::Less => format!("{row}{column}s"),
                std::cmp::Ordering::Greater => format!("{column}{row}o"),
            };
            line += &format!("{hand:>4}:{:>3.0}", frequency[&hand] * 100.0);
        }
        println!("{line}");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stack: f64 = match std::env::args().nth(1) {
        Some(text) => text.parse()?,
        None => 10.0,
    };
    let chips = |amount: f64| Chips::from_f64(amount).ok_or("not a valid chip amount");
    let config = PreflopConfig::new(chips(0.5)?, chips(1.0)?, chips(stack)?)
        .with_all_in(true)
        .with_max_raises(1)
        .with_limp(false);
    let all: Range = all_canonical_hands().join(",").parse()?;
    let ranges = [all.combos(CardSet::EMPTY), all.combos(CardSet::EMPTY)];
    let mut trainer = Trainer::new(
        build_preflop_tree(&config)?,
        ranges,
        TrainerOptions::default(),
    )?;

    let start = std::time::Instant::now();
    let mut iterations = 0;
    let exploitability = loop {
        trainer.train(100);
        iterations += 100;
        let exploitability = trainer.exploitability().total;
        if exploitability < 0.001 || iterations >= 2000 {
            break exploitability;
        }
    };
    println!(
        "stack {stack} bb: {iterations} iterations in {:.2?}, exploitability {exploitability:.5} bb",
        start.elapsed()
    );

    let game = trainer.game();
    grid(
        "small blind shoves",
        game.combos(Player::Ip),
        &trainer.average_strategy(0),
        1,
    );
    grid(
        "big blind calls",
        game.combos(Player::Oop),
        &trainer.average_strategy(1),
        1,
    );
    Ok(())
}
