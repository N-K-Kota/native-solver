# native-solver

A heads-up postflop No-Limit Hold'em solver in Rust. It includes:

- a table-driven 5- and 7-card hand evaluator,
- weighted preflop ranges that expand into concrete two-card combos,
- betting-tree construction from the flop, turn, or river with configurable
  bet sizes and raise caps,
- a combo-vectorized CFR trainer with CFR, CFR+, and Discounted CFR,
  simultaneous or alternating updates, optional suit-isomorphism reduction
  and card abstraction,
- per-node EV and river equity under the average strategy, and an exact
  exploitability (best-response) metric.

## Example

Solve a river spot, measure how far the result is from equilibrium, and read
the first player's strategy:

```rust
use native_solver::{
    build_tree, card_from_str, set_from_cards, Chips, GameConfig, Range, Trainer, TrainerOptions,
};

let board: Vec<_> = ["Ks", "Th", "7s", "4d", "2s"]
    .iter()
    .map(|card| card_from_str(card).unwrap())
    .collect();
let config = GameConfig {
    board: board.clone(),
    pot: Chips::whole(100),
    effective_stack: Chips::whole(100),
    bet_sizes: vec![0.5, 1.0],
    max_raises: 1,
    suit_isomorphism: false,
};

// Combos that share a card with the board are removed.
let dead = set_from_cards(&board);
let oop = Range::parse("AA,KK,KQs,QJs,JTs:0.5,AQo").unwrap().combos(dead);
let ip = Range::parse("AKs,AKo,KJs,TT,99,A5s:0.5").unwrap().combos(dead);

let mut trainer = Trainer::new(build_tree(&config), board, [oop, ip], TrainerOptions::default());
trainer.train(1000);

// Average gain, in chips, that a best response could still win.
let exploitability = trainer.exploitability().total;
assert!(exploitability < 1.0, "{exploitability}");

// The root is OOP's first decision. Rows are actions, columns are OOP's combos.
let strategy = trainer.average_strategy(0);
for row in &strategy {
    assert_eq!(row.len(), trainer.game().combos(0).len());
}
for combo in 0..strategy[0].len() {
    let total: f64 = strategy.iter().map(|row| row[combo]).sum();
    assert!((total - 1.0).abs() < 1e-9);
}
```

Training more iterations lowers exploitability; the iteration count by itself
says nothing about convergence. Check `exploitability()` instead.

## Conventions

Raw counterfactual values are not EVs, and exploitability is reported in
chips. See [docs/conventions.md](docs/conventions.md) for units, value
normalization, and the requirements of suit-isomorphism reduction, and
[docs/algorithms.md](docs/algorithms.md) for the update rules.

## Limitations

- Heads-up and postflop only: no preflop tree, multiway pots, or side pots.
- No rake.
- Bet sizes are fractions of the pot. An all-in is not added automatically,
  so include a large enough fraction if you want one.
- Equity is reported only on a complete five-card board.
- Memory grows with tree size × combo count. Use `build_tree_bounded` for
  untrusted input.

## Stability

The API may change in any `0.x` minor release. Breaking changes are listed
in [CHANGELOG.md](CHANGELOG.md).

## Minimum supported Rust version

Rust 1.80 for the library. The test suite needs a newer stable compiler.

## License

[MIT](LICENSE)
