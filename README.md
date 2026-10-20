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

## Installation

```sh
cargo add native-solver
```

Or add `native-solver = "0.1"` under `[dependencies]` in `Cargo.toml`.

## Example

Solve a river spot, measure how far the result is from equilibrium, and print
the first player's strategy for each combo:

```rust
use native_solver::{
    build_tree, card_from_str, card_to_string, set_from_cards, Action, Chips, GameConfig, Node,
    Player, Range, Trainer, TrainerOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let board = ["Ks", "Th", "7s", "4d", "2s"]
        .iter()
        .map(|card| card_from_str(card))
        .collect::<Result<Vec<_>, _>>()?;
    let config = GameConfig {
        board: board.clone(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(100),
        bet_sizes: vec![0.5, 1.0],
        max_raises: 1,
        suit_isomorphism: false,
    };

    // Two different ranks need an `s` (suited) or `o` (offsuit) marker.
    // Passing the board as dead cards removes combos that share a card with it.
    let dead = set_from_cards(&board);
    let oop = Range::parse("AA,KK,KQs,QJs,JTs:0.5,AQo")?.combos(dead);
    let ip = Range::parse("AKs,AKo,KJs,TT,99,A5s:0.5")?.combos(dead);

    // Both steps validate their input and return an error instead of panicking.
    let tree = build_tree(&config)?;
    let mut trainer = Trainer::new(tree, [oop, ip], TrainerOptions::default())?;
    trainer.train(1000);

    // Average gain, in chips, that a best response could still win.
    println!("exploitability: {:.3} chips", trainer.exploitability().total);

    // The root is OOP's first decision. Strategy rows follow the root's actions;
    // columns follow OOP's combos.
    let Node::Decision { actions, .. } = trainer.game().tree() else {
        unreachable!("a river tree with chips behind starts with a decision");
    };
    let strategy = trainer.average_strategy(0);
    for (i, combo) in trainer.game().combos(Player::Oop).iter().enumerate() {
        let [a, b] = combo.cards;
        print!("{}{}", card_to_string(a), card_to_string(b));
        for (action, row) in actions.iter().zip(&strategy) {
            let label = match action {
                Action::Check => "check".to_string(),
                Action::Bet(amount) => format!("bet {amount}"),
                other => format!("{other:?}"),
            };
            print!("  {label} {:.0}%", row[i] * 100.0);
        }
        println!();
    }
    Ok(())
}
```

The output starts like this:

```text
exploitability: 0.060 chips
JcTc  check 100%  bet 50 0%  bet 100 0%
JdTd  check 100%  bet 50 0%  bet 100 0%
JsTs  check 0%  bet 50 0%  bet 100 100%
...
```

Training more iterations lowers exploitability; the iteration count by itself
says nothing about convergence. Check `exploitability()` instead.

Training runs in parallel on Rayon's global thread pool. The thread count
does not change the results; see [docs/algorithms.md](docs/algorithms.md).

## Errors

`build_tree` returns a `TreeError` for an illegal board, a non-positive pot,
a negative stack, or a bet size that is not a positive finite number.
`Trainer::new` returns a `TrainerError` for inputs it cannot solve, for
example when the tree does not continue from the initial board, or when a
range is empty, contains invalid, repeated, or board-blocked combos, or
cannot be dealt against the other range. Pass the board as dead cards to
`Range::combos` to remove blocked combos.

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

## Benchmarks

`examples/` holds benchmark programs that use only the public API. For
example, `cargo run --release --example solver_performance -- --help` lists
the workloads of the end-to-end solver benchmark.

## Stability

The API may change in any `0.x` minor release. Breaking changes are listed
in [CHANGELOG.md](CHANGELOG.md).

## Minimum supported Rust version

Rust 1.80 for the library. The tests and benchmarks need Rust 1.85 or newer,
because a dev-dependency (Criterion) requires it.

## License

[MIT](LICENSE)
