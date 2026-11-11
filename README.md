# native-solver

A heads-up No-Limit Hold'em solver in Rust, for postflop spots and for the
preflop betting round. It includes:

- a table-driven 5- and 7-card hand evaluator,
- weighted preflop ranges that expand into concrete two-card combos,
- betting-tree construction from the flop, turn, or river with configurable
  bet sizes and raise caps,
- a heads-up preflop betting tree (blinds, limps, raise sizes, all-ins) whose
  called pots are settled over every runout from an exact equity table,
- a combo-vectorized CFR trainer with CFR, CFR+, and Discounted CFR,
  simultaneous or alternating updates, optional suit-isomorphism reduction
  and card abstraction,
- per-node EV and river equity under the average strategy, and an exact
  exploitability (best-response) metric.

## Installation

```sh
cargo add native-solver
```

Or add `native-solver = "0.2"` under `[dependencies]` in `Cargo.toml`.

## Example

Solve a river spot, measure how far the result is from equilibrium, and print
the first player's strategy for each combo:

```rust
use native_solver::{
    build_tree, Action, Card, CardSet, Chips, GameConfig, Node, Player, Range, Trainer,
    TrainerOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let board = ["Ks", "Th", "7s", "4d", "2s"]
        .iter()
        .map(|card| card.parse())
        .collect::<Result<Vec<Card>, _>>()?;
    let config = GameConfig::new(board.clone(), Chips::whole(100), Chips::whole(100))
        .with_bet_sizes([0.5, 1.0])
        .with_max_raises(1);

    // Two different ranks need an `s` (suited) or `o` (offsuit) marker.
    // Passing the board as dead cards removes combos that share a card with it.
    let dead = CardSet::from_cards(&board);
    let oop = "AA,KK,KQs,QJs,JTs:0.5,AQo".parse::<Range>()?.combos(dead);
    let ip = "AKs,AKo,KJs,TT,99,A5s:0.5".parse::<Range>()?.combos(dead);

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
        print!("{a}{b}");
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

## Preflop

`build_preflop_tree` builds the preflop betting round: the small blind (the
button, `Player::Ip`) acts first against the big blind (`Player::Oop`). The
same `Trainer` solves it. This finds the push-or-fold equilibrium with 10 big
blinds and prints how often each hand shoves:

```rust
use native_solver::{
    all_canonical_hands, average_by_hand, build_preflop_tree, CardSet, Chips, Player,
    PreflopConfig, Range, Trainer, TrainerOptions,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let half = Chips::from_f64(0.5).ok_or("0.5 is a valid chip amount")?;
    let config = PreflopConfig::new(half, Chips::whole(1), Chips::whole(10))
        .with_all_in(true)
        .with_max_raises(1)
        .with_limp(false);

    // Every hand for both players; nothing is dead before the flop.
    let all: Range = all_canonical_hands().join(",").parse()?;
    let ranges = [all.combos(CardSet::EMPTY), all.combos(CardSet::EMPTY)];

    let mut trainer = Trainer::new(build_preflop_tree(&config)?, ranges, TrainerOptions::default())?;
    trainer.train(400);
    println!("exploitability: {:.4} big blinds", trainer.exploitability().total);

    // Node 0 is the small blind's decision; row 1 is the shove.
    let strategy = trainer.average_strategy(0);
    for (hand, row) in average_by_hand(trainer.game().combos(Player::Ip), &strategy) {
        println!("{hand}: shove {:.0}%", row[1] * 100.0);
    }
    Ok(())
}
```

No cards are dealt inside a preflop tree. When a bet is called, or the big
blind checks behind a limp, the pot goes to a showdown over all 1,712,304
possible boards, using exact win and tie counts for each pair of hands. That
is the true value of an all-in. For a call with chips behind it assumes the
hand is checked down, so trees with small raises and calls are approximations
of real play; push-or-fold and all-in lines are exact.

## Errors

`build_tree` returns a `TreeError` for an illegal board, a non-positive pot,
a negative stack, or a bet size that is not a positive finite number.
`build_preflop_tree` returns one unless
`0 < small blind < big blind <= effective stack` and every raise size is a
finite multiple above 1.
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

- Heads-up only: no multiway pots or side pots.
- A preflop tree does not continue onto the flop. A called pot is settled as
  if checked down, which is exact only when a player is all in. Card
  abstraction does not apply to preflop trees.
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

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
