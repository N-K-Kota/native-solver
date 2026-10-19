# Contributing

Bug reports and pull requests are welcome. This is maintained in spare time,
so replies may take a while.

## What fits

- Correctness fixes, especially with a failing test.
- Performance work that keeps results unchanged. Show before/after numbers
  from `examples/solver_performance.rs` with the same arguments, and confirm
  the checksums match.
- Documentation fixes.

Open an issue before starting larger features (new game types, multiway
pots, preflop trees) so we can agree on the scope first.

## Before opening a pull request

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --release
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

Every public item needs a doc comment. State units and normalization for
any new number the API returns (see `docs/conventions.md`).

Solver changes should be checked against known values (the Kuhn poker
equilibrium, pot-odds and minimum-defense-frequency cases, the exact
reference evaluator), not only against sums of probabilities. Changes to the
hand evaluator must pass the exhaustive comparison over all 2,598,960 hands:

```sh
cargo test --release evaluate_5_matches_reference_for_every_possible_hand
```
