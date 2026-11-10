# Changelog

All notable changes are listed here. The API may change in any `0.x` minor
release; such changes are marked **Breaking**.

## Unreleased

Performance only: every strategy, value, and exploitability is bit-identical
to 0.1.0 at any thread count.

- Training, `exploitability`, and `evaluate` walk a tree compiled once per
  game. Small subtrees run sequentially in one scratch buffer instead of
  allocating per node and spawning a Rayon task per child.
- Fold and showdown kernels read flat per-board card arrays and skip the
  cancellation-limit division in the common case.
- `evaluate_7` scores seven cards directly instead of taking the best of 21
  five-card lookups. Scores are unchanged for every hand.
- `Trainer::policy` normalizes each node in one pass and uses Rayon on large
  trees.

## 0.1.0

First public release, imported from the solver core of a private project.
Licensed under MIT OR Apache-2.0.

- Table-driven 5- and 7-card hand evaluator.
- Weighted preflop ranges that expand into deterministic `Combo` lists.
- Betting trees from the flop, turn, or river, with exact integer chip
  amounts and an optional decision-node budget.
- Combo-vectorized CFR, CFR+, and DCFR (the default) with simultaneous or
  alternating updates, suit-isomorphism reduction, and per-street card
  abstraction.
- Node EVs, river equity, and exploitability under the average strategy;
  `Policy` snapshots that can be read while training continues.
- An independent exact reference evaluator for small trees.
- Input validation: `build_tree` and `Trainer::new` return `TreeError` and
  `TrainerError` for illegal boards, chip amounts, bet sizes, trees, and
  ranges instead of panicking or returning NaN.
