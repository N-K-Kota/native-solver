# Changelog

All notable changes are listed here. The API may change in any `0.x` minor
release; such changes are marked **Breaking**.

## Unreleased

Additions only. Postflop results are unchanged.

- Heads-up preflop trees: `PreflopConfig` and `build_preflop_tree` build the
  preflop betting round (blinds, limps, raise sizes as multiples of the bet,
  all-ins, a raise cap) as a `Tree` with an empty board. `Trainer`,
  `exploitability`, and `evaluate` work on it unchanged.
- A `Terminal::Showdown` on an empty board is settled over every five-card
  runout, from a shipped table of exact win and tie counts for all 47,008
  suit-isomorphism classes of two hands. This is exact for all-ins; a call
  with chips behind is valued as checked down. Earlier versions rejected
  such a tree with `TrainerError::Board`.
- `Combo::hand` names a combo's canonical hand, and `average_by_hand` folds
  per-combo rows into the 169-hand view.
- `ReferenceEvaluator` accepts preflop trees and enumerates every runout.
- New error variants: `TreeError::InvalidBlinds`,
  `TreeError::InvalidRaiseSize`, `TrainerError::PreflopAbstraction`.
- The crate grows by about 0.6 MB for the equity table.

## 0.2.0

Performance only. No public API changed, and every strategy, value, and
exploitability is bit-identical to 0.1.0 at any thread count.

- Training, `exploitability`, and `evaluate` walk a tree compiled once per
  game. Small subtrees run sequentially in one scratch buffer instead of
  allocating per node and spawning a Rayon task per child. Training is about
  1.6x to 3.3x faster on one thread in `examples/solver_performance.rs`, and
  small trees no longer slow down when threads are added.
- `evaluate` now runs in parallel on large trees.
- Fold and showdown kernels read flat per-board card arrays and skip the
  cancellation-limit division in the common case.
- `evaluate_7` scores seven cards directly instead of taking the best of 21
  five-card lookups. Scores are unchanged for every hand.
- `Trainer::policy` normalizes each node in one pass and uses Rayon on large
  trees.
- Memory use grows by up to about 10% on trees with very small ranges, where
  per-node bookkeeping dominates.

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
