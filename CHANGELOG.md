# Changelog

All notable changes are listed here. The API may change in any `0.x` minor
release; such changes are marked **Breaking**.

## 0.1.0

First public release, imported from the solver core of a private project.

- Table-driven 5- and 7-card hand evaluator.
- Weighted preflop ranges that expand into deterministic `Combo` lists.
- Betting trees from the flop, turn, or river, with exact integer chip
  amounts and an optional decision-node budget.
- Combo-vectorized CFR, CFR+, and DCFR with simultaneous or alternating
  updates, suit-isomorphism reduction, and per-street card abstraction.
- Node EVs, river equity, and exploitability under the average strategy;
  `Policy` snapshots that can be read while training continues.
- An independent exact reference evaluator for small trees.
