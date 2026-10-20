# Algorithms

## Combo-vectorized CFR

The tree contains only public information: actions and dealt community
cards. Private information lives in vectors. Each decision node stores one
regret row and one strategy-sum row per action, with one entry per combo of
the acting player, or per bucket under card abstraction. One traversal
updates every combo of both ranges at once.

Terminal values come from sums over the opponent's combos with card removal:

- At a fold, a combo's value is the payoff times the opponent's compatible
  reach: `total - reach through card a - reach through card b + reach of the
  identical pair`.
- At a showdown, both ranges are sorted by hand strength once per board, and
  two sweeps over strictly weaker and strictly stronger opponents give every
  combo's win and loss mass in time linear in the two range sizes. When a
  subtraction would lose a tiny compatible reach to rounding, the code falls
  back to a direct sum.

Sibling subtrees (a node's actions, a chance node's cards) are evaluated in
parallel with `rayon`. Each decision node's tables are behind their own
`Mutex`, which only the threads passing through that node touch.

## Update rules

`SolverAlgorithm` selects the regret and averaging rule. Iterations are
numbered from one.

| Algorithm | Regret after adding the iteration's delta | Average strategy |
| --- | --- | --- |
| `Cfr` | kept as is | uniform weights |
| `CfrPlus` | clipped at `0` after the whole delta is added | uniform weights, no delay |
| `Dcfr` (default) | positive × `t^1.5 / (t^1.5 + 1)`, negative × `1/2` | discounted by `(t / (t + 1))^2`, which gives iteration `s` relative weight `s^2` |

`Dcfr` uses the fixed parameters `(alpha, beta, gamma) = (1.5, 0, 2)` from
Brown and Sandholm, "Solving Imperfect-Information Games via Discounted
Regret Minimization" (AAAI 2019). The sign used for discounting is taken
after adding the iteration's regret.

`CfrPlus` keeps this crate's original rule. It is not the linearly weighted,
alternating CFR+ of the literature, nor the quadratically weighted baseline
from the DCFR paper.

Average-strategy contributions are weighted by the acting player's own reach.

## Update schedule

`UpdateSchedule` selects the player-update order:

- `Simultaneous` (default): both players update against the same
  pre-iteration strategy profile.
- `Alternating`: one iteration updates OOP, then IP against OOP's updated
  strategy. Each player's regrets and average strategy are updated exactly
  once per iteration, and the iteration clock advances once.

The schedule is fixed for a trainer's lifetime, including across separate
`train` calls.

## Reading results without blocking training

`Trainer::policy()` takes a `Policy` snapshot: the normalized average
strategy of every decision node at an iteration boundary. `SolverGame` holds
everything training never changes, and its read methods (`evaluate`,
`exploitability`, `average_strategy`) take a `Policy`. A reader can therefore
use a snapshot from `trainer.game()` while training continues.
`Policy::rows` and `SolverGame::policy_from_rows` persist and restore a
snapshot. The restored policy reads bit-identically.

## Determinism

`Range::combos` returns combos in a fixed order, and the trainer keeps that
order. Results depend on the order of floating-point operations: reordering
combos or swapping the two cards inside a combo can change the last bits.
The thread count does not: each decision node is updated once per pass, by
the one traversal path that reaches it, and parallel child values are
combined in a fixed order.
