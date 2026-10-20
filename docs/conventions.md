# Conventions

This page states the units and normalizations of every number the crate
returns. Read it before you show a solver value to a person or feed it into
another calculation.

## Cards and combos

- A `Card` is one of the 52 cards, with index `rank * 4 + suit` in `0..52`
  (`Card::index`). Ranks run `23456789TJQKA` (`0` is a deuce, `12` an
  ace). Suits run `cdhs`. Parse one with `"As".parse::<Card>()`; it
  displays the same way.
- A `CardSet` is a set of cards, stored as a bitmask with bit
  `card.index()` set for each card.
- `Range::parse` (or `str::parse`) takes the 169 canonical hands (`AA`,
  `AKs`, `AKo`, …), each with an optional `:weight` in `0.0..=1.0`. Two
  different ranks need an `s` or `o` marker. A range displays in the same
  notation.
- `Range::combos(dead)` expands a range into `Combo { cards, weight }`
  values and drops every combo that shares a card with `dead`. The order is
  deterministic: by lower card, then higher card.
- A trainer keeps the combo order it is given. Every per-combo result
  (strategies, EVs, equities) is indexed in that order.

## Chips and payoffs

- Bet sizes are fractions of the pot. Pots, stacks, and bets are stored as
  exact integers of `1 / CHIP_SCALE` (`0.001`) input chips, so all-in
  detection and bet deduplication are exact. Each bet amount is rounded
  half-to-even to that unit, clipped to the remaining stack, and
  deduplicated.
- No all-in size is added automatically. Include a large enough fraction if
  you want one.
- Terminal payoffs count half the starting pot as each player's own
  investment. `invested[p]` is that half plus everything `p` put in since. At
  a fold the winner gains the loser's `invested` and the loser loses their
  own. At showdown the better hand gains the opponent's `invested`, and a tie
  pays `(invested[opp] - invested[me]) / 2`.
- Every value below is in input chips, the same units as `pot`. Divide by the
  pot yourself for a percentage. Nothing is reported in big blinds or mbb.

## Counterfactual values versus EVs

The solver works with raw counterfactual values (CFVs). A raw CFV for one of
a player's combos is a sum over the opponent's combos, weighted by the
opponent's unnormalized reach. It is not the expected value of holding that
combo.

| API | What it returns |
| --- | --- |
| `Trainer::train` | Root `[OOP, IP]` raw range-weighted values under the last iteration's current strategy. A progress signal, not an EV. |
| `evaluate().root_values` | Root `[OOP, IP]` raw range-weighted values under the average strategy. |
| `NodeEvaluation::ev` | Per combo: the raw CFV divided by the opponent's compatible reach, which is the node EV in chips. `0.0` when no opponent combo is compatible. |
| `NodeEvaluation::equity` | Per combo, complete boards only: showdown win probability against the opponent's compatible, reach-weighted range. Ties count half; `0.5` when nothing is compatible. |

"Compatible" excludes opponent combos that share a card with the board or
with the hero's combo.

`NodeEvaluation::ev` is the value of the whole decision node under the
average strategy. It is not the value of any single action. To get an
action's value, read the child node.

## Exploitability

`Exploitability { oop, ip, total }`:

- `oop` is OOP's best-response value against IP's fixed average strategy, and
  `ip` the reverse.
- `total = (oop + ip) / 2`. It is `0` at an exact Nash equilibrium and grows
  with how much either player could still gain by deviating.
- Values are expectations over compatible initial private deals, normalized
  by their joint range weight, in chips. Every field is `NaN` when that joint
  weight is zero or not finite. Reject such a result.

Exploitability is measured in the game the tree models: the given ranges,
bet sizes, and raise caps. It says nothing about bet sizes the tree leaves
out. With card abstraction the best response still picks per combo, so a
bucketed strategy is measured in the full private-card game.

A fixed iteration count does not establish convergence. Compare
`exploitability().total` with a target instead.

## Chance nodes

A chance node deals the next community card. Each undealt card is equally
likely given a legal private deal. Both players' hole cards are excluded, so
every compatible deal has `N - 4` next cards among the `N` cards not on the
board. Raw CFVs at a chance node divide by `N - 4`.

## Suit isomorphism

With `GameConfig::suit_isomorphism`, a chance node groups cards of the same
rank whose suits are both absent from the board. The tree recurses only into
a representative card of each class, and the trainer derives the other
members' values by relabeling suits.

This is exact only when the game is suit-symmetric:

- Each player's combos must be closed under every suit swap that fixes the
  board, and swapped combos must have equal weights. Ranges from
  `Range::combos` satisfy this. `Trainer::new` returns
  `TrainerError::AsymmetricRange` for a reduced tree whose ranges do not.
- Floating-point training breaks symmetry by tiny amounts, and regret
  matching can turn tiny differences into large strategy differences. The
  trainer therefore averages current and average strategies over each
  suit-equivalent group of combos. Reduced and unreduced trees reach the same
  equilibrium but not bit-identical numbers.

For arbitrary asymmetric combo weights, build the tree without suit
isomorphism.

## Card abstraction

`CardAbstraction` buckets a street's combos by hand strength. `Some(n)` makes
at most `n` percentile buckets, and tied strengths always share a bucket.
River buckets use the exact score. Flop and turn buckets use the mean river
score over the remaining runouts. Combos in a bucket share one regret and
strategy row, and the bucket's regret is the own-reach-weighted average of its
members' regrets. Strategies are expanded back to one row per combo when you
read them.

## Reference evaluator

`solver::quality::ReferenceEvaluator` computes exact profile values,
best-response values, NashConv, and exploitability by enumerating private
deals, without sharing any code path with the trainer. It is for small trees,
tests, and audits. It rejects trees with suit-isomorphism classes; build an
unreduced tree for it.
