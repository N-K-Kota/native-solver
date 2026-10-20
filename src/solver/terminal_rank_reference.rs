// Frozen pre-cache showdown sweep. Test-only numerical regression oracle.
use super::*;

impl TerminalIndex {
    #[allow(clippy::too_many_arguments)] // Explicit immutable inputs keep the hot kernel allocation-free.
    pub(in crate::solver) fn showdown_before_rank_cache(
        &self,
        combos: &[Vec<Combo>; 2],
        player: usize,
        reach: &[f64],
        scores: &[Vec<u32>; 2],
        order: &BoardOrder,
        board: u64,
        invested: &[f64; 2],
    ) -> Vec<f64> {
        let opp = 1 - player;
        let small = invested[0].abs().min(invested[1].abs());
        let large = invested[0].abs().max(invested[1].abs());
        if invested[0] != invested[1] && small < large * 1e-8 {
            // Constructed poker showdowns always have matched investments.
            // Public custom trees may not: avoid cancellation between the
            // affine tie offset and half-pot when their scales differ greatly.
            return combos[player]
                .iter()
                .enumerate()
                .map(|(i, &hero)| {
                    if !live(hero, board) {
                        return 0.0;
                    }
                    order.order[opp]
                        .iter()
                        .copied()
                        .filter(|&j| disjoint(hero, combos[opp][j]))
                        .map(|j| {
                            let net = match scores[player][i].cmp(&scores[opp][j]) {
                                std::cmp::Ordering::Greater => invested[opp],
                                std::cmp::Ordering::Less => -invested[player],
                                std::cmp::Ordering::Equal => {
                                    (invested[opp] - invested[player]) / 2.0
                                }
                            };
                            reach[j] * net
                        })
                        .sum()
                })
                .collect();
        }
        let half_pot = (invested[0] + invested[1]) / 2.0;
        let tie_value = (invested[opp] - invested[player]) / 2.0;
        let mut values = if tie_value == 0.0 {
            vec![0.0; combos[player].len()]
        } else {
            self.compatible_mass(combos, player, reach, board)
                .into_iter()
                .map(|mass| mass * tie_value)
                .collect()
        };
        let mut weaker = ReachMass::new();
        let mut j = 0;
        for &i in &order.order[player] {
            while j < order.order[opp].len() && scores[opp][order.order[opp][j]] < scores[player][i]
            {
                let k = order.order[opp][j];
                weaker.add(combos[opp][k], reach[k]);
                j += 1;
            }
            let mut mass = weaker.excluding(combos[player][i]);
            if weaker.support_excluding(combos[player][i]) > 0 && weaker.cancellation_prone(mass) {
                mass = order.order[opp][..j]
                    .iter()
                    .copied()
                    .filter(|&k| disjoint(combos[player][i], combos[opp][k]))
                    .map(|k| reach[k])
                    .sum();
            }
            values[i] += half_pot * mass;
        }
        let mut stronger = ReachMass::new();
        let mut j = order.order[opp].len();
        for &i in order.order[player].iter().rev() {
            while j > 0 && scores[opp][order.order[opp][j - 1]] > scores[player][i] {
                j -= 1;
                let k = order.order[opp][j];
                stronger.add(combos[opp][k], reach[k]);
            }
            let mut mass = stronger.excluding(combos[player][i]);
            if stronger.support_excluding(combos[player][i]) > 0
                && stronger.cancellation_prone(mass)
            {
                mass = order.order[opp][j..]
                    .iter()
                    .copied()
                    .filter(|&k| disjoint(combos[player][i], combos[opp][k]))
                    .map(|k| reach[k])
                    .sum();
            }
            values[i] -= half_pot * mass;
        }
        values
    }
}

#[test]
fn cached_rank_bounds_preserve_ties_blockers_and_cancellation_fallback() {
    // Deliberately unsorted and repeated/reversed pairs; scores are tied and
    // identical for identical pairs. A second range can be empty.
    let full: Vec<Combo> = (0..12)
        .flat_map(|a| (a + 1..13).map(move |b| Combo::new(Card(a), Card(b), 1.0)))
        .collect();
    let mut sparse: Vec<_> = full.iter().step_by(3).rev().copied().collect();
    sparse.push(Combo::new(sparse[2].cards[1], sparse[2].cards[0], 1.0));
    for combos in [
        [full.clone(), sparse.clone()],
        [sparse.clone(), full.clone()],
        [full.clone(), vec![]],
    ] {
        for board in [0, (1 << 0) | (1 << 4), (1 << 13) - 1] {
            for all_tied in [false, true] {
                let scores: [Vec<u32>; 2] = std::array::from_fn(|p| {
                    combos[p]
                        .iter()
                        .map(|&Combo { cards: [a, b], .. }| {
                            if all_tied {
                                7
                            } else {
                                ((a.0 as u32 + b.0 as u32) % 5) + 1
                            }
                        })
                        .collect()
                });
                let order = BoardOrder::new(&combos, &scores, board);
                let index = TerminalIndex::new(&combos);
                for p in 0..2 {
                    // Independent counts establish strict boundaries even if
                    // every reach is zero and numerical equality is vacuous.
                    for (&i, &(lo, hi)) in order.order[p].iter().zip(&order.rank_bounds[p]) {
                        assert_eq!(
                            lo,
                            combos[1 - p]
                                .iter()
                                .enumerate()
                                .filter(
                                    |(j, c)| live(**c, board) && scores[1 - p][*j] < scores[p][i]
                                )
                                .count()
                        );
                        assert_eq!(
                            hi,
                            combos[1 - p]
                                .iter()
                                .enumerate()
                                .filter(
                                    |(j, c)| live(**c, board) && scores[1 - p][*j] <= scores[p][i]
                                )
                                .count()
                        );
                    }
                    for phase in 0..4 {
                        let reach: Vec<_> = (0..combos[1 - p].len())
                            .map(|i| match phase {
                                0 => 0.0,
                                1 => {
                                    if i == 0 {
                                        1.0
                                    } else {
                                        1e-24
                                    }
                                }
                                2 => {
                                    if i % 3 == 0 {
                                        0.0
                                    } else {
                                        (i + 1) as f64 / 97.0
                                    }
                                }
                                _ => {
                                    if i == 2 {
                                        1e24
                                    } else {
                                        0.25
                                    }
                                }
                            })
                            .collect();
                        for invested in [[50.0, 50.0], [37.0, 121.0], [1e-12, 1e12]] {
                            let actual = index
                                .showdown(&combos, p, &reach, &scores, &order, board, &invested);
                            let before = index.showdown_before_rank_cache(
                                &combos, p, &reach, &scores, &order, board, &invested,
                            );
                            assert_eq!(
                                actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                before.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
    }
}
