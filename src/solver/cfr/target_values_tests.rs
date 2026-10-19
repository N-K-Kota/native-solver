//! Frozen two-player traversal from before the target-only optimization.
use super::*;

impl Trainer {
    fn legacy_cfr(
        &self,
        node: &Node,
        reach: &[Vec<f64>; 2],
        board: &[Card],
        board_eval: &[Vec<u32>; 2],
        update: IterationUpdate,
        updating: Option<usize>,
    ) -> [Vec<f64>; 2] {
        match node {
            Node::Terminal(t) => self.terminal_value(t, reach, board, board_eval),
            Node::Chance { children } => {
                self.legacy_chance_value(children, reach, board, update, updating)
            }
            Node::Decision {
                id,
                player,
                actions,
                children,
            } => {
                let p = player.index();
                let num_own_combos = self.combos[p].len();
                let num_other_combos = self.combos[1 - p].len();
                let (num_buckets, buckets) = self.combo_buckets(p, board);
                let strategy = self.current_strategy(*id, actions.len(), num_buckets);

                let child_cfvs: Vec<[Vec<f64>; 2]> = children
                    .par_iter()
                    .enumerate()
                    .map(|(a, child)| {
                        let mut child_reach = reach.clone();
                        for i in 0..num_own_combos {
                            child_reach[p][i] *= strategy[a][buckets[i]];
                        }
                        self.legacy_cfr(child, &child_reach, board, board_eval, update, updating)
                    })
                    .collect();

                let mut own_cfv = vec![0.0; num_own_combos];
                let mut other_cfv = vec![0.0; num_other_combos];
                for (a, cfv) in child_cfvs.iter().enumerate() {
                    let own = &cfv[p];
                    let other = &cfv[1 - p];
                    for i in 0..num_own_combos {
                        own_cfv[i] += strategy[a][buckets[i]] * own[i];
                    }
                    for (j, v) in other_cfv.iter_mut().enumerate() {
                        *v += other[j];
                    }
                }

                // A non-updating player's strategy affects both returned CFVs,
                // but its regrets and strategy mass must remain untouched.
                if updating.is_some_and(|target| target != p) {
                    return if p == 0 {
                        [own_cfv, other_cfv]
                    } else {
                        [other_cfv, own_cfv]
                    };
                }

                // `effective_reach` treats a combo whose hole card physically
                // conflicts with `board` (impossible on this specific
                // runout — see `eval_combos_if_complete`'s doc comment) as
                // reach 0 *only* for the bucket-merge weighting below, so a
                // phantom combo sharing a bucket with a real one can never
                // skew the real combo's learned regret. When every combo is
                // its own bucket (unabstracted, or `river_buckets` capped to
                // a bijection) each bucket's total is either that one
                // combo's own reach or — if masked or genuinely zero — falls
                // through to the equal-weight branch below with a single
                // member, which reduces exactly to the plain per-combo
                // regret update this replaces (`own[i] - own_cfv[i]`, no
                // reach weighting at all), so this is an exact
                // generalization, not a behavior change, for that case.
                let board_set = set_from_cards(board);
                let effective_reach: Vec<f64> = (0..num_own_combos)
                    .map(|i| {
                        let [h1, h2] = self.combos[p][i].cards;
                        if set_contains(board_set, h1) || set_contains(board_set, h2) {
                            0.0
                        } else {
                            reach[p][i]
                        }
                    })
                    .collect();
                let mut bucket_reach_total = vec![0.0; num_buckets];
                let mut bucket_size = vec![0usize; num_buckets];
                for i in 0..num_own_combos {
                    bucket_reach_total[buckets[i]] += effective_reach[i];
                    bucket_size[buckets[i]] += 1;
                }

                {
                    let mut data = self.node_data[*id].lock().unwrap();
                    for (a, cfv) in child_cfvs.iter().enumerate() {
                        let own = &cfv[p];
                        let mut regret_delta = vec![0.0; num_buckets];
                        for i in 0..num_own_combos {
                            let b = buckets[i];
                            let delta = own[i] - own_cfv[i];
                            let weight = if bucket_reach_total[b] > 0.0 {
                                effective_reach[i] / bucket_reach_total[b]
                            } else {
                                1.0 / bucket_size[b] as f64
                            };
                            regret_delta[b] += weight * delta;
                            data.strategy_sum[a][b] += reach[p][i] * strategy[a][b];
                        }
                        for (b, delta) in regret_delta.iter().enumerate() {
                            data.regret_sum[a][b] = update.regret(data.regret_sum[a][b], *delta);
                            // Once per bucket, after every combo's contribution:
                            // never apply DCFR discounting inside the combo loop.
                            data.strategy_sum[a][b] = update.strategy_sum(data.strategy_sum[a][b]);
                        }
                    }
                }

                if p == 0 {
                    [own_cfv, other_cfv]
                } else {
                    [other_cfv, own_cfv]
                }
            }
        }
    }

    /// Average raw `cfv` over every legal next card of each compatible
    /// private deal. Hero-card conflicts are excluded here; opponent-card
    /// conflicts are excluded at terminals. Together they leave N - 4
    /// possible cards, even though the outer vector is hero-combo indexed.
    ///
    /// `children` is grouped into suit-isomorphism classes (see
    /// `tree::deal_classes`): only the class's representative
    /// (`members[0]`) is actually recursed into. Every other member's
    /// counterfactual values are derived from the representative's by
    /// `relabel_cfv` — a swap of the two suits involved is a symmetry of
    /// the game (it doesn't change tree structure or `evaluate_7`
    /// outcomes), so member `card`'s cfv for combo `j` equals the
    /// representative's cfv for the combo obtained by applying that same
    /// suit swap to combo `j`'s two cards. With isomorphism disabled every
    /// class is a singleton and this reduces to the original per-card loop.
    fn legacy_chance_value(
        &self,
        children: &[(Vec<Card>, Node)],
        reach: &[Vec<f64>; 2],
        board: &[Card],
        update: IterationUpdate,
        updating: Option<usize>,
    ) -> [Vec<f64>; 2] {
        let n: usize = children.iter().map(|(members, _)| members.len()).sum();
        let divisor = (n as f64) - 4.0;

        let class_results: Vec<(&Vec<Card>, [Vec<f64>; 2])> = children
            .par_iter()
            .map(|(members, child)| {
                let rep = members[0];
                let mut child_board = board.to_vec();
                child_board.push(rep);
                let empty_eval = [Vec::new(), Vec::new()];
                let child_board_eval = self
                    .board_eval_cache
                    .get(&child_board)
                    .unwrap_or(&empty_eval);
                let child_cfv = self.legacy_cfr(
                    child,
                    reach,
                    &child_board,
                    child_board_eval,
                    update,
                    updating,
                );
                (members, child_cfv)
            })
            .collect();

        let mut cfv0 = vec![0.0; self.combos[0].len()];
        let mut cfv1 = vec![0.0; self.combos[1].len()];

        for (members, rep_cfv) in &class_results {
            let rep = members[0];
            for &card in members.iter() {
                let (cfv0_card, cfv1_card) = if card == rep {
                    (rep_cfv[0].clone(), rep_cfv[1].clone())
                } else {
                    (
                        self.relabel_cfv(0, &rep_cfv[0], suit(rep), suit(card)),
                        self.relabel_cfv(1, &rep_cfv[1], suit(rep), suit(card)),
                    )
                };
                for (
                    i,
                    &Combo {
                        cards: [h1, h2], ..
                    },
                ) in self.combos[0].iter().enumerate()
                {
                    if h1 != card && h2 != card {
                        cfv0[i] += cfv0_card[i];
                    }
                }
                for (
                    j,
                    &Combo {
                        cards: [h1, h2], ..
                    },
                ) in self.combos[1].iter().enumerate()
                {
                    if h1 != card && h2 != card {
                        cfv1[j] += cfv1_card[j];
                    }
                }
            }
        }

        for v in &mut cfv0 {
            *v /= divisor;
        }
        for v in &mut cfv1 {
            *v /= divisor;
        }
        [cfv0, cfv1]
    }
}

impl Trainer {
    fn legacy_train(&mut self, iterations: u32) -> [f64; 2] {
        let mut last = [0.0; 2];
        let game = std::sync::Arc::clone(&self.game);
        let board = game.initial_board.clone();
        let empty_eval = [Vec::new(), Vec::new()];
        let board_eval = game.board_eval_cache.get(&board).unwrap_or(&empty_eval);
        for _ in 0..iterations {
            let iteration = self.iterations_completed.checked_add(1).unwrap();
            let update = self.algorithm.iteration(iteration);
            let reach: [Vec<f64>; 2] =
                std::array::from_fn(|p| self.combos[p].iter().map(|c| c.weight).collect());
            let updating = match self.update_schedule {
                UpdateSchedule::Simultaneous => None,
                UpdateSchedule::Alternating => {
                    self.legacy_cfr(&self.tree.root, &reach, &board, board_eval, update, Some(0));
                    Some(1)
                }
            };
            let cfv = self.legacy_cfr(
                &self.tree.root,
                &reach,
                &board,
                board_eval,
                update,
                updating,
            );
            last = std::array::from_fn(|p| {
                cfv[p]
                    .iter()
                    .zip(&self.combos[p])
                    .map(|(v, c)| v * c.weight)
                    .sum()
            });
            self.iterations_completed = iteration;
        }
        last
    }
}

fn assert_bits(a: &[Vec<f64>], b: &[Vec<f64>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.len(), b.len());
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn target_values_match_frozen_traversal_bit_for_bit() {
    use crate::solver::Chips;
    use crate::solver::{build_tree, GameConfig};
    use crate::{card_from_str, range::Range};
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        for board_text in ["Ks7d2c", "Ks7s2s", "2c7c9dJd", "2c7c9dJd3h"] {
            let board: Vec<_> = board_text.as_bytes().as_chunks::<2>().0.iter().map(|c|card_from_str(std::str::from_utf8(c).unwrap()).unwrap()).collect();
            for iso in [false,true] { for bucketed in [false,true] {
                let cfg=GameConfig{board:board.clone(),pot:Chips::whole(100),effective_stack:Chips::whole(100),bet_sizes:vec![1.0],max_raises:0,suit_isomorphism:iso};
                let mut combos=["AA:0.5,KK,QQ", "KK:0.25,JJ,TT"].map(|s|Range::parse(s).unwrap().combos(set_from_cards(&board)));
                // Preserve suit symmetry while including zero own reach and
                // blocked/overlapping private deals in bucket and chance paths.
                for c in &mut combos[0] { if rank(c.cards[0])==10 && rank(c.cards[1])==10 {c.weight=0.0;} }
                for algorithm in [SolverAlgorithm::Cfr,SolverAlgorithm::CfrPlus,SolverAlgorithm::Dcfr] {
                    for schedule in [UpdateSchedule::Simultaneous,UpdateSchedule::Alternating] {
                        let make=||Trainer::new(build_tree(&cfg).unwrap(), board.clone(), combos.clone(), TrainerOptions { algorithm, schedule, abstraction: if bucketed {CardAbstraction{flop:Some(2),turn:Some(2),river:Some(2)}} else {CardAbstraction::default()} }).unwrap();
                        let mut candidate=make();let mut legacy=make();
                        for batch in [0,1,3] {
                            assert_eq!(candidate.train(batch).map(f64::to_bits),legacy.legacy_train(batch).map(f64::to_bits),"{board_text} iso={iso} bucketed={bucketed} {algorithm:?} {schedule:?}");
                            assert_eq!(candidate.iterations_completed,legacy.iterations_completed);
                            for (a,b) in candidate.node_data.iter().zip(&legacy.node_data) {
                                let a=a.lock().unwrap();let b=b.lock().unwrap();
                                assert_bits(&a.regret_sum,&b.regret_sum);
                                assert_bits(&a.strategy_sum,&b.strategy_sum);
                            }
                            let a=candidate.exploitability();let b=legacy.exploitability();
                            assert_eq!([a.oop,a.ip,a.total].map(f64::to_bits),[b.oop,b.ip,b.total].map(f64::to_bits));
                            assert_eq!(candidate.evaluate().1.map(f64::to_bits),legacy.evaluate().1.map(f64::to_bits));
                        }
                    }
                }
            }}
        }
    });
}

#[test]
fn target_values_skip_unused_terminal_work_but_preserve_final_return() {
    use crate::solver::Chips;
    use crate::{
        card_from_str,
        range::Range,
        solver::{build_tree, GameConfig},
    };
    use std::sync::atomic::Ordering;
    let board = ["2c", "7d", "9h", "Ks", "3c"]
        .map(|s| card_from_str(s).unwrap())
        .to_vec();
    let cfg = GameConfig {
        board: board.clone(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(100),
        bet_sizes: vec![1.0],
        max_raises: 0,
        suit_isomorphism: false,
    };
    let combos = ["AA:0.5,QQ", "KK,JJ:0.25"]
        .map(|s| Range::parse(s).unwrap().combos(set_from_cards(&board)));
    let make = || {
        Trainer::new(
            build_tree(&cfg).unwrap(),
            board.clone(),
            combos.clone(),
            TrainerOptions {
                algorithm: SolverAlgorithm::Dcfr,
                schedule: UpdateSchedule::Alternating,
                ..TrainerOptions::default()
            },
        )
        .unwrap()
    };
    let mut candidate = make();
    let mut legacy = make();
    let expected = legacy.legacy_train(5);
    let actual = candidate.train(5);
    assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
    // Five terminals in this river tree. Old path: 2 players * 2 passes * 5
    // iterations. New path: 5 OOP passes + 5 IP passes + final IP's OOP vector.
    assert_eq!(
        legacy
            .terminal_calls
            .each_ref()
            .map(|c| c.load(Ordering::Relaxed)),
        [50, 50]
    );
    assert_eq!(
        candidate
            .terminal_calls
            .each_ref()
            .map(|c| c.load(Ordering::Relaxed)),
        [30, 25]
    );
}

#[test]
fn direct_combo_updates_preserve_long_batches_and_sparse_tied_buckets() {
    use crate::solver::Chips;
    use crate::{
        card_from_str,
        solver::{build_tree, GameConfig},
    };
    let board = ["2h", "3d", "7c", "9s", "Jd"]
        .map(|s| card_from_str(s).unwrap())
        .to_vec();
    let make_combo =
        |a: &str, b: &str, w| Combo::new(card_from_str(a).unwrap(), card_from_str(b).unwrap(), w);
    // The two kings tie below AA, so requesting three buckets produces
    // [0, 0, 2]: max(bucket)+1 equals combo count despite a merged bucket.
    let combos = [
        vec![
            make_combo("Kc", "Kd", 0.0),
            make_combo("Kh", "Ks", 0.5),
            make_combo("Ac", "Ad", 1.0),
        ],
        vec![
            make_combo("Qc", "Qd", 0.25),
            make_combo("Qh", "Qs", 1.0),
            make_combo("Ah", "As", 0.0),
        ],
    ];
    let cfg = GameConfig {
        board: board.clone(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(200),
        bet_sizes: vec![0.5, 1.0],
        max_raises: 1,
        suit_isomorphism: false,
    };
    for bucketed in [false, true] {
        for algorithm in [
            SolverAlgorithm::Cfr,
            SolverAlgorithm::CfrPlus,
            SolverAlgorithm::Dcfr,
        ] {
            for schedule in [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating] {
                let make = || {
                    Trainer::new(
                        build_tree(&cfg).unwrap(),
                        board.clone(),
                        combos.clone(),
                        TrainerOptions {
                            algorithm,
                            schedule,
                            abstraction: CardAbstraction {
                                river: bucketed.then_some(3),
                                ..CardAbstraction::default()
                            },
                        },
                    )
                    .unwrap()
                };
                let mut candidate = make();
                let mut legacy = make();
                if bucketed {
                    assert_eq!(candidate.combo_buckets(0, &board), (3, &[0, 0, 2][..]));
                } else {
                    assert!(candidate.bucket_maps.is_empty());
                }
                for batch in [0, 1, 25, 75] {
                    assert_eq!(
                        candidate.train(batch).map(f64::to_bits),
                        legacy.legacy_train(batch).map(f64::to_bits)
                    );
                    for (a, b) in candidate.node_data.iter().zip(&legacy.node_data) {
                        let a = a.lock().unwrap();
                        let b = b.lock().unwrap();
                        assert_bits(&a.regret_sum, &b.regret_sum);
                        assert_bits(&a.strategy_sum, &b.strategy_sum);
                    }
                    assert_eq!(
                        candidate.exploitability().total.to_bits(),
                        legacy.exploitability().total.to_bits()
                    );
                    assert_eq!(
                        candidate.evaluate().1.map(f64::to_bits),
                        legacy.evaluate().1.map(f64::to_bits)
                    );
                }
            }
        }
    }
}

#[test]
fn borrowed_reach_matches_owned_traversal_on_unequal_ranges_and_threads() {
    use crate::solver::Chips;
    use crate::{
        card_from_str,
        range::Range,
        solver::{build_tree, GameConfig},
    };
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let board = ["2c", "7c", "9d", "Jd"]
                .map(|s| card_from_str(s).unwrap())
                .to_vec();
            let mut combos = ["AA:0.5,QQ", "JJ"]
                .map(|s| Range::parse(s).unwrap().combos(set_from_cards(&board)));
            for combo in &mut combos[0] {
                if rank(combo.cards[0]) == rank(card_from_str("Qc").unwrap()) {
                    combo.weight = 0.0;
                }
            }
            assert_ne!(combos[0].len(), combos[1].len());
            for iso in [false, true] {
                for bucketed in [false, true] {
                    let cfg = GameConfig {
                        board: board.clone(),
                        pot: Chips::whole(100),
                        effective_stack: Chips::whole(150),
                        bet_sizes: vec![0.5, 1.0],
                        max_raises: 1,
                        suit_isomorphism: iso,
                    };
                    for algorithm in [
                        SolverAlgorithm::Cfr,
                        SolverAlgorithm::CfrPlus,
                        SolverAlgorithm::Dcfr,
                    ] {
                        for schedule in [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating]
                        {
                            let make = || {
                                Trainer::new(
                                    build_tree(&cfg).unwrap(),
                                    board.clone(),
                                    combos.clone(),
                                    TrainerOptions {
                                        algorithm,
                                        schedule,
                                        abstraction: if bucketed {
                                            CardAbstraction {
                                                turn: Some(2),
                                                river: Some(2),
                                                ..CardAbstraction::default()
                                            }
                                        } else {
                                            CardAbstraction::default()
                                        },
                                    },
                                )
                                .unwrap()
                            };
                            let mut candidate = make();
                            let mut legacy = make();
                            for batch in [0, 1, 4] {
                                assert_eq!(
                                    candidate.train(batch).map(f64::to_bits),
                                    legacy.legacy_train(batch).map(f64::to_bits)
                                );
                                for (a, b) in candidate.node_data.iter().zip(&legacy.node_data) {
                                    let a = a.lock().unwrap();
                                    let b = b.lock().unwrap();
                                    assert_bits(&a.regret_sum, &b.regret_sum);
                                    assert_bits(&a.strategy_sum, &b.strategy_sum);
                                }
                                assert_eq!(
                                    candidate.exploitability().total.to_bits(),
                                    legacy.exploitability().total.to_bits()
                                );
                                assert_eq!(
                                    candidate.evaluate().1.map(f64::to_bits),
                                    legacy.evaluate().1.map(f64::to_bits)
                                );
                            }
                        }
                    }
                }
            }
        });
    }
}
