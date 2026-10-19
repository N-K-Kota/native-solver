//! Fixed-policy audit, independent of equality between two learning runs.
//! The adapter is test-only: complete trees, symmetric ranges, no abstraction.
use super::*;
use crate::card::card_from_str;
use crate::range::Range;
use crate::solver::Chips;
use crate::solver::{build_tree_bounded, quality::ReferenceEvaluator, GameConfig};
use std::collections::BTreeMap;

type Policies = BTreeMap<usize, Vec<Vec<f64>>>;

/// Concrete history -> representative history. The same composed permutation
/// must map BOTH public cards and hole cards. Returns the count of nontrivial
/// swaps applied after an earlier nontrivial permutation (flop coverage).
fn expand(
    full: &Node,
    reduced: &Node,
    trainer: &Trainer,
    map: [u8; 52],
    out: &mut Policies,
    get: &impl Fn(usize) -> Vec<Vec<f64>>,
) -> usize {
    match (full, reduced) {
        (
            Node::Decision {
                id,
                player,
                actions,
                children,
            },
            Node::Decision {
                id: rid,
                player: rp,
                actions: ra,
                children: rc,
            },
        ) => {
            assert_eq!(player, rp);
            assert_eq!(format!("{actions:?}"), format!("{ra:?}"));
            let combos = trainer.combos(player.index());
            let policy = get(*rid);
            let indices: Vec<_> = combos
                .iter()
                .map(
                    |&Combo {
                         cards: [a, b],
                         weight: w,
                     }| {
                        let (x, y) = (map[a as usize], map[b as usize]);
                        let i = combos
                            .iter()
                            .position(|&Combo { cards: [c, d], .. }| {
                                (c == x && d == y) || (c == y && d == x)
                            })
                            .unwrap();
                        assert_eq!(w, combos[i].weight, "range weights must be suit symmetric");
                        i
                    },
                )
                .collect();
            assert!(out
                .insert(
                    *id,
                    policy
                        .iter()
                        .map(|row| indices.iter().map(|&i| row[i]).collect())
                        .collect()
                )
                .is_none());
            assert_eq!(children.len(), rc.len());
            children
                .iter()
                .zip(rc)
                .map(|(a, b)| expand(a, b, trainer, map, out, get))
                .sum()
        }
        (Node::Chance { children }, Node::Chance { children: rc }) => children
            .iter()
            .map(|(cards, child)| {
                assert_eq!(cards.len(), 1);
                let card = map[cards[0] as usize];
                let (members, rep) = rc
                    .iter()
                    .find(|(members, _)| members.contains(&card))
                    .unwrap();
                let (a, b) = (card % 4, members[0] % 4);
                let composed =
                    usize::from(a != b && map.iter().enumerate().any(|(i, &c)| i != c as usize));
                let mut next = map;
                for c in &mut next {
                    let s = *c % 4;
                    *c = *c / 4 * 4
                        + if s == a {
                            b
                        } else if s == b {
                            a
                        } else {
                            s
                        };
                }
                composed + expand(child, rep, trainer, next, out, get)
            })
            .sum(),
        (Node::Terminal(a), Node::Terminal(b)) => {
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            0
        }
        _ => panic!("tree shape mismatch"),
    }
}

fn projected(
    full: &Node,
    trainer: &Trainer,
    get: impl Fn(usize) -> Vec<Vec<f64>>,
) -> (Policies, usize) {
    let mut policies = Policies::new();
    let composed = expand(
        full,
        trainer.tree(),
        trainer,
        std::array::from_fn(|i| i as u8),
        &mut policies,
        &get,
    );
    (policies, composed)
}

fn config(board: &[&str]) -> GameConfig {
    GameConfig {
        board: board.iter().map(|s| card_from_str(s).unwrap()).collect(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(150),
        bet_sizes: vec![0.33, 0.75],
        max_raises: 1,
        suit_isomorphism: false,
    }
}

fn combos(cfg: &GameConfig, ranges: [&str; 2]) -> [Vec<Combo>; 2] {
    ranges.map(|s| {
        let mut cs = Range::parse(s).unwrap().combos(set_from_cards(&cfg.board));
        for combo in &mut cs {
            combo.cards.sort_unstable();
        }
        cs.sort_by_key(|&Combo { cards: [a, b], .. }| (a, b));
        cs
    })
}

fn trainer(cfg: &GameConfig, cs: &[Vec<Combo>; 2], iso: bool) -> Trainer {
    let mut cfg = cfg.clone();
    cfg.suit_isomorphism = iso;
    Trainer::new(
        build_tree_bounded(&cfg, 100_000).unwrap(),
        cfg.board,
        cs.clone(),
        TrainerOptions {
            algorithm: SolverAlgorithm::Dcfr,
            schedule: UpdateSchedule::Alternating,
            ..TrainerOptions::default()
        },
    )
}

/// Compare individual normalized EVs and BRs, not just their zero-sum total.
fn audit(oracle: &ReferenceEvaluator<'_>, full: &Node, t: &Trainer, expected_composition: bool) {
    let (policy, composed) = projected(full, t, |id| t.average_strategy(id));
    if expected_composition {
        assert!(composed > 0, "exercise successive nontrivial suit swaps");
    }
    let q = oracle.evaluate(|id| policy[&id].clone()).unwrap();
    let actual = t.exploitability();
    let ev = t.evaluate().1.map(|v| v / q.joint_mass);
    for (name, a, b) in [
        ("OOP EV", ev[0], q.profile_ev[0]),
        ("IP EV", ev[1], q.profile_ev[1]),
        ("OOP BR", actual.oop, q.best_response_ev[0]),
        ("IP BR", actual.ip, q.best_response_ev[1]),
        ("exploitability", actual.total, q.exploitability),
    ] {
        assert!(
            (a - b).abs() < 1e-8,
            "{name} at {}: trainer={a} reference={b}",
            t.iterations_completed()
        );
    }
    println!(
        "audit iterations={} quality={} ev={ev:?} br={:?} max_error={:.3e}",
        t.iterations_completed(),
        q.exploitability,
        q.best_response_ev,
        [
            ev[0] - q.profile_ev[0],
            ev[1] - q.profile_ev[1],
            actual.oop - q.best_response_ev[0],
            actual.ip - q.best_response_ev[1]
        ]
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max)
    );
}

fn current(t: &Trainer, id: usize) -> Vec<Vec<f64>> {
    let data = t.node_data[id].lock().unwrap();
    let (actions, hands) = (data.regret_sum.len(), data.regret_sum[0].len());
    drop(data);
    t.current_strategy(id, actions, hands)
}

/// Diagnostics compare the same public history and concrete, board-live hand.
/// Never require two independently learned profiles to be numerically equal.
fn trace(full: &Trainer, reduced: &Trainer) {
    let (regrets, _) = projected(full.tree(), reduced, |id| {
        reduced.node_data[id].lock().unwrap().regret_sum.clone()
    });
    let (strategy, _) = projected(full.tree(), reduced, |id| current(reduced, id));
    let mut max_regret: f64 = 0.0;
    let mut max_policy = (0.0, 0, 0);
    for (&id, rows) in &regrets {
        let own = full.node_data[id].lock().unwrap();
        for (row, other) in rows.iter().zip(&own.regret_sum) {
            for (a, b) in row.iter().zip(other) {
                max_regret = max_regret.max((a - b).abs());
            }
        }
        drop(own);
        let own = current(full, id);
        let (player, board) = &full.node_meta[id];
        for (a, b) in own.iter().zip(&strategy[&id]) {
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                let [h1, h2] = full.combos[*player][i].cards;
                if !board.contains(&h1) && !board.contains(&h2) && (x - y).abs() > max_policy.0 {
                    max_policy = ((x - y).abs(), id, i);
                }
            }
        }
    }
    let (gap, id, i) = max_policy;
    println!("trace iterations={} regret_gap={max_regret:.3e} current_gap={gap} node={id} hand={i} board={:?} full_regrets={:?} reduced_regrets={:?}",
        full.iterations_completed(),full.node_board(id),full.node_data[id].lock().unwrap().regret_sum.iter().map(|r|r[i]).collect::<Vec<_>>(),regrets[&id].iter().map(|r|r[i]).collect::<Vec<_>>());
    println!(
        "trajectory full={} reduced={}",
        full.exploitability().total,
        reduced.exploitability().total
    );
}

#[test]
fn turn_fixed_profiles_match_independent_ev_and_each_best_response() {
    for threads in [1, 2] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let cfg = config(&["Js", "9s", "4d", "2d"]);
                let cs = combos(&cfg, ["AKs,AQs", "AKs,AQs"]);
                let tree = build_tree_bounded(&cfg, 100_000).unwrap();
                let oracle =
                    ReferenceEvaluator::new(&tree.root, &cfg.board, [&cs[0], &cs[1]]).unwrap();
                let (mut full, mut reduced) = (trainer(&cfg, &cs, false), trainer(&cfg, &cs, true));
                for step in 0..=25 {
                    if [0, 1, 2, 3, 4, 5, 25].contains(&step) {
                        audit(&oracle, &tree.root, &full, false);
                        audit(&oracle, &tree.root, &reduced, false);
                        if threads == 1 {
                            trace(&full, &reduced);
                        }
                    }
                    if step < 25 {
                        full.train(1);
                        reduced.train(1);
                    }
                }
            });
    }
}

#[test]
fn flop_fixed_profile_audits_composed_swaps_and_weighted_ranges() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let mut cfg = config(&["Ks", "7s", "2s"]);
            cfg.effective_stack = Chips::whole(100);
            cfg.bet_sizes = vec![0.5];
            cfg.max_raises = 0;
            let cs = combos(&cfg, ["AKs:0.5,AQs", "AKs,AQs:0.25"]);
            let tree = build_tree_bounded(&cfg, 100_000).unwrap();
            assert!(tree.num_decision_nodes * cs[0].len() * cs[1].len() < 3_000_000);
            let oracle = ReferenceEvaluator::new(&tree.root, &cfg.board, [&cs[0], &cs[1]]).unwrap();
            let mut reduced = trainer(&cfg, &cs, true);
            for step in [0, 1, 5, 25] {
                reduced.train(step - reduced.iterations_completed() as u32);
                audit(&oracle, &tree.root, &reduced, true);
            }
        });
}

#[test]
fn tiny_positive_regrets_retain_scale_and_explain_zero_discontinuity() {
    let cfg = config(&["Js", "9s", "4d", "2d", "3c"]);
    let cs = combos(&cfg, ["AKs,AQs", "AKs,AQs"]);
    let t = trainer(&cfg, &cs, false);
    for scale in [1.0, 1e-18, 1e-100] {
        let mut data = t.node_data[0].lock().unwrap();
        for (row, value) in data.regret_sum.iter_mut().zip([-1.0, 1.0, 3.0]) {
            row.fill(value * scale);
        }
        drop(data);
        let policy = current(&t, 0);
        assert_eq!(
            [policy[0][0], policy[1][0], policy[2][0]],
            [0.0, 0.25, 0.75]
        );
    }
    let mut data = t.node_data[0].lock().unwrap();
    for row in &mut data.regret_sum {
        row.fill(0.0);
    }
    drop(data);
    assert_eq!(current(&t, 0)[0][0], 1.0 / 3.0);
}

// Manual audits of the two archived matrix fixtures. More expensive than the
// small CI cases; bounded trees and explicit pair-work caps remain enforced.
fn audit_matrix(mut cfg: GameConfig, ranges: [&str; 2], checkpoints: &[u32]) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            cfg.suit_isomorphism = false;
            let cs = combos(&cfg, ranges);
            let tree = build_tree_bounded(&cfg, 100_000).unwrap();
            let work = tree.num_decision_nodes * cs[0].len() * cs[1].len();
            assert!(work < 300_000_000, "manual audit work budget");
            println!(
                "matrix nodes={} combos={:?} pair_work={work}",
                tree.num_decision_nodes,
                cs.each_ref().map(Vec::len)
            );
            let oracle = ReferenceEvaluator::new(&tree.root, &cfg.board, [&cs[0], &cs[1]]).unwrap();
            let (mut full, mut reduced) = (trainer(&cfg, &cs, false), trainer(&cfg, &cs, true));
            for &step in checkpoints {
                full.train(step - full.iterations_completed() as u32);
                reduced.train(step - reduced.iterations_completed() as u32);
                audit(&oracle, &tree.root, &full, false);
                audit(&oracle, &tree.root, &reduced, cfg.board.len() == 3);
                println!(
                    "matrix trajectory step={step} full={} reduced={}",
                    full.exploitability().total,
                    reduced.exploitability().total
                );
            }
        });
}

#[test]
#[ignore = "bounded manual audit of archived wide matrix case"]
fn matrix_turn_wide_fixed_profiles() {
    audit_matrix(config(&["Js","9s","4d","2d"]),[
        "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,KQo",
        "AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,K8s,QJs,QTs,Q9s,JTs,J9s,T9s,T8s,98s,97s,87s,86s,76s,75s,65s,64s,54s,AKo,AQo,AJo,ATo,KQo,KJo,QJo"
    ],&[25,150]);
}

#[test]
#[ignore = "bounded manual audit of archived medium matrix case"]
fn matrix_flop_medium_fixed_profiles() {
    let mut cfg = config(&["Ks", "7s", "2s"]);
    cfg.effective_stack = Chips::whole(100);
    cfg.bet_sizes = vec![0.5];
    cfg.max_raises = 0;
    audit_matrix(
        cfg,
        ["AA,KK,QQ,JJ,TT,AKs,AQs,KQs", "QQ,JJ,TT,99,88,AQs,AJs,KQs"],
        &[25, 50, 75],
    );
}

fn assert_policy_symmetry(t: &Trainer) {
    let mut compared = 0;
    for (id, (player, board)) in t.node_meta.iter().enumerate() {
        let free: Vec<_> = (0..4)
            .filter(|s| !board.iter().any(|&c| suit(c) == *s))
            .collect();
        let (_, buckets) = t.combo_buckets(*player, board);
        let current = current(t, id);
        let average = t.average_strategy(id);
        for &a in &free {
            for &b in &free {
                if a >= b {
                    continue;
                }
                for (
                    i,
                    &Combo {
                        cards: [h1, h2], ..
                    },
                ) in t.combos[*player].iter().enumerate()
                {
                    let mapped = normalize_pair(swap_suit(h1, a, b), swap_suit(h2, a, b));
                    let j = t.combos[*player]
                        .iter()
                        .position(|&Combo { cards: [c, d], .. }| normalize_pair(c, d) == mapped)
                        .unwrap();
                    if i == j {
                        continue;
                    }
                    for row in &current {
                        assert_eq!(row[buckets[i]], row[buckets[j]], "current node {id}");
                    }
                    for row in &average {
                        assert_eq!(row[i], row[j], "average node {id}");
                    }
                    compared += 1;
                }
            }
        }
        for i in 0..t.combos[*player].len() {
            assert!((average.iter().map(|r| r[i]).sum::<f64>() - 1.0).abs() < 1e-12);
            assert!((current.iter().map(|r| r[buckets[i]]).sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }
    assert!(compared > 0);
}

#[test]
fn reduced_policies_are_exactly_symmetric_across_algorithms_and_buckets() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| {
            let mut cfg = config(&["Js", "9s", "4d", "2d"]);
            let cs = combos(&cfg, ["AKs:0.5,AQs", "AKs,AQs:0.25"]);
            let full = build_tree_bounded(&cfg, 100_000).unwrap();
            let oracle = ReferenceEvaluator::new(&full.root, &cfg.board, [&cs[0], &cs[1]]).unwrap();
            cfg.suit_isomorphism = true;
            for algorithm in [
                SolverAlgorithm::Cfr,
                SolverAlgorithm::CfrPlus,
                SolverAlgorithm::Dcfr,
            ] {
                for schedule in [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating] {
                    for abstraction in [
                        CardAbstraction::default(),
                        CardAbstraction {
                            flop: None,
                            turn: Some(3),
                            river: Some(3),
                        },
                    ] {
                        let mut t = Trainer::new(
                            build_tree_bounded(&cfg, 100_000).unwrap(),
                            cfg.board.clone(),
                            cs.clone(),
                            TrainerOptions {
                                algorithm,
                                schedule,
                                abstraction,
                            },
                        );
                        t.train(25);
                        assert_policy_symmetry(&t);
                        audit(&oracle, &full.root, &t, false);
                    }
                }
            }
        });
}

#[test]
#[should_panic(expected = "suit reduction requires suit-symmetric range weights")]
fn reduced_tree_rejects_asymmetric_initial_weights() {
    let cfg = config(&["Js", "9s", "4d", "2d"]);
    let mut cs = combos(&cfg, ["AKs,AQs", "AKs,AQs"]);
    cs[0][0].weight = 0.25;
    let _ = trainer(&cfg, &cs, true);
}
