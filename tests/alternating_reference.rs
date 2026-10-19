//! Independent scalar Kuhn recurrence: no production traversal/discount helpers.
use native_solver::solver::Chips;
use native_solver::solver::TrainerOptions;
use native_solver::{
    card_from_str,
    solver::{build_tree, GameConfig, SolverAlgorithm, Trainer, UpdateSchedule},
    Combo,
};

type Table = [[[f64; 2]; 3]; 4];
const ACTOR: [usize; 4] = [0, 1, 0, 1];

fn policy(regret: &Table) -> Table {
    let mut result = [[[0.5; 2]; 3]; 4];
    for n in 0..4 {
        for h in 0..3 {
            let positive = regret[n][h].map(|r| r.max(0.0));
            let sum: f64 = positive.iter().sum();
            if sum > 0.0 {
                result[n][h] = positive.map(|r| r / sum);
            }
        }
    }
    result
}

// Histories are literal Kuhn actions, independent of the NLHE game tree.
fn visit(
    history: &str,
    hands: [usize; 2],
    reach: [f64; 2],
    strategy: &Table,
    regret: &mut Table,
    mass: &mut Table,
) -> [f64; 2] {
    let showdown = if hands[0] > hands[1] { 1.0 } else { -1.0 };
    let payoff = match history {
        "xx" => Some(showdown),
        "bc" | "xbc" => Some(2.0 * showdown),
        "bf" => Some(1.0),
        "xbf" => Some(-1.0),
        _ => None,
    };
    if let Some(v) = payoff {
        return [v, -v];
    }
    let (n, actions) = match history {
        "" => (0, ['x', 'b']),
        "x" => (1, ['x', 'b']),
        "xb" => (2, ['f', 'c']),
        "b" => (3, ['f', 'c']),
        _ => unreachable!(),
    };
    let p = ACTOR[n];
    let h = hands[p];
    let sigma = strategy[n][h];
    let mut values = [[0.0; 2]; 2];
    for a in 0..2 {
        let mut next = reach;
        next[p] *= sigma[a];
        values[a] = visit(
            &format!("{history}{}", actions[a]),
            hands,
            next,
            strategy,
            regret,
            mass,
        );
    }
    let value = [
        sigma[0] * values[0][0] + sigma[1] * values[1][0],
        sigma[0] * values[0][1] + sigma[1] * values[1][1],
    ];
    for a in 0..2 {
        regret[n][h][a] += reach[1 - p] * (values[a][p] - value[p]);
        // Each own hand appears in exactly two legal deals. Own reach depends
        // only on own hand/history, so count the averaging contribution once.
        mass[n][h][a] += 0.5 * reach[p] * sigma[a];
    }
    value
}

#[test]
fn alternating_dcfr_matches_independent_scalar_history_recurrence() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let card = |s| card_from_str(s).unwrap();
        let cfg = GameConfig {
            board: ["2h", "3c", "7d", "8h", "9c"].map(card).to_vec(),
            pot: Chips::whole(2),
            effective_stack: Chips::whole(1),
            bet_sizes: vec![0.5],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let combos = vec![
            Combo::new(card("Js"), card("Jd"), 1.0),
            Combo::new(card("Qs"), card("Qd"), 1.0),
            Combo::new(card("Ks"), card("Kd"), 1.0),
        ];
        for schedule in [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating] {
            let mut trainer = Trainer::new(
                build_tree(&cfg).unwrap(),
                cfg.board.clone(),
                [combos.clone(), combos.clone()],
                TrainerOptions {
                    algorithm: SolverAlgorithm::Dcfr,
                    schedule,
                    ..TrainerOptions::default()
                },
            )
            .unwrap();
            let mut regrets: Table = [[[0.0; 2]; 3]; 4];
            let mut sums = regrets;
            for t in 1..=128 {
                let players: &[Option<usize>] = if schedule == UpdateSchedule::Alternating {
                    &[Some(0), Some(1)]
                } else {
                    &[None]
                };
                for &target in players {
                    // Freeze all policies during each scalar player pass.
                    let strategy = policy(&regrets);
                    let mut deltas: Table = [[[0.0; 2]; 3]; 4];
                    let mut mass = deltas;
                    for oop in 0..3 {
                        for ip in 0..3 {
                            if oop != ip {
                                visit("", [oop, ip], [1.0, 1.0], &strategy, &mut deltas, &mut mass);
                            }
                        }
                    }
                    let time = t as f64;
                    for n in 0..4 {
                        if target.is_none_or(|p| p == ACTOR[n]) {
                            for h in 0..3 {
                                for a in 0..2 {
                                    let accumulated = regrets[n][h][a] + deltas[n][h][a];
                                    regrets[n][h][a] = accumulated
                                        * if accumulated > 0.0 {
                                            time.powf(1.5) / (time.powf(1.5) + 1.0)
                                        } else {
                                            0.5
                                        };
                                    sums[n][h][a] = (sums[n][h][a] + mass[n][h][a])
                                        * (time / (time + 1.0)).powi(2);
                                }
                            }
                        }
                    }
                }
                trainer.train(1);
                for (n, node_sums) in sums.iter().enumerate() {
                    let actual = trainer.average_strategy(n);
                    for (h, sum) in node_sums.iter().enumerate() {
                        let total = sum[0] + sum[1];
                        for a in 0..2 {
                            let expected = if total > 0.0 { sum[a] / total } else { 0.5 };
                            assert!(
                                (actual[a][h] - expected).abs() < 1e-11,
                                "{schedule:?} t={t} node={n} hand={h} action={a}: {} != {expected}",
                                actual[a][h]
                            );
                        }
                    }
                }
            }
        }
    });
}

#[test]
fn alternating_suit_reduction_matches_full_runouts_and_parallel_execution() {
    use native_solver::{range::Range, set_from_cards};
    let board = ["2c", "7c", "9d", "Jd"]
        .map(|s| card_from_str(s).unwrap())
        .to_vec();
    let combos = ["AA:0.5,QQ", "KK,TT:0.25"]
        .map(|s| Range::parse(s).unwrap().combos(set_from_cards(&board)));
    let mut reference: Option<(f64, Vec<Vec<f64>>)> = None;
    let mut full_nodes = 0;
    for (threads, iso) in [(1, false), (1, true), (2, true)] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let tree = build_tree(&GameConfig {
                board: board.clone(),
                pot: Chips::whole(100),
                effective_stack: Chips::whole(100),
                bet_sizes: vec![1.0],
                max_raises: 0,
                suit_isomorphism: iso,
            })
            .unwrap();
            if iso {
                assert!(tree.num_decision_nodes < full_nodes);
            } else {
                full_nodes = tree.num_decision_nodes;
            }
            let mut trainer = Trainer::new(
                tree,
                board.clone(),
                combos.clone(),
                TrainerOptions {
                    algorithm: SolverAlgorithm::Dcfr,
                    schedule: UpdateSchedule::Alternating,
                    ..TrainerOptions::default()
                },
            )
            .unwrap();
            trainer.train(30);
            let q = trainer.exploitability().total;
            let strategy = trainer.average_strategy(0);
            if let Some((expected, policy)) = &reference {
                assert!(
                    (q - expected).abs() < 1e-9,
                    "{threads} threads, iso={iso}: {q} != {expected}"
                );
                for (actual, expected) in strategy.iter().flatten().zip(policy.iter().flatten()) {
                    assert!((actual - expected).abs() < 1e-9);
                }
            } else {
                reference = Some((q, strategy));
            }
        });
    }
}
