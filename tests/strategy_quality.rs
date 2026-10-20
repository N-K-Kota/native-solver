//! Independent quality-oracle regressions. Expectations below are closed-form
//! betting-game solutions or direct legal-runout enumeration, not solver CFVs.
use native_solver::solver::quality::{QualityError, ReferenceEvaluator, StrategyQuality};
use native_solver::solver::Chips;
use native_solver::solver::TrainerOptions;
use native_solver::solver::{
    build_tree, Action, CardAbstraction, GameConfig, Node, Player, SolverAlgorithm, Terminal,
    Trainer, TrainerError, Tree, UpdateSchedule,
};
use native_solver::{card_from_str, evaluate_7, set_from_cards, Card, Combo};
use std::collections::HashMap;

fn card(s: &str) -> Card {
    card_from_str(s).unwrap()
}
fn board(names: &[&str]) -> Vec<Card> {
    names.iter().map(|s| card(s)).collect()
}
fn combo(a: &str, b: &str, w: f64) -> Combo {
    Combo::new(card(a), card(b), w)
}
/// Drops combos that share a card with `board`, as `Range::combos` does.
fn live(combos: &[Vec<Combo>; 2], board: &[Card]) -> [Vec<Combo>; 2] {
    let dead = set_from_cards(board);
    combos.clone().map(|range| {
        range
            .into_iter()
            .filter(|c| !c.conflicts_with(dead))
            .collect()
    })
}
fn river() -> Vec<Card> {
    board(&["2c", "7d", "9h", "Js", "Qd"])
}
fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "actual {actual:.15}, expected {expected:.15}"
    );
}

fn score(hand: Combo, b: &[Card]) -> u32 {
    evaluate_7([hand.cards[0], hand.cards[1], b[0], b[1], b[2], b[3], b[4]])
}

// OOP may check directly to showdown or bet. IP then calls or folds.
fn one_bet_tree(bet: f64) -> Tree {
    Tree {
        board: river(),
        root: Node::Decision {
            id: 0,
            player: Player::Oop,
            actions: vec![Action::Check, Action::Bet(Chips::from_f64(bet).unwrap())],
            children: vec![
                Node::Terminal(Terminal::Showdown {
                    invested: [5.0, 5.0],
                }),
                Node::Decision {
                    id: 1,
                    player: Player::Ip,
                    actions: vec![Action::Call, Action::Fold],
                    children: vec![
                        Node::Terminal(Terminal::Showdown {
                            invested: [5.0 + bet, 5.0 + bet],
                        }),
                        Node::Terminal(Terminal::Fold {
                            winner: Player::Oop,
                            invested: [5.0 + bet, 5.0],
                        }),
                    ],
                },
            ],
        },
        num_decision_nodes: 2,
    }
}

fn uniform(node: &Node, sizes: [usize; 2]) -> HashMap<usize, Vec<Vec<f64>>> {
    fn visit(node: &Node, sizes: [usize; 2], out: &mut HashMap<usize, Vec<Vec<f64>>>) {
        match node {
            Node::Decision {
                id,
                player,
                children,
                ..
            } => {
                out.insert(
                    *id,
                    vec![vec![1.0 / children.len() as f64; sizes[player.index()]]; children.len()],
                );
                for child in children {
                    visit(child, sizes, out);
                }
            }
            Node::Chance { children } => {
                for (_, child) in children {
                    visit(child, sizes, out);
                }
            }
            Node::Terminal(_) => {}
        }
    }
    let mut out = HashMap::new();
    visit(node, sizes, &mut out);
    out
}

fn compare_trainer(trainer: &Trainer, initial_board: &[Card]) -> StrategyQuality {
    let evaluator = ReferenceEvaluator::new(
        trainer.game().tree(),
        initial_board,
        Player::ALL.map(|player| trainer.game().combos(player)),
    )
    .unwrap();
    let reference = evaluator
        .evaluate(|id| trainer.average_strategy(id))
        .unwrap();
    let production = trainer.exploitability();
    let raw_ev = trainer.evaluate().root_values;
    near(reference.profile_ev[0], raw_ev[0] / reference.joint_mass);
    near(reference.profile_ev[1], raw_ev[1] / reference.joint_mass);
    near(reference.best_response_ev[0], production.oop);
    near(reference.best_response_ev[1], production.ip);
    near(reference.exploitability, production.total);
    near(reference.profile_ev.iter().sum(), 0.0);
    near(reference.nash_conv, reference.best_response_ev.iter().sum());
    assert!(reference
        .best_response_ev
        .iter()
        .zip(reference.profile_ev)
        .all(|(br, ev)| br + 1e-10 >= ev));
    reference
}

#[test]
fn weighted_blocked_deals_have_hand_calculated_nonzero_exploitability() {
    let b = river();
    let base = [
        vec![
            combo("Ac", "Ad", 0.7),
            combo("Ah", "As", 0.3),
            combo("Qd", "Qc", 999.0),
        ],
        vec![
            combo("Kc", "Kd", 0.2),
            combo("Ac", "Ks", 0.8),
            combo("2c", "3d", 888.0),
        ],
    ];
    for oop in &base[0][..2] {
        for ip in &base[1][..2] {
            assert!(score(*oop, &b) > score(*ip, &b));
        }
    }
    for factors in [[1.0, 1.0], [13.0, 7.0], [1e-100, 0.5]] {
        let mut combos = base.clone();
        for p in 0..2 {
            for c in &mut combos[p] {
                c.weight *= factors[p];
            }
        }
        let new = |combos| {
            Trainer::new(
                one_bet_tree(5.0),
                combos,
                TrainerOptions {
                    algorithm: SolverAlgorithm::CfrPlus,
                    ..TrainerOptions::default()
                },
            )
        };
        // The heavy board-blocked combos are rejected rather than silently
        // weighed, so they cannot leak into the joint mass.
        assert!(matches!(
            new(combos.clone()).err(),
            Some(TrainerError::ComboConflictsWithBoard { .. })
        ));
        let trainer = new(live(&combos, &b)).unwrap();
        let result = compare_trainer(&trainer, &b);
        // Only AA_c/d versus K_c/d and AA_h/s versus either IP hand survive.
        let expected_mass = 0.44 * factors[0] * factors[1];
        assert!((result.joint_mass / expected_mass - 1.0).abs() < 1e-12);
        // Check wins 5. Bet wins (10 + 5)/2. Uniform profile wins 6.25.
        near(result.profile_ev[0], 6.25);
        near(result.profile_ev[1], -6.25);
        near(result.best_response_ev[0], 7.5);
        near(result.best_response_ev[1], -5.0);
        near(result.nash_conv, 2.5);
        near(result.exploitability, 1.25);
    }
}

#[test]
fn two_type_betting_game_matches_closed_form_for_arbitrary_policies() {
    let b = river();
    let combos = [
        vec![combo("Ac", "Ad", 1.0), combo("3c", "3d", 1.0)],
        vec![combo("Kc", "Kd", 1.0)],
    ];
    assert!(score(combos[0][0], &b) > score(combos[1][0], &b));
    assert!(score(combos[0][1], &b) < score(combos[1][0], &b));
    let tree = one_bet_tree(10.0);
    let evaluator = ReferenceEvaluator::new(&tree.root, &b, [&combos[0], &combos[1]]).unwrap();
    for value_bet in [0.0, 0.2, 0.5, 1.0] {
        for bluff_bet in [0.0, 0.2, 0.5, 1.0] {
            for call in [0.0, 0.2, 0.5, 1.0] {
                let result = evaluator
                    .evaluate(|id| match id {
                        0 => vec![
                            vec![1.0 - value_bet, 1.0 - bluff_bet],
                            vec![value_bet, bluff_bet],
                        ],
                        1 => vec![vec![call], vec![1.0 - call]],
                        _ => unreachable!(),
                    })
                    .unwrap();
                let ev = 0.5 * ((1.0 - value_bet) * 5.0 + value_bet * (5.0 + 10.0 * call))
                    + 0.5 * ((1.0 - bluff_bet) * -5.0 + bluff_bet * (5.0 - 20.0 * call));
                let oop_br =
                    0.5 * 5.0_f64.max(5.0 + 10.0 * call) + 0.5 * (-5.0_f64).max(5.0 - 20.0 * call);
                let ip_check = -2.5 * (1.0 - value_bet) + 2.5 * (1.0 - bluff_bet);
                let ip_call = 7.5 * (bluff_bet - value_bet);
                let ip_fold = -2.5 * (value_bet + bluff_bet);
                let ip_br = ip_check + ip_call.max(ip_fold);
                near(result.profile_ev[0], ev);
                near(result.profile_ev[1], -ev);
                near(result.best_response_ev[0], oop_br);
                near(result.best_response_ev[1], ip_br);
                near(result.exploitability, (oop_br + ip_br) / 2.0);
            }
        }
    }
    // Pot-size equilibrium: value always bets, half of air bluffs, IP calls half.
    let equilibrium = evaluator
        .evaluate(|id| {
            if id == 0 {
                vec![vec![0.0, 0.5], vec![1.0, 0.5]]
            } else {
                vec![vec![0.5], vec![0.5]]
            }
        })
        .unwrap();
    near(equilibrium.profile_ev[0], 2.5);
    near(equilibrium.best_response_ev[0], 2.5);
    near(equilibrium.best_response_ev[1], -2.5);
    near(equilibrium.exploitability, 0.0);
}

#[test]
fn best_response_cannot_see_opponents_private_cards() {
    let b = river();
    let combos = [
        vec![combo("Ac", "Ad", 1.0), combo("3c", "3d", 1.0)],
        vec![combo("Kc", "Kd", 1.0)],
    ];
    let root = Node::Decision {
        id: 0,
        player: Player::Ip,
        actions: vec![Action::Call, Action::Fold],
        children: vec![
            Node::Terminal(Terminal::Showdown {
                invested: [15.0, 15.0],
            }),
            Node::Terminal(Terminal::Fold {
                winner: Player::Oop,
                invested: [15.0, 5.0],
            }),
        ],
    };
    let result = ReferenceEvaluator::new(&root, &b, [&combos[0], &combos[1]])
        .unwrap()
        .evaluate(|_| vec![vec![0.5], vec![0.5]])
        .unwrap();
    // One information set: call yields (-15 + 15)/2 = 0; folding yields -5.
    // A broken max-per-pair oracle would fold against AA and call against 33,
    // claiming (-5 + 15)/2 = 5 despite never observing OOP's private cards.
    near(result.best_response_ev[1], 0.0);
    near(result.profile_ev[1], -2.5);
    near(result.exploitability, 1.25);
    assert!((result.best_response_ev[1] - 5.0).abs() > 1.0);
}

#[test]
fn production_quality_matches_reference_before_and_after_training_all_algorithms() {
    let b = river();
    let combos = [
        vec![
            combo("Ac", "Ad", 0.3),
            combo("Ah", "As", 0.8),
            combo("3c", "3d", 1.0),
        ],
        vec![
            combo("Kc", "Kd", 0.7),
            combo("Ac", "Ks", 0.1),
            combo("9c", "9d", 0.4),
        ],
    ];
    let cfg = GameConfig {
        board: b.clone(),
        pot: Chips::whole(10),
        effective_stack: Chips::whole(20),
        bet_sizes: vec![0.5, 1.0],
        max_raises: 1,
        suit_isomorphism: false,
    };
    for (algorithm, schedule) in [
        SolverAlgorithm::Cfr,
        SolverAlgorithm::CfrPlus,
        SolverAlgorithm::Dcfr,
    ]
    .into_iter()
    .flat_map(|a| [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating].map(|s| (a, s)))
    {
        let mut trainer = Trainer::new(
            build_tree(&cfg).unwrap(),
            combos.clone(),
            TrainerOptions {
                algorithm,
                schedule,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        for iterations in [0, 1, 7, 31] {
            trainer.train(iterations);
            compare_trainer(&trainer, &b);
        }
    }
}

fn direct_showdown_ev(initial: &[Card], combos: &[Vec<Combo>; 2], invested: [f64; 2]) -> f64 {
    let mut total = 0.0;
    let mut mass = 0.0;
    for &a in &combos[0] {
        for &b in &combos[1] {
            let mut dead = initial.to_vec();
            dead.extend([a.cards[0], a.cards[1], b.cards[0], b.cards[1]]);
            let mut unique = dead.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != dead.len() {
                continue;
            }
            let remaining: Vec<Card> = (0..52).filter(|c| !dead.contains(c)).collect();
            let mut payoff = 0.0;
            let mut count = 0;
            let mut record = |completed: &[Card]| {
                payoff += match score(a, completed).cmp(&score(b, completed)) {
                    std::cmp::Ordering::Greater => invested[1],
                    std::cmp::Ordering::Less => -invested[0],
                    std::cmp::Ordering::Equal => (invested[1] - invested[0]) / 2.0,
                };
                count += 1;
            };
            if initial.len() == 4 {
                for &river in &remaining {
                    let mut completed = initial.to_vec();
                    completed.push(river);
                    record(&completed);
                }
                assert_eq!(count, 44);
            } else {
                for (i, &turn) in remaining.iter().enumerate() {
                    // Unordered runouts provide a different enumeration from
                    // the reference tree's ordered turn/river chance paths.
                    for &river in &remaining[i + 1..] {
                        let mut completed = initial.to_vec();
                        completed.extend([turn, river]);
                        record(&completed);
                    }
                }
                assert_eq!(count, 990);
            }
            let weight = a.weight * b.weight;
            total += weight * payoff / count as f64;
            mass += weight;
        }
    }
    total / mass
}

#[test]
fn turn_and_flop_all_in_match_direct_pair_runout_enumeration() {
    for initial in [board(&["Ts", "9s", "4d", "2c"]), board(&["Ts", "9s", "4d"])] {
        let combos = [
            vec![
                combo("As", "Ks", 0.3),
                combo("Th", "Td", 0.8),
                combo("4d", "4c", 7.0),
            ],
            vec![
                combo("Qh", "Qd", 0.6),
                combo("As", "Jc", 0.4),
                combo("4d", "5h", 9.0),
            ],
        ];
        let cfg = GameConfig {
            board: initial.clone(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(0),
            bet_sizes: vec![],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let combos = live(&combos, &initial);
        let trainer = Trainer::new(
            build_tree(&cfg).unwrap(),
            combos.clone(),
            TrainerOptions {
                algorithm: SolverAlgorithm::CfrPlus,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        let result = compare_trainer(&trainer, &initial);
        let expected = direct_showdown_ev(&initial, &combos, [50.0, 50.0]);
        near(result.profile_ev[0], expected);
        near(result.best_response_ev[0], expected);
        near(result.exploitability, 0.0);
    }
}

#[test]
fn turn_betting_with_private_blockers_matches_all_algorithm_profiles() {
    let b = board(&["Ts", "9s", "4d", "2c"]);
    let combos = [
        vec![combo("As", "Ks", 0.3), combo("Th", "Td", 0.8)],
        vec![combo("Qh", "Qd", 0.6), combo("As", "Jc", 0.4)],
    ];
    let cfg = GameConfig {
        board: b.clone(),
        pot: Chips::whole(10),
        effective_stack: Chips::whole(5),
        bet_sizes: vec![0.5],
        max_raises: 0,
        suit_isomorphism: false,
    };
    for (algorithm, schedule) in [
        SolverAlgorithm::Cfr,
        SolverAlgorithm::CfrPlus,
        SolverAlgorithm::Dcfr,
    ]
    .into_iter()
    .flat_map(|a| [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating].map(|s| (a, s)))
    {
        let mut trainer = Trainer::new(
            build_tree(&cfg).unwrap(),
            combos.clone(),
            TrainerOptions {
                algorithm,
                schedule,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        for iterations in [0, 1, 9] {
            trainer.train(iterations);
            compare_trainer(&trainer, &b);
        }
    }
}

#[test]
fn expanded_bucket_policy_is_evaluated_against_unrestricted_combo_responses() {
    let b = river();
    let combos = [
        vec![combo("Ac", "Ad", 1.0), combo("3c", "3d", 1.0)],
        vec![combo("Kc", "Kd", 1.0)],
    ];
    let mut trainer = Trainer::new(
        one_bet_tree(10.0),
        combos,
        TrainerOptions {
            algorithm: SolverAlgorithm::CfrPlus,
            abstraction: CardAbstraction {
                river: Some(1),
                ..Default::default()
            },
            ..TrainerOptions::default()
        },
    )
    .unwrap();
    trainer.train(5);
    let strategy = trainer.average_strategy(0);
    near(strategy[0][0], strategy[0][1]);
    let reference = compare_trainer(&trainer, &b);
    // Even though both OOP hands share one policy bucket, their full-game best
    // responses differ when q > 1/2: value bets, air checks.
    let call = trainer.average_strategy(1)[0][0];
    let expected = 0.5 * (5.0 + 10.0 * call) + 0.5 * (-5.0_f64).max(5.0 - 20.0 * call);
    near(reference.best_response_ev[0], expected);
}

#[test]
fn ties_and_unequal_investments_use_net_chip_payoffs() {
    let b = board(&["Ac", "Kc", "Qc", "Jc", "Tc"]);
    let combos = [vec![combo("2d", "3d", 0.3)], vec![combo("4h", "5h", 0.7)]];
    assert_eq!(score(combos[0][0], &b), score(combos[1][0], &b));
    let root = Node::Terminal(Terminal::Showdown {
        invested: [20.0, 10.0],
    });
    let result = ReferenceEvaluator::new(&root, &b, [&combos[0], &combos[1]])
        .unwrap()
        .evaluate(|_| unreachable!())
        .unwrap();
    near(result.profile_ev[0], -5.0);
    near(result.profile_ev[1], 5.0);
    near(result.exploitability, 0.0);
}

#[test]
fn malformed_boards_ranges_and_impossible_deals_are_errors() {
    let b = river();
    let root = Node::Terminal(Terminal::Showdown {
        invested: [5.0, 5.0],
    });
    let ranges = [vec![combo("Ac", "Ad", 1.0)], vec![combo("Kc", "Kd", 1.0)]];
    for invalid in [
        vec![],
        vec![0, 1],
        vec![0, 1, 2, 3, 4, 5],
        vec![0, 0, 2],
        vec![0, 1, 52],
    ] {
        assert!(matches!(
            ReferenceEvaluator::new(&root, &invalid, [&ranges[0], &ranges[1]]),
            Err(QualityError::InvalidBoard(_))
        ));
    }
    for bad in [
        vec![Combo::new(0, 0, 1.0)],
        vec![Combo::new(0, 52, 1.0)],
        vec![Combo::new(0, 1, -1.0)],
        vec![Combo::new(0, 1, f64::NAN)],
        vec![Combo::new(0, 1, f64::INFINITY)],
        vec![Combo::new(0, 1, 1.0), Combo::new(1, 0, 0.3)],
    ] {
        assert!(matches!(
            ReferenceEvaluator::new(&root, &b, [&bad, &ranges[1]]),
            Err(QualityError::InvalidRange { player: 0, .. })
        ));
    }
    for no_deals in [
        vec![],
        vec![combo("Ac", "Ad", 0.0)],
        vec![combo("Kc", "Ah", 1.0)],
        vec![combo("2c", "Ah", 1.0)],
        vec![combo("Ac", "Ad", f64::MAX), combo("Ah", "As", f64::MAX)],
    ] {
        assert!(matches!(
            ReferenceEvaluator::new(&root, &b, [&no_deals, &ranges[1]]),
            Err(QualityError::NoCompatibleDeals)
        ));
    }
}

#[test]
fn malformed_and_reduced_trees_are_rejected_instead_of_silently_misread() {
    let b = board(&["2c", "7c", "9c", "Jc"]);
    let ranges = [vec![combo("Ac", "Ad", 1.0)], vec![combo("Kc", "Kd", 1.0)]];
    let cfg = GameConfig {
        board: b.clone(),
        pot: Chips::whole(10),
        effective_stack: Chips::whole(0),
        bet_sizes: vec![],
        max_raises: 0,
        suit_isomorphism: true,
    };
    let reduced = build_tree(&cfg).unwrap();
    assert!(matches!(
        ReferenceEvaluator::new(&reduced.root, &b, [&ranges[0], &ranges[1]]),
        Err(QualityError::UnsupportedChanceReduction)
    ));
    let terminal = Node::Terminal(Terminal::Fold {
        winner: Player::Oop,
        invested: [5.0, 5.0],
    });
    let full_chance = || Node::Chance {
        children: (0..52)
            .filter(|c| !b.contains(c))
            .map(|c| (vec![c], terminal.clone()))
            .collect(),
    };
    let mut duplicate = full_chance();
    if let Node::Chance { children } = &mut duplicate {
        children[1].0 = children[0].0.clone();
    }
    let mut empty_class = full_chance();
    if let Node::Chance { children } = &mut empty_class {
        children[0].0.clear();
    }
    let decision = |id| Node::Decision {
        id,
        player: Player::Oop,
        actions: vec![Action::Check],
        children: vec![terminal.clone()],
    };
    let malformed = [
        Node::Chance { children: vec![] },
        duplicate,
        empty_class,
        Node::Terminal(Terminal::Showdown {
            invested: [5.0, 5.0],
        }),
        Node::Terminal(Terminal::Fold {
            winner: Player::Oop,
            invested: [f64::NAN, 5.0],
        }),
        Node::Terminal(Terminal::Fold {
            winner: Player::Oop,
            invested: [-1.0, 5.0],
        }),
        Node::Decision {
            id: 0,
            player: Player::Oop,
            actions: vec![],
            children: vec![],
        },
        Node::Decision {
            id: 0,
            player: Player::Oop,
            actions: vec![Action::Check],
            children: vec![],
        },
        Node::Decision {
            id: 0,
            player: Player::Oop,
            actions: vec![Action::Bet(Chips::ZERO)],
            children: vec![terminal.clone()],
        },
        Node::Decision {
            id: 0,
            player: Player::Oop,
            actions: vec![Action::Check, Action::Bet(Chips::whole(1))],
            children: vec![decision(1), decision(1)],
        },
    ];
    for root in malformed {
        assert!(matches!(
            ReferenceEvaluator::new(&root, &b, [&ranges[0], &ranges[1]]),
            Err(QualityError::InvalidTree(_))
        ));
    }
    assert!(matches!(
        ReferenceEvaluator::new(&full_chance(), &river(), [&ranges[0], &ranges[1]]),
        Err(QualityError::InvalidTree(_))
    ));
}

#[test]
fn every_strategy_column_is_validated_even_at_unreachable_nodes() {
    let b = river();
    let ranges = [vec![combo("Ac", "Ad", 1.0)], vec![combo("Kc", "Kd", 1.0)]];
    let tree = one_bet_tree(5.0);
    let evaluator = ReferenceEvaluator::new(&tree.root, &b, [&ranges[0], &ranges[1]]).unwrap();
    for bad in [
        vec![],
        vec![vec![1.0]],
        vec![vec![], vec![]],
        vec![vec![-0.1], vec![1.1]],
        vec![vec![f64::NAN], vec![0.0]],
        vec![vec![f64::INFINITY], vec![0.0]],
        vec![vec![0.0], vec![0.0]],
        vec![vec![0.4], vec![0.4]],
    ] {
        assert!(matches!(
            evaluator.evaluate(|id| if id == 0 {
                vec![vec![1.0], vec![0.0]]
            } else {
                bad.clone()
            }),
            Err(QualityError::InvalidStrategy { node_id: 1, .. })
        ));
    }
    let policies = uniform(&tree.root, [1, 1]);
    let mut calls = HashMap::<usize, usize>::new();
    evaluator
        .evaluate(|id| {
            *calls.entry(id).or_default() += 1;
            policies[&id].clone()
        })
        .unwrap();
    assert_eq!(calls, HashMap::from([(0, 1), (1, 1)]));
}

#[test]
fn public_river_changes_hidden_range_before_best_response_choice() {
    let initial = board(&["2c", "7d", "9h", "Js"]);
    let combos = [
        vec![combo("Ac", "Ad", 0.7), combo("3c", "3d", 0.3)],
        vec![combo("Kc", "Kd", 1.0)],
    ];
    // IP faces one pot-size bet after the river is public. The river may
    // eliminate only one OOP hand, so treating the posterior as the initial
    // 70/30 mixture would select the wrong response on some public histories.
    let root = Node::Chance {
        children: (0..52)
            .filter(|c| !initial.contains(c))
            .enumerate()
            .map(|(id, c)| {
                (
                    vec![c],
                    Node::Decision {
                        id,
                        player: Player::Ip,
                        actions: vec![Action::Call, Action::Fold],
                        children: vec![
                            Node::Terminal(Terminal::Showdown {
                                invested: [15.0, 15.0],
                            }),
                            Node::Terminal(Terminal::Fold {
                                winner: Player::Oop,
                                invested: [15.0, 5.0],
                            }),
                        ],
                    },
                )
            })
            .collect(),
    };
    let result = ReferenceEvaluator::new(&root, &initial, [&combos[0], &combos[1]])
        .unwrap()
        .evaluate(|_| vec![vec![0.4], vec![0.6]])
        .unwrap();
    let hero = combos[1][0];
    let mut expected_br = 0.0;
    let mut expected_profile = 0.0;
    let mut clairvoyant_br = 0.0;
    let mut choose_call = 0;
    let mut choose_fold = 0;
    for river in
        (0..52).filter(|c| !initial.contains(c) && *c != hero.cards[0] && *c != hero.cards[1])
    {
        let mut completed = initial.clone();
        completed.push(river);
        let mut call = 0.0;
        let mut fold = 0.0;
        for &villain in &combos[0] {
            if river == villain.cards[0] || river == villain.cards[1] {
                continue;
            }
            let probability = villain.weight / 44.0;
            let payoff = match score(hero, &completed).cmp(&score(villain, &completed)) {
                std::cmp::Ordering::Greater => 15.0,
                std::cmp::Ordering::Less => -15.0,
                std::cmp::Ordering::Equal => 0.0,
            };
            call += probability * payoff;
            fold += probability * -5.0;
            clairvoyant_br += probability * payoff.max(-5.0);
        }
        if call > fold {
            choose_call += 1;
        } else {
            choose_fold += 1;
        }
        expected_br += call.max(fold);
        expected_profile += 0.4 * call + 0.6 * fold;
    }
    assert!(
        choose_call > 0 && choose_fold > 0,
        "fixture must require both public-history responses"
    );
    assert!(
        clairvoyant_br - expected_br > 1.0,
        "fixture must penalize private-card clairvoyance"
    );
    near(result.profile_ev[1], expected_profile);
    near(result.best_response_ev[1], expected_br);
}
