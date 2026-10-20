//! Removing forced checks must preserve public chance, payoffs and real policies.
use native_solver::solver::quality::ReferenceEvaluator;
use native_solver::solver::Chips;
use native_solver::solver::TrainerOptions;
use native_solver::solver::{
    build_tree_bounded, Action, GameConfig, Node, Player, SolverAlgorithm, Trainer, UpdateSchedule,
};
use native_solver::{
    card::{card_from_str, set_from_cards},
    range::Range,
};

fn contains_decision(node: &Node) -> bool {
    match node {
        Node::Decision { .. } => true,
        Node::Chance { children } => children.iter().any(|(_, n)| contains_decision(n)),
        _ => false,
    }
}

/// Recreate the old forced OOP/IP checks on each all-in future street.
/// This test-only tree keeps the same chance classes and investments.
fn restore_forced_checks(node: &mut Node) {
    match node {
        Node::Decision { children, .. } => {
            for child in children {
                restore_forced_checks(child);
            }
        }
        Node::Chance { children } => {
            for (_, child) in children {
                let forced = !contains_decision(child);
                restore_forced_checks(child);
                if forced {
                    *child = Node::Decision {
                        id: 0,
                        player: Player::Oop,
                        actions: vec![Action::Check],
                        children: vec![Node::Decision {
                            id: 0,
                            player: Player::Ip,
                            actions: vec![Action::Check],
                            children: vec![child.clone()],
                        }],
                    };
                }
            }
        }
        _ => {}
    }
}

fn renumber(node: &mut Node, next: &mut usize) {
    match node {
        Node::Decision { id, children, .. } => {
            *id = *next;
            *next += 1;
            for child in children {
                renumber(child, next);
            }
        }
        Node::Chance { children } => {
            for (_, child) in children {
                renumber(child, next);
            }
        }
        _ => {}
    }
}

fn compare_real_policies(a: &Node, b: &Node, new: &Trainer, old: &Trainer) {
    if let Node::Decision {
        actions, children, ..
    } = b
    {
        if actions.len() == 1 && matches!(actions[0], Action::Check) {
            compare_real_policies(a, &children[0], new, old);
            return;
        }
    }
    match (a, b) {
        (
            Node::Decision {
                id: ai,
                actions: aa,
                children: ac,
                ..
            },
            Node::Decision {
                id: bi,
                actions: ba,
                children: bc,
                ..
            },
        ) => {
            assert_eq!(format!("{aa:?}"), format!("{ba:?}"));
            assert_eq!(new.average_strategy(*ai), old.average_strategy(*bi));
            for (a, b) in ac.iter().zip(bc) {
                compare_real_policies(a, b, new, old);
            }
        }
        (Node::Chance { children: ac }, Node::Chance { children: bc }) => {
            assert_eq!(ac.len(), bc.len());
            for ((am, a), (bm, b)) in ac.iter().zip(bc) {
                assert_eq!(am, bm);
                compare_real_policies(a, b, new, old);
            }
        }
        (Node::Terminal(a), Node::Terminal(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
        _ => panic!("tree mismatch: {a:?} versus {b:?}"),
    }
}

#[test]
fn all_in_runouts_preserve_strategy_values_and_reference_quality() {
    for board_text in ["Ks7d2c", "Ks7d2c9h", "Ks7s2s", "Ks7s2s9h"] {
        for iso in [false, true] {
            for (algorithm, schedule) in [
                SolverAlgorithm::Cfr,
                SolverAlgorithm::CfrPlus,
                SolverAlgorithm::Dcfr,
            ]
            .into_iter()
            .flat_map(|a| {
                [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating].map(|s| (a, s))
            }) {
                let board: Vec<_> = board_text
                    .as_bytes()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| card_from_str(std::str::from_utf8(c).unwrap()).unwrap())
                    .collect();
                let cfg = GameConfig {
                    board: board.clone(),
                    pot: Chips::whole(100),
                    effective_stack: Chips::whole(100),
                    bet_sizes: vec![1.0],
                    max_raises: 0,
                    suit_isomorphism: iso,
                };
                let tree = build_tree_bounded(&cfg, 20_000).unwrap();
                let mut legacy = build_tree_bounded(&cfg, 20_000).unwrap();
                restore_forced_checks(&mut legacy.root);
                let mut count = 0;
                renumber(&mut legacy.root, &mut count);
                legacy.num_decision_nodes = count;
                assert!(legacy.num_decision_nodes > tree.num_decision_nodes);
                // A flop with this simple tree used to exceed the service's node cap.
                if board.len() == 3 && !iso {
                    assert_eq!((tree.num_decision_nodes, count), (9608, 28620));
                }
                let combos = [
                    Range::parse("AA:0.5,KK")
                        .unwrap()
                        .combos(set_from_cards(&board)),
                    Range::parse("QQ,JJ:0.25")
                        .unwrap()
                        .combos(set_from_cards(&board)),
                ];
                let mut new = Trainer::new(
                    tree,
                    board.clone(),
                    combos.clone(),
                    TrainerOptions {
                        algorithm,
                        schedule,
                        ..TrainerOptions::default()
                    },
                )
                .unwrap();
                let mut old = Trainer::new(
                    legacy,
                    board.clone(),
                    combos.clone(),
                    TrainerOptions {
                        algorithm,
                        schedule,
                        ..TrainerOptions::default()
                    },
                )
                .unwrap();
                assert_eq!(new.train(3), old.train(3));
                compare_real_policies(new.game().tree(), old.game().tree(), &new, &old);
                let q = new.exploitability();
                let previous = old.exploitability();
                assert_eq!(
                    [q.oop, q.ip, q.total],
                    [previous.oop, previous.ip, previous.total]
                );
                assert_eq!(new.evaluate().1, old.evaluate().1);
                if !iso {
                    let reference = ReferenceEvaluator::new(
                        new.game().tree(),
                        &board,
                        [&combos[0], &combos[1]],
                    )
                    .unwrap();
                    let independent = reference.evaluate(|id| new.average_strategy(id)).unwrap();
                    assert!((independent.exploitability - q.total).abs() < 1e-9);
                }
            }
        }
    }
}

#[test]
fn uncalled_all_in_still_requires_opponent_fold_or_call_and_ids_are_dense() {
    let cfg = GameConfig {
        board: ["Ks", "7d", "2c", "9h"]
            .map(|c| card_from_str(c).unwrap())
            .to_vec(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(100),
        bet_sizes: vec![1.0],
        max_raises: 0,
        suit_isomorphism: false,
    };
    let tree = build_tree_bounded(&cfg, 196).unwrap();
    assert_eq!(tree.num_decision_nodes, 196);
    assert!(build_tree_bounded(&cfg, 195).is_err());
    let Node::Decision { children, .. } = &tree.root else {
        panic!()
    };
    let Node::Decision {
        actions, children, ..
    } = &children[1]
    else {
        panic!("uncalled bet must remain a decision")
    };
    assert!(matches!(actions.as_slice(), [Action::Fold, Action::Call]));
    assert!(!contains_decision(&children[1]));
    let mut duplicate = tree.root.clone();
    let mut count = 0;
    renumber(&mut duplicate, &mut count);
    assert_eq!(format!("{:?}", tree.root), format!("{duplicate:?}"));
}
