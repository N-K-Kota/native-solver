//! Preflop trees: an empty board, betting only, and called pots settled over
//! every five-card runout from the shipped equity table.
use native_solver::solver::quality::ReferenceEvaluator;
use native_solver::{
    all_canonical_hands, average_by_hand, build_preflop_tree, build_tree, Action, Card,
    CardAbstraction, CardSet, Chips, Combo, GameConfig, Node, Player, PreflopConfig, Range,
    Terminal, Trainer, TrainerError, TrainerOptions, Tree,
};

fn half() -> Chips {
    Chips::from_f64(0.5).unwrap()
}

fn combo(first: &str, second: &str, weight: f64) -> Combo {
    Combo::new(first.parse().unwrap(), second.parse().unwrap(), weight)
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

/// The trainer's values against the independent enumeration of every board.
fn assert_matches_reference(trainer: &Trainer) {
    let game = trainer.game();
    let reference = ReferenceEvaluator::new(
        game.tree(),
        game.board(),
        Player::ALL.map(|player| game.combos(player)),
    )
    .unwrap()
    .evaluate(|id| trainer.average_strategy(id))
    .unwrap();
    let production = trainer.exploitability();
    let raw_ev = trainer.evaluate().root_values;
    near(raw_ev[0] / reference.joint_mass, reference.profile_ev[0]);
    near(raw_ev[1] / reference.joint_mass, reference.profile_ev[1]);
    near(production.oop, reference.best_response_ev[0]);
    near(production.ip, reference.best_response_ev[1]);
    near(production.total, reference.exploitability);
    near(reference.profile_ev[0] + reference.profile_ev[1], 0.0);
}

/// A few combos per side, with shared cards, ties, and unequal weights.
fn small_ranges() -> [Vec<Combo>; 2] {
    [
        vec![
            combo("As", "Ah", 1.0),
            combo("Kd", "Qd", 0.5),
            combo("7c", "2d", 1.0),
            combo("Ts", "9s", 0.25),
        ],
        vec![
            combo("As", "Kh", 1.0),
            combo("Qc", "Qh", 1.0),
            combo("Td", "9d", 0.75),
            combo("8c", "3h", 1.0),
        ],
    ]
}

#[test]
fn trained_values_match_the_reference_enumeration() {
    let config = PreflopConfig::new(half(), Chips::whole(1), Chips::whole(12))
        .with_raise_sizes([2.5])
        .with_all_in(true)
        .with_max_raises(2);
    let tree = build_preflop_tree(&config).unwrap();
    let mut trainer = Trainer::new(tree, small_ranges(), TrainerOptions::default()).unwrap();
    // Before training (uniform strategies) and after.
    assert_matches_reference(&trainer);
    trainer.train(300);
    assert_matches_reference(&trainer);
    assert!(trainer.exploitability().total < 0.01);
}

#[test]
fn unequal_investments_at_a_runout_match_the_reference() {
    // Not a legal poker line, but a legal tree: the runout pays the usual
    // showdown rule, so a tie returns each player's own investment.
    let tree = Tree {
        board: Vec::new(),
        root: Node::Decision {
            id: 0,
            player: Player::Ip,
            actions: vec![Action::Fold, Action::Call],
            children: vec![
                Node::Terminal(Terminal::Fold {
                    winner: Player::Oop,
                    invested: [2.0, 1.0],
                }),
                Node::Terminal(Terminal::Showdown {
                    invested: [7.0, 3.0],
                }),
            ],
        },
        num_decision_nodes: 1,
    };
    let mut trainer = Trainer::new(tree, small_ranges(), TrainerOptions::default()).unwrap();
    assert_matches_reference(&trainer);
    trainer.train(50);
    assert_matches_reference(&trainer);
}

fn full_range() -> Vec<Combo> {
    all_canonical_hands()
        .join(",")
        .parse::<Range>()
        .unwrap()
        .combos(CardSet::EMPTY)
}

#[test]
fn push_or_fold_with_ten_big_blinds_converges_to_a_sensible_equilibrium() {
    let config = PreflopConfig::new(half(), Chips::whole(1), Chips::whole(10))
        .with_all_in(true)
        .with_max_raises(1)
        .with_limp(false);
    let tree = build_preflop_tree(&config).unwrap();
    let mut trainer = Trainer::new(
        tree,
        [full_range(), full_range()],
        TrainerOptions::default(),
    )
    .unwrap();
    assert_eq!(trainer.game().combos(Player::Ip).len(), 1326);
    trainer.train(400);
    // Within a hundredth of a big blind of equilibrium.
    let exploitability = trainer.exploitability().total;
    assert!(exploitability < 0.01, "{exploitability}");

    // Node 0: the small blind folds or shoves. Node 1: the big blind folds or calls.
    let by_hand = |node: usize, player: Player, action: usize| {
        let rows = trainer.average_strategy(node);
        let hands = average_by_hand(trainer.game().combos(player), &rows);
        assert_eq!(hands.len(), 169);
        move |hand: &str| {
            hands
                .iter()
                .find(|(name, _)| name == hand)
                .map(|(_, row)| row[action])
                .unwrap()
        }
    };
    let shove = by_hand(0, Player::Ip, 1);
    let call = by_hand(1, Player::Oop, 1);
    for hand in ["AA", "KK", "22", "AKs", "A2o", "KQo", "T9s"] {
        assert!(shove(hand) > 0.99, "{hand} shoves {}", shove(hand));
    }
    for hand in ["72o", "32o", "83o", "T2o"] {
        assert!(shove(hand) < 0.01, "{hand} shoves {}", shove(hand));
    }
    for hand in ["AA", "KK", "99", "AKo", "A9s", "KQs"] {
        assert!(call(hand) > 0.99, "{hand} calls {}", call(hand));
    }
    for hand in ["72o", "32o", "T5o", "J4s", "96o"] {
        assert!(call(hand) < 0.01, "{hand} calls {}", call(hand));
    }

    // The small blind shoves more than half its hands; the big blind calls
    // with fewer than it folds.
    let frequency = |node: usize, player: Player| {
        let rows = trainer.average_strategy(node);
        let combos = trainer.game().combos(player);
        rows[1].iter().sum::<f64>() / combos.len() as f64
    };
    let (shoved, called) = (frequency(0, Player::Ip), frequency(1, Player::Oop));
    assert!((0.5..0.7).contains(&shoved), "{shoved}");
    assert!((0.3..0.5).contains(&called), "{called}");

    // The game is zero-sum.
    let [oop, ip] = trainer.evaluate().root_values;
    assert!((oop + ip).abs() < 1e-6 * oop.abs().max(1.0));
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let solve = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let config = PreflopConfig::new(half(), Chips::whole(1), Chips::whole(25))
                    .with_raise_sizes([2.5, 3.0])
                    .with_all_in(true)
                    .with_max_raises(3);
                let ranges = [
                    "AA,KK,QQ,JJ,TT,AKs,AKo,AQs,KQs,76s,72o",
                    "AA,99,88,AJo,KTs,T9s,54s,83o",
                ]
                .map(|range| range.parse::<Range>().unwrap().combos(CardSet::EMPTY));
                let mut trainer = Trainer::new(
                    build_preflop_tree(&config).unwrap(),
                    ranges,
                    TrainerOptions::default(),
                )
                .unwrap();
                let mut bits: Vec<u64> = trainer.train(20).map(f64::to_bits).to_vec();
                let policy = trainer.policy();
                bits.extend(
                    policy
                        .rows()
                        .iter()
                        .flatten()
                        .flatten()
                        .map(|p| p.to_bits()),
                );
                let exploitability = trainer.exploitability();
                bits.extend([exploitability.oop, exploitability.ip].map(f64::to_bits));
                for node in trainer.evaluate().nodes {
                    assert!(node.equity.is_none());
                    bits.extend(node.ev.iter().map(|ev| ev.to_bits()));
                }
                bits
            })
    };
    let sequential = solve(1);
    assert!(sequential
        .iter()
        .all(|&bits| f64::from_bits(bits).is_finite()));
    assert!(solve(4) == sequential);
}

#[test]
fn preflop_trees_reject_abstraction_and_dealt_cards() {
    let config = PreflopConfig::new(half(), Chips::whole(1), Chips::whole(10)).with_all_in(true);
    let abstraction = CardAbstraction {
        river: Some(4),
        ..CardAbstraction::default()
    };
    let error = Trainer::new(
        build_preflop_tree(&config).unwrap(),
        small_ranges(),
        TrainerOptions {
            abstraction,
            ..TrainerOptions::default()
        },
    )
    .unwrap_err();
    assert_eq!(error, TrainerError::PreflopAbstraction);

    // A chance node on an empty board would deal one card, not a flop.
    let dealing = Tree {
        board: Vec::new(),
        root: Node::Chance {
            children: Card::all()
                .map(|card| {
                    (
                        vec![card],
                        Node::Terminal(Terminal::Showdown {
                            invested: [1.0, 1.0],
                        }),
                    )
                })
                .collect(),
        },
        num_decision_nodes: 0,
    };
    assert!(matches!(
        Trainer::new(dealing, small_ranges(), TrainerOptions::default()),
        Err(TrainerError::InvalidTree { .. })
    ));
}

#[test]
fn postflop_trees_still_need_a_flop_turn_or_river() {
    for cards in [1, 2, 6] {
        let board: Vec<Card> = Card::all().take(cards).collect();
        assert!(build_tree(&GameConfig::new(
            board,
            Chips::whole(100),
            Chips::whole(100)
        ))
        .is_err());
    }
    assert!(build_tree(&GameConfig::new(
        Vec::new(),
        Chips::whole(100),
        Chips::whole(100)
    ))
    .is_err());
}
