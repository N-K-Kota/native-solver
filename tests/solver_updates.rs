//! Regression and convergence checks for the trainer's update algorithms.
use native_solver::solver::Chips;
use native_solver::solver::TrainerOptions;
use native_solver::solver::{
    build_tree, CardAbstraction, GameConfig, Node, SolverAlgorithm, Trainer, UpdateSchedule,
};
use native_solver::{card_from_str, evaluate_7, Card, Combo};

fn card(s: &str) -> Card {
    card_from_str(s).unwrap()
}
fn river_fixture(algorithm: SolverAlgorithm, bucketed: bool) -> (Trainer, usize) {
    scheduled_fixture(algorithm, bucketed, UpdateSchedule::Simultaneous)
}
fn scheduled_fixture(
    algorithm: SolverAlgorithm,
    bucketed: bool,
    schedule: UpdateSchedule,
) -> (Trainer, usize) {
    let board = ["2h", "7c", "9d", "Jc", "4s"].map(card).to_vec();
    let tree = build_tree(&GameConfig {
        board: board.clone(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(200),
        bet_sizes: vec![0.5, 1.0],
        max_raises: 1,
        suit_isomorphism: false,
    })
    .unwrap();
    let nodes = tree.num_decision_nodes;
    let combos = [
        vec![
            Combo::new(card("Jh"), card("Jd"), 0.75),
            Combo::new(card("Kc"), card("Qd"), 1.0),
            Combo::new(card("Ks"), card("Qh"), 0.5),
        ],
        vec![Combo::new(card("9c"), card("9s"), 1.0)],
    ];
    let abstraction = CardAbstraction {
        river: bucketed.then_some(2),
        ..CardAbstraction::default()
    };
    (
        Trainer::new(
            tree,
            board,
            combos,
            TrainerOptions {
                algorithm,
                schedule,
                abstraction,
            },
        )
        .unwrap(),
        nodes,
    )
}

fn trajectory_fingerprint(trainer: &mut Trainer, nodes: usize) -> Vec<u64> {
    let mut fingerprints = Vec::new();
    for batch in [1, 1, 1, 4, 16, 41] {
        let mut hash = 0xcbf29ce484222325u64;
        let mut absorb = |value: f64| {
            for byte in value.to_bits().to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
            }
        };
        for value in trainer.train(batch) {
            absorb(value);
        }
        for node in 0..nodes {
            for row in trainer.average_strategy(node) {
                for value in row {
                    absorb(value);
                }
            }
        }
        fingerprints.push(hash);
    }
    fingerprints
}

// Frozen from the pre-policy-refactor implementation, not recomputed by a
// second entry point into the new algorithm. Each fingerprint includes every
// strategy entry and both training-return values, with exact f64 bit equality.
#[test]
fn legacy_trajectories_match_pre_refactor_bit_for_bit() {
    let cases = [
        (
            SolverAlgorithm::Cfr,
            false,
            [
                9846889724022330851,
                5372216759972904762,
                15167244231596758895,
                16575681813458622309,
                17874133840296056996,
                9120114657009126904,
            ],
        ),
        (
            SolverAlgorithm::Cfr,
            true,
            [
                9846889724022330851,
                6785688272709962706,
                6374937423166547191,
                5680045390404753521,
                15478301554639931763,
                11212587698736649502,
            ],
        ),
        (
            SolverAlgorithm::CfrPlus,
            false,
            [
                9846889724022330851,
                5372216759972904762,
                4787141966008914913,
                797257280035474678,
                18219019228296156926,
                4101240763380972683,
            ],
        ),
        (
            SolverAlgorithm::CfrPlus,
            true,
            [
                9846889724022330851,
                6785688272709962706,
                15268233800149601593,
                10582716865448510774,
                14388008353551589265,
                15104751954510384905,
            ],
        ),
    ];
    for (algorithm, bucketed, expected) in cases {
        let (mut trainer, nodes) = river_fixture(algorithm, bucketed);
        assert_eq!(
            trajectory_fingerprint(&mut trainer, nodes),
            expected,
            "algorithm={algorithm:?}, bucketed={bucketed}"
        );
        assert_eq!(trainer.iterations_completed(), 64);
    }
}

const ALGORITHMS: [SolverAlgorithm; 3] = [
    SolverAlgorithm::Cfr,
    SolverAlgorithm::CfrPlus,
    SolverAlgorithm::Dcfr,
];

#[test]
fn chunked_training_is_bit_identical_to_one_batch() {
    for (algorithm, schedule) in ALGORITHMS
        .into_iter()
        .flat_map(|a| [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating].map(|s| (a, s)))
    {
        for bucketed in [false, true] {
            let (mut whole, nodes) = scheduled_fixture(algorithm, bucketed, schedule);
            let (mut chunks, _) = scheduled_fixture(algorithm, bucketed, schedule);
            assert_eq!(whole.update_schedule(), schedule);
            assert_eq!(whole.algorithm(), algorithm);
            assert_eq!(chunks.train(0), [0.0, 0.0]);
            assert_eq!(chunks.iterations_completed(), 0);
            let expected = whole.train(137);
            chunks.train(1);
            chunks.train(0);
            chunks.train(6);
            chunks.train(30);
            let actual = chunks.train(100);
            assert_eq!(actual, expected, "{algorithm:?}, bucketed={bucketed}");
            assert_eq!(chunks.iterations_completed(), 137);
            assert_eq!(whole.iterations_completed(), 137);
            for id in 0..nodes {
                assert_eq!(
                    chunks.average_strategy(id),
                    whole.average_strategy(id),
                    "{algorithm:?}, bucketed={bucketed}, node={id}"
                );
            }
            assert_eq!(chunks.evaluate().1, whole.evaluate().1);
            assert_eq!(chunks.exploitability().total, whole.exploitability().total);
            // Equal averages alone do not guarantee equal hidden regrets.
            assert_eq!(chunks.train(3), whole.train(3));
        }
    }
}

type WeightedCombos = [Vec<Combo>; 2];

fn kuhn_equivalent_config() -> (GameConfig, WeightedCombos) {
    // Three disjoint pocket pairs represent Kuhn's J/Q/K. Giving both
    // players the same three combos excludes identical rank deals by card
    // removal, leaving exactly Kuhn's six equiprobable legal private deals.
    let board = ["2h", "3c", "7d", "8h", "9c"].map(card).to_vec();
    let combos = vec![
        Combo::new(card("Js"), card("Jd"), 1.0),
        Combo::new(card("Qs"), card("Qd"), 1.0),
        Combo::new(card("Ks"), card("Kd"), 1.0),
    ];
    let score = |c: Combo| {
        evaluate_7([
            c.cards[0], c.cards[1], board[0], board[1], board[2], board[3], board[4],
        ])
    };
    assert!(score(combos[0]) < score(combos[1]));
    assert!(score(combos[1]) < score(combos[2]));
    let config = GameConfig {
        board,
        pot: Chips::whole(2),
        effective_stack: Chips::whole(1),
        bet_sizes: vec![0.5],
        max_raises: 0,
        suit_isomorphism: false,
    };
    (config, [combos.clone(), combos])
}

#[test]
fn legacy_boolean_constructor_keeps_its_algorithm_and_trajectory() {
    let (config, combos) = kuhn_equivalent_config();
    for (plus, algorithm) in [
        (false, SolverAlgorithm::Cfr),
        (true, SolverAlgorithm::CfrPlus),
    ] {
        let mut legacy = Trainer::new(
            build_tree(&config).unwrap(),
            config.board.clone(),
            combos.clone(),
            TrainerOptions {
                algorithm: if plus {
                    SolverAlgorithm::CfrPlus
                } else {
                    SolverAlgorithm::Cfr
                },
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        let mut explicit = Trainer::new(
            build_tree(&config).unwrap(),
            config.board.clone(),
            combos.clone(),
            TrainerOptions {
                algorithm,
                ..TrainerOptions::default()
            },
        )
        .unwrap();
        assert_eq!(legacy.algorithm(), algorithm);
        for iterations in [1, 2, 5, 31] {
            assert_eq!(legacy.train(iterations), explicit.train(iterations));
            for id in 0..4 {
                assert_eq!(legacy.average_strategy(id), explicit.average_strategy(id));
            }
        }
    }
}

#[test]
fn every_algorithm_converges_to_known_kuhn_value_and_policy() {
    // The actual vectorized NLHE Trainer, not the separate toy KuhnTrainer,
    // must recover Kuhn's -1/18 chip value on this exactly equivalent game.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let (config, combos) = kuhn_equivalent_config();
        for (algorithm, schedule) in ALGORITHMS.into_iter().flat_map(|a| {
            [UpdateSchedule::Simultaneous, UpdateSchedule::Alternating].map(|s| (a, s))
        }) {
            let tree = build_tree(&config).unwrap();
            let Node::Decision {
                id: root_id,
                children,
                ..
            } = &tree.root
            else {
                panic!("Kuhn must start at a decision");
            };
            let root_id = *root_id;
            let Node::Decision {
                id: after_check, ..
            } = &children[0]
            else {
                panic!("IP must act after a check");
            };
            let after_check = *after_check;
            let Node::Decision { id: facing_bet, .. } = &children[1] else {
                panic!("IP must act facing a bet");
            };
            let facing_bet = *facing_bet;
            let mut trainer = Trainer::new(
                tree,
                config.board.clone(),
                combos.clone(),
                TrainerOptions {
                    algorithm,
                    schedule,
                    ..TrainerOptions::default()
                },
            )
            .unwrap();
            trainer.train(20_000);
            // evaluate() returns raw joint-range weighted totals. There are
            // six unit-weight legal deals, so divide by six for chips/deal.
            let ev = trainer.evaluate().1[0] / 6.0;
            let exploitability = trainer.exploitability().total;
            eprintln!("{algorithm:?}: value={ev}, exploitability={exploitability}");
            assert!((ev + 1.0 / 18.0).abs() < 0.001, "{algorithm:?}: EV={ev}");
            // Exploitability is (BR_OOP + BR_IP)/2 in chips/legal deal.
            assert!(
                exploitability < 0.01,
                "{algorithm:?}: exploitability={exploitability}"
            );
            let root = trainer.average_strategy(root_id);
            assert!(root[1][1] < 0.02, "{algorithm:?}: Q should check");
            assert!(
                (root[1][2] - 3.0 * root[1][0]).abs() < 0.03,
                "{algorithm:?}: Kuhn K/J opening ratio: {root:?}"
            );
            let check = trainer.average_strategy(after_check);
            assert!(
                (check[1][0] - 1.0 / 3.0).abs() < 0.02,
                "{algorithm:?}: IP J bluff frequency: {check:?}"
            );
            assert!(check[1][1] < 0.02 && check[1][2] > 0.98);
            let response = trainer.average_strategy(facing_bet);
            assert!(response[1][0] < 0.02 && response[1][2] > 0.98);
            assert!(
                (response[1][1] - 1.0 / 3.0).abs() < 0.02,
                "{algorithm:?}: IP Q call frequency: {response:?}"
            );
        }
    });
}
