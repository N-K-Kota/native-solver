//! Public accessors that other tests only use indirectly: combo helpers,
//! per-node boards, bucket assignments, and the trainer's settings.
use native_solver::solver::{
    build_tree, CardAbstraction, Chips, GameConfig, Player, SolverAlgorithm, Trainer,
    TrainerOptions, UpdateSchedule,
};
use native_solver::{card_from_str, evaluate_7, set_from_cards, Card, Combo, Range};

fn board(names: &[&str]) -> Vec<Card> {
    names.iter().map(|s| card_from_str(s).unwrap()).collect()
}

fn trainer(board: &[Card], oop: &str, ip: &str, options: TrainerOptions) -> Trainer {
    let cfg = GameConfig {
        board: board.to_vec(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(100),
        bet_sizes: vec![1.0],
        max_raises: 0,
        suit_isomorphism: false,
    };
    let dead = set_from_cards(board);
    let combos = [
        Range::parse(oop).unwrap().combos(dead),
        Range::parse(ip).unwrap().combos(dead),
    ];
    Trainer::new(build_tree(&cfg).unwrap(), combos, options).unwrap()
}

#[test]
fn combo_card_helpers() {
    let c = |s| card_from_str(s).unwrap();
    let combo = Combo::new(c("Kd"), c("2c"), 0.5);
    assert_eq!(combo.mask(), set_from_cards(&[c("2c"), c("Kd")]));
    assert_eq!(
        combo.sorted_cards(),
        Combo::new(c("2c"), c("Kd"), 1.0).sorted_cards()
    );
    assert!(combo.sorted_cards().0 < combo.sorted_cards().1);
    assert!(combo.conflicts_with(set_from_cards(&[c("Kd"), c("Ah")])));
    assert!(!combo.conflicts_with(set_from_cards(&[c("Kh"), c("2d")])));
    assert!(!combo.conflicts_with(0));
}

#[test]
fn players_alternate() {
    assert_eq!(Player::Oop.other(), Player::Ip);
    assert_eq!(Player::Ip.other(), Player::Oop);
    assert_eq!([Player::Oop.index(), Player::Ip.index()], [0, 1]);
}

#[test]
fn trainer_reports_its_settings_and_progress() {
    let river = board(&["Ks", "Th", "7s", "4d", "2s"]);
    let options = TrainerOptions {
        algorithm: SolverAlgorithm::Dcfr,
        schedule: UpdateSchedule::Alternating,
        abstraction: CardAbstraction {
            river: Some(2),
            ..CardAbstraction::default()
        },
    };
    let mut trainer = trainer(&river, "AA,KQs", "QQ,JJ", options);
    assert_eq!(trainer.algorithm(), SolverAlgorithm::Dcfr);
    assert_eq!(trainer.update_schedule(), UpdateSchedule::Alternating);
    let abstraction = trainer.game().abstraction();
    assert_eq!(
        [abstraction.flop, abstraction.turn, abstraction.river],
        [None, None, Some(2)]
    );

    assert_eq!(trainer.iterations_completed(), 0);
    trainer.train(3);
    trainer.train(4);
    // An alternating iteration (OOP pass, then IP pass) counts once.
    assert_eq!(trainer.iterations_completed(), 7);
}

#[test]
fn node_board_is_the_public_board_at_each_decision() {
    let turn = board(&["Ks", "Th", "7s", "4d"]);
    let trainer = trainer(&turn, "AA,KQs", "QQ,JJ", TrainerOptions::default());
    let game = trainer.game();
    assert!(game.num_decision_nodes() > 0);
    assert_eq!(game.node_board(0), turn.as_slice());

    let mut river_cards = std::collections::HashSet::new();
    for id in 0..game.num_decision_nodes() {
        let node_board = game.node_board(id);
        assert_eq!(&node_board[..4], turn.as_slice(), "node {id}");
        match node_board.len() {
            4 => {}
            5 => {
                assert!(!turn.contains(&node_board[4]), "node {id}");
                river_cards.insert(node_board[4]);
            }
            len => panic!("node {id} has a {len}-card board"),
        }
    }
    // Without suit reduction, every undealt card gets its own river nodes.
    assert_eq!(river_cards.len(), 48);
}

#[test]
fn bucket_assignments_follow_hand_strength() {
    let river = board(&["Ks", "Th", "7s", "4d", "2s"]);
    let unbucketed = trainer(
        &river,
        "AA,KK,QQ,AKs",
        "JJ,TT,99",
        TrainerOptions::default(),
    );
    assert!(unbucketed.game().bucket_assignments().is_empty());

    let buckets = 3;
    let options = TrainerOptions {
        abstraction: CardAbstraction {
            river: Some(buckets),
            ..CardAbstraction::default()
        },
        ..TrainerOptions::default()
    };
    let bucketed = trainer(&river, "AA,KK,QQ,AKs", "JJ,TT,99", options);
    let game = bucketed.game();
    let assignments = game.bucket_assignments();
    assert_eq!(assignments.len(), 1);
    let (assigned_board, maps) = assignments[0];
    assert_eq!(assigned_board, river.as_slice());

    for (player, map) in Player::ALL.into_iter().zip(maps) {
        let combos = game.combos(player);
        assert_eq!(map.len(), combos.len());
        assert!(map.iter().all(|&bucket| bucket < buckets));
        let score = |c: &Combo| {
            evaluate_7([
                c.cards[0], c.cards[1], river[0], river[1], river[2], river[3], river[4],
            ])
        };
        for i in 0..combos.len() {
            for j in 0..combos.len() {
                if score(&combos[i]) < score(&combos[j]) {
                    assert!(map[i] <= map[j], "player {player:?}: {i} vs {j}");
                }
                if score(&combos[i]) == score(&combos[j]) {
                    assert_eq!(map[i], map[j], "player {player:?}: tie {i} vs {j}");
                }
            }
        }
    }
    // OOP's four hand classes (AA, KK set, QQ, AK pair) cannot share one bucket.
    assert!(maps[0].iter().max() > maps[0].iter().min());
}

#[test]
fn river_buckets_on_a_flop_tree_only_bucket_river_boards() {
    // Regression: river-only bucketing used to give turn decisions zero
    // buckets, which panicked with an index out of bounds.
    let flop = board(&["2h", "7h", "9h"]);
    let dead = set_from_cards(&flop);
    for suit_isomorphism in [false, true] {
        let cfg = GameConfig {
            board: flop.clone(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(50),
            bet_sizes: vec![1.0],
            max_raises: 0,
            suit_isomorphism,
        };
        let combos = [
            Range::parse("AA,KK").unwrap().combos(dead),
            Range::parse("QQ,AKs").unwrap().combos(dead),
        ];
        let options = TrainerOptions {
            abstraction: CardAbstraction {
                river: Some(2),
                ..CardAbstraction::default()
            },
            ..TrainerOptions::default()
        };
        let mut trainer = Trainer::new(build_tree(&cfg).unwrap(), combos, options).unwrap();
        trainer.train(2);
        assert!(trainer.exploitability().total.is_finite());
        let assignments = trainer.game().bucket_assignments();
        assert!(!assignments.is_empty());
        assert!(assignments.iter().all(|(board, _)| board.len() == 5));
    }
}
