//! Invalid inputs are reported as errors by `build_tree` and `Trainer::new`
//! instead of producing NaN results, silently wrong values, or panics.
use native_solver::solver::{
    build_tree, build_tree_bounded, Action, BoardError, CardAbstraction, Chips, GameConfig, Node,
    Player, Terminal, Trainer, TrainerError, TrainerOptions, Tree, TreeError,
};
use native_solver::{card_from_str, set_from_cards, Card, Combo, Range};

fn board(names: &[&str]) -> Vec<Card> {
    names.iter().map(|s| card_from_str(s).unwrap()).collect()
}

fn river() -> Vec<Card> {
    board(&["Ks", "Th", "7s", "4d", "2s"])
}

fn config(board: Vec<Card>) -> GameConfig {
    GameConfig {
        board,
        pot: Chips::whole(100),
        effective_stack: Chips::whole(100),
        bet_sizes: vec![0.5, 1.0],
        max_raises: 1,
        suit_isomorphism: false,
    }
}

fn combos(oop: &str, ip: &str, board: &[Card]) -> [Vec<Combo>; 2] {
    let dead = set_from_cards(board);
    [
        Range::parse(oop).unwrap().combos(dead),
        Range::parse(ip).unwrap().combos(dead),
    ]
}

fn combo(a: &str, b: &str, weight: f64) -> Combo {
    Combo::new(card_from_str(a).unwrap(), card_from_str(b).unwrap(), weight)
}

/// `Trainer::new` on a river tree for `river()` with the given combos.
fn river_trainer(combos: [Vec<Combo>; 2]) -> Result<Trainer, TrainerError> {
    let board = river();
    Trainer::new(
        build_tree(&config(board.clone())).unwrap(),
        board,
        combos,
        TrainerOptions::default(),
    )
}

fn showdown() -> Node {
    Node::Terminal(Terminal::Showdown {
        invested: [50.0, 50.0],
    })
}

#[test]
fn build_tree_rejects_illegal_boards() {
    let cases = [
        (board(&["Ks", "Th"]), BoardError::WrongSize { len: 2 }),
        (
            board(&["Ks", "Th", "7s", "4d", "2s", "3c"]),
            BoardError::WrongSize { len: 6 },
        ),
        (vec![0, 1, 52], BoardError::InvalidCard { card: 52 }),
        (
            board(&["Ks", "Th", "Ks"]),
            BoardError::DuplicateCard {
                card: card_from_str("Ks").unwrap(),
            },
        ),
    ];
    for (cards, expected) in cases {
        let error = build_tree(&config(cards)).err();
        assert_eq!(error, Some(TreeError::Board(expected)));
    }
}

#[test]
fn build_tree_rejects_illegal_chip_amounts_and_bet_sizes() {
    for pot in [Chips::ZERO, Chips::whole(-10)] {
        let cfg = GameConfig {
            pot,
            ..config(river())
        };
        assert_eq!(build_tree(&cfg).err(), Some(TreeError::NonPositivePot));
    }

    let cfg = GameConfig {
        effective_stack: Chips::whole(-1),
        ..config(river())
    };
    assert_eq!(build_tree(&cfg).err(), Some(TreeError::NegativeStack));

    for size in [0.0, -0.5, f64::NAN, f64::INFINITY] {
        let cfg = GameConfig {
            bet_sizes: vec![0.5, size],
            ..config(river())
        };
        assert_eq!(
            build_tree(&cfg).err(),
            Some(TreeError::InvalidBetSize { index: 1 }),
            "bet size {size}"
        );
    }

    // The bounded builder validates before spending any budget.
    let cfg = GameConfig {
        pot: Chips::ZERO,
        ..config(river())
    };
    assert_eq!(
        build_tree_bounded(&cfg, 0).err(),
        Some(TreeError::NonPositivePot)
    );
}

#[test]
fn build_tree_accepts_degenerate_but_legal_configs() {
    // No bet sizes: a check-down game.
    let cfg = GameConfig {
        bet_sizes: vec![],
        ..config(river())
    };
    assert_eq!(build_tree(&cfg).unwrap().num_decision_nodes, 2);

    // No stack behind: nobody can bet, so the root is the showdown.
    let cfg = GameConfig {
        effective_stack: Chips::ZERO,
        ..config(river())
    };
    assert_eq!(build_tree(&cfg).unwrap().num_decision_nodes, 0);
}

#[test]
fn trainer_rejects_an_illegal_initial_board() {
    let tree = build_tree(&config(river())).unwrap();
    let error = Trainer::new(
        tree,
        board(&["Ks", "Ks", "7s", "4d", "2s"]),
        combos("AA", "QQ", &river()),
        TrainerOptions::default(),
    )
    .err();
    assert_eq!(
        error,
        Some(TrainerError::Board(BoardError::DuplicateCard {
            card: card_from_str("Ks").unwrap()
        }))
    );
}

#[test]
fn trainer_rejects_a_tree_built_for_another_street() {
    let turn = board(&["Ks", "Th", "7s", "4d"]);

    // River tree, turn board: the showdown would happen on four cards.
    let error = Trainer::new(
        build_tree(&config(river())).unwrap(),
        turn.clone(),
        combos("AA", "QQ", &turn),
        TrainerOptions::default(),
    )
    .err();
    assert_eq!(error, Some(TrainerError::TreeBoardMismatch));

    // Turn tree, river board: the chance node would deal a sixth card.
    let error = Trainer::new(
        build_tree(&config(turn)).unwrap(),
        river(),
        combos("AA", "QQ", &river()),
        TrainerOptions::default(),
    )
    .err();
    assert_eq!(error, Some(TrainerError::TreeBoardMismatch));
}

#[test]
fn trainer_rejects_malformed_hand_built_trees() {
    let decision = |id, actions: Vec<Action>, children: Vec<Node>| Node::Decision {
        id,
        player: Player::Oop,
        actions,
        children,
    };
    let trees = [
        // Duplicate decision id.
        Tree {
            root: decision(
                0,
                vec![Action::Check, Action::Bet(Chips::whole(10))],
                vec![
                    showdown(),
                    decision(0, vec![Action::Check], vec![showdown()]),
                ],
            ),
            num_decision_nodes: 2,
        },
        // Id outside 0..num_decision_nodes.
        Tree {
            root: decision(3, vec![Action::Check], vec![showdown()]),
            num_decision_nodes: 1,
        },
        // A declared id that no node uses.
        Tree {
            root: decision(0, vec![Action::Check], vec![showdown()]),
            num_decision_nodes: 2,
        },
        // More actions than children.
        Tree {
            root: decision(
                0,
                vec![Action::Check, Action::Bet(Chips::whole(10))],
                vec![showdown()],
            ),
            num_decision_nodes: 1,
        },
        // Negative terminal investment.
        Tree {
            root: Node::Terminal(Terminal::Showdown {
                invested: [-1.0, 50.0],
            }),
            num_decision_nodes: 0,
        },
    ];
    for tree in trees {
        let board = river();
        let error = Trainer::new(
            tree,
            board.clone(),
            combos("AA", "QQ", &board),
            TrainerOptions::default(),
        )
        .err();
        assert!(
            matches!(error, Some(TrainerError::InvalidTree { .. })),
            "{error:?}"
        );
    }

    // A chance node that skips an undealt card.
    let turn = board(&["Ks", "Th", "7s", "4d"]);
    let dead = set_from_cards(&turn);
    let children = (0..52u8)
        .filter(|&c| dead & (1 << c) == 0)
        .skip(1)
        .map(|c| (vec![c], showdown()))
        .collect();
    let tree = Tree {
        root: Node::Chance { children },
        num_decision_nodes: 0,
    };
    let error = Trainer::new(
        tree,
        turn.clone(),
        combos("AA", "QQ", &turn),
        TrainerOptions::default(),
    )
    .err();
    assert!(
        matches!(error, Some(TrainerError::InvalidTree { .. })),
        "{error:?}"
    );
}

#[test]
fn trainer_rejects_empty_ranges() {
    let [oop, ip] = combos("AA", "QQ", &river());
    assert_eq!(
        river_trainer([Vec::new(), ip.clone()]).err(),
        Some(TrainerError::EmptyRange {
            player: Player::Oop
        })
    );
    let zero_weight = oop.iter().map(|c| Combo { weight: 0.0, ..*c }).collect();
    assert_eq!(
        river_trainer([oop, zero_weight]).err(),
        Some(TrainerError::EmptyRange { player: Player::Ip })
    );
}

#[test]
fn trainer_rejects_invalid_combos_and_weights() {
    let ip = vec![combo("Qc", "Qd", 1.0)];
    let invalid_cards = [
        Combo {
            cards: [0, 52],
            weight: 1.0,
        },
        combo("Ac", "Ac", 1.0),
    ];
    for bad in invalid_cards {
        assert_eq!(
            river_trainer([vec![combo("Ad", "Ah", 1.0), bad], ip.clone()]).err(),
            Some(TrainerError::InvalidCombo {
                player: Player::Oop,
                index: 1
            })
        );
    }

    for weight in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            river_trainer([vec![combo("Ad", "Ah", weight)], ip.clone()]).err(),
            Some(TrainerError::InvalidWeight {
                player: Player::Oop,
                index: 0
            }),
            "weight {weight}"
        );
    }

    // The same two cards in the reverse order are the same combo.
    assert_eq!(
        river_trainer([
            vec![combo("Ad", "Ah", 1.0), combo("Ah", "Ad", 1.0)],
            ip.clone()
        ])
        .err(),
        Some(TrainerError::DuplicateCombo {
            player: Player::Oop,
            index: 1
        })
    );
}

#[test]
fn trainer_rejects_combos_that_use_board_cards() {
    // Without the board as dead cards, KK keeps combos that hold the Ks.
    let oop = Range::parse("KK").unwrap().combos(0);
    let ip = Range::parse("AA").unwrap().combos(set_from_cards(&river()));
    let error = river_trainer([oop, ip]).err();
    assert!(
        matches!(
            error,
            Some(TrainerError::ComboConflictsWithBoard {
                player: Player::Oop,
                ..
            })
        ),
        "{error:?}"
    );
}

#[test]
fn trainer_rejects_ranges_that_can_never_be_dealt_together() {
    let error = river_trainer([vec![combo("Ac", "Ad", 1.0)], vec![combo("Ac", "Ah", 1.0)]]).err();
    assert_eq!(error, Some(TrainerError::NoCompatibleDeals));

    // A zero-weight combo does not make a deal possible.
    let error = river_trainer([
        vec![combo("Ac", "Ad", 1.0), combo("Qc", "Qd", 0.0)],
        vec![combo("Ac", "Ah", 1.0)],
    ])
    .err();
    assert_eq!(error, Some(TrainerError::NoCompatibleDeals));
}

#[test]
fn trainer_rejects_zero_bucket_abstraction() {
    let board = river();
    let error = Trainer::new(
        build_tree(&config(board.clone())).unwrap(),
        board.clone(),
        combos("AA,KQs", "QQ,JJ", &board),
        TrainerOptions {
            abstraction: CardAbstraction {
                river: Some(0),
                ..CardAbstraction::default()
            },
            ..TrainerOptions::default()
        },
    )
    .err();
    assert_eq!(error, Some(TrainerError::ZeroBuckets));
}

#[test]
fn trainer_rejects_asymmetric_ranges_on_suit_reduced_trees() {
    let turn = board(&["Js", "9s", "4d", "2d"]);
    let cfg = GameConfig {
        suit_isomorphism: true,
        ..config(turn.clone())
    };
    let [mut oop, ip] = combos("AKs,AQs", "AKs,AQs", &turn);
    oop[0].weight = 0.25;
    let error = Trainer::new(
        build_tree(&cfg).unwrap(),
        turn,
        [oop, ip],
        TrainerOptions::default(),
    )
    .err();
    assert_eq!(
        error,
        Some(TrainerError::AsymmetricRange {
            player: Player::Oop
        })
    );
}

#[test]
fn errors_explain_what_is_wrong() {
    let tree_error = build_tree(&config(board(&["Ks", "Th"]))).err().unwrap();
    assert_eq!(
        tree_error.to_string(),
        "invalid board: board has 2 cards; expected 3, 4, or 5"
    );
    assert!(std::error::Error::source(&tree_error).is_some());

    let trainer_error = river_trainer([Vec::new(), vec![combo("Qc", "Qd", 1.0)]])
        .err()
        .unwrap();
    assert_eq!(
        trainer_error.to_string(),
        "Oop range has no combo with positive weight"
    );
}
