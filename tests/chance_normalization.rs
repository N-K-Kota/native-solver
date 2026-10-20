use native_solver::solver::Chips;
use native_solver::solver::{build_tree, GameConfig, Trainer};
use native_solver::solver::{SolverAlgorithm, TrainerOptions};
use native_solver::{evaluate_7, Card, CardSet, Combo};

fn card(s: &str) -> Card {
    s.parse::<Card>().unwrap()
}

fn check_forced_showdown(board_text: &[&str]) {
    let board: Vec<Card> = board_text.iter().map(|s| card(s)).collect();
    let hero = Combo::new(card("Jc"), card("Jd"), 1.0);
    let villain = Combo::new(card("3c"), card("3d"), 1.0);
    let mut dead = board.clone();
    dead.extend([
        hero.cards[0],
        hero.cards[1],
        villain.cards[0],
        villain.cards[1],
    ]);
    let dead = CardSet::from_cards(&dead);
    let remaining: Vec<Card> = Card::all().filter(|&c| !dead.contains(c)).collect();
    let mut runouts = 0;
    let mut check = |river_board: &[Card]| {
        assert_eq!(river_board.len(), 5);
        assert!(
            evaluate_7([
                hero.cards[0],
                hero.cards[1],
                river_board[0],
                river_board[1],
                river_board[2],
                river_board[3],
                river_board[4]
            ]) > evaluate_7([
                villain.cards[0],
                villain.cards[1],
                river_board[0],
                river_board[1],
                river_board[2],
                river_board[3],
                river_board[4]
            ])
        );
        runouts += 1;
    };
    if board.len() == 4 {
        for &river in &remaining {
            let mut b = board.clone();
            b.push(river);
            check(&b);
        }
    } else {
        for &turn in &remaining {
            for &river in &remaining {
                if turn == river {
                    continue;
                }
                let mut b = board.clone();
                b.extend([turn, river]);
                check(&b);
            }
        }
    }
    let cfg = GameConfig::new(board.clone(), Chips::whole(100), Chips::whole(0));
    let mut trainer = Trainer::new(
        build_tree(&cfg).unwrap(),
        [vec![hero], vec![villain]],
        TrainerOptions {
            algorithm: SolverAlgorithm::CfrPlus,
            ..TrainerOptions::default()
        },
    )
    .unwrap();
    let trained = trainer.train(1);
    let evaluation = trainer.evaluate();
    let (nodes, evaluated) = (evaluation.nodes, evaluation.root_values);
    let exploitability = trainer.exploitability();
    println!("board={board_text:?}, legal ordered runouts={runouts}, train={trained:?}, evaluate={evaluated:?}, best_response={exploitability:?}");
    assert!(
        (trained[0] - 50.0).abs() < 1e-10,
        "constant +50 payoff must survive chance averaging: got {}",
        trained[0]
    );
    assert!((trained[1] + 50.0).abs() < 1e-10);
    assert!((evaluated[0] - 50.0).abs() < 1e-10);
    assert!((evaluated[1] + 50.0).abs() < 1e-10);
    assert!(
        nodes.is_empty(),
        "zero-stack runouts have no betting decisions"
    );
    assert!((exploitability.oop - 50.0).abs() < 1e-10);
    assert!((exploitability.ip + 50.0).abs() < 1e-10);
    assert!(exploitability.total.abs() < 1e-10);
}

#[test]
fn forced_showdown_turn_preserves_constant_payoff() {
    check_forced_showdown(&["Jh", "Js", "2c", "2d"]);
}

#[test]
fn forced_showdown_flop_preserves_constant_payoff() {
    check_forced_showdown(&["Jh", "Js", "2c"]);
}
