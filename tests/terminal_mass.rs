//! Public-API regressions for blocker subtraction near zero compatible reach.
use native_solver::solver::{Action, Node, Player, Terminal, Trainer, Tree};
use native_solver::solver::{SolverAlgorithm, TrainerOptions};
use native_solver::{card_from_str, evaluate_7, Combo};

fn combo(a: &str, b: &str, weight: f64) -> Combo {
    Combo::new(card_from_str(a).unwrap(), card_from_str(b).unwrap(), weight)
}

fn trainer(compatible_weight: Option<f64>, fold: bool) -> Trainer {
    let board: Vec<_> = ["2c", "7d", "9h", "Js", "Qd"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let hero = combo("Ac", "Ad", 1.0);
    let mut opponents = vec![
        combo("Ac", "Kc", 0.1),
        combo("Ad", "Qc", 0.2),
        combo("Ac", "Jc", 0.3),
    ];
    if let Some(weight) = compatible_weight {
        let villain = combo("Kh", "Kd", weight);
        let score = |c: Combo| {
            evaluate_7([
                c.cards[0], c.cards[1], board[0], board[1], board[2], board[3], board[4],
            ])
        };
        assert!(score(hero) > score(villain), "fixture needs AA to beat KK");
        opponents.push(villain);
    }
    let terminal = if fold {
        Terminal::Fold {
            winner: Player::Oop,
            invested: [50.0, 50.0],
        }
    } else {
        Terminal::Showdown {
            invested: [50.0, 50.0],
        }
    };
    let tree = Tree {
        root: Node::Decision {
            id: 0,
            player: Player::Oop,
            actions: vec![Action::Check],
            children: vec![Node::Terminal(terminal)],
        },
        num_decision_nodes: 1,
    };
    Trainer::new(
        tree,
        board,
        [vec![hero], opponents],
        TrainerOptions {
            algorithm: SolverAlgorithm::CfrPlus,
            ..TrainerOptions::default()
        },
    )
}

#[test]
fn all_opponent_reach_blocked_means_zero_mass_and_undefined_exploitability() {
    for fold in [false, true] {
        let mut trainer = trainer(None, fold);
        let raw = trainer.train(1);
        let (nodes, evaluated) = trainer.evaluate();
        assert_eq!(raw, [0.0, 0.0]);
        assert_eq!(evaluated, [0.0, 0.0]);
        assert_eq!(nodes[&0].ev[0], 0.0);
        assert_eq!(nodes[&0].equity.as_ref().unwrap()[0], 0.5);
        let exploitability = trainer.exploitability();
        assert!(
            exploitability.oop.is_nan()
                && exploitability.ip.is_nan()
                && exploitability.total.is_nan(),
            "no legal deals must not yield finite exploitability: {exploitability:?}"
        );
    }
}

#[test]
fn tiny_compatible_reach_survives_larger_blocked_reach() {
    for weight in [1e-12, 1e-18, 1e-100] {
        for fold in [false, true] {
            let mut trainer = trainer(Some(weight), fold);
            let raw = trainer.train(1);
            let expected = 50.0 * weight;
            assert!(
                (raw[0] - expected).abs() <= expected * 1e-10,
                "fold={fold}, weight={weight}, raw={} expected={expected}",
                raw[0]
            );
            assert!((raw[1] + expected).abs() <= expected * 1e-10);
            let (nodes, _) = trainer.evaluate();
            assert!(
                (nodes[&0].ev[0] - 50.0).abs() < 1e-8,
                "weight={weight}, ev={:?}",
                nodes[&0].ev
            );
            assert!((nodes[&0].equity.as_ref().unwrap()[0] - 1.0).abs() < 1e-10);
            let exploitability = trainer.exploitability();
            assert!(
                (exploitability.oop - 50.0).abs() < 1e-8,
                "weight={weight}: {exploitability:?}"
            );
            assert!(
                (exploitability.ip + 50.0).abs() < 1e-8,
                "weight={weight}: {exploitability:?}"
            );
            assert!(
                exploitability.total.abs() < 1e-8,
                "weight={weight}: {exploitability:?}"
            );
        }
    }
}

#[test]
fn highly_unequal_showdown_investments_preserve_small_payouts() {
    let board: Vec<_> = ["2c", "7d", "9h", "Js", "Qd"]
        .iter()
        .map(|s| card_from_str(s).unwrap())
        .collect();
    let aces = combo("Ac", "Ad", 1.0);
    let kings = combo("Kh", "Kd", 1.0);
    let score = |c: Combo| {
        evaluate_7([
            c.cards[0], c.cards[1], board[0], board[1], board[2], board[3], board[4],
        ])
    };
    assert!(score(aces) > score(kings));
    for invested in [[1e9, 1e-6], [1e-6, 1e9]] {
        for winner in 0..2 {
            let combos = if winner == 0 {
                [vec![aces], vec![kings]]
            } else {
                [vec![kings], vec![aces]]
            };
            let tree = Tree {
                root: Node::Terminal(Terminal::Showdown { invested }),
                num_decision_nodes: 0,
            };
            let mut trainer = Trainer::new(
                tree,
                board.clone(),
                combos,
                TrainerOptions {
                    algorithm: SolverAlgorithm::CfrPlus,
                    ..TrainerOptions::default()
                },
            );
            let won = invested[1 - winner];
            let expected = if winner == 0 {
                [won, -won]
            } else {
                [-won, won]
            };
            let raw = trainer.train(1);
            let (_, evaluated) = trainer.evaluate();
            let exploitability = trainer.exploitability();
            for (label, actual) in [
                ("train", raw),
                ("evaluate", evaluated),
                ("best response", [exploitability.oop, exploitability.ip]),
            ] {
                for player in 0..2 {
                    assert!((actual[player] - expected[player]).abs() <= expected[player].abs() * 1e-12,
                        "{label}: winner={winner}, invested={invested:?}, player={player}, actual={} expected={}", actual[player], expected[player]);
                }
            }
        }
    }
}
