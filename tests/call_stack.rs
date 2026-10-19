use native_solver::card_from_str;
use native_solver::solver::Chips;
use native_solver::solver::{build_tree, Action, GameConfig, Node};

fn action<'a>(node: &'a Node, name: &str) -> &'a Node {
    let Node::Decision {
        actions, children, ..
    } = node
    else {
        panic!("expected decision: {name}");
    };
    let index = actions
        .iter()
        .position(|a| match (name, a) {
            ("bet50", Action::Bet(amount)) => *amount == Chips::whole(50),
            ("check", Action::Check) | ("call", Action::Call) => true,
            _ => false,
        })
        .unwrap_or_else(|| panic!("missing {name}; actions={actions:?}"));
    &children[index]
}
fn chance<'a>(node: &'a Node, card: &str) -> &'a Node {
    let Node::Chance { children } = node else {
        panic!("expected chance");
    };
    &children
        .iter()
        .find(|(cards, _)| cards.contains(&card_from_str(card).unwrap()))
        .unwrap()
        .1
}
#[test]
fn callers_spend_their_remaining_stack_before_next_street() {
    let cfg = GameConfig {
        board: ["2c", "7d", "Jh"]
            .iter()
            .map(|s| card_from_str(s).unwrap())
            .collect(),
        pot: Chips::whole(100),
        effective_stack: Chips::whole(100),
        bet_sizes: vec![0.5],
        max_raises: 0,
        suit_isomorphism: false,
    };
    let tree = build_tree(&cfg).unwrap();
    let ip_flop = action(&tree.root, "bet50");
    let turn = chance(action(ip_flop, "call"), "Qh");
    let ip_turn = action(turn, "check");
    let oop_turn = action(ip_turn, "bet50");
    let river = chance(action(oop_turn, "call"), "Ks");
    let Node::Terminal(native_solver::solver::Terminal::Showdown { invested }) = river else {
        panic!("both stacks are spent; the river must go directly to showdown")
    };
    assert_eq!(*invested, [150.0, 150.0]);
}

#[test]
fn no_terminal_exceeds_the_effective_stack_after_calls_and_raises() {
    fn check(node: &Node, maximum: f64) {
        match node {
            Node::Decision { children, .. } => {
                for child in children {
                    check(child, maximum);
                }
            }
            Node::Chance { children } => {
                for (_, child) in children {
                    check(child, maximum);
                }
            }
            Node::Terminal(terminal) => {
                let invested = match terminal {
                    native_solver::solver::Terminal::Fold { invested, .. }
                    | native_solver::solver::Terminal::Showdown { invested } => invested,
                };
                assert!(
                    invested.iter().all(|&x| x <= maximum + 1e-10),
                    "a player invested past its stack: {invested:?}, maximum={maximum}"
                );
            }
        }
    }
    for raises in [0, 1, 2] {
        let cfg = GameConfig {
            board: ["2c", "7d", "Jh", "Qs"]
                .iter()
                .map(|s| card_from_str(s).unwrap())
                .collect(),
            pot: Chips::whole(100),
            effective_stack: Chips::whole(80),
            bet_sizes: vec![0.25, 0.5, 1.0],
            max_raises: raises,
            suit_isomorphism: false,
        };
        check(
            &build_tree(&cfg).unwrap().root,
            cfg.pot.to_f64() / 2.0 + cfg.effective_stack.to_f64(),
        );
    }
}
