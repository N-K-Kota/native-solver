//! Reproducible, end-to-end solver benchmark using only the public API.
//!
//! Build separately from measurement:
//!   cargo build --release --example solver_performance
//!   target/release/examples/solver_performance --workload river-broad --threads 1
//! Use the identical file, arguments, compiler and host for both revisions.
//! Run each workload in a fresh process for comparable cold initialization, and
//! repeat/alternate revisions to measure noise. No bucketing or suit reduction.
//! Output is JSON Lines (default), or CSV with a JSON checkpoint-array column.
//! Times exclude compilation, thread-pool creation and result serialization.
//! A fixed-iteration run is explicitly selected with --fixed-iterations; it does
//! not claim convergence. Otherwise hitting the cap without the target exits 2.

use native_solver::card::{card_from_str, set_contains, set_from_cards, Card};
use native_solver::range::{all_canonical_hands, Combo, Range};
use native_solver::solver::quality::ReferenceEvaluator;
use native_solver::solver::Chips;
use native_solver::solver::TrainerOptions;
use native_solver::solver::{
    build_tree_bounded, Exploitability, GameConfig, Node, SolverAlgorithm, Trainer,
};
use rayon::ThreadPoolBuilder;
use std::fmt::Write;
use std::time::{Duration, Instant};

const MAX_REFERENCE_WORK: usize = 6_000_000;
const MAX_DECISION_NODES: usize = 250_000;

const HELP: &str = "solver_performance [OPTIONS]
  --workload NAME       river-narrow, river-broad, turn-moderate, river-paired,
                        river-monotone, turn-draws, flop-dry, flop-two-tone,
                        flop-monotone, flop-paired, or all (legacy six; default)
  --algorithm NAME      cfr, cfr-plus (default), dcfr
  --iterations N        maximum iterations, or exact count in fixed mode (1000)
  --check-every N       exploitability checkpoint spacing in iterations (50)
  --target-chips X      exploitability target, in chips, not percent (0.1)
  --threads N           dedicated Rayon thread count (1)
  --format FORMAT       json (one object per workload) or csv (json)
  --fixed-iterations    train to cap unless timed out; convergence is separate
  --reference-check     independent final profile/BR audit (bounded trees only)
  --max-seconds X       soft solve budget; checked between batches (default 180)
  --help                show this help

All games use pot=100, no abstraction or suit reduction.
CFR+ means the legacy simultaneous, uniformly averaged RM+ variant.
DCFR uses simultaneous updates and published alpha=1.5, beta=0, gamma=2.
river-narrow: 37 OOP / 27 IP combos, two bet sizes, one raise.
river-broad: every legal combo for each player, two bet sizes, one raise.
turn-moderate: 37 OOP / 27 IP combos, one bet size, no raises.
Flop fixtures use one bet size, no raises, and exact full runouts.
Use the Python runner for a hard process timeout and RSS guard.
The three original held-out fixtures are river-paired (weighted, stack=75),
river-monotone (weighted, stack=150), turn-draws (stack=100).
Checkpoints include iteration 0 and the exact final cap. Exit 2 means at least
one target-mode workload hit its cap without convergence; invalid input or
nonfinite results exit 1. Each workload's first initialization may include
process-global evaluator table initialization; prefer separate processes.
";

struct Options {
    workload: String,
    algorithm: SolverAlgorithm,
    iterations: u32,
    check_every: u32,
    target: f64,
    threads: usize,
    csv: bool,
    fixed: bool,
    reference: bool,
    max_seconds: f64,
}

impl Options {
    fn parse() -> Result<Option<Self>, String> {
        let mut options = Self {
            workload: "all".into(),
            algorithm: SolverAlgorithm::CfrPlus,
            iterations: 1000,
            check_every: 50,
            target: 0.1,
            threads: 1,
            csv: false,
            fixed: false,
            reference: false,
            max_seconds: 180.0,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            if arg == "--help" || arg == "-h" {
                print!("{HELP}");
                return Ok(None);
            }
            if arg == "--reference-check" {
                options.reference = true;
                continue;
            }
            if arg == "--fixed-iterations" {
                options.fixed = true;
                continue;
            }
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {arg}"))?;
            match arg.as_str() {
                "--workload" => options.workload = value,
                "--algorithm" => {
                    options.algorithm = match value.as_str() {
                        "cfr" => SolverAlgorithm::Cfr,
                        "cfr-plus" => SolverAlgorithm::CfrPlus,
                        "dcfr" => SolverAlgorithm::Dcfr,
                        _ => return Err("--algorithm must be cfr, cfr-plus or dcfr".into()),
                    }
                }
                "--iterations" => {
                    options.iterations = value.parse().map_err(|_| "invalid --iterations")?
                }
                "--check-every" => {
                    options.check_every = value.parse().map_err(|_| "invalid --check-every")?
                }
                "--threads" => options.threads = value.parse().map_err(|_| "invalid --threads")?,
                "--max-seconds" => {
                    options.max_seconds = value.parse().map_err(|_| "invalid --max-seconds")?
                }
                "--target-chips" => {
                    options.target = value.parse().map_err(|_| "invalid --target-chips")?
                }
                "--format" => {
                    options.csv = match value.as_str() {
                        "json" => false,
                        "csv" => true,
                        _ => return Err("--format must be json or csv".into()),
                    }
                }
                _ => return Err(format!("unknown argument {arg}")),
            }
        }
        if ![
            "all",
            "river-narrow",
            "river-broad",
            "turn-moderate",
            "river-paired",
            "river-monotone",
            "turn-draws",
            "flop-dry",
            "flop-two-tone",
            "flop-monotone",
            "flop-paired",
        ]
        .contains(&options.workload.as_str())
        {
            return Err("unknown workload (see --help)".into());
        }
        if options.iterations == 0 || options.check_every == 0 || options.threads == 0 {
            return Err("iterations, check-every and threads must all be positive".into());
        }
        if !options.target.is_finite() || options.target <= 0.0 {
            return Err("--target-chips must be positive and finite".into());
        }
        if !options.max_seconds.is_finite() || options.max_seconds <= 0.0 {
            return Err("--max-seconds must be positive and finite".into());
        }
        Ok(Some(options))
    }
}

struct Checkpoint {
    elapsed_seconds: f64,
    iterations: u32,
    train_seconds: f64,
    diagnostic_seconds: f64,
    exploitability: Exploitability,
}

struct ResultRow {
    name: String,
    reference_json: String,
    board: String,
    range_specs: [String; 2],
    range_sizes: [usize; 2],
    decision_nodes: usize,
    bet_sizes: Vec<f64>,
    effective_stack: f64,
    peak_rss_kib: Option<u64>,
    max_raises: u32,
    joint_mass: f64,
    initialization_seconds: f64,
    train_seconds: f64,
    diagnostic_seconds: f64,
    evaluate_seconds: f64,
    solve_seconds: f64,
    total_seconds: f64,
    iterations: u32,
    converged: bool,
    time_budget_reached: bool,
    first_target_iteration: Option<u32>,
    root_ev: [f64; 2],
    strategy_checksum: f64,
    evaluation_checksum: f64,
    checkpoints: Vec<Checkpoint>,
}

fn sorted_combos(spec: &str, board: &[Card]) -> Result<Vec<Combo>, String> {
    let mut combos = Range::parse(spec)
        .map_err(|e| e.to_string())?
        .combos(set_from_cards(board));
    // Fix each pair's orientation as well as the combo order so reductions
    // are repeatable.
    for combo in &mut combos {
        combo.cards.sort_unstable();
    }
    combos.sort_by_key(|c| c.cards);
    Ok(combos)
}

fn run(name: &str, options: &Options) -> Result<ResultRow, String> {
    let total_start = Instant::now();
    let (board_text, range_specs, bet_sizes, max_raises, effective_stack) = match name {
        "river-narrow" => (
            "2c,7d,9h,Ks,3c",
            ["QQ,JJ,TT,99,AQs,AQo".into(), "88,77,66,KTs,KTo".into()],
            vec![0.5, 1.0],
            1,
            200.0,
        ),
        "river-broad" => {
            let all = all_canonical_hands().join(",");
            (
                "2c,7d,9h,Ks,3c",
                [all.clone(), all],
                vec![0.5, 1.0],
                1,
                200.0,
            )
        }
        "turn-moderate" => (
            "2c,7d,9h,Ks",
            ["QQ,JJ,TT,99,AQs,AQo".into(), "88,77,66,KTs,KTo".into()],
            vec![1.0],
            0,
            200.0,
        ),
        "river-paired" => (
            "Qc,Qd,7h,4s,2c",
            [
                "AA:0.5,KK,JJ,AQs,AQo:0.25".into(),
                "77,88,99,KQs,KQo:0.5".into(),
            ],
            vec![0.33, 0.75, 1.5],
            1,
            75.0,
        ),
        "river-monotone" => (
            "Ah,Kh,8h,5h,2c",
            [
                "QJs,QJo:0.5,TT,99,76s".into(),
                "QQ,JJ,T9s,T9o:0.4,AQs".into(),
            ],
            vec![0.5, 1.0],
            1,
            150.0,
        ),
        "turn-draws" => (
            "Ts,9s,4d,2c",
            ["AKs,AQs,KQs,JJ,TT".into(), "AJs,KJs,QJs,99,88".into()],
            vec![0.75],
            0,
            100.0,
        ),
        "flop-dry" => (
            "Ks,7d,2c",
            ["AA,AKs".into(), "QQ,KQs".into()],
            vec![0.5],
            0,
            100.0,
        ),
        "flop-two-tone" => (
            "Js,9s,4d",
            ["AA,KK,QQ,AQs,AKs,KQs".into(), "JJ,TT,99,QTs,T8s,AJs".into()],
            vec![0.75],
            0,
            100.0,
        ),
        "flop-monotone" => (
            "Ah,8h,3h",
            ["AKs,AQo:0.5,KK".into(), "QJs,JTs,88:0.5".into()],
            vec![0.5],
            0,
            75.0,
        ),
        "flop-paired" => (
            "Qc,Qd,5s",
            ["AA:0.5,KK,AQs,KQs".into(), "JJ,TT,55,QJs".into()],
            vec![1.0],
            0,
            150.0,
        ),
        _ => return Err(format!("unknown workload: {name}")),
    };
    let board: Vec<Card> = board_text
        .split(',')
        .map(|c| card_from_str(c).expect("valid fixture board"))
        .collect();
    let combos = [
        sorted_combos(&range_specs[0], &board)?,
        sorted_combos(&range_specs[1], &board)?,
    ];
    let range_sizes = [combos[0].len(), combos[1].len()];
    if options.reference && range_sizes[0] * range_sizes[1] > 10000 {
        return Err(
            "--reference-check is limited to at most 10000 private-pair entries; use small ranges"
                .into(),
        );
    }
    // This one-time normalization is deliberately outside timed training and
    // included in initialization. evaluate() returns raw range-weighted CFVs;
    // divide by compatible initial joint range mass for an EV in chips.
    let joint_mass: f64 = combos[0]
        .iter()
        .map(|a| {
            combos[1]
                .iter()
                .filter(|b| {
                    a.cards[0] != b.cards[0]
                        && a.cards[0] != b.cards[1]
                        && a.cards[1] != b.cards[0]
                        && a.cards[1] != b.cards[1]
                })
                .map(|b| a.weight * b.weight)
                .sum::<f64>()
        })
        .sum();
    if !joint_mass.is_finite() || joint_mass <= 0.0 {
        return Err("invalid compatible initial range mass".into());
    }
    let cfg = GameConfig {
        board: board.clone(),
        pot: Chips::whole(100),
        effective_stack: Chips::from_f64(effective_stack).expect("workload stacks are whole chips"),
        bet_sizes: bet_sizes.clone(),
        max_raises,
        suit_isomorphism: false,
    };
    let tree = build_tree_bounded(&cfg, MAX_DECISION_NODES).map_err(|e| e.to_string())?;
    let decision_nodes = tree.num_decision_nodes;
    // Pair count alone is insufficient on a flop: reject a large independent
    // traversal before allocating its state. This is a conservative work proxy,
    // not an estimate of resident bytes. The runner provides hard limits too.
    let reference_work = decision_nodes
        .saturating_mul(range_sizes[0])
        .saturating_mul(range_sizes[1]);
    if options.reference && reference_work > MAX_REFERENCE_WORK {
        return Err(format!("--reference-check exceeds {MAX_REFERENCE_WORK} decision/private-pair work units ({reference_work}); use the production exact metric without an independent audit"));
    }
    let mut trainer = Trainer::new(
        tree,
        board,
        combos,
        TrainerOptions {
            algorithm: options.algorithm,
            ..TrainerOptions::default()
        },
    );
    let initialization_seconds = total_start.elapsed().as_secs_f64();
    let mut train_time = Duration::ZERO;
    let mut diagnostic_time = Duration::ZERO;
    let mut iterations = 0;
    let mut first_target_iteration = None;
    let mut checkpoints = Vec::new();
    loop {
        let start = Instant::now();
        let exploitability = trainer.exploitability();
        diagnostic_time += start.elapsed();
        if ![exploitability.oop, exploitability.ip, exploitability.total]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(format!(
                "{name}: nonfinite exploitability at iteration {iterations}"
            ));
        }
        let converged = exploitability.total <= options.target;
        if converged && first_target_iteration.is_none() {
            first_target_iteration = Some(iterations);
        }
        checkpoints.push(Checkpoint {
            elapsed_seconds: total_start.elapsed().as_secs_f64(),
            iterations,
            train_seconds: train_time.as_secs_f64(),
            diagnostic_seconds: diagnostic_time.as_secs_f64(),
            exploitability,
        });
        if iterations >= options.iterations
            || (!options.fixed && converged)
            || total_start.elapsed().as_secs_f64() >= options.max_seconds
        {
            break;
        }
        let batch = options.check_every.min(options.iterations - iterations);
        let start = Instant::now();
        let current_raw_cfv = trainer.train(batch);
        train_time += start.elapsed();
        if !current_raw_cfv.iter().all(|v| v.is_finite()) {
            return Err(format!("{name}: nonfinite current-strategy CFV"));
        }
        iterations += batch;
    }
    let solve_seconds = total_start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (evaluations, raw_root_cfv) = trainer.evaluate();
    let evaluate_seconds = start.elapsed().as_secs_f64();
    let root_ev = [raw_root_cfv[0] / joint_mass, raw_root_cfv[1] / joint_mass];
    let mut strategy_checksum = 0.0;
    let mut evaluation_checksum = 0.0;
    // Weighted numeric checksums are compact drift indicators, not proofs of
    // numerical equivalence. Include every node's board-legal acting-player
    // combos only: impossible rows have no meaning and must not affect output.
    // Calculation is excluded from solve/evaluate timing, included in total.
    let mut players = vec![0; decision_nodes];
    collect_players(trainer.tree(), &mut players);
    for (node_id, &player) in players.iter().enumerate() {
        let board_mask = set_from_cards(trainer.node_board(node_id));
        let live: Vec<bool> = trainer
            .combos(player)
            .iter()
            .map(|c| !set_contains(board_mask, c.cards[0]) && !set_contains(board_mask, c.cards[1]))
            .collect();
        let strategy = trainer.average_strategy(node_id);
        for (action, row) in strategy.iter().enumerate() {
            for (combo, &value) in row.iter().enumerate() {
                if !live[combo] {
                    continue;
                }
                if !value.is_finite() {
                    return Err(format!("{name}: nonfinite strategy at node {node_id}"));
                }
                strategy_checksum += value
                    * ((node_id + 1) as f64
                        + (action + 1) as f64 / 17.0
                        + (combo + 1) as f64 / 65537.0);
            }
        }
        let evaluation = evaluations
            .get(&node_id)
            .ok_or_else(|| format!("{name}: missing evaluation node {node_id}"))?;
        for (combo, &value) in evaluation.ev.iter().enumerate() {
            if !live[combo] {
                continue;
            }
            if !value.is_finite() {
                return Err(format!("{name}: nonfinite EV at node {node_id}"));
            }
            evaluation_checksum += value * ((node_id + 1) as f64 + (combo + 1) as f64 / 65537.0);
        }
        if let Some(equity) = &evaluation.equity {
            for (combo, &value) in equity.iter().enumerate() {
                if !live[combo] {
                    continue;
                }
                if !value.is_finite() {
                    return Err(format!("{name}: nonfinite equity at node {node_id}"));
                }
                evaluation_checksum +=
                    value * ((node_id + 1) as f64 / 17.0 + (combo + 1) as f64 / 65537.0);
            }
        }
    }
    if !root_ev
        .iter()
        .chain([&strategy_checksum, &evaluation_checksum])
        .all(|v| v.is_finite())
    {
        return Err(format!("{name}: nonfinite normalized EV or checksum"));
    }
    let converged = checkpoints
        .last()
        .expect("initial checkpoint exists")
        .exploitability
        .total
        <= options.target;
    let time_budget_reached = iterations < options.iterations
        && (options.fixed || !converged)
        && solve_seconds >= options.max_seconds;
    let total_seconds = total_start.elapsed().as_secs_f64();
    let peak_rss_kib = peak_rss_kib();
    // Keep the expensive independent audit outside the production solver's
    // total time and peak RSS. The audit has its own explicit timing field.
    let reference_json = if options.reference {
        let start = Instant::now();
        let evaluator = ReferenceEvaluator::new(
            trainer.tree(),
            &cfg.board,
            [trainer.combos(0), trainer.combos(1)],
        )
        .map_err(|e| e.to_string())?;
        let quality = evaluator
            .evaluate(|id| trainer.average_strategy(id))
            .map_err(|e| e.to_string())?;
        let production = checkpoints.last().unwrap().exploitability;
        let error = (quality.profile_ev[0] - root_ev[0])
            .abs()
            .max((quality.profile_ev[1] - root_ev[1]).abs())
            .max((quality.best_response_ev[0] - production.oop).abs())
            .max((quality.best_response_ev[1] - production.ip).abs());
        if !error.is_finite() || error > 1e-8 {
            return Err(format!("independent reference disagreement: {error} chips"));
        }
        format!("{{\"seconds\":{:.9},\"exploitability_chips\":{:.12},\"max_value_error_chips\":{:.12}}}",
            start.elapsed().as_secs_f64(), quality.exploitability, error)
    } else {
        "null".into()
    };
    Ok(ResultRow {
        name: name.into(),
        reference_json,
        board: board_text.into(),
        range_specs,
        range_sizes,
        decision_nodes,
        bet_sizes,
        effective_stack,
        peak_rss_kib,
        max_raises,
        joint_mass,
        initialization_seconds,
        train_seconds: train_time.as_secs_f64(),
        diagnostic_seconds: diagnostic_time.as_secs_f64(),
        evaluate_seconds,
        solve_seconds,
        total_seconds,
        iterations,
        converged,
        time_budget_reached,
        first_target_iteration,
        root_ev,
        strategy_checksum,
        evaluation_checksum,
        checkpoints,
    })
}

fn collect_players(node: &Node, players: &mut [usize]) {
    match node {
        Node::Decision {
            id,
            player,
            children,
            ..
        } => {
            players[*id] = player.index();
            for child in children {
                collect_players(child, players);
            }
        }
        Node::Chance { children } => {
            for (_, child) in children {
                collect_players(child, players);
            }
        }
        Node::Terminal(_) => {}
    }
}

fn checkpoint_json(checkpoints: &[Checkpoint]) -> String {
    let mut out = String::from("[");
    for (i, c) in checkpoints.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write!(out, "{{\"elapsed_seconds\":{:.9},\"iterations\":{},\"train_seconds\":{:.9},\"exploitability_seconds_cumulative\":{:.9},\"exploitability_chips\":{:.12},\"best_response_oop_chips\":{:.12},\"best_response_ip_chips\":{:.12}}}",
            c.elapsed_seconds, c.iterations, c.train_seconds, c.diagnostic_seconds, c.exploitability.total, c.exploitability.oop, c.exploitability.ip).unwrap();
    }
    out.push(']');
    out
}

fn algorithm_name(algorithm: SolverAlgorithm) -> &'static str {
    match algorithm {
        SolverAlgorithm::Cfr => "cfr",
        SolverAlgorithm::CfrPlus => "cfr-plus",
        SolverAlgorithm::Dcfr => "dcfr",
    }
}

// Linux VmHWM is sampled by the solver itself after all work. Unlike a
// launcher's wait4 high water it does not carry the launcher's pre-exec RSS.
// This is whole-process resident memory, including tables, Rayon and allocator;
// unsupported platforms return null rather than a misleading zero.
fn peak_rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")
                .and_then(|value| value.split_whitespace().next()?.parse().ok())
        })
}

fn emit(row: &ResultRow, options: &Options, process_workload_index: usize) {
    let status = if row.time_budget_reached {
        "time_budget"
    } else if options.fixed {
        "fixed_iterations"
    } else if row.converged {
        "target_reached"
    } else {
        "iteration_cap"
    };
    let target_iteration = row
        .first_target_iteration
        .map_or_else(|| "null".into(), |n| n.to_string());
    let final_exploitability = row.checkpoints.last().unwrap().exploitability.total;
    let checkpoints = checkpoint_json(&row.checkpoints);
    if options.csv {
        // The fixed fixture strings contain no quotes; escape the JSON column.
        println!("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.12},{:.12},{:.12},{:.12},{:.12},\"{}\",\"{}\",4,{},exact_finite_action_game,{}",
            row.name, algorithm_name(options.algorithm), row.effective_stack, row.peak_rss_kib.map_or_else(String::new, |n| n.to_string()), options.threads, process_workload_index, row.range_sizes[0], row.range_sizes[1], row.decision_nodes,
            options.iterations, options.check_every, options.target, row.iterations, status, row.converged, target_iteration,
            row.max_raises, row.bet_sizes.len(), row.joint_mass,
            row.initialization_seconds, row.train_seconds, row.diagnostic_seconds, row.evaluate_seconds, row.solve_seconds, row.total_seconds,
            final_exploitability, row.root_ev[0], row.root_ev[1], row.strategy_checksum, row.evaluation_checksum, checkpoints.replace('"', "\"\""), row.reference_json.replace('"', "\"\""), options.max_seconds, options.reference);
    } else {
        // All strings here are fixed validated fixture names/specifications.
        println!(concat!(
            "{{\"schema_version\":4,\"workload\":\"{}\",\"algorithm\":\"{}\",\"threads\":{},\"process_workload_index\":{},",
            "\"board\":\"{}\",\"range_specs\":[\"{}\",\"{}\"],\"range_sizes\":[{},{}],\"decision_nodes\":{},",
            "\"pot_chips\":100,\"effective_stack_chips\":{},\"bet_sizes\":{:?},\"max_raises\":{},",
            "\"checksum_scope\":\"all_nodes_board_legal_combos\",\"update_schedule\":\"simultaneous\",\"peak_rss_kib\":{},\"card_abstraction\":false,\"suit_isomorphism\":false,\"joint_range_mass\":{},",
            "\"max_seconds\":{},\"quality_scope\":\"exact_finite_action_game\",\"independent_audit\":{},\"iteration_cap\":{},\"check_every\":{},\"target_chips\":{},\"iterations\":{},\"status\":\"{}\",\"converged\":{},\"first_target_iteration\":{},",
            "\"initialization_seconds\":{:.9},\"train_seconds\":{:.9},\"exploitability_seconds\":{:.9},\"evaluate_seconds\":{:.9},\"solve_seconds\":{:.9},\"total_seconds\":{:.9},",
            "\"exploitability_chips\":{:.12},\"root_average_ev_chips\":[{:.12},{:.12}],\"strategy_checksum\":{:.12},\"evaluation_checksum\":{:.12},\"checkpoints\":{},\"reference_check\":{}}}"),
            row.name, algorithm_name(options.algorithm), options.threads, process_workload_index, row.board, row.range_specs[0], row.range_specs[1], row.range_sizes[0], row.range_sizes[1], row.decision_nodes,
            row.effective_stack, row.bet_sizes, row.max_raises, row.peak_rss_kib.map_or_else(|| "null".into(), |n| n.to_string()), row.joint_mass, options.max_seconds, options.reference, options.iterations, options.check_every, options.target, row.iterations, status, row.converged, target_iteration,
            row.initialization_seconds, row.train_seconds, row.diagnostic_seconds, row.evaluate_seconds, row.solve_seconds, row.total_seconds,
            final_exploitability, row.root_ev[0], row.root_ev[1], row.strategy_checksum, row.evaluation_checksum, checkpoints, row.reference_json);
    }
}

fn main() -> std::process::ExitCode {
    let options = match Options::parse() {
        Ok(Some(options)) => options,
        Ok(None) => return std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            return std::process::ExitCode::from(1);
        }
    };
    if cfg!(debug_assertions) {
        eprintln!("error: benchmark requires a release build (--release)");
        return std::process::ExitCode::from(1);
    }
    let pool = match ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()
    {
        Ok(pool) => pool,
        Err(error) => {
            eprintln!("error creating Rayon pool: {error}");
            return std::process::ExitCode::from(1);
        }
    };
    if options.csv {
        println!("workload,algorithm,effective_stack_chips,peak_rss_kib,threads,process_workload_index,oop_combos,ip_combos,decision_nodes,iteration_cap,check_every,target_chips,iterations,status,converged,first_target_iteration,max_raises,bet_size_count,joint_range_mass,initialization_seconds,train_seconds,exploitability_seconds,evaluate_seconds,solve_seconds,total_seconds,exploitability_chips,root_oop_ev_chips,root_ip_ev_chips,strategy_checksum,evaluation_checksum,checkpoints_json,reference_check_json,schema_version,max_seconds,quality_scope,independent_audit");
    }
    let workloads = if options.workload == "all" {
        vec![
            "river-narrow",
            "river-broad",
            "turn-moderate",
            "river-paired",
            "river-monotone",
            "turn-draws",
        ]
    } else {
        vec![options.workload.as_str()]
    };
    let mut missed_target = false;
    for (i, name) in workloads.iter().enumerate() {
        match pool.install(|| run(name, &options)) {
            Ok(row) => {
                emit(&row, &options, i);
                if row.time_budget_reached || (!options.fixed && !row.converged) {
                    eprintln!(
                        "{name}: solve budget/cap reached; target is {} chips",
                        options.target
                    );
                    missed_target = true;
                }
            }
            Err(error) => {
                eprintln!("error: {error}");
                return std::process::ExitCode::from(1);
            }
        }
    }
    std::process::ExitCode::from(if missed_target { 2 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dry_flop_independent_audit_fits_work_guard() {
        let board: Vec<_> = ["Ks", "7d", "2c"]
            .iter()
            .map(|c| card_from_str(c).unwrap())
            .collect();
        let oop = sorted_combos("AA,AKs", &board).unwrap();
        let ip = sorted_combos("QQ,KQs", &board).unwrap();
        let cfg = GameConfig {
            board,
            pot: Chips::whole(100),
            effective_stack: Chips::whole(100),
            bet_sizes: vec![0.5],
            max_raises: 0,
            suit_isomorphism: false,
        };
        let tree = build_tree_bounded(&cfg, MAX_DECISION_NODES).unwrap();
        assert_eq!([oop.len(), ip.len()], [9, 9]);
        // Four decisions per live street (OOP, IP after check, and the
        // two fold/call responses). Three flop continuations reach the turn.
        // Five histories reach a live river: three after flop check-check,
        // plus turn check-check after either flop bet-call. The other four
        // turn bet-calls are all-in and deal the river without forced checks.
        let expected_nodes = 4 + 49 * (3 * 4) + 49 * 48 * (5 * 4);
        assert_eq!(tree.num_decision_nodes, expected_nodes);
        assert!(tree.num_decision_nodes * oop.len() * ip.len() <= MAX_REFERENCE_WORK);
    }
}
