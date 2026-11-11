//! Regenerates `src/solver/preflop_equity.bin`, the exact all-in equity of
//! every preflop matchup up to suit renaming. Takes tens of CPU-minutes.
//!
//!   cargo run --release --example gen_preflop_equity -- src/solver/preflop_equity.bin
//!
//! The output is deterministic. `cargo test --release
//! the_shipped_table_matches_the_generator -- --ignored` checks the shipped
//! file against a fresh run.

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: gen_preflop_equity OUTPUT_FILE");
        std::process::exit(1);
    };
    let start = std::time::Instant::now();
    let bytes = native_solver::solver::generate_preflop_equity_table();
    std::fs::write(&path, &bytes).expect("the output file must be writable");
    println!(
        "wrote {} bytes to {path} in {:.0?}",
        bytes.len(),
        start.elapsed()
    );
}
