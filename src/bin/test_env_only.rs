//! Minimal test - only environment, NO model
//!
//! This isolates the stack overflow issue.

use std::path::Path;

use clap::Parser;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use eris::config::FullTrainingConfig;
use eris::env::IOBufferEnv;
use eris::{Environment, Space};

#[derive(Clone, Debug, clap::ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Trace file format
#[derive(Clone, Debug, clap::ValueEnum)]
enum TraceFormat {
    Autodetect,
    Recorder,
    Dftracer,
}

#[derive(Parser, Debug)]
#[command(name = "eris-test-env", version = "0.1.0")]
struct Args {
    #[arg(short, long, default_value = "config/tiers.toml")]
    config: String,

    #[arg(short, long, default_value = "recorder-csv/NWChem-64_combined.csv")]
    trace: String,

    /// Trace format: recorder (CSV), dftracer (pfw.gz), or autodetect
    #[arg(long, value_enum, default_value = "autodetect")]
    trace_format: TraceFormat,

    #[arg(short, long, default_value = "10")]
    max_steps: usize,

    #[arg(long, value_enum, default_value = "info")]
    log_level: LogLevel,
}

fn main() {
    let args = Args::parse();

    // Initialize logging with user-specified level
    let level = match args.log_level {
        LogLevel::Trace => Level::TRACE,
        LogLevel::Debug => Level::DEBUG,
        LogLevel::Info => Level::INFO,
        LogLevel::Warn => Level::WARN,
        LogLevel::Error => Level::ERROR,
    };
    let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    println!("=== Testing Environment Only (NO Model) ===");
    println!("This isolates the stack overflow issue");
    println!();

    if let Err(e) = test_env_only(&args) {
        tracing::error!("Test failed: {}", e);
        std::process::exit(1);
    }
}

fn to_trace_format(format: &TraceFormat) -> eris::TraceFormat {
    match format {
        TraceFormat::Autodetect => eris::TraceFormat::Autodetect,
        TraceFormat::Recorder => eris::TraceFormat::Recorder,
        TraceFormat::Dftracer => eris::TraceFormat::Dftracer,
    }
}

fn test_env_only(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = Path::new(&args.config);
    let trace_path = Path::new(&args.trace);

    println!("Loading config...");
    let _config = FullTrainingConfig::from_file(config_path)?;
    println!("✓ Config loaded");

    println!("\nCreating environment...");
    let env = IOBufferEnv::new(
        config_path,
        trace_path,
        to_trace_format(&args.trace_format),
        args.max_steps,
        None,
        None,
    )?;
    println!("✓ Environment created");

    println!("\nEnvironment info:");
    println!("  Observation dim: {}", env.observation_space().dim());
    println!("  Action dim: {}", env.action_space().n);

    println!("\nResetting environment...");
    let mut env = env;
    let obs = env.reset();
    println!("✓ Environment reset");
    println!("  Initial observation: {} values", obs.len());

    println!("\nTaking 5 random steps...");
    for step in 0..5 {
        let action = step % 10; // Simple deterministic action
        let (_next_obs, reward, done) = env.step(action);
        println!(
            "  Step {}: action={}, reward={:.2}, done={}",
            step + 1,
            action,
            reward,
            done
        );
        if done {
            break;
        }
    }

    println!("\n=== Test Complete ===");
    println!("✓ Environment initialization successful");
    println!("✓ Environment reset successful");
    println!("✓ Environment interaction successful");
    println!("\nEnvironment works fine without Burn model!");
    println!("The stack overflow is in the model initialization.");

    Ok(())
}
