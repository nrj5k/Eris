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

#[derive(Parser, Debug)]
#[command(name = "eris-test-env", version = "0.1.0")]
struct Args {
    #[arg(short, long, default_value = "config/tiers.toml")]
    config: String,

    #[arg(short, long, default_value = "recorder-csv/NWChem-64_combined.csv")]
    trace: String,

    #[arg(short, long, default_value = "10")]
    max_steps: usize,
}

fn main() {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    let args = Args::parse();

    println!("=== Testing Environment Only (NO Model) ===");
    println!("This isolates the stack overflow issue");
    println!();

    if let Err(e) = test_env_only(&args) {
        tracing::error!("Test failed: {}", e);
        std::process::exit(1);
    }
}

fn test_env_only(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = Path::new(&args.config);
    let trace_path = Path::new(&args.trace);

    println!("Loading config...");
    let config = FullTrainingConfig::from_file(config_path)?;
    println!("✓ Config loaded");

    println!("\nCreating environment...");
    let env = IOBufferEnv::new(config_path, trace_path, args.max_steps, None, None)?;
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
        let (next_obs, reward, done) = env.step(action);
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
