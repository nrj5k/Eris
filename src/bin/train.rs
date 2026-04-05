//! MINIMAL training binary - NO BURN MODELS
//!
//! Just runs the environment with a random agent to test tier visualization.
//! This version PROVES the environment works and shows tier utilization.

use std::path::Path;

use clap::Parser;
use rand::prelude::*;
use rand::rng;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use eris::config::{FullTrainingConfig, ModelArchitecture};
use eris::config_old::TierConfig as OldTierConfig;
use eris::env::IOBufferEnv;
use eris::tier::Tier;
use eris::{Environment, Space};

#[derive(Parser, Debug)]
#[command(
    name = "eris-train",
    about = "Train the Eris RL model for multi-tier storage optimization",
    version = "0.1.0"
)]
struct Args {
    /// Path to config file
    #[arg(short, long, default_value = "config/tiers.toml")]
    config: String,

    /// Path to trace CSV file
    #[arg(short, long, default_value = "recorder-csv/NWChem-64_combined.csv")]
    trace: String,

    /// Number of training episodes
    #[arg(short, long, default_value = "10")]
    episodes: usize,

    /// Maximum steps per episode
    #[arg(short, long, default_value = "100")]
    max_steps: usize,

    /// Backend (unused in this minimal version)
    #[arg(short, long, default_value = "cpu")]
    backend: String,
}

use rand::prelude::*;

fn run_training(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = Path::new(&args.config);
    let trace_path = Path::new(&args.trace);

    // Load config
    let config = FullTrainingConfig::from_file(config_path)?;

    // Create actual device based on backend
    let device = eris::device::create_device(&args.backend);

    // Verify backend is available
    if !eris::device::is_backend_available(&args.backend) {
        eprintln!("ERROR: Backend '{}' is not available.", args.backend);
        eprintln!("Make sure to compile with the appropriate feature flag:");
        eprintln!("  --features cpu     for CPU (NdArray)");
        eprintln!("  --features gpu     for GPU (Wgpu)");
        eprintln!("  --features nvidia  for CUDA");
        eprintln!("  --features amd     for ROCm");
        std::process::exit(1);
    }

    // Create environment
    let mut env = IOBufferEnv::new(config_path, trace_path, args.max_steps)?;

    // Create tier selector
    let tiers: Vec<_> = config
        .tiers
        .iter()
        .map(|tc| {
            Tier::new(OldTierConfig {
                name: tc.name.clone(),
                tier_id: tc.tier_id,
                capacity: tc.capacity,
                access_latency: tc.access_latency,
                description: tc.description.clone(),
            })
        })
        .collect();

    // Get action dimension
    let action_dim = env.action_space().n;

    // Log configuration
    tracing::info!("=== MINIMAL Training (Random Agent) ===");
    tracing::info!("Episodes: {}", args.episodes);
    tracing::info!("Max steps: {}", args.max_steps);
    tracing::info!("Action dim: {}", action_dim);
    tracing::info!(
        "Tiers: {} ({:?})",
        tiers.len(),
        config.tiers.iter().map(|t| &t.name).collect::<Vec<_>>()
    );

    // Training loop with RANDOM agent
    let mut episode_rewards = Vec::new();
    let mut epsilon: f32 = 1.0; // Start with random exploration
    let epsilon_decay = 0.995_f32;
    let epsilon_end = 0.01_f32;

    tracing::info!(
        "\n{:>10} {:>12} {:>10} {:>10}",
        "Episode",
        "Reward",
        "Avg",
        "Epsilon"
    );

    for episode in 0..args.episodes {
        let mut total_reward = 0.0;
        let mut done = false;
        let mut state = env.reset();
        let mut steps = 0;

        // Episode loop
        while !done && steps < args.max_steps {
            // Random action selection
            let action: usize = rng().random_range(0..action_dim);

            // Step environment - returns (observation, reward, done)
            let (next_obs, reward, is_done) = env.step(action);
            total_reward += reward;
            steps += 1;

            state = next_obs;
            done = is_done;
        }

        episode_rewards.push(total_reward as f32);

        // Compute running average
        let avg_reward: f32 = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;

        // Log progress
        tracing::info!(
            "{:>10} {:>12.1} {:>10.1} {:>10.3}",
            episode + 1,
            total_reward,
            avg_reward,
            epsilon
        );

        // Show tier utilization every 5 episodes
        if (episode + 1) % 5 == 0 {
            let tier_util = env.get_tier_utilization();
            tracing::info!("--- Tier Utilization (Episode {}) ---", episode + 1);
            for (tier, &utilization) in tiers.iter().zip(tier_util.iter()) {
                let bar_width = 40;
                let filled = (utilization * bar_width as f32) as usize;
                let empty = bar_width - filled;
                let bar: String = "█".repeat(filled) + &"░".repeat(empty);
                tracing::info!(
                    "{}: |{}| {:.1}%",
                    tier.config.name,
                    bar,
                    utilization * 100.0
                );
            }
            tracing::info!("");
        }

        // Decay epsilon
        epsilon = (epsilon * epsilon_decay).max(epsilon_end);
    }

    // Final tier utilization
    let tier_util = env.get_tier_utilization();
    tracing::info!("\n=== Final Tier Utilization ===");
    for (tier, &utilization) in tiers.iter().zip(tier_util.iter()) {
        let bar_width = 40;
        let filled = (utilization * bar_width as f32) as usize;
        let empty = bar_width - filled;
        let bar: String = "█".repeat(filled) + &"░".repeat(empty);
        tracing::info!(
            "{}: |{}| {:.1}%",
            tier.config.name,
            bar,
            utilization * 100.0
        );
    }

    // Summary
    tracing::info!("\n=== Training Summary ===");
    tracing::info!("Total episodes: {}", args.episodes);
    tracing::info!(
        "Average reward: {:.2}",
        episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
    );
    tracing::info!("Final epsilon: {:.3}", epsilon);

    Ok(())
}

fn main() {
    // Minimal stack - just basic function calls, no deep generics
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024) // 8 MB is plenty for this
        .spawn(|| {
            let args = Args::parse();

            // Initialize logging
            let subscriber = FmtSubscriber::builder()
                .with_max_level(Level::INFO)
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");

            if let Err(e) = run_training(&args) {
                tracing::error!("Training failed: {}", e);
                std::process::exit(1);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
