#![recursion_limit = "512"]

use std::path::Path;

use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use eris::config::Config;
use eris::env::IOBufferEnv;
use eris::models::CombinedModelConfig;
use eris::tier::Tier;
use eris::training::{train_agent, CombinedAgent, ConsoleMonitor, TrainingConfig};
use eris::{Environment, Space};

#[derive(Parser, Debug)]
#[command(
    name = "eris-train",
    about = "Train the Eris RL model for multi-tier storage optimization",
    version = "0.1.0"
)]
struct Args {
    /// Path to tier configuration file (TOML)
    #[arg(short, long, default_value = "config/tiers.toml")]
    config: String,

    /// Path to trace CSV file
    #[arg(short, long, default_value = "recorder-csv/NWChem-64_combined.csv")]
    trace: String,

    /// Path to output model file
    #[arg(short, long, default_value = "output/model.postcard")]
    output: String,

    /// Number of training episodes
    #[arg(short, long, default_value = "100")]
    episodes: usize,

    /// Maximum steps per episode
    #[arg(short, long, default_value = "1000")]
    max_steps: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    learning_rate: f64,

    /// Batch size for training
    #[arg(short = 'B', long, default_value = "512")]
    batch_size: usize,

    /// Gamma (discount factor)
    #[arg(short, long, default_value = "0.99")]
    gamma: f32,

    /// Backend to use for training (cpu or gpu)
    #[arg(short, long, default_value = "cpu")]
    backend: String,

    /// Enable verbose output with progress bar
    #[arg(short, long)]
    verbose: bool,
}

fn create_agent<B: AutodiffBackend>(
    training_config: TrainingConfig,
    model_config: CombinedModelConfig,
    device: B::Device,
) -> CombinedAgent<B> {
    CombinedAgent::new(training_config, model_config, device)
}

fn run_training(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = Path::new(&args.config);
    let trace_path = Path::new(&args.trace);
    let output_path = Path::new(&args.output);

    // Load config
    let config = Config::from_file(config_path)?;

    // Create environment with CLI max_steps
    let mut env = IOBufferEnv::new(config_path, trace_path, args.max_steps)?;

    // Create tier selector
    let tiers: Vec<_> = config.tier.iter().map(|tc| Tier::new(tc.clone())).collect();
    let tier_selector = eris::tier::TierSelector::new(tiers);

    // Get dimensions from environment
    let state_dim = env.observation_space().dim();
    let action_dim = env.action_space().n;
    let model_config = CombinedModelConfig::new(state_dim, 20, 128, action_dim);

    // Training configuration with CLI arguments
    let training_config = TrainingConfig {
        learning_rate: args.learning_rate,
        gamma: args.gamma,
        batch_size: args.batch_size,
        ..Default::default()
    };

    // Select backend and run training
    match args.backend.to_lowercase().as_str() {
        "cpu" | "ndarray" => {
            println!("Using CPU (NdArray) backend...");
            let device = burn::backend::ndarray::NdArrayDevice::Cpu;
            let mut agent =
                create_agent::<Autodiff<NdArray>>(training_config, model_config, device);

            let mut monitor = if args.verbose {
                Some(ConsoleMonitor::new(args.episodes).with_tier_configs(config.tier.clone()))
            } else {
                None
            };

            let result = train_agent(
                &mut env,
                &mut agent,
                args.episodes,
                &tier_selector,
                monitor.as_mut(),
            );

            tracing::info!("Training complete!");
            tracing::info!(
                "Average reward: {:.2}",
                result.episode_rewards.iter().sum::<f32>() / result.episode_rewards.len() as f32
            );

            // Save model
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            agent.save(output_path);

            tracing::info!("Model saved to {:?}", output_path);
        }
        "gpu" | "wgpu" => {
            println!("Using GPU (Wgpu) backend...");
            let device = burn::backend::wgpu::WgpuDevice::default();
            let mut agent = create_agent::<Autodiff<Wgpu>>(training_config, model_config, device);

            let mut monitor = if args.verbose {
                Some(ConsoleMonitor::new(args.episodes).with_tier_configs(config.tier.clone()))
            } else {
                None
            };

            let result = train_agent(
                &mut env,
                &mut agent,
                args.episodes,
                &tier_selector,
                monitor.as_mut(),
            );

            tracing::info!("Training complete!");
            tracing::info!(
                "Average reward: {:.2}",
                result.episode_rewards.iter().sum::<f32>() / result.episode_rewards.len() as f32
            );

            // Save model
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent).ok();
            }
            agent.save(output_path);

            tracing::info!("Model saved to {:?}", output_path);
        }
        _ => {
            return Err(format!("Unknown backend: {}. Use 'cpu' or 'gpu'.", args.backend).into());
        }
    }

    Ok(())
}

fn main() {
    let args = Args::parse();

    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    println!("=== Eris Training ===");
    println!("Config: {}", args.config);
    println!("Trace: {}", args.trace);
    println!("Episodes: {}", args.episodes);
    println!("Output: {}", args.output);
    println!("Max steps: {}", args.max_steps);
    println!("Learning rate: {}", args.learning_rate);
    println!("Batch size: {}", args.batch_size);
    println!("Gamma: {}", args.gamma);
    println!("Backend: {}", args.backend);
    println!();

    if let Err(e) = run_training(&args) {
        tracing::error!("Training failed: {}", e);
        std::process::exit(1);
    }
}
