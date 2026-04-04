#![recursion_limit = "512"]

use std::path::Path;

use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use eris::config::{FullTrainingConfig, ModelArchitecture};
use eris::config_old::TierConfig as OldTierConfig;
use eris::env::IOBufferEnv;
use eris::models::CombinedModelConfig;
use eris::tier::Tier;
use eris::training::{train_agent_with_metrics, CombinedAgent, TrainingConfig};
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

    /// Path to output model file
    #[arg(short, long, default_value = "output/model.postcard")]
    output: String,

    /// Number of training episodes (overrides config)
    #[arg(short, long)]
    episodes: Option<usize>,

    /// Maximum steps per episode (overrides config)
    #[arg(short, long)]
    max_steps: Option<usize>,

    /// Learning rate (overrides config)
    #[arg(short, long)]
    learning_rate: Option<f64>,

    /// Batch size for training (overrides config)
    #[arg(short = 'B', long)]
    batch_size: Option<usize>,

    /// Gamma discount factor (overrides config)
    #[arg(short, long)]
    gamma: Option<f32>,

    /// Backend to use: cpu, gpu, torch, cuda, rocm
    #[arg(short, long, default_value = "cpu")]
    backend: String,

    /// Model architecture: dueling_dqn, bandit_dqn, simple_dqn
    #[arg(long)]
    model: Option<String>,

    /// Device ID for GPU/accelerator
    #[arg(long, default_value = "0")]
    device_id: usize,
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

    // Load config (supports both old and new format)
    let mut config = FullTrainingConfig::from_file(config_path)?;

    // Apply CLI overrides
    config.apply_overrides(
        args.episodes,
        args.max_steps,
        args.batch_size,
        args.learning_rate,
        args.gamma,
        Some(args.backend.clone()),
        args.model.clone(),
    );

    // Create environment with config max_steps
    let mut env = IOBufferEnv::new(config_path, trace_path, config.training.max_steps)?;

    // Create tier selector - convert from new config format
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
    let tier_selector = eris::tier::TierSelector::new(tiers);

    // Get dimensions from environment
    let state_dim = env.observation_space().dim();
    let action_dim = env.action_space().n;

    // Create model config based on config
    let model_config = match config.model.architecture {
        ModelArchitecture::DuelingDQN | ModelArchitecture::SimpleDQN => CombinedModelConfig::new(
            state_dim,
            config.model.feature_dim,
            config.model.dqn_hidden[0],
            action_dim,
        ),
        ModelArchitecture::BanditDQN => CombinedModelConfig::new(
            state_dim,
            config.model.feature_dim,
            config.model.dqn_hidden[0],
            action_dim,
        ),
    };

    // Training configuration from config
    let training_config = TrainingConfig {
        learning_rate: config.training.learning_rate,
        gamma: config.training.gamma,
        batch_size: config.training.batch_size,
        ..Default::default()
    };

    // Print configuration
    println!("=== Configuration ===");
    println!("Model: {}", config.model.architecture);
    println!("Backend: {}", config.backend.backend_type);
    println!("Episodes: {}", config.training.episodes);
    println!("Max steps: {}", config.training.max_steps);
    println!("Batch size: {}", config.training.batch_size);
    println!("Learning rate: {}", config.training.learning_rate);
    println!("Gamma: {}", config.training.gamma);
    println!(
        "Epsilon: [{}, {}], decay={}",
        config.training.epsilon_start, config.training.epsilon_end, config.training.epsilon_decay
    );
    println!("Replay buffer: {}", config.training.replay_buffer_size);
    println!();

    // Select backend and run training
    match config.backend.backend_type {
        eris::config::BackendType::Cpu => {
            println!("Using CPU (NdArray) backend...");
            let device = burn::backend::ndarray::NdArrayDevice::Cpu;
            let mut agent =
                create_agent::<Autodiff<NdArray>>(training_config, model_config, device);
            run_training_loop(&mut env, &mut agent, args, &tier_selector, output_path)?;
        }
        eris::config::BackendType::Gpu => {
            println!("Using GPU (Wgpu) backend...");
            let device = burn::backend::wgpu::WgpuDevice::default();
            let mut agent = create_agent::<Autodiff<Wgpu>>(training_config, model_config, device);
            run_training_loop(&mut env, &mut agent, args, &tier_selector, output_path)?;
        }
        eris::config::BackendType::Torch => {
            #[cfg(feature = "torch")]
            {
                println!("Using LibTorch backend...");
                use burn::backend::libtorch::LibTorchDevice;
                use burn::backend::LibTorch;
                let device = LibTorchDevice::Cpu;
                let mut agent =
                    create_agent::<Autodiff<LibTorch>>(training_config, model_config, device);
                run_training_loop(&mut env, &mut agent, args, &tier_selector, output_path)?;
            }
            #[cfg(not(feature = "torch"))]
            {
                eprintln!("ERROR: Torch backend not enabled. Recompile with --features torch");
                std::process::exit(1);
            }
        }
        eris::config::BackendType::Cuda => {
            #[cfg(feature = "cuda")]
            {
                println!("Using CUDA backend...");
                use burn::backend::Cuda;
                let device = Cuda::device(config.backend.device_id);
                let mut agent =
                    create_agent::<Autodiff<Cuda>>(training_config, model_config, device);
                run_training_loop(&mut env, &mut agent, args, &tier_selector, output_path)?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("ERROR: CUDA backend not enabled. Recompile with --features cuda");
                std::process::exit(1);
            }
        }
        eris::config::BackendType::Rocm => {
            #[cfg(feature = "rocm")]
            {
                println!("Using ROCm backend...");
                use burn::backend::Rocm;
                let device = Rocm::device(config.backend.device_id);
                let mut agent =
                    create_agent::<Autodiff<Rocm>>(training_config, model_config, device);
                run_training_loop(&mut env, &mut agent, args, &tier_selector, output_path)?;
            }
            #[cfg(not(feature = "rocm"))]
            {
                eprintln!("ERROR: ROCm backend not enabled. Recompile with --features rocm");
                std::process::exit(1);
            }
        }
    }

    Ok(())
}

fn run_training_loop<B: AutodiffBackend>(
    env: &mut IOBufferEnv,
    agent: &mut CombinedAgent<B>,
    args: &Args,
    tier_selector: &eris::tier::TierSelector,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load full config for episode count
    let config_path = Path::new(&args.config);
    let config = FullTrainingConfig::from_file(config_path)?;
    let episodes = args.episodes.unwrap_or(config.training.episodes);

    let result = train_agent_with_metrics(env, agent, episodes, tier_selector);

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

    Ok(())
}

fn main() {
    let args = Args::parse();

    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Failed to set tracing subscriber");

    if let Err(e) = run_training(&args) {
        tracing::error!("Training failed: {}", e);
        std::process::exit(1);
    }
}
