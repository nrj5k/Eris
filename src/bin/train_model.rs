//! Generic model training binary with checkpoint support.
//!
//! Usage:
//!   train_model --model dqn --episodes 100 --max-steps 100
//!
//! NOTE: Backend selection is done via Cargo features (compile-time):
//!   cargo run --bin train_model --features cpu-only -- --model dqn
//!   cargo run --bin train_model --features wgpu-only -- --model dqn
//!
//! The --backend CLI flag validates that you're using the expected backend.

use clap::Parser;
use eris::training::CombinedAgent;

// ============================================================================
// BACKEND TYPE DEFINITIONS (Compile-time selection)
// ============================================================================

#[cfg(feature = "cpu-only")]
type Backend = burn::backend::NdArray<f32>;

#[cfg(feature = "wgpu-only")]
type Backend = burn::backend::Wgpu<f32, i32>;

#[cfg(feature = "cuda-only")]
type Backend = burn::backend::Cuda<f32, i32>;

#[cfg(feature = "rocm-only")]
type Backend = burn::backend::Rocm<f32, i32>;

#[derive(Parser, Clone)]
#[command(name = "train_model")]
#[command(about = "Train ONE model: dqn, cbandit, or combined")]
struct Args {
    /// Which model to train: dqn, cbandit, or combined
    #[arg(short, long, default_value = "dqn")]
    model: String,

    /// Number of episodes
    #[arg(short, long, default_value = "100")]
    episodes: usize,

    /// Max steps per episode
    #[arg(short = 's', long, default_value = "100")]
    max_steps: usize,

    /// Batch size for training
    #[arg(short = 'B', long, default_value = "512")]
    batch_size: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.001")]
    learning_rate: f64,

    /// Backend selection (informational - validates against compiled backend)
    #[arg(long, default_value = "cpu")]
    backend: String,
}

// ============================================================================
// ENTRY POINT
// ============================================================================

fn main() {
    let args = Args::parse();

    // Validate backend arg matches compiled backend
    let compiled_backend = get_compiled_backend();
    if args.backend.to_lowercase() != compiled_backend {
        eprintln!(
            "Error: Requested backend '{}' doesn't match compiled backend '{}'",
            args.backend, compiled_backend
        );
        eprintln!(
            "Recompile with: cargo build --features {}-only",
            args.backend.to_lowercase()
        );
        std::process::exit(1);
    }

    println!("=== Training Model: {} ===", args.model);
    println!("Episodes: {}", args.episodes);
    println!("Max steps: {}", args.max_steps);
    println!("Batch size: {}", args.batch_size);
    println!("Learning rate: {}", args.learning_rate);
    println!("Backend: {} ✓", compiled_backend);
    println!();

    // Spawn training in a thread with increased stack size (512MB for Burn)
    use std::thread;
    let stack_size: usize = 512 * 1024 * 1024; // 512MB

    let result = thread::Builder::new()
        .stack_size(stack_size)
        .spawn(move || train_model_generic(&args))
        .expect("Failed to spawn thread")
        .join();

    if let Err(e) = result {
        eprintln!("Training thread panicked: {:?}", e);
        std::process::exit(1);
    }
}

// ============================================================================
// BACKEND SELECTION
// ============================================================================

/// Get compiled backend name
fn get_compiled_backend() -> &'static str {
    #[cfg(feature = "cpu-only")]
    {
        "cpu"
    }

    #[cfg(feature = "wgpu-only")]
    {
        "wgpu"
    }

    #[cfg(feature = "cuda-only")]
    {
        "cuda"
    }

    #[cfg(feature = "rocm-only")]
    {
        "rocm"
    }

    #[cfg(not(any(
        feature = "cpu-only",
        feature = "wgpu-only",
        feature = "cuda-only",
        feature = "rocm-only"
    )))]
    {
        "none"
    }
}

/// Generic training function that dispatches to the correct backend.
fn train_model_generic(args: &Args) -> Result<(), String> {
    let model_type = args.model.as_str();
    println!("Initializing {} training...", model_type);
    eprintln!("DEBUG: Starting train_model_generic for {}", model_type);

    // Setup device and backend based on compiled features
    #[cfg(feature = "cpu-only")]
    {
        use burn::backend::ndarray::NdArrayDevice;
        use burn::backend::Autodiff;

        type AutodiffBackend = Autodiff<Backend>;
        let device = NdArrayDevice::default();
        println!("Running on CPU (NdArray)");
        run_training::<AutodiffBackend>(args, device)
    }

    #[cfg(feature = "wgpu-only")]
    {
        use burn::backend::wgpu::WgpuDevice;
        use burn::backend::Autodiff;

        type AutodiffBackend = Autodiff<Backend>;
        let device = WgpuDevice::default();
        println!("Running on GPU (Wgpu)");
        run_training::<AutodiffBackend>(args, device)
    }

    #[cfg(feature = "cuda-only")]
    {
        use burn::backend::cuda::CudaDevice;
        use burn::backend::Autodiff;

        type AutodiffBackend = Autodiff<Backend>;
        let device = CudaDevice::default();
        println!("Running on CUDA");
        run_training::<AutodiffBackend>(args, device)
    }

    #[cfg(feature = "rocm-only")]
    {
        use burn::backend::rocm::RocmDevice;
        use burn::backend::Autodiff;

        type AutodiffBackend = Autodiff<Backend>;
        let device = RocmDevice::default();
        println!("Running on ROCm");
        run_training::<AutodiffBackend>(args, device)
    }

    #[cfg(not(any(
        feature = "cpu-only",
        feature = "wgpu-only",
        feature = "cuda-only",
        feature = "rocm-only"
    )))]
    {
        Err("No backend feature enabled. Please compile with: --features cpu-only|wgpu-only|cuda-only|rocm-only".to_string())
    }
}

// ============================================================================
// GENERIC TRAIN FUNCTION
// ============================================================================

/// Generic training function that works with any backend.
fn run_training<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    device: B::Device,
) -> Result<(), String> {
    use eris::env::Environment;
    use eris::env::IOBufferEnv;
    use eris::space::Space;
    use eris::training::Transition;
    use std::path::Path;

    let model_type = args.model.as_str();

    // Create environment
    let config_path = Path::new("config/tiers.toml");
    let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");
    eprintln!("DEBUG: Creating environment");
    let mut env = IOBufferEnv::new(config_path, trace_path, args.max_steps)
        .map_err(|e| format!("Failed to create environment: {}", e))?;
    eprintln!("DEBUG: Environment created");

    // Get dimensions from environment
    let state_dim = env.observation_space().dim();
    let action_dim = env.action_space().n;
    eprintln!("DEBUG: state_dim={}, action_dim={}", state_dim, action_dim);

    // MODEL-SPECIFIC SETUP
    let mut agent = match model_type {
        "dqn" => setup_dqn_agent::<B>(args, state_dim, action_dim, &device)
            .map_err(|e| format!("DQN setup failed: {}", e))?,
        "cbandit" | "bandit" => setup_cbandit_agent::<B>(args, state_dim, action_dim, &device)
            .map_err(|e| format!("CBandit setup failed: {}", e))?,
        "combined" => setup_combined_agent::<B>(args, state_dim, action_dim, &device)
            .map_err(|e| format!("Combined setup failed: {}", e))?,
        _ => {
            return Err(format!(
                "Unknown model type '{}'. Available: dqn, cbandit, combined",
                model_type
            ));
        }
    };

    println!("Starting training for {} episodes...", args.episodes);
    println!("State dim: {}, Action dim: {}", state_dim, action_dim);
    println!("Model: {}", model_type);
    println!();

    // GENERIC TRAINING LOOP
    let mut episode_rewards = Vec::with_capacity(args.episodes);

    for episode in 0..args.episodes {
        let mut state = env.reset();
        let mut episode_reward = 0.0;
        let mut done = false;
        let mut step = 0;

        while !done && step < args.max_steps {
            // Select action using epsilon-greedy policy
            let action = select_action(&agent, &state, &env, agent.epsilon);

            // Step environment
            let (next_state, reward, done_flag) = env.step(action);
            episode_reward += reward;

            // Store transition
            agent.buffer.push(Transition {
                state: state.iter().map(|&x| x as f32).collect(),
                action,
                reward: reward as f32,
                next_state: next_state.iter().map(|&x| x as f32).collect(),
                done: done_flag,
            });

            // Train if enough samples
            #[allow(deprecated)]
            if agent.buffer.len() >= agent.config.batch_size {
                if let Some(batch) = agent.buffer.sample_batch(agent.config.batch_size) {
                    let _loss = agent.train_step(batch);
                }
            }

            state = next_state;
            done = done_flag;
            step += 1;
        }

        // Decay epsilon
        agent.epsilon = (agent.epsilon * agent.config.epsilon_decay).max(agent.config.epsilon_end);

        episode_rewards.push(episode_reward);

        // Print progress
        println!(
            "Episode {}/{}: reward={:.2}, steps={}, epsilon={:.3}",
            episode + 1,
            args.episodes,
            episode_reward,
            step,
            agent.epsilon
        );

        // SAVE CHECKPOINT every 10 episodes
        if (episode + 1) % 10 == 0 || episode == args.episodes - 1 {
            let avg_reward = if !episode_rewards.is_empty() {
                episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64
            } else {
                0.0
            };
            let checkpoint_path = format!("checkpoints/{}_episode_{}", model_type, episode + 1);
            save_checkpoint(&agent, &checkpoint_path, episode + 1, avg_reward as f32)
                .map_err(|e| format!("Failed to save checkpoint: {}", e))?;
            println!("  Saved checkpoint: {}.mpk", checkpoint_path);
        }
    }

    // FINAL SAVE
    let avg_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
    let final_path = format!("checkpoints/{}_final", model_type);
    save_checkpoint(&agent, &final_path, args.episodes, avg_reward as f32)
        .map_err(|e| format!("Failed to save final checkpoint: {}", e))?;

    // PRINT SUMMARY
    println!("\nTraining complete!");
    println!("Model: {}", model_type);
    println!("Total episodes: {}", args.episodes);
    println!("Average reward: {:.2}", avg_reward);
    println!("Final epsilon: {:.3}", agent.epsilon);
    println!("Checkpoints saved to: checkpoints/{}_*.mpk", model_type);

    Ok(())
}

// ============================================================================
// ACTION SELECTION
// ============================================================================

/// Select action using epsilon-greedy policy.
fn select_action<B: burn::tensor::backend::AutodiffBackend>(
    agent: &CombinedAgent<B>,
    state: &[f64],
    env: &eris::env::IOBufferEnv,
    epsilon: f32,
) -> usize {
    use burn::tensor::{Tensor, TensorData};
    use eris::config_old::Config;
    use eris::env::Environment;
    use eris::tier::{Tier, TierSelector};
    use rand::prelude::*;
    use std::path::Path;

    // Epsilon-greedy: random action with probability epsilon
    if rand::random::<f32>() < epsilon {
        let action_space = env.action_space();
        return rand::rng().random_range(0..action_space.n);
    }

    // Convert state to tensor and get Q-values from model
    let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();
    let state_data = TensorData::new(state_f32.clone(), [1, state_f32.len()]);
    let state_tensor: Tensor<B, 2> = Tensor::from_data(state_data.convert::<f32>(), &agent.device);

    // Create TierSelector for action mapping
    let config_path = Path::new("config/tiers.toml");
    let tier_configs = match Config::from_file(config_path) {
        Ok(cfg) => cfg.tier,
        Err(_) => (0..5)
            .map(|i| eris::config_old::TierConfig {
                name: format!("tier_{}", i),
                tier_id: i as u32,
                capacity: 100.0,
                access_latency: 0.01,
                description: String::new(),
            })
            .collect(),
    };
    let tier_selector = TierSelector::new(tier_configs.into_iter().map(Tier::new).collect());

    // Forward pass through model
    agent
        .model
        .select_action(state_tensor, &tier_selector, epsilon)
}

// ============================================================================
// MODEL-SPECIFIC SETUP
// ============================================================================

fn setup_dqn_agent<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    state_dim: usize,
    action_dim: usize,
    device: &B::Device,
) -> Result<CombinedAgent<B>, Box<dyn std::error::Error>> {
    use eris::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};
    use eris::model::Activation;
    use eris::training::{CombinedAgent, TrainingConfig};

    let feature_dim = 20;

    let bandit_config = BanditConfig::builder()
        .input_dim(state_dim)
        .hidden_layers(vec![64, 128])
        .feature_dim(feature_dim)
        .activation(Activation::Sigmoid)
        .build()?;

    let dqn_config = DQNConfig::builder()
        .input_dim(feature_dim)
        .hidden_layers(vec![128, 128])
        .action_dim(action_dim)
        .dueling(true)
        .build()?;

    let model_config = CombinedBanditDQNConfig::builder()
        .bandit(bandit_config)
        .dqn(dqn_config)
        .build()?;

    let training_config = TrainingConfig {
        learning_rate: args.learning_rate,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.01,
        epsilon_decay: 0.995,
        batch_size: args.batch_size,
        buffer_capacity: 10_000,
        target_update_freq: 10,
        tau: 0.005,
        backend: "ndarray".to_string(),
        checkpoint_interval: 10,
        max_gradient_norm: 1.0,
    };

    let agent = CombinedAgent::new(training_config, model_config, device.clone());

    Ok(agent)
}

fn setup_cbandit_agent<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    state_dim: usize,
    action_dim: usize,
    device: &B::Device,
) -> Result<CombinedAgent<B>, Box<dyn std::error::Error>> {
    use eris::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};
    use eris::training::TrainingConfig;

    println!("Setting up Contextual Bandit model...");

    let feature_dim = action_dim;

    let bandit_config = BanditConfig::builder()
        .input_dim(state_dim)
        .hidden_layers(vec![64, 128])
        .feature_dim(feature_dim)
        .build()?;

    let dqn_config = DQNConfig::builder()
        .input_dim(feature_dim)
        .hidden_layers(vec![128, 128])
        .action_dim(action_dim)
        .dueling(true)
        .build()?;

    let model_config = CombinedBanditDQNConfig::builder()
        .bandit(bandit_config)
        .dqn(dqn_config)
        .build()?;

    let training_config = TrainingConfig {
        learning_rate: args.learning_rate,
        gamma: 0.0,
        epsilon_start: 0.5,
        epsilon_end: 0.01,
        epsilon_decay: 0.99,
        batch_size: args.batch_size,
        buffer_capacity: 10_000,
        target_update_freq: 0,
        tau: 0.0,
        backend: "ndarray".to_string(),
        checkpoint_interval: 10,
        max_gradient_norm: 1.0,
    };

    let agent = CombinedAgent::new(training_config, model_config, device.clone());

    println!("Contextual Bandit agent ready!");
    println!("  Input: {} features", state_dim);
    println!("  Output: {} action importance scores", action_dim);

    Ok(agent)
}

fn setup_combined_agent<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    state_dim: usize,
    action_dim: usize,
    device: &B::Device,
) -> Result<CombinedAgent<B>, Box<dyn std::error::Error>> {
    use eris::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};
    use eris::training::TrainingConfig;

    println!("Setting up Combined Model (Bandit + DQN)...");

    let bandit_config = BanditConfig::builder()
        .input_dim(state_dim)
        .hidden_layers(vec![64, 128])
        .feature_dim(20)
        .build()?;

    let dqn_config = DQNConfig::builder()
        .input_dim(20)
        .action_dim(action_dim)
        .hidden_layers(vec![128, 128])
        .dueling(true)
        .build()?;

    let model_config = CombinedBanditDQNConfig::builder()
        .bandit(bandit_config)
        .dqn(dqn_config)
        .build()?;

    let training_config = TrainingConfig {
        learning_rate: args.learning_rate,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.01,
        epsilon_decay: 0.995,
        batch_size: args.batch_size,
        buffer_capacity: 10_000,
        target_update_freq: 10,
        tau: 0.005,
        backend: "ndarray".to_string(),
        checkpoint_interval: 10,
        max_gradient_norm: 1.0,
    };

    let agent = CombinedAgent::<B>::new(training_config, model_config, device.clone());

    println!("Combined agent ready!");
    println!("  State dim: {}", state_dim);
    println!("  Bandit: {} -> [64,128] -> 20 features", state_dim);
    println!("  DQN: 20 -> [128,128] -> {} Q-values", action_dim);

    Ok(agent)
}

// ============================================================================
// CHECKPOINT SAVING
// ============================================================================

fn save_checkpoint<B: burn::tensor::backend::AutodiffBackend>(
    agent: &CombinedAgent<B>,
    path: &str,
    episode: usize,
    avg_reward: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    fs::create_dir_all("checkpoints")?;
    agent.save_checkpoint(path, episode, avg_reward)?;

    Ok(())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_parsing() {
        let args = Args::try_parse_from(&["train_model", "--model", "dqn", "--episodes", "50"]);
        assert!(args.is_ok());
        let args = args.unwrap();
        assert_eq!(args.model, "dqn");
        assert_eq!(args.episodes, 50);
    }

    #[test]
    fn test_args_defaults() {
        let args = Args::try_parse_from(&["train_model"]);
        assert!(args.is_ok());
        let args = args.unwrap();
        assert_eq!(args.model, "dqn");
        assert_eq!(args.episodes, 100);
        assert_eq!(args.max_steps, 100);
        assert_eq!(args.batch_size, 512);
    }

    #[test]
    fn test_compiled_backend() {
        let backend = get_compiled_backend();
        // Should be one of the valid backends
        assert!(["cpu", "wgpu", "cuda", "rocm", "none"].contains(&backend));
    }
}
