#![recursion_limit = "256"]

//! Generic model training binary with checkpoint support.
//!
//! Usage:
//!   train_model --model dqn --episodes 100 --max-steps 100
//!
//! Backend selection is done via --backend flag (runtime):
//!   train_model --model dqn --backend cpu
//!   train_model --model dqn --backend wgpu
//!   train_model --model dqn --backend cuda
//!   train_model --model dqn --backend rocm

use burn::tensor::TensorData;
use clap::{Parser, ValueEnum};
use eris::device::{available_backends, Device};

use eris::training::CombinedAgent;
use std::path::PathBuf;

#[derive(Clone, Debug, ValueEnum)]
enum ModelType {
    /// Metis: Combined DQN + Bandit
    Metis,
    /// Cacheus: Contextual Multi-Armed Bandit
    Cacheus,
    /// Catcher: DDPG Actor-Critic
    Catcher,
    /// DQN: Pure Deep Q-Network
    Dqn,
    /// Bandit: Standalone Contextual Bandit
    Bandit,
}

/// Exploration strategy for action selection
#[derive(Clone, Debug, ValueEnum)]
enum ExplorationStrategy {
    /// Epsilon-greedy: random with probability epsilon
    EpsilonGreedy,
    /// Thompson Sampling: Bayesian posterior sampling
    ThompsonSampling,
    /// Upper Confidence Bound: theoretically optimal exploration
    Ucb,
}

/// Validate batch size is a multiple of 32 and within reasonable bounds
fn validate_batch_size(s: &str) -> Result<usize, String> {
    let size: usize = s.parse().map_err(|_| "Invalid number")?;
    if size % 32 != 0 {
        return Err(format!(
            "Batch size must be multiple of 32 for GPU warp alignment (got {})",
            size
        ));
    }
    if size < 32 {
        return Err(format!("Batch size must be at least 32 (got {})", size));
    }
    if size > 16384 {
        return Err(format!("Batch size should not exceed 16384 (got {})", size));
    }
    Ok(size)
}

#[derive(Parser, Clone)]
#[command(name = "train_model")]
#[command(about = "Train cache policy: metis, cacheus, or catcher")]
struct Args {
    /// Which policy to train
    #[arg(short, long, value_enum, default_value = "metis")]
    model: ModelType,

    /// Number of episodes
    #[arg(short, long, default_value = "100")]
    episodes: usize,

    /// Max steps per episode
    #[arg(short = 's', long, default_value = "100")]
    max_steps: usize,

    /// Batch size for training (must be multiple of 32 for GPU warp alignment)
    #[arg(short = 'B', long, default_value = "2048", value_parser = validate_batch_size)]
    batch_size: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.0001")]
    learning_rate: f64,

    /// Backend selection at runtime: cpu, wgpu, cuda, rocm
    #[arg(long, default_value = "cpu")]
    backend: String,

    /// Path to configuration file
    #[arg(short, long, default_value = "config/tiers.toml")]
    config: PathBuf,

    /// Number of parallel environments
    #[arg(long, default_value = "16")]
    num_envs: usize,

    /// Replay buffer capacity
    ///
    /// For large batch sizes with gradient accumulation, ensure buffer_capacity >= batch_size * accumulation_steps.
    /// Example: batch_size=4096 with 4x accumulation requires buffer_capacity >= 16,384
    /// Default (100,000) accommodates most use cases safely
    #[arg(long, default_value = "100000")]
    buffer_capacity: usize,

    /// Exploration strategy for DQN and Bandit policies
    #[arg(long, value_enum, default_value = "epsilon-greedy")]
    exploration: ExplorationStrategy,

    /// Epsilon start (for epsilon-greedy)
    #[arg(long, default_value = "1.0")]
    epsilon_start: f32,

    /// Epsilon end (for epsilon-greedy)
    #[arg(long, default_value = "0.01")]
    epsilon_end: f32,

    /// Epsilon decay (for epsilon-greedy)
    #[arg(long, default_value = "0.995")]
    epsilon_decay: f32,

    /// UCB coefficient (for UCB)
    #[arg(long, default_value = "2.0")]
    ucb_c: f32,

    /// Thompson sampling prior std (for Thompson sampling)
    #[arg(long, default_value = "1.0")]
    thompson_std: f32,

    /// Thompson sampling prior mean (for Thompson sampling)
    #[arg(long, default_value = "0.0")]
    thompson_mean: f32,

    /// Number of DataLoader worker threads (0 = single-threaded)
    ///
    /// For GPU training with VecEnv (16 parallel environments), the optimal
    /// value is 2 workers to minimize threading contention while maintaining
    /// some batching parallelism. Data is already GPU-resident via TensorRingBuffer.
    ///
    /// CPU-only training or single-env training may benefit from more workers.
    /// Use CLI override: --num_workers 4 for CPU-heavy preprocessing.
    #[arg(long, default_value = "2")]
    num_workers: usize,
}

// ============================================================================
// EXPLORATION CONFIG HELPER
// ============================================================================

/// Convert CLI exploration args to ExplorationConfig
fn create_exploration_config(args: &Args) -> eris::policies::ExplorationConfig {
    use eris::policies::ExplorationConfig;

    match args.exploration {
        ExplorationStrategy::EpsilonGreedy => ExplorationConfig::EpsilonGreedy {
            epsilon_start: args.epsilon_start,
            epsilon_end: args.epsilon_end,
            epsilon_decay: args.epsilon_decay,
        },
        ExplorationStrategy::ThompsonSampling => ExplorationConfig::ThompsonSampling {
            prior_mean: args.thompson_mean,
            prior_std: args.thompson_std,
        },
        ExplorationStrategy::Ucb => ExplorationConfig::UCB { c: args.ucb_c },
    }
}

// ============================================================================
// ENTRY POINT
// ============================================================================

fn main() {
    let args = Args::parse();

    // Validate config file exists
    if !args.config.exists() {
        eprintln!(
            "Error: Configuration file not found: {}",
            args.config.display()
        );
        eprintln!(
            "Create it with: mkdir -p config && touch {}",
            args.config.display()
        );
        std::process::exit(1);
    }

    // Validate config is a file (not a directory)
    if !args.config.is_file() {
        eprintln!(
            "Error: Config path is not a file: {}",
            args.config.display()
        );
        std::process::exit(1);
    }

    // Parse backend selection at runtime
    let device = match Device::from_str(&args.backend) {
        Some(d) => d,
        None => {
            eprintln!("Error: Unknown backend '{}'", args.backend);
            eprintln!("Available backends: {}", available_backends().join(", "));
            std::process::exit(1);
        }
    };
    println!("Using backend: {}", device.name());

    // GPU DIAGNOSTIC: Verify compiled backend features
    println!("🔍 GPU DIAGNOSTIC: Compiled features:");
    #[cfg(feature = "cuda")]
    println!("  ✅ cuda feature: ENABLED");
    #[cfg(not(feature = "cuda"))]
    println!("  ❌ cuda feature: DISABLED (binary compiled without CUDA support!)");
    #[cfg(feature = "wgpu")]
    println!("  ✅ wgpu feature: ENABLED");
    #[cfg(not(feature = "wgpu"))]
    println!("  ⚠️  wgpu feature: DISABLED");
    #[cfg(feature = "cpu")]
    println!("  ✅ cpu feature: ENABLED");
    #[cfg(not(feature = "cpu"))]
    println!("  ⚠️  cpu feature: DISABLED");

    // GPU DIAGNOSTIC: Show which Device variant was selected at runtime
    match &device {
        #[cfg(feature = "cuda")]
        Device::Cuda(ref dev) => println!(
            "🔍 GPU DIAGNOSTIC: Device::Cuda variant selected - CUDA device: {:?}",
            dev
        ),
        #[cfg(feature = "wgpu")]
        Device::Wgpu(ref dev) => println!(
            "🔍 GPU DIAGNOSTIC: Device::Wgpu variant selected - WGPU device: {:?}",
            dev
        ),
        #[cfg(feature = "cpu")]
        Device::Cpu(ref dev) => println!(
            "🔍 GPU DIAGNOSTIC: Device::Cpu variant selected - CPU device: {:?}",
            dev
        ),
        #[cfg(feature = "rocm")]
        Device::Rocm(ref dev) => println!(
            "🔍 GPU DIAGNOSTIC: Device::Rocm variant selected - ROCm device: {:?}",
            dev
        ),
    }

    let model_name = format!("{:?}", args.model).to_lowercase();
    println!("=== Training Model: {} ===", model_name);
    println!("Episodes: {}", args.episodes);
    println!("Max steps: {}", args.max_steps);
    println!("Batch size: {}", args.batch_size);
    println!("Learning rate: {}", args.learning_rate);
    println!("Backend: {} ✓", device.name());
    println!();

    // Spawn training in a thread with increased stack size (512MB for Burn)
    use std::thread;
    let stack_size: usize = 512 * 1024 * 1024; // 512MB

    let result = thread::Builder::new()
        .stack_size(stack_size)
        .spawn(move || train_model_generic(&args, device))
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

/// Generic training function that dispatches to the correct backend.
fn train_model_generic(args: &Args, device: Device) -> Result<(), String> {
    let model_name = format!("{:?}", args.model).to_lowercase();
    println!("Initializing {} training...", model_name);
    #[cfg(debug_assertions)]
    eprintln!("DEBUG: Starting train_model_generic for {}", model_name);

    // Check if using Cacheus (tabular, no backend needed)
    if matches!(args.model, ModelType::Cacheus) {
        return run_cacheus_training(args);
    }

    // Create exploration config for DQN and Bandit
    let exploration = create_exploration_config(args);

    // Check if using Catcher (DDPG - requires backend)
    if matches!(args.model, ModelType::Catcher) {
        // Dispatch to Catcher training with appropriate backend
        return match device {
            #[cfg(feature = "cpu")]
            Device::Cpu(ndarray_device) => {
                use burn::backend::{Autodiff, NdArray};
                type Backend = Autodiff<NdArray<f32>>;
                println!("Running on CPU (NdArray)");
                run_catcher_training::<Backend>(args, ndarray_device)
            }

            #[cfg(feature = "wgpu")]
            Device::Wgpu(wgpu_device) => {
                use burn::backend::{Autodiff, Wgpu};
                type Backend = Autodiff<Wgpu<f32, i32>>;
                println!("Running on GPU (Wgpu)");
                run_catcher_training::<Backend>(args, wgpu_device)
            }

            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_device) => {
                use burn::backend::{Autodiff, Cuda};
                type Backend = Autodiff<Cuda<f32, i32>>;
                println!("Running on CUDA");
                run_catcher_training::<Backend>(args, cuda_device)
            }

            #[cfg(feature = "rocm")]
            Device::Rocm(rocm_device) => {
                use burn::backend::{Autodiff, Rocm};
                type Backend = Autodiff<Rocm<f32, i32>>;
                println!("Running on ROCm");
                run_catcher_training::<Backend>(args, rocm_device)
            }
        };
    }

    // Check if using DQN (requires backend)
    if matches!(args.model, ModelType::Dqn) {
        return match device {
            #[cfg(feature = "cpu")]
            Device::Cpu(ndarray_device) => {
                use burn::backend::{Autodiff, NdArray};
                type Backend = Autodiff<NdArray<f32>>;
                println!("Running DQN on CPU (NdArray)");
                run_dqn_training::<Backend>(args, ndarray_device, exploration.clone())
            }

            #[cfg(feature = "wgpu")]
            Device::Wgpu(wgpu_device) => {
                use burn::backend::{Autodiff, Wgpu};
                type Backend = Autodiff<Wgpu<f32, i32>>;
                println!("Running DQN on GPU (Wgpu)");
                run_dqn_training::<Backend>(args, wgpu_device, exploration.clone())
            }

            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_device) => {
                use burn::backend::{Autodiff, Cuda};
                type Backend = Autodiff<Cuda<f32, i32>>;

                // 🔥 DEBUG: DQN CUDA PATH ACTIVE
                eprintln!("🔥 DQN CUDA PATH ACTIVE 🔥");
                eprintln!("🔥 Backend type: {}", std::any::type_name::<Backend>());
                eprintln!("🔥 CUDA device: {:?}", cuda_device);

                println!("Running DQN on CUDA");
                // GPU DIAGNOSTIC: Verify the Backend type at compile time
                println!(
                    "🔍 GPU DIAGNOSTIC: Backend type = {}",
                    std::any::type_name::<Backend>()
                );
                // Quick sanity check: time a small tensor operation
                let diag_start = std::time::Instant::now();
                let _test_tensor =
                    burn::tensor::Tensor::<Backend, 2>::zeros([64, 128], &cuda_device);
                let _ = _test_tensor.into_data(); // Force GPU sync
                let diag_elapsed = diag_start.elapsed();
                println!(
                    "🔍 GPU DIAGNOSTIC: Zero tensor [64,128] + sync: {:?} (GPU<1ms, CPU>5ms)",
                    diag_elapsed
                );
                run_dqn_training::<Backend>(args, cuda_device, exploration.clone())
            }

            #[cfg(feature = "rocm")]
            Device::Rocm(rocm_device) => {
                use burn::backend::{Autodiff, Rocm};
                type Backend = Autodiff<Rocm<f32, i32>>;
                println!("Running DQN on ROCm");
                run_dqn_training::<Backend>(args, rocm_device, exploration.clone())
            }
        };
    }

    // Check if using Bandit (requires backend)
    if matches!(args.model, ModelType::Bandit) {
        return match device {
            #[cfg(feature = "cpu")]
            Device::Cpu(ndarray_device) => {
                use burn::backend::{Autodiff, NdArray};
                type Backend = Autodiff<NdArray<f32>>;
                println!("Running Bandit on CPU (NdArray)");
                run_bandit_training::<Backend>(args, ndarray_device, exploration.clone())
            }

            #[cfg(feature = "wgpu")]
            Device::Wgpu(wgpu_device) => {
                use burn::backend::{Autodiff, Wgpu};
                type Backend = Autodiff<Wgpu<f32, i32>>;
                println!("Running Bandit on GPU (Wgpu)");
                run_bandit_training::<Backend>(args, wgpu_device, exploration.clone())
            }

            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_device) => {
                use burn::backend::{Autodiff, Cuda};
                type Backend = Autodiff<Cuda<f32, i32>>;
                println!("Running Bandit on CUDA");
                run_bandit_training::<Backend>(args, cuda_device, exploration.clone())
            }

            #[cfg(feature = "rocm")]
            Device::Rocm(rocm_device) => {
                use burn::backend::{Autodiff, Rocm};
                type Backend = Autodiff<Rocm<f32, i32>>;
                println!("Running Bandit on ROCm");
                run_bandit_training::<Backend>(args, rocm_device, exploration.clone())
            }
        };
    }

    // Use VecEnv for parallel environments if num_envs > 1
    if args.num_envs > 1 {
        println!("Using parallel environments: {}", args.num_envs);
        // Dispatch to VecEnv training with appropriate backend
        match device {
            #[cfg(feature = "cpu")]
            Device::Cpu(ndarray_device) => {
                use burn::backend::{Autodiff, NdArray};
                type Backend = Autodiff<NdArray<f32>>;
                println!("Running on CPU (NdArray)");
                run_training_vecenv::<Backend>(args, ndarray_device)
            }

            #[cfg(feature = "wgpu")]
            Device::Wgpu(wgpu_device) => {
                use burn::backend::{Autodiff, Wgpu};
                type Backend = Autodiff<Wgpu<f32, i32>>;
                println!("Running on GPU (Wgpu)");
                run_training_vecenv::<Backend>(args, wgpu_device)
            }

            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_device) => {
                use burn::backend::{Autodiff, Cuda};
                type Backend = Autodiff<Cuda<f32, i32>>;
                println!("Running on CUDA");
                run_training_vecenv::<Backend>(args, cuda_device)
            }

            #[cfg(feature = "rocm")]
            Device::Rocm(rocm_device) => {
                use burn::backend::{Autodiff, Rocm};
                type Backend = Autodiff<Rocm<f32, i32>>;
                println!("Running on ROCm");
                run_training_vecenv::<Backend>(args, rocm_device)
            }
        }
    } else {
        // Use original single-env training
        match device {
            #[cfg(feature = "cpu")]
            Device::Cpu(ndarray_device) => {
                use burn::backend::{Autodiff, NdArray};
                type Backend = Autodiff<NdArray<f32>>;
                println!("Running on CPU (NdArray)");
                run_training::<Backend>(args, ndarray_device)
            }

            #[cfg(feature = "wgpu")]
            Device::Wgpu(wgpu_device) => {
                use burn::backend::{Autodiff, Wgpu};
                type Backend = Autodiff<Wgpu<f32, i32>>;
                println!("Running on GPU (Wgpu)");
                run_training::<Backend>(args, wgpu_device)
            }

            #[cfg(feature = "cuda")]
            Device::Cuda(cuda_device) => {
                use burn::backend::{Autodiff, Cuda};
                type Backend = Autodiff<Cuda<f32, i32>>;
                println!("Running on CUDA");
                run_training::<Backend>(args, cuda_device)
            }

            #[cfg(feature = "rocm")]
            Device::Rocm(rocm_device) => {
                use burn::backend::{Autodiff, Rocm};
                type Backend = Autodiff<Rocm<f32, i32>>;
                println!("Running on ROCm");
                run_training::<Backend>(args, rocm_device)
            }
        }
    }
}

// ============================================================================
// VECENV TRAINING (PARALLEL ENVIRONMENTS)
// ============================================================================

/// Train Metis using vectorized environments for parallel experience collection.
///
/// MultiThreadDataLoader Integration:
/// This function uses Burn's MultiThreadDataLoader for efficient batch processing.
/// Burn automatically handles prefetching in background threads, maximizing GPU utilization.
fn run_training_vecenv<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    device: B::Device,
) -> Result<(), String> {
    // Validate num_envs to prevent division by zero
    if args.num_envs == 0 {
        return Err("num_envs must be greater than 0".to_string());
    }

    // Validate buffer capacity for training stability
    if args.buffer_capacity < args.batch_size * 4 {
        return Err(format!(
            "buffer_capacity ({}) must be >= batch_size * 4 ({})",
            args.buffer_capacity,
            args.batch_size * 4
        ));
    }

    use eris::env::VecEnv;
    use eris::training::Transition;
    use std::path::Path;

    let config_path = &args.config;
    let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");

    // Create vectorized environment
    println!("Creating {} parallel environments...", args.num_envs);
    let mut vec_env = VecEnv::new(args.num_envs, config_path, trace_path, args.max_steps)
        .map_err(|e| format!("Failed to create VecEnv: {}", e))?;

    // Reset all environments
    let mut observations = vec_env
        .reset_all()
        .map_err(|e| format!("Failed to reset envs: {}", e))?;

    let state_dim = vec_env.observation_dim();
    let action_dim = vec_env.action_space().n;

    // Setup Metis agent (single agent controls all envs)
    let mut agent = setup_combined_agent::<B>(args, state_dim, action_dim, &device)
        .map_err(|e| format!("Agent setup failed: {}", e))?;

    println!("Training with {} parallel environments", args.num_envs);
    println!(
        "GPU-native training: batch_size={}, device={:?}",
        args.batch_size, device
    );

    // Collect experience from all envs
    let mut total_steps = 0;
    let mut episode_count = 0;
    let mut episode_rewards: Vec<f64> = Vec::new();
    let mut env_cumulative_rewards: Vec<f64> = vec![0.0; args.num_envs];
    let mut env_steps: Vec<usize> = vec![0; args.num_envs];

    while episode_count < args.episodes {
        // Step all environments with batched action selection (SINGLE forward pass)
        // GPU-native: avoids GPU→CPU→GPU transfer for better performance
        let actions: Vec<usize> =
            select_actions_batched_gpu(&agent, &observations, &device, action_dim, agent.epsilon);

        #[cfg(feature = "parallel")]
        let step_results = vec_env
            .step_all_parallel(actions.clone())
            .map_err(|e| format!("Step failed: {}", e))?;

        #[cfg(not(feature = "parallel"))]
        let step_results = vec_env
            .step_all(actions.clone())
            .map_err(|e| format!("Step failed: {}", e))?;

        // Reset environments that are done and get new observations
        let reset_obs = vec_env.reset_done_environments(&step_results);

        // Store all transitions in both agent buffer and async buffer
        let mut done_count = 0;
        for (i, result) in step_results.iter().enumerate() {
            // Use reset observation if environment was reset
            let next_obs = reset_obs[i].as_ref().unwrap_or(&result.observation);

            // Accumulate rewards for this environment
            env_cumulative_rewards[i] += result.reward as f64;
            env_steps[i] += 1;

            // Clip reward to prevent extreme values (standard DQN: ±1.0)
            let clipped_reward = result.reward.clamp(-1.0, 1.0) as f32;

            let transition = Transition {
                state: observations[i].iter().map(|&x| x as f32).collect(),
                action: actions[i],
                reward: clipped_reward,
                next_state: next_obs.iter().map(|&x| x as f32).collect(),
                done: result.done,
            };

            // Store in agent's GPU-native buffer
            agent.buffer.push(
                transition.state.clone(),
                transition.action,
                transition.reward,
                transition.next_state.clone(),
                transition.done,
            );

            if result.done {
                // Print episode completion with reward
                println!(
                    "✓ Episode {} completed: reward={:.2}, steps={}, env={}",
                    episode_count + 1,
                    env_cumulative_rewards[i],
                    env_steps[i],
                    i
                );

                episode_rewards.push(env_cumulative_rewards[i]);
                env_cumulative_rewards[i] = 0.0;
                env_steps[i] = 0;
                episode_count += 1;
                done_count += 1;
            }
        }

        if done_count > 0 {
            println!(
                "  → {} episode(s) completed (total: {}/{})",
                done_count, episode_count, args.episodes
            );
        }

        // Use reset observations for next step
        observations = VecEnv::get_current_observations(&step_results, &reset_obs);

        total_steps += args.num_envs;

        // Show buffer fill progress during warmup
        let buffer_len = agent.buffer.len();
        let warmup_threshold = agent.warmup_batch_size;
        if !agent.warmup_complete && buffer_len < warmup_threshold {
            if total_steps % 100 == 0 {
                println!(
                    "⏳ Warming up: {}/{} samples ({:.1}%) - training starts at {}",
                    buffer_len,
                    warmup_threshold,
                    100.0 * buffer_len as f32 / warmup_threshold as f32,
                    warmup_threshold
                );
            }
            continue; // Skip training until we have enough for warmup batch
        }

        // Train using GPU-native sampling with warmup batch sizing
        // During warmup: train every step with small batch (256)
        // After warmup: train every 4 steps with full batch (4096)
        let steps_since_last_train = if agent.warmup_complete { 4 } else { 1 };
        let mut total_loss = 0.0;
        let mut train_steps = 0;

        // Sample directly from GPU buffer and train
        if let Some(loss) = agent.train_step_gpu_native(steps_since_last_train) {
            total_loss += loss;
            train_steps += 1;
        }

        if train_steps > 0 && total_steps % 1000 == 0 {
            let avg_loss = total_loss / train_steps as f32;
            let avg_reward =
                episode_rewards.iter().sum::<f64>() / episode_rewards.len().max(1) as f64;
            println!(
                "📊 Training: avg_loss={:.4}, avg_reward={:.2}, epsilon={:.3}",
                avg_loss, avg_reward, agent.epsilon
            );
        }

        // Progress display
        if total_steps % 1000 == 0 {
            let steps_per_env = total_steps / args.num_envs;
            let avg_reward = if !episode_rewards.is_empty() {
                episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64
            } else {
                0.0
            };

            println!(
                "⏱️  Steps: {} total ({} per env) | Episodes: {}/{} | Avg Reward: {:.2} | ε: {:.3} | Buffer: {}/{}",
                total_steps,
                steps_per_env,
                episode_count,
                args.episodes,
                avg_reward,
                agent.epsilon,
                agent.buffer.len(),
                args.buffer_capacity
            );
        }
    }

    Ok(())
}

/// Select actions for ALL environments in a single batched forward pass.
///
/// This is ~16x faster than calling forward() 16 times because:
/// - One GPU kernel launch vs 16 launches
/// - One memory transfer vs 16 transfers
/// - Better GPU utilization
///
/// The epsilon-greedy decision is made PER-ENVIRONMENT after getting all Q-values.
fn select_actions_batched<B: burn::tensor::backend::AutodiffBackend>(
    agent: &CombinedAgent<B>,
    observations: &[Vec<f64>],
    device: &B::Device,
    action_dim: usize,
    epsilon: f32,
) -> Vec<usize> {
    use burn::tensor::Tensor;
    use rand::prelude::*;

    let batch_size = observations.len();
    if batch_size == 0 {
        return Vec::new();
    }

    // Step 1: Stack all observations into a single batch tensor
    let state_dim = observations[0].len();
    let states_flat: Vec<f32> = observations
        .iter()
        .flat_map(|obs| obs.iter().map(|&x| x as f32))
        .collect();

    let states_data = TensorData::new(states_flat, [batch_size, state_dim]);
    let states_tensor: Tensor<B, 2> = Tensor::from_data(states_data.convert::<f32>(), device);

    // Step 2: SINGLE forward pass for ALL environments
    let (_, _, q_values) = agent.model.forward(states_tensor);

    // Step 3: Get Q-values as slice for per-environment processing
    let q_data = q_values.into_data().convert::<f32>();
    let q_slice: &[f32] = q_data.as_slice().unwrap();

    // Step 4: For EACH environment, decide: random or greedy?
    (0..batch_size)
        .map(|i| {
            // Epsilon-greedy: per-environment exploration
            if rand::random::<f32>() < epsilon {
                // EXPLORE: random action for this specific environment
                rand::rng().random_range(0..action_dim)
            } else {
                // EXPLOIT: use the Q-values we computed
                let start = i * action_dim;
                let end = start + action_dim;
                let q_for_this_env = &q_slice[start..end];

                // Argmax: select action with highest Q-value
                q_for_this_env
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            }
        })
        .collect()
}

/// Select actions for ALL environments using GPU-native epsilon-greedy.
///
/// This version avoids GPU→CPU→GPU transfer by keeping all computation on GPU
/// until the final Vec<usize> result is needed.
///
/// All tensors are 1D [batch_size] for consistent shape handling.
fn select_actions_batched_gpu<B: burn::tensor::backend::AutodiffBackend>(
    agent: &CombinedAgent<B>,
    observations: &[Vec<f64>],
    device: &B::Device,
    action_dim: usize,
    epsilon: f32,
) -> Vec<usize> {
    use burn::tensor::{Distribution, Int, Tensor, TensorData};

    let batch_size = observations.len();
    if batch_size == 0 {
        return Vec::new();
    }

    // Step 1: Stack all observations into a single batch tensor (already GPU)
    let state_dim = observations[0].len();
    let states_flat: Vec<f32> = observations
        .iter()
        .flat_map(|obs| obs.iter().map(|&x| x as f32))
        .collect();

    let states_data = TensorData::new(states_flat, [batch_size, state_dim]);
    let states_tensor: Tensor<B, 2> = Tensor::from_data(states_data.convert::<f32>(), device);

    // Step 2: SINGLE forward pass for ALL environments (GPU)
    let (_, _, q_values) = agent.model.forward(states_tensor);

    // Step 3: Generate random actions on GPU [batch_size] with values in [0, action_dim)
    let random_float = Tensor::<B, 1>::random(
        [batch_size],
        Distribution::Uniform(0.0, action_dim as f64),
        device,
    );
    let random_actions: Tensor<B, 1, Int> = random_float.int(); // [batch_size]

    // Step 4: Get greedy actions on GPU using argmax, then reshape to 1D
    let greedy_actions_2d = q_values.clone().argmax(1); // [batch_size, 1]
    let greedy_actions: Tensor<B, 1, Int> = greedy_actions_2d.reshape([batch_size]); // [batch_size]

    // Step 5: Generate random values [0,1] for epsilon-greedy decision
    let random_vals = Tensor::<B, 1>::random([batch_size], Distribution::Uniform(0.0, 1.0), device);

    // Step 6: Create explore mask: random_vals < epsilon
    let explore_mask = random_vals.lower_elem(epsilon as f64); // Tensor<B, 1, Bool>
    let explore_int: Tensor<B, 1, Int> = explore_mask.int(); // 1 for explore, 0 for exploit [batch_size]

    // Step 7: Select actions using mask_where
    // mask_where: where condition == 0, use second arg; else use first arg
    // So: where explore_int == 0 (exploit), use greedy; else use random
    // Both tensors are [batch_size], no reshape needed
    let selected = random_actions.mask_where(explore_int.equal_elem(0), greedy_actions);

    // Step 8: Convert to Vec<usize> - only sync point
    let actions_data = selected.into_data().convert::<i64>();
    let actions_slice: &[i64] = actions_data.as_slice().unwrap();
    actions_slice.iter().map(|&x| x as usize).collect()
}

// ============================================================================
// GENERIC TRAIN FUNCTION
// ============================================================================

/// Train Metis (single environment).
fn run_training<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    device: B::Device,
) -> Result<(), String> {
    use eris::env::Environment;
    use eris::env::IOBufferEnv;
    use eris::space::Space;
    use eris::training::Transition;
    use std::path::Path;

    // Create environment
    let config_path = &args.config;
    let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");
    #[cfg(debug_assertions)]
    eprintln!("DEBUG: Creating environment");
    let mut env = IOBufferEnv::new(config_path, trace_path, args.max_steps, None, None)
        .map_err(|e| format!("Failed to create environment: {}", e))?;
    #[cfg(debug_assertions)]
    eprintln!("DEBUG: Environment created");

    // Get dimensions from environment
    let state_dim = env.observation_space().dim();
    let action_dim = env.action_space().n;
    #[cfg(debug_assertions)]
    eprintln!("DEBUG: state_dim={}, action_dim={}", state_dim, action_dim);

    // Setup Metis agent
    let mut agent = setup_combined_agent::<B>(args, state_dim, action_dim, &device)
        .map_err(|e| format!("Combined setup failed: {}", e))?;

    println!("Starting training for {} episodes...", args.episodes);
    println!("State dim: {}, Action dim: {}", state_dim, action_dim);
    println!("Model: metis");
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
            let action = select_action(&agent, &state, &env, agent.epsilon, &args.config);

            // Step environment
            let (next_state, reward, done_flag) = env.step(action);
            episode_reward += reward;

            // Clip reward to prevent extreme values (standard DQN: ±1.0)
            let clipped_reward = reward.clamp(-1.0, 1.0) as f32;

            // Store transition in GPU-native buffer
            agent.buffer.push(
                state.iter().map(|&x| x as f32).collect(),
                action,
                clipped_reward,
                next_state.iter().map(|&x| x as f32).collect(),
                done_flag,
            );

            // Train using GPU-native sampling with warmup batch sizing
            // During warmup: train every step with small batch (256)
            // After warmup: train every 4 steps with full batch (4096)
            let buffer_len = agent.buffer.len();
            let warmup_threshold = agent.warmup_batch_size;
            if buffer_len >= warmup_threshold {
                let steps_since_last_train = if agent.warmup_complete { 4 } else { 1 };
                if let Some(_loss) = agent.train_step_gpu_native(steps_since_last_train) {
                    // Loss reported asynchronously
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
            let checkpoint_path = format!("checkpoints/metis_episode_{}", episode + 1);
            save_checkpoint(&agent, &checkpoint_path, episode + 1, avg_reward as f32)
                .map_err(|e| format!("Failed to save checkpoint: {}", e))?;
            println!("  Saved checkpoint: {}.mpk", checkpoint_path);
        }
    }

    // FINAL SAVE
    let avg_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
    let final_path = "checkpoints/metis_final";
    save_checkpoint(&agent, final_path, args.episodes, avg_reward as f32)
        .map_err(|e| format!("Failed to save final checkpoint: {}", e))?;

    // PRINT SUMMARY
    println!("\nTraining complete!");
    println!("Model: metis");
    println!("Total episodes: {}", args.episodes);
    println!("Average reward: {:.2}", avg_reward);
    println!("Final epsilon: {:.3}", agent.epsilon);
    println!("Checkpoints saved to: checkpoints/metis_*.mpk");

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
    config_path: &PathBuf,
) -> usize {
    use burn::tensor::Tensor;
    use eris::config_old::Config;
    use eris::env::Environment;
    use eris::tier::{Tier, TierSelector};
    use rand::prelude::*;

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
// METIS AGENT SETUP
// ============================================================================

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
        buffer_capacity: args.buffer_capacity,
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
// CACHEUS TRAINING (ONLINE LEARNING - NO BACKEND NEEDED)
// ============================================================================

/// Train CACHEUS policy (tabular bandit, no deep learning).
fn run_cacheus_training(args: &Args) -> Result<(), String> {
    use eris::env::IOBufferEnv;
    use eris::policies::cacheus::CacheusPolicy;
    use eris::policies::{Action, CachePolicy, OnlinePolicy, State, Transition};
    use std::path::Path;

    println!("=== Training CACHEUS Policy ===");
    println!("Episodes: {}", args.episodes);
    println!("Learning rate: {}", args.learning_rate);

    // Create CACHEUS policy (no device needed - tabular)
    let mut policy = CacheusPolicy::new(2, args.learning_rate as f32);

    // Create environment
    let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");
    let config_path = &args.config;
    let mut env = IOBufferEnv::new(config_path, trace_path, args.max_steps, None, None)
        .map_err(|e| format!("Failed to create env: {}", e))?;

    // Training loop (online learning)
    let mut episode_rewards = Vec::new();

    for episode in 0..args.episodes {
        let mut state_vec = env.reset();
        let mut episode_reward = 0.0;
        let mut done = false;
        let mut steps = 0;

        while !done && steps < args.max_steps {
            // Extract features for CACHEUS
            let features = extract_cacheus_features(&[state_vec.clone()], &env);
            let state = State::Features(features);

            // Select action
            let action = policy.select_action(&state);
            let action_idx = match action {
                Action::Discrete(idx) => idx,
                _ => 0,
            };

            // Step environment
            let (next_state_vec, reward, done_flag) = env.step(action_idx);

            // Create transition
            let next_features = extract_cacheus_features(&[next_state_vec.clone()], &env);
            let transition = Transition {
                state: state.clone(),
                action: action.clone(),
                reward: reward as f32,
                next_state: State::Features(next_features),
                done: done_flag,
            };

            // Online update
            let _regret = policy.update(&transition);

            state_vec = next_state_vec;
            episode_reward += reward;
            done = done_flag;
            steps += 1;
        }

        episode_rewards.push(episode_reward);

        // Decay learning rate
        if episode % 100 == 0 {
            let new_lr = policy.learning_rate() * 0.95;
            policy.set_learning_rate(new_lr);
        }

        // Print progress
        if (episode + 1) % 10 == 0 {
            let avg_reward: f64 =
                episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
            println!(
                "Episode {}/{}: reward={:.2}, steps={}, avg={:.2}, lr={:.4}",
                episode + 1,
                args.episodes,
                episode_reward,
                steps,
                avg_reward,
                policy.learning_rate()
            );
        }
    }

    // Save policy
    std::fs::create_dir_all("checkpoints")
        .map_err(|e| format!("Failed to create checkpoints dir: {}", e))?;
    let save_path = "checkpoints/cacheus_policy.json";
    policy
        .save(Path::new(save_path))
        .map_err(|e| format!("Failed to save: {}", e))?;
    println!("Policy saved to {}", save_path);

    Ok(())
}

/// Extract features for CACHEUS: [recency, frequency, size].
fn extract_cacheus_features(state: &[Vec<f64>], _env: &eris::env::IOBufferEnv) -> Vec<f32> {
    // Simplified: use state statistics
    // Real implementation would track per-blob access patterns
    let recency = state.iter().flatten().cloned().fold(0.0f64, f64::max) as f32;
    let frequency = state.len() as f32;
    let size = state.iter().flatten().count() as f32;

    vec![recency, frequency, size]
}

// ============================================================================
// CATCHER TRAINING (DDPG ACTOR-CRITIC WITH COORDINATOR)
// ============================================================================

/// Train Catcher policy (DDPG Actor-Critic with continuous actions) using VecEnv and GpuTrainingCoordinator.
fn run_catcher_training<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    device: B::Device,
) -> Result<(), String> {
    use eris::env::VecEnv;
    use eris::policies::CatcherPolicy;
    use eris::training::{GpuTrainable, GpuTrainingCoordinator};
    use std::path::Path;

    println!("=== Training Catcher (DDPG) with GpuTrainingCoordinator ===");
    println!("Episodes: {}", args.episodes);
    println!("Max steps: {}", args.max_steps);
    println!("Batch size: {}", args.batch_size);
    println!("Learning rate: {}", args.learning_rate);
    println!("Parallel environments: {}", args.num_envs);

    // Validate num_envs
    if args.num_envs == 0 {
        return Err("num_envs must be greater than 0".to_string());
    }

    // Validate buffer capacity
    if args.buffer_capacity < args.batch_size * 4 {
        return Err(format!(
            "buffer_capacity ({}) must be >= batch_size * 4 ({})",
            args.buffer_capacity,
            args.batch_size * 4
        ));
    }

    // Create vectorized environment
    let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");
    let config_path = &args.config;

    println!("Creating {} parallel environments...", args.num_envs);
    let mut vec_env = VecEnv::new(args.num_envs, config_path, trace_path, args.max_steps)
        .map_err(|e| format!("Failed to create VecEnv: {}", e))?;

    // Get dimensions from environment
    let state_dim = vec_env.observation_dim();
    let action_dim = vec_env.action_space().n;

    println!("State dim: {}", state_dim);
    println!("Action dim: {}", action_dim);

    // Create Catcher policy with custom buffer capacity
    let mut policy = CatcherPolicy::<B>::with_config(
        device.clone(),
        args.buffer_capacity,
        100, // target_update_freq
    );

    println!("Catcher policy initialized!");
    println!(
        "GPU-native training: batch_size={}, warmup_batch_size={}, device={:?}",
        args.batch_size,
        policy.warmup_batch_size(),
        device
    );

    // Create training coordinator with configuration
    let coordinator = GpuTrainingCoordinator::with_config(
        args.episodes,
        args.max_steps,
        args.batch_size,
        256, // warmup_batch_size
        10,  // checkpoint_interval
        4,   // train_frequency
    );

    println!("\nStarting Catcher training with GpuTrainingCoordinator...");
    println!(
        "Warmup: {} samples → {} samples",
        coordinator.warmup_batch_size, coordinator.batch_size
    );
    println!(
        "Train frequency: every {} steps after warmup",
        coordinator.train_frequency
    );

    // Run training using coordinator
    let start_time = std::time::Instant::now();
    let metrics = coordinator
        .run_training::<CatcherPolicy<B>, VecEnv, B>(
            &mut policy,
            &mut vec_env,
            &device,
            "checkpoints/catcher",
        )
        .map_err(|e| format!("Training failed: {}", e))?;
    let elapsed = start_time.elapsed();

    // Print final metrics
    println!("\n{}", "=".repeat(60));
    println!("🏁 Catcher Training Complete!");
    println!("{}", "=".repeat(60));
    println!("Total episodes: {}", metrics.total_episodes);
    println!(
        "Total steps: {} ({:.0} per env)",
        metrics.total_steps,
        metrics.total_steps as f64 / args.num_envs as f64
    );
    println!("Average reward: {:.2}", metrics.avg_reward);
    println!("Final loss: {:.4}", metrics.final_loss);
    println!("Final noise_std: {:.3}", policy.epsilon());
    println!("Elapsed time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "Throughput: {:.0} steps/sec",
        metrics.total_steps as f64 / elapsed.as_secs_f64()
    );
    println!("\nCheckpoints saved to: checkpoints/catcher_episode_*.mpk");

    Ok(())
}

// ============================================================================
// DQN TRAINING (PURE DQN WITH EXPLORATION)
// ============================================================================

/// Train DQN policy (standalone DQN without bandit) using VecEnv and GpuTrainingCoordinator.
fn run_dqn_training<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    device: B::Device,
    exploration: eris::policies::ExplorationConfig,
) -> Result<(), String> {
    use eris::env::VecEnv;
    use eris::policies::{DQNExplorerConfig, DQNPolicy};
    use eris::training::{GpuTrainable, GpuTrainingCoordinator};
    use std::path::Path;

    println!("=== Training DQN Policy with VecEnv ===");
    println!("Episodes: {}", args.episodes);
    println!("Max steps: {}", args.max_steps);
    println!("Batch size: {}", args.batch_size);
    println!("Learning rate: {}", args.learning_rate);
    println!("Parallel environments: {}", args.num_envs);
    println!("Exploration: {:?}", exploration);

    // Validate num_envs
    if args.num_envs == 0 {
        return Err("num_envs must be greater than 0".to_string());
    }

    // Validate buffer capacity
    if args.buffer_capacity < args.batch_size * 4 {
        return Err(format!(
            "buffer_capacity ({}) must be >= batch_size * 4 ({})",
            args.buffer_capacity,
            args.batch_size * 4
        ));
    }

    // Create vectorized environment (like Metis)
    let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");
    let config_path = &args.config;

    println!("Creating {} parallel environments...", args.num_envs);
    let mut vec_env = VecEnv::new(args.num_envs, config_path, trace_path, args.max_steps)
        .map_err(|e| format!("Failed to create VecEnv: {}", e))?;

    // Get dimensions from environment
    let state_dim = vec_env.observation_dim();
    let action_dim = vec_env.action_space().n;

    println!("State dim: {}", state_dim);
    println!("Action dim: {}", action_dim);

    // ADJUST BUFFER CAPACITY FOR LARGE STATE DIMENSIONS
    let effective_buffer_capacity = if state_dim > 20 {
        let adjusted = (args.buffer_capacity / 2).max(10000); // Minimum 10k
        println!(
            "⚠️  State dimension {} > 20, reducing buffer capacity: {} → {} (for GPU memory safety)",
            state_dim, args.buffer_capacity, adjusted
        );
        adjusted
    } else {
        args.buffer_capacity
    };

    // Calculate estimated GPU memory usage
    let buffer_mem_mb = (effective_buffer_capacity * state_dim * 4 * 2) / (1024 * 1024); // ×2 for states + next_states
    println!(
        "📊 Estimated GPU buffer memory: ~{} MB (capacity={}, state_dim={})",
        buffer_mem_mb, effective_buffer_capacity, state_dim
    );

    // Create DQN config
    let dqn_config = eris::config::DQNConfig::builder()
        .input_dim(state_dim)
        .hidden_layers(vec![128, 128])
        .action_dim(action_dim)
        .dueling(true)
        .build()
        .map_err(|e| format!("DQN config failed: {}", e))?;

    // Create DQN explorer config
    let dqn_explorer_config = DQNExplorerConfig::new(dqn_config, exploration)
        .with_learning_rate(args.learning_rate as f32)
        .with_gamma(0.99)
        .with_batch_size(args.batch_size)
        .with_buffer_capacity(effective_buffer_capacity);

    // Create DQN policy
    let mut policy = DQNPolicy::<B>::new(dqn_explorer_config, device.clone());

    println!("DQN policy initialized!");
    println!(
        "GPU-native training: batch_size={}, warmup_batch_size={}, device={:?}",
        args.batch_size,
        policy.warmup_batch_size(),
        device
    );

    // GPU DIAGNOSTIC: Verify training will actually use GPU
    {
        let backend_name = std::any::type_name::<B>();
        println!("🔍 GPU DIAGNOSTIC: Policy backend type = {}", backend_name);
        if backend_name.contains("NdArray") {
            println!(
                "❌ GPU DIAGNOSTIC ERROR: Backend is NdArray (CPU)! Training will happen on CPU!"
            );
            println!(
                "   → This means the CUDA path was NOT taken despite selecting --backend cuda"
            );
        } else if backend_name.contains("Cuda") {
            println!("✅ GPU DIAGNOSTIC: Backend is CUDA! Training should happen on GPU.");
        } else if backend_name.contains("Wgpu") {
            println!(
                "⚠️  GPU DIAGNOSTIC: Backend is WGPU (WebGPU). May work but CUDA is preferred."
            );
        } else {
            println!("⚠️  GPU DIAGNOSTIC: Unknown backend type: {}", backend_name);
        }

        // Benchmark: run tensor operations to verify GPU is actually used
        use burn::tensor::Tensor;
        println!("🔍 GPU DIAGNOSTIC: Benchmarking tensor operations on device...");

        // Test 1: Create and sync a zero tensor
        let bench_start1 = std::time::Instant::now();
        let test_zeros = Tensor::<B, 2>::zeros([64, 128], &device);
        let _ = test_zeros.into_data(); // Force sync
        let bench_elapsed1 = bench_start1.elapsed();
        println!(
            "🔍 GPU DIAGNOSTIC: Zero tensor [64,128] + sync: {:?}",
            bench_elapsed1
        );
        println!("   → GPU: <1ms | CPU: 1-5ms");

        // Test 2: Create a random tensor and compute
        let bench_start2 = std::time::Instant::now();
        let test_random = Tensor::<B, 2>::random(
            [256, state_dim],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let test_result = test_random.clone() * test_random; // Element-wise multiply
        let _ = test_result.into_data(); // Force sync
        let bench_elapsed2 = bench_start2.elapsed();
        println!(
            "🔍 GPU DIAGNOSTIC: Random tensor [256,{}] + multiply + sync: {:?}",
            state_dim, bench_elapsed2
        );
        println!("   → GPU: <5ms | CPU: 5-50ms");

        // Test 3: Large matmul-like operation
        let bench_start3 = std::time::Instant::now();
        let a = Tensor::<B, 2>::random(
            [2048, 128],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let b = Tensor::<B, 2>::random(
            [128, 128],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let mut result = a.clone();
        for _ in 0..10 {
            result = result.clone().matmul(b.clone());
        }
        let _ = result.into_data(); // Force sync
        let bench_elapsed3 = bench_start3.elapsed();
        println!(
            "🔍 GPU DIAGNOSTIC: [2048x128] x [128x128] matmul x10 + sync: {:?}",
            bench_elapsed3
        );
        println!("   → GPU: <50ms | CPU: 500-5000ms");
    }

    // Create training coordinator with configuration
    let coordinator = GpuTrainingCoordinator::new(args.episodes, args.max_steps, args.batch_size);

    println!("\nStarting training with GpuTrainingCoordinator...");
    println!(
        "Warmup: {} samples → {} samples",
        coordinator.warmup_batch_size, coordinator.batch_size
    );
    println!(
        "Train frequency: every {} steps after warmup",
        coordinator.train_frequency
    );

    // Run training using coordinator (like Metis)
    let start_time = std::time::Instant::now();
    let metrics = coordinator
        .run_training::<DQNPolicy<B>, VecEnv, B>(
            &mut policy,
            &mut vec_env,
            &device,
            "checkpoints/dqn",
        )
        .map_err(|e| format!("Training failed: {}", e))?;
    let elapsed = start_time.elapsed();

    // Print final metrics
    println!("\n{}", "=".repeat(60));
    println!("🏁 Training Complete!");
    println!("{}", "=".repeat(60));
    println!("Total episodes: {}", metrics.total_episodes);
    println!(
        "Total steps: {} ({:.0} per env)",
        metrics.total_steps,
        metrics.total_steps as f64 / args.num_envs as f64
    );
    println!("Average reward: {:.2}", metrics.avg_reward);
    println!("Final loss: {:.4}", metrics.final_loss);
    println!("Final epsilon: {:.3}", policy.get_exploration_param());
    println!("Elapsed time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "Throughput: {:.0} steps/sec",
        metrics.total_steps as f64 / elapsed.as_secs_f64()
    );
    println!("\nCheckpoints saved to: checkpoints/dqn_episode_*.mpk");

    Ok(())
}

// ============================================================================
// BANDIT TRAINING (STANDALONE CONTEXTUAL BANDIT)
// ============================================================================

/// Train Bandit policy (standalone contextual bandit without DQN).
fn run_bandit_training<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    device: B::Device,
    exploration: eris::policies::ExplorationConfig,
) -> Result<(), String> {
    use eris::env::Environment;
    use eris::env::IOBufferEnv;
    use eris::policies::{
        Action, BanditPolicy, BanditPolicyConfig, CachePolicy, OnlinePolicy, State,
    };
    use eris::space::Space;
    use std::path::Path;

    println!("=== Training Bandit Policy ===");
    println!("Episodes: {}", args.episodes);
    println!("Learning rate: {}", args.learning_rate);
    println!("Exploration: {:?}", exploration);

    // Create environment
    let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");
    let config_path = &args.config;
    let mut env = IOBufferEnv::new(config_path, trace_path, args.max_steps, None, None)
        .map_err(|e| format!("Failed to create env: {}", e))?;

    // Get dimensions from environment
    let state_dim = env.observation_space().dim();
    let action_dim = env.action_space().n;

    // Number of tiers is action_dim / 2 (read/write per tier)
    let num_tiers = action_dim / 2;

    println!("State dim: {}", state_dim);
    println!("Action dim: {}", action_dim);
    println!("Num tiers: {}", num_tiers);

    // Create bandit config
    let bandit_config = eris::config::BanditConfig::builder()
        .input_dim(state_dim)
        .hidden_layers(vec![64, 128])
        .feature_dim(20)
        .build()
        .map_err(|e| format!("Bandit config failed: {}", e))?;

    // Create bandit policy config
    let policy_config = BanditPolicyConfig::new(
        bandit_config,
        exploration,
        args.learning_rate as f32,
        num_tiers,
    );

    // Create bandit policy
    let mut policy = BanditPolicy::<B>::new(policy_config, &device);

    println!("Bandit policy initialized!");

    // Training loop
    let mut episode_rewards = Vec::with_capacity(args.episodes);

    for episode in 0..args.episodes {
        let mut state_vec = env.reset();
        let mut episode_reward = 0.0;
        let mut done = false;
        let mut step = 0;

        while !done && step < args.max_steps {
            // Convert state to State enum
            let state = State::Raw(state_vec.clone());

            // Select action
            let action = policy.select_action(&state);
            let action_idx = match action {
                Action::Discrete(idx) => idx,
                _ => 0,
            };

            // Step environment
            let (next_state_vec, reward, done_flag) = env.step(action_idx);

            // Create transition
            let next_state = State::Raw(next_state_vec.clone());
            let transition = eris::policies::Transition {
                state: state.clone(),
                action: Action::Discrete(action_idx),
                reward: reward as f32,
                next_state,
                done: done_flag,
            };

            // Update policy (online learning)
            let _loss = policy.update(&transition);

            state_vec = next_state_vec;
            episode_reward += reward;
            done = done_flag;
            step += 1;
        }

        episode_rewards.push(episode_reward);

        // Decay learning rate
        if episode % 100 == 0 {
            let new_lr = policy.learning_rate() * 0.95;
            policy.set_learning_rate(new_lr);
        }

        // Print progress
        if (episode + 1) % 10 == 0 {
            let avg_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
            println!(
                "Episode {}/{}: reward={:.2}, steps={}, avg={:.2}, lr={:.4}",
                episode + 1,
                args.episodes,
                episode_reward,
                step,
                avg_reward,
                policy.learning_rate()
            );
        }

        // Save checkpoint every 10 episodes
        if (episode + 1) % 10 == 0 || episode == args.episodes - 1 {
            let checkpoint_path = format!("checkpoints/bandit_episode_{}", episode + 1);
            if let Err(e) = policy.save(Path::new(&checkpoint_path)) {
                eprintln!("Failed to save checkpoint: {}", e);
            } else {
                println!("Checkpoint saved: {}", checkpoint_path);
            }
        }
    }

    // Final save
    let final_path = "checkpoints/bandit_final";
    if let Err(e) = policy.save(Path::new(final_path)) {
        eprintln!("Failed to save final checkpoint: {}", e);
    } else {
        println!("Final checkpoint saved: {}", final_path);
    }

    // Print summary
    let avg_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
    println!("\nTraining complete!");
    println!("Model: Bandit");
    println!("Total episodes: {}", args.episodes);
    println!("Average reward: {:.2}", avg_reward);
    println!("Final learning rate: {:.4}", policy.learning_rate());

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
        let args = Args::try_parse_from(&["train_model", "--model", "metis", "--episodes", "50"]);
        assert!(args.is_ok());
        let args = args.unwrap();
        assert!(matches!(args.model, ModelType::Metis));
        assert_eq!(args.episodes, 50);
    }

    #[test]
    fn test_args_defaults() {
        let args = Args::try_parse_from(&["train_model"]);
        assert!(args.is_ok());
        let args = args.unwrap();
        assert!(matches!(args.model, ModelType::Metis));
        assert_eq!(args.episodes, 100);
        assert_eq!(args.max_steps, 100);
        assert_eq!(args.batch_size, 2048); // Updated default for GPU optimization
        assert!(matches!(
            args.exploration,
            ExplorationStrategy::EpsilonGreedy
        ));
        assert_eq!(args.epsilon_start, 1.0);
        assert_eq!(args.epsilon_end, 0.01);
        assert_eq!(args.epsilon_decay, 0.995);
    }

    #[test]
    fn test_model_types() {
        // Test all model types compile and parse correctly
        assert!(matches!(
            Args::try_parse_from(&["train_model", "--model", "metis"])
                .unwrap()
                .model,
            ModelType::Metis
        ));
        assert!(matches!(
            Args::try_parse_from(&["train_model", "--model", "cacheus"])
                .unwrap()
                .model,
            ModelType::Cacheus
        ));
        assert!(matches!(
            Args::try_parse_from(&["train_model", "--model", "catcher"])
                .unwrap()
                .model,
            ModelType::Catcher
        ));
        assert!(matches!(
            Args::try_parse_from(&["train_model", "--model", "dqn"])
                .unwrap()
                .model,
            ModelType::Dqn
        ));
        assert!(matches!(
            Args::try_parse_from(&["train_model", "--model", "bandit"])
                .unwrap()
                .model,
            ModelType::Bandit
        ));
    }

    #[test]
    fn test_exploration_strategies() {
        // Test epsilon-greedy
        let args = Args::try_parse_from(&[
            "train_model",
            "--exploration",
            "epsilon-greedy",
            "--epsilon-start",
            "0.9",
            "--epsilon-end",
            "0.05",
            "--epsilon-decay",
            "0.99",
        ])
        .unwrap();
        assert!(matches!(
            args.exploration,
            ExplorationStrategy::EpsilonGreedy
        ));
        assert_eq!(args.epsilon_start, 0.9);
        assert_eq!(args.epsilon_end, 0.05);
        assert_eq!(args.epsilon_decay, 0.99);

        // Test Thompson sampling
        let args =
            Args::try_parse_from(&["train_model", "--exploration", "thompson-sampling"]).unwrap();
        assert!(matches!(
            args.exploration,
            ExplorationStrategy::ThompsonSampling
        ));

        // Test UCB
        let args = Args::try_parse_from(&["train_model", "--exploration", "ucb", "--ucb-c", "1.5"])
            .unwrap();
        assert!(matches!(args.exploration, ExplorationStrategy::Ucb));
        assert_eq!(args.ucb_c, 1.5);
    }

    #[test]
    fn test_create_exploration_config() {
        // Test epsilon-greedy conversion
        let args = Args::try_parse_from(&[
            "train_model",
            "--exploration",
            "epsilon-greedy",
            "--epsilon-start",
            "0.8",
            "--epsilon-end",
            "0.02",
            "--epsilon-decay",
            "0.99",
        ])
        .unwrap();
        let config = create_exploration_config(&args);
        match config {
            eris::policies::ExplorationConfig::EpsilonGreedy {
                epsilon_start,
                epsilon_end,
                epsilon_decay,
            } => {
                assert!((epsilon_start - 0.8).abs() < 1e-6);
                assert!((epsilon_end - 0.02).abs() < 1e-6);
                assert!((epsilon_decay - 0.99).abs() < 1e-6);
            }
            _ => panic!("Expected EpsilonGreedy config"),
        }

        // Test Thompson sampling conversion
        let args = Args::try_parse_from(&[
            "train_model",
            "--exploration",
            "thompson-sampling",
            "--thompson-mean",
            "0.5",
            "--thompson-std",
            "0.3",
        ])
        .unwrap();
        let config = create_exploration_config(&args);
        match config {
            eris::policies::ExplorationConfig::ThompsonSampling {
                prior_mean,
                prior_std,
            } => {
                assert!((prior_mean - 0.5).abs() < 1e-6);
                assert!((prior_std - 0.3).abs() < 1e-6);
            }
            _ => panic!("Expected ThompsonSampling config"),
        }

        // Test UCB conversion
        let args = Args::try_parse_from(&["train_model", "--exploration", "ucb", "--ucb-c", "2.5"])
            .unwrap();
        let config = create_exploration_config(&args);
        match config {
            eris::policies::ExplorationConfig::UCB { c } => {
                assert!((c - 2.5).abs() < 1e-6);
            }
            _ => panic!("Expected UCB config"),
        }
    }
}
