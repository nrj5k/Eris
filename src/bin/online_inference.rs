//! Production inference mode with online learning capability
//!
//! This binary runs inference continuously while:
//! - Storing recent experience in a sliding-window buffer
//! - Periodically retrains on recent data (online learning)
//! - Can hot-reload from new checkpoints

use burn::backend::autodiff::Autodiff;
use burn::backend::NdArray;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use burnme_rly::CheckpointMetadata;
use clap::Parser;
use eris::tier::TierSelector;
use eris::training::{CombinedAgent, HybridRingBuffer, MockEnv, TrainingConfig};
use eris::{CombinedBanditDQNConfig, Tier, TierConfig};
use rand::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// A single transition stored in the replay buffer
struct Transition {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
}

/// Type alias for the autodiff backend
type B = Autodiff<NdArray>;

/// Command-line arguments for online inference mode
#[derive(Parser, Debug)]
#[command(name = "online_inference")]
#[command(about = "Production inference with online learning")]
struct Args {
    /// Path to checkpoint file to load
    #[arg(short, long)]
    checkpoint: PathBuf,

    /// Config file for environment
    #[arg(short = 'f', long, default_value = "config/tiers.toml")]
    config: String,

    /// Retrain every N steps (0 = no retraining, pure inference)
    #[arg(long, default_value = "1000")]
    retrain_interval: usize,

    /// Batch size for retraining
    #[arg(long, default_value = "64")]
    retrain_batch_size: usize,

    /// Replay window size (how many recent transitions to keep)
    #[arg(long, default_value = "10000")]
    replay_window: usize,

    /// Number of inference steps to run
    #[arg(long, default_value = "10000")]
    steps: usize,

    /// Epsilon for epsilon-greedy action selection
    #[arg(long, default_value = "0.1")]
    epsilon: f32,

    /// Batch size for replay buffer sampling
    #[arg(long, default_value = "64")]
    batch_size: usize,

    /// Checkpoint saving interval (0 = no saving)
    #[arg(long, default_value = "1000")]
    checkpoint_interval: usize,
}

fn main() {
    // Use 64MB stack to prevent overflow during model initialization
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(|| {
            let args = Args::parse();
            println!("=== Online Inference Mode ===");
            println!("Loading checkpoint: {:?}", args.checkpoint);
            println!("Retrain interval: {} steps", args.retrain_interval);
            println!("Replay window: {} transitions", args.replay_window);
            println!("Retrain batch size: {}", args.retrain_batch_size);
            println!();

            if let Err(e) = run_online_inference(&args) {
                eprintln!("Online inference failed: {}", e);
                std::process::exit(1);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Accumulates and logs inference metrics during online learning
#[derive(Default)]
struct InferenceMetrics {
    hit_rate: f64,
    avg_latency_us: u64,
    retrain_loss: Option<f32>,
    retrain_count: usize,
    total_reward: f64,
    total_hits: usize,
    total_steps: usize,
    latency_sum_us: u64,
}

impl InferenceMetrics {
    pub fn log_step(&mut self, reward: f64, latency_us: u64) {
        self.total_reward += reward;
        self.total_steps += 1;
        self.latency_sum_us += latency_us;
        if reward > 0.0 {
            self.total_hits += 1;
        }
        self.hit_rate = self.total_hits as f64 / self.total_steps as f64;
        self.avg_latency_us = self.latency_sum_us / self.total_steps as u64;
    }

    pub fn log_retrain(&mut self, loss: f32) {
        self.retrain_loss = Some(loss);
        self.retrain_count += 1;
    }

    pub fn format_summary(&self, step: usize) -> String {
        format!(
            "Step {} | hit_rate={:.3} | avg_latency={}μs | retrain_loss={} | retrain_count={}",
            step,
            self.hit_rate,
            self.avg_latency_us,
            self.retrain_loss
                .map(|l| format!("{:.6}", l))
                .unwrap_or_else(|| "N/A".to_string()),
            self.retrain_count,
        )
    }
}

/// Run a single retraining step on samples from replay buffer
fn retrain_step<B: AutodiffBackend>(
    agent: &mut CombinedAgent<B>,
    buffer: &HybridRingBuffer<B>,
    device: &B::Device,
    batch_size: usize,
) -> Option<f32> {
    // Check if buffer has enough samples
    if !buffer.can_sample(batch_size) {
        return None;
    }

    // Sample batch from buffer
    let batch = buffer.sample_batch(batch_size, device)?;

    // Run training step
    let loss = agent.train_step_gpu(&batch);

    println!("[RETRAIN] Loss: {:.6}", loss);

    Some(loss)
}

/// Check if checkpoint file was modified and needs reload
fn check_checkpoint_modified(
    checkpoint_path: &std::path::Path,
    last_modified: &mut std::time::SystemTime,
) -> bool {
    match std::fs::metadata(checkpoint_path) {
        Ok(metadata) => match metadata.modified() {
            Ok(modified) => {
                if modified > *last_modified {
                    *last_modified = modified;
                    true
                } else {
                    false
                }
            }
            Err(_) => false,
        },
        Err(_) => false,
    }
}

/// Reload checkpoint into agent
fn reload_checkpoint<B: AutodiffBackend>(
    agent: &mut CombinedAgent<B>,
    checkpoint_path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    agent.reload_checkpoint(checkpoint_path)?;
    println!("[HOT-RELOAD] Loaded checkpoint from {:?}", checkpoint_path);
    Ok(())
}

/// Save model checkpoint
fn save_checkpoint<B: AutodiffBackend>(
    agent: &CombinedAgent<B>,
    checkpoint_path: &std::path::Path,
    step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint_dir = checkpoint_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let checkpoint_name = format!("online_model_step_{}", step);

    let full_path = checkpoint_dir.join(&checkpoint_name);

    // Save using agent's checkpoint method (requires episode and avg_reward)
    agent.save_checkpoint(full_path.to_str().unwrap(), step, 0.0)?;

    println!("[CHECKPOINT] Saved to {:?}", full_path);

    Ok(())
}

/// Select action using epsilon-greedy policy
///
/// # Arguments
/// * `agent` - The combined agent with model and target model
/// * `observation` - Current state observation vector
/// * `epsilon` - Exploration rate [0, 1] - probability of random action
/// * `device` - Compute device for tensor operations
/// * `tier_selector` - Tier selector for mapping importance to tier
/// * `action_dim` - Number of possible actions
///
/// # Returns
/// * Action index (usize) in range [0, action_dim)
///
/// # Behavior
/// * With probability `epsilon`: random action (exploration)
/// * With probability `1 - epsilon`: greedy action from model (exploitation)
fn select_action<B: AutodiffBackend>(
    agent: &CombinedAgent<B>,
    observation: &[f64],
    epsilon: f32,
    device: &B::Device,
    tier_selector: &TierSelector,
    action_dim: usize,
) -> usize {
    // Epsilon-greedy: random action with probability epsilon
    if rand::random::<f32>() < epsilon {
        rand::rng().random_range(0..action_dim)
    } else {
        // Greedy: use model to select best action
        let state_data = TensorData::new(
            observation.iter().map(|v| *v as f32).collect(),
            [1, observation.len()],
        );
        let state_tensor = Tensor::from_data(state_data.convert::<f32>(), device);
        agent.model.select_action(state_tensor, tier_selector, 0.0) // 0.0 for greedy (no exploration)
    }
}

fn run_online_inference(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize device
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    // Initialize tier selector with default tiers
    let tiers = vec![
        Tier::new(TierConfig {
            name: "fast".to_string(),
            tier_id: 0,
            capacity: 100.0,
            access_latency: 10.0,
            description: String::new(),
        }),
        Tier::new(TierConfig {
            name: "medium".to_string(),
            tier_id: 1,
            capacity: 500.0,
            access_latency: 50.0,
            description: String::new(),
        }),
        Tier::new(TierConfig {
            name: "slow".to_string(),
            tier_id: 2,
            capacity: 1000.0,
            access_latency: 100.0,
            description: String::new(),
        }),
    ];
    let tier_selector = TierSelector::new(tiers);

    // Load agent from checkpoint or create new one
    // First, try to load checkpoint metadata to get dimensions
    println!(
        "[INIT] Loading agent from checkpoint: {:?}",
        args.checkpoint
    );

    // Default dimensions
    let mut state_dim = 20;
    let mut action_dim = 10;

    // Try to load metadata from checkpoint to get actual dimensions
    let metadata_opt = std::fs::read_to_string(&args.checkpoint)
        .ok()
        .and_then(|_| {
            let meta_path = args
                .checkpoint
                .parent()?
                .join(format!("{}.json", args.checkpoint.file_stem()?.to_str()?));
            std::fs::read_to_string(&meta_path).ok()
        })
        .and_then(|content| serde_json::from_str::<CheckpointMetadata>(&content).ok());

    if let Some(metadata) = metadata_opt {
        state_dim = metadata.state_dim.unwrap_or(state_dim);
        action_dim = metadata.action_dim.unwrap_or(action_dim);
        println!(
            "[INIT] Loaded dimensions from checkpoint: state_dim={}, action_dim={}",
            state_dim, action_dim
        );
    } else {
        println!(
            "[INIT] Using default dimensions: state_dim={}, action_dim={}",
            state_dim, action_dim
        );
    }

    let training_config = TrainingConfig::default();
    let model_config = CombinedBanditDQNConfig::builder()
        .bandit(
            eris::BanditConfig::builder()
                .input_dim(state_dim)
                .hidden_layers(vec![64])
                .feature_dim(16)
                .build()
                .expect("Failed to build bandit config"),
        )
        .dqn(
            eris::DQNConfig::builder()
                .input_dim(16)
                .hidden_layers(vec![64, 64])
                .action_dim(action_dim)
                .build()
                .expect("Failed to build dqn config"),
        )
        .build()
        .expect("Failed to build model config");

    let mut agent = CombinedAgent::<B>::load_checkpoint(
        args.checkpoint.to_str().unwrap(),
        training_config.clone(),
        model_config.clone(),
        device.clone(),
    )
    .unwrap_or_else(|_| {
        println!("[WARN] Failed to load checkpoint, using untrained agent");
        CombinedAgent::new(training_config, model_config, device.clone())
    });

    // Initialize replay buffer with sliding window
    let mut buffer = HybridRingBuffer::<B>::new(args.replay_window, state_dim);
    println!(
        "[INIT] Replay buffer: capacity={}, state_dim={}",
        args.replay_window, state_dim
    );

    // Initialize environment (using MockEnv for now)
    // In production, this would be IoBufferEnv loaded from args.config
    let mut env = MockEnv::new_with_dims(action_dim, state_dim, 100);
    let mut observation = env.reset();

    println!(
        "[INIT] Environment: action_dim={}, state_dim={}",
        action_dim, state_dim
    );

    // Initialize metrics
    let mut metrics = InferenceMetrics::default();

    // Setup Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n[SHUTDOWN] Received Ctrl+C, shutting down gracefully...");
        r.store(false, Ordering::SeqCst);
    })?;

    // Record checkpoint mtime for hot-reload
    let mut last_checkpoint_mtime = std::fs::metadata(&args.checkpoint)
        .and_then(|m| m.modified())
        .unwrap_or_else(|_| std::time::SystemTime::now());

    println!("[START] Running online inference for {} steps", args.steps);

    // Main inference loop
    for step in 0..args.steps {
        // Check for shutdown signal
        if !running.load(Ordering::SeqCst) {
            break;
        }

        let start = Instant::now();

        // 1. Select action
        let action = select_action(
            &agent,
            &observation,
            args.epsilon,
            &device,
            &tier_selector,
            action_dim,
        );

        // 2. Execute action in environment
        let (next_observation, reward, done) = env.step(action);

        // 3. Store transition in replay buffer
        let transition = Transition {
            state: observation.iter().map(|v| *v as f32).collect(),
            action,
            reward: reward as f32,
            next_state: next_observation.iter().map(|v| *v as f32).collect(),
            done,
        };
        buffer.push(
            transition.state,
            transition.action,
            transition.reward,
            transition.next_state,
            transition.done,
        );

        // 4. Periodic retraining
        if args.retrain_interval > 0
            && step > 0
            && step % args.retrain_interval == 0
            && buffer.can_sample(args.batch_size)
        {
            if let Some(loss) = retrain_step(&mut agent, &buffer, &device, args.batch_size) {
                metrics.log_retrain(loss);
            }
        }

        // 5. Update observations
        if done {
            observation = env.reset();
        } else {
            observation = next_observation;
        }

        // 6. Update metrics
        let latency = start.elapsed().as_micros() as u64;
        metrics.log_step(reward, latency);

        // 7. Periodic checkpoint saving
        if args.checkpoint_interval > 0 && step > 0 && step % args.checkpoint_interval == 0 {
            if let Err(e) = save_checkpoint(&agent, &args.checkpoint, step) {
                eprintln!("[WARN] Failed to save checkpoint: {}", e);
            }
        }

        // 8. Check for hot-reload
        if step % 100 == 0
            && check_checkpoint_modified(&args.checkpoint, &mut last_checkpoint_mtime)
        {
            if let Err(e) = reload_checkpoint(&mut agent, &args.checkpoint) {
                eprintln!("[WARN] Failed to reload checkpoint: {}", e);
            }
        }

        // 9. Log progress periodically
        if step % 100 == 0 {
            println!("{}", metrics.format_summary(step));
        }
    }

    // Final checkpoint and summary
    if let Err(e) = save_checkpoint(&agent, &args.checkpoint, args.steps) {
        eprintln!("[WARN] Failed to save final checkpoint: {}", e);
    }

    println!("\n{}", "=".repeat(60));
    println!("[COMPLETE] Online inference finished");
    println!("{}", metrics.format_summary(args.steps));
    println!("{}", "=".repeat(60));

    Ok(())
}
