//! Generic model training binary with checkpoint support.
//!
//! Usage:
//!   train_model --model dqn --episodes 100 --max-steps 100

use clap::Parser;
use eris::training::CombinedAgent;

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

    /// Backend: cpu, gpu, cuda, rocm
    #[arg(short, long, default_value = "cpu")]
    backend: String,
}

// ============================================================================
// ENTRY POINT
// ============================================================================

fn main() {
    let args = Args::parse();

    println!("=== Training Model: {} ===", args.model);
    println!("Episodes: {}", args.episodes);
    println!("Max steps: {}", args.max_steps);
    println!("Batch size: {}", args.batch_size);
    println!("Learning rate: {}", args.learning_rate);
    println!("Backend: {}", args.backend);
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
// GENERIC TRAIN FUNCTION
// ============================================================================

/// Generic training function that works with any model type.
///
/// This function handles the common training loop for all model types,
/// while delegating model-specific setup to specialized functions.
fn train_model_generic(args: &Args) -> Result<(), String> {
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::backend::Backend;
    use eris::env::Environment;
    use eris::env::IOBufferEnv;
    use eris::space::Space;
    use eris::training::Transition;
    use std::path::Path;

    let model_type = args.model.as_str();
    println!("Initializing {} training...", model_type);
    eprintln!("DEBUG: Starting train_model_generic for {}", model_type);

    // Setup device and backend
    type B = Autodiff<NdArray>;
    eprintln!("DEBUG: Creating device");
    let device = <NdArray as Backend>::Device::default();
    eprintln!("DEBUG: Device created");

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

    // GENERIC TRAINING LOOP - same for all model types
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
            println!("  💾 Saved checkpoint: {}.mpk", checkpoint_path);
        }
    }

    // FINAL SAVE
    let avg_reward = episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64;
    let final_path = format!("checkpoints/{}_final", model_type);
    save_checkpoint(&agent, &final_path, args.episodes, avg_reward as f32)
        .map_err(|e| format!("Failed to save final checkpoint: {}", e))?;

    // PRINT SUMMARY
    println!("\n✅ Training complete!");
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
/// This is shared across all model types using the CombinedModel's select_action.
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
        // Random action: uniform over action space
        let action_space = env.action_space();
        return rand::rng().random_range(0..action_space.n);
    }

    // Convert state to tensor and get Q-values from model
    let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();
    let state_data = TensorData::new(state_f32.clone(), [1, state_f32.len()]);
    let state_tensor: Tensor<B, 2> = Tensor::from_data(state_data.convert::<f32>(), &agent.device);

    // Create TierSelector for action mapping
    // Load tier configs from file
    let config_path = Path::new("config/tiers.toml");
    let tier_configs = match Config::from_file(config_path) {
        Ok(cfg) => cfg.tier,
        Err(_) => {
            // Fallback: create default tiers
            (0..5)
                .map(|i| eris::config_old::TierConfig {
                    name: format!("tier_{}", i),
                    tier_id: i as u32,
                    capacity: 100.0,
                    access_latency: 0.01,
                    description: String::new(),
                })
                .collect()
        }
    };
    let tier_selector = TierSelector::new(tier_configs.into_iter().map(Tier::new).collect());

    // Forward pass through model using select_action
    agent
        .model
        .select_action(state_tensor, &tier_selector, epsilon)
}

// ============================================================================
// MODEL-SPECIFIC SETUP
// ============================================================================

/// Setup DQN agent with configuration from args.
fn setup_dqn_agent<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    state_dim: usize,
    action_dim: usize,
    device: &B::Device,
) -> Result<CombinedAgent<B>, Box<dyn std::error::Error>> {
    use eris::models::CombinedModelConfig;
    use eris::training::{CombinedAgent, TrainingConfig};

    // Legacy model config (CombinedAgent still uses deprecated config)
    let feature_dim = 20;
    let hidden_dim = 128;
    #[allow(deprecated)]
    let model_config = CombinedModelConfig::new(state_dim, feature_dim, hidden_dim, action_dim);

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

/// Setup Contextual Bandit agent.
/// TODO: Implement CBandit-specific setup
fn setup_cbandit_agent<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    state_dim: usize,
    action_dim: usize,
    device: &B::Device,
) -> Result<CombinedAgent<B>, Box<dyn std::error::Error>> {
    // For now, use DQN setup as placeholder
    // TODO: Implement CBandit-specific model and agent
    println!("Note: CBandit not fully implemented, using DQN-like setup");
    setup_dqn_agent(args, state_dim, action_dim, device)
}

/// Setup Combined agent.
/// TODO: Implement Combined-specific setup
fn setup_combined_agent<B: burn::tensor::backend::AutodiffBackend>(
    args: &Args,
    state_dim: usize,
    action_dim: usize,
    device: &B::Device,
) -> Result<CombinedAgent<B>, Box<dyn std::error::Error>> {
    // For now, use DQN setup as placeholder
    // TODO: Implement Combined-specific model and agent
    println!("Note: Combined model not fully implemented, using DQN-like setup");
    setup_dqn_agent(args, state_dim, action_dim, device)
}

// ============================================================================
// CHECKPOINT SAVING
// ============================================================================

/// Save checkpoint using agent's save_checkpoint method.
fn save_checkpoint<B: burn::tensor::backend::AutodiffBackend>(
    agent: &CombinedAgent<B>,
    path: &str,
    episode: usize,
    avg_reward: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs;

    // Ensure checkpoints directory exists
    fs::create_dir_all("checkpoints")?;

    // Use agent's save_checkpoint method
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
}
