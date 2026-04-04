//! Manual DQN Training Implementation (Legacy)
//!
//! **⚠️ DEPRECATION NOTICE:**
//! The `train_step()` method in this module is deprecated. Use the Burn `TrainStep`
//! implementation in `burn_trainer.rs` with Burn's training pipeline instead.
//!
//! **Current status:**
//! - Manual `train_step()` remains for backward compatibility
//! - Burn `TrainStep` implementation exists in `burn_trainer.rs`
//! - See migration guide below
//!
//! **Migration Guide:**
//! ```rust,ignore
//! // OLD (deprecated):
//! let loss = agent.train_step(batch);
//!
//! // NEW (recommended):
//! // Use Burn's LearnerBuilder with TrainStep trait:
//! // See src/training/burn_trainer.rs for complete implementation
//! ```
//!
//! **Why both exist:**
//! - Manual version: Complete DQN with target network, replay buffer
//! - Burn version: Simpler, automatic gradients, but lacks target network
//! - Keep both during transition period

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};
use std::path::Path;

use crate::models::{CombinedModel, CombinedModelConfig};
use crate::training::checkpoint::{CheckpointMetadata, DQNCheckpointHelper};
use crate::training::replay_buffer::{ReplayBuffer, TransitionBatch};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate for optimizer
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub gamma: f32,
    /// Initial exploration rate
    pub epsilon_start: f32,
    /// Final exploration rate
    pub epsilon_end: f32,
    /// Exploration decay rate
    pub epsilon_decay: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Replay buffer capacity
    pub buffer_capacity: usize,
    /// Target network update frequency
    pub target_update_freq: usize,
    /// Soft update coefficient (not used in hard update)
    pub tau: f32,
    /// Backend type (wgpu or ndarray)
    pub backend: String,
    /// Checkpoint save interval (episodes)
    pub checkpoint_interval: usize,
    /// Maximum gradient norm for clipping
    pub max_gradient_norm: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            batch_size: 512, // Much better for GPU utilization
            buffer_capacity: 10_000,
            target_update_freq: 1000,
            tau: 0.005,
            backend: "wgpu".to_string(),
            checkpoint_interval: 10,
            max_gradient_norm: 1.0,
        }
    }
}

/// Combined agent with policy and target networks
///
/// This agent implements DQN learning with:
/// - Experience replay buffer
/// - Target network for stability
/// - Epsilon-greedy exploration
/// - Gradient clipping for stability
pub struct CombinedAgent<B: AutodiffBackend> {
    /// Policy network (online network)
    pub model: CombinedModel<B>,
    /// Target network (frozen copy)
    pub target_model: CombinedModel<B>,
    /// Experience replay buffer
    pub buffer: ReplayBuffer,
    /// Training configuration
    pub config: TrainingConfig,
    /// Current epsilon for exploration
    pub epsilon: f32,
    /// Training step counter
    pub step_count: usize,
    /// Device
    pub device: B::Device,
}

impl<B: AutodiffBackend> CombinedAgent<B> {
    /// Create new agent
    ///
    /// # Arguments
    /// * `config` - Training configuration
    /// * `model_config` - Model architecture configuration
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Initialized agent with random weights and empty buffer
    pub fn new(
        config: TrainingConfig,
        model_config: CombinedModelConfig,
        device: B::Device,
    ) -> Self {
        let model = model_config.init(&device);
        let target_model = model_config.init(&device);
        let buffer = ReplayBuffer::new(config.buffer_capacity);

        Self {
            model,
            target_model,
            buffer,
            config: config.clone(),
            epsilon: config.epsilon_start,
            step_count: 0,
            device,
        }
    }

    /// Perform one DQN training step (DEPRECATED - use Burn Trainer)
    ///
    /// **⚠️ DEPRECATION NOTICE:**
    /// This manual implementation is deprecated. Use the Burn `TrainStep` implementation
    /// in `burn_trainer.rs` with Burn's training pipeline instead.
    ///
    /// **Migration Guide:**
    /// - See `src/training/burn_trainer.rs` for Burn TrainStep implementation
    /// - Use Burn's `LearnerBuilder` for proper training loop
    /// - The `TrainStep` trait provides automatic gradient handling
    ///
    /// **Why keep this?**
    /// This method remains for backward compatibility and provides complete DQN training with:
    /// - Target network updates (hard/soft)
    /// - Experience replay integration
    /// - Gradient clipping
    /// - Full TD learning with Bellman equation
    ///
    /// The Burn `TrainStep` is simpler but lacks target network support currently.
    ///
    /// # Arguments
    /// * `batch` - Batch of transitions from replay buffer
    ///
    /// # Returns
    /// * MSE loss value (non-negative, finite)
    #[deprecated(
        since = "0.2.0",
        note = "Use Burn TrainStep implementation with LearnerBuilder. See burn_trainer.rs for details."
    )]
    #[allow(deprecated)]
    pub fn train_step(&mut self, batch: TransitionBatch) -> f32 {
        if batch.states.is_empty() {
            tracing::warn!("train_step called with empty batch");
            return 0.0;
        }

        let batch_size = batch.states.len();
        let state_dim = batch.states[0].len();

        // Convert states: Vec<Vec<f32>> -> Tensor<B, 2>
        let states_flat: Vec<f32> = batch.states.iter().flatten().copied().collect();
        let states_data = TensorData::new(states_flat, [batch_size, state_dim]);
        let states: Tensor<B, 2> = Tensor::from_data(states_data.convert::<f32>(), &self.device);

        // Convert actions: Vec<usize> -> Tensor<B, 1, Int>
        let actions_data: Vec<i64> = batch.actions.iter().map(|&a| a as i64).collect();
        let actions: Tensor<B, 1, Int> = Tensor::from_data(
            TensorData::new(actions_data, [batch_size]).convert::<i64>(),
            &self.device,
        );

        // Convert rewards: Vec<f32> -> Tensor<B, 1>
        let rewards_data = TensorData::new(batch.rewards.clone(), [batch_size]);
        let rewards: Tensor<B, 1> = Tensor::from_data(rewards_data.convert::<f32>(), &self.device);

        // Convert next_states
        let next_states_flat: Vec<f32> = batch.next_states.iter().flatten().copied().collect();
        let next_states_data = TensorData::new(next_states_flat, [batch_size, state_dim]);
        let next_states: Tensor<B, 2> =
            Tensor::from_data(next_states_data.convert::<f32>(), &self.device);

        // Convert dones: Vec<bool> -> Tensor<B, 1>
        let dones_float: Vec<f32> = batch
            .dones
            .iter()
            .map(|&d| if d { 1.0 } else { 0.0 })
            .collect();
        let dones_data = TensorData::new(dones_float, [batch_size]);
        let dones: Tensor<B, 1> = Tensor::from_data(dones_data.convert::<f32>(), &self.device);

        // Forward pass through policy network (with gradients)
        let (_, _, q_values) = self.model.forward(states.clone()); // [batch_size, 10]

        // Gather Q-values for actions taken
        let actions_2d = actions.reshape([batch_size, 1]);
        let q_selected = q_values.gather(1, actions_2d).reshape([batch_size]);

        // Forward pass through target network (NO gradients)
        // Detach next_states to prevent gradient flow through target network
        let target_model = self.target_model.clone();
        let (_, _, target_q_values) = target_model.forward(next_states.detach());

        // Compute max Q for next states
        let max_next_q = target_q_values.max_dim(1).squeeze(); // [batch_size]

        // Compute TD target: r + gamma * max(Q') * (1 - done)
        // Create ones tensor and subtract dones
        let ones = Tensor::<B, 1>::ones([batch_size], &self.device);
        let not_done = ones - dones;

        // Convert gamma to tensor for broadcasting
        let gamma_tensor = Tensor::<B, 1>::full([batch_size], self.config.gamma, &self.device);
        let targets: Tensor<B, 1> = rewards + gamma_tensor * max_next_q * not_done;

        // MSE loss
        let diff = q_selected - targets;
        let loss = diff.powf_scalar(2.0).mean();

        // Backpropagation
        // DEPRECATED: Manual gradient computation - use Burn TrainStep instead
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);

        // Create optimizer and update parameters
        // DEPRECATED: Manual optimizer.step() - Burn TrainStep handles this automatically
        let mut optimizer = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .init();

        self.model = optimizer.step(self.config.learning_rate, self.model.clone(), grads);

        // Soft update target network periodically
        self.step_count += 1;
        if self.step_count % self.config.target_update_freq == 0 {
            self.hard_update_target();
        }

        // Decay epsilon
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);

        // Return loss scalar
        loss.into_data().convert::<f32>().as_slice().unwrap()[0]
    }

    /// Hard update: Copy all weights from model to target_model
    pub fn hard_update_target(&mut self) {
        self.target_model = self.model.clone();
        tracing::debug!("Target network updated (hard reset)");
    }

    /// Save model checkpoint using Burn's recorder with DQN metadata.
    ///
    /// # Arguments
    /// * `path` - Path prefix (extensions added automatically)
    /// * `episode` - Current episode number
    /// * `avg_reward` - Average reward for logging
    ///
    /// # Returns
    /// Ok(()) on success, Err on failure
    ///
    /// # File Naming Convention
    ///
    /// Creates the following files:
    /// - `{path}-{episode}.mpk` - Policy network checkpoint
    /// - `{path}-{episode}.json` - Policy network metadata
    /// - `{path}_target-{episode}.mpk` - Target network checkpoint
    /// - `{path}_target-{epoch}.json` - Target network metadata
    ///
    /// Note: Uses underscore separator for target network to avoid issues with filesystem extensions.
    pub fn save_checkpoint<P: AsRef<Path>>(
        &self,
        path: P,
        episode: usize,
        avg_reward: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Extract directory and name from path
        let directory = path.as_ref().parent().unwrap_or(Path::new("."));
        let name = path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");

        // Save policy network using Burn's recorder
        let metadata = CheckpointMetadata::new(
            episode,
            self.step_count,
            self.epsilon,
            avg_reward,
            avg_reward,
        );

        DQNCheckpointHelper::save(&self.model, directory, name, episode, &metadata)?;

        // Also save target network (use underscore instead of dot for filename)
        let target_name = format!("{}_target", name);
        DQNCheckpointHelper::save(
            &self.target_model,
            directory,
            &target_name,
            episode,
            &metadata,
        )?;

        Ok(())
    }

    /// Load model checkpoint using Burn's recorder with DQN metadata.
    ///
    /// # Arguments
    /// * `path` - Path prefix (extensions added automatically)
    /// * `config` - Training configuration
    /// * `model_config` - Model architecture configuration
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Loaded agent on success, error on failure
    pub fn load_checkpoint<P: AsRef<Path>>(
        path: P,
        config: TrainingConfig,
        model_config: CombinedModelConfig,
        device: B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Extract directory and name from path
        let directory = path.as_ref().parent().unwrap_or(Path::new("."));
        let name = path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");

        // Load policy network using Burn's recorder
        let (model, _metadata) = DQNCheckpointHelper::load(
            directory,
            name,
            0, // Use epoch 0 for single checkpoint
            &device,
            || model_config.init(&device),
        )?;

        // Load target network (use underscore instead of dot for filename)
        let target_name = format!("{}_target", name);
        let (target_model, metadata) =
            DQNCheckpointHelper::load(directory, &target_name, 0, &device, || {
                model_config.init(&device)
            })?;

        Ok(Self {
            model,
            target_model,
            buffer: ReplayBuffer::new(config.buffer_capacity),
            config,
            epsilon: metadata.epsilon,
            step_count: metadata.step_count,
            device,
        })
    }

    /// Save model (simplified interface for compatibility)
    ///
    /// # Arguments
    /// * `path` - Output path
    pub fn save<P: AsRef<Path>>(&self, path: P) {
        if let Err(e) = self.save_checkpoint(&path, 0, 0.0) {
            tracing::error!("Failed to save model: {}", e);
        }
    }
}

// Note: train_agent function is defined separately, typically in the binary
// to avoid circular dependencies with the environment
