use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};
use std::path::Path;

use crate::models::{CombinedModel, CombinedModelConfig};
use crate::training::checkpoint::CheckpointMetadata;
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

    /// Perform one DQN training step
    ///
    /// # Arguments
    /// * `batch` - Batch of transitions from replay buffer
    ///
    /// # Returns
    /// * MSE loss value (non-negative, finite)
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
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);

        // Create optimizer and update parameters
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

    /// Save model checkpoint
    ///
    /// # Arguments
    /// * `path` - Path prefix (extensions added automatically)
    /// * `episode` - Current episode number
    /// * `avg_reward` - Average reward for logging
    ///
    /// # Returns
    /// Ok(()) on success, Err on failure
    pub fn save_checkpoint<P: AsRef<Path>>(
        &self,
        path: P,
        episode: usize,
        avg_reward: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        // Create directory if needed
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Save policy network
        self.model
            .clone()
            .save_file(path.as_ref(), &recorder)
            .map_err(|e| format!("Failed to save model: {:?}", e))?;

        // Save target network
        let target_path = format!("{}.target.mpk", path.as_ref().display());
        self.target_model
            .clone()
            .save_file(&target_path, &recorder)
            .map_err(|e| format!("Failed to save target: {:?}", e))?;

        // Save metadata
        let metadata = CheckpointMetadata::new(
            episode,
            self.step_count,
            self.epsilon,
            avg_reward,
            avg_reward,
        );

        let meta_path = format!("{}.json", path.as_ref().display());
        let json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(&meta_path, json)?;

        tracing::info!("Checkpoint saved to {}", path.as_ref().display());
        Ok(())
    }

    /// Load model checkpoint
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
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        // Load policy network
        let model = model_config.init(&device);
        let model = model
            .load_file(path.as_ref(), &recorder, &device)
            .map_err(|e| format!("Failed to load model: {:?}", e))?;

        // Load target network
        let target_path = format!("{}.target.mpk", path.as_ref().display());
        let target_model = model_config.init(&device);
        let target_model = target_model
            .load_file(&target_path, &recorder, &device)
            .map_err(|e| format!("Failed to load target: {:?}", e))?;

        // Load metadata
        let meta_path = format!("{}.json", path.as_ref().display());
        let metadata = if std::path::Path::new(&meta_path).exists() {
            let json = std::fs::read_to_string(&meta_path)?;
            serde_json::from_str(&json)?
        } else {
            tracing::warn!("No checkpoint metadata found, using defaults");
            CheckpointMetadata::default()
        };

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
