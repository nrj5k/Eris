//! Manual DQN Training Implementation (Legacy)
//!
//! **[STAGE:WARN] DEPRECATION NOTICE:**
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

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor, TensorData};
use std::path::Path;

#[cfg(feature = "profiling")]
use tracy_client::span;

use crate::config::CombinedBanditDQNConfig;
use crate::models::CombinedModel;
use crate::training::checkpoint::{CheckpointMetadata, CheckpointMetadataExt};
use crate::training::replay_buffer::TransitionBatch;
use crate::training::HybridRingBuffer;
use crate::training::TensorTransitionBatch;

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
            learning_rate: 0.0001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            batch_size: 2048, // Optimized for GPU utilization (multiple of 32 for warp alignment)
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
/// - Experience replay buffer (hybrid CPU/GPU HybridRingBuffer)
/// - Target network for stability
/// - Epsilon-greedy exploration
/// - Gradient clipping for stability
pub struct CombinedAgent<B: AutodiffBackend> {
    /// Policy network (online network)
    pub model: CombinedModel<B>,
    /// Target network (frozen copy)
    pub target_model: CombinedModel<B>,
    /// Experience replay buffer (hybrid CPU/GPU ring buffer for efficient sampling)
    pub buffer: HybridRingBuffer<B>,
    /// Training configuration
    pub config: TrainingConfig,
    /// Current epsilon for exploration
    pub epsilon: f32,
    /// Training step counter
    pub step_count: usize,
    /// Device
    pub device: B::Device,
    /// Accumulated gradients for gradient accumulation
    accumulated_grads: GradientsAccumulator<CombinedModel<B>>,
    /// Current accumulation step counter
    accumulation_counter: usize,
    /// Accumulated loss on GPU for async reporting
    accumulated_loss: Tensor<B, 1>,
    /// Number of losses accumulated
    accumulated_loss_count: usize,
    /// Frequency of loss sync to CPU (0 = sync every step)
    loss_sync_freq: usize,
    /// Minimum batch size during warmup (starts training immediately)
    pub warmup_batch_size: usize,
    /// Whether warmup phase is complete
    pub warmup_complete: bool,
    /// Cached Adam optimizer (reused across steps instead of re-created)
    optimizer: OptimizerAdaptor<Adam, CombinedModel<B>, B>,
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
        model_config: CombinedBanditDQNConfig,
        device: B::Device,
    ) -> Self {
        #[cfg(all(feature = "profiling", debug_assertions))]
        let _init_span = span!("agent_init", 0);

        let model = model_config.init(&device);
        let target_model = model_config.init(&device);
        let state_dim = model_config.bandit.input_dim;
        let buffer = HybridRingBuffer::new(config.buffer_capacity, state_dim);

        // Initialize optimizer once (cached for reuse)
        let optimizer = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(
                config.max_gradient_norm,
            )))
            .init();

        Self {
            model,
            target_model,
            buffer,
            config: config.clone(),
            epsilon: config.epsilon_start,
            step_count: 0,
            device: device.clone(),
            accumulated_grads: GradientsAccumulator::new(),
            accumulation_counter: 0,
            accumulated_loss: Tensor::zeros([1], &device),
            accumulated_loss_count: 0,
            loss_sync_freq: 500,    // Sync every 500 steps
            warmup_batch_size: 256, // Start with 256, fills in 8 steps (with 32 envs)
            warmup_complete: false,
            optimizer,
        }
    }

    /// GPU-native training step that accepts pre-batched GPU tensors.
    ///
    /// This method avoids GPU→CPU→GPU transfer by keeping tensors on GPU throughout.
    /// Use with Burn's MultiThreadDataLoader for maximum performance.
    ///
    /// # Arguments
    /// * `batch` - TensorTransitionBatch with GPU tensors from DataLoader
    ///
    /// # Returns
    /// Training loss as f32
    pub fn train_step_gpu(&mut self, batch: &TensorTransitionBatch<B>) -> f32 {
        let batch_size = batch.states.dims()[0];
        if batch_size == 0 {
            tracing::warn!("train_step_gpu called with empty batch");
            return 0.0;
        }

        // Tensors are already on GPU - no conversion needed!
        // Actions are [batch_size, 1] Int tensor -> squeeze to [batch_size]
        let actions: Tensor<B, 1, Int> = batch.actions.clone().squeeze(); // squeeze() creates new tensor
        // Rewards are [batch_size, 1] -> squeeze to [batch_size]
        let rewards: Tensor<B, 1> = batch.rewards.clone().squeeze(); // squeeze() creates new tensor
        let states = batch.states.clone(); // still needed - forward() takes ownership
        let next_states = batch.next_states.clone(); // still needed
        // Dones are [batch_size, 1] -> squeeze to [batch_size]
        let dones: Tensor<B, 1> = batch.dones.clone().squeeze(); // squeeze() creates new tensor

        // Forward pass through policy network (with gradients)
        let (_, _, q_values) = self.model.forward(states);

        // Gather Q-values for actions taken
        // actions is [batch_size], need to reshape to [batch_size, 1] for gather
        let actions_2d = actions.reshape([batch_size, 1]);
        let q_selected: Tensor<B, 1> = q_values.gather(1, actions_2d).squeeze();

        // Forward pass through target network (NO gradients)
        let (_, _, target_q_values) = self.target_model.forward(next_states.detach());

        // Compute max Q for next states
        let max_next_q = target_q_values.max_dim(1).squeeze();

        // Compute TD target: r + gamma * max(Q') * (1 - done)
        let ones = Tensor::<B, 1>::ones([batch_size], &self.device);
        let not_done = ones - dones;
        let gamma_tensor = Tensor::<B, 1>::full([batch_size], self.config.gamma, &self.device);
        let targets = rewards + gamma_tensor * max_next_q * not_done;

        // MSE loss
        let diff = q_selected - targets;
        let loss = diff.powf_scalar(2.0).mean();

        // Backpropagation
        let grads = loss.backward();
        // Detach loss immediately to drop computation graph before optimizer step
        let loss_detached = loss.detach();
        
        let grads_params = GradientsParams::from_grads(grads, &self.model);

        // Update model with CACHED optimizer (massive speedup!)
        self.model = self.optimizer.step(self.config.learning_rate, self.model.clone(), grads_params);

        // Soft update target network
        self.step_count += 1;
        if self.step_count % self.config.target_update_freq == 0 {
            self.hard_update_target();
        }

        // Return loss value (async - may not sync every step)
        self.report_loss_async(loss_detached).unwrap_or(0.0)
    }

    /// Training step using GPU-native batch sampling (no CPU→GPU transfer).
    ///
    /// This method uses HybridRingBuffer::sample_batch() to get batches
    /// directly as GPU tensors, eliminating the DataLoader sync overhead.
    ///
    /// Uses progressive batch sizing: starts with small batches during warmup
    /// to enable training after ~8 steps instead of waiting for full buffer fill.
    ///
    /// # Arguments
    /// * `steps_since_last_train` - Number of steps since last training (for warmup logic)
    ///
    /// # Returns
    /// * Loss value if batch was sampled, None if buffer has insufficient samples
    pub fn train_step_gpu_native(&mut self, steps_since_last_train: usize) -> Option<f32> {
        // Determine batch size based on warmup state
        let batch_size = if self.warmup_complete {
            self.config.batch_size
        } else {
            // During warmup: use smaller batch
            let effective = self.warmup_batch_size.min(self.config.batch_size);

            // Check if we can mark warmup complete
            if self.buffer.len() >= self.config.batch_size {
                self.warmup_complete = true;
                tracing::info!(
                    "Warmup complete! Using full batch size: {}",
                    self.config.batch_size
                );
            }

            effective
        };

        // Train every step during warmup, then every 4 steps after warmup
        if !self.warmup_complete || steps_since_last_train >= 4 {
            // Sample directly from hybrid buffer - CPU to GPU conversion only here!
            let batch = self.buffer.sample_batch(batch_size, &self.device)?;

            let batch_size_actual = batch.states.dims()[0];
            if batch_size_actual == 0 {
                return None;
            }

            // Tensors are already on GPU - no conversion needed!
            let states = batch.states;
            let actions: Tensor<B, 1, Int> = batch.actions.squeeze();
            let rewards: Tensor<B, 1> = batch.rewards.squeeze();
            let next_states = batch.next_states;
            let dones: Tensor<B, 1> = batch.dones.squeeze();

            // Forward pass through policy network (with gradients)
            let (_, _, q_values) = self.model.forward(states);

            // Gather Q-values for actions taken
            let actions_2d = actions.reshape([batch_size_actual, 1]);
            let q_selected: Tensor<B, 1> = q_values.gather(1, actions_2d).squeeze();

            // Forward pass through target network (NO gradients)
            let (_, _, target_q_values) = self.target_model.forward(next_states.detach());

            // Compute max Q for next states
            let max_next_q = target_q_values.max_dim(1).squeeze();

            // Compute TD target: r + gamma * max(Q') * (1 - done)
            let ones = Tensor::<B, 1>::ones([batch_size_actual], &self.device);
            let not_done = ones - dones;
            let gamma_tensor =
                Tensor::<B, 1>::full([batch_size_actual], self.config.gamma, &self.device);
            let targets = rewards + gamma_tensor * max_next_q * not_done;

            // MSE loss
            let diff = q_selected - targets;
            let loss = diff.powf_scalar(2.0).mean();

            // Backpropagation
            let grads = loss.backward();
            // Detach loss immediately to drop computation graph before optimizer step
            let loss_detached = loss.detach();
            let grads_params = GradientsParams::from_grads(grads, &self.model);

            // Update model with CACHED optimizer (massive speedup!)
            self.model =
                self.optimizer.step(self.config.learning_rate, self.model.clone(), grads_params);

            // Update target network and epsilon
            self.step_count += 1;
            if self.step_count % self.config.target_update_freq == 0 {
                self.hard_update_target();
            }

            self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);

            // Return loss value (async)
            return self.report_loss_async(loss_detached);
        }

        None
    }

    /// Report loss asynchronously, only syncing to CPU every N steps.
    /// Returns the loss value if synced, None if accumulated on GPU.
    pub fn report_loss_async(&mut self, loss: Tensor<B, 1>) -> Option<f32> {
        // Accumulate loss on GPU
        self.accumulated_loss = self.accumulated_loss.clone() + loss.detach();
        self.accumulated_loss_count += 1;

        // Only sync to CPU periodically
        if self.loss_sync_freq == 0 || self.accumulated_loss_count % self.loss_sync_freq == 0 {
            let avg_loss = self.accumulated_loss.clone() / self.accumulated_loss_count as f32;
            let loss_value = avg_loss.into_data().convert::<f32>().as_slice().unwrap()[0];

            // Reset accumulation
            self.accumulated_loss = Tensor::zeros([1], &self.device);
            self.accumulated_loss_count = 0;

            Some(loss_value)
        } else {
            None // Loss accumulated, not synced yet
        }
    }

    /// Perform one DQN training step (DEPRECATED - use Burn Trainer)
    ///
    /// **[STAGE:WARN] DEPRECATION NOTICE:**
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
        #[cfg(all(feature = "profiling", debug_assertions))]
        let _step_span = span!("train_step", 0);

        if batch.states.is_empty() {
            tracing::warn!("train_step called with empty batch");
            return 0.0;
        }

        let batch_size = batch.states.len();
        let state_dim = batch.states[0].len();

        // Data preparation section
        let (states, actions, rewards, next_states, dones) = {
            #[cfg(all(feature = "profiling", debug_assertions))]
            let _data_span = span!("data_prep", 0);

            // Convert states: Vec<Vec<f32>> -> Tensor<B, 2>
            let states_flat: Vec<f32> = batch.states.iter().flatten().copied().collect();
            let states_data = TensorData::new(states_flat, [batch_size, state_dim]);
            let states: Tensor<B, 2> =
                Tensor::from_data(states_data.convert::<f32>(), &self.device);

            // Convert actions: Vec<usize> -> Tensor<B, 1, Int>
            let actions_data: Vec<i32> = batch.actions.iter().map(|&a| a as i32).collect();
            let actions: Tensor<B, 1, Int> = Tensor::from_data(
                TensorData::new(actions_data, [batch_size]).convert::<i32>(),
                &self.device,
            );

            // Convert rewards: Vec<f32> -> Tensor<B, 1>
            let rewards_data = TensorData::new(batch.rewards.clone(), [batch_size]);
            let rewards: Tensor<B, 1> =
                Tensor::from_data(rewards_data.convert::<f32>(), &self.device);

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

            (states, actions, rewards, next_states, dones)
        };

        // Forward pass section
        #[cfg(all(feature = "profiling", debug_assertions))]
        let _forward_span = span!("forward_pass", 0);

        // Forward pass through policy network (with gradients)
        let (_, _, q_values) = self.model.forward(states.clone()); // [batch_size, 10]

        // Gather Q-values for actions taken
        let actions_2d = actions.reshape([batch_size, 1]);
        let q_selected = q_values.gather(1, actions_2d).reshape([batch_size]);

        // Forward pass through target network (NO gradients)
        // Detach next_states to prevent gradient flow through target network
        let (_, _, target_q_values) = self.target_model.forward(next_states.detach());

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

        // Backward pass section
        {
            #[cfg(all(feature = "profiling", debug_assertions))]
            let _backward_span = span!("backward_pass", 0);

            // Backpropagation
            // DEPRECATED: Manual gradient computation - use Burn TrainStep instead
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.model);

            // Update model with CACHED optimizer (massive speedup!)
            // DEPRECATED: Manual optimizer.step() - Burn TrainStep handles this automatically
            self.model =
                self.optimizer.step(self.config.learning_rate, self.model.clone(), grads_params);
        }

        // Soft update target network periodically
        self.step_count += 1;
        if self.config.target_update_freq > 0
            && self.step_count % self.config.target_update_freq == 0
        {
            self.hard_update_target();
        }

        // Decay epsilon
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);

        // Return loss scalar
        loss.into_data().convert::<f32>().as_slice().unwrap()[0]
    }

    /// Compute loss without applying gradients (for accumulation)
    fn compute_loss(&self, batch: TransitionBatch) -> Tensor<B, 1> {
        if batch.states.is_empty() {
            tracing::warn!("compute_loss called with empty batch");
            return Tensor::zeros([1], &self.device);
        }

        let batch_size = batch.states.len();
        let state_dim = batch.states[0].len();

        // Data preparation
        let (states, actions, rewards, next_states, dones) = {
            // Convert states: Vec<Vec<f32>> -> Tensor<B, 2>
            let states_flat: Vec<f32> = batch.states.iter().flatten().copied().collect();
            let states_data = TensorData::new(states_flat, [batch_size, state_dim]);
            let states: Tensor<B, 2> =
                Tensor::from_data(states_data.convert::<f32>(), &self.device);

            // Convert actions: Vec<usize> -> Tensor<B, 1, Int>
            let actions_data: Vec<i32> = batch.actions.iter().map(|&a| a as i32).collect();
            let actions: Tensor<B, 1, Int> = Tensor::from_data(
                TensorData::new(actions_data, [batch_size]).convert::<i32>(),
                &self.device,
            );

            // Convert rewards: Vec<f32> -> Tensor<B, 1>
            let rewards_data = TensorData::new(batch.rewards.clone(), [batch_size]);
            let rewards: Tensor<B, 1> =
                Tensor::from_data(rewards_data.convert::<f32>(), &self.device);

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

            (states, actions, rewards, next_states, dones)
        };

        // Forward pass through policy network
        let (_, _, q_values) = self.model.forward(states);

        // Gather Q-values for actions taken
        let actions_2d = actions.reshape([batch_size, 1]);
        let q_selected = q_values.gather(1, actions_2d).reshape([batch_size]);

        // Double DQN: Use policy network to select action, target network to evaluate
        // 1. Get Q-values from policy network for next states
        let next_q_policy = self.model.forward(next_states.clone()).2;

        // 2. Select best action using policy network (argmax)
        let best_actions = next_q_policy.argmax(1);

        // 3. Get Q-values from target network for next states (NO gradients)
        let (_, _, target_q_values) = self.target_model.forward(next_states.detach());

        // 4. Evaluate selected action using target network (gather)
        let max_next_q = target_q_values
            .gather(1, best_actions.reshape([batch_size, 1]))
            .reshape([batch_size]);
        let max_next_q = max_next_q.detach();

        // Compute td target: r + gamma * max(Q') * (1 - done)
        let ones = Tensor::<B, 1>::ones([batch_size], &self.device);
        let not_done = ones - dones;
        let gamma_tensor = Tensor::<B, 1>::full([batch_size], self.config.gamma, &self.device);
        let targets: Tensor<B, 1> = rewards + gamma_tensor * max_next_q * not_done;

        // MSE loss
        let diff = q_selected - targets;
        diff.powf_scalar(2.0).mean()
    }

    /// Train with gradient accumulation over multiple mini-batches
    ///
    /// # Arguments
    /// * `batches` - Vector of transition batches to accumulate over
    /// * `accumulation_steps` - Number of steps to accumulate (effective batch = batch_size × accumulation_steps)
    ///
    /// # Returns
    /// Average loss over all accumulation steps
    pub fn train_step_accumulated(
        &mut self,
        batches: Vec<TransitionBatch>,
        accumulation_steps: usize,
    ) -> f32 {
        if batches.len() != accumulation_steps {
            tracing::warn!(
                "Batch count {} doesn't match accumulation_steps {}",
                batches.len(),
                accumulation_steps
            );
        }

        if accumulation_steps == 0 {
            tracing::error!("accumulation_steps must be greater than 0");
            return 0.0;
        }

        let mut total_loss = 0.0;

        // Zero gradients at start of accumulation cycle
        self.accumulation_counter = 0;
        self.accumulated_grads = GradientsAccumulator::new();

        for (step, batch) in batches.into_iter().enumerate() {
            self.accumulation_counter = step + 1;

            // Compute loss
            let loss = self.compute_loss(batch);

            // Get scalar loss value BEFORE scaling
            let loss_value: f32 = loss
                .clone()
                .into_data()
                .convert::<f32>()
                .as_slice()
                .expect("Failed to get loss value")[0];
            total_loss += loss_value;

            // Scale loss by accumulation_steps for proper gradient averaging
            // Create a scalar tensor with the scale value for division
            let scale_value = accumulation_steps as f32;
            let scale_tensor: Tensor<B, 1> = Tensor::from_data(
                TensorData::new(vec![scale_value], [1]).convert::<f32>(),
                &loss.device(),
            );
            let scaled_loss = loss.div(scale_tensor);

            // Compute gradients on SCALED loss
            let grads = scaled_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.model);

            // Accumulate scaled gradients
            self.accumulated_grads.accumulate(&self.model, grads_params);
        }

        // Apply accumulated gradients
        let grads = self.accumulated_grads.grads();

        // Update model with CACHED optimizer (massive speedup!)
        // Gradients were already scaled by 1/accumulation_steps during backward,
        // so use the original learning rate
        self.model = self.optimizer.step(self.config.learning_rate, self.model.clone(), grads);

        // Update target network and epsilon (like regular train_step)
        self.step_count += 1;
        if self.config.target_update_freq > 0
            && self.step_count % self.config.target_update_freq == 0
        {
            self.hard_update_target();
        }

        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);

        // Average loss over accumulation steps
        total_loss / accumulation_steps as f32
    }

    /// Clip gradients by global norm
    ///
    /// # Arguments
    /// * `grads` - Parameter gradients to clip
    /// * `max_norm` - Maximum gradient norm (from config)
    ///
    /// # Returns
    /// Gradients (as-is, clipping is handled by optimizer config)
    ///
    /// # Note
    /// Gradient clipping is now handled by optimizer configuration via `with_grad_clipping()`.
    /// This method exists for compatibility but returns gradients unchanged.
    fn clip_gradients(&self, grads: GradientsParams, _max_norm: f32) -> GradientsParams {
        // Gradient clipping is handled by Adam's with_grad_clipping() configuration
        // Applied during optimizer.step() automatically
        grads
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

        // Save policy network using Burn's recorder with dimension info
        // Note: dimensions stored for checkpoint compatibility checking
        let metadata = CheckpointMetadata::new_with_dims(
            "combined_policy".to_string(),
            episode,
            0, // state_dim - not directly accessible from model, will be added to config
            0, // action_dim - not directly accessible from model
            0, // feature_dim - not directly accessible from model
        )
        .with_training_state(self.step_count, episode, self.epsilon, avg_reward);

        crate::training::checkpoint::save_checkpoint::<B, _>(
            &self.model,
            directory,
            name,
            episode,
            &metadata,
        )?;

        // Also save target network (use underscore instead of dot for filename)
        let target_name = format!("{}_target", name);
        crate::training::checkpoint::save_checkpoint::<B, _>(
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
    ///
    /// # Errors
    /// Returns an error if the checkpoint was trained with different model dimensions
    /// (e.g., state_dim changed from 15 to 32). In this case, delete old checkpoints
    /// and retrain with the new dimensions.
    pub fn load_checkpoint<P: AsRef<Path>>(
        path: P,
        config: TrainingConfig,
        model_config: CombinedBanditDQNConfig,
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
        let (model, metadata) = crate::training::checkpoint::load_checkpoint::<B, _>(
            directory,
            name,
            0, // Use epoch 0 for single checkpoint
            &device,
            || model_config.init(&device),
        )?;

        // Check dimension compatibility if checkpoint has dimension info
        // Expected dimensions from model_config
        let expected_state_dim = model_config.bandit.input_dim;
        let expected_feature_dim = model_config.bandit.feature_dim;
        let expected_action_dim = model_config.dqn.action_dim;

        if let Err(e) = metadata.check_dimensions(
            expected_state_dim,
            expected_action_dim,
            expected_feature_dim,
        ) {
            return Err(format!(
                "Checkpoint dimension mismatch detected!\n\
                 {}\n\
                 \n\
                 This error occurs when trying to load a checkpoint trained with different\n\
                 model dimensions (e.g., state_dim changed from 15 to 32).\n\
                 \n\
                 To fix this:\n\
                 1. Delete old checkpoints: rm -rf checkpoints/*\n\
                 2. Retrain from scratch with the new dimensions\n\
                 \n\
                 Expected dimensions: state={}, action={}, feature={}",
                e, expected_state_dim, expected_action_dim, expected_feature_dim
            )
            .into());
        }

        // Load target network (use underscore instead of dot for filename)
        let target_name = format!("{}_target", name);
        let (target_model, _metadata) = crate::training::checkpoint::load_checkpoint::<B, _>(
            directory,
            &target_name,
            0,
            &device,
            || model_config.init(&device),
        )?;

        // Create hybrid buffer with state dimension from model config
        let state_dim = model_config.bandit.input_dim;
        let buffer = HybridRingBuffer::new(config.buffer_capacity, state_dim);

        // Initialize optimizer once (cached for reuse)
        let optimizer = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(
                config.max_gradient_norm,
            )))
            .init();

        Ok(Self {
            model,
            target_model,
            buffer,
            config,
            epsilon: metadata.epsilon,
            step_count: metadata.step_count,
            device: device.clone(),
            accumulated_grads: GradientsAccumulator::new(),
            accumulation_counter: 0,
            accumulated_loss: Tensor::zeros([1], &device),
            accumulated_loss_count: 0,
            loss_sync_freq: 500,    // Sync every 500 steps
            warmup_batch_size: 256, // Start with 256, fills in 8 steps (with 32 envs)
            warmup_complete: false,
            optimizer,
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
