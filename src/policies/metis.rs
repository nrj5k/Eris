//! Metis: Combined DQN + Contextual Bandit Policy
//!
//! Refactored to implement CachePolicy trait using existing CombinedModel infrastructure
//! Migrated to use GpuTrainable + GpuTrainingCoordinator pattern

use super::exploration::{ExplorationConfig, ExplorationStrategy};
use super::policy::*;
use super::tensor_utils::{batch_to_tensors, state_to_tensor};
use crate::config::CombinedBanditDQNConfig;
use crate::models::CombinedModel;
use crate::training::checkpoint::{CheckpointMetadata, CheckpointMetadataExt, Checkpointable};
use crate::training::BatchedActionSelector;
use crate::training::{GpuTrainable, HybridRingBuffer};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Distribution, Int, Tensor};
use burnme_rly::buffer::TensorTransitionBatch;
use std::error::Error;
use std::path::Path;

/// Metis Policy - Combined DQN + Bandit using existing model infrastructure
pub struct MetisPolicy<B: AutodiffBackend> {
    /// Policy network (online network)
    pub model: CombinedModel<B>,
    /// Target network (frozen copy)
    pub target_model: CombinedModel<B>,
    /// Optimizer for model parameter updates using Adam with gradient clipping
    optimizer: OptimizerAdaptor<Adam, CombinedModel<B>, B>,
    /// Exploration strategy
    explorer: Box<dyn ExplorationStrategy<B>>,
    /// Configuration
    config: MetisConfig,
    /// Current device
    device: B::Device,
    /// Training step counter
    step_count: usize,
    /// GPU replay buffer for batch training
    pub gpu_buffer: HybridRingBuffer<B>,
    /// Warmup batch size (starts small, ramps up to full batch)
    pub warmup_batch_size: usize,
    /// Full batch size for training
    pub full_batch_size: usize,
    /// Whether warmup phase is complete
    pub warmup_complete: bool,
}

/// Configuration for Metis
#[derive(Clone, Debug)]
pub struct MetisConfig {
    /// Input state dimension
    pub state_dim: usize,
    /// Feature dimension (bandit output)
    pub feature_dim: usize,
    /// Action dimension (number of tiers * 2 for read/write ops)
    pub action_dim: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Discount factor
    pub gamma: f32,
    /// Initial exploration rate (deprecated, use exploration config)
    pub epsilon_start: f32,
    /// Final exploration rate (deprecated, use exploration config)
    pub epsilon_end: f32,
    /// Exploration decay rate (deprecated, use exploration config)
    pub epsilon_decay: f32,
    /// Target network update frequency
    pub target_update_freq: usize,
    /// Batch size
    pub batch_size: usize,
    /// Replay buffer capacity
    pub buffer_capacity: usize,
    /// Exploration strategy configuration
    pub exploration: ExplorationConfig,
    /// Weight for bandit loss in joint training (default: 0.5)
    /// The joint loss = dqn_loss + bandit_loss_weight * bandit_loss
    pub bandit_loss_weight: f32,
    /// Maximum gradient norm for gradient clipping (default: 1.0)
    pub max_gradient_norm: f32,
}

impl Default for MetisConfig {
    fn default() -> Self {
        Self {
            state_dim: 32,   // Warp-aligned dimension (5 tier sizes + 10 features + 17 padding)
            feature_dim: 20, // Bandit output dimension
            action_dim: 10,  // Number of actions (5 tiers * 2 ops)
            learning_rate: 0.0001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            target_update_freq: 1000,
            batch_size: 2048, // Optimized for GPU utilization (multiple of 32 for warp alignment)
            buffer_capacity: 100_000,
            exploration: ExplorationConfig::EpsilonGreedy {
                epsilon_start: 1.0,
                epsilon_end: 0.01,
                epsilon_decay: 0.995,
            },
            bandit_loss_weight: 0.5,
            max_gradient_norm: 1.0,
        }
    }
}

impl<B: AutodiffBackend> MetisPolicy<B> {
    /// Create new MetisPolicy
    ///
    /// # Arguments
    /// * `config` - Metis configuration
    /// * `model_config` - Combined model architecture configuration
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Initialized policy with random weights and empty replay buffer
    pub fn new(
        config: MetisConfig,
        model_config: CombinedBanditDQNConfig,
        device: B::Device,
    ) -> Self {
        let model = model_config.init(&device);
        let target_model = model_config.init(&device);

        // Build exploration strategy from config
        let explorer = config.exploration.build(config.action_dim);

        // Initialize optimizer with Adam and gradient clipping
        let optimizer = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_gradient_norm)))
            .init();

        Self {
            model,
            target_model,
            optimizer,
            explorer,
            config: config.clone(),
            device: device.clone(),
            step_count: 0,
            gpu_buffer: HybridRingBuffer::new(config.buffer_capacity, config.state_dim),
            warmup_batch_size: 256.min(config.batch_size),
            full_batch_size: config.batch_size,
            warmup_complete: false,
        }
    }

    /// Create with epsilon-greedy (backward compatible)
    ///
    /// This method provides backward compatibility with the original METIS implementation
    /// that used hardcoded epsilon-greedy exploration.
    ///
    /// # Arguments
    /// * `config` - Metis configuration
    /// * `model_config` - Combined model architecture configuration
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Initialized policy with epsilon-greedy exploration
    pub fn new_epsilon_greedy(
        config: MetisConfig,
        model_config: CombinedBanditDQNConfig,
        device: B::Device,
    ) -> Self {
        let mut config = config;
        config.exploration = ExplorationConfig::EpsilonGreedy {
            epsilon_start: config.epsilon_start,
            epsilon_end: config.epsilon_end,
            epsilon_decay: config.epsilon_decay,
        };
        Self::new(config, model_config, device)
    }

    /// Forward pass through both bandit and DQN
    ///
    /// # Arguments
    /// * `state` - Input tensor [batch_size, state_dim]
    ///
    /// # Returns
    /// * (features, importance, q_values)
    fn forward(&self, state: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        self.model.forward(state)
    }

    /// Select action using exploration strategy with combined bandit + DQN
    ///
    /// # Arguments
    /// * `state` - Current state
    ///
    /// # Returns
    /// * Action (discrete action index)
    pub fn select_action(&self, state: &State) -> Action {
        // Convert state to tensor
        let state_tensor = state_to_tensor(state, self.config.state_dim, &self.device);

        // Forward pass: get importance and Q-values
        let (_features, importance, q_values) = self.forward(state_tensor);

        // Get importance score as scalar
        let importance_val: f32 = importance
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert importance")[0];

        // Map importance [0, 1] to tier selection
        // Lower importance -> lower tier (cold storage)
        // Higher importance -> higher tier (hot storage)
        let importance_scaled: f32 = importance_val * 5.0;
        let tier_idx = importance_scaled.min(4.0) as usize;

        // Get Q-values for this tier's actions (2 actions per tier: read/write)
        let tier_start = tier_idx * 2;
        let tier_q_values = q_values.slice([0..1, tier_start..tier_start + 2]);

        // Use exploration strategy to select action within tier
        let action_tensor = self.explorer.select_action(&tier_q_values, 2);

        let action_idx = action_tensor
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("Failed to convert action")[0] as usize;

        // Encode action: tier * 2 + operation
        let action = tier_idx * 2 + action_idx;

        Action::Discrete(action)
    }

    /// Joint training step with DQN loss + Bandit loss
    ///
    /// # Arguments
    /// * `states` - Current states [batch_size, state_dim]
    /// * `actions` - Taken actions [batch_size, 1]
    /// * `rewards` - Rewards [batch_size, 1]
    /// * `next_states` - Next states [batch_size, state_dim]
    /// * `dones` - Terminal flags [batch_size, 1]
    ///
    /// # Returns
    /// * Joint loss value (DQN + bandit)
    fn train_joint_step(
        &mut self,
        states: &Tensor<B, 2>,
        actions: &Tensor<B, 2, Int>,
        rewards: &Tensor<B, 2>,
        next_states: &Tensor<B, 2>,
        dones: &Tensor<B, 2>,
    ) -> f32 {
        // Forward pass through the model
        let (_features, importance, q_values) = self.model.forward(states.clone());

        // Gather Q-values for taken actions
        let current_q = q_values.gather(1, actions.clone());

        // Double DQN target computation
        // 1. Get next Q-values from target model
        let next_q_target = self.target_model.forward(next_states.clone()).2;

        // 2. Get best actions from policy model (not target)
        let (_, _, next_q_policy) = self.model.forward(next_states.clone());
        let best_actions = next_q_policy.argmax(1);
        let max_next_q = next_q_target.gather(1, best_actions);

        // 3. Compute TD target: r + γ * max_next_q * (1 - done)
        let target_q = rewards.clone()
            + Tensor::full(rewards.dims(), self.config.gamma, &self.device)
                * max_next_q
                * (Tensor::ones_like(dones) - dones.clone());

        // DQN loss
        let dqn_loss = (current_q - target_q.detach()).powf_scalar(2.0).mean();

        // Bandit loss: MSE between importance scores and normalized rewards
        // Normalize rewards to [0, 1] using min-max scaling
        let min_reward = rewards.clone().min().reshape([1, 1]);
        let max_reward = rewards.clone().max().reshape([1, 1]);
        let reward_range = max_reward.clone() - min_reward.clone();
        let epsilon = Tensor::<B, 2>::full([1, 1], 1e-8, &self.device);
        let normalized_rewards = (rewards.clone() - min_reward) / (reward_range + epsilon);

        // Bandit loss: MSE(importance, normalized_rewards)
        let bandit_loss = (importance - normalized_rewards).powf_scalar(2.0).mean();

        // Joint loss = DQN loss + bandit_loss_weight * bandit loss
        let joint_loss = dqn_loss + self.config.bandit_loss_weight * bandit_loss;

        // Backward pass — gradients flow to BOTH bandit and DQN heads
        let grads = joint_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);

        // Optimizer step
        self.model = self.optimizer.step(
            self.config.learning_rate as f64,
            self.model.clone(),
            grads_params,
        );

        // Increment step count after optimizer step
        self.step_count += 1;

        // Decay exploration parameters
        self.explorer.decay();

        // Update target network periodically
        if self.step_count % self.config.target_update_freq == 0 {
            self.target_model = self.model.clone();
        }

        // Convert to scalar for logging
        let loss_scalar: f32 = joint_loss.into_data().convert::<f32>().as_slice().unwrap()[0];
        loss_scalar
    }

    /// Update target network
    fn update_target_network(&mut self) {
        self.target_model = self.model.clone();
    }
}

impl<B: AutodiffBackend> CachePolicy for MetisPolicy<B> {
    fn select_action(&self, state: &State) -> Action {
        self.select_action(state)
    }

    fn update(&mut self, _transition: &Transition) -> f32 {
        // Updates happen in train_step with batches
        // This method is for online policies
        0.0
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        // Use checkpoint module's save function
        // path is the full path without extension, e.g., "checkpoints/metis_episode_100"
        let parent = path.parent().unwrap_or(Path::new("."));
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("metis");

        crate::training::checkpoint::save_checkpoint(
            &self.model,
            parent,
            name,
            0,
            &self.checkpoint_metadata(),
        )?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
        // Use checkpoint module's load function
        let parent = path.parent().unwrap_or(Path::new("."));
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("metis");

        let (loaded_model, _metadata) = crate::training::checkpoint::load_checkpoint::<B, _>(
            parent,
            name,
            0,
            &self.device,
            || self.model.clone(),
        )?;
        self.model = loaded_model;
        Ok(())
    }

    fn policy_type(&self) -> PolicyType {
        PolicyType::Metis
    }

    fn action_dim(&self) -> usize {
        self.config.action_dim
    }
}

// ============================================================================
// Checkpointable Implementation for MetisPolicy
// ============================================================================

impl<B: AutodiffBackend> Checkpointable<B> for MetisPolicy<B> {
    fn checkpoint_name(&self) -> &str {
        "metis_policy"
    }

    fn checkpoint_metadata(&self) -> CheckpointMetadata {
        CheckpointMetadata::new_with_dims(
            "metis".to_string(),
            self.step_count,
            self.config.state_dim,
            self.config.action_dim,
            self.config.feature_dim,
        )
        .with_training_state(self.step_count, 0, self.epsilon(), 0.0)
    }

    fn model(&self) -> &impl burn::module::Module<B> {
        &self.model
    }
}

// ============================================================================
// BatchedActionSelector Implementation for MetisPolicy
// ============================================================================

impl<B: AutodiffBackend> BatchedActionSelector<B> for MetisPolicy<B> {
    fn select_actions_batched(
        &self,
        observations: &[Vec<f64>],
        device: &B::Device,
        action_dim: usize,
        epsilon: f32,
    ) -> Vec<usize> {
        let batch_size = observations.len();
        if batch_size == 0 {
            return Vec::new();
        }

        // Convert observations to batched tensor [batch_size, state_dim]
        let states_flat: Vec<f32> = observations.iter().flatten().map(|&x| x as f32).collect();
        let states_tensor: Tensor<B, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(states_flat, [batch_size, self.config.state_dim])
                .convert::<f32>(),
            device,
        );

        // Forward pass: get importance and Q-values
        let (_features, importance, q_values) = self.forward(states_tensor);

        // Get importance scores as Vec<f32>
        let importance_vec: Vec<f32> = importance
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert importance");

        // Get Q-values as 2D Vec for exploration
        let q_values_data: Vec<f32> = q_values
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert Q-values");

        // Generate random values for epsilon-greedy on GPU
        let random_vals =
            Tensor::<B, 1>::random([batch_size], Distribution::Uniform(0.0, 1.0), device);
        let random_slice: Vec<f64> = random_vals
            .into_data()
            .convert::<f64>()
            .to_vec::<f64>()
            .expect("Failed to convert random values");

        let mut actions = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            // Map importance [0, 1] to tier selection
            let importance_scaled = (importance_vec[i] * 5.0).min(4.0);
            let tier_idx = importance_scaled as usize;

            // Get Q-values for this tier's actions (2 actions per tier: read/write)
            let tier_start = tier_idx * 2;
            let tier_q_read = q_values_data[i * action_dim + tier_start];
            let tier_q_write = q_values_data[i * action_dim + tier_start + 1];

            // Apply epsilon-greedy within tier
            let action_in_tier = if random_slice[i] < epsilon as f64 {
                // Explore: random action within tier (0 or 1)
                if random_slice[i] < 0.5 {
                    0
                } else {
                    1
                }
            } else {
                // Exploit: choose best action within tier
                if tier_q_read > tier_q_write {
                    0
                } else {
                    1
                }
            };

            // Encode action: tier * 2 + operation
            let action = tier_idx * 2 + action_in_tier;
            actions.push(action);
        }

        actions
    }
}

impl<B: AutodiffBackend> ReplayPolicy for MetisPolicy<B> {
    fn train_step(&mut self, batch: &[Transition]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        // Extract batch components using utility
        let (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor) =
            batch_to_tensors(batch, self.config.state_dim, &self.device);

        // Train with joint loss (DQN + bandit)
        let loss = self.train_joint_step(
            &states_tensor,
            &actions_tensor,
            &rewards_tensor,
            &next_states_tensor,
            &dones_tensor,
        );

        loss
    }

    fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    fn update_target(&mut self) {
        self.update_target_network();
    }
}

// ============================================================================
// GpuTrainable Implementation for MetisPolicy
// ============================================================================

impl<B: AutodiffBackend> GpuTrainable<B> for MetisPolicy<B> {
    fn gpu_buffer_mut(&mut self) -> &mut HybridRingBuffer<B> {
        &mut self.gpu_buffer
    }

    fn gpu_buffer(&self) -> &HybridRingBuffer<B> {
        &self.gpu_buffer
    }

    fn warmup_batch_size(&self) -> usize {
        self.warmup_batch_size
    }

    fn full_batch_size(&self) -> usize {
        self.full_batch_size
    }

    fn is_warmup_complete(&self) -> bool {
        self.warmup_complete
    }

    fn set_warmup_complete(&mut self, complete: bool) {
        self.warmup_complete = complete;
    }

    fn target_update_freq(&self) -> usize {
        self.config.target_update_freq
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn increment_step_count(&mut self) {
        self.step_count += 1;
    }

    fn epsilon(&self) -> f32 {
        self.explorer.get_param()
    }

    fn update_epsilon(&mut self) {
        self.explorer.decay();
    }

    fn train_step_gpu(&mut self, batch: &TensorTransitionBatch<B>) -> f32 {
        let batch_size = batch.states.dims()[0];
        if batch_size == 0 {
            tracing::warn!("Metis train_step_gpu called with empty batch");
            return 0.0;
        }

        // Tensors are already on GPU - no conversion needed!
        // Batch already has rank-2 format [batch_size, 1], reshape for type signature
        let states = batch.states.clone();
        let next_states = batch.next_states.clone();
        let actions = batch.actions.clone().reshape([batch_size, 1]);
        let rewards = batch.rewards.clone().reshape([batch_size, 1]);
        let dones = batch.dones.clone().reshape([batch_size, 1]);

        // Train joint step (DQN + bandit)
        let loss = self.train_joint_step(&states, &actions, &rewards, &next_states, &dones);

        loss
    }

    fn maybe_update_target(&mut self, step_count: usize) {
        if step_count % self.config.target_update_freq == 0 {
            self.update_target_network();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn test_metis_config_default() {
        let config = MetisConfig::default();
        assert_eq!(
            config.state_dim, 32,
            "Default state_dim should be warp-aligned (32)"
        );
        assert_eq!(config.action_dim, 10);
        assert_eq!(config.epsilon_start, 1.0);

        // Verify exploration config is set
        match config.exploration {
            ExplorationConfig::EpsilonGreedy { epsilon_start, .. } => {
                assert_eq!(epsilon_start, 1.0);
            }
            _ => panic!("Expected EpsilonGreedy exploration config"),
        }
    }

    #[test]
    fn test_state_to_tensor() {
        type TestBackend = Autodiff<NdArray>;
        let device = <NdArray as burn::prelude::Backend>::Device::default();
        let config = MetisConfig::default();

        // Test state_to_tensor utility directly
        let state = State::Features(vec![1.0; 32]);
        let tensor = state_to_tensor::<TestBackend>(&state, config.state_dim, &device);

        assert_eq!(tensor.shape().dims, [1, 32]);
    }

    #[test]
    fn test_select_action_exploration() {
        type TestBackend = Autodiff<NdArray>;
        let device = <NdArray as burn::prelude::Backend>::Device::default();
        let mut config = MetisConfig::default();

        // Use epsilon-greedy with high exploration
        config.exploration = ExplorationConfig::EpsilonGreedy {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        #[allow(deprecated)]
        let model_config = crate::config::CombinedBanditDQNConfig::builder()
            .bandit(
                crate::config::BanditConfig::builder()
                    .input_dim(config.state_dim)
                    .hidden_layers(vec![64])
                    .feature_dim(config.feature_dim)
                    .build()
                    .expect("Valid bandit config"),
            )
            .dqn(
                crate::config::DQNConfig::builder()
                    .input_dim(config.feature_dim)
                    .hidden_layers(vec![128])
                    .action_dim(config.action_dim)
                    .build()
                    .expect("Valid DQN config"),
            )
            .build()
            .expect("Valid combined config");

        let policy = MetisPolicy::<TestBackend>::new(config, model_config, device);

        // With epsilon = 1.0, should explore (random actions)
        let state = State::Features(vec![1.0; 32]);
        let action = policy.select_action(&state);

        match action {
            Action::Discrete(a) => assert!(a < 10),
            _ => panic!("Expected discrete action"),
        }
    }

    #[test]
    fn test_policy_type() {
        type TestBackend = Autodiff<NdArray>;
        let device = <NdArray as burn::prelude::Backend>::Device::default();
        let config = MetisConfig::default();

        #[allow(deprecated)]
        let model_config = crate::config::CombinedBanditDQNConfig::builder()
            .bandit(
                crate::config::BanditConfig::builder()
                    .input_dim(config.state_dim)
                    .hidden_layers(vec![64])
                    .feature_dim(config.feature_dim)
                    .build()
                    .expect("Valid bandit config"),
            )
            .dqn(
                crate::config::DQNConfig::builder()
                    .input_dim(config.feature_dim)
                    .hidden_layers(vec![128])
                    .action_dim(config.action_dim)
                    .build()
                    .expect("Valid DQN config"),
            )
            .build()
            .expect("Valid combined config");

        let policy = MetisPolicy::<TestBackend>::new(config, model_config, device);

        assert_eq!(policy.policy_type(), PolicyType::Metis);
        assert_eq!(policy.action_dim(), 10);
    }

    #[test]
    fn test_new_epsilon_greedy_backward_compat() {
        type TestBackend = Autodiff<NdArray>;
        let device = <NdArray as burn::prelude::Backend>::Device::default();
        let config = MetisConfig::default();

        #[allow(deprecated)]
        let model_config = crate::config::CombinedBanditDQNConfig::builder()
            .bandit(
                crate::config::BanditConfig::builder()
                    .input_dim(config.state_dim)
                    .hidden_layers(vec![64])
                    .feature_dim(config.feature_dim)
                    .build()
                    .expect("Valid bandit config"),
            )
            .dqn(
                crate::config::DQNConfig::builder()
                    .input_dim(config.feature_dim)
                    .hidden_layers(vec![128])
                    .action_dim(config.action_dim)
                    .build()
                    .expect("Valid DQN config"),
            )
            .build()
            .expect("Valid combined config");

        // Test backward compatibility method
        let policy = MetisPolicy::<TestBackend>::new_epsilon_greedy(config, model_config, device);

        assert_eq!(policy.policy_type(), PolicyType::Metis);
        assert_eq!(policy.action_dim(), 10);

        // Verify exploration parameter
        let param = policy.explorer.get_param();
        assert!((param - 1.0).abs() < 1e-6); // epsilon_start should be 1.0
    }
}
