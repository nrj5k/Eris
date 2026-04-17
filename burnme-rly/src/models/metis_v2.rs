//! MetisV2: Joint Bandit + DQN training with SequentialCompose
//!
//! This is the "true Metis" implementation with joint loss:
//! joint_loss = dqn_loss + bandit_loss_weight * bandit_loss

use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};

use crate::buffer::{CpuRingBuffer, TensorTransitionBatch};
use crate::models::composable::SequentialCompose;
use crate::models::ComposableModel;
use crate::traits::{BatchedActionSelector, GpuTrainable};

/// Configuration for MetisV2Policy
#[derive(Debug, Clone)]
pub struct MetisV2Config {
    /// Weight for bandit loss in joint training (default: 0.5)
    /// The joint loss = dqn_loss + bandit_loss_weight * bandit_loss
    pub bandit_loss_weight: f32,

    /// Maximum gradient norm for gradient clipping (default: 1.0)
    pub max_gradient_norm: f32,

    /// Learning rate for Adam optimizer
    pub learning_rate: f64,

    /// Discount factor for Q-learning
    pub gamma: f32,

    /// Starting epsilon for exploration
    pub epsilon_start: f32,

    /// Ending epsilon for exploration
    pub epsilon_end: f32,

    /// Epsilon decay rate
    pub epsilon_decay: f32,

    /// Batch size for training
    pub batch_size: usize,

    /// Warmup batch size (before full training)
    pub warmup_batch_size: usize,

    /// Replay buffer capacity
    pub buffer_capacity: usize,

    /// Target network update frequency (in steps)
    pub target_update_freq: usize,

    /// Loss synchronization frequency (how often to report loss)
    pub loss_sync_freq: usize,
}

impl Default for MetisV2Config {
    fn default() -> Self {
        Self {
            bandit_loss_weight: 0.5,
            max_gradient_norm: 1.0,
            learning_rate: 0.0001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            batch_size: 32,
            warmup_batch_size: 256,
            buffer_capacity: 10000,
            target_update_freq: 100,
            loss_sync_freq: 100,
        }
    }
}

impl MetisV2Config {
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.bandit_loss_weight < 0.0 {
            return Err("bandit_loss_weight must be >= 0".to_string());
        }
        if self.max_gradient_norm <= 0.0 {
            return Err("max_gradient_norm must be > 0".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be > 0".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be > 0".to_string());
        }
        Ok(())
    }

    /// Builder pattern: with_bandit_loss_weight
    pub fn with_bandit_loss_weight(mut self, weight: f32) -> Self {
        self.bandit_loss_weight = weight;
        self
    }

    /// Builder pattern: with_max_gradient_norm
    pub fn with_max_gradient_norm(mut self, norm: f32) -> Self {
        self.max_gradient_norm = norm;
        self
    }
}

/// MetisV2 Policy: Joint Bandit + DQN training
///
/// Uses SequentialCompose to chain a bandit (perception) with a DQN (decision).
/// The bandit provides features and importance scores, the DQN uses features
/// to compute Q-values. Both are trained jointly with:
///   joint_loss = dqn_loss + bandit_loss_weight * bandit_loss
pub struct MetisV2Policy<B, A, M>
where
    B: AutodiffBackend,
    A: ComposableModel<B> + burn::module::AutodiffModule<B> + Send + Clone + std::fmt::Debug,
    M: ComposableModel<B> + burn::module::AutodiffModule<B> + Send + Clone + std::fmt::Debug,
    A::InnerModule: ComposableModel<B::InnerBackend>,
    M::InnerModule: ComposableModel<B::InnerBackend>,
{
    /// The composed model: Bandit -> DQN
    pub model: SequentialCompose<B, A, M>,

    /// Target model for Double DQN
    pub target_model: SequentialCompose<B, A, M>,

    /// Adam optimizer with gradient clipping
    pub optimizer: OptimizerAdaptor<Adam, SequentialCompose<B, A, M>, B>,

    /// Replay buffer for experience replay
    pub buffer: CpuRingBuffer,

    /// Configuration
    pub config: MetisV2Config,

    /// Current training step count
    pub step_count: usize,

    /// Current exploration parameter (epsilon)
    pub epsilon: f32,

    /// Device for tensor operations
    pub device: B::Device,

    /// Whether warmup is complete
    pub warmup_complete: bool,

    /// Accumulated loss for async reporting
    accumulated_loss: Tensor<B, 1>,

    /// Count of accumulated loss samples
    accumulated_count: usize,

    /// Closure to compute importance from bandit output
    /// Takes features tensor, returns importance scores
    importance_fn: Box<dyn Fn(Tensor<B, 2>) -> Tensor<B, 2>>,
}

impl<B, A, M> MetisV2Policy<B, A, M>
where
    B: AutodiffBackend,
    A: crate::models::ComposableModel<B>
        + AutodiffModule<B>
        + Send
        + Clone
        + std::fmt::Debug
        + 'static,
    M: crate::models::ComposableModel<B>
        + AutodiffModule<B>
        + Send
        + Clone
        + std::fmt::Debug
        + 'static,
    A::InnerModule: ComposableModel<B::InnerBackend>,
    M::InnerModule: ComposableModel<B::InnerBackend>,
{
    /// Create new MetisV2Policy
    pub fn new(
        model: SequentialCompose<B, A, M>,
        config: MetisV2Config,
        device: B::Device,
        importance_fn: Box<dyn Fn(Tensor<B, 2>) -> Tensor<B, 2>>,
    ) -> Result<Self, String> {
        config.validate()?;

        let target_model = model.clone();
        let optimizer = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_gradient_norm)))
            .init();

        let buffer = CpuRingBuffer::new(config.buffer_capacity);
        let accumulated_loss = Tensor::<B, 1>::zeros([1], &device);

        let epsilon = config.epsilon_start;
        Ok(Self {
            model,
            target_model,
            optimizer,
            buffer,
            config,
            step_count: 0,
            epsilon,
            device: device.clone(),
            warmup_complete: false,
            accumulated_loss,
            accumulated_count: 0,
            importance_fn,
        })
    }

    /// Check if should train based on warmup and step count
    #[allow(dead_code)]
    fn should_train(&self) -> bool {
        if !self.warmup_complete && self.buffer.len() >= self.config.batch_size {
            // Can complete warmup
            true
        } else {
            self.warmup_complete
        }
    }

    /// Decay exploration (epsilon)
    fn decay_exploration(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }

    /// Maybe update target network
    fn maybe_update_target(&mut self) {
        if self.step_count > 0
            && self
                .step_count
                .is_multiple_of(self.config.target_update_freq)
        {
            self.target_model = self.model.clone();
        }
    }

    /// State dimension (from model input)
    #[allow(dead_code)]
    fn state_dim(&self) -> usize {
        // This is a placeholder - in practice we'd need to track this
        // For now, return a reasonable default or extract from model
        32
    }

    /// Compute joint training step with DQN + Bandit loss
    ///
    /// Joint loss = dqn_loss + bandit_loss_weight * bandit_loss
    pub fn train_step_gpu(&mut self, batch: &TensorTransitionBatch<B>) -> f32 {
        let batch_size = batch.states.shape().dims[0];

        // 1. Forward through model_a (bandit) for features
        let features = self.model.forward_a(batch.states.clone());

        // 2. Compute importance using the closure
        let importance = (self.importance_fn)(features.clone());

        // 3. Forward through full model for Q-values
        // We need features detached for DQN (stop-gradient between bandit and DQN)
        let features_detached = features.detach();
        let q_values = self.model.forward_b(features_detached);

        // 4. Gather Q(s, a) for taken actions
        let current_q = q_values.gather(1, batch.actions.clone());

        // 5. Double DQN target
        // Select actions using policy model
        let next_q_policy = self.model.forward(batch.next_states.clone());
        let best_actions = next_q_policy.argmax(1).reshape([batch_size, 1]);

        // Evaluate using target model (detached)
        let next_q_target = self.target_model.forward(batch.next_states.clone());
        let max_next_q = next_q_target.gather(1, best_actions).detach();

        // Compute TD target: r + γ * max_next_q * (1 - done)
        let target_q = batch.rewards.clone()
            + Tensor::<B, 2>::full_like(&batch.rewards, self.config.gamma)
                * max_next_q
                * (Tensor::<B, 2>::ones_like(&batch.dones) - batch.dones.clone());

        // 6. DQN loss: MSE(Q(s,a), target)
        let dqn_loss = (current_q - target_q).powf_scalar(2.0).mean();

        // 7. Bandit loss: MSE(importance, normalized_rewards)
        // Normalize rewards to [0, 1]
        let min_reward = batch.rewards.clone().min().reshape([1, 1]);
        let max_reward = batch.rewards.clone().max().reshape([1, 1]);
        let reward_range = max_reward.clone() - min_reward.clone();
        let epsilon = Tensor::<B, 2>::full([1, 1], 1e-8, &self.device);
        let normalized_rewards = (batch.rewards.clone() - min_reward) / (reward_range + epsilon);

        let bandit_loss = (importance - normalized_rewards).powf_scalar(2.0).mean();

        // 8. Joint loss
        let joint_loss = dqn_loss + self.config.bandit_loss_weight * bandit_loss;

        // 9. Backward pass
        let grads = joint_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);

        // 10. Optimizer step
        self.model =
            self.optimizer
                .step(self.config.learning_rate, self.model.clone(), grads_params);

        // 11. Update counters
        self.step_count += 1;
        self.decay_exploration();
        self.maybe_update_target();

        if !self.warmup_complete && self.buffer.len() >= self.config.batch_size {
            self.warmup_complete = true;
        }

        // 12. Async loss accumulation
        self.accumulated_loss = self.accumulated_loss.clone() + joint_loss.clone();
        self.accumulated_count += 1;

        if self
            .accumulated_count
            .is_multiple_of(self.config.loss_sync_freq)
        {
            let avg_loss = self.accumulated_loss.clone() / self.accumulated_count as f32;
            let loss_scalar: f32 = avg_loss.into_data().convert::<f32>().as_slice().unwrap()[0];
            self.accumulated_loss = Tensor::<B, 1>::zeros([1], &self.device);
            self.accumulated_count = 0;
            loss_scalar
        } else {
            0.0 // Don't sync every step
        }
    }

    /// Get effective batch size (warmup vs full)
    fn effective_batch_size(&self) -> usize {
        if self.warmup_complete {
            self.config.batch_size
        } else {
            self.config.warmup_batch_size.min(self.config.batch_size)
        }
    }
}

// ============================================================================
// GpuTrainable Trait Implementation
// ============================================================================

impl<B, A, M> GpuTrainable<B> for MetisV2Policy<B, A, M>
where
    B: AutodiffBackend,
    A: crate::models::ComposableModel<B>
        + AutodiffModule<B>
        + Send
        + Clone
        + std::fmt::Debug
        + 'static,
    M: crate::models::ComposableModel<B>
        + AutodiffModule<B>
        + Send
        + Clone
        + std::fmt::Debug
        + 'static,
    A::InnerModule: ComposableModel<B::InnerBackend>,
    M::InnerModule: ComposableModel<B::InnerBackend>,
{
    fn buffer_mut(&mut self) -> &mut CpuRingBuffer {
        &mut self.buffer
    }

    fn buffer(&self) -> &CpuRingBuffer {
        &self.buffer
    }

    fn train_step_gpu(&mut self, batch: &TensorTransitionBatch<B>) -> f32 {
        // Call the existing train_step_gpu method
        self.train_step_gpu(batch)
    }

    fn train_step_gpu_native(
        &mut self,
        _steps_since_last_train: usize,
        device: &B::Device,
    ) -> Option<f32> {
        // Sample batch and train
        let batch_size = self.effective_batch_size();
        let transitions = self.buffer.sample(batch_size)?;
        let batch = TensorTransitionBatch::from_transitions(&transitions, batch_size, device);
        let loss = self.train_step_gpu(&batch);
        Some(loss)
    }

    fn is_warmup_complete(&self) -> bool {
        self.warmup_complete
    }

    fn set_warmup_complete(&mut self, complete: bool) {
        self.warmup_complete = complete;
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn increment_step_count(&mut self) {
        self.step_count += 1;
    }

    fn epsilon(&self) -> f32 {
        self.epsilon
    }

    fn decay_exploration(&mut self) {
        self.decay_exploration();
    }

    fn target_update_freq(&self) -> usize {
        self.config.target_update_freq
    }

    fn update_target_network(&mut self) {
        self.maybe_update_target();
    }

    fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    fn warmup_batch_size(&self) -> usize {
        self.config.warmup_batch_size
    }

    fn device(&self) -> &B::Device {
        &self.device
    }

    fn state_dim(&self) -> usize {
        // Extract from model or use a reasonable default
        // For MetisV2, this would typically come from the bandit model's input dim
        32
    }

    fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    fn learning_rate(&self) -> f32 {
        self.config.learning_rate as f32
    }

    fn gamma(&self) -> f32 {
        self.config.gamma
    }

    fn save_checkpoint(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

        let checkpoint_dir = std::path::Path::new(path);
        std::fs::create_dir_all(checkpoint_dir)?;

        // Save model
        let model_path = checkpoint_dir.join("model.mpk");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.model.clone().save_file(&model_path, &recorder)?;

        // Save target model
        let target_path = checkpoint_dir.join("target_model.mpk");
        self.target_model
            .clone()
            .save_file(&target_path, &recorder)?;

        // Save metadata
        use crate::checkpoint::Checkpointable as _;
        let metadata = self.checkpoint_metadata();
        let meta_path = checkpoint_dir.join("metadata.json");
        std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)?;

        Ok(())
    }

    fn load_checkpoint(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

        let checkpoint_dir = std::path::Path::new(path);

        // Load model
        let model_path = checkpoint_dir.join("model.mpk");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        self.model = self
            .model
            .clone()
            .load_file(&model_path, &recorder, &self.device)?;

        // Load target model
        let target_path = checkpoint_dir.join("target_model.mpk");
        self.target_model =
            self.target_model
                .clone()
                .load_file(&target_path, &recorder, &self.device)?;

        Ok(())
    }
}

// ============================================================================
// Checkpointable Trait Implementation
// ============================================================================

impl<B, A, M> crate::Checkpointable<B> for MetisV2Policy<B, A, M>
where
    B: AutodiffBackend,
    A: crate::models::ComposableModel<B>
        + AutodiffModule<B>
        + Send
        + Clone
        + std::fmt::Debug
        + 'static,
    M: crate::models::ComposableModel<B>
        + AutodiffModule<B>
        + Send
        + Clone
        + std::fmt::Debug
        + 'static,
    A::InnerModule: ComposableModel<B::InnerBackend>,
    M::InnerModule: ComposableModel<B::InnerBackend>,
{
    fn checkpoint_name(&self) -> &str {
        "metis_v2"
    }

    fn checkpoint_metadata(&self) -> crate::CheckpointMetadata {
        crate::CheckpointMetadata::new_with_dims(
            self.checkpoint_name().to_string(),
            self.step_count,
            self.state_dim(),
            0, // action_dim - not applicable for MetisV2
            0, // feature_dim - not applicable
        )
    }

    fn model(&self) -> &impl Module<B> {
        &self.model
    }
}

// ============================================================================
// BatchedActionSelector Trait Implementation
// ============================================================================

impl<B, A, M> BatchedActionSelector<B> for MetisV2Policy<B, A, M>
where
    B: AutodiffBackend,
    A: crate::models::ComposableModel<B>
        + AutodiffModule<B>
        + Send
        + Clone
        + std::fmt::Debug
        + 'static,
    M: crate::models::ComposableModel<B>
        + AutodiffModule<B>
        + Send
        + Clone
        + std::fmt::Debug
        + 'static,
    A::InnerModule: ComposableModel<B::InnerBackend>,
    M::InnerModule: ComposableModel<B::InnerBackend>,
{
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

        // Convert observations to tensor
        let obs_data: Vec<f32> = observations
            .iter()
            .flat_map(|obs| obs.iter().map(|&x| x as f32))
            .collect();
        let state_dim = observations[0].len();
        let states = Tensor::from_data(TensorData::new(obs_data, [batch_size, state_dim]), device);

        // Forward through model to get Q-values
        let q_values = self.model.forward(states);

        // Epsilon-greedy selection using GPU-native approach
        // Generate random values on GPU for all samples at once
        let random_vals =
            Tensor::<B, 1>::random([batch_size], Distribution::Uniform(0.0, 1.0), device);

        // Get greedy actions: argmax of Q-values
        let greedy_actions_2d = q_values.argmax(1); // [batch_size, 1]
        let greedy_actions: Tensor<B, 1, Int> = greedy_actions_2d.reshape([batch_size]); // [batch_size]

        // Generate random actions on GPU [batch_size] with values in [0, action_dim)
        let random_float = Tensor::<B, 1>::random(
            [batch_size],
            Distribution::Uniform(0.0, action_dim as f64),
            device,
        );
        let random_actions: Tensor<B, 1, Int> = random_float.int(); // [batch_size]

        // Create explore mask: random_vals < epsilon
        let explore_mask = random_vals.lower_elem(epsilon as f64); // Tensor<B, 1, Bool>
        let explore_int: Tensor<B, 1, Int> = explore_mask.int(); // 1 for explore, 0 for exploit

        // Select actions: where explore_int == 1, use random; else use greedy
        let selected = random_actions.mask_where(explore_int.equal_elem(0), greedy_actions);

        // Convert to Vec<usize>
        let actions_data = selected.into_data().convert::<i64>();
        let actions_slice: &[i64] = actions_data.as_slice().unwrap();
        actions_slice.iter().map(|&x| x as usize).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MetisV2Config::default();
        assert!((config.bandit_loss_weight - 0.5).abs() < 1e-6);
        assert!((config.max_gradient_norm - 1.0).abs() < 1e-6);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_validate_passes() {
        let config = MetisV2Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_fails_negative_weight() {
        let config = MetisV2Config::default().with_bandit_loss_weight(-1.0);
        assert!(config.validate().is_err());
    }
}
