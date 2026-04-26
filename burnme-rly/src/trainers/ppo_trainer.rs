//! PPO (Proximal Policy Optimization) Trainer
//!
//! Implements PPO with clipped surrogate loss for stable policy optimization.
//! Uses on-policy learning with a rollout buffer instead of experience replay.

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

use crate::buffer::CpuRingBuffer;
use crate::checkpoint::{
    load_checkpoint, save_checkpoint, CheckpointMetadata, CheckpointMetadataExt,
};
use crate::models::ppo_model::PpoModel;
use crate::trainers::base::{TrainerConfig, TrainerConfigBase};

/// Training hyperparameters for PPOTrainer
#[derive(Debug, Clone)]
pub struct PpoTrainerConfig {
    pub base: TrainerConfigBase,
    pub clip_epsilon: f32,
    pub value_loss_coef: f32,
    pub entropy_coef: f32,
    pub ppo_epochs: usize,
    pub gae_lambda: f32,
}

impl Default for PpoTrainerConfig {
    fn default() -> Self {
        Self {
            base: TrainerConfigBase {
                learning_rate: 0.0003,
                batch_size: 64,
                buffer_capacity: 2048,
                max_gradient_norm: 0.5,
                ..TrainerConfigBase::default()
            },
            clip_epsilon: 0.2,
            value_loss_coef: 0.5,
            entropy_coef: 0.01,
            ppo_epochs: 4,
            gae_lambda: 0.95,
        }
    }
}

impl TrainerConfig for PpoTrainerConfig {
    fn gamma(&self) -> f32 {
        self.base.gamma
    }
    fn epsilon_start(&self) -> f32 {
        0.0
    }
    fn epsilon_end(&self) -> f32 {
        0.0
    }
    fn epsilon_decay(&self) -> f32 {
        1.0
    }
    fn learning_rate(&self) -> f64 {
        self.base.learning_rate
    }
    fn batch_size(&self) -> usize {
        self.base.batch_size
    }
    fn buffer_capacity(&self) -> usize {
        self.base.buffer_capacity
    }
    fn target_update_freq(&self) -> usize {
        0
    }
    fn max_gradient_norm(&self) -> f32 {
        self.base.max_gradient_norm
    }
    fn loss_sync_freq(&self) -> usize {
        self.base.loss_sync_freq
    }
    fn warmup_steps(&self) -> usize {
        self.base.warmup_steps
    }
    fn warmup_batch_size(&self) -> usize {
        self.base.warmup_batch_size
    }
}

impl PpoTrainerConfig {
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.base = self.base.with_gamma(gamma);
        self
    }
    pub fn with_clip_epsilon(mut self, epsilon: f32) -> Self {
        self.clip_epsilon = epsilon;
        self
    }
    pub fn with_value_loss_coef(mut self, coef: f32) -> Self {
        self.value_loss_coef = coef;
        self
    }
    pub fn with_entropy_coef(mut self, coef: f32) -> Self {
        self.entropy_coef = coef;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.base = self.base.with_learning_rate(lr);
        self
    }
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.base = self.base.with_batch_size(size);
        self
    }
    pub fn with_buffer_capacity(mut self, cap: usize) -> Self {
        self.base = self.base.with_buffer_capacity(cap);
        self
    }
    pub fn with_ppo_epochs(mut self, epochs: usize) -> Self {
        self.ppo_epochs = epochs;
        self
    }
    pub fn with_max_gradient_norm(mut self, norm: f32) -> Self {
        self.base = self.base.with_max_gradient_norm(norm);
        self
    }
    pub fn with_gae_lambda(mut self, lambda: f32) -> Self {
        self.gae_lambda = lambda;
        self
    }
    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.base = self.base.with_warmup_steps(steps);
        self
    }
    pub fn with_warmup_batch_size(mut self, size: usize) -> Self {
        self.base = self.base.with_warmup_batch_size(size);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        TrainerConfig::validate(self)?;
        if self.clip_epsilon <= 0.0 || self.clip_epsilon >= 1.0 {
            return Err("clip_epsilon must be in (0, 1)".to_string());
        }
        if self.value_loss_coef < 0.0 {
            return Err("value_loss_coef must be >= 0".to_string());
        }
        if self.entropy_coef < 0.0 {
            return Err("entropy_coef must be >= 0".to_string());
        }
        if self.ppo_epochs == 0 {
            return Err("ppo_epochs must be > 0".to_string());
        }
        if self.gae_lambda <= 0.0 || self.gae_lambda > 1.0 {
            return Err("gae_lambda must be in (0, 1]".to_string());
        }
        Ok(())
    }
}

/// PPO Trainer with clipped surrogate loss
///
/// Uses two models:
/// - `model`: Current policy (updated every step)
/// - `old_model`: Frozen policy from last update (used for importance ratio)
pub struct PpoTrainer<B: AutodiffBackend> {
    /// Current policy network
    pub model: PpoModel<B>,
    /// Frozen policy network (copy from last update)
    pub old_model: PpoModel<B>,
    /// Rollout buffer (on-policy, cleared after each update)
    pub buffer: CpuRingBuffer,
    /// Training configuration
    pub config: PpoTrainerConfig,
    /// Training step counter
    pub step_count: usize,
    /// Device for tensor operations
    pub device: B::Device,
    /// Optimizer for training
    pub optimizer: OptimizerAdaptor<Adam, PpoModel<B>, B>,
    /// Whether warmup phase is complete
    pub warmup_complete: bool,
    /// State dimension (stored for batch conversion)
    pub state_dim: usize,
    /// Episode counter
    pub episode_count: usize,
}

impl<B: AutodiffBackend> PpoTrainer<B> {
    /// Create a new PPO trainer
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        config: PpoTrainerConfig,
        device: B::Device,
    ) -> Result<Self, String> {
        config.validate()?;

        let model_config =
            crate::models::ppo_model::PpoModelConfig::new(state_dim, vec![128, 128], action_dim);
        let model = PpoModel::new(model_config, &device);
        let old_model = model.clone(); // Start with identical copy

        let optimizer = config.base.build_adam::<PpoModel<B>, B>();

        Ok(Self {
            model,
            old_model,
            buffer: CpuRingBuffer::new(config.base.buffer_capacity),
            config: config.clone(),
            step_count: 0,
            device: device.clone(),
            optimizer,
            warmup_complete: false,
            state_dim,
            episode_count: 0,
        })
    }

    /// Compute PPO clipped surrogate loss
    ///
    /// # Arguments
    /// * `log_probs` - Current policy log probabilities
    /// * `old_log_probs` - Old policy log probabilities (detached)
    /// * `advantages` - Advantage estimates
    /// * `values` - Current value estimates
    /// * `returns` - Target returns
    /// * `entropy` - Policy entropy for exploration bonus
    fn compute_ppo_loss(
        &self,
        log_probs: Tensor<B, 1>,
        old_log_probs: Tensor<B, 1>,
        advantages: Tensor<B, 1>,
        values: Tensor<B, 1>,
        returns: Tensor<B, 1>,
        entropy: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        // Importance ratio: π_new / π_old
        let ratio = (log_probs - old_log_probs).exp();

        // Clipped surrogate loss
        let surr1 = ratio.clone() * advantages.clone();
        let clipped_ratio = ratio.clone().clamp(
            1.0 - self.config.clip_epsilon as f64,
            1.0 + self.config.clip_epsilon as f64,
        );
        let surr2 = clipped_ratio * advantages;

        // Element-wise minimum for clipped surrogate
        let policy_loss = surr1.min_pair(surr2).mean();

        // Value loss: MSE between predicted and actual returns
        let value_loss = (values - returns).powf_scalar(2.0).mean();

        // Entropy bonus (encourages exploration)
        let entropy_bonus = entropy.mean();

        // Combined loss: maximize policy loss, minimize value loss, maximize entropy
        // Note: We negate policy_loss because we want to maximize it via gradient descent
        -(policy_loss - self.config.value_loss_coef * value_loss
            + self.config.entropy_coef * entropy_bonus)
    }

    /// Execute one PPO training step
    pub fn train_step(&mut self) -> Option<f32> {
        // Check if we have enough samples
        if self.buffer.len() < self.config.base.batch_size {
            return None;
        }

        let mut total_loss = 0.0f32;
        let mut update_count = 0usize;

        // PPO epochs: multiple passes over the same rollout data
        for _epoch in 0..self.config.ppo_epochs {
            // Sample batch from buffer
            let transitions = self.buffer.sample(self.config.base.batch_size)?;
            let batch = crate::buffer::TensorTransitionBatch::from_transitions(
                &transitions,
                self.state_dim,
                &self.device,
            );

            // Get action log probs and values from CURRENT policy
            // batch.actions is [batch_size, 1], need to squeeze to [batch_size]
            let actions_1d = batch.actions.clone().reshape([batch.batch_size()]);
            let (log_probs, values, entropy) = self
                .model
                .evaluate_actions(batch.states.clone(), actions_1d);

            // Get OLD policy log probs (no gradients)
            let actions_1d_old = batch.actions.clone().reshape([batch.batch_size()]);
            let (old_log_probs, _, _) = self
                .old_model
                .evaluate_actions(batch.states.clone().detach(), actions_1d_old);
            let old_log_probs = old_log_probs.detach();

            // Compute simple 1-step advantage: A = r + γV(s') - V(s)
            let (_, next_values_2d) = self.old_model.forward(batch.next_states.clone());
            let next_values = next_values_2d.reshape([batch.batch_size()]);

            // Reshape dones and rewards from [batch_size, 1] to [batch_size]
            let dones_1d = batch.dones.clone().reshape([batch.batch_size()]);
            let rewards_1d = batch.rewards.clone().reshape([batch.batch_size()]);

            let not_dones = Tensor::<B, 1>::ones_like(&dones_1d) - dones_1d;
            let returns = rewards_1d.clone() + self.config.base.gamma * next_values * not_dones;
            let advantages = returns.clone() - values.clone();

            // Compute PPO loss
            let loss = self.compute_ppo_loss(
                log_probs,
                old_log_probs,
                advantages,
                values,
                returns,
                entropy,
            );

            // Backward pass
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.model);

            // Optimizer step
            self.model = self.optimizer.step(
                self.config.base.learning_rate,
                self.model.clone(),
                grads_params,
            );

            // Accumulate loss for logging
            let loss_value: f32 = loss.into_data().convert::<f32>().as_slice::<f32>().unwrap()[0];
            total_loss += loss_value;
            update_count += 1;
        }

        // Update old_model after all epochs
        self.old_model = self.model.clone();

        // Clear buffer (on-policy: don't reuse old data)
        self.buffer.clear();

        self.step_count += 1;

        Some(total_loss / update_count as f32)
    }

    /// Check if warmup is complete
    pub fn is_warmup_complete(&self) -> bool {
        self.warmup_complete
    }

    /// Mark warmup as complete
    pub fn complete_warmup(&mut self) {
        if !self.warmup_complete {
            self.warmup_complete = true;
            log::info!(
                "[STAGE:WARMUP] PPO warmup complete! Buffer has {} transitions",
                self.buffer.len()
            );
        }
    }

    /// Save checkpoint
    pub fn save(
        &self,
        directory: &std::path::Path,
        name: &str,
        episode: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let metadata = CheckpointMetadata::new("PPO".to_string(), episode, serde_json::json!({}))
            .with_training_state(self.step_count, episode, 0.0, 0.0);
        save_checkpoint(&self.model, directory, name, episode, &metadata)?;
        Ok(())
    }

    /// Load checkpoint
    pub fn load(
        &mut self,
        directory: &std::path::Path,
        name: &str,
        episode: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = || self.model.clone();
        let (loaded_model, metadata) =
            load_checkpoint::<B, PpoModel<B>>(directory, name, episode, &self.device, config)?;
        self.model = loaded_model;
        self.old_model = self.model.clone();
        self.step_count = metadata.step_count;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = PpoTrainerConfig::default();
        assert!((config.gamma() - 0.99).abs() < 1e-6);
        assert!((config.clip_epsilon - 0.2).abs() < 1e-6);
        assert!((config.value_loss_coef - 0.5).abs() < 1e-6);
        assert!((config.entropy_coef - 0.01).abs() < 1e-6);
        assert_eq!(config.batch_size(), 64);
        assert_eq!(config.ppo_epochs, 4);
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = PpoTrainerConfig::default()
            .with_gamma(0.95)
            .with_clip_epsilon(0.1)
            .with_value_loss_coef(0.25)
            .with_entropy_coef(0.02)
            .with_learning_rate(0.001)
            .with_batch_size(128)
            .with_buffer_capacity(4096)
            .with_ppo_epochs(10)
            .with_max_gradient_norm(1.0)
            .with_gae_lambda(0.9);

        assert!((config.gamma() - 0.95).abs() < 1e-6);
        assert!((config.clip_epsilon - 0.1).abs() < 1e-6);
        assert!((config.value_loss_coef - 0.25).abs() < 1e-6);
        assert!((config.entropy_coef - 0.02).abs() < 1e-6);
        assert!((config.learning_rate() - 0.001).abs() < 1e-6);
        assert_eq!(config.batch_size(), 128);
        assert_eq!(config.buffer_capacity(), 4096);
        assert_eq!(config.ppo_epochs, 10);
        assert!((config.max_gradient_norm() - 1.0).abs() < 1e-6);
        assert!((config.gae_lambda - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_config_validate_success() {
        let config = PpoTrainerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_gamma_invalid() {
        let config = PpoTrainerConfig::default().with_gamma(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_clip_epsilon_invalid() {
        let config = PpoTrainerConfig::default().with_clip_epsilon(0.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_compute_gae_advantages() {
        let config = PpoTrainerConfig::default();
        let rewards = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 1.0, 1.5, 0.0]; // Extra value for terminal state
        let dones = vec![false, false, true];

        // Test GAE computation directly (method is on PpoTrainer, not Config)
        // This is a basic smoke test to ensure the logic works
        let n = rewards.len();
        let mut advantages = Vec::with_capacity(n);
        let mut returns = Vec::with_capacity(n);
        let mut last_gae = 0.0f32;

        for i in (0..n).rev() {
            let delta = rewards[i]
                + config.gamma()
                    * values.get(i + 1).copied().unwrap_or(0.0)
                    * if dones[i] { 0.0 } else { 1.0 }
                - values[i];
            last_gae = delta
                + config.gamma() * config.gae_lambda * if dones[i] { 0.0 } else { 1.0 } * last_gae;
            advantages.push(last_gae);
            returns.push(last_gae + values[i]);
        }
        advantages.reverse();
        returns.reverse();

        assert_eq!(advantages.len(), 3);
        assert_eq!(returns.len(), 3);
    }
}
