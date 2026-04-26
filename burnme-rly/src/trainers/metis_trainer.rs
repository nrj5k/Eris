//! MetisTrainer - Combined DQN + Contextual Bandit training
//!
//! Implements joint training with loss = DQN loss + bandit_weight * bandit_loss

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

use crate::buffer::{GpuRingBuffer, GpuTransitionBatch};
use crate::checkpoint::{
    load_checkpoint, save_checkpoint, CheckpointMetadata, CheckpointMetadataExt,
};
use crate::loss;
use crate::models::CombinedModel;
use crate::trainers::base::{TrainerConfig, TrainerConfigBase};

/// Configuration for MetisTrainer
#[derive(Debug, Clone)]
pub struct MetisTrainerConfig {
    pub base: TrainerConfigBase,
    pub bandit_loss_weight: f32,
}

impl Default for MetisTrainerConfig {
    fn default() -> Self {
        Self {
            base: TrainerConfigBase::default(),
            bandit_loss_weight: 0.5,
        }
    }
}

impl TrainerConfig for MetisTrainerConfig {
    fn gamma(&self) -> f32 {
        self.base.gamma
    }
    fn epsilon_start(&self) -> f32 {
        self.base.epsilon_start
    }
    fn epsilon_end(&self) -> f32 {
        self.base.epsilon_end
    }
    fn epsilon_decay(&self) -> f32 {
        self.base.epsilon_decay
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
        self.base.target_update_freq
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

impl MetisTrainerConfig {
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.base = self.base.with_gamma(gamma);
        self
    }
    pub fn with_epsilon_start(mut self, epsilon: f32) -> Self {
        self.base = self.base.with_epsilon_start(epsilon);
        self
    }
    pub fn with_epsilon_end(mut self, epsilon: f32) -> Self {
        self.base = self.base.with_epsilon_end(epsilon);
        self
    }
    pub fn with_epsilon_decay(mut self, decay: f32) -> Self {
        self.base = self.base.with_epsilon_decay(decay);
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
    pub fn with_target_update_freq(mut self, freq: usize) -> Self {
        self.base = self.base.with_target_update_freq(freq);
        self
    }
    pub fn with_max_gradient_norm(mut self, norm: f32) -> Self {
        self.base = self.base.with_max_gradient_norm(norm);
        self
    }
    pub fn with_loss_sync_freq(mut self, freq: usize) -> Self {
        self.base = self.base.with_loss_sync_freq(freq);
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
    pub fn with_bandit_loss_weight(mut self, weight: f32) -> Self {
        self.bandit_loss_weight = weight;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        self.base.validate()?;
        if self.bandit_loss_weight < 0.0 {
            return Err("bandit_loss_weight must be >= 0".to_string());
        }
        Ok(())
    }
}

/// Metis Trainer - Combined DQN + Contextual Bandit
pub struct MetisTrainer<B: AutodiffBackend> {
    pub model: CombinedModel<B>,
    pub target_model: CombinedModel<B>,
    pub buffer: GpuRingBuffer<B>,
    pub config: MetisTrainerConfig,
    pub step_count: usize,
    pub epsilon: f32,
    pub device: B::Device,
    pub optimizer: OptimizerAdaptor<Adam, CombinedModel<B>, B>,
    /// Async loss accumulation - avoids GPU→CPU sync every step
    loss_accumulator: crate::loss::LossAccumulator<B>,
    pub warmup_complete: bool,
}

impl<B: AutodiffBackend> MetisTrainer<B> {
    /// Create new Metis trainer
    pub fn new(
        model: CombinedModel<B>,
        state_dim: usize,
        config: MetisTrainerConfig,
        device: B::Device,
    ) -> Result<Self, String> {
        config.validate()?;

        let target_model = model.clone();
        let buffer = GpuRingBuffer::new(config.buffer_capacity(), state_dim, &device);

        let optimizer = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(
                config.max_gradient_norm(),
            )))
            .init();

        Ok(Self {
            model,
            target_model,
            buffer,
            config: config.clone(),
            step_count: 0,
            epsilon: config.epsilon_start(),
            device: device.clone(),
            optimizer,
            loss_accumulator: crate::loss::LossAccumulator::new(config.loss_sync_freq(), &device),
            warmup_complete: false,
        })
    }

    /// Get effective batch size (handles warmup)
    fn effective_batch_size(&self) -> usize {
        if self.warmup_complete {
            self.config.batch_size()
        } else {
            self.config
                .warmup_batch_size()
                .min(self.config.batch_size())
        }
    }

    /// Sample batch from buffer
    fn sample_batch(&self) -> Option<GpuTransitionBatch<B>> {
        self.buffer.sample(self.effective_batch_size())
    }

    /// Compute bandit loss (MSE between importance and normalized rewards)
    fn compute_bandit_loss(
        &self,
        importance: &Tensor<B, 2>,
        rewards: &Tensor<B, 1>,
        batch_size: usize,
    ) -> Tensor<B, 1> {
        // Min-max normalization to [0, 1]
        let rewards_2d = rewards.clone().reshape([batch_size, 1]);
        let min_reward = rewards.clone().min().reshape([1, 1]);
        let max_reward = rewards.clone().max().reshape([1, 1]);
        let range = max_reward.clone() - min_reward.clone();
        let epsilon = Tensor::<B, 2>::full([1, 1], 1e-8, &self.device);

        let normalized_rewards = (rewards_2d.clone() - min_reward) / (range + epsilon);

        let diff = importance.clone() - normalized_rewards;
        diff.powf_scalar(2.0).mean()
    }

    /// Compute joint loss: DQN loss + bandit_weight * bandit_loss
    fn compute_joint_loss(
        &self,
        dqn_loss: &Tensor<B, 1>,
        bandit_loss: &Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let weighted_bandit = bandit_loss.clone() * self.config.bandit_loss_weight;
        dqn_loss.clone() + weighted_bandit
    }

    /// Execute one Metis training step
    pub fn train_step(&mut self) -> Option<f32> {
        // Check warmup completion
        if !self.warmup_complete && self.step_count >= self.config.warmup_steps() {
            self.warmup_complete = true;
            log::info!(
                "Warmup complete! Using full batch size: {}",
                self.config.batch_size()
            );
        }

        // 1. Sample batch
        let batch = self.sample_batch()?;
        let batch_size = batch.states.dims()[0];
        if batch_size == 0 {
            return None;
        }

        // 2. Forward through model (both bandit and DQN)
        let (_features, importance, q_values) = self.model.forward(batch.states.clone());

        // 3. Gather current Q(s, a)
        let current_q = loss::gather_q_values(&q_values, &batch.actions);

        // 4. Compute bandit loss
        let bandit_loss = self.compute_bandit_loss(&importance, &batch.rewards, batch_size);

        // 5. Double DQN: select best actions using policy model
        let (_features_next, _, next_q_policy) = self.model.forward(batch.next_states.clone());
        let best_actions = next_q_policy.argmax(1).squeeze::<1>();

        // 6. Target model evaluation
        let (_features_target, _, next_q_target) =
            self.target_model.forward(batch.next_states.detach());
        let best_actions_2d = best_actions.reshape([batch_size, 1]);
        let max_next_q = next_q_target
            .gather(1, best_actions_2d)
            .squeeze::<1>()
            .detach();

        // 7. Compute TD target
        let target_q = loss::compute_td_target(
            &batch.rewards,
            &max_next_q,
            &batch.dones,
            self.config.gamma(),
        );

        // 8. Compute DQN loss
        let dqn_loss = loss::compute_double_dqn_loss(&current_q, &target_q);

        // 9. Compute joint loss
        let joint_loss = self.compute_joint_loss(&dqn_loss, &bandit_loss);

        // 10. Backward and optimize
        let grads = joint_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);

        self.model = self.optimizer.step(
            self.config.learning_rate(),
            self.model.clone(),
            grads_params,
        );

        // 11. Update step count
        self.step_count += 1;

        // 12. Update target network periodically
        if self.config.target_update_freq() > 0
            && self.step_count > 0
            && self
                .step_count
                .is_multiple_of(self.config.target_update_freq())
        {
            self.target_model = self.model.clone();
        }

        // 13. Decay epsilon
        self.epsilon = (self.epsilon * self.config.epsilon_decay()).max(self.config.epsilon_end());

        // 14. Accumulate loss on GPU (async - no sync!)
        self.loss_accumulator.accumulate(joint_loss.detach());

        // 15. Try to sync accumulated loss
        self.loss_accumulator.try_sync()
    }

    /// Save checkpoint
    pub fn save(
        &self,
        directory: &std::path::Path,
        name: &str,
        episode: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let metadata =
            CheckpointMetadata::new("Combined".to_string(), episode, serde_json::json!({}))
                .with_training_state(self.step_count, episode, self.epsilon, 0.0);
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
            load_checkpoint::<B, CombinedModel<B>>(directory, name, episode, &self.device, config)?;
        self.model = loaded_model;
        self.step_count = metadata.step_count;
        self.epsilon = metadata.epsilon;
        Ok(())
    }

    /// Force sync accumulated loss (for end of training)
    pub fn flush_loss(&mut self) -> Option<f32> {
        self.loss_accumulator.force_sync()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_metis_config_defaults() {
        let config = MetisTrainerConfig::default();
        assert!((config.gamma() - 0.99).abs() < 1e-6);
        assert!((config.epsilon_start() - 1.0).abs() < 1e-6);
        assert!((config.epsilon_end() - 0.01).abs() < 1e-6);
        assert!((config.epsilon_decay() - 0.995).abs() < 1e-6);
        assert!((config.learning_rate() - 0.0001).abs() < 1e-6);
        assert_eq!(config.batch_size(), 2048);
        assert_eq!(config.buffer_capacity(), 100_000);
        assert_eq!(config.target_update_freq(), 1000);
        assert!((config.max_gradient_norm() - 1.0).abs() < 1e-6);
        assert!((config.bandit_loss_weight - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_metis_trainer_creation() {
        use burn::backend::{Autodiff, NdArray};
        type TestBackend = Autodiff<NdArray>;

        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let config = MetisTrainerConfig::default();
        let model_config = crate::models::CombinedModelConfig::new(
            10,       // state_dim
            vec![16], // bandit_hidden
            8,        // feature_dim
            vec![32], // dqn_hidden
            4,        // action_dim
        );
        let model = CombinedModel::new(model_config, &device);

        let _trainer = MetisTrainer::<TestBackend>::new(model, 10, config, device).unwrap();
    }

    #[test]
    fn test_metis_config_trait_impl() {
        let config = MetisTrainerConfig::default();
        assert!((config.gamma() - 0.99).abs() < 1e-6);
        assert!((config.epsilon_start() - 1.0).abs() < 1e-6);
        assert!((config.epsilon_end() - 0.01).abs() < 1e-6);
        assert!((config.epsilon_decay() - 0.995).abs() < 1e-6);
        assert!((config.learning_rate() - 0.0001).abs() < 1e-6);
        assert_eq!(config.batch_size(), 2048);
        assert_eq!(config.buffer_capacity(), 100_000);
        assert_eq!(config.target_update_freq(), 1000);
        assert!((config.max_gradient_norm() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_warmup_config_defaults() {
        let config = MetisTrainerConfig::default();
        assert_eq!(config.warmup_steps(), 1000);
        assert_eq!(config.warmup_batch_size(), 256);
    }

    #[test]
    fn test_metis_config_builder_pattern() {
        let config = MetisTrainerConfig::default()
            .with_gamma(0.95)
            .with_epsilon_start(0.9)
            .with_epsilon_end(0.05)
            .with_learning_rate(0.001)
            .with_batch_size(1024)
            .with_buffer_capacity(50_000)
            .with_target_update_freq(500)
            .with_max_gradient_norm(0.5)
            .with_bandit_loss_weight(0.75)
            .with_warmup_steps(500)
            .with_warmup_batch_size(128);

        assert!((config.gamma() - 0.95).abs() < 1e-6);
        assert!((config.epsilon_start() - 0.9).abs() < 1e-6);
        assert!((config.epsilon_end() - 0.05).abs() < 1e-6);
        assert!((config.learning_rate() - 0.001).abs() < 1e-6);
        assert_eq!(config.batch_size(), 1024);
        assert_eq!(config.buffer_capacity(), 50_000);
        assert_eq!(config.target_update_freq(), 500);
        assert!((config.max_gradient_norm() - 0.5).abs() < 1e-6);
        assert!((config.bandit_loss_weight - 0.75).abs() < 1e-6);
        assert_eq!(config.warmup_steps(), 500);
        assert_eq!(config.warmup_batch_size(), 128);
    }

    #[test]
    fn test_warmup_initial_state() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let config = MetisTrainerConfig::default();
        let model = CombinedModel::new(
            crate::models::CombinedModelConfig::new(10, vec![16], 8, vec![32], 4),
            &device,
        );
        let trainer = MetisTrainer::<TestBackend>::new(model, 10, config, device).unwrap();
        assert!(!trainer.warmup_complete);
    }

    #[test]
    fn test_effective_batch_size_during_warmup() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let config = MetisTrainerConfig::default()
            .with_batch_size(2048)
            .with_warmup_batch_size(256)
            .with_warmup_steps(100);
        let model = CombinedModel::new(
            crate::models::CombinedModelConfig::new(10, vec![16], 8, vec![32], 4),
            &device,
        );
        let trainer = MetisTrainer::<TestBackend>::new(model, 10, config, device).unwrap();
        assert_eq!(trainer.effective_batch_size(), 256);
    }

    #[test]
    fn test_effective_batch_size_after_warmup() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let config = MetisTrainerConfig::default()
            .with_batch_size(2048)
            .with_warmup_batch_size(256)
            .with_warmup_steps(100);
        let model = CombinedModel::new(
            crate::models::CombinedModelConfig::new(10, vec![16], 8, vec![32], 4),
            &device,
        );
        let mut trainer = MetisTrainer::<TestBackend>::new(model, 10, config, device).unwrap();
        trainer.warmup_complete = true;
        assert_eq!(trainer.effective_batch_size(), 2048);
    }

    #[test]
    fn test_metis_config_validate_success() {
        let config = MetisTrainerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_metis_config_validate_gamma_invalid() {
        let config = MetisTrainerConfig::default().with_gamma(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_metis_config_validate_bandit_loss_weight_negative() {
        let config = MetisTrainerConfig::default().with_bandit_loss_weight(-0.1);
        assert!(config.validate().is_err());
    }
}
