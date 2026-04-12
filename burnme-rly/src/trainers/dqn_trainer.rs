use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor};

use crate::buffer::{GpuRingBuffer, GpuTransitionBatch};
use crate::checkpoint::{load_checkpoint, save_checkpoint, CheckpointMetadata};
use crate::loss;
use crate::trainers::base::TrainerConfig;

/// Trait for modules that can compute Q-values
///
/// This extends AutodiffModule with a forward method for Q-value computation.
pub trait QNetwork<B: AutodiffBackend>: AutodiffModule<B> {
    /// Forward pass returning Q-values
    ///
    /// # Arguments
    /// * `states` - Input states tensor \[batch_size, state_dim\]
    ///
    /// # Returns
    /// Q-values tensor \[batch_size, action_dim\]
    fn forward_q(&self, states: Tensor<B, 2>) -> Tensor<B, 2>;
}

/// Training hyperparameters for DQNTrainer
#[derive(Debug, Clone)]
pub struct DQNTrainerConfig {
    pub gamma: f32,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay: f32,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub buffer_capacity: usize,
    pub target_update_freq: usize,
    pub max_gradient_norm: f32,
}

impl Default for DQNTrainerConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            learning_rate: 0.0001,
            batch_size: 2048,
            buffer_capacity: 100_000,
            target_update_freq: 1000,
            max_gradient_norm: 1.0,
        }
    }
}

impl TrainerConfig for DQNTrainerConfig {
    fn gamma(&self) -> f32 {
        self.gamma
    }
    fn epsilon_start(&self) -> f32 {
        self.epsilon_start
    }
    fn epsilon_end(&self) -> f32 {
        self.epsilon_end
    }
    fn epsilon_decay(&self) -> f32 {
        self.epsilon_decay
    }
    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }
    fn batch_size(&self) -> usize {
        self.batch_size
    }
    fn buffer_capacity(&self) -> usize {
        self.buffer_capacity
    }
    fn target_update_freq(&self) -> usize {
        self.target_update_freq
    }
    fn max_gradient_norm(&self) -> f32 {
        self.max_gradient_norm
    }
}

impl DQNTrainerConfig {
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }
    pub fn with_epsilon_start(mut self, epsilon: f32) -> Self {
        self.epsilon_start = epsilon;
        self
    }
    pub fn with_epsilon_end(mut self, epsilon: f32) -> Self {
        self.epsilon_end = epsilon;
        self
    }
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }
    pub fn with_buffer_capacity(mut self, cap: usize) -> Self {
        self.buffer_capacity = cap;
        self
    }
    pub fn with_target_update_freq(mut self, freq: usize) -> Self {
        self.target_update_freq = freq;
        self
    }
    pub fn with_max_gradient_norm(mut self, norm: f32) -> Self {
        self.max_gradient_norm = norm;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.gamma <= 0.0 || self.gamma > 1.0 {
            return Err("gamma must be in (0, 1]".to_string());
        }
        if self.epsilon_start < 0.0 || self.epsilon_start > 1.0 {
            return Err("epsilon_start must be in [0, 1]".to_string());
        }
        if self.epsilon_end < 0.0 || self.epsilon_end > 1.0 {
            return Err("epsilon_end must be in [0, 1]".to_string());
        }
        if self.epsilon_end > self.epsilon_start {
            return Err("epsilon_end must be <= epsilon_start".to_string());
        }
        if self.epsilon_decay <= 0.0 || self.epsilon_decay > 1.0 {
            return Err("epsilon_decay must be in (0, 1]".to_string());
        }
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be > 0".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be > 0".to_string());
        }
        if self.buffer_capacity == 0 {
            return Err("buffer_capacity must be > 0".to_string());
        }
        if self.max_gradient_norm <= 0.0 {
            return Err("max_gradient_norm must be > 0".to_string());
        }
        Ok(())
    }
}

/// DQN Trainer with Double DQN and GPU-native operations
pub struct DQNTrainer<B: AutodiffBackend, M: QNetwork<B>> {
    /// Policy network (online)
    pub q_network: M,
    /// Target network (frozen copy)
    pub target_network: M,
    /// Experience replay buffer
    pub buffer: GpuRingBuffer<B>,
    /// Training configuration
    pub config: DQNTrainerConfig,
    /// Training step counter
    pub step_count: usize,
    /// Current exploration rate
    pub epsilon: f32,
    /// Device for tensor operations
    pub device: B::Device,
    /// Optimizer for training
    pub optimizer: OptimizerAdaptor<Adam, M, B>,
}

impl<B: AutodiffBackend, M: QNetwork<B> + Clone> DQNTrainer<B, M> {
    /// Create a new DQN trainer
    pub fn new(
        q_network: M,
        state_dim: usize,
        config: DQNTrainerConfig,
        device: B::Device,
    ) -> Result<Self, String> {
        config.validate()?;

        let target_network = q_network.clone();
        let buffer = GpuRingBuffer::new(config.buffer_capacity, state_dim, &device);

        let optimizer = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_gradient_norm)))
            .init();

        Ok(Self {
            q_network,
            target_network,
            buffer,
            config: config.clone(),
            step_count: 0,
            epsilon: config.epsilon_start,
            device,
            optimizer,
        })
    }

    /// Sample batch from buffer
    fn sample_batch(&self) -> Option<GpuTransitionBatch<B>> {
        self.buffer.sample(self.config.batch_size)
    }

    /// Forward pass through policy network
    fn forward_q_values(&self, states: Tensor<B, 2>) -> Tensor<B, 2> {
        self.q_network.forward_q(states)
    }

    /// Select best actions for next states (Double DQN)
    fn select_best_actions(&self, next_states: Tensor<B, 2>) -> Tensor<B, 1, Int> {
        let next_q = self.q_network.forward_q(next_states);
        next_q.argmax(1).squeeze::<1>()
    }

    /// Forward pass through target network
    fn forward_target_q(&self, next_states: Tensor<B, 2>) -> Tensor<B, 2> {
        self.target_network.forward_q(next_states)
    }

    /// Backward pass and optimizer step
    fn backward_and_step(&mut self, loss: Tensor<B, 1>) {
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.q_network);

        self.q_network = self.optimizer.step(
            self.config.learning_rate,
            self.q_network.clone(),
            grads_params,
        );
    }

    /// Update target network periodically
    fn maybe_update_target(&mut self) {
        if self.config.target_update_freq > 0
            && self.step_count > 0
            && self
                .step_count
                .is_multiple_of(self.config.target_update_freq)
        {
            self.target_network = self.q_network.clone();
        }
    }

    /// Decay epsilon
    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.config.epsilon_decay).max(self.config.epsilon_end);
    }

    /// Execute one DQN training step with Double DQN
    pub fn train_step(&mut self) -> Option<f32> {
        // 1. Sample batch
        let batch = self.sample_batch()?;
        let batch_size = batch.states.dims()[0];
        if batch_size == 0 {
            return None;
        }

        // 2. Forward pass: current Q-values
        let q_values = self.forward_q_values(batch.states);

        // 3. Gather Q(s, a) using loss module
        let current_q = loss::gather_q_values(&q_values, &batch.actions);

        // 4. Double DQN: select best actions using policy network
        let best_actions = self.select_best_actions(batch.next_states.clone());

        // 5. Target network forward
        let target_q_values = self.forward_target_q(batch.next_states);

        // 6. Gather max Q from target using best actions
        let best_actions_2d = best_actions.reshape([batch_size, 1]);
        let max_next_q = target_q_values
            .gather(1, best_actions_2d)
            .squeeze::<1>()
            .detach();

        // 7. Compute TD target using loss module
        let targets =
            loss::compute_td_target(&batch.rewards, &max_next_q, &batch.dones, self.config.gamma);

        // 8. MSE loss using loss module
        let loss = loss::compute_double_dqn_loss(&current_q, &targets);

        // 9. Backward + optimizer step
        self.backward_and_step(loss.clone());

        // 10. Update step count and target
        self.step_count += 1;
        self.maybe_update_target();

        // 11. Decay epsilon
        self.decay_epsilon();

        // 12. Return loss using loss module
        Some(loss::loss_to_scalar(loss))
    }

    /// Save checkpoint (simple wrapper)
    pub fn save(
        &self,
        path: &std::path::Path,
        episode: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let metadata = CheckpointMetadata {
            step_count: self.step_count,
            epsilon: self.epsilon,
            episode,
        };
        save_checkpoint(&self.q_network, &metadata, path)
    }

    /// Load checkpoint (simple wrapper)
    pub fn load(&mut self, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        let config = || self.q_network.clone();
        let (loaded_model, metadata) = load_checkpoint::<B, M>(path, &self.device, config)?;
        self.q_network = loaded_model;
        self.step_count = metadata.step_count;
        self.epsilon = metadata.epsilon;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = DQNTrainerConfig::default();
        assert!((config.gamma - 0.99).abs() < 1e-6);
        assert!((config.epsilon_start - 1.0).abs() < 1e-6);
        assert_eq!(config.batch_size, 2048);
    }

    #[test]
    fn test_config_builder_pattern() {
        let config = DQNTrainerConfig::default()
            .with_gamma(0.95)
            .with_epsilon_start(0.9)
            .with_epsilon_end(0.05)
            .with_learning_rate(0.001)
            .with_batch_size(1024)
            .with_buffer_capacity(50_000)
            .with_target_update_freq(500)
            .with_max_gradient_norm(0.5);

        assert!((config.gamma - 0.95).abs() < 1e-6);
        assert!((config.epsilon_start - 0.9).abs() < 1e-6);
        assert!((config.epsilon_end - 0.05).abs() < 1e-6);
        assert!((config.learning_rate - 0.001).abs() < 1e-6);
        assert_eq!(config.batch_size, 1024);
        assert_eq!(config.buffer_capacity, 50_000);
        assert_eq!(config.target_update_freq, 500);
        assert!((config.max_gradient_norm - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_config_validate_success() {
        let config = DQNTrainerConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validate_gamma_invalid() {
        let config = DQNTrainerConfig::default().with_gamma(1.5);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_epsilon_end_greater_than_start() {
        let config = DQNTrainerConfig::default()
            .with_epsilon_start(0.1)
            .with_epsilon_end(0.9);
        assert!(config.validate().is_err());
    }
}
