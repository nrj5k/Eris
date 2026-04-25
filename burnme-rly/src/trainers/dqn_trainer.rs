use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor};

use crate::buffer::GpuRingBuffer;
use crate::checkpoint::{
    load_checkpoint, save_checkpoint, CheckpointMetadata, CheckpointMetadataExt,
};
use crate::loss::{self, LossAccumulator};
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
    /// Batch size for training. Should be warp-aligned (multiple of 32) for optimal
    /// GPU utilization. Default: 2048 (32 × 64 warps).
    ///
    /// # Why Warp Alignment Matters
    ///
    /// NVIDIA GPUs execute instructions in groups of 32 threads called "warps".
    /// Non-warp-aligned batch sizes waste GPU compute cycles because the GPU must
    /// execute partial warps, leaving some threads idle.
    ///
    /// For example, with batch_size=100:
    /// - GPU allocates 4 warps (128 threads)
    /// - Only 100 threads do work
    /// - 28 threads (22%) are wasted
    ///
    /// With batch_size=128 (warp-aligned):
    /// - GPU allocates exactly 4 warps
    /// - All 128 threads do work
    /// - 0% waste, maximum occupancy
    ///
    /// # Recommended Values
    ///
    /// - 2048 = 64 warps (excellent occupancy, Metis default)
    /// - 1024 = 32 warps (good for smaller GPUs)
    /// - 512  = 16 warps (minimum for decent utilization)
    ///
    /// The default 2048 was benchmarked to provide best throughput on most GPUs.
    pub batch_size: usize,
    pub buffer_capacity: usize,
    pub target_update_freq: usize,
    pub max_gradient_norm: f32,

    // NEW: Async loss configuration (from Metis optimization)
    /// Loss sync frequency. Controls how often loss is synced from GPU to CPU.
    /// Higher values reduce GPU→CPU synchronization overhead.
    /// Default: 100 (sync every 100 steps, recommended for performance)
    /// Set to 1 for backward compatibility (sync every step)
    pub loss_sync_freq: usize,

    // NEW: Warmup configuration (from Metis optimization)
    /// Number of warmup steps before using full batch size.
    /// During warmup, uses smaller batches to stabilize training.
    /// Default: 1000
    pub warmup_steps: usize,

    /// Batch size during warmup period.
    /// Should be smaller than batch_size.
    /// Default: 256
    pub warmup_batch_size: usize,
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
            // NEW defaults:
            loss_sync_freq: 500,    // Sync every 500 steps (Metis optimization)
            warmup_steps: 1000,     // 1000 steps of warmup
            warmup_batch_size: 256, // Smaller batches during warmup
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

    /// Set loss sync frequency.
    /// Higher values = fewer GPU→CPU syncs = better performance.
    /// Recommended: 100 (Metis default)
    /// Backward compatible: 1 (sync every step)
    pub fn with_loss_sync_freq(mut self, freq: usize) -> Self {
        self.loss_sync_freq = freq;
        self
    }

    /// Set warmup steps.
    /// Number of steps to use warmup_batch_size before switching to full batch_size.
    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    /// Set warmup batch size.
    /// Smaller batch size used during warmup period.
    pub fn with_warmup_batch_size(mut self, size: usize) -> Self {
        self.warmup_batch_size = size;
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

        // NEW validations:
        if self.loss_sync_freq == 0 {
            return Err("loss_sync_freq must be > 0".to_string());
        }

        if self.warmup_batch_size == 0 {
            return Err("warmup_batch_size must be > 0".to_string());
        }

        if self.warmup_batch_size > self.batch_size {
            return Err(format!(
                "warmup_batch_size ({}) cannot exceed batch_size ({})",
                self.warmup_batch_size, self.batch_size
            ));
        }

        // Warn about non-warp-aligned batch size
        if !self.is_batch_size_warp_aligned() {
            log::warn!(
                "[STAGE:WARN] batch_size {} is not warp-aligned (not a multiple of 32). \
                 Consider using {} for better GPU utilization. \
                 Warp alignment reduces wasted GPU cycles.",
                self.batch_size,
                self.align_batch_size_to_warp()
            );
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
    /// Async loss accumulation - avoids GPU→CPU sync every step (Metis optimization)
    loss_accumulator: LossAccumulator<B>,
    /// Whether warmup phase is complete
    pub warmup_complete: bool,
    /// Episode counter
    pub episode_count: usize,
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

        // Initialize loss accumulator (async loss - Metis optimization)
        let loss_accumulator = LossAccumulator::new(config.loss_sync_freq, &device);

        Ok(Self {
            q_network,
            target_network,
            buffer,
            config: config.clone(),
            step_count: 0,
            epsilon: config.epsilon_start,
            device: device.clone(),
            optimizer,
            loss_accumulator,
            warmup_complete: false,
            episode_count: 0,
        })
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

    /// Check if warmup is complete
    pub fn is_warmup_complete(&self) -> bool {
        self.warmup_complete
    }

    /// Get effective batch size (accounts for warmup)
    pub fn effective_batch_size(&self) -> usize {
        if self.warmup_complete {
            self.config.batch_size
        } else {
            // During warmup, use smaller batch size
            self.config.warmup_batch_size.min(self.config.batch_size)
        }
    }

    /// Mark warmup as complete
    fn complete_warmup(&mut self) {
        if !self.warmup_complete {
            self.warmup_complete = true;
            log::info!(
                "[STAGE:WARMUP] Warmup complete! Using full batch size: {}",
                self.config.batch_size
            );
        }
    }

    /// Force sync accumulated loss (for end of training)
    pub fn flush_loss(&mut self) -> Option<f32> {
        self.loss_accumulator.force_sync()
    }

    /// Execute one DQN training step with Double DQN
    pub fn train_step(&mut self) -> Option<f32> {
        // Check warmup completion
        if !self.warmup_complete && self.step_count >= self.config.warmup_steps {
            self.complete_warmup();
        }

        // 1. Sample batch using effective batch size
        let batch = self.buffer.sample(self.effective_batch_size())?;
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

        // 12. Accumulate loss on GPU (async - Metis optimization)
        self.loss_accumulator.accumulate(loss.detach());

        // 13. Try to sync accumulated loss
        self.loss_accumulator.try_sync()
    }

    /// Save checkpoint (simple wrapper)
    pub fn save(
        &self,
        directory: &std::path::Path,
        name: &str,
        episode: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let metadata = CheckpointMetadata::new("DQN".to_string(), episode, serde_json::json!({}))
            .with_training_state(self.step_count, episode, self.epsilon, 0.0);
        save_checkpoint(&self.q_network, directory, name, episode, &metadata)?;
        Ok(())
    }

    /// Load checkpoint (simple wrapper)
    pub fn load(
        &mut self,
        directory: &std::path::Path,
        name: &str,
        episode: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = || self.q_network.clone();
        let (loaded_model, metadata) =
            load_checkpoint::<B, M>(directory, name, episode, &self.device, config)?;
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

    #[test]
    fn test_config_new_async_fields_defaults() {
        let config = DQNTrainerConfig::default();
        assert_eq!(config.loss_sync_freq, 500);
        assert_eq!(config.warmup_steps, 1000);
        assert_eq!(config.warmup_batch_size, 256);
    }

    #[test]
    fn test_config_new_builder_methods() {
        let config = DQNTrainerConfig::default()
            .with_loss_sync_freq(50)
            .with_warmup_steps(500)
            .with_warmup_batch_size(128);

        assert_eq!(config.loss_sync_freq, 50);
        assert_eq!(config.warmup_steps, 500);
        assert_eq!(config.warmup_batch_size, 128);
    }

    #[test]
    fn test_config_validate_loss_sync_freq_zero() {
        let config = DQNTrainerConfig::default().with_loss_sync_freq(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_warmup_batch_size_zero() {
        let config = DQNTrainerConfig::default().with_warmup_batch_size(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validate_warmup_batch_size_exceeds_batch_size() {
        let config = DQNTrainerConfig::default()
            .with_batch_size(512)
            .with_warmup_batch_size(1024);
        assert!(config.validate().is_err());
    }
}
