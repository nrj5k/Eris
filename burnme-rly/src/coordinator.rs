//! GPU training coordinator extracted from Metis VecEnv pattern

use crate::{
    buffer::Transition,
    traits::{BatchedActionSelector, GpuTrainable, VecEnvironment},
};
use burn::tensor::backend::AutodiffBackend;
use std::error::Error;

/// Configuration for GPU training coordinator.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of training episodes
    pub episodes: usize,
    /// Maximum steps per episode
    pub max_steps: usize,
    /// Full batch size for training
    pub batch_size: usize,
    /// Initial batch size during warmup
    pub warmup_batch_size: usize,
    /// How often to save checkpoints (episodes)
    pub checkpoint_interval: usize,
    /// Training frequency (train every N steps after warmup)
    pub train_frequency: usize,
}

impl TrainingConfig {
    /// Create new training configuration.
    pub fn new(episodes: usize, max_steps: usize, batch_size: usize) -> Self {
        Self {
            episodes,
            max_steps,
            batch_size,
            warmup_batch_size: 256.min(batch_size),
            checkpoint_interval: 10,
            train_frequency: 4,
        }
    }

    /// Set warmup batch size.
    pub fn with_warmup_batch_size(mut self, size: usize) -> Self {
        self.warmup_batch_size = size.min(self.batch_size);
        self
    }

    /// Set checkpoint interval.
    pub fn with_checkpoint_interval(mut self, interval: usize) -> Self {
        self.checkpoint_interval = interval;
        self
    }

    /// Set training frequency.
    pub fn with_train_frequency(mut self, frequency: usize) -> Self {
        self.train_frequency = frequency;
        self
    }

    /// Validate training configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.episodes == 0 {
            return Err("episodes must be > 0".to_string());
        }
        if self.max_steps == 0 {
            return Err("max_steps must be > 0".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be > 0".to_string());
        }
        if self.checkpoint_interval == 0 {
            return Err("checkpoint_interval must be > 0".to_string());
        }
        if self.train_frequency == 0 {
            return Err("train_frequency must be > 0".to_string());
        }
        Ok(())
    }
}

/// Training metrics returned after training completes.
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Total training steps across all episodes
    pub total_steps: usize,
    /// Total episodes completed
    pub total_episodes: usize,
    /// Average reward per episode
    pub avg_reward: f64,
    /// Final average loss
    pub final_loss: f32,
    /// Total training time (if measured)
    pub training_time_secs: Option<f64>,
}

/// Generic GPU training coordinator based on Metis VecEnv implementation.
///
/// This coordinator extracts the training loop pattern from Metis and makes it
/// reusable across different policy types (DQN, Bandit, Catcher, etc.).
///
/// # Example
/// ```rust,ignore
/// use burnme_rly::{GpuTrainingCoordinator, TrainingConfig};
/// use burn::backend::Cuda;
///
/// let config = TrainingConfig::new(1000, 500, 512);
/// let coordinator = GpuTrainingCoordinator::new(config);
/// let metrics = coordinator.run_training(
///     &mut agent, &mut env, &device, "checkpoints"
/// )?;
/// ```
pub struct GpuTrainingCoordinator {
    /// Training configuration
    pub config: TrainingConfig,
}

impl GpuTrainingCoordinator {
    /// Create new coordinator with configuration.
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Run training with the given agent and environment.
    ///
    /// This method implements the Metis VecEnv training pattern:
    /// 1. Initialize tracking
    /// 2. Episode loop with batched action selection
    /// 3. Parallel environment stepping
    /// 4. CPU buffer storage (O(1) push, no GPU allocation)
    /// 5. Training with warmup (batch conversion at train time)
    /// 6. Checkpointing and metrics
    ///
    /// # Type Parameters
    /// * `A` - Agent type implementing GpuTrainable and BatchedActionSelector
    /// * `E` - Environment type implementing VecEnvironment
    /// * `B` - Burn backend (CUDA, WGPU, etc.)
    ///
    /// # Arguments
    /// * `agent` - The learning agent
    /// * `env` - Vectorized environment
    /// * `device` - GPU device for tensor operations
    /// * `checkpoint_prefix` - Prefix for checkpoint filenames
    ///
    /// # Returns
    /// Training metrics on success
    pub fn run_training<A, E, B>(
        &self,
        agent: &mut A,
        env: &mut E,
        device: &B::Device,
        checkpoint_prefix: &str,
    ) -> Result<TrainingMetrics, Box<dyn Error>>
    where
        A: GpuTrainable<B> + BatchedActionSelector<B>,
        E: VecEnvironment,
        B: AutodiffBackend,
    {
        // Validate config first
        self.config.validate().map_err(|e| {
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, e)) as Box<dyn Error>
        })?;

        // Initialize tracking
        let mut episode_count = 0usize;
        let mut total_steps = 0usize;
        let mut episode_rewards: Vec<f64> = Vec::new();
        let mut total_loss = 0.0f32;
        let mut train_steps = 0usize;

        // Initialize per-environment tracking
        let num_envs = env.num_envs();
        let mut env_cumulative_rewards: Vec<f64> = vec![0.0; num_envs];
        let mut env_steps: Vec<usize> = vec![0; num_envs];

        // Reset all environments
        let mut observations = env.reset_all()?;

        log::info!("Starting training: {} episodes", self.config.episodes);

        // Episode loop
        while episode_count < self.config.episodes && total_steps < self.config.max_steps {
            let action_dim = env.action_space().n();

            // Batched action selection (single forward pass)
            let actions =
                agent.select_actions_batched(&observations, device, action_dim, agent.epsilon());

            // Environment stepping
            let step_results = env.step_all(actions)?;

            // Reset done environments
            let reset_obs = env.reset_done_environments(&step_results)?;

            // Store transitions and track rewards
            for (i, result) in step_results.iter().enumerate() {
                let clipped_reward = result.reward.clamp(-1.0, 1.0);

                // Store in CPU buffer (O(1), no GPU allocation)
                let transition = Transition {
                    state: observations[i].iter().map(|&x| x as f32).collect(),
                    action: result.action,
                    reward: clipped_reward as f32,
                    next_state: reset_obs
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; env.observation_dim()])
                        .iter()
                        .map(|&x| x as f32)
                        .collect(),
                    done: result.done,
                };

                agent.buffer_mut().push(transition);
            }

            // Update observations for next step - get fresh observations after resets
            observations = env.get_current_observations(&step_results, &reset_obs)?;

            // Accumulate per-environment rewards
            for (i, result) in step_results.iter().enumerate() {
                env_cumulative_rewards[i] += result.reward;
                env_steps[i] += 1;
            }

            // Determine if we should train
            let should_train = if agent.is_warmup_complete() {
                // Train every train_frequency steps after warmup
                total_steps.is_multiple_of(self.config.train_frequency)
            } else {
                // Always train during warmup
                true
            };

            if should_train {
                if let Some(loss) = agent.train_step_gpu_native(total_steps) {
                    total_loss += loss;
                    train_steps += 1;
                    // Decay exploration after each training step
                    agent.decay_exploration();

                    // Update target network periodically for Double DQN
                    if agent.step_count() % agent.target_update_freq() == 0 {
                        agent.update_target_network();
                    }
                }
            }

            // Only count episodes when environment is done
            for (i, result) in step_results.iter().enumerate() {
                if result.done {
                    episode_count += 1;
                    episode_rewards.push(env_cumulative_rewards[i]);
                    env_cumulative_rewards[i] = 0.0;
                    env_steps[i] = 0;
                }
            }
            total_steps += 1;

            // Checkpointing
            if episode_count > 0 && episode_count.is_multiple_of(self.config.checkpoint_interval) {
                let checkpoint_path = format!("{}_episode_{}", checkpoint_prefix, episode_count);
                match agent.save_checkpoint(&checkpoint_path) {
                    Ok(_) => log::info!("Saved checkpoint: {}", checkpoint_path),
                    Err(e) => log::error!("Failed to save checkpoint: {}", e),
                }
            }
        }

        // Calculate final metrics
        let avg_reward = if episode_rewards.is_empty() {
            0.0
        } else {
            episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64
        };
        let final_loss = if train_steps > 0 {
            total_loss / train_steps as f32
        } else {
            0.0
        };

        log::info!("Training complete!");
        log::info!(
            "Episodes: {} | Steps: {} | Avg Reward: {:.2} | Final Loss: {:.4}",
            episode_count,
            total_steps,
            avg_reward,
            final_loss
        );

        Ok(TrainingMetrics {
            total_steps,
            total_episodes: episode_count,
            avg_reward,
            final_loss,
            training_time_secs: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::new(100, 500, 256);
        assert_eq!(config.episodes, 100);
        assert_eq!(config.max_steps, 500);
        assert_eq!(config.batch_size, 256);
        assert_eq!(config.warmup_batch_size, 256); // Min of 256 and 256
        assert_eq!(config.checkpoint_interval, 10);
        assert_eq!(config.train_frequency, 4);
    }

    #[test]
    fn test_training_metrics() {
        let metrics = TrainingMetrics {
            total_steps: 1000,
            total_episodes: 10,
            avg_reward: 15.5,
            final_loss: 0.25,
            training_time_secs: Some(60.0),
        };

        assert_eq!(metrics.total_steps, 1000);
        assert_eq!(metrics.total_episodes, 10);
        assert!((metrics.avg_reward - 15.5).abs() < 0.001);
    }

    #[test]
    fn test_training_config_builder() {
        let config = TrainingConfig::new(100, 500, 512)
            .with_warmup_batch_size(128)
            .with_checkpoint_interval(5)
            .with_train_frequency(2);

        assert_eq!(config.episodes, 100);
        assert_eq!(config.max_steps, 500);
        assert_eq!(config.batch_size, 512);
        assert_eq!(config.warmup_batch_size, 128);
        assert_eq!(config.checkpoint_interval, 5);
        assert_eq!(config.train_frequency, 2);
    }

    #[test]
    fn test_training_config_warmup_cap() {
        // Test that warmup_batch_size is capped at batch_size
        let config = TrainingConfig::new(100, 500, 128).with_warmup_batch_size(512);

        assert_eq!(config.warmup_batch_size, 128); // Capped at batch_size
    }

    #[test]
    fn test_training_config_validate_success() {
        let config = TrainingConfig::new(100, 500, 256);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_training_config_validate_episodes_zero() {
        let config = TrainingConfig::new(0, 500, 256);
        let result = config.validate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "episodes must be > 0");
    }

    #[test]
    fn test_training_config_validate_max_steps_zero() {
        let config = TrainingConfig::new(100, 0, 256);
        let result = config.validate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "max_steps must be > 0");
    }

    #[test]
    fn test_training_config_validate_batch_size_zero() {
        let config = TrainingConfig::new(100, 500, 0);
        let result = config.validate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "batch_size must be > 0");
    }

    #[test]
    fn test_training_config_validate_checkpoint_interval_zero() {
        let config = TrainingConfig::new(100, 500, 256).with_checkpoint_interval(0);
        let result = config.validate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "checkpoint_interval must be > 0");
    }

    #[test]
    fn test_training_config_validate_train_frequency_zero() {
        let config = TrainingConfig::new(100, 500, 256).with_train_frequency(0);
        let result = config.validate();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "train_frequency must be > 0");
    }

    #[test]
    fn test_training_metrics_default() {
        let metrics = TrainingMetrics::default();

        assert_eq!(metrics.total_steps, 0);
        assert_eq!(metrics.total_episodes, 0);
        assert!((metrics.avg_reward - 0.0).abs() < 0.001);
        assert!((metrics.final_loss - 0.0).abs() < 0.001);
        assert!(metrics.training_time_secs.is_none());
    }

    #[test]
    fn test_coordinator_creation() {
        let config = TrainingConfig::new(100, 500, 256);
        let coordinator = GpuTrainingCoordinator::new(config.clone());

        assert_eq!(coordinator.config.episodes, config.episodes);
        assert_eq!(coordinator.config.max_steps, config.max_steps);
        assert_eq!(coordinator.config.batch_size, config.batch_size);
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::buffer::CpuRingBuffer;
    use crate::env::{Info, StepResult};
    use crate::space::DiscreteSpace;
    use crate::traits::{BatchedActionSelector, GpuTrainable, VecEnvironment};
    use burn::backend::{Autodiff, NdArray};

    type TestBackend = Autodiff<NdArray>;

    // Simple toy agent for testing
    struct ToyAgent {
        buffer: CpuRingBuffer,
        warmup_complete: bool,
        step_count: usize,
        epsilon: f32,
        epsilon_end: f32,
        epsilon_decay: f32,
        device: <NdArray as burn::prelude::Backend>::Device,
        state_dim: usize,
    }

    impl ToyAgent {
        fn new() -> Self {
            Self {
                buffer: CpuRingBuffer::new(1000),
                warmup_complete: false,
                step_count: 0,
                epsilon: 1.0,
                epsilon_end: 0.01,
                epsilon_decay: 0.995,
                device: <NdArray as burn::prelude::Backend>::Device::default(),
                state_dim: 4,
            }
        }
    }

    impl GpuTrainable<TestBackend> for ToyAgent {
        fn buffer_mut(&mut self) -> &mut CpuRingBuffer {
            &mut self.buffer
        }

        fn buffer(&self) -> &CpuRingBuffer {
            &self.buffer
        }

        fn device(&self) -> &<TestBackend as burn::tensor::backend::Backend>::Device {
            &self.device
        }

        fn state_dim(&self) -> usize {
            self.state_dim
        }

        fn buffer_len(&self) -> usize {
            self.buffer.len()
        }

        fn train_step_gpu_native(&mut self, _steps: usize) -> Option<f32> {
            // Mock training: just return a fake loss
            self.step_count += 1;
            Some(0.5)
        }

        fn warmup_batch_size(&self) -> usize {
            32
        }

        fn is_warmup_complete(&self) -> bool {
            self.warmup_complete
        }

        fn set_warmup_complete(&mut self, complete: bool) {
            self.warmup_complete = complete;
        }

        fn epsilon(&self) -> f32 {
            self.epsilon
        }

        fn step_count(&self) -> usize {
            self.step_count
        }

        fn increment_step_count(&mut self) {
            self.step_count += 1;
        }

        fn batch_size(&self) -> usize {
            64
        }

        fn target_update_freq(&self) -> usize {
            10
        }

        fn learning_rate(&self) -> f32 {
            0.001
        }

        fn gamma(&self) -> f32 {
            0.99
        }

        fn decay_exploration(&mut self) {
            self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_end);
        }

        fn update_target_network(&mut self) {
            // Mock: just increment step count to track calls
            self.step_count += 1;
        }

        fn save_checkpoint(&self, _path: &str) -> Result<(), Box<dyn std::error::Error>> {
            Ok(())
        }

        fn load_checkpoint(&mut self, _path: &str) -> Result<(), Box<dyn std::error::Error>> {
            Ok(())
        }
    }

    impl BatchedActionSelector<TestBackend> for ToyAgent {
        fn select_actions_batched(
            &self,
            _observations: &[Vec<f64>],
            _device: &<TestBackend as burn::tensor::backend::Backend>::Device,
            action_dim: usize,
            _epsilon: f32,
        ) -> Vec<usize> {
            // Return random actions
            use rand::RngExt;
            let mut rng = rand::rng();
            (0.._observations.len())
                .map(|_| rng.random_range(0..action_dim))
                .collect()
        }
    }

    // Simple toy environment
    struct ToyEnv {
        num_envs: usize,
        observations: Vec<Vec<f64>>,
        step_count: Vec<usize>,
        action_space: DiscreteSpace,
    }

    impl ToyEnv {
        fn new(num_envs: usize) -> Self {
            Self {
                num_envs,
                observations: vec![vec![0.0; 4]; num_envs],
                step_count: vec![0; num_envs],
                action_space: DiscreteSpace::new(2),
            }
        }
    }

    impl VecEnvironment for ToyEnv {
        fn num_envs(&self) -> usize {
            self.num_envs
        }

        fn action_space(&self) -> &DiscreteSpace {
            &self.action_space
        }

        fn observation_dim(&self) -> usize {
            4
        }

        fn reset_all(&mut self) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
            self.observations = vec![vec![0.0; 4]; self.num_envs];
            self.step_count = vec![0; self.num_envs];
            Ok(self.observations.clone())
        }

        fn step_all(
            &mut self,
            actions: Vec<usize>,
        ) -> Result<Vec<StepResult>, Box<dyn std::error::Error>> {
            let mut results = Vec::new();
            for (i, action) in actions.iter().enumerate() {
                self.step_count[i] += 1;
                // Simple environment: reward = action as f64, done after 10 steps
                let reward = *action as f64;
                let done = self.step_count[i] >= 10;
                self.observations[i] = vec![self.step_count[i] as f64; 4];
                results.push(StepResult {
                    action: *action,
                    observation: self.observations[i].clone(),
                    reward,
                    done,
                    info: Info::default(),
                });
            }
            Ok(results)
        }

        fn reset_done_environments(
            &mut self,
            results: &[StepResult],
        ) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
            let mut reset_obs = Vec::new();
            for (i, result) in results.iter().enumerate() {
                if result.done {
                    self.step_count[i] = 0;
                    self.observations[i] = vec![0.0; 4];
                    reset_obs.push(self.observations[i].clone());
                }
            }
            Ok(reset_obs)
        }

        fn get_current_observations(
            &self,
            results: &[StepResult],
            _reset_obs: &[Vec<f64>],
        ) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
            Ok(results.iter().map(|r| r.observation.clone()).collect())
        }
    }

    #[test]
    fn test_full_training_loop_completes() {
        let config = TrainingConfig::new(5, 20, 32).with_checkpoint_interval(2);
        let coordinator = GpuTrainingCoordinator::new(config);
        let mut agent = ToyAgent::new();
        let mut env = ToyEnv::new(2);
        let device = <NdArray as burn::prelude::Backend>::Device::default();

        let result =
            coordinator.run_training(&mut agent, &mut env, &device, "/tmp/test_checkpoint");

        assert!(result.is_ok(), "Training should complete without errors");
        let metrics = result.unwrap();
        assert!(
            metrics.total_episodes > 0,
            "Should complete at least 1 episode"
        );
    }

    #[test]
    fn test_exploration_decays() {
        let mut agent = ToyAgent::new();
        let initial_epsilon = agent.epsilon();

        // Decay exploration multiple times
        for _ in 0..100 {
            agent.decay_exploration();
        }

        assert!(
            agent.epsilon() < initial_epsilon,
            "Epsilon should decay over time"
        );
        assert!(
            agent.epsilon() >= agent.epsilon_end,
            "Epsilon should not go below epsilon_end"
        );
    }

    #[test]
    fn test_target_network_updates() {
        let mut agent = ToyAgent::new();
        let initial_steps = agent.step_count();

        // Call update_target_network
        agent.update_target_network();

        assert!(
            agent.step_count() > initial_steps,
            "Target network update should be called"
        );
    }
}
