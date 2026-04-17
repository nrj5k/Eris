//! GPU Training Coordinator - Reusable training loop for GPU-native agents
//!
//! This module provides a generic training coordinator that extracts the VecEnv
//! training pattern from Metis and makes it reusable across different policy types
//! (DQN, Bandit, Catcher, etc.).
//!
//! ## Architecture
//!
//! The coordinator works with:
//! - Any agent implementing [`GpuTrainable`] and [`BatchedActionSelector`]
//! - Any environment implementing [`VecEnvironment`]
//! - Any Burn backend (CUDA, WGPU, NdArray)
//!
//! ## Features
//!
//! - **Warmup handling**: Automatic batch size ramp-up (256 → full size)
//! - **Episode management**: Checkpointing, metrics, progress reporting
//! - **Batched action selection**: Single forward pass for all environments
//! - **Parallel environment stepping**: Optional Rayon-based parallelism
//! - **GPU-native storage**: Direct tensor storage without CPU conversion
//!
//! ## Example
//!
//! ```rust,ignore
//! use eris::training::coordinator::GpuTrainingCoordinator;
//! use eris::training::GpuTrainable;
//! use eris::env::VecEnvironment;
//! use burn::backend::Cuda;
//!
//! let coordinator = GpuTrainingCoordinator::new(1000, 500, 512);
//! let metrics = coordinator.run_training(
//!     &mut agent,
//!     &mut vec_env,
//!     &device,
//!     "checkpoints/mymodel",
//! )?;
//!
//! println!("Training complete! Avg reward: {:.2}", metrics.avg_reward);
//! ```

use crate::training::checkpoint::CheckpointMetadataExt;
use crate::utils::backend_diagnostics::log_backend_info;
use burn::tensor::backend::AutodiffBackend;
use std::error::Error;

/// Training metrics returned after completing training.
/// Re-exported from burnme-rly (superset with best_reward and training_time_secs).
///
/// # Fields
///
/// * `total_steps` - Total environment steps across all environments
/// * `total_episodes` - Number of completed episodes
/// * `avg_reward` - Average reward across all episodes
/// * `final_loss` - Average training loss from the final training steps
/// * `steps_per_second` - Optional throughput metric (if measured)
/// * `training_time_secs` - Optional total training time in seconds
/// * `best_reward` - Best single-episode reward during training
///
/// # Example
///
/// ```
/// use eris::training::coordinator::TrainingMetrics;
///
/// let metrics = TrainingMetrics {
///     total_steps: 500_000,
///     total_episodes: 1000,
///     avg_reward: 245.3,
///     final_loss: 0.152,
///     steps_per_second: Some(12500.0),
///     training_time_secs: Some(60.0),
///     best_reward: 312.5,
/// };
///
/// println!(
///     "Trained {} episodes, avg reward: {:.1}",
///     metrics.total_episodes, metrics.avg_reward
/// );
/// ```
pub use burnme_rly::coordinator::TrainingMetrics;

/// Trait for selecting actions in batch for vectorized environments.
/// Re-exported from burnme-rly for DRY.
pub use burnme_rly::traits::BatchedActionSelector;

/// Trait for vectorized environments running multiple instances in parallel.
///
/// This trait abstracts over vectorized environment implementations,
/// enabling the training coordinator to work with any VecEnv variant.
///
/// # Example
///
/// ```rust,ignore
/// impl VecEnvironment for VecEnv {
///     fn num_envs(&self) -> usize {
///         self.num_envs()
///     }
///
///     fn action_space(&self) -> &DiscreteSpace {
///         self.action_space()
///     }
///
///     fn observation_dim(&self) -> usize {
///         self.observation_dim()
///     }
///
///     fn reset_all(&mut self) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
///         self.reset_all()
///     }
///
///     fn step_all(&mut self, actions: Vec<usize>) -> Result<Vec<StepResult>, Box<dyn Error>> {
///         self.step_all(actions)
///     }
///
///     fn step_all_parallel(&mut self, actions: Vec<usize>) -> Result<Vec<StepResult>, Box<dyn Error>> {
///         self.step_all_parallel(actions)
///     }
///
///     fn reset_done_environments(&mut self, results: &[StepResult]) -> Result<Vec<Option<Vec<f64>>>, Box<dyn Error>> {
///         Ok(self.reset_done_environments(results))
///     }
///
///     fn get_current_observations(&self, results: &[StepResult], reset_obs: &[Option<Vec<f64>>]) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
///         Ok(VecEnv::get_current_observations(results, reset_obs))
///     }
/// }
/// ```
pub trait VecEnvironment {
    /// Get number of parallel environments.
    fn num_envs(&self) -> usize;

    /// Get action space (same for all environments).
    fn action_space(&self) -> &crate::space::DiscreteSpace;

    /// Get observation dimension.
    fn observation_dim(&self) -> usize;

    /// Reset all environments and return initial observations.
    fn reset_all(&mut self) -> Result<Vec<Vec<f64>>, Box<dyn Error>>;

    /// Step all environments sequentially.
    ///
    /// # Arguments
    ///
    /// * `actions` - Vec of action indices, one per environment
    ///
    /// # Returns
    ///
    /// Vec of StepResult, one per environment (same order as input)
    fn step_all(
        &mut self,
        actions: Vec<usize>,
    ) -> Result<Vec<crate::env::StepResult>, Box<dyn Error>>;

    /// Step all environments in parallel (if supported).
    ///
    /// Default implementation falls back to sequential stepping.
    /// Implementations using Rayon should override this.
    ///
    /// # Arguments
    ///
    /// * `actions` - Vec of action indices, one per environment
    ///
    /// # Returns
    ///
    /// Vec of StepResult, one per environment (same order as input)
    fn step_all_parallel(
        &mut self,
        actions: Vec<usize>,
    ) -> Result<Vec<crate::env::StepResult>, Box<dyn Error>> {
        // Default fallback to sequential
        self.step_all(actions)
    }

    /// Reset environments that are done and return new observations.
    ///
    /// # Arguments
    ///
    /// * `results` - Results from step_all or step_all_parallel
    ///
    /// # Returns
    ///
    /// Vec of Option<Vec<f64>>: Some(new_obs) for reset envs, None for others
    fn reset_done_environments(
        &mut self,
        results: &[crate::env::StepResult],
    ) -> Result<Vec<Option<Vec<f64>>>, Box<dyn Error>>;

    /// Get current observations for all environments.
    ///
    /// Uses reset observations where available, otherwise uses step results.
    ///
    /// # Arguments
    ///
    /// * `results` - Previous step results
    /// * `reset_obs` - Optional observations from reset_done_environments
    ///
    /// # Returns
    ///
    /// Vec of observations (reset observation if env was reset, otherwise from results)
    fn get_current_observations(
        &self,
        results: &[crate::env::StepResult],
        reset_obs: &[Option<Vec<f64>>],
    ) -> Result<Vec<Vec<f64>>, Box<dyn Error>>;
}

/// Generic GPU training coordinator based on Metis VecEnv implementation.
///
/// This coordinator extracts the training loop pattern from Metis and makes it
/// reusable across different policy types (DQN, Bandit, Catcher, etc.).
///
/// # Type Parameters
///
/// The coordinator is generic over:
/// - Agent type `A` implementing `GpuTrainable` and `BatchedActionSelector`
/// - Environment type `E` implementing `VecEnvironment`
/// - Backend type `B` implementing `AutodiffBackend`
///
/// # Example
///
/// ```rust,ignore
/// use eris::training::coordinator::GpuTrainingCoordinator;
///
/// let coordinator = GpuTrainingCoordinator::new(1000, 500, 512);
///
/// let metrics = coordinator.run_training(
///     &mut agent,
///     &mut vec_env,
///     &device,
///     "checkpoints/model",
/// )?;
/// ```
pub struct GpuTrainingCoordinator {
    /// Number of training episodes
    pub episodes: usize,
    /// Maximum steps per episode
    pub max_steps: usize,
    /// Full batch size for training
    pub batch_size: usize,
    /// Initial batch size during warmup
    pub warmup_batch_size: usize,
    /// How often to save checkpoints (in episodes)
    pub checkpoint_interval: usize,
    /// Training frequency after warmup (train every N steps)
    pub train_frequency: usize,
    /// How often to report progress (in steps)
    pub progress_interval: usize,
    /// Whether to use parallel environment stepping
    pub use_parallel: bool,
}

impl GpuTrainingCoordinator {
    /// Create a new coordinator with default settings.
    ///
    /// # Arguments
    ///
    /// * `episodes` - Number of training episodes
    /// * `max_steps` - Maximum steps per episode
    /// * `batch_size` - Full batch size for training after warmup
    ///
    /// # Returns
    ///
    /// A new coordinator with sensible defaults:
    /// - warmup_batch_size: min(256, batch_size)
    /// - checkpoint_interval: 10 episodes
    /// - train_frequency: 4 steps
    /// - progress_interval: 1000 steps
    /// - use_parallel: true (if parallel feature enabled)
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::coordinator::GpuTrainingCoordinator;
    ///
    /// let coordinator = GpuTrainingCoordinator::new(1000, 500, 512);
    /// assert_eq!(coordinator.episodes, 1000);
    /// assert_eq!(coordinator.batch_size, 512);
    /// assert_eq!(coordinator.warmup_batch_size, 256);
    /// ```
    pub fn new(episodes: usize, max_steps: usize, batch_size: usize) -> Self {
        Self {
            episodes,
            max_steps,
            batch_size,
            warmup_batch_size: 256.min(batch_size),
            checkpoint_interval: 10,
            train_frequency: 4,
            progress_interval: 1000,
            use_parallel: true,
        }
    }

    /// Create a coordinator with custom settings.
    ///
    /// # Arguments
    ///
    /// * `episodes` - Number of training episodes
    /// * `max_steps` - Maximum steps per episode
    /// * `batch_size` - Full batch size for training
    /// * `warmup_batch_size` - Initial batch size during warmup
    /// * `checkpoint_interval` - How often to save checkpoints (episodes)
    /// * `train_frequency` - Training frequency after warmup (steps)
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::coordinator::GpuTrainingCoordinator;
    ///
    /// let coordinator = GpuTrainingCoordinator::with_config(
    ///     1000,   // episodes
    ///     500,    // max_steps
    ///     512,    // batch_size
    ///     128,    // warmup_batch_size
    ///     5,      // checkpoint_interval
    ///     2,      // train_frequency
    /// );
    /// ```
    pub fn with_config(
        episodes: usize,
        max_steps: usize,
        batch_size: usize,
        warmup_batch_size: usize,
        checkpoint_interval: usize,
        train_frequency: usize,
    ) -> Self {
        Self {
            episodes,
            max_steps,
            batch_size,
            warmup_batch_size,
            checkpoint_interval,
            train_frequency,
            progress_interval: 1000,
            use_parallel: true,
        }
    }

    /// Run training with the given agent and environment.
    ///
    /// This method implements the Metis VecEnv training pattern:
    /// 1. Episode loop with tracking
    /// 2. Batched action selection (single forward pass)
    /// 3. Parallel environment stepping
    /// 4. GPU buffer storage (zero-copy)
    /// 5. Training with warmup (small → full batch size)
    /// 6. Checkpointing and metrics reporting
    ///
    /// # Type Parameters
    ///
    /// * `A` - Agent type implementing `GpuTrainable<B>` and `BatchedActionSelector<B>`
    /// * `E` - Environment type implementing `VecEnvironment`
    /// * `B` - Burn backend implementing `AutodiffBackend`
    ///
    /// # Arguments
    ///
    /// * `agent` - The learning agent (DQN, Metis, etc.)
    /// * `env` - Vectorized environment (VecEnv)
    /// * `device` - GPU device for tensor operations
    /// * `checkpoint_prefix` - Prefix for checkpoint filenames
    ///
    /// # Returns
    ///
    /// TrainingMetrics with aggregated statistics, or an error if training fails
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Environment reset fails
    /// - Environment step fails
    /// - Checkpoint saving fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let coordinator = GpuTrainingCoordinator::new(1000, 500, 512);
    ///
    /// let metrics = coordinator.run_training(
    ///     &mut agent,
    ///     &mut vec_env,
    ///     &device,
    ///     "checkpoints/mymodel",
    /// )?;
    ///
    /// println!("Training complete!");
    /// println!("  Episodes: {}", metrics.total_episodes);
    /// println!("  Avg reward: {:.2}", metrics.avg_reward);
    /// ```
    pub fn run_training<A, E, B>(
        &self,
        agent: &mut A,
        env: &mut E,
        device: &B::Device,
        checkpoint_prefix: &str,
    ) -> Result<TrainingMetrics, Box<dyn Error>>
    where
        A: crate::training::GpuTrainable<B>
            + BatchedActionSelector<B>
            + crate::training::checkpoint::Checkpointable<B>,
        E: VecEnvironment,
        B: AutodiffBackend,
    {
        log_backend_info::<B>("GpuTrainingCoordinator::run_training", device);

        // Initialize tracking
        let mut episode_count = 0;
        let mut total_steps = 0;
        let mut episode_rewards: Vec<f64> = Vec::new();
        let mut total_loss = 0.0;
        let mut train_steps = 0;
        let mut steps_since_last_train = 0;
        let mut last_checkpoint_episode = 0; // Track last checkpoint to prevent duplicates
        let mut best_reward: f32 = 0.0; // Track best reward for checkpoint metadata

        // Initialize per-environment tracking
        let num_envs = env.num_envs();
        let mut env_cumulative_rewards: Vec<f64> = vec![0.0; num_envs];
        let mut env_steps: Vec<usize> = vec![0; num_envs];

        // Reset all environments
        let mut observations = env.reset_all()?;

        // Pre-fill buffer with random transitions for fast warmup
        // (standard DQN practice — avoids slow cold-start where GPU is idle)
        let buffer_len = agent.gpu_buffer().len();
        if buffer_len < self.warmup_batch_size {
            let needed = self.warmup_batch_size - buffer_len;
            let action_dim = env.action_space().n;
            let state_dim = env.observation_dim();
            agent
                .gpu_buffer_mut()
                .fill_random(needed, action_dim, state_dim);
            log::info!(
                "[STAGE:WARMUP] Pre-filled buffer with {} random transitions (buffer now has {})",
                needed,
                agent.gpu_buffer().len()
            );
        }

        // Episode loop
        while episode_count < self.episodes {
            // Batched action selection - single forward pass for all envs
            let action_dim = env.action_space().n;
            let actions =
                agent.select_actions_batched(&observations, device, action_dim, agent.epsilon());

            // Environment stepping (parallel or sequential)
            let step_results = if self.use_parallel {
                env.step_all_parallel(actions.clone())?
            } else {
                env.step_all(actions.clone())?
            };

            // Reset done environments
            let reset_obs = env.reset_done_environments(&step_results)?;

            // Collect all transitions for batch push (more efficient than individual pushes)
            let mut states_batch = Vec::with_capacity(num_envs);
            let mut actions_batch = Vec::with_capacity(num_envs);
            let mut rewards_batch = Vec::with_capacity(num_envs);
            let mut next_states_batch = Vec::with_capacity(num_envs);
            let mut dones_batch = Vec::with_capacity(num_envs);

            for (i, (result, &action)) in step_results.iter().zip(actions.iter()).enumerate() {
                let clipped_reward = result.reward.clamp(-1.0, 1.0);

                // Get next state: use reset observation if env was reset, otherwise use result observation
                let next_state = reset_obs
                    .get(i)
                    .cloned()
                    .flatten()
                    .unwrap_or_else(|| result.observation.clone());

                // Collect for batch push
                states_batch.push(observations[i].iter().map(|&x| x as f32).collect());
                actions_batch.push(action);
                rewards_batch.push(clipped_reward as f32);
                next_states_batch.push(next_state.iter().map(|&x| x as f32).collect());
                dones_batch.push(result.done);

                // Track rewards
                env_cumulative_rewards[i] += result.reward;
                env_steps[i] += 1;

                // Episode end handling
                if result.done {
                    let episode_reward = env_cumulative_rewards[i] as f32;
                    episode_rewards.push(env_cumulative_rewards[i]);
                    best_reward = best_reward.max(episode_reward);
                    env_cumulative_rewards[i] = 0.0;
                    env_steps[i] = 0;
                    episode_count += 1;
                }
            }

            // Single batch push to GPU buffer (reduces lock/unlock overhead from N× to 1×)
            agent.gpu_buffer_mut().push_batch(
                states_batch,
                actions_batch,
                rewards_batch,
                next_states_batch,
                dones_batch,
            );

            // Update observations for next step
            observations = env.get_current_observations(&step_results, &reset_obs)?;

            total_steps += num_envs;

            // Training with warmup
            let should_train = crate::training::should_train(
                agent.is_warmup_complete(),
                steps_since_last_train,
                self.train_frequency,
            );

            if should_train {
                let effective_batch_size = if agent.is_warmup_complete() {
                    self.batch_size
                } else {
                    self.warmup_batch_size.min(self.batch_size)
                };

                // Sample directly from GPU buffer - no prefetch overhead
                match agent
                    .gpu_buffer_mut()
                    .sample_batch(effective_batch_size, device)
                {
                    Some(batch) => {
                        let train_result = agent.train_step_gpu(&batch);
                        tracing::debug!(
                            "train_step_gpu returned Some({:.4}), step_count: {}",
                            train_result,
                            agent.step_count()
                        );
                        total_loss += train_result as f64;
                        train_steps += 1;
                    }
                    None => {
                        tracing::debug!(
                            "sample_batch returned None! buffer_len: {}, batch_size: {}, warmup_complete: {}",
                            agent.gpu_buffer().len(),
                            effective_batch_size,
                            agent.is_warmup_complete()
                        );
                    }
                }
                steps_since_last_train = 0;
            } else {
                tracing::debug!(
                    "Skipping training, steps_since_last_train: {}, train_frequency: {}",
                    steps_since_last_train,
                    self.train_frequency
                );
                steps_since_last_train += num_envs;
            }

            // Progress reporting
            if total_steps % self.progress_interval == 0 {
                let avg_reward = if episode_rewards.is_empty() {
                    0.0
                } else {
                    episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64
                };
                let avg_loss = if train_steps > 0 {
                    (total_loss / train_steps as f64) as f32
                } else {
                    0.0
                };

                tracing::info!(
                    "[STAGE:TIME]  Steps: {} ({:.1} per env) | Episodes: {}/{} | Avg Reward: {:.2} | Avg Loss: {:.4} | ε: {:.3}",
                    total_steps,
                    total_steps as f64 / num_envs as f64,
                    episode_count,
                    self.episodes,
                    avg_reward,
                    avg_loss,
                    agent.epsilon()
                );
            }

            // Checkpointing
            if episode_count > 0
                && episode_count % self.checkpoint_interval == 0
                && episode_count != last_checkpoint_episode
            {
                let checkpoint_dir = format!("{}", checkpoint_prefix);
                let checkpoint_name = format!("episode_{}", episode_count);

                // Get metadata from agent and update with training state
                let metadata = agent.checkpoint_metadata().with_training_state(
                    total_steps,
                    episode_count,
                    agent.epsilon(),
                    best_reward,
                );

                match crate::training::checkpoint::save_checkpoint::<B, _>(
                    agent.model(),
                    &checkpoint_dir,
                    &checkpoint_name,
                    episode_count,
                    &metadata,
                ) {
                    Ok(path) => {
                        tracing::info!("[STAGE:checkpoint_saved] {}", path.display());
                        last_checkpoint_episode = episode_count;
                    }
                    Err(e) => {
                        tracing::error!("[STAGE:checkpoint_error] Failed to save: {}", e);
                    }
                }
            }
        }

        // Compute final metrics
        let avg_reward = if episode_rewards.is_empty() {
            0.0
        } else {
            episode_rewards.iter().sum::<f64>() / episode_rewards.len() as f64
        };
        let final_loss = if train_steps > 0 {
            (total_loss / train_steps as f64) as f32
        } else {
            0.0
        };

        Ok(TrainingMetrics {
            total_steps,
            total_episodes: episode_count,
            avg_reward,
            final_loss,
            steps_per_second: None,
            training_time_secs: None,
            best_reward: best_reward as f64,
        })
    }

    /// Run training with progress timing (measures steps per second).
    ///
    /// Same as `run_training` but measures throughput.
    ///
    /// # Arguments
    ///
    /// * `agent` - The learning agent
    /// * `env` - Vectorized environment
    /// * `device` - GPU device
    /// * `checkpoint_prefix` - Checkpoint path prefix
    ///
    /// # Returns
    ///
    /// TrainingMetrics with steps_per_second populated
    pub fn run_training_timed<A, E, B>(
        &self,
        agent: &mut A,
        env: &mut E,
        device: &B::Device,
        checkpoint_prefix: &str,
    ) -> Result<TrainingMetrics, Box<dyn Error>>
    where
        A: crate::training::GpuTrainable<B>
            + BatchedActionSelector<B>
            + crate::training::checkpoint::Checkpointable<B>,
        E: VecEnvironment,
        B: AutodiffBackend,
    {
        let start = std::time::Instant::now();
        let mut metrics = self.run_training(agent, env, device, checkpoint_prefix)?;
        let elapsed = start.elapsed();

        metrics.training_time_secs = Some(elapsed.as_secs_f64());
        metrics.steps_per_second = Some(metrics.total_steps as f64 / elapsed.as_secs_f64());

        tracing::info!(
            "\n[STAGE:DONE] Training complete in {:.2}s ({:.0} steps/sec)",
            elapsed.as_secs_f64(),
            metrics.steps_per_second.unwrap_or(0.0)
        );

        Ok(metrics)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::backend::Backend;

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_coordinator_creation() {
        let coordinator = GpuTrainingCoordinator::new(1000, 500, 512);
        assert_eq!(coordinator.episodes, 1000);
        assert_eq!(coordinator.max_steps, 500);
        assert_eq!(coordinator.batch_size, 512);
        assert_eq!(coordinator.warmup_batch_size, 256);
        assert_eq!(coordinator.checkpoint_interval, 10);
        assert_eq!(coordinator.train_frequency, 4);
    }

    #[test]
    fn test_coordinator_with_config() {
        let coordinator = GpuTrainingCoordinator::with_config(100, 50, 256, 64, 5, 2);
        assert_eq!(coordinator.episodes, 100);
        assert_eq!(coordinator.batch_size, 256);
        assert_eq!(coordinator.warmup_batch_size, 64);
        assert_eq!(coordinator.checkpoint_interval, 5);
        assert_eq!(coordinator.train_frequency, 2);
    }

    #[test]
    fn test_training_metrics() {
        let metrics = TrainingMetrics {
            total_steps: 100_000,
            total_episodes: 200,
            avg_reward: 150.5,
            final_loss: 0.25,
            steps_per_second: Some(5000.0),
            training_time_secs: None,
            best_reward: 0.0,
        };

        assert_eq!(metrics.total_steps, 100_000);
        assert_eq!(metrics.total_episodes, 200);
        assert!((metrics.avg_reward - 150.5).abs() < f64::EPSILON);
        assert!((metrics.final_loss - 0.25).abs() < f32::EPSILON);
        assert_eq!(metrics.steps_per_second, Some(5000.0));
    }
}
