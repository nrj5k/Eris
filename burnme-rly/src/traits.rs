//! Core traits for GPU-native training
//!
//! This module defines the key abstractions for GPU-native training
//! across different policy types (DQN, Bandit, Catcher, Metis, etc.).

#![allow(unexpected_cfgs)]

use crate::env::StepResult;
use crate::space::DiscreteSpace;
use burn::tensor::backend::AutodiffBackend;
use std::error::Error;

/// Trait for agents that support GPU-native training with configurable buffer type.
///
/// This trait abstracts GPU-native training operations that work
/// across different policy types (DQN, Bandit, Catcher, Metis, etc.).
///
/// # Example
/// ```ignore
/// use burnme_rly::traits::GpuTrainable;
/// use burnme_rly::buffer::CpuRingBuffer;
/// use burn::tensor::backend::AutodiffBackend;
///
/// // Implement for your policy
/// struct MyPolicy<B: AutodiffBackend> {
///     buffer: CpuRingBuffer,
///     warmup_complete: bool,
///     step_count: usize,
///     // ... other fields
/// }
///
/// impl<B: AutodiffBackend> GpuTrainable<B, CpuRingBuffer> for MyPolicy<B> {
///     fn buffer_mut(&mut self) -> &mut CpuRingBuffer {
///         &mut self.buffer
///     }
///     
///     fn buffer(&self) -> &CpuRingBuffer {
///         &self.buffer
///     }
///     
///     fn train_step_gpu_native(&mut self, steps_since_last_train: usize, device: &B::Device) -> Option<f32> {
///         // GPU-native training implementation
///         None
///     }
///     
///     fn warmup_batch_size(&self) -> usize { 256 }
///     fn is_warmup_complete(&self) -> bool { self.warmup_complete }
///     fn set_warmup_complete(&mut self, complete: bool) { self.warmup_complete = complete; }
///     fn epsilon(&self) -> f32 { 1.0 }
///     fn step_count(&self) -> usize { self.step_count }
///     fn increment_step_count(&mut self) { self.step_count += 1; }
///     fn batch_size(&self) -> usize { 512 }
///     fn target_update_freq(&self) -> usize { 1000 }
///     fn learning_rate(&self) -> f32 { 0.001 }
///     fn gamma(&self) -> f32 { 0.99 }
/// }
/// ```
pub trait GpuTrainable<B: AutodiffBackend, Buf> {
    /// Get mutable reference to the replay buffer.
    fn buffer_mut(&mut self) -> &mut Buf;

    /// Get immutable reference to the replay buffer.
    fn buffer(&self) -> &Buf;

    /// Perform a GPU-native training step with warmup support.
    ///
    /// # Double DQN Implementation Pattern (from Metis reference)
    ///
    /// This method should implement the complete Double DQN training pipeline:
    ///
    /// ```ignore
    /// use burnme_rly::buffer::{CpuRingBuffer, TensorTransitionBatch};
    /// use burn::tensor::{backend::AutodiffBackend, Tensor, Int};
    ///
    /// impl<B: AutodiffBackend> MyPolicy<B> {
    ///     fn train_step_gpu_native(&mut self, _steps: usize, device: &B::Device) -> Option<f32> {
    ///         let batch_size = self.effective_batch_size();
    ///         
    ///         // 1. Sample from buffer (CPU → GPU conversion)
    ///         let transitions = self.buffer().sample(batch_size)?;
    ///         let batch = TensorTransitionBatch::<B>::from_transitions(
    ///             &transitions, self.state_dim(), &self.device()
    ///         );
    ///         
    ///         // 2. Forward pass: current Q-values for all actions
    ///         //    Model output: \[batch_size, action_dim\]
    ///         let q_values = self.model().forward(batch.states);
    ///         
    ///         // 3. Gather Q-values for taken actions using gather()
    ///         //    actions: \[batch_size, 1\], q_values: \[batch_size, action_dim\]
    ///         //    current_q: \[batch_size, 1\]
    ///         let current_q = q_values.gather(1, batch.actions);
    ///         
    ///         // 4. Double DQN: Policy network selects actions for next states
    ///         //    best_actions: \[batch_size, 1\]
    ///         let next_q_policy = self.model().forward(batch.next_states);
    ///         let best_actions = next_q_policy.argmax(1);
    ///         
    ///         // 5. Target network evaluates selected actions
    ///         //    max_next_q: \[batch_size, 1\]
    ///         let next_q_target = self.target_model().forward(batch.next_states);
    ///         let max_next_q = next_q_target.gather(1, best_actions);
    ///         
    ///         // 6. Compute target: r + γ * max_a' Q_target(s', a') * (1 - done)
    ///         //    rewards: \[batch_size, 1\], dones: \[batch_size, 1\]
    ///         let gamma = self.gamma();
    ///         let target_q = batch.rewards
    ///             + Tensor::full([1], gamma, &self.device())
    ///                 * max_next_q
    ///                 * (Tensor::ones_like(&batch.dones) - batch.dones);
    ///         
    ///         // 7. MSE loss: (current_q - target_q.detach())^2.mean()
    ///         let diff = current_q - target_q.detach();
    ///         let squared = diff.powf(Tensor::full([1], 2.0_f32, &self.device()));
    ///         let loss = squared.mean();
    ///         
    ///         // 8. Backward pass and optimizer step
    ///         let grads = loss.backward();
    ///         self.optimizer().step(&mut self.model(), grads);
    ///         
    ///         // 9. Return loss value for logging
    ///         Some(loss.into_data().convert::<f32>().as_slice().unwrap()[0])
    ///     }
    /// }
    /// ```
    ///
    /// # Arguments
    /// * `steps_since_last_train` - Number of steps since last training
    /// * `device` - GPU device for tensor operations (REQUIRED for GPU training)
    ///
    /// # Returns
    /// * `Some(loss)` if training occurred
    /// * `None` if training was skipped (e.g., during warmup, insufficient samples)
    fn train_step_gpu_native(
        &mut self,
        steps_since_last_train: usize,
        device: &B::Device,
    ) -> Option<f32>;

    /// Perform a GPU-native training step on a pre-built batch.
    ///
    /// This method takes a pre-sampled batch and performs the training step.
    /// It does NOT handle sampling, warmup checks, or training frequency -
    /// those should be handled by the caller.
    ///
    /// # Arguments
    /// * `batch` - Pre-sampled batch from buffer
    ///
    /// # Returns
    /// Loss value from training step
    fn train_step_gpu(&mut self, batch: &crate::buffer::TensorTransitionBatch<B>) -> f32;

    /// Perform a training step with an optional pre-built batch.
    ///
    /// When `prebuilt_batch` is `Some(batch)`, uses that batch directly.
    /// When `None`, falls back to internal sampling.
    ///
    /// This enables double-buffering: while the GPU trains on batch N, the CPU can
    /// prepare batch N+1 via `PrefetchBuffer`, then pass it here.
    ///
    /// # Arguments
    /// * `steps_since_last_train` - Number of steps since last training
    /// * `device` - GPU device for tensor operations
    /// * `prebuilt_batch` - Optional pre-built batch from prefetch buffer
    ///
    /// # Returns
    /// * `Some(loss)` if training occurred
    /// * `None` if training was skipped (e.g., during warmup, insufficient samples)
    fn train_step_gpu_native_with_prefetch(
        &mut self,
        steps_since_last_train: usize,
        device: &B::Device,
        prebuilt_batch: Option<crate::buffer::TensorTransitionBatch<B>>,
    ) -> Option<f32> {
        match prebuilt_batch {
            Some(batch) => {
                let loss = self.train_step_gpu(&batch);
                self.increment_step_count();
                self.maybe_update_target();
                self.update_epsilon();
                Some(loss)
            }
            None => self.train_step_gpu_native(steps_since_last_train, device),
        }
    }

    /// Get the device for tensor operations.
    fn device(&self) -> &B::Device;

    /// Get state dimension (needed for buffer → tensor conversion).
    fn state_dim(&self) -> usize;

    /// Get current buffer length (for warmup checking).
    fn buffer_len(&self) -> usize;

    /// Get warmup batch size.
    fn warmup_batch_size(&self) -> usize;

    /// Check if warmup phase is complete.
    fn is_warmup_complete(&self) -> bool;

    /// Mark warmup as complete.
    fn set_warmup_complete(&mut self, complete: bool);

    /// Get current epsilon (exploration rate).
    fn epsilon(&self) -> f32;

    /// Get current training step count.
    fn step_count(&self) -> usize;

    /// Increment step count after training.
    fn increment_step_count(&mut self);

    /// Get full batch size for training.
    fn batch_size(&self) -> usize;

    /// Get full batch size for training (alias for batch_size).
    /// Used by coordinator to determine warmup completion threshold.
    fn full_batch_size(&self) -> usize {
        self.batch_size()
    }

    /// Get target network update frequency.
    fn target_update_freq(&self) -> usize;

    /// Get learning rate.
    fn learning_rate(&self) -> f32;

    /// Get discount factor (gamma).
    fn gamma(&self) -> f32;

    /// Decay exploration parameters (e.g., epsilon).
    /// Called after each training step.
    fn decay_exploration(&mut self);

    /// Update exploration parameter (alias for decay_exploration).
    /// Called after each training step by the coordinator.
    fn update_epsilon(&mut self) {
        self.decay_exploration();
    }

    /// Update target network weights from policy network.
    /// Called periodically based on `target_update_freq()`.
    fn update_target_network(&mut self);

    /// Update target network if step count is a multiple of target_update_freq.
    /// Convenience method that combines step_count() and update_target_network().
    fn maybe_update_target(&mut self) {
        if self.step_count() % self.target_update_freq() == 0 {
            self.update_target_network();
        }
    }

    /// Save model checkpoint to path.
    ///
    /// # Arguments
    /// * `path` - Path to save the checkpoint
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(...)` on failure
    fn save_checkpoint(&self, path: &str) -> Result<(), Box<dyn std::error::Error>>;

    /// Load model checkpoint from path.
    ///
    /// # Arguments
    /// * `path` - Path to load the checkpoint from
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err(...)` on failure
    fn load_checkpoint(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>>;
}

/// Helper methods for GpuTrainable (default implementations)
pub trait GpuTrainableExt<B: AutodiffBackend, Buf>: GpuTrainable<B, Buf> {
    /// Get effective batch size based on warmup state.
    ///
    /// During warmup: uses warmup_batch_size
    /// After warmup: uses full batch_size
    fn effective_batch_size(&mut self) -> usize {
        if self.is_warmup_complete() {
            return self.batch_size();
        }

        let effective = self.warmup_batch_size().min(self.batch_size());

        // Check if we should complete warmup
        let buffer_len: usize = self.buffer_len();
        if buffer_len >= self.batch_size() {
            self.set_warmup_complete(true);
        }

        effective
    }

    /// Get effective batch size with coordinator-provided config override.
    ///
    /// This variant allows the coordinator to override the agent's warmup_batch_size,
    /// fixing the bug where the agent's default was used instead of the coordinator's config.
    ///
    /// # Arguments
    /// * `full_batch_size` - Full batch size from coordinator config
    /// * `warmup_batch_size` - Warmup batch size from coordinator config
    fn effective_batch_size_with_config(
        &mut self,
        full_batch_size: usize,
        warmup_batch_size: usize,
    ) -> usize {
        if self.is_warmup_complete() {
            return full_batch_size;
        }

        let effective = warmup_batch_size.min(full_batch_size);

        // Check if we should complete warmup
        let buffer_len: usize = self.buffer_len();
        if buffer_len >= full_batch_size {
            self.set_warmup_complete(true);
        }

        effective
    }
}

// Blanket implementation
impl<B: AutodiffBackend, Buf, T: GpuTrainable<B, Buf>> GpuTrainableExt<B, Buf> for T {}

/// Trait for agents that can select actions in batches.
///
/// This enables single forward pass for all environments (GPU-efficient).
pub trait BatchedActionSelector<B: AutodiffBackend> {
    /// Select actions for all environments using a single forward pass.
    ///
    /// # Arguments
    /// * `observations` - Batch of observations (one per environment)
    /// * `device` - GPU device for tensor operations
    /// * `action_dim` - Number of possible actions
    /// * `epsilon` - Exploration rate (for epsilon-greedy)
    ///
    /// # Returns
    /// Vector of selected actions (one per environment)
    fn select_actions_batched(
        &self,
        observations: &[Vec<f64>],
        device: &B::Device,
        action_dim: usize,
        epsilon: f32,
    ) -> Vec<usize>;
}

/// Trait for vectorized environments (multiple environments in parallel).
///
/// This trait abstracts the VecEnv pattern from Metis.
pub trait VecEnvironment {
    /// Get number of parallel environments.
    fn num_envs(&self) -> usize;

    /// Get action space (number of discrete actions).
    fn action_space(&self) -> &DiscreteSpace;

    /// Get observation dimension.
    fn observation_dim(&self) -> usize;

    /// Reset all environments.
    ///
    /// # Returns
    /// Initial observations for all environments
    fn reset_all(&mut self) -> Result<Vec<Vec<f64>>, Box<dyn Error>>;

    /// Step all environments (sequential).
    ///
    /// # Arguments
    /// * `actions` - Action for each environment
    fn step_all(&mut self, actions: Vec<usize>) -> Result<Vec<StepResult>, Box<dyn Error>>;

    /// Step all environments in parallel (default: falls back to sequential).
    ///
    /// # Arguments
    /// * `actions` - Action for each environment
    fn step_all_parallel(
        &mut self,
        actions: Vec<usize>,
    ) -> Result<Vec<StepResult>, Box<dyn Error>> {
        self.step_all(actions)
    }

    /// Reset environments that are done.
    ///
    /// # Arguments
    /// * `results` - Step results indicating which envs are done
    ///
    /// # Returns
    /// Vector of Option<Vec<f64>> where Some(obs) means the environment was reset
    /// and None means it wasn't. Length equals num_envs, with each index corresponding
    /// to the environment at that index.
    fn reset_done_environments(
        &mut self,
        results: &[StepResult],
    ) -> Result<Vec<Option<Vec<f64>>>, Box<dyn Error>>;

    /// Get current observations after stepping.
    ///
    /// Combines step results and reset observations.
    fn get_current_observations(
        &self,
        results: &[StepResult],
        reset_obs: &[Option<Vec<f64>>],
    ) -> Result<Vec<Vec<f64>>, Box<dyn Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::CpuRingBuffer;
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::tensor::backend::AutodiffBackend;

    // Type alias for test backend
    type TestBackend = Autodiff<NdArray>;

    // Mock implementations for testing
    struct MockAgent<B: AutodiffBackend> {
        warmup_complete: bool,
        buffer: CpuRingBuffer,
        state_dim: usize,
        device: B::Device,
        _phantom: std::marker::PhantomData<B>,
    }

    impl<B: AutodiffBackend> MockAgent<B> {
        fn new() -> Self {
            Self {
                warmup_complete: false,
                buffer: CpuRingBuffer::new(1000),
                state_dim: 4,
                device: Default::default(),
                _phantom: std::marker::PhantomData,
            }
        }
    }

    impl<B: AutodiffBackend> GpuTrainable<B, CpuRingBuffer> for MockAgent<B> {
        fn buffer_mut(&mut self) -> &mut CpuRingBuffer {
            &mut self.buffer
        }

        fn buffer(&self) -> &CpuRingBuffer {
            &self.buffer
        }

        fn train_step_gpu_native(&mut self, _: usize, _device: &B::Device) -> Option<f32> {
            None
        }

        fn train_step_gpu(&mut self, _batch: &crate::buffer::TensorTransitionBatch<B>) -> f32 {
            0.0
        }

        fn warmup_batch_size(&self) -> usize {
            256
        }

        fn is_warmup_complete(&self) -> bool {
            self.warmup_complete
        }

        fn set_warmup_complete(&mut self, complete: bool) {
            self.warmup_complete = complete;
        }

        fn epsilon(&self) -> f32 {
            1.0
        }

        fn step_count(&self) -> usize {
            0
        }

        fn increment_step_count(&mut self) {}

        fn batch_size(&self) -> usize {
            512
        }

        fn full_batch_size(&self) -> usize {
            512
        }

        fn target_update_freq(&self) -> usize {
            1000
        }

        fn learning_rate(&self) -> f32 {
            0.001
        }

        fn gamma(&self) -> f32 {
            0.99
        }

        fn decay_exploration(&mut self) {
            // No-op for mock
        }

        fn update_epsilon(&mut self) {
            // No-op for mock
        }

        fn update_target_network(&mut self) {
            // No-op for mock
        }

        fn maybe_update_target(&mut self) {
            // No-op for mock
        }

        fn device(&self) -> &B::Device {
            &self.device
        }

        fn state_dim(&self) -> usize {
            self.state_dim
        }

        fn buffer_len(&self) -> usize {
            self.buffer.len()
        }

        fn save_checkpoint(&self, _path: &str) -> Result<(), Box<dyn std::error::Error>> {
            // Mock implementation: just succeed
            Ok(())
        }

        fn load_checkpoint(&mut self, _path: &str) -> Result<(), Box<dyn std::error::Error>> {
            // Mock implementation: just succeed
            Ok(())
        }
    }

    #[test]
    fn test_gpu_trainable_ext_warmup() {
        let mut agent = MockAgent::<TestBackend>::new();
        // Test effective_batch_size during warmup
        assert_eq!(agent.effective_batch_size(), 256); // Warmup batch size
    }

    #[test]
    fn test_gpu_trainable_ext_post_warmup() {
        let mut agent = MockAgent::<TestBackend>::new();
        agent.set_warmup_complete(true);
        // Test effective_batch_size after warmup
        assert_eq!(agent.effective_batch_size(), 512); // Full batch size
    }

    #[test]
    fn test_mock_agent_properties() {
        let agent = MockAgent::<TestBackend>::new();
        assert_eq!(agent.warmup_batch_size(), 256);
        assert_eq!(agent.batch_size(), 512);
        assert_eq!(agent.target_update_freq(), 1000);
        assert!((agent.learning_rate() - 0.001).abs() < f32::EPSILON);
        assert!((agent.gamma() - 0.99).abs() < f32::EPSILON);
    }

    #[test]
    fn test_effective_batch_size_with_config_warmup() {
        let mut agent = MockAgent::<TestBackend>::new();
        // During warmup: should use warmup_batch_size from config
        let batch_size = agent.effective_batch_size_with_config(512, 128);
        assert_eq!(batch_size, 128);
    }

    #[test]
    fn test_effective_batch_size_with_config_post_warmup() {
        let mut agent = MockAgent::<TestBackend>::new();
        agent.set_warmup_complete(true);
        // After warmup: should use full_batch_size from config
        let batch_size = agent.effective_batch_size_with_config(512, 128);
        assert_eq!(batch_size, 512);
    }
}
