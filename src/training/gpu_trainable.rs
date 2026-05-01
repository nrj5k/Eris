//! GPU-native training trait for unified training interface.
//!
//! This trait abstracts GPU-native training operations that work
//! across different policy types (DQN, Bandit, Catcher, etc.).
//!
//! ## Purpose
//!
//! Instead of copy-pasting GPU training logic into each policy,
//! this trait provides a unified interface for:
//! - Pushing transitions to GPU buffer
//! - Sampling batches for training
//! - Performing GPU-native training steps
//! - Managing warmup and async loss reporting
//!
//! ## Future Work
//!
//! Extract episode loop to make this fully generic (Option 3 unification).

use burn::tensor::backend::AutodiffBackend;
use burnme_rly::buffer::TensorTransitionBatch;
use burnme_rly::warmup;
use tracing;

/// Trait for policies that support GPU-native training with TensorRingBuffer.
///
/// This trait provides a unified interface for GPU training operations
/// that are common across DQN, Bandit, Catcher, and other policies.
///
/// # Example
///
/// ```rust,ignore
/// impl<B: AutodiffBackend> GpuTrainable<B> for DQNPolicy<B> {
///     fn gpu_buffer_mut(&mut self) -> &mut TensorRingBuffer<B> {
///         &mut self.gpu_buffer
///     }
///
///     fn train_step_gpu(&mut self, batch: &TensorTransitionBatch<B>) -> f32 {
///         // DQN-specific training logic
///     }
///
///     // ... other methods
/// }
/// ```
pub trait GpuTrainable<B: AutodiffBackend> {
    /// Get mutable reference to the GPU replay buffer.
    fn gpu_buffer_mut(&mut self) -> &mut crate::training::HybridRingBuffer<B>;

    /// Get immutable reference to the GPU replay buffer.
    fn gpu_buffer(&self) -> &crate::training::HybridRingBuffer<B>;

    /// Get warmup batch size.
    fn warmup_batch_size(&self) -> usize;

    /// Get the full batch size for training.
    /// This is used to determine when warmup is complete.
    fn full_batch_size(&self) -> usize;

    /// Check if warmup is complete.
    fn is_warmup_complete(&self) -> bool;

    /// Mark warmup as complete.
    fn set_warmup_complete(&mut self, complete: bool);

    /// Get the target network update frequency.
    fn target_update_freq(&self) -> usize;

    /// Get current step count.
    fn step_count(&self) -> usize;

    /// Increment step count.
    fn increment_step_count(&mut self);

    /// Get current epsilon value.
    fn epsilon(&self) -> f32;

    /// Update epsilon after step.
    fn update_epsilon(&mut self);

    /// Perform a GPU-native training step.
    ///
    /// # Arguments
    /// * `batch` - GPU tensor batch from TensorRingBuffer
    ///
    /// # Returns
    /// Loss value from training step
    fn train_step_gpu(&mut self, batch: &TensorTransitionBatch<B>) -> f32;

    /// Update target network if needed.
    fn maybe_update_target(&mut self, step_count: usize);

    /// Perform a self-contained GPU-native training step.
    ///
    /// Unlike `train_step_gpu()` which takes a pre-sampled batch,
    /// this method handles warmup, sampling, and training internally.
    ///
    /// # Arguments
    /// * `steps_since_last_train` - Steps since last training (used for training frequency)
    /// * `device` - GPU device for tensor operations
    ///
    /// # Returns
    /// * `Some(loss)` if training occurred
    /// * `None` if skipped (not time to train yet)
    fn train_step_gpu_native(
        &mut self,
        steps_since_last_train: usize,
        device: &B::Device,
    ) -> Option<f32> {
        // Use agent's warmup_batch_size for backward compatibility
        self.train_step_gpu_native_with_config(
            steps_since_last_train,
            device,
            self.warmup_batch_size(),
            self.full_batch_size(),
        )
    }

    /// Perform a self-contained GPU-native training step with an optional pre-built batch.
    ///
    /// When `prebuilt_batch` is `Some(batch)`, uses that batch directly (skips internal `sample_batch()`).
    /// When `prebuilt_batch` is `None`, falls back to calling `sample_batch()` internally.
    ///
    /// This enables double-buffering: while the GPU trains on batch N, the CPU can
    /// prepare batch N+1 via `PrefetchBuffer`, then pass it here.
    ///
    /// # Arguments
    /// * `steps_since_last_train` - Steps since last training (used for training frequency)
    /// * `device` - GPU device for tensor operations
    /// * `warmup_batch_size` - Batch size during warmup (from coordinator)
    /// * `full_batch_size` - Full batch size after warmup (from coordinator)
    /// * `prebuilt_batch` - Optional pre-built batch from prefetch buffer
    ///
    /// # Returns
    /// * `Some(loss)` if training occurred
    /// * `None` if skipped (not time to train yet)
    fn train_step_gpu_native_with_prefetch(
        &mut self,
        steps_since_last_train: usize,
        device: &B::Device,
        warmup_batch_size: usize,
        full_batch_size: usize,
        prebuilt_batch: Option<TensorTransitionBatch<B>>,
    ) -> Option<f32> {
        // CRITICAL DEBUG: Entry point logging
        tracing::debug!(
            "train_step_gpu_native_with_prefetch ENTRY, step_count: {}, steps_since_last_train: {}, buffer_len: {}, warmup_batch_size: {}, full_batch_size: {}, has_prefetch: {}",
            self.step_count(),
            steps_since_last_train,
            self.gpu_buffer().len(),
            warmup_batch_size,
            full_batch_size,
            prebuilt_batch.is_some()
        );

        // Check training frequency using lib's canonical should_train
        let should_train = warmup::should_train(
            self.is_warmup_complete(),
            steps_since_last_train,
            4, // train_frequency
        );
        tracing::debug!(
            "should_train: {} (warmup_complete={}, steps_since_last_train={})",
            should_train,
            self.is_warmup_complete(),
            steps_since_last_train
        );

        if !should_train {
            tracing::debug!("train_step_gpu_native_with_prefetch: SKIPPING (should_train=false)");
            return None;
        }

        // Get the batch either from the prefetch or by sampling internally
        let batch = match prebuilt_batch {
            Some(batch) => {
                tracing::debug!(
                    "Using pre-built batch, batch.states.shape: {:?}",
                    batch.states.shape()
                );
                batch
            }
            None => {
                // Fall back to internal sampling
                let effective_batch_size = if self.is_warmup_complete() {
                    full_batch_size
                } else {
                    let effective = warmup_batch_size.min(full_batch_size);
                    let buffer_len: usize = self.gpu_buffer().len();
                    if buffer_len >= full_batch_size {
                        self.set_warmup_complete(true);
                    }
                    effective
                };
                tracing::debug!(
                    "Calling sample_batch(batch_size={}), buffer_len: {}",
                    effective_batch_size,
                    self.gpu_buffer().len()
                );
                match self
                    .gpu_buffer_mut()
                    .sample_batch(effective_batch_size, device)
                {
                    Some(batch) => {
                        tracing::debug!(
                            "sample_batch SUCCESS, batch.states.shape: {:?}",
                            batch.states.shape()
                        );
                        batch
                    }
                    None => {
                        tracing::debug!(
                            "sample_batch returned None! buffer_len: {}, batch_size requested: {}",
                            self.gpu_buffer().len(),
                            effective_batch_size
                        );
                        return None;
                    }
                }
            }
        };

        // GPU DIAGNOSTIC: Time the entire train_step_gpu_native call
        static FIRST_CALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        let is_first = FIRST_CALL.load(std::sync::atomic::Ordering::Relaxed);
        let step_start = std::time::Instant::now();

        // Perform training step
        let train_start = std::time::Instant::now();
        tracing::debug!(
            "Calling train_step_gpu, batch.states.shape: {:?}",
            batch.states.shape()
        );
        let loss = self.train_step_gpu(&batch);
        let train_elapsed = train_start.elapsed();
        tracing::debug!("train_step_gpu returned loss: {:.4}", loss);

        if is_first || self.step_count() <= 3 || self.step_count() % 500 == 0 {
            tracing::debug!(
                "[STAGE:DIAG] train_step_gpu_native_with_prefetch #{}: train_step_gpu took {:?} (loss={:.4})",
                self.step_count(),
                train_elapsed,
                loss
            );
        }

        // Update step count and target network
        self.increment_step_count();
        let step_count = self.step_count();
        tracing::debug!(
            "increment_step_count called, new step_count: {}",
            step_count
        );
        self.maybe_update_target(step_count);

        // Decay epsilon
        self.update_epsilon();
        tracing::debug!("update_epsilon called, new epsilon: {:.4}", self.epsilon());

        // GPU DIAGNOSTIC: Total step timing
        let total_elapsed = step_start.elapsed();
        if is_first || self.step_count() <= 3 || self.step_count() % 500 == 0 {
            tracing::debug!(
                "[STAGE:DIAG] train_step_gpu_native_with_prefetch #{}: TOTAL took {:?}",
                self.step_count(),
                total_elapsed
            );
            FIRST_CALL.store(false, std::sync::atomic::Ordering::Relaxed);
        }

        tracing::debug!(
            "train_step_gpu_native_with_prefetch EXIT, returning Some({:.4})",
            loss
        );
        Some(loss)
    }

    /// Perform a self-contained GPU-native training step with configurable batch sizes.
    ///
    /// This variant allows the coordinator to override batch sizes, fixing the issue
    /// where the agent's hardcoded warmup_batch_size (256) was used instead of the
    /// coordinator's configured value (e.g., 1024).
    ///
    /// Unlike `train_step_gpu()` which takes a pre-sampled batch,
    /// this method handles warmup, sampling, and training internally.
    ///
    /// # Arguments
    /// * `steps_since_last_train` - Steps since last training (used for training frequency)
    /// * `device` - GPU device for tensor operations
    /// * `warmup_batch_size` - Batch size during warmup (from coordinator)
    /// * `full_batch_size` - Full batch size after warmup (from coordinator)
    ///
    /// # Returns
    /// * `Some(loss)` if training occurred
    /// * `None` if skipped (not time to train yet)
    fn train_step_gpu_native_with_config(
        &mut self,
        steps_since_last_train: usize,
        device: &B::Device,
        warmup_batch_size: usize,
        full_batch_size: usize,
    ) -> Option<f32> {
        self.train_step_gpu_native_with_prefetch(
            steps_since_last_train,
            device,
            warmup_batch_size,
            full_batch_size,
            None,
        )
    }

    /// Get effective batch size (handles warmup logic).
    ///
    /// During warmup, returns the minimum of current buffer size
    /// and configured batch size. After warmup, returns full batch size.
    fn effective_batch_size(&mut self, config_batch_size: usize) -> usize {
        // Use agent's warmup_batch_size for backward compatibility
        self.effective_batch_size_with_config(config_batch_size, self.warmup_batch_size())
    }

    /// Get effective batch size with external warmup parameter.
    ///
    /// This allows the coordinator to override the agent's warmup_batch_size.
    ///
    /// During warmup, returns the minimum of current buffer size
    /// and configured batch size. After warmup, returns full batch size.
    ///
    /// # Arguments
    ///
    /// * `config_batch_size` - Full batch size from configuration
    /// * `warmup_batch_size` - Warmup batch size from coordinator (overrides agent's)
    ///
    /// # Returns
    ///
    /// Effective batch size to use for current training step
    fn effective_batch_size_with_config(
        &mut self,
        config_batch_size: usize,
        warmup_batch_size: usize,
    ) -> usize {
        if self.is_warmup_complete() {
            return config_batch_size;
        }

        // Check if we should complete warmup BEFORE calculating effective batch size
        if self.gpu_buffer().len() >= config_batch_size {
            self.set_warmup_complete(true);
            tracing::info!(
                "[STAGE:WARMUP] Warmup complete! Using full batch size: {}",
                config_batch_size
            );
            return config_batch_size;
        }

        // Use coordinator's warmup_batch_size, not agent's hardcoded value!
        warmup_batch_size.min(config_batch_size)
    }
}

/// Helper function for GPU-native training with warmup.
///
/// This can be called from any training loop that implements GpuTrainable.
/// It handles:
/// - Warmup batch sizing
/// - Buffer sampling
/// - Training step execution
/// - Target network updates
/// - Epsilon decay
///
/// # Arguments
/// * `agent` - Policy implementing GpuTrainable
/// * `config_batch_size` - Full batch size from config
/// * `device` - GPU device
///
/// # Returns
/// * `Some(loss)` if training occurred
/// * `None` if buffer insufficient or skipped
pub fn train_step_with_warmup<B: AutodiffBackend, T: GpuTrainable<B>>(
    agent: &mut T,
    config_batch_size: usize,
    device: &B::Device,
) -> Option<f32> {
    let batch_size = agent.effective_batch_size(config_batch_size);

    // Sample from GPU buffer
    let batch = agent.gpu_buffer_mut().sample_batch(batch_size, device)?;

    // Perform training step
    let loss = agent.train_step_gpu(&batch);

    // Update step count and target network
    agent.increment_step_count();
    let step_count = agent.step_count();
    agent.maybe_update_target(step_count);

    // Decay epsilon
    agent.update_epsilon();

    Some(loss)
}

/// Helper function for GPU-native training with coordinator-configured warmup.
///
/// This variant allows the coordinator to override the agent's warmup_batch_size,
/// fixing the bug where hardcoded 256 was used instead of the configured value.
///
/// # Arguments
/// * `agent` - Policy implementing GpuTrainable
/// * `full_batch_size` - Full batch size from coordinator config
/// * `warmup_batch_size` - Warmup batch size from coordinator config
/// * `device` - GPU device
///
/// # Returns
/// * `Some(loss)` if training occurred
/// * `None` if buffer insufficient or skipped
pub fn train_step_with_warmup_config<B: AutodiffBackend, T: GpuTrainable<B>>(
    agent: &mut T,
    full_batch_size: usize,
    warmup_batch_size: usize,
    device: &B::Device,
) -> Option<f32> {
    let batch_size = agent.effective_batch_size_with_config(full_batch_size, warmup_batch_size);

    // Sample from GPU buffer
    let batch = agent.gpu_buffer_mut().sample_batch(batch_size, device)?;

    // Perform training step
    let loss = agent.train_step_gpu(&batch);

    // Update step count and target network
    agent.increment_step_count();
    let step_count = agent.step_count();
    agent.maybe_update_target(step_count);

    // Decay epsilon
    agent.update_epsilon();

    Some(loss)
}

/// Check if training should occur based on warmup state.
/// Delegates to burnme-rly's canonical implementation.
pub use burnme_rly::warmup::should_train;

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::backend::Backend;

    type TestBackend = Autodiff<NdArray<f32>>;

    // Mock implementation for testing
    struct MockPolicy {
        buffer: crate::training::HybridRingBuffer<TestBackend>,
        warmup_batch_size: usize,
        full_batch_size: usize,
        warmup_complete: bool,
        target_update_freq: usize,
        step_count: usize,
        epsilon: f32,
        state_dim: usize,
    }

    impl MockPolicy {
        fn new(capacity: usize, state_dim: usize) -> Self {
            Self {
                buffer: crate::training::HybridRingBuffer::new(capacity, state_dim),
                warmup_batch_size: 256,
                full_batch_size: 512,
                warmup_complete: false,
                target_update_freq: 100,
                step_count: 0,
                epsilon: 1.0,
                state_dim,
            }
        }

        fn full_batch_size(&self) -> usize {
            self.full_batch_size
        }
    }

    impl GpuTrainable<TestBackend> for MockPolicy {
        fn gpu_buffer_mut(&mut self) -> &mut crate::training::HybridRingBuffer<TestBackend> {
            &mut self.buffer
        }

        fn gpu_buffer(&self) -> &crate::training::HybridRingBuffer<TestBackend> {
            &self.buffer
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
            self.target_update_freq
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

        fn update_epsilon(&mut self) {
            self.epsilon = (self.epsilon * 0.995).max(0.01);
        }

        fn train_step_gpu(&mut self, _batch: &TensorTransitionBatch<TestBackend>) -> f32 {
            // Mock loss
            0.5
        }

        fn maybe_update_target(&mut self, _step_count: usize) {
            // Mock target update
        }
    }

    #[test]
    fn test_should_train_during_warmup() {
        // During warmup, should train every step
        assert!(should_train(false, 0, 4));
        assert!(should_train(false, 1, 4));
        assert!(should_train(false, 100, 4));
    }

    #[test]
    fn test_should_train_after_warmup() {
        // After warmup, should train every N steps
        assert!(should_train(true, 0, 4)); // First step
        assert!(!should_train(true, 1, 4));
        assert!(!should_train(true, 2, 4));
        assert!(!should_train(true, 3, 4));
        assert!(should_train(true, 4, 4)); // Train again
        assert!(should_train(true, 5, 4)); // Train again (>= threshold)
    }

    #[test]
    fn test_effective_batch_size_warmup() {
        let device = <NdArray as Backend>::Device::default();
        let mut policy = MockPolicy::new(1000, 10);
        let full_batch_size = policy.full_batch_size();
        let warmup_batch_size = policy.warmup_batch_size();

        // Initially not warmed up
        assert!(!policy.is_warmup_complete());

        // During warmup, effective batch size is min of warmup_batch_size and config
        assert_eq!(
            policy.effective_batch_size(full_batch_size),
            warmup_batch_size
        );
        assert_eq!(policy.effective_batch_size(128), 128);

        // Add warmup_batch_size samples - should NOT complete warmup
        for i in 0..warmup_batch_size {
            policy.gpu_buffer_mut().push(
                vec![i as f32; 10],
                0,
                1.0,
                vec![(i + 1) as f32; 10],
                false,
            );
        }

        // Warmup should NOT complete at warmup_batch_size (256)
        assert!(
            !policy.is_warmup_complete(),
            "Warmup should NOT complete at warmup_batch_size ({})",
            warmup_batch_size
        );

        // Effective batch size should still be warmup_batch_size
        let batch_size = policy.effective_batch_size(full_batch_size);
        assert_eq!(batch_size, warmup_batch_size);

        // Add remaining samples to reach full_batch_size (512)
        for i in warmup_batch_size..full_batch_size {
            policy.gpu_buffer_mut().push(
                vec![i as f32; 10],
                0,
                1.0,
                vec![(i + 1) as f32; 10],
                false,
            );
        }

        // NOW warmup should complete (buffer has >= full_batch_size samples)
        let batch_size = policy.effective_batch_size(full_batch_size);
        assert!(
            policy.is_warmup_complete(),
            "Warmup should complete at full_batch_size ({})",
            full_batch_size
        );
        assert_eq!(batch_size, full_batch_size);
    }

    #[test]
    fn test_train_step_with_warmup() {
        let mut policy = MockPolicy::new(1000, 10);
        let full_batch_size = policy.full_batch_size();

        // Fill buffer with enough samples to complete warmup (512 samples)
        for i in 0..full_batch_size {
            policy.gpu_buffer_mut().push(
                vec![i as f32; 10],
                0,
                1.0,
                vec![(i + 1) as f32; 10],
                false,
            );
        }

        // Training should work with full_batch_size
        let device = <NdArray as Backend>::Device::default();
        let loss = train_step_with_warmup(&mut policy, full_batch_size, &device);
        assert!(loss.is_some());
        assert_eq!(loss.unwrap(), 0.5);
        assert_eq!(policy.step_count(), 1);
        // After training with full batch, warmup should be complete
        assert!(policy.is_warmup_complete());
    }

    #[test]
    fn test_effective_batch_size_with_config_override() {
        // Test that coordinator's warmup_batch_size overrides agent's hardcoded value
        let mut policy = MockPolicy::new(2000, 10);
        let full_batch_size = 1024;
        let coordinator_warmup_batch_size = 1024; // Coordinator wants 1024
        let agent_warmup_batch_size = policy.warmup_batch_size(); // Agent has 256

        // Verify agent's default warmup size is 256
        assert_eq!(agent_warmup_batch_size, 256);

        // Initially not warmed up
        assert!(!policy.is_warmup_complete());

        // With old method, would use agent's 256
        let old_batch_size = policy.effective_batch_size(full_batch_size);
        assert_eq!(old_batch_size, agent_warmup_batch_size);

        // With new method, uses coordinator's 1024
        let new_batch_size =
            policy.effective_batch_size_with_config(full_batch_size, coordinator_warmup_batch_size);
        assert_eq!(new_batch_size, coordinator_warmup_batch_size);

        // Verify coordinator's override is larger than agent's default
        assert!(new_batch_size > old_batch_size);
    }

    #[test]
    fn test_train_step_gpu_native_with_config() {
        // Test that train_step_gpu_native_with_config uses coordinator's batch sizes
        let mut policy = MockPolicy::new(2000, 10);
        let full_batch_size = 1024;
        let coordinator_warmup_batch_size = 1024;

        // Fill buffer with coordinator's warmup_batch_size samples
        for i in 0..coordinator_warmup_batch_size {
            policy.gpu_buffer_mut().push(
                vec![i as f32; 10],
                0,
                1.0,
                vec![(i + 1) as f32; 10],
                false,
            );
        }

        let device = <NdArray as Backend>::Device::default();

        // Call the new method with coordinator's batch sizes
        let loss = policy.train_step_gpu_native_with_config(
            0, // steps_since_last_train
            &device,
            coordinator_warmup_batch_size,
            full_batch_size,
        );

        // Should train successfully
        assert!(loss.is_some());
        assert_eq!(loss.unwrap(), 0.5);

        // Should have trained with coordinator's batch size (1024), not agent's (256)
        assert_eq!(policy.step_count(), 1);
    }

    #[test]
    fn test_train_step_with_warmup_config() {
        // Test the helper function with coordinator's batch sizes
        let mut policy = MockPolicy::new(2000, 10);
        let full_batch_size = 1024;
        let coordinator_warmup_batch_size = 1024;

        // Fill buffer with coordinator's warmup_batch_size samples
        for i in 0..coordinator_warmup_batch_size {
            policy.gpu_buffer_mut().push(
                vec![i as f32; 10],
                0,
                1.0,
                vec![(i + 1) as f32; 10],
                false,
            );
        }

        let device = <NdArray as Backend>::Device::default();

        // Use the new helper function
        let loss = train_step_with_warmup_config(
            &mut policy,
            full_batch_size,
            coordinator_warmup_batch_size,
            &device,
        );

        assert!(loss.is_some());
        assert_eq!(loss.unwrap(), 0.5);
        assert_eq!(policy.step_count(), 1);
    }
}
