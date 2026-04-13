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

use crate::training::tensor_buffer::TensorTransitionBatch;
use burn::tensor::backend::AutodiffBackend;

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
        // Check training frequency (every 4 steps after warmup)
        let should_train = if self.is_warmup_complete() {
            steps_since_last_train >= 4
        } else {
            true // Train every step during warmup
        };

        if !should_train {
            return None;
        }

        // Get effective batch size (handles warmup logic)
        let batch_size = self.effective_batch_size(self.full_batch_size());

        // GPU DIAGNOSTIC: Time the entire train_step_gpu_native call
        static FIRST_CALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        let is_first = FIRST_CALL.load(std::sync::atomic::Ordering::Relaxed);
        let step_start = std::time::Instant::now();

        // Sample from GPU buffer
        let sample_start = std::time::Instant::now();
        let batch = self.gpu_buffer_mut().sample_batch(batch_size, device)?;
        let sample_elapsed = sample_start.elapsed();

        if is_first || self.step_count() <= 3 || self.step_count() % 500 == 0 {
            println!(
                "[STAGE:DIAG] train_step_gpu_native #{}: sample_batch(batch_size={}) took {:?}",
                self.step_count(),
                batch_size,
                sample_elapsed
            );
            if sample_elapsed.as_millis() > 10 {
                println!(
                    "   [STAGE:WARN]  SLOW sample_batch (>10ms) - data transfer may be on CPU"
                );
            }
        }

        // Perform training step
        let train_start = std::time::Instant::now();
        let loss = self.train_step_gpu(&batch);
        let train_elapsed = train_start.elapsed();

        if is_first || self.step_count() <= 3 || self.step_count() % 500 == 0 {
            println!(
                "[STAGE:DIAG] train_step_gpu_native #{}: train_step_gpu took {:?} (loss={:.4})",
                self.step_count(),
                train_elapsed,
                loss
            );
        }

        // Update step count and target network
        self.increment_step_count();
        let step_count = self.step_count();
        self.maybe_update_target(step_count);

        // Decay epsilon
        self.update_epsilon();

        // GPU DIAGNOSTIC: Total step timing
        let total_elapsed = step_start.elapsed();
        if is_first || self.step_count() <= 3 || self.step_count() % 500 == 0 {
            println!(
                "[STAGE:DIAG] train_step_gpu_native #{}: TOTAL took {:?} (sample={:?}, train={:?})",
                self.step_count(),
                total_elapsed,
                sample_elapsed,
                train_elapsed
            );
            // GPU: total step < 50ms for batch_size=2048
            // CPU: total step > 100ms for batch_size=2048
            FIRST_CALL.store(false, std::sync::atomic::Ordering::Relaxed);
        }

        Some(loss)
    }

    /// Get effective batch size (handles warmup logic).
    ///
    /// During warmup, returns the minimum of current buffer size
    /// and configured batch size. After warmup, returns full batch size.
    fn effective_batch_size(&mut self, config_batch_size: usize) -> usize {
        if self.is_warmup_complete() {
            return config_batch_size;
        }

        // Check if we should complete warmup BEFORE calculating effective batch size
        if self.gpu_buffer().len() >= config_batch_size {
            self.set_warmup_complete(true);
            println!(
                "Warmup complete! Using full batch size: {}",
                config_batch_size
            );
            return config_batch_size;
        }

        self.warmup_batch_size().min(config_batch_size)
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

/// Check if training should occur based on warmup state.
///
/// During warmup: trains every step
/// After warmup: trains every N steps (e.g., every 4)
///
/// # Arguments
/// * `warmup_complete` - Whether warmup phase is complete
/// * `steps_since_last_train` - Steps since last training
/// * `train_frequency` - Training frequency after warmup
///
/// # Returns
/// * `true` if training should occur
/// * `false` otherwise
pub fn should_train(
    warmup_complete: bool,
    steps_since_last_train: usize,
    train_frequency: usize,
) -> bool {
    if !warmup_complete {
        // Train every step during warmup
        true
    } else {
        // Train every N steps after warmup
        // steps_since_last_train == 0 means first step after warmup, should train
        steps_since_last_train == 0 || steps_since_last_train >= train_frequency
    }
}

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
}
