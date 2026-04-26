//! Warmup and training frequency helpers
//!
//! This module provides utility functions for managing warmup phases
//! and training frequency in GPU-native reinforcement learning training.

use crate::buffer::BufferOps;
use crate::traits::GpuTrainable;
use burn::tensor::backend::AutodiffBackend;

/// Determine if training should occur based on warmup state.
///
/// During warmup: trains every step
/// After warmup: trains every N steps
///
/// # Arguments
/// * `warmup_complete` - Whether warmup phase is finished
/// * `steps_since_last_train` - Number of steps since last training
/// * `train_frequency` - How often to train after warmup (e.g., every 4 steps)
///
/// # Returns
/// `true` if training should occur, `false` otherwise
///
/// # Example
/// ```
/// use burnme_rly::warmup::should_train;
///
/// // During warmup: train every step
/// assert!(should_train(false, 1, 4));  // Always true during warmup
///
/// // After warmup: train every 4 steps
/// assert!(should_train(true, 4, 4));   // Steps == frequency
/// assert!(!should_train(true, 3, 4)); // Steps < frequency
/// ```
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
        // Also train on first step (steps_since_last_train == 0) after warmup completes
        steps_since_last_train == 0 || steps_since_last_train >= train_frequency
    }
}

/// Execute a training step with automatic warmup handling.
///
/// This is a convenience wrapper that reads the agent's own configuration
/// and delegates to `train_step_with_warmup_config`.
///
/// # Arguments
/// * `agent` - The learning agent implementing GpuTrainable
/// * `steps_since_last_train` - Steps since last training call
/// * `device` - GPU device for tensor operations
///
/// # Returns
/// * `Some(loss)` if training occurred
/// * `None` if training was skipped (insufficient samples, frequency not met)
///
/// # Example
/// ```rust,ignore
/// use burnme_rly::warmup::train_step_with_warmup;
///
/// if let Some(loss) = train_step_with_warmup(&mut agent, steps, &device) {
///     total_loss += loss;
/// }
/// ```
pub fn train_step_with_warmup<B: AutodiffBackend, Buf: BufferOps>(
    agent: &mut impl GpuTrainable<B, Buf>,
    steps_since_last_train: usize,
    device: &B::Device,
) -> Option<f32> {
    let warmup_batch_size = agent.warmup_batch_size();
    let full_batch_size = agent.full_batch_size();
    train_step_with_warmup_config(
        agent,
        full_batch_size,
        warmup_batch_size,
        steps_since_last_train,
        device,
    )
}

/// Execute a training step with automatic warmup handling and configurable batch sizes.
///
/// This variant allows the coordinator to override the agent's warmup_batch_size,
/// fixing the bug where hardcoded values were used instead of the configured value.
///
/// # Arguments
/// * `agent` - The learning agent implementing GpuTrainable
/// * `full_batch_size` - Full batch size from coordinator config
/// * `warmup_batch_size` - Warmup batch size from coordinator config (overrides agent's default)
/// * `steps_since_last_train` - Steps since last training call
/// * `device` - GPU device for tensor operations
///
/// # Returns
/// * `Some(loss)` if training occurred
/// * `None` if training was skipped (insufficient samples, frequency not met)
pub fn train_step_with_warmup_config<B: AutodiffBackend, Buf: BufferOps>(
    agent: &mut impl GpuTrainable<B, Buf>,
    full_batch_size: usize,
    warmup_batch_size: usize,
    steps_since_last_train: usize,
    device: &B::Device,
) -> Option<f32> {
    // Determine effective batch size using coordinator-provided config
    let batch_size = if agent.is_warmup_complete() {
        full_batch_size
    } else {
        let effective = warmup_batch_size.min(full_batch_size);
        // Check if we should complete warmup
        let buffer_len: usize = agent.buffer().len();
        if buffer_len >= full_batch_size {
            agent.set_warmup_complete(true);
        }
        effective
    };

    // Check if buffer has enough samples
    if !agent.buffer().can_sample(batch_size) {
        return None;
    }

    // Training step handles its own sampling internally
    agent.train_step_gpu_native(steps_since_last_train, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_train_during_warmup() {
        // During warmup: always train
        assert!(should_train(false, 0, 4));
        assert!(should_train(false, 1, 4));
        assert!(should_train(false, 100, 4));
    }

    #[test]
    fn test_should_train_after_warmup() {
        // After warmup: train based on frequency
        assert!(should_train(true, 0, 4)); // First step after warmup should train
        assert!(!should_train(true, 3, 4)); // Not yet
        assert!(should_train(true, 4, 4)); // Time to train
        assert!(should_train(true, 5, 4)); // Past due
    }

    #[test]
    fn test_should_train_frequency_1() {
        // Train every step
        assert!(should_train(true, 1, 1));
        assert!(should_train(true, 2, 1));
    }

    #[test]
    fn test_should_train_edge_cases() {
        // Zero frequency (should always train after warmup)
        assert!(should_train(true, 0, 0));
        assert!(should_train(true, 1, 0));

        // Large frequency
        assert!(!should_train(true, 99, 100));
        assert!(should_train(true, 100, 100));
    }

    // Note: train_step_with_warmup integration tests require full buffer setup
    // These are better tested in integration tests with actual agent implementations
}
