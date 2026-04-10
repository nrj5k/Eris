//! Batch size auto-tuner for GPU memory optimization
//!
//! This module provides automatic batch size tuning based on available GPU memory
//! and model configuration to maximize GPU utilization while avoiding OOM errors.
//!
//! # Example
//! ```
//! use eris::config::batch_tuner::BatchTuner;
//!
//! let tuner = BatchTuner::new(32); // state_dim = 32
//! let gpu_memory_mb = 8192; // 8GB GPU
//! let optimal_batch_size = tuner.tune(gpu_memory_mb);
//! println!("Optimal batch size: {}", optimal_batch_size);
//! ```

/// Auto-tune batch size based on available GPU memory
///
/// The tuner calculates an optimal batch size that:
/// - Maximizes GPU utilization (larger batches = better throughput)
/// - Fits within GPU memory constraints
/// - Is a multiple of 32 for GPU warp alignment
/// - Stays within reasonable bounds for training stability
pub struct BatchTuner {
    /// Minimum batch size for training stability
    pub min_batch_size: usize,
    /// Maximum batch size to prevent OOM
    pub max_batch_size: usize,
    /// State dimension for memory estimation
    pub state_dim: usize,
    /// Action dimension (affects Q-value tensor size)
    pub action_dim: usize,
    /// Hidden layer sizes for memory estimation
    pub hidden_layers: Vec<usize>,
}

impl BatchTuner {
    /// Create a new batch tuner with default bounds
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of the state vector
    pub fn new(state_dim: usize) -> Self {
        Self {
            min_batch_size: 256,
            max_batch_size: 8192,
            state_dim,
            action_dim: 10,                // Default assumption
            hidden_layers: vec![128, 128], // Default assumption
        }
    }

    /// Create a new batch tuner with custom bounds
    ///
    /// # Arguments
    /// * `state_dim` - Dimension of the state vector
    /// * `min_batch_size` - Minimum batch size (must be multiple of 32)
    /// * `max_batch_size` - Maximum batch size (must be multiple of 32)
    pub fn with_bounds(state_dim: usize, min_batch_size: usize, max_batch_size: usize) -> Self {
        // Validate bounds are multiples of 32
        let min_batch_size = (min_batch_size / 32) * 32;
        let max_batch_size = (max_batch_size / 32) * 32;

        Self {
            min_batch_size: min_batch_size.max(32),
            max_batch_size: max_batch_size.min(16384),
            state_dim,
            action_dim: 10,
            hidden_layers: vec![128, 128],
        }
    }

    /// Set action dimension for more accurate memory estimation
    pub fn with_action_dim(mut self, action_dim: usize) -> Self {
        self.action_dim = action_dim;
        self
    }

    /// Set hidden layers for more accurate memory estimation
    pub fn with_hidden_layers(mut self, hidden_layers: Vec<usize>) -> Self {
        self.hidden_layers = hidden_layers;
        self
    }

    /// Calculate optimal batch size based on GPU memory
    ///
    /// # Arguments
    /// * `gpu_memory_mb` - Available GPU memory in megabytes
    ///
    /// # Returns
    /// Optimal batch size (multiple of 32)
    ///
    /// # Memory Estimation
    ///
    /// Memory per sample includes:
    /// - State tensor: state_dim * 4 bytes (f32)
    /// - Next state tensor: state_dim * 4 bytes (f32)
    /// - Action tensor: 8 bytes (i64)
    /// - Reward tensor: 4 bytes (f32)
    /// - Done tensor: 4 bytes (f32)
    /// - Gradient buffers: ~2x forward memory
    /// - Optimizer state: ~3x for Adam (momentum + variance)
    /// - Safety headroom: 25% reserved
    ///
    /// Total: ~300-500 bytes per sample depending on model size
    pub fn tune(&self, gpu_memory_mb: usize) -> usize {
        // Estimate memory per sample
        // State (32 f32) + next_state (32 f32) + action (1 i64) + reward (1 f32) + done (1 f32)
        let bytes_per_sample_data = self.state_dim * 2 * 4 + 8 + 4 + 4;

        // Estimate model parameters
        let hidden_params: usize = self
            .hidden_layers
            .iter()
            .zip(self.hidden_layers.iter().skip(1))
            .map(|(&a, &b)| a * b)
            .sum();
        let input_to_hidden = self.state_dim * self.hidden_layers.first().copied().unwrap_or(128);
        let hidden_to_output = self.hidden_layers.last().copied().unwrap_or(128) * self.action_dim;
        let total_params = input_to_hidden + hidden_params + hidden_to_output;

        // Model memory per sample (activations + gradients)
        // Rough estimate: 4 bytes per param * 2 (forward + backward)
        let bytes_per_sample_model = (total_params * 8) / self.min_batch_size;

        // Total bytes per sample
        let bytes_per_sample = bytes_per_sample_data + bytes_per_sample_model;

        // Use 75% of GPU memory (leave 25% headroom for kernel overhead, OS, etc.)
        let usable_memory_bytes = (gpu_memory_mb * 1024 * 1024 * 3) / 4;

        // Calculate max samples that fit
        let max_samples = usable_memory_bytes / bytes_per_sample;

        // Round down to nearest multiple of 32 (GPU warp size)
        let batch_size = (max_samples / 32) * 32;

        // Clamp to reasonable range
        batch_size.clamp(self.min_batch_size, self.max_batch_size)
    }

    /// Calculate batch size for a specific target memory usage
    ///
    /// # Arguments
    /// * `target_memory_mb` - Target memory usage in megabytes
    ///
    /// # Returns
    /// Batch size that uses approximately the target memory
    pub fn tune_for_target(&self, target_memory_mb: usize) -> usize {
        let bytes_per_sample = self.state_dim * 2 * 4 + 8 + 4 + 4;
        let target_bytes = target_memory_mb * 1024 * 1024;
        let max_samples = target_bytes / bytes_per_sample;
        let batch_size = (max_samples / 32) * 32;
        batch_size.clamp(self.min_batch_size, self.max_batch_size)
    }

    /// Validate a batch size value
    ///
    /// # Arguments
    /// * `batch_size` - Batch size to validate
    ///
    /// # Returns
    /// Ok(()) if valid, Err with message if invalid
    pub fn validate(batch_size: usize) -> Result<(), String> {
        if batch_size % 32 != 0 {
            return Err(format!(
                "Batch size must be multiple of 32 (got {})",
                batch_size
            ));
        }
        if batch_size < 32 {
            return Err(format!(
                "Batch size must be at least 32 (got {})",
                batch_size
            ));
        }
        if batch_size > 16384 {
            return Err(format!(
                "Batch size should not exceed 16384 (got {})",
                batch_size
            ));
        }
        Ok(())
    }
}

impl Default for BatchTuner {
    fn default() -> Self {
        Self::new(32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_tuner_new() {
        let tuner = BatchTuner::new(32);
        assert_eq!(tuner.min_batch_size, 256);
        assert_eq!(tuner.max_batch_size, 8192);
        assert_eq!(tuner.state_dim, 32);
    }

    #[test]
    fn test_batch_tuner_with_bounds() {
        let tuner = BatchTuner::with_bounds(64, 100, 5000);
        // Should round to multiples of 32
        assert_eq!(tuner.min_batch_size, 96); // 100 -> 96
        assert_eq!(tuner.max_batch_size, 4992); // 5000 -> 4992
    }

    #[test]
    fn test_batch_tuner_tune_small_gpu() {
        let tuner = BatchTuner::new(32);
        // 2GB GPU - should get smaller batch size
        let batch_size = tuner.tune(2048);
        assert!(batch_size >= tuner.min_batch_size);
        assert!(batch_size <= tuner.max_batch_size);
        assert_eq!(batch_size % 32, 0);
    }

    #[test]
    fn test_batch_tuner_tune_large_gpu() {
        let tuner = BatchTuner::new(32);
        // 24GB GPU - should get larger batch size
        let batch_size = tuner.tune(24576);
        assert!(batch_size >= tuner.min_batch_size);
        assert!(batch_size <= tuner.max_batch_size);
        assert_eq!(batch_size % 32, 0);
    }

    #[test]
    fn test_batch_tuner_validate_valid() {
        assert!(BatchTuner::validate(32).is_ok());
        assert!(BatchTuner::validate(64).is_ok());
        assert!(BatchTuner::validate(256).is_ok());
        assert!(BatchTuner::validate(2048).is_ok());
        assert!(BatchTuner::validate(8192).is_ok());
    }

    #[test]
    fn test_batch_tuner_validate_invalid_not_multiple_of_32() {
        assert!(BatchTuner::validate(31).is_err());
        assert!(BatchTuner::validate(33).is_err());
        assert!(BatchTuner::validate(100).is_err());
        assert!(BatchTuner::validate(513).is_err());
    }

    #[test]
    fn test_batch_tuner_validate_invalid_too_small() {
        assert!(BatchTuner::validate(0).is_err());
        assert!(BatchTuner::validate(1).is_err());
        assert!(BatchTuner::validate(31).is_err());
    }

    #[test]
    fn test_batch_tuner_validate_invalid_too_large() {
        assert!(BatchTuner::validate(16385).is_err());
        assert!(BatchTuner::validate(32768).is_err());
    }

    #[test]
    fn test_batch_tuner_tune_for_target() {
        let tuner = BatchTuner::new(32);
        let batch_size = tuner.tune_for_target(100); // 100MB target
        assert!(batch_size >= tuner.min_batch_size);
        assert_eq!(batch_size % 32, 0);
    }

    #[test]
    fn test_batch_tuner_with_action_dim() {
        let tuner = BatchTuner::new(32).with_action_dim(20);
        assert_eq!(tuner.action_dim, 20);
    }

    #[test]
    fn test_batch_tuner_with_hidden_layers() {
        let tuner = BatchTuner::new(32).with_hidden_layers(vec![64, 128, 256]);
        assert_eq!(tuner.hidden_layers, vec![64, 128, 256]);
    }
}
