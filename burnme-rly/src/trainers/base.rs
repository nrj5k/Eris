/// Shared configuration trait for all RL trainers
pub trait TrainerConfig: Clone {
    /// Discount factor for TD target
    fn gamma(&self) -> f32;

    /// Initial exploration rate (epsilon)
    fn epsilon_start(&self) -> f32;

    /// Minimum exploration rate
    fn epsilon_end(&self) -> f32;

    /// Multiplicative decay per step
    fn epsilon_decay(&self) -> f32;

    /// Learning rate
    fn learning_rate(&self) -> f64;

    /// Batch size for training
    fn batch_size(&self) -> usize;

    /// Replay buffer capacity
    fn buffer_capacity(&self) -> usize;

    /// Target network update frequency (steps)
    fn target_update_freq(&self) -> usize;

    /// Maximum gradient norm for clipping
    fn max_gradient_norm(&self) -> f32;

    /// Loss sync frequency for async loss accumulation (Metis optimization)
    /// Default: 100 (sync every 100 steps)
    fn loss_sync_freq(&self) -> usize {
        100
    }

    /// Number of warmup steps before using full batch size
    /// Default: 1000
    fn warmup_steps(&self) -> usize {
        1000
    }

    /// Batch size during warmup period
    /// Default: 256
    fn warmup_batch_size(&self) -> usize {
        256
    }

    /// Check if batch size is warp-aligned (multiple of 32 for NVIDIA GPUs).
    ///
    /// Warp-aligned batches maximize GPU occupancy and utilization by ensuring
    /// all threads in a warp (32 threads on NVIDIA GPUs) are fully utilized.
    /// Non-aligned sizes waste GPU cycles and reduce throughput.
    ///
    /// # Returns
    /// * `true` if batch_size is a multiple of 32 and positive
    /// * `false` otherwise
    fn is_batch_size_warp_aligned(&self) -> bool {
        let bs = self.batch_size();
        bs.is_multiple_of(32) && bs > 0
    }

    /// Get the nearest warp-aligned batch size.
    ///
    /// Rounds up to the nearest multiple of 32 (NVIDIA warp size).
    /// For example: 100 -> 128, 2000 -> 2048, 2048 -> 2048
    ///
    /// # Returns
    /// Recommended batch size that is warp-aligned
    fn align_batch_size_to_warp(&self) -> usize {
        let bs = self.batch_size();
        if bs.is_multiple_of(32) {
            bs
        } else {
            ((bs / 32) + 1) * 32
        }
    }

    /// Validate all configuration parameters
    fn validate(&self) -> Result<(), String> {
        // Gamma must be in (0, 1]
        let gamma = self.gamma();
        if gamma <= 0.0 || gamma > 1.0 {
            return Err(format!("gamma must be in (0, 1], got {}", gamma));
        }

        // Batch size must be positive
        let batch_size = self.batch_size();
        if batch_size == 0 {
            return Err("batch_size must be > 0".to_string());
        }

        // Buffer capacity must be positive
        let buffer_capacity = self.buffer_capacity();
        if buffer_capacity == 0 {
            return Err("buffer_capacity must be > 0".to_string());
        }

        // Learning rate must be positive
        let lr = self.learning_rate();
        if lr <= 0.0 {
            return Err(format!("learning_rate must be > 0, got {}", lr));
        }

        // Epsilon validation
        let eps_start = self.epsilon_start();
        let eps_end = self.epsilon_end();
        if eps_start <= 0.0 || eps_start > 1.0 {
            return Err(format!(
                "epsilon_start must be in (0, 1], got {}",
                eps_start
            ));
        }
        if eps_end < 0.0 || eps_end > eps_start {
            return Err(format!(
                "epsilon_end must be in [0, epsilon_start], got {}",
                eps_end
            ));
        }

        // Max gradient norm must be positive
        let max_grad = self.max_gradient_norm();
        if max_grad <= 0.0 {
            return Err(format!("max_gradient_norm must be > 0, got {}", max_grad));
        }

        // Warn about non-warp-aligned batch size
        if !self.is_batch_size_warp_aligned() {
            log::warn!(
                "[STAGE:WARN] batch_size {} is not warp-aligned (not a multiple of 32). \
                 Consider using {} for better GPU utilization. \
                 Warp alignment reduces wasted GPU cycles.",
                self.batch_size(),
                self.align_batch_size_to_warp()
            );
        }

        Ok(())
    }
}
