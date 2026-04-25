/// Base configuration struct shared by all RL trainers
#[derive(Debug, Clone)]
pub struct TrainerConfigBase {
    pub gamma: f32,
    pub epsilon_start: f32,
    pub epsilon_end: f32,
    pub epsilon_decay: f32,
    pub learning_rate: f64,
    pub batch_size: usize,
    pub buffer_capacity: usize,
    pub target_update_freq: usize,
    pub max_gradient_norm: f32,
    pub loss_sync_freq: usize,
    pub warmup_steps: usize,
    pub warmup_batch_size: usize,
}

impl Default for TrainerConfigBase {
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
            loss_sync_freq: 500,
            warmup_steps: 1000,
            warmup_batch_size: 256,
        }
    }
}

impl TrainerConfigBase {
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

    pub fn with_epsilon_decay(mut self, decay: f32) -> Self {
        self.epsilon_decay = decay;
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

    pub fn with_loss_sync_freq(mut self, freq: usize) -> Self {
        self.loss_sync_freq = freq;
        self
    }

    pub fn with_warmup_steps(mut self, steps: usize) -> Self {
        self.warmup_steps = steps;
        self
    }

    pub fn with_warmup_batch_size(mut self, size: usize) -> Self {
        self.warmup_batch_size = size;
        self
    }
}

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

        // Target update frequency must be positive
        let target_freq = self.target_update_freq();
        if target_freq == 0 {
            return Err("target_update_freq must be > 0".to_string());
        }

        // Warmup batch size must be positive
        let warmup_bs = self.warmup_batch_size();
        if warmup_bs == 0 {
            return Err("warmup_batch_size must be > 0".to_string());
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

impl TrainerConfig for TrainerConfigBase {
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

    fn loss_sync_freq(&self) -> usize {
        self.loss_sync_freq
    }

    fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    fn warmup_batch_size(&self) -> usize {
        self.warmup_batch_size
    }
}
