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

        Ok(())
    }
}
