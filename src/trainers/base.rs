//! Base configuration shared across all RL trainers.
//!
//! This module re-exports [`burnme_rly::trainers::TrainerConfigBase`] as the canonical
//! configuration for all trainers in eris.
//
// Migration guide (derive_builder -> manual builder):
//   OLD: TrainerConfigBase::builder().gamma(0.95).build().unwrap()
//   NEW: TrainerConfigBase::default().with_gamma(0.95)
//
// Type change: learning_rate is now f64 (was f32). Cast with `as f32` where needed.

// Re-export burnme-rly's canonical types
pub use burnme_rly::trainers::base::{TrainerConfig, TrainerConfigBase};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_config_base_default() {
        let config = TrainerConfigBase::default();
        assert_eq!(config.gamma, 0.99);
        assert_eq!(config.learning_rate, 0.0001);
        assert_eq!(config.batch_size, 2048);
        assert_eq!(config.warmup_batch_size, 256);
        assert_eq!(config.buffer_capacity, 100_000);
        assert_eq!(config.target_update_freq, 1000);
        assert_eq!(config.max_gradient_norm, 1.0);
        assert_eq!(config.epsilon_start, 1.0);
        assert_eq!(config.epsilon_end, 0.01);
        assert_eq!(config.epsilon_decay, 0.995);
        assert_eq!(config.loss_sync_freq, 500);
        assert_eq!(config.warmup_steps, 1000);
    }

    #[test]
    fn test_trainer_config_base_builder() {
        let config = TrainerConfigBase::default()
            .with_gamma(0.95)
            .with_learning_rate(0.001)
            .with_batch_size(1024)
            .with_warmup_batch_size(128)
            .with_buffer_capacity(50_000)
            .with_target_update_freq(50)
            .with_max_gradient_norm(0.5)
            .with_epsilon_start(0.9)
            .with_epsilon_end(0.05)
            .with_epsilon_decay(0.99);

        assert_eq!(config.gamma, 0.95);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 1024);
        assert_eq!(config.warmup_batch_size, 128);
        assert_eq!(config.buffer_capacity, 50_000);
        assert_eq!(config.target_update_freq, 50);
        assert!((config.max_gradient_norm - 0.5).abs() < 1e-6);
        assert_eq!(config.epsilon_start, 0.9);
        assert_eq!(config.epsilon_end, 0.05);
        assert_eq!(config.epsilon_decay, 0.99);
    }

    #[test]
    fn test_trainer_config_base_clone() {
        let config1 = TrainerConfigBase::default().with_gamma(0.95);
        let config2 = config1.clone();
        assert_eq!(config1.gamma, config2.gamma);
        assert_eq!(config1.learning_rate, config2.learning_rate);
    }

    #[test]
    fn test_trainer_config_base_debug() {
        let config = TrainerConfigBase::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("TrainerConfigBase"));
        assert!(debug_str.contains("gamma"));
    }
}
