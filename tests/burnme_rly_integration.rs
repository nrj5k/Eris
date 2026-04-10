//! Integration test for burnme-rly types
//! Verifies that all exported types are accessible from the main project

#[cfg(test)]
mod integration_tests {
    use burnme_rly::{
        should_train, train_step_with_warmup, BatchedActionSelector, DiscreteSpace, GpuTrainable,
        GpuTrainingCoordinator, Info, StepResult, TensorRingBuffer, TensorTransitionBatch,
        TrainingConfig, TrainingMetrics, Transition, VecEnvironment,
    };

    #[test]
    fn test_types_are_accessible() {
        // Test TrainingConfig
        let config = TrainingConfig::new(1000, 500, 512);
        assert_eq!(config.episodes, 1000);
        assert_eq!(config.max_steps, 500);
        assert_eq!(config.batch_size, 512);

        // Test DiscreteSpace
        let space = DiscreteSpace::new(10);
        assert_eq!(space.n(), 10);

        // Test warmup functions
        // During warmup: train every step
        assert!(should_train(false, 1, 4)); // Always true during warmup

        // After warmup: train every 4 steps
        assert!(should_train(true, 4, 4)); // Steps == frequency
        assert!(!should_train(true, 3, 4)); // Steps < frequency

        // Test VERSION constant
        assert_eq!(burnme_rly::VERSION, "0.1.0");
    }

    #[test]
    fn test_config_builder() {
        let config = TrainingConfig::new(1000, 500, 512).with_warmup_batch_size(256);
        assert_eq!(config.warmup_batch_size, 256);
    }

    #[test]
    fn test_training_metrics_default() {
        let metrics = TrainingMetrics::default();
        assert_eq!(metrics.total_steps, 0);
        assert_eq!(metrics.total_episodes, 0);
        assert_eq!(metrics.avg_reward, 0.0);
    }
}
