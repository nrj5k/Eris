//! Burn-compatible metrics for DQN training.
//!
//! This module provides custom metrics that integrate with Burn's metric system
//! for automatic progress tracking and logging during training.

use burn::tensor::backend::Backend;
use burn::train::metric::{Metric, MetricMetadata, Numeric, NumericEntry, SerializedEntry};
use std::sync::Arc;

/// Input type for RewardMetric.
///
/// Contains the total reward for an episode.
#[derive(Debug, Clone)]
pub struct RewardInput {
    /// Total reward for the episode
    pub reward: f32,
}

/// Metric for tracking episode rewards.
///
/// Tracks the average reward over episodes during training.
///
/// # Example
///
/// ```
/// use eris::training::burn_metrics::{RewardMetric, RewardInput};
/// use burn::train::metric::{Metric, MetricMetadata};
///
/// let mut metric = RewardMetric::new();
/// let input = RewardInput { reward: 50.0 };
/// let metadata = MetricMetadata::default();
///
/// metric.update(&input, &metadata);
/// assert!(metric.value() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct RewardMetric {
    /// Running sum of rewards
    sum: f32,
    /// Number of episodes tracked
    count: usize,
}

impl Default for RewardMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardMetric {
    /// Create a new reward metric.
    pub fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }
}

impl Metric for RewardMetric {
    type Input = RewardInput;

    fn name(&self) -> Arc<String> {
        Arc::new("Reward".to_string())
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.sum += input.reward;
        self.count += 1;

        let avg = if self.count > 0 {
            self.sum / self.count as f32
        } else {
            0.0
        };

        let formatted = format!("{:.2}", avg);
        SerializedEntry::new(formatted.clone(), formatted)
    }

    fn clear(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
}

impl Numeric for RewardMetric {
    fn value(&self) -> NumericEntry {
        let avg = if self.count > 0 {
            self.sum / self.count as f32
        } else {
            0.0
        };
        NumericEntry::Value(avg as f64)
    }

    fn running_value(&self) -> NumericEntry {
        self.value()
    }
}

/// Input type for EpsilonMetric.
///
/// Contains the current exploration rate.
#[derive(Debug, Clone)]
pub struct EpsilonInput {
    /// Current epsilon value (0.0 to 1.0)
    pub epsilon: f32,
}

/// Metric for tracking exploration rate (epsilon).
///
/// Tracks the exploration rate decay during epsilon-greedy training.
///
/// # Example
///
/// ```
/// use eris::training::burn_metrics::{EpsilonMetric, EpsilonInput};
/// use burn::train::metric::{Metric, MetricMetadata};
///
/// let mut metric = EpsilonMetric::new();
/// let input = EpsilonInput { epsilon: 0.95 };
/// let metadata = MetricMetadata::default();
///
/// metric.update(&input, &metadata);
/// assert!(metric.value() > 0.9);
/// ```
#[derive(Debug, Clone)]
pub struct EpsilonMetric {
    /// Current epsilon value
    current: f32,
}

impl Default for EpsilonMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl EpsilonMetric {
    /// Create a new epsilon metric.
    pub fn new() -> Self {
        Self { current: 1.0 }
    }
}

impl Metric for EpsilonMetric {
    type Input = EpsilonInput;

    fn name(&self) -> Arc<String> {
        Arc::new("Epsilon".to_string())
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.current = input.epsilon;
        let formatted = format!("{:.3}", self.current);
        SerializedEntry::new(formatted.clone(), formatted)
    }

    fn clear(&mut self) {
        self.current = 1.0;
    }
}

impl Numeric for EpsilonMetric {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.current as f64)
    }

    fn running_value(&self) -> NumericEntry {
        self.value()
    }
}

/// Input type for TierUtilizationMetric.
///
/// Contains utilization ratios for each tier (0.0 to 1.0).
#[derive(Debug, Clone)]
pub struct TierUtilizationInput {
    /// Utilization ratio for each tier
    pub tier_utilizations: Vec<f32>,
}

/// Metric for tracking tier utilization.
///
/// Tracks the average utilization across all cache tiers.
///
/// # Example
///
/// ```
/// use eris::training::burn_metrics::{TierUtilizationMetric, TierUtilizationInput};
/// use burn::train::metric::{Metric, MetricMetadata};
///
/// let mut metric = TierUtilizationMetric::new();
/// let input = TierUtilizationInput {
///     tier_utilizations: vec![0.7, 0.9, 0.5],
/// };
/// let metadata = MetricMetadata::default();
///
/// metric.update(&input, &metadata);
/// assert!(metric.value() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct TierUtilizationMetric {
    /// Running sum of average tier utilization
    sum: f32,
    /// Number of updates
    count: usize,
}

impl Default for TierUtilizationMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl TierUtilizationMetric {
    /// Create a new tier utilization metric.
    pub fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }
}

impl Metric for TierUtilizationMetric {
    type Input = TierUtilizationInput;

    fn name(&self) -> Arc<String> {
        Arc::new("TierUtilization".to_string())
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        if !input.tier_utilizations.is_empty() {
            let avg: f32 =
                input.tier_utilizations.iter().sum::<f32>() / input.tier_utilizations.len() as f32;
            self.sum += avg;
            self.count += 1;

            let running_avg = self.sum / self.count as f32;
            let formatted = format!("{:.2}%", running_avg * 100.0);
            SerializedEntry::new(formatted.clone(), formatted)
        } else {
            let formatted = "0.00%".to_string();
            SerializedEntry::new(formatted.clone(), formatted)
        }
    }

    fn clear(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
}

impl Numeric for TierUtilizationMetric {
    fn value(&self) -> NumericEntry {
        let avg = if self.count > 0 {
            self.sum / self.count as f32
        } else {
            0.0
        };
        NumericEntry::Value(avg as f64)
    }

    fn running_value(&self) -> NumericEntry {
        self.value()
    }
}

/// Input type for MeanQMetric.
///
/// Contains mean Q-value from training step.
#[derive(Debug, Clone)]
pub struct MeanQInput<B: Backend> {
    /// Mean Q-value
    pub mean_q: f32,
    /// Phantom data for backend
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> MeanQInput<B> {
    /// Create a new MeanQInput.
    pub fn new(mean_q: f32) -> Self {
        Self {
            mean_q,
            _backend: std::marker::PhantomData,
        }
    }
}

/// Metric for tracking mean Q-values during training.
///
/// Q-values represent the expected future rewards for actions.
/// Tracking their mean helps monitor training progress.
///
/// # Example
///
/// ```
/// use eris::training::burn_metrics::{MeanQMetric, MeanQInput};
/// use burn::backend::NdArray;
/// use burn::train::metric::{Metric, MetricMetadata};
///
/// let mut metric = MeanQMetric::<NdArray>::new();
/// let input = MeanQInput::new(15.5);
/// let metadata = MetricMetadata::default();
///
/// metric.update(&input, &metadata);
/// assert!(metric.value() > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct MeanQMetric<B: Backend> {
    /// Running sum of mean Q-values
    sum: f32,
    /// Number of updates
    count: usize,
    /// Phantom data for backend type
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for MeanQMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MeanQMetric<B> {
    /// Create a new mean Q-value metric.
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Metric for MeanQMetric<B> {
    type Input = MeanQInput<B>;

    fn name(&self) -> Arc<String> {
        Arc::new("MeanQ".to_string())
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        self.sum += input.mean_q;
        self.count += 1;

        let avg = if self.count > 0 {
            self.sum / self.count as f32
        } else {
            0.0
        };

        let formatted = format!("{:.2}", avg);
        SerializedEntry::new(formatted.clone(), formatted)
    }

    fn clear(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }
}

impl<B: Backend> Numeric for MeanQMetric<B> {
    fn value(&self) -> NumericEntry {
        let avg = if self.count > 0 {
            self.sum / self.count as f32
        } else {
            0.0
        };
        NumericEntry::Value(avg as f64)
    }

    fn running_value(&self) -> NumericEntry {
        self.value()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::data::dataloader::Progress;

    /// Helper to create a minimal metadata for tests
    fn dummy_metadata() -> MetricMetadata {
        // Create minimal progress info
        MetricMetadata {
            progress: Progress {
                items_processed: 0,
                items_total: 0,
            },
            epoch: 0,
            epoch_total: 0,
            iteration: 0,
            lr: None,
        }
    }

    #[test]
    fn test_reward_metric() {
        let mut metric = RewardMetric::new();

        // Track some rewards
        let metadata = dummy_metadata();
        metric.update(&RewardInput { reward: 10.0 }, &metadata);
        metric.update(&RewardInput { reward: 20.0 }, &metadata);
        metric.update(&RewardInput { reward: 30.0 }, &metadata);

        // Average should be 20.0
        match metric.value() {
            NumericEntry::Value(avg) => assert!((avg - 20.0).abs() < 0.01),
            _ => panic!("Expected Value variant"),
        }
    }

    #[test]
    fn test_reward_metric_clear() {
        let mut metric = RewardMetric::new();

        let metadata = dummy_metadata();
        metric.update(&RewardInput { reward: 100.0 }, &metadata);

        metric.clear();

        // After clear, average should be 0.0
        match metric.value() {
            NumericEntry::Value(avg) => assert!((avg - 0.0).abs() < 0.01),
            _ => panic!("Expected Value variant"),
        }
    }

    #[test]
    fn test_epsilon_metric() {
        let mut metric = EpsilonMetric::new();

        let metadata = dummy_metadata();
        metric.update(&EpsilonInput { epsilon: 0.95 }, &metadata);

        match metric.value() {
            NumericEntry::Value(eps) => assert!((eps - 0.95).abs() < 0.01),
            _ => panic!("Expected Value variant"),
        }

        metric.update(&EpsilonInput { epsilon: 0.90 }, &metadata);

        match metric.value() {
            NumericEntry::Value(eps) => assert!((eps - 0.90).abs() < 0.01),
            _ => panic!("Expected Value variant"),
        }
    }

    #[test]
    fn test_tier_utilization_metric() {
        let mut metric = TierUtilizationMetric::new();

        let metadata = dummy_metadata();
        metric.update(
            &TierUtilizationInput {
                tier_utilizations: vec![0.5, 1.0, 0.5],
            },
            &metadata,
        );

        // Average should be (0.5 + 1.0 + 0.5) / 3 = 0.667
        match metric.value() {
            NumericEntry::Value(avg) => assert!((avg - 0.667).abs() < 0.01),
            _ => panic!("Expected Value variant"),
        }
    }

    #[test]
    fn test_mean_q_metric() {
        use burn::backend::NdArray;

        let mut metric = MeanQMetric::<NdArray>::new();

        let metadata = dummy_metadata();
        metric.update(&MeanQInput::new(15.0), &metadata);
        metric.update(&MeanQInput::new(25.0), &metadata);

        // Average should be 20.0
        match metric.value() {
            NumericEntry::Value(avg) => assert!((avg - 20.0).abs() < 0.01),
            _ => panic!("Expected Value variant"),
        }
    }

    #[test]
    fn test_metric_names() {
        let reward = RewardMetric::new();
        let epsilon = EpsilonMetric::new();
        let tier_util = TierUtilizationMetric::new();

        assert_eq!(*reward.name(), "Reward");
        assert_eq!(*epsilon.name(), "Epsilon");
        assert_eq!(*tier_util.name(), "TierUtilization");
    }
}
