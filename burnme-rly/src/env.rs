//! Environment types for step results

/// Result of a single environment step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Action taken
    pub action: usize,
    /// Observation after step
    pub observation: Vec<f64>,
    /// Reward received (before clipping)
    pub reward: f64,
    /// Whether episode is done
    pub done: bool,
    /// Additional info
    pub info: Info,
}

impl StepResult {
    /// Create new step result
    pub fn new(action: usize, observation: Vec<f64>, reward: f64, done: bool) -> Self {
        Self {
            action,
            observation,
            reward,
            done,
            info: Info::default(),
        }
    }
}

/// Additional step information
#[derive(Debug, Clone, Default)]
pub struct Info {
    /// Current step number
    pub current_step: usize,
    /// Current time in milliseconds
    pub current_time_ms: u64,
    /// Tier utilization
    pub tier_utilization: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_result() {
        let result = StepResult::new(2, vec![1.0, 2.0], 1.5, false);
        assert_eq!(result.action, 2);
        assert_eq!(result.reward, 1.5);
        assert!(!result.done);
    }
}
