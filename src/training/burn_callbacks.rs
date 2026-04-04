//! Burn training callbacks for DQN-specific training logic.

use std::sync::{Arc, Mutex};

/// Callback for updating target network during DQN training.
///
/// Implements periodic hard updates of the target network from the policy network.
/// This stabilizes training by preventing moving target chasing.
///
/// # Update Strategy
///
/// Hard update: θ_target = θ_policy (full copy)
/// Update frequency: Every `update_freq` training steps
///
/// # Example
///
/// ```
/// use eris::training::TargetUpdateCallback;
///
/// // Create callback to track updates
/// let callback = TargetUpdateCallback::new(1000);
///
/// // Manually check and update
/// if callback.should_update() {
///     // Update target network
/// }
/// ```
pub struct TargetUpdateCallback {
    /// Update frequency (in training steps)
    update_freq: usize,
    /// Step counter
    step_count: Arc<Mutex<usize>>,
}

impl TargetUpdateCallback {
    /// Create new target update callback.
    ///
    /// # Arguments
    /// * `update_freq` - Update target network every N training steps
    pub fn new(update_freq: usize) -> Self {
        Self {
            update_freq,
            step_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get reference to step counter
    pub fn step_count(&self) -> Arc<Mutex<usize>> {
        Arc::clone(&self.step_count)
    }

    /// Increment step counter and check if should update target network
    pub fn should_update(&self) -> bool {
        let mut count = self.step_count.lock().unwrap();
        *count += 1;
        *count % self.update_freq == 0
    }

    /// Reset step counter
    pub fn reset(&self) {
        let mut count = self.step_count.lock().unwrap();
        *count = 0;
    }
}

/// Callback for epsilon decay during DQN training.
///
/// Implements exponential decay of exploration rate (epsilon).
/// Starts from epsilon_start and decays to epsilon_end.
///
/// # Decay Formula
///
/// ε_new = max(ε_end, ε_current * decay_rate)
///
/// # Example
///
/// ```
/// use eris::training::EpsilonDecayCallback;
///
/// // Decay from 1.0 to 0.01 with rate 0.995
/// let callback = EpsilonDecayCallback::new(1.0, 0.01, 0.995);
/// ```
#[derive(Debug, Clone)]
pub struct EpsilonDecayCallback {
    /// Current epsilon value
    epsilon: Arc<Mutex<f32>>,
    /// Final epsilon (minimum)
    epsilon_end: f32,
    /// Decay rate per episode
    decay_rate: f32,
}

impl EpsilonDecayCallback {
    /// Create new epsilon decay callback.
    ///
    /// # Arguments
    /// * `epsilon_start` - Initial exploration rate (usually 1.0)
    /// * `epsilon_end` - Minimum exploration rate (usually 0.01)
    /// * `decay_rate` - Decay multiplier per episode (usually 0.995)
    pub fn new(epsilon_start: f32, epsilon_end: f32, decay_rate: f32) -> Self {
        Self {
            epsilon: Arc::new(Mutex::new(epsilon_start)),
            epsilon_end,
            decay_rate,
        }
    }

    /// Get current epsilon value
    pub fn epsilon(&self) -> f32 {
        *self.epsilon.lock().unwrap()
    }

    /// Get reference to epsilon for external modification
    pub fn epsilon_ref(&self) -> Arc<Mutex<f32>> {
        Arc::clone(&self.epsilon)
    }

    /// Apply decay once
    pub fn decay(&mut self) {
        let mut eps = self.epsilon.lock().unwrap();
        *eps = (*eps * self.decay_rate).max(self.epsilon_end);
    }
}

/// Callback for tracking episode rewards.
///
/// Tracks training progress through episode rewards.
/// Used for logging and monitoring training stability.
///
/// # Example
///
/// ```
/// use eris::training::RewardTrackingCallback;
///
/// let callback = RewardTrackingCallback::new();
/// callback.add_reward(50.0);
/// assert_eq!(callback.average_reward(), 50.0);
/// ```
#[derive(Debug, Clone)]
pub struct RewardTrackingCallback {
    /// Running sum of rewards
    sum: Arc<Mutex<f32>>,
    /// Episode count
    count: Arc<Mutex<usize>>,
}

impl RewardTrackingCallback {
    /// Create new reward tracking callback.
    pub fn new() -> Self {
        Self {
            sum: Arc::new(Mutex::new(0.0)),
            count: Arc::new(Mutex::new(0)),
        }
    }

    /// Add reward from completed episode
    pub fn add_reward(&self, reward: f32) {
        let mut sum = self.sum.lock().unwrap();
        let mut count = self.count.lock().unwrap();
        *sum += reward;
        *count += 1;
    }

    /// Get average reward
    pub fn average_reward(&self) -> f32 {
        let sum = *self.sum.lock().unwrap();
        let count = *self.count.lock().unwrap();
        if count > 0 {
            sum / count as f32
        } else {
            0.0
        }
    }

    /// Get total reward
    pub fn total_reward(&self) -> f32 {
        *self.sum.lock().unwrap()
    }

    /// Get episode count
    pub fn episode_count(&self) -> usize {
        *self.count.lock().unwrap()
    }

    /// Reset tracking
    pub fn reset(&self) {
        let mut sum = self.sum.lock().unwrap();
        let mut count = self.count.lock().unwrap();
        *sum = 0.0;
        *count = 0;
    }
}

impl Default for RewardTrackingCallback {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epsilon_decay_callback() {
        let mut callback = EpsilonDecayCallback::new(1.0, 0.01, 0.995);

        // Initial epsilon
        assert!((callback.epsilon() - 1.0).abs() < 1e-6);

        // Decay once
        callback.decay();
        assert!((callback.epsilon() - 0.995).abs() < 1e-6);

        // Decay to minimum
        for _ in 0..1000 {
            callback.decay();
        }
        assert!((callback.epsilon() - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_reward_tracking_callback() {
        let callback = RewardTrackingCallback::new();

        // No rewards yet
        assert_eq!(callback.average_reward(), 0.0);
        assert_eq!(callback.episode_count(), 0);

        // Add rewards
        callback.add_reward(10.0);
        callback.add_reward(20.0);
        callback.add_reward(30.0);

        assert_eq!(callback.episode_count(), 3);
        assert!((callback.average_reward() - 20.0).abs() < 1e-6);
        assert!((callback.total_reward() - 60.0).abs() < 1e-6);

        // Reset
        callback.reset();
        assert_eq!(callback.episode_count(), 0);
        assert_eq!(callback.average_reward(), 0.0);
    }

    #[test]
    fn test_target_update_callback_creation() {
        let callback = TargetUpdateCallback::new(100);
        assert_eq!(callback.update_freq, 100);

        // Step counter should start at 0
        let count = callback.step_count.lock().unwrap();
        assert_eq!(*count, 0);
    }

    #[test]
    fn test_target_update_should_update() {
        let callback = TargetUpdateCallback::new(5);

        // First 4 steps should not trigger update
        for _ in 0..4 {
            assert!(!callback.should_update());
        }

        // 5th step should trigger update
        assert!(callback.should_update());

        // Next 4 steps should not trigger
        for _ in 0..4 {
            assert!(!callback.should_update());
        }

        // 10th step should trigger
        assert!(callback.should_update());
    }

    #[test]
    fn test_target_update_reset() {
        let callback = TargetUpdateCallback::new(3);

        // Increment a few times
        callback.should_update();
        callback.should_update();

        // Reset
        callback.reset();

        // Counter should be 0
        let count = callback.step_count.lock().unwrap();
        assert_eq!(*count, 0);
    }
}
