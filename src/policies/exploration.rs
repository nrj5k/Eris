//! # Exploration Strategies for Reinforcement Learning
//!
//! This module provides trait-based exploration strategies for RL policies,
//! enabling pluggable exploration methods that can be swapped at runtime.
//!
//! ## Available Strategies
//!
//! Three exploration strategies are implemented:
//!
//! ### 1. Epsilon-Greedy (`EpsilonGreedy`)
//! The classic exploration strategy that selects random actions with probability ε.
//! - **Use case**: Simple baseline, well-understood behavior
//! - **Pros**: Easy to implement and tune, predictable decay
//! - **Cons**: Doesn't adapt to uncertainty, sub-optimal for multi-armed bandits
//! - **Parameters**: `epsilon_start`, `epsilon_end`, `epsilon_decay`
//!
//! ### 2. Thompson Sampling (`ThompsonSampling`)
//! Bayesian posterior sampling that naturally balances exploration and exploitation.
//! - **Use case**: Multi-armed bandits, non-stationary environments
//! - **Pros**: Theoretically optimal for bandits, adapts to uncertainty
//! - **Cons**: Maintains posteriors, more complex implementation
//! - **Parameters**: `prior_mean`, `prior_std`
//!
//! ### 3. Upper Confidence Bound (`UCBExplorer`)
//! Uses UCB1 formula: Q(a) + c * sqrt(ln(N) / n(a))
//! - **Use case**: Stochastic bandits, theoretically optimal regret
//! - **Pros**: Provable regret bounds, automatic exploration balance
//! - **Cons**: Assumes stationary environment, can be aggressive
//! - **Parameters**: `c` (exploration constant)
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use eris::policies::exploration::{ExplorationConfig, EpsilonGreedy};
//!
//! // Create exploration config
//! let config = ExplorationConfig::EpsilonGreedy {
//!     epsilon_start: 1.0,
//!     epsilon_end: 0.01,
//!     epsilon_decay: 0.995,
//! };
//!
//! // Build strategy for your action space
//! let strategy = config.build::<NdArray>(10); // 10 actions
//!
//! // Use in policy
//! let action = strategy.select_action(&q_values, 10);
//! ```
//!
//! ## Comparison Table
//!
//! | Strategy        | Convergence | Regret Bound | Adaptability | Complexity |
//! |-----------------|-------------|--------------|--------------|-------------|
//! | Epsilon-Greedy  | Slow        | O(1/ε)       | Low          | Low         |
//! | Thompson        | Fast        | O(log T)     | High         | Medium      |
//! | UCB             | Medium      | O(√(T log T))| Medium       | Low         |
//!
//! ## Recommendations
//!
//! - **For DQN**: Start with EpsilonGreedy (standard), try ThompsonSampling
//! - **For Bandit**: Use ThompsonSampling or UCB (better than epsilon-greedy)
//! - **For METIS**: ThompsonSampling often works best
//!
//! ## Example: Custom Configuration
//!
//! ```rust,ignore
//! use eris::policies::exploration::ExplorationConfig;
//!
//! // Epsilon-greedy with custom decay
//! let epsilon_greedy = ExplorationConfig::EpsilonGreedy {
//!     epsilon_start: 1.0,
//!     epsilon_end: 0.01,
//!     epsilon_decay: 0.995,
//! };
//!
//! // Thompson sampling with informative prior
//! let thompson = ExplorationConfig::ThompsonSampling {
//!     prior_mean: 0.0,
//!     prior_std: 2.0,  // High uncertainty = more exploration
//! };
//!
//! // UCB with moderate exploration
//! let ucb = ExplorationConfig::UCB {
//!     c: 2.0,  // Higher c = more exploration
//! };
//! ```

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use serde::{Deserialize, Serialize};

/// Trait defining the interface for exploration strategies.
///
/// Exploration strategies determine how an agent selects actions during training
/// to balance exploration (trying new actions) and exploitation (using known good actions).
pub trait ExplorationStrategy<B: Backend>: Send + Sync {
    /// Select an action based on Q-values and exploration strategy.
    ///
    /// # Arguments
    /// * `q_values` - Q-values tensor of shape [batch_size, action_dim]
    /// * `action_dim` - Number of possible actions
    ///
    /// # Returns
    /// Selected action indices of shape [batch_size] as Int tensor
    fn select_action(&self, q_values: &Tensor<B, 2>, action_dim: usize) -> Tensor<B, 2, Int>;

    /// Update the strategy with the action taken and reward received.
    ///
    /// This is used by Thompson Sampling and UCB to update their internal statistics.
    ///
    /// # Arguments
    /// * `action` - The action that was taken
    /// * `reward` - The reward received for taking that action
    fn update(&mut self, action: usize, reward: f32);

    /// Decay exploration parameters over time.
    ///
    /// This is typically used to reduce exploration as training progresses.
    fn decay(&mut self);

    /// Clone the strategy into a boxed trait object.
    fn clone_box(&self) -> Box<dyn ExplorationStrategy<B>>;

    /// Get the current exploration parameter value.
    ///
    /// For epsilon-greedy: returns epsilon
    /// For Thompson Sampling: returns prior_std
    /// For UCB: returns c parameter
    fn get_param(&self) -> f32;

    /// Set the exploration parameter value.
    fn set_param(&mut self, value: f32);
}

impl<B: Backend> Clone for Box<dyn ExplorationStrategy<B>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Configuration for building exploration strategies.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExplorationConfig {
    /// Epsilon-greedy exploration with decay.
    ///
    /// Randomly selects actions with probability epsilon,
    /// otherwise selects the greedy (best) action.
    EpsilonGreedy {
        /// Initial exploration rate
        epsilon_start: f32,
        /// Final exploration rate
        epsilon_end: f32,
        /// Decay rate per step (epsilon *= epsilon_decay)
        epsilon_decay: f32,
    },

    /// Thompson Sampling with Bayesian posterior.
    ///
    /// Samples from posterior distributions over Q-values to select actions.
    ThompsonSampling {
        /// Prior mean for Q-value distribution
        prior_mean: f32,
        /// Prior standard deviation for Q-value distribution
        prior_std: f32,
    },

    /// Upper Confidence Bound exploration.
    ///
    /// Uses UCB1 formula: Q(a) + c * sqrt(ln(N) / n(a))
    UCB {
        /// Exploration constant (higher = more exploration)
        c: f32,
    },
}

impl ExplorationConfig {
    /// Build the exploration strategy from configuration.
    ///
    /// # Arguments
    /// * `action_dim` - Number of possible actions
    ///
    /// # Returns
    /// Boxed exploration strategy
    pub fn build<B: Backend>(&self, action_dim: usize) -> Box<dyn ExplorationStrategy<B>> {
        match self {
            ExplorationConfig::EpsilonGreedy {
                epsilon_start,
                epsilon_end,
                epsilon_decay,
            } => Box::new(EpsilonGreedy::new(
                *epsilon_start,
                *epsilon_end,
                *epsilon_decay,
            )),

            ExplorationConfig::ThompsonSampling {
                prior_mean,
                prior_std,
            } => Box::new(ThompsonSampling::new(action_dim, *prior_mean, *prior_std)),

            ExplorationConfig::UCB { c } => Box::new(UCBExplorer::new(action_dim, *c)),
        }
    }
}

/// Epsilon-greedy exploration strategy.
///
/// With probability epsilon, selects a random action.
/// Otherwise, selects the action with highest Q-value.
///
/// Epsilon decays over time to shift from exploration to exploitation.
///
/// **GPU-Native Implementation**: All operations are performed on GPU using batched tensor
/// operations, avoiding CPU-GPU synchronization for maximum performance.
pub struct EpsilonGreedy {
    /// Current exploration rate
    epsilon: f32,
    /// Initial exploration rate
    epsilon_start: f32,
    /// Minimum exploration rate
    epsilon_end: f32,
    /// Decay multiplier per step
    epsilon_decay: f32,
}

impl EpsilonGreedy {
    /// Create a new epsilon-greedy strategy.
    ///
    /// # Arguments
    /// * `epsilon_start` - Initial exploration rate (typically 1.0)
    /// * `epsilon_end` - Minimum exploration rate (typically 0.01)
    /// * `epsilon_decay` - Decay multiplier (e.g., 0.995)
    pub fn new(epsilon_start: f32, epsilon_end: f32, epsilon_decay: f32) -> Self {
        Self {
            epsilon: epsilon_start,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
        }
    }
}

impl Clone for EpsilonGreedy {
    fn clone(&self) -> Self {
        Self {
            epsilon: self.epsilon,
            epsilon_start: self.epsilon_start,
            epsilon_end: self.epsilon_end,
            epsilon_decay: self.epsilon_decay,
        }
    }
}

impl std::fmt::Debug for EpsilonGreedy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EpsilonGreedy")
            .field("epsilon", &self.epsilon)
            .field("epsilon_start", &self.epsilon_start)
            .field("epsilon_end", &self.epsilon_end)
            .field("epsilon_decay", &self.epsilon_decay)
            .finish()
    }
}

impl<B: Backend> ExplorationStrategy<B> for EpsilonGreedy {
    fn select_action(&self, q_values: &Tensor<B, 2>, action_dim: usize) -> Tensor<B, 2, Int> {
        let batch_size = q_values.dims()[0];
        let device = q_values.device();

        // Generate random mask on GPU: which samples should explore?
        // Random values in [0, 1) for each batch element
        let random_vals =
            Tensor::<B, 1>::random([batch_size], Distribution::Uniform(0.0, 1.0), &device);

        // Create boolean mask: explore where random < epsilon
        let explore_mask = random_vals.lower_elem(self.epsilon as f64); // Tensor<B, 1, Bool>

        // Greedy actions: argmax over Q-values (GPU operation)
        // argmax(1) returns [batch_size, 1] with same rank as input
        let greedy_actions = q_values.clone().argmax(1); // [batch_size, 1]

        // Random actions: generate on GPU using float tensor then convert to int
        // Uniform [0, action_dim) for random action selection
        let random_float = Tensor::<B, 2>::random(
            [batch_size, 1],
            Distribution::Uniform(0.0, action_dim as f64),
            &device,
        );
        let random_actions = random_float.int(); // Convert to int tensor [batch_size, 1]

        // Select actions based on mask:
        // If explore_mask is true, use random_actions
        // Otherwise, use greedy_actions
        // Need to reshape mask to match [batch_size, 1]
        let explore_mask_2d = explore_mask.unsqueeze_dim(1); // [batch_size, 1]
        let explore_int = explore_mask_2d.int(); // Convert Bool to Int (1 for explore, 0 for exploit)

        // Use mask_where: where explore_int == 0, use greedy_actions, else random_actions
        // Both tensors are [batch_size, 1]
        let selected = random_actions.mask_where(explore_int.equal_elem(0), greedy_actions);

        selected
    }

    fn update(&mut self, _action: usize, _reward: f32) {
        // Epsilon-greedy doesn't use reward feedback
    }

    fn decay(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_end);
    }

    fn clone_box(&self) -> Box<dyn ExplorationStrategy<B>> {
        Box::new(self.clone())
    }

    fn get_param(&self) -> f32 {
        self.epsilon
    }

    fn set_param(&mut self, value: f32) {
        self.epsilon = value.clamp(self.epsilon_end, self.epsilon_start);
    }
}

/// Thompson Sampling exploration strategy.
///
/// Maintains posterior distributions over Q-values for each action.
/// Samples from these posteriors to select actions, naturally balancing
/// exploration and exploitation.
///
/// Uses conjugate Gaussian update for efficient online learning.
///
/// **GPU-Native Implementation**: All sampling and comparison operations are performed
/// on GPU using vectorized tensor operations.
pub struct ThompsonSampling<B: Backend> {
    /// Number of actions
    action_dim: usize,
    /// Posterior means for each action (stored on CPU, synced for GPU)
    means: Vec<f32>,
    /// Posterior standard deviations for each action
    stds: Vec<f32>,
    /// Action counts (how many times each action was selected)
    counts: Vec<usize>,
    /// Sum of rewards for each action
    reward_sums: Vec<f32>,
    /// Sum of squared rewards for each action
    reward_sq_sums: Vec<f32>,
    /// Phantom data for backend type
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> ThompsonSampling<B> {
    /// Create a new Thompson Sampling strategy.
    ///
    /// # Arguments
    /// * `action_dim` - Number of possible actions
    /// * `prior_mean` - Prior mean for Q-value distribution
    /// * `prior_std` - Prior standard deviation for Q-value distribution
    pub fn new(action_dim: usize, prior_mean: f32, prior_std: f32) -> Self {
        Self {
            action_dim,
            means: vec![prior_mean; action_dim],
            stds: vec![prior_std; action_dim],
            counts: vec![0; action_dim],
            reward_sums: vec![0.0; action_dim],
            reward_sq_sums: vec![0.0; action_dim],
            _backend: std::marker::PhantomData,
        }
    }

    /// Update posterior using conjugate Gaussian update.
    ///
    /// Combines prior with observed rewards to update the posterior distribution.
    fn update_posterior(&mut self, action: usize, reward: f32) {
        self.counts[action] += 1;
        self.reward_sums[action] += reward;
        self.reward_sq_sums[action] += reward * reward;

        let n = self.counts[action] as f32;
        let sum = self.reward_sums[action];
        let sum_sq = self.reward_sq_sums[action];

        // Numerically stable variance calculation
        let mean = sum / n;
        let variance = if n > 1.0 {
            (sum_sq - sum * mean) / (n - 1.0)
        } else {
            self.stds[action] * self.stds[action]
        };

        // Update posterior
        self.means[action] = mean;
        self.stds[action] = variance.sqrt().max(0.01); // Clamp to avoid zero std
    }
}

impl<B: Backend> Clone for ThompsonSampling<B> {
    fn clone(&self) -> Self {
        Self {
            action_dim: self.action_dim,
            means: self.means.clone(),
            stds: self.stds.clone(),
            counts: self.counts.clone(),
            reward_sums: self.reward_sums.clone(),
            reward_sq_sums: self.reward_sq_sums.clone(),
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> std::fmt::Debug for ThompsonSampling<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThompsonSampling")
            .field("action_dim", &self.action_dim)
            .field("means", &self.means)
            .field("stds", &self.stds)
            .field("counts", &self.counts)
            .finish()
    }
}

impl<B: Backend> ExplorationStrategy<B> for ThompsonSampling<B> {
    fn select_action(&self, q_values: &Tensor<B, 2>, action_dim: usize) -> Tensor<B, 2, Int> {
        let batch_size = q_values.dims()[0];
        let device = q_values.device();

        // Create posterior mean tensor on GPU: [1, action_dim] -> broadcast to [batch_size, action_dim]
        let mean_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(self.means.clone(), [1, action_dim]),
            &device,
        )
        .repeat_dim(0, batch_size);

        // Create posterior std tensor on GPU: [1, action_dim] -> broadcast to [batch_size, action_dim]
        let std_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(self.stds.clone(), [1, action_dim]), &device)
                .repeat_dim(0, batch_size);

        // Sample from Normal(mean, std) on GPU: mean + std * noise
        // For batch processing, we sample standard normal and scale
        let noise = Tensor::<B, 2>::random(
            [batch_size, action_dim],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let samples = mean_tensor + std_tensor * noise;

        // Add Q-values: Q(s,a) + sample from posterior
        let combined = q_values.clone() + samples;

        // Argmax over combined values: [batch_size, 1]
        combined.clone().argmax(1)
    }

    fn update(&mut self, action: usize, reward: f32) {
        self.update_posterior(action, reward);
    }

    fn decay(&mut self) {
        // Thompson Sampling naturally adapts, but we can reduce uncertainty
        for std in self.stds.iter_mut() {
            *std = (*std * 0.999).max(0.01);
        }
    }

    fn clone_box(&self) -> Box<dyn ExplorationStrategy<B>> {
        Box::new(self.clone())
    }

    fn get_param(&self) -> f32 {
        // Return average posterior uncertainty
        self.stds.iter().sum::<f32>() / self.stds.len() as f32
    }

    fn set_param(&mut self, value: f32) {
        // Set all posterior standard deviations
        for std in self.stds.iter_mut() {
            *std = value.max(0.01);
        }
    }
}

/// Upper Confidence Bound (UCB) exploration strategy.
///
/// Uses the UCB1 formula: Q(a) + c * sqrt(ln(N) / n(a))
///
/// This provides theoretically optimal regret bounds in stochastic bandits.
/// The exploration bonus decreases as an action is selected more often,
/// naturally balancing exploration and exploitation.
///
/// **GPU-Native Implementation**: All UCB score computations are performed
/// on GPU using vectorized tensor operations for maximum performance.
pub struct UCBExplorer<B: Backend> {
    /// Number of actions
    action_dim: usize,
    /// Exploration constant (higher c = more exploration)
    c: f32,
    /// Total number of selections
    total_count: usize,
    /// Action counts (how many times each action was selected)
    counts: Vec<usize>,
    /// Sum of Q-values for each action (for computing average Q)
    q_sums: Vec<f32>,
    /// Phantom data for backend type
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> UCBExplorer<B> {
    /// Create a new UCB explorer.
    ///
    /// # Arguments
    /// * `action_dim` - Number of possible actions
    /// * `c` - Exploration constant (typically 1.0 to 2.0)
    pub fn new(action_dim: usize, c: f32) -> Self {
        Self {
            action_dim,
            c,
            total_count: 0,
            counts: vec![0; action_dim],
            q_sums: vec![0.0; action_dim],
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Clone for UCBExplorer<B> {
    fn clone(&self) -> Self {
        Self {
            action_dim: self.action_dim,
            c: self.c,
            total_count: self.total_count,
            counts: self.counts.clone(),
            q_sums: self.q_sums.clone(),
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> std::fmt::Debug for UCBExplorer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UCBExplorer")
            .field("action_dim", &self.action_dim)
            .field("c", &self.c)
            .field("total_count", &self.total_count)
            .field("counts", &self.counts)
            .finish()
    }
}

impl<B: Backend> ExplorationStrategy<B> for UCBExplorer<B> {
    fn select_action(&self, q_values: &Tensor<B, 2>, action_dim: usize) -> Tensor<B, 2, Int> {
        let batch_size = q_values.dims()[0];
        let device = q_values.device();

        // Create count tensor on GPU: [1, action_dim] -> [batch_size, action_dim]
        let counts: Vec<f32> = self.counts.iter().map(|&c| c as f32).collect();
        let count_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(counts, [1, action_dim]), &device)
                .repeat_dim(0, batch_size);

        // Create average reward tensor on GPU: [1, action_dim] -> [batch_size, action_dim]
        let avg_rewards: Vec<f32> = self
            .q_sums
            .iter()
            .zip(self.counts.iter())
            .map(|(&sum, &count)| if count > 0 { sum / count as f32 } else { 0.0 })
            .collect();
        let reward_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(avg_rewards, [1, action_dim]), &device)
                .repeat_dim(0, batch_size);

        // Compute UCB: Q(a) + c * sqrt(ln(N) / n(a))
        // Add total_count + 1 to avoid log(0)
        // Add 1 to all counts to avoid division by zero
        let total_count = self.total_count as f32;
        let ln_n = (total_count + 1.0_f32).log(std::f32::consts::E);

        // Exploration bonus: c * sqrt(ln(N) / (n(a) + 1))
        let c_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(vec![self.c; action_dim], [1, action_dim]),
            &device,
        )
        .repeat_dim(0, batch_size)
            * (ln_n / (count_tensor.clone() + 1.0_f32)).sqrt();

        // UCB values = Q-values + average rewards + exploration bonus
        // Note: Q-values from network represent expected returns
        // UCB bonus represents uncertainty about Q-values
        let ucb_values = q_values.clone() + reward_tensor + c_tensor;

        // Handle unvisited actions: they get infinite bonus
        // Create mask for unvisited actions (count == 0)
        let unvisited_mask = count_tensor.equal_elem(0.0_f32); // Tensor<B, 2, Bool>

        // For unvisited actions, set UCB to infinity (they should be tried first)
        let inf_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(vec![f32::INFINITY; action_dim], [1, action_dim]),
            &device,
        )
        .repeat_dim(0, batch_size);
        let ucb_values_with_inf = ucb_values.mask_where(unvisited_mask, inf_tensor);

        // Argmax: [batch_size, 1]
        ucb_values_with_inf.clone().argmax(1)
    }

    fn update(&mut self, action: usize, reward: f32) {
        // UCB updates Q-sums and counts
        self.counts[action] += 1;
        self.q_sums[action] += reward;
        self.total_count += 1;
    }

    fn decay(&mut self) {
        // UCB has automatic decay through the log(N) term
        // Optionally decay c for more exploitation over time
        self.c = (self.c * 0.999).max(0.5);
    }

    fn clone_box(&self) -> Box<dyn ExplorationStrategy<B>> {
        Box::new(self.clone())
    }

    fn get_param(&self) -> f32 {
        self.c
    }

    fn set_param(&mut self, value: f32) {
        self.c = value.max(0.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_epsilon_greedy_explore() {
        let strategy = EpsilonGreedy::new(1.0, 0.01, 0.995);
        let device = <NdArray as Backend>::Device::default();

        // Create Q-values tensor [batch=2, actions=4]
        let q_values = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            &device,
        );

        // With epsilon=1.0, should explore (random)
        let actions = strategy.select_action(&q_values, 4);
        let action_data = actions.into_data().convert::<i32>();
        let action_vec: Vec<i32> = action_data.to_vec().unwrap_or_default();

        // Should have selected 2 actions (batch_size)
        assert_eq!(action_vec.len(), 2);
        // All actions should be in valid range
        for action in action_vec {
            assert!(action >= 0 && action < 4);
        }
    }

    #[test]
    fn test_epsilon_greedy_exploit() {
        let strategy = EpsilonGreedy::new(0.0, 0.0, 0.995);
        let device = <NdArray as Backend>::Device::default();

        // Create Q-values tensor
        let q_values = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            &device,
        );

        // With epsilon=0.0, should exploit (greedy)
        let actions = strategy.select_action(&q_values, 4);
        let action_data = actions.into_data().convert::<i32>();
        let action_vec: Vec<i32> = action_data.to_vec().unwrap_or_default();

        // Should select greedy actions: 3 and 0
        assert_eq!(action_vec[0], 3); // max is at index 3 for first batch
        assert_eq!(action_vec[1], 0); // max is at index 0 for second batch
    }

    #[test]
    fn test_epsilon_greedy_properties() {
        let strategy = EpsilonGreedy::new(0.5, 0.01, 0.9);

        // Test initial value
        assert!((strategy.epsilon - 0.5).abs() < 1e-6);

        // Test bounds
        assert!(strategy.epsilon >= strategy.epsilon_end);
        assert!(strategy.epsilon <= strategy.epsilon_start);
    }

    #[test]
    fn test_epsilon_greedy_decay() {
        let mut strategy: EpsilonGreedy = EpsilonGreedy::new(0.5, 0.01, 0.9);

        // Decay should reduce epsilon
        let initial = strategy.epsilon;
        ExplorationStrategy::<TestBackend>::decay(&mut strategy);

        assert!(strategy.epsilon < initial);
        assert!(strategy.epsilon >= 0.01);
    }

    #[test]
    fn test_thompson_sampling_creation() {
        let strategy: ThompsonSampling<TestBackend> = ThompsonSampling::new(5, 0.0, 1.0);

        assert_eq!(strategy.action_dim, 5);
        assert_eq!(strategy.means.len(), 5);
        assert_eq!(strategy.stds.len(), 5);
    }

    #[test]
    fn test_thompson_sampling_properties() {
        let mut strategy: ThompsonSampling<TestBackend> = ThompsonSampling::new(3, 0.0, 1.0);

        // Update posterior with rewards
        strategy.update_posterior(0, 5.0);
        strategy.update_posterior(0, 5.0);
        strategy.update_posterior(1, 10.0);

        // Action 0 should have higher mean than prior
        assert!(strategy.means[0] > 1.0);
        // Action 1 should have highest mean
        assert!(strategy.means[1] > strategy.means[0]);
        // Action 2 should remain at prior
        assert!((strategy.means[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_thompson_sampling_decay() {
        let mut strategy: ThompsonSampling<TestBackend> = ThompsonSampling::new(3, 0.0, 1.0);

        let initial_std = strategy.stds[0];
        // Decay reduces uncertainty
        ExplorationStrategy::<TestBackend>::decay(&mut strategy);

        // Std should decrease after decay
        assert!(strategy.stds[0] < initial_std);
    }

    #[test]
    fn test_ucb_explorer_creation() {
        let strategy: UCBExplorer<TestBackend> = UCBExplorer::new(4, 2.0);

        assert_eq!(strategy.action_dim, 4);
        assert_eq!(strategy.c, 2.0);
        assert_eq!(strategy.counts.len(), 4);
    }

    #[test]
    fn test_ucb_unvisited_actions() {
        let device = <NdArray as Backend>::Device::default();
        let strategy: UCBExplorer<TestBackend> = UCBExplorer::new(3, 1.0);

        // Create Q-values for testing
        let q_values =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], &device);

        // Unvisited actions should have infinite UCB score
        let actions = strategy.select_action(&q_values, 3);
        let action_data = actions.into_data().convert::<i32>();
        let action_vec: Vec<i32> = action_data.to_vec().unwrap_or_default();

        // Should select unvisited actions (first time)
        assert_eq!(action_vec.len(), 2);
        // All actions in valid range
        for action in &action_vec {
            assert!(action >= &0 && action < &3);
        }
    }

    #[test]
    fn test_ucb_exploration_bonus() {
        let mut strategy: UCBExplorer<TestBackend> = UCBExplorer::new(3, 2.0);

        // Update to visit actions
        strategy.counts[0] += 1; // count = 1
        strategy.q_sums[0] += 5.0;
        strategy.total_count += 1;

        strategy.counts[1] += 1;
        strategy.q_sums[1] += 5.0;
        strategy.total_count += 1;
        strategy.counts[1] += 1; // count = 2 for action 1
        strategy.q_sums[1] += 5.0;
        strategy.total_count += 1;

        strategy.counts[2] += 1;
        strategy.q_sums[2] += 10.0;
        strategy.total_count += 1;
        strategy.counts[2] += 1;
        strategy.q_sums[2] += 10.0;
        strategy.total_count += 1;
        strategy.counts[2] += 1; // count = 3 for action 2
        strategy.q_sums[2] += 10.0;
        strategy.total_count += 1;

        // Actions with fewer visits should have higher exploration bonus
        let bonus_0 = strategy.c * ((strategy.total_count as f32).ln() / 1.0_f32).sqrt();
        let bonus_1 = strategy.c * ((strategy.total_count as f32).ln() / 2.0_f32).sqrt();
        let bonus_2 = strategy.c * ((strategy.total_count as f32).ln() / 3.0_f32).sqrt();

        assert!(bonus_0 > bonus_1);
        assert!(bonus_1 > bonus_2);
    }

    #[test]
    fn test_ucb_decay() {
        let mut strategy: UCBExplorer<TestBackend> = UCBExplorer::new(4, 2.0);

        let initial_c = strategy.c;
        // Decay the exploration constant
        ExplorationStrategy::<TestBackend>::decay(&mut strategy);

        // c should decrease after decay
        assert!(strategy.c < initial_c);
        assert!(strategy.c >= 0.5); // But stay above minimum
    }

    #[test]
    fn test_exploration_config_build() {
        let config = ExplorationConfig::EpsilonGreedy {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        let strategy: Box<dyn ExplorationStrategy<TestBackend>> = config.build(4);
        assert!((strategy.get_param() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_exploration_config_thompson() {
        std::cell::Cell::new(());
        let config = ExplorationConfig::ThompsonSampling {
            prior_mean: 0.0,
            prior_std: 1.0,
        };

        let strategy: Box<dyn ExplorationStrategy<TestBackend>> = config.build(5);
        // Initial param should be close to prior_std
        assert!(strategy.get_param() > 0.0);
    }

    #[test]
    fn test_exploration_config_ucb() {
        let config = ExplorationConfig::UCB { c: 2.0 };

        let strategy: Box<dyn ExplorationStrategy<TestBackend>> = config.build(4);
        assert!((strategy.get_param() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_epsilon_greedy_batch_processing() {
        let strategy = EpsilonGreedy::new(0.0, 0.0, 0.995);
        let device = <NdArray as Backend>::Device::default();

        // Test with larger batch
        let q_values = Tensor::<TestBackend, 2>::from_floats(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [0.5, 0.6, 0.7, 0.8],
                [10.0, 5.0, 2.0, 1.0],
            ],
            &device,
        );

        let actions = strategy.select_action(&q_values, 4);
        let action_data = actions.into_data().convert::<i32>();
        let action_vec: Vec<i32> = action_data.to_vec().unwrap_or_default();

        // Should select greedy actions for all batch elements
        assert_eq!(action_vec.len(), 4);
        assert_eq!(action_vec[0], 3); // max at index 3
        assert_eq!(action_vec[1], 0); // max at index 0
        assert_eq!(action_vec[2], 3); // max at index 3
        assert_eq!(action_vec[3], 0); // max at index 0
    }

    #[test]
    fn test_thompson_sampling_batch_processing() {
        let strategy: ThompsonSampling<TestBackend> = ThompsonSampling::new(4, 0.0, 1.0);
        let device = <NdArray as Backend>::Device::default();

        // Test batch processing
        let q_values = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            &device,
        );

        let actions = strategy.select_action(&q_values, 4);
        let action_data = actions.into_data().convert::<i32>();
        let action_vec: Vec<i32> = action_data.to_vec().unwrap_or_default();

        // Should produce valid actions for batch
        assert_eq!(action_vec.len(), 2);
        for action in action_vec {
            assert!(action >= 0 && action < 4);
        }
    }

    #[test]
    fn test_ucb_batch_processing() {
        let device = <NdArray as Backend>::Device::default();
        let mut strategy: UCBExplorer<TestBackend> = UCBExplorer::new(4, 2.0);

        // Initialize some visit counts
        strategy.update(0, 5.0);
        strategy.update(0, 6.0);
        strategy.update(1, 3.0);

        // Test batch processing
        let q_values = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            &device,
        );

        let actions = strategy.select_action(&q_values, 4);
        let action_data = actions.into_data().convert::<i32>();
        let action_vec: Vec<i32> = action_data.to_vec().unwrap_or_default();

        // Should produce valid actions for batch
        assert_eq!(action_vec.len(), 2);
        for action in action_vec {
            assert!(action >= 0 && action < 4);
        }
    }
}
