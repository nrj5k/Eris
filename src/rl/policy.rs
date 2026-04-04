//! Policy implementation for burn-rl integration
//!
//! This module implements the Policy trait from burn-rl for our models.

use burn::{module::Module, prelude::*, record::Record};
use burn_rl::{Batchable, Policy, PolicyState};
use rand::{rng, RngExt};

use super::types::{Action, ActionDistribution, Observation};
use crate::models::QNetwork;

/// Policy state wrapping a Q-network
///
/// This holds the current model parameters and can be
/// converted to/from a record for saving/loading.
#[derive(Clone, Debug)]
pub struct DQNPolicyState<B: Backend> {
    /// The Q-network parameters
    pub model: QNetwork<B>,
}

impl<B: Backend> PolicyState<B> for DQNPolicyState<B> {
    type Record = <QNetwork<B> as Module<B>>::Record;

    fn into_record(self) -> Self::Record {
        self.model.into_record()
    }

    fn load_record(&self, record: Self::Record) -> Self {
        DQNPolicyState {
            model: self.model.clone().load_record(record),
        }
    }
}

/// DQN Policy implementing burn-rl's Policy trait
///
/// This wraps a Q-network and provides:
/// - Forward pass to get action distributions (Q-values)
/// - Action selection with epsilon-greedy exploration
/// - State management for training
#[derive(Clone, Debug)]
pub struct DQNPolicy<B: Backend> {
    /// The Q-network model
    model: QNetwork<B>,
    /// Exploration rate (epsilon)
    epsilon: f32,
    /// Number of actions
    action_dim: usize,
}

impl<B: Backend> DQNPolicy<B> {
    /// Create a new DQN policy
    ///
    /// # Arguments
    /// * `model` - The Q-network
    /// * `action_dim` - Number of discrete actions
    /// * `epsilon` - Exploration rate [0, 1]
    pub fn new(model: QNetwork<B>, action_dim: usize, epsilon: f32) -> Self {
        Self {
            model,
            epsilon,
            action_dim,
        }
    }

    /// Update exploration rate
    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }

    /// Get the current epsilon value
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get reference to the model
    pub fn model(&self) -> &QNetwork<B> {
        &self.model
    }

    /// Get mutable reference to the model  
    pub fn model_mut(&mut self) -> &mut QNetwork<B> {
        &mut self.model
    }
}

impl<B: Backend> Policy<B> for DQNPolicy<B> {
    type Observation = Observation<B>;
    type ActionDistribution = ActionDistribution<B>;
    type Action = Action<B>;
    type ActionContext = ();
    type PolicyState = DQNPolicyState<B>;

    /// Forward pass to get Q-values (action distribution)
    ///
    /// # Arguments
    /// * `obs` - Batch of observations
    ///
    /// # Returns
    /// * Action distribution (Q-values) for each observation
    fn forward(&mut self, obs: Self::Observation) -> Self::ActionDistribution {
        let q_values = self.model.forward(obs.tensor);
        ActionDistribution { logits: q_values }
    }

    /// Select actions from observations
    ///
    /// # Arguments
    /// * `obs` - Batch of observations
    /// * `deterministic` - If true, use greedy selection; if false, use epsilon-greedy
    ///
    /// # Returns
    /// * (Actions, Action contexts)
    ///   - Actions: batch of selected action indices
    ///   - Contexts: empty vec (no context for DQN)
    fn action(
        &mut self,
        obs: Self::Observation,
        deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>) {
        // Get Q-values
        let q_values = self.model.forward(obs.tensor.clone());
        let device = q_values.device();

        // Unbatch observations to process each one
        let batch_obs = obs.unbatch();
        let batch_size = batch_obs.len();

        let mut actions: Vec<Tensor<B, 2>> = Vec::with_capacity(batch_size);
        let mut rng = rng();

        for (i, _single_obs) in batch_obs.iter().enumerate() {
            // For deterministic: always use greedy
            // For non-deterministic: epsilon-greedy
            let use_greedy = deterministic || {
                let random_val: f32 = rng.random_range(0.0..1.0);
                random_val > self.epsilon
            };

            if use_greedy {
                // Greedy: select argmax for this sample
                let single_q = q_values.clone().slice([i..i + 1]);
                let argmax_idx = single_q.argmax(1); // Int tensor [batch=1, 1]
                                                     // Convert to float for consistent storage
                let argmax_float: Tensor<B, 2> = argmax_idx.float();
                actions.push(argmax_float);
            } else {
                // Random: sample uniformly from action space
                let random_action: usize = rng.random_range(0..self.action_dim);
                let action_tensor: Tensor<B, 2> =
                    Tensor::from_floats([[random_action as f32]], &device);
                actions.push(action_tensor);
            }
        }

        // Stack actions into batch tensor
        let action_tensor = Tensor::cat(actions, 0);

        (
            Action {
                indices: action_tensor,
            },
            vec![(); batch_size],
        )
    }

    /// Update policy parameters
    fn update(&mut self, update: Self::PolicyState) {
        self.model = update.model;
    }

    /// Get current policy state
    fn state(&self) -> Self::PolicyState {
        DQNPolicyState {
            model: self.model.clone(),
        }
    }

    /// Load policy from a record
    fn load_record(self, record: <Self::PolicyState as PolicyState<B>>::Record) -> Self {
        let state = self.state().load_record(record);
        Self {
            model: state.model,
            epsilon: self.epsilon,
            action_dim: self.action_dim,
        }
    }
}

/// Policy for combined bandit + DQN model
///
/// This combines:
/// 1. Bandit feature extraction + importance scoring
/// 2. DQN Q-value prediction
/// 3. Tier-aware action selection
#[derive(Clone, Debug)]
pub struct CombinedPolicyState<B: Backend> {
    /// Bandit model parameters
    pub bandit: crate::models::ContextualBandit<B>,
    /// Q-network model parameters
    pub qnetwork: QNetwork<B>,
}

impl<B: Backend> PolicyState<B> for CombinedPolicyState<B> {
    type Record = (
        <crate::models::ContextualBandit<B> as Module<B>>::Record,
        <QNetwork<B> as Module<B>>::Record,
    );

    fn into_record(self) -> Self::Record {
        (self.bandit.into_record(), self.qnetwork.into_record())
    }

    fn load_record(&self, record: Self::Record) -> Self {
        CombinedPolicyState {
            bandit: self.bandit.clone().load_record(record.0),
            qnetwork: self.qnetwork.clone().load_record(record.1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DQNConfig;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_dqn_policy_state_roundtrip() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let config = DQNConfig::builder()
            .input_dim(10)
            .action_dim(5)
            .build()
            .unwrap();
        let model: QNetwork<TestBackend> = config.init(&device);

        let state = DQNPolicyState {
            model: model.clone(),
        };
        let record = state.clone().into_record();
        let loaded = state.load_record(record);

        // Just verify we can load - the model structure is preserved
        let _model = loaded.model;
    }

    #[test]
    fn test_dqn_policy_forward() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let config = DQNConfig::builder()
            .input_dim(10)
            .action_dim(5)
            .dueling(false)
            .build()
            .unwrap();
        let model: QNetwork<TestBackend> = config.init(&device);

        let mut policy = DQNPolicy::new(model, 5, 0.1);

        // Create batch of 3 observations
        let obs = Observation {
            tensor: Tensor::zeros([3, 10], &device),
        };

        let dist = policy.forward(obs);
        assert_eq!(dist.logits.dims(), [3, 5]);
    }

    #[test]
    fn test_dqn_policy_action_deterministic() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let config = DQNConfig::builder()
            .input_dim(10)
            .action_dim(5)
            .dueling(false)
            .build()
            .unwrap();
        let model: QNetwork<TestBackend> = config.init(&device);

        let mut policy = DQNPolicy::new(model, 5, 0.0); // epsilon = 0, always greedy

        let obs = Observation {
            tensor: Tensor::ones([2, 10], &device),
        };

        let (actions, contexts) = policy.action(obs, true);
        assert_eq!(actions.indices.dims(), [2, 1]);
        assert_eq!(contexts.len(), 2);

        // Actions should be in valid range
        let action_data: Vec<f32> = actions.indices.into_data().to_vec().unwrap();
        for action in action_data {
            assert!(
                action >= 0.0 && action < 5.0,
                "Action {} out of range",
                action
            );
        }
    }

    #[test]
    fn test_dqn_policy_action_exploration() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let config = DQNConfig::builder()
            .input_dim(10)
            .action_dim(5)
            .dueling(false)
            .build()
            .unwrap();
        let model: QNetwork<TestBackend> = config.init(&device);

        // High epsilon to ensure some exploration
        let mut policy = DQNPolicy::new(model, 5, 0.5);

        let obs = Observation {
            tensor: Tensor::ones([10, 10], &device),
        };

        let (actions, contexts) = policy.action(obs, false);
        assert_eq!(actions.indices.dims(), [10, 1]);
        assert_eq!(contexts.len(), 10);

        // All actions should be in valid range [0, 5)
        let action_data: Vec<f32> = actions.indices.into_data().to_vec().unwrap();
        for action in action_data {
            assert!(
                action >= 0.0 && action < 5.0,
                "Action {} out of range",
                action
            );
        }
    }

    #[test]
    fn test_dqn_policy_update() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let config = DQNConfig::builder()
            .input_dim(10)
            .action_dim(5)
            .build()
            .unwrap();
        let model1: QNetwork<TestBackend> = config.init(&device);
        let model2: QNetwork<TestBackend> = config.init(&device);

        let mut policy = DQNPolicy::new(model1, 5, 0.1);

        // Update to new model
        let new_state = DQNPolicyState { model: model2 };
        policy.update(new_state);

        // Verify update happened (no panic means success)
        let _state = policy.state();
    }
}
