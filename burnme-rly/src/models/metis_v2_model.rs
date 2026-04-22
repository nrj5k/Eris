//! MetisV2Model: Concrete fused Bandit + DQN model
//!
//! This replaces the generic SequentialCompose pattern with a concrete
//! struct for maximum performance. Single forward pass returns all outputs.

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{relu, sigmoid};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

/// Concrete MetisV2 model with fused bandit + DQN
#[derive(Module, Debug)]
pub struct MetisV2Model<B: AutodiffBackend> {
    // Bandit layers
    pub bandit_fc1: Linear<B>,
    pub bandit_fc2: Linear<B>,
    pub bandit_feature_head: Linear<B>,
    pub bandit_score_head: Linear<B>,

    // DQN layers (dueling architecture)
    pub dqn_fc1: Linear<B>,
    pub dqn_fc2: Linear<B>,
    pub dqn_value_fc1: Linear<B>,
    pub dqn_value_fc2: Linear<B>,
    pub dqn_advantage_fc1: Linear<B>,
    pub dqn_advantage_fc2: Linear<B>,

    // Dimensions
    pub state_dim: usize,
    pub feature_dim: usize,
    pub action_dim: usize,
}

impl<B: AutodiffBackend> MetisV2Model<B> {
    /// Create new MetisV2Model with given dimensions
    pub fn new(
        state_dim: usize,
        feature_dim: usize,
        action_dim: usize,
        device: &B::Device,
    ) -> Self {
        // Bandit: state_dim -> 64 -> 128 -> feature_dim + 1 (score)
        let bandit_fc1 = LinearConfig::new(state_dim, 64).init(device);
        let bandit_fc2 = LinearConfig::new(64, 128).init(device);
        let bandit_feature_head = LinearConfig::new(128, feature_dim).init(device);
        let bandit_score_head = LinearConfig::new(128, 1).init(device);

        // DQN: feature_dim -> 128 -> 128 -> value(1) + advantage(action_dim)
        let dqn_fc1 = LinearConfig::new(feature_dim, 128).init(device);
        let dqn_fc2 = LinearConfig::new(128, 128).init(device);
        let dqn_value_fc1 = LinearConfig::new(128, 128).init(device);
        let dqn_value_fc2 = LinearConfig::new(128, 1).init(device);
        let dqn_advantage_fc1 = LinearConfig::new(128, 128).init(device);
        let dqn_advantage_fc2 = LinearConfig::new(128, action_dim).init(device);

        Self {
            bandit_fc1,
            bandit_fc2,
            bandit_feature_head,
            bandit_score_head,
            dqn_fc1,
            dqn_fc2,
            dqn_value_fc1,
            dqn_value_fc2,
            dqn_advantage_fc1,
            dqn_advantage_fc2,
            state_dim,
            feature_dim,
            action_dim,
        }
    }

    /// Single fused forward pass
    /// Returns: (features, importance, q_values)
    pub fn forward(&self, states: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = states.shape().dims[0];

        // Bandit forward
        let x = relu(self.bandit_fc1.forward(states));
        let x = relu(self.bandit_fc2.forward(x));
        let features = self.bandit_feature_head.forward(x.clone());
        let scores = self.bandit_score_head.forward(x);
        let importance = sigmoid(scores);

        // DQN forward (dueling) - use detached features
        let features_detached = features.clone().detach();
        let x = relu(self.dqn_fc1.forward(features_detached));
        let x = relu(self.dqn_fc2.forward(x));

        // Value stream
        let value = relu(self.dqn_value_fc1.forward(x.clone()));
        let value = self.dqn_value_fc2.forward(value);

        // Advantage stream
        let advantage = relu(self.dqn_advantage_fc1.forward(x));
        let advantage = self.dqn_advantage_fc2.forward(advantage);

        // Combine: Q(s,a) = V(s) + A(s,a) - mean(A)
        let mean_advantage = advantage.clone().mean_dim(1).reshape([batch_size, 1]);
        let q_values = value + advantage - mean_advantage;

        (features, importance, q_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::autodiff::Autodiff;
    use burn::backend::NdArray;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_metis_v2_model_creation() {
        let device = Default::default();
        let model = MetisV2Model::<TestBackend>::new(32, 20, 10, &device);
        assert_eq!(model.state_dim, 32);
        assert_eq!(model.feature_dim, 20);
        assert_eq!(model.action_dim, 10);
    }

    #[test]
    fn test_metis_v2_forward() {
        let device = Default::default();
        let model = MetisV2Model::<TestBackend>::new(32, 20, 10, &device);
        let states = Tensor::<TestBackend, 2>::zeros([4, 32], &device);
        let (features, importance, q_values) = model.forward(states);
        assert_eq!(features.shape().dims, [4, 20]);
        assert_eq!(importance.shape().dims, [4, 1]);
        assert_eq!(q_values.shape().dims, [4, 10]);
    }
}
