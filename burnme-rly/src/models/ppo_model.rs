//! PPO (Proximal Policy Optimization) Model
//!
//! Two-headed neural network:
//! - Policy head: outputs action logits (action_dim values per state)
//! - Value head: outputs state value estimate (1 value per state)

use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{log_softmax, relu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

#[derive(Clone, Debug)]
pub struct PpoModelConfig {
    pub state_dim: usize,
    pub hidden_layers: Vec<usize>,
    pub action_dim: usize,
}

impl Default for PpoModelConfig {
    fn default() -> Self {
        Self {
            state_dim: 32,
            hidden_layers: vec![64],
            action_dim: 10,
        }
    }
}

impl PpoModelConfig {
    pub fn new(state_dim: usize, hidden_layers: Vec<usize>, action_dim: usize) -> Self {
        Self {
            state_dim,
            hidden_layers,
            action_dim,
        }
    }

    pub fn with_state_dim(mut self, dim: usize) -> Self {
        self.state_dim = dim;
        self
    }

    pub fn with_hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = layers;
        self
    }

    pub fn with_action_dim(mut self, dim: usize) -> Self {
        self.action_dim = dim;
        self
    }
}

#[derive(Module, Debug)]
pub struct PpoModel<B: Backend> {
    shared_layers: Vec<Linear<B>>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: Backend> PpoModel<B> {
    pub fn new(config: PpoModelConfig, device: &B::Device) -> Self {
        let mut shared_layers = Vec::new();
        let mut prev_dim = config.state_dim;
        for &hidden_dim in &config.hidden_layers {
            shared_layers.push(LinearConfig::new(prev_dim, hidden_dim).init(device));
            prev_dim = hidden_dim;
        }

        let policy_head = LinearConfig::new(prev_dim, config.action_dim).init(device);
        let value_head = LinearConfig::new(prev_dim, 1).init(device);

        Self {
            shared_layers,
            policy_head,
            value_head,
        }
    }

    /// Forward: returns (logits, value)
    /// - logits: [batch, action_dim] - action probabilities before softmax
    /// - value: [batch, 1] - state value estimate
    pub fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = input;
        for layer in &self.shared_layers {
            x = relu(layer.forward(x));
        }

        let logits = self.policy_head.forward(x.clone());
        let value = self.value_head.forward(x);

        (logits, value)
    }

    /// Evaluate actions: returns (log_probs, values, entropy)
    ///
    /// # Arguments
    /// * `states` - Batch of states [batch, state_dim]
    /// * `actions` - Taken actions [batch] (integer tensor)
    ///
    /// # Returns
    /// Tuple of:
    /// - log_probs: [batch] - log probability of taken actions
    /// - values: [batch] - state value estimates
    /// - entropy: [batch] - policy entropy for each state
    pub fn evaluate_actions(
        &self,
        states: Tensor<B, 2>,
        actions: Tensor<B, 1, Int>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let (logits, values) = self.forward(states);

        // Compute log_probs for taken actions
        let log_probs_full = log_softmax(logits.clone(), 1); // [batch, action_dim]
        let batch_size = actions.shape().dims[0];
        let actions_2d = actions.reshape([batch_size, 1]);
        let log_probs: Tensor<B, 1> = log_probs_full.clone().gather(1, actions_2d).squeeze();

        // Compute entropy: -sum(p * log_p)
        let probs = softmax(logits, 1);
        let entropy: Tensor<B, 1> = -(probs * log_probs_full).sum_dim(1).squeeze();

        // Squeeze values from [batch, 1] to [batch]
        let values_squeezed: Tensor<B, 1> = values.squeeze();

        (log_probs, values_squeezed, entropy)
    }

    /// Select action using greedy selection (no exploration)
    pub fn select_action(&self, state: Tensor<B, 2>) -> usize {
        let (logits, _) = self.forward(state);
        let best_action = logits.argmax(1);
        let action_data = best_action.into_data().convert::<i32>();
        action_data
            .as_slice::<i32>()
            .map(|s| s.first().copied().unwrap_or(0) as usize)
            .unwrap_or(0)
    }

    /// Select actions for a batch of states using greedy selection
    pub fn select_action_batched(&self, states: Tensor<B, 2>) -> Vec<usize> {
        let (logits, _) = self.forward(states);
        let best_actions = logits.argmax(1);
        let best_actions_data: Vec<i32> = best_actions
            .into_data()
            .convert::<i32>()
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();

        best_actions_data.into_iter().map(|a| a as usize).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_ppo_model_creation() {
        let device = Default::default();
        let config = PpoModelConfig::new(32, vec![64], 10);
        let model: PpoModel<NdArray> = PpoModel::new(config, &device);
        let (logits, value) = model.forward(Tensor::zeros([1, 32], &device));
        assert_eq!(logits.dims(), [1, 10]);
        assert_eq!(value.dims(), [1, 1]);
    }

    #[test]
    fn test_ppo_model_batch() {
        let device = Default::default();
        let config = PpoModelConfig::new(16, vec![32, 64], 5);
        let model: PpoModel<NdArray> = PpoModel::new(config, &device);
        let (logits, value) = model.forward(Tensor::ones([4, 16], &device));
        assert_eq!(logits.dims(), [4, 5]);
        assert_eq!(value.dims(), [4, 1]);
    }

    #[test]
    fn test_evaluate_actions() {
        let device = Default::default();
        let config = PpoModelConfig::new(10, vec![16], 4);
        let model: PpoModel<NdArray> = PpoModel::new(config, &device);
        let states = Tensor::ones([3, 10], &device);
        let actions = Tensor::from_data([0, 1, 2], &device);
        let (log_probs, values, entropy) = model.evaluate_actions(states, actions);
        assert_eq!(log_probs.dims(), [3]);
        assert_eq!(values.dims(), [3]);
        assert_eq!(entropy.dims(), [3]);
    }

    #[test]
    fn test_select_action() {
        let device = Default::default();
        let config = PpoModelConfig::new(10, vec![16], 4);
        let model: PpoModel<NdArray> = PpoModel::new(config, &device);
        let action = model.select_action(Tensor::ones([1, 10], &device));
        assert!(action < 4);
    }

    #[test]
    fn test_select_action_batched() {
        let device = Default::default();
        let config = PpoModelConfig::new(10, vec![16], 4);
        let model: PpoModel<NdArray> = PpoModel::new(config, &device);
        let states = Tensor::ones([5, 10], &device);
        let actions = model.select_action_batched(states);
        assert_eq!(actions.len(), 5);
        for action in actions {
            assert!(action < 4);
        }
    }

    #[test]
    fn test_ppo_config_builder_pattern() {
        let config = PpoModelConfig::default()
            .with_state_dim(64)
            .with_hidden_layers(vec![128, 64])
            .with_action_dim(8);

        assert_eq!(config.state_dim, 64);
        assert_eq!(config.hidden_layers, vec![128, 64]);
        assert_eq!(config.action_dim, 8);
    }
}
