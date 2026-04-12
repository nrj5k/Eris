//! Combined DQN + Contextual Bandit Model
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::RngExt;

#[derive(Clone, Debug)]
pub struct CombinedModelConfig {
    pub state_dim: usize,
    pub bandit_hidden: Vec<usize>,
    pub feature_dim: usize,
    pub dqn_hidden: Vec<usize>,
    pub action_dim: usize,
}

impl Default for CombinedModelConfig {
    fn default() -> Self {
        Self {
            state_dim: 32,
            bandit_hidden: vec![64],
            feature_dim: 20,
            dqn_hidden: vec![128],
            action_dim: 10,
        }
    }
}

impl CombinedModelConfig {
    pub fn new(
        state_dim: usize,
        bandit_hidden: Vec<usize>,
        feature_dim: usize,
        dqn_hidden: Vec<usize>,
        action_dim: usize,
    ) -> Self {
        Self {
            state_dim,
            bandit_hidden,
            feature_dim,
            dqn_hidden,
            action_dim,
        }
    }

    pub fn with_state_dim(mut self, dim: usize) -> Self {
        self.state_dim = dim;
        self
    }
    pub fn with_bandit_hidden(mut self, hidden: Vec<usize>) -> Self {
        self.bandit_hidden = hidden;
        self
    }
    pub fn with_feature_dim(mut self, dim: usize) -> Self {
        self.feature_dim = dim;
        self
    }
    pub fn with_dqn_hidden(mut self, hidden: Vec<usize>) -> Self {
        self.dqn_hidden = hidden;
        self
    }
    pub fn with_action_dim(mut self, dim: usize) -> Self {
        self.action_dim = dim;
        self
    }
}

#[derive(Module, Debug)]
pub struct CombinedModel<B: Backend> {
    bandit_layers: Vec<Linear<B>>,
    bandit_output: Linear<B>,
    dqn_layers: Vec<Linear<B>>,
    dqn_output: Linear<B>,
    feature_dim: usize,
    action_dim: usize,
}

impl<B: Backend> CombinedModel<B> {
    pub fn new(config: CombinedModelConfig, device: &B::Device) -> Self {
        let mut bandit_layers = Vec::new();
        let mut prev_dim = config.state_dim;
        for &h in &config.bandit_hidden {
            bandit_layers.push(LinearConfig::new(prev_dim, h).init(device));
            prev_dim = h;
        }
        let bandit_output = LinearConfig::new(prev_dim, config.feature_dim).init(device);
        let mut dqn_layers = Vec::new();
        prev_dim = config.feature_dim;
        for &h in &config.dqn_hidden {
            dqn_layers.push(LinearConfig::new(prev_dim, h).init(device));
            prev_dim = h;
        }
        let dqn_output = LinearConfig::new(prev_dim, config.action_dim).init(device);
        Self {
            bandit_layers,
            bandit_output,
            dqn_layers,
            dqn_output,
            feature_dim: config.feature_dim,
            action_dim: config.action_dim,
        }
    }

    /// Forward: (features, importance, q_values)
    pub fn forward(&self, states: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let mut x = states;
        for layer in &self.bandit_layers {
            x = relu(layer.forward(x));
        }
        let features = self.bandit_output.forward(x);
        // Tanh maps [-inf, +inf] to [-1, 1], then scale to [0, 1]
        // Better gradient flow near boundaries
        let importance = (features.clone().mean_dim(1).tanh() * 0.5) + 0.5;
        let mut q = features.clone();
        for layer in &self.dqn_layers {
            q = relu(layer.forward(q));
        }
        let q_values = self.dqn_output.forward(q);
        (features, importance, q_values)
    }

    pub fn select_action(&self, state: Tensor<B, 2>, epsilon: f32) -> usize {
        let (_, _, q_values) = self.forward(state);
        let mut rng = rand::rng();
        if rng.random::<f32>() < epsilon {
            rng.random_range(0..self.action_dim)
        } else {
            let best_action = q_values.argmax(1);
            let action_data = best_action.into_data().convert::<i32>();
            action_data
                .as_slice::<i32>()
                .map(|s| s.first().copied().unwrap_or(0) as usize)
                .unwrap_or(0)
        }
    }

    /// Select actions for a batch of states (efficient - single GPU sync)
    ///
    /// # Arguments
    /// * `states` - Batch of states \[batch_size, state_dim\]
    /// * `epsilon` - Exploration rate
    ///
    /// # Returns
    /// Vector of selected actions
    pub fn select_action_batched(&self, states: Tensor<B, 2>, epsilon: f32) -> Vec<usize> {
        let batch_size = states.dims()[0];
        let (_, _, q_values) = self.forward(states);
        let best_actions = q_values.argmax(1);
        let best_actions_data: Vec<i32> = best_actions
            .into_data()
            .convert::<i32>()
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let mut rng = rand::rng();
        let mut actions = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            if rng.random::<f32>() < epsilon {
                actions.push(rng.random_range(0..self.action_dim));
            } else {
                let action = best_actions_data.get(i).copied().unwrap_or(0) as usize;
                actions.push(action);
            }
        }
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    #[test]
    fn test_combined_model_creation() {
        let device = Default::default();
        let config = CombinedModelConfig::new(32, vec![64], 20, vec![128], 10);
        let model: CombinedModel<NdArray> = CombinedModel::new(config, &device);
        let (f, i, q) = model.forward(Tensor::zeros([1, 32], &device));
        assert_eq!(f.dims(), [1, 20]);
        assert_eq!(i.dims(), [1, 1]);
        assert_eq!(q.dims(), [1, 10]);
    }
    #[test]
    fn test_combined_model_batch() {
        let device = Default::default();
        let config = CombinedModelConfig::new(16, vec![32], 8, vec![64], 5);
        let model: CombinedModel<NdArray> = CombinedModel::new(config, &device);
        let (f, i, q) = model.forward(Tensor::ones([4, 16], &device));
        assert_eq!(f.dims(), [4, 8]);
        assert_eq!(i.dims(), [4, 1]);
        assert_eq!(q.dims(), [4, 5]);
    }
    #[test]
    fn test_select_action() {
        let device = Default::default();
        let config = CombinedModelConfig::new(10, vec![16], 8, vec![32], 4);
        let model: CombinedModel<NdArray> = CombinedModel::new(config, &device);
        let action = model.select_action(Tensor::ones([1, 10], &device), 0.0);
        assert!(action < 4);
    }
    #[test]
    fn test_select_action_batched() {
        let device = Default::default();
        let config = CombinedModelConfig::new(10, vec![16], 8, vec![32], 4);
        let model: CombinedModel<NdArray> = CombinedModel::new(config, &device);
        let states = Tensor::ones([5, 10], &device);
        let actions = model.select_action_batched(states, 0.0);
        assert_eq!(actions.len(), 5);
        for action in actions {
            assert!(action < 4);
        }
    }

    #[test]
    fn test_combined_config_builder_pattern() {
        let config = CombinedModelConfig::default()
            .with_state_dim(64)
            .with_bandit_hidden(vec![128, 64])
            .with_feature_dim(32)
            .with_dqn_hidden(vec![256, 128])
            .with_action_dim(8);

        assert_eq!(config.state_dim, 64);
        assert_eq!(config.bandit_hidden, vec![128, 64]);
        assert_eq!(config.feature_dim, 32);
        assert_eq!(config.dqn_hidden, vec![256, 128]);
        assert_eq!(config.action_dim, 8);
    }
}
