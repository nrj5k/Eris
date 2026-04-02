use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    prelude::*,
};

/// Contextual Bandit network for tier selection
///
/// Takes state features and outputs:
/// 1. Enhanced features for the Q-network (feature_dim)
/// 2. Importance score for tier selection [0, 1]
///
/// The importance score is used to select which tier to use for a blob,
/// while the enhanced features are used by the Q-network to determine
/// the Q-values for actions.
#[derive(Module, Debug)]
pub struct ContextualBandit<B: Backend> {
    // Feature extraction layers
    fc1: Linear<B>, // state_dim -> hidden_dim
    fc2: Linear<B>, // hidden_dim -> hidden_dim * 2

    // Output heads
    feature_head: Linear<B>, // hidden_dim * 2 -> feature_dim
    score_head: Linear<B>,   // hidden_dim * 2 -> 1

    activation: Relu,
    sigmoid: Sigmoid,
}

#[derive(Config, Debug)]
pub struct ContextualBanditConfig {
    /// Input state dimension
    pub state_dim: usize,
    /// Hidden layer dimension (fc1)
    pub hidden_dim: usize,
    /// Output feature dimension
    pub feature_dim: usize,
    /// Whether to include bias in linear layers
    #[config(default = true)]
    pub bias: bool,
}

impl ContextualBanditConfig {
    /// Initialize the Contextual Bandit network
    ///
    /// # Arguments
    /// * `device` - Device to initialize the network on
    ///
    /// # Returns
    /// Initialized ContextualBandit with random weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> ContextualBandit<B> {
        ContextualBandit {
            fc1: LinearConfig::new(self.state_dim, self.hidden_dim)
                .with_bias(self.bias)
                .init(device),
            fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim * 2)
                .with_bias(self.bias)
                .init(device),

            feature_head: LinearConfig::new(self.hidden_dim * 2, self.feature_dim)
                .with_bias(self.bias)
                .init(device),
            score_head: LinearConfig::new(self.hidden_dim * 2, 1)
                .with_bias(self.bias)
                .init(device),

            activation: Relu::new(),
            sigmoid: Sigmoid::new(),
        }
    }
}

impl<B: Backend> ContextualBandit<B> {
    /// Forward pass for contextual bandit
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, state_dim]
    ///
    /// # Returns
    /// * (features, importance_score) where:
    ///   - features: [batch_size, feature_dim] - Enhanced features for Q-network
    ///   - importance_score: [batch_size, 1] - Importance score in [0, 1]
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Feature extraction
        let x = self.activation.forward(self.fc1.forward(x));
        let x = self.activation.forward(self.fc2.forward(x));

        // Feature output (for Q-network input)
        let features = self.feature_head.forward(x.clone());

        // Importance score output (for tier selector)
        let score = self.sigmoid.forward(self.score_head.forward(x));

        (features, score)
    }
}
