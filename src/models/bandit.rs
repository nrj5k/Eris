use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    prelude::*,
};

use crate::training::checkpoint::{CheckpointMetadata, Checkpointable};

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
    ///
    /// # Deprecation Notice
    ///
    /// This config type is deprecated. Use `eris::config::BanditConfig` with the builder pattern instead:
    /// ```rust,ignore
    /// use eris::config::BanditConfig;
    /// use eris::model::Activation;
    /// use eris::training::mock_env::MockEnv;
    ///
    /// // Get dimensions from environment
    /// let env = MockEnv::new_with_dims(100, 50, 20);
    /// let obs_dim = env.observation_space().dim();
    ///
    /// let config = BanditConfig::builder()
    ///     .input_dim(obs_dim)
    ///     .hidden_layers(vec![64, 128])
    ///     .feature_dim(20)
    ///     .activation(Activation::Sigmoid)
    ///     .build()?;
    /// ```
    #[deprecated(
        since = "0.2.0",
        note = "Use `eris::config::BanditConfig` with builder pattern instead"
    )]
    pub fn init<B: Backend>(&self, device: &B::Device) -> ContextualBandit<B> {
        log::warn!(
            "ContextualBanditConfig is deprecated. Use eris::config::BanditConfig with builder pattern"
        );

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

impl<B: Backend> Checkpointable<B> for ContextualBandit<B> {
    fn checkpoint_name(&self) -> &str {
        "contextual_bandit"
    }

    fn checkpoint_metadata(&self) -> CheckpointMetadata {
        // Get dimensions from the FC layer weights
        // fc1.weight shape: [hidden_dim, state_dim]
        let state_dim = self.fc1.weight.shape().dims[1];
        // feature_head.weight shape: [feature_dim, hidden_dim * 2]
        let feature_dim = self.feature_head.weight.shape().dims[0];

        CheckpointMetadata::new_with_dims(
            "ContextualBandit".to_string(),
            0, // epoch - will be updated by training loop
            state_dim,
            1, // action_dim = 1 (importance score output)
            feature_dim,
        )
    }

    fn model(&self) -> &impl Module<B> {
        self
    }
}
