use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

/// Q-Network with dueling architecture
///
/// This implements a dueling DQN architecture that separates the estimation of:
/// - V(s): Value of being in state s
/// - A(s, a): Advantage of taking action a in state s
///
/// The final Q-values are: Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
///
/// This architecture helps with value function approximation by allowing
/// the network to learn the value of states independently from the advantage
/// of actions.
#[derive(Module, Debug)]
pub struct QNetwork<B: Backend> {
    // Shared feature extraction layers
    fc1: Linear<B>,
    fc2: Linear<B>,

    // Value stream: V(s)
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,

    // Advantage stream: A(s, a)
    advantage_fc1: Linear<B>,
    advantage_fc2: Linear<B>,

    activation: Relu,
}

#[derive(Config, Debug)]
pub struct QNetworkConfig {
    /// Input dimension (features from bandit)
    pub input_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Action dimension (5 tiers × 2 ops = 10 actions)
    pub action_dim: usize,
    /// Whether to include bias in linear layers
    #[config(default = true)]
    pub bias: bool,
}

impl QNetworkConfig {
    /// Initialize the Q-Network
    ///
    /// # Arguments
    /// * `device` - Device to initialize the network on
    ///
    /// # Returns
    /// Initialized QNetwork with random weights
    pub fn init<B: Backend>(&self, device: &B::Device) -> QNetwork<B> {
        QNetwork {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim)
                .with_bias(self.bias)
                .init(device),
            fc2: LinearConfig::new(self.hidden_dim, self.hidden_dim)
                .with_bias(self.bias)
                .init(device),

            value_fc1: LinearConfig::new(self.hidden_dim, self.hidden_dim)
                .with_bias(self.bias)
                .init(device),
            value_fc2: LinearConfig::new(self.hidden_dim, 1)
                .with_bias(self.bias)
                .init(device),

            advantage_fc1: LinearConfig::new(self.hidden_dim, self.hidden_dim)
                .with_bias(self.bias)
                .init(device),
            advantage_fc2: LinearConfig::new(self.hidden_dim, self.action_dim)
                .with_bias(self.bias)
                .init(device),

            activation: Relu::new(),
        }
    }
}

impl<B: Backend> QNetwork<B> {
    /// Forward pass for Q-network with dueling architecture
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    /// * Q-values of shape [batch_size, action_dim]
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Shared feature extraction
        let x = self.activation.forward(self.fc1.forward(x));
        let x = self.activation.forward(self.fc2.forward(x));

        // Value stream: V(s)
        let value = self.activation.forward(self.value_fc1.forward(x.clone()));
        let value = self.value_fc2.forward(value); // [batch, 1]

        // Advantage stream: A(s, a)
        let advantage = self.activation.forward(self.advantage_fc1.forward(x));
        let advantage = self.advantage_fc2.forward(advantage); // [batch, action_dim]

        // Combine: Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
        // Broadcasting: value [batch, 1] + advantage [batch, action_dim] - mean_advantage [batch, 1]
        let mean_advantage = advantage.clone().mean_dim(1);
        let q_values = value + (advantage - mean_advantage);

        q_values
    }
}
