//! Metis: Combined DQN + Contextual Bandit Policy
//!
//! Refactored to implement CachePolicy trait using existing CombinedModel infrastructure

use super::exploration::{ExplorationConfig, ExplorationStrategy};
use super::policy::*;
use super::tensor_utils::{batch_to_tensors, state_to_tensor};
use crate::config::CombinedBanditDQNConfig;
use crate::models::CombinedModel;
use crate::training::ring_buffer::RingBuffer;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor};
use std::error::Error;
use std::path::Path;

/// Metis Policy - Combined DQN + Bandit using existing model infrastructure
pub struct MetisPolicy<B: AutodiffBackend> {
    /// Policy network (online network)
    model: CombinedModel<B>,
    /// Target network (frozen copy)
    target_model: CombinedModel<B>,
    /// Experience replay buffer
    buffer: RingBuffer,
    /// Exploration strategy
    explorer: Box<dyn ExplorationStrategy<B>>,
    /// Configuration
    config: MetisConfig,
    /// Current device
    device: B::Device,
    /// Training step counter
    step_count: usize,
}

/// Configuration for Metis
#[derive(Clone, Debug)]
pub struct MetisConfig {
    /// Input state dimension
    pub state_dim: usize,
    /// Feature dimension (bandit output)
    pub feature_dim: usize,
    /// Action dimension (number of tiers * 2 for read/write ops)
    pub action_dim: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Discount factor
    pub gamma: f32,
    /// Initial exploration rate (deprecated, use exploration config)
    pub epsilon_start: f32,
    /// Final exploration rate (deprecated, use exploration config)
    pub epsilon_end: f32,
    /// Exploration decay rate (deprecated, use exploration config)
    pub epsilon_decay: f32,
    /// Target network update frequency
    pub target_update_freq: usize,
    /// Batch size
    pub batch_size: usize,
    /// Replay buffer capacity
    pub buffer_capacity: usize,
    /// Exploration strategy configuration
    pub exploration: ExplorationConfig,
}

impl Default for MetisConfig {
    fn default() -> Self {
        Self {
            state_dim: 32,   // Warp-aligned dimension (5 tier sizes + 10 features + 17 padding)
            feature_dim: 20, // Bandit output dimension
            action_dim: 10,  // Number of actions (5 tiers * 2 ops)
            learning_rate: 0.0001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            target_update_freq: 1000,
            batch_size: 2048, // Optimized for GPU utilization (multiple of 32 for warp alignment)
            buffer_capacity: 10_000,
            exploration: ExplorationConfig::EpsilonGreedy {
                epsilon_start: 1.0,
                epsilon_end: 0.01,
                epsilon_decay: 0.995,
            },
        }
    }
}

impl<B: AutodiffBackend> MetisPolicy<B> {
    /// Create new MetisPolicy
    ///
    /// # Arguments
    /// * `config` - Metis configuration
    /// * `model_config` - Combined model architecture configuration
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Initialized policy with random weights and empty replay buffer
    pub fn new(
        config: MetisConfig,
        model_config: CombinedBanditDQNConfig,
        device: B::Device,
    ) -> Self {
        let model = model_config.init(&device);
        let target_model = model_config.init(&device);
        let buffer = RingBuffer::new(config.buffer_capacity);

        // Build exploration strategy from config
        let explorer = config.exploration.build(config.action_dim);

        Self {
            model,
            target_model,
            buffer,
            explorer,
            config,
            device,
            step_count: 0,
        }
    }

    /// Create with epsilon-greedy (backward compatible)
    ///
    /// This method provides backward compatibility with the original METIS implementation
    /// that used hardcoded epsilon-greedy exploration.
    ///
    /// # Arguments
    /// * `config` - Metis configuration
    /// * `model_config` - Combined model architecture configuration
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Initialized policy with epsilon-greedy exploration
    pub fn new_epsilon_greedy(
        config: MetisConfig,
        model_config: CombinedBanditDQNConfig,
        device: B::Device,
    ) -> Self {
        let mut config = config;
        config.exploration = ExplorationConfig::EpsilonGreedy {
            epsilon_start: config.epsilon_start,
            epsilon_end: config.epsilon_end,
            epsilon_decay: config.epsilon_decay,
        };
        Self::new(config, model_config, device)
    }

    /// Forward pass through both bandit and DQN
    ///
    /// # Arguments
    /// * `state` - Input tensor [batch_size, state_dim]
    ///
    /// # Returns
    /// * (features, importance, q_values)
    fn forward(&self, state: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        self.model.forward(state)
    }

    /// Select action using exploration strategy with combined bandit + DQN
    ///
    /// # Arguments
    /// * `state` - Current state
    ///
    /// # Returns
    /// * Action (discrete action index)
    pub fn select_action(&self, state: &State) -> Action {
        // Convert state to tensor
        let state_tensor = state_to_tensor(state, self.config.state_dim, &self.device);

        // Forward pass: get importance and Q-values
        let (_features, importance, q_values) = self.forward(state_tensor);

        // Get importance score as scalar
        let importance_val: f32 = importance
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert importance")[0];

        // Map importance [0, 1] to tier selection
        // Lower importance -> lower tier (cold storage)
        // Higher importance -> higher tier (hot storage)
        let importance_scaled: f32 = importance_val * 5.0;
        let tier_idx = importance_scaled.min(4.0) as usize;

        // Get Q-values for this tier's actions (2 actions per tier: read/write)
        let tier_start = tier_idx * 2;
        let tier_q_values = q_values.slice([0..1, tier_start..tier_start + 2]);

        // Use exploration strategy to select action within tier
        let action_tensor = self.explorer.select_action(&tier_q_values, 2);

        let action_idx = action_tensor
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("Failed to convert action")[0] as usize;

        // Encode action: tier * 2 + operation
        let action = tier_idx * 2 + action_idx;

        Action::Discrete(action)
    }

    /// DQN training step with Double DQN
    ///
    /// # Arguments
    /// * `batch` - Batch of transitions from replay buffer
    ///
    /// # Returns
    /// * Loss value
    fn train_dqn_step(
        &mut self,
        states: &Tensor<B, 2>,
        actions: &Tensor<B, 2, Int>,
        rewards: &Tensor<B, 2>,
        next_states: &Tensor<B, 2>,
        dones: &Tensor<B, 2>,
    ) -> f32 {
        // Current Q values
        let (_, _, q_values) = self.model.forward(states.clone());

        // Gather Q values for taken actions
        let current_q = q_values.gather(1, actions.clone());

        // Double DQN: Use policy network to select actions
        let (_, _, next_q_policy) = self.model.forward(next_states.clone());
        let best_actions = next_q_policy.argmax(1);

        // Use target network to evaluate actions
        let (_, _, next_q_target) = self.target_model.forward(next_states.clone());
        let max_next_q = next_q_target.gather(1, best_actions);

        // Compute target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        let gamma_val: f32 = self.config.gamma.into();
        let target_q = rewards.clone()
            + Tensor::full([1], gamma_val, &self.device)
                * max_next_q
                * (Tensor::ones_like(dones) - dones.clone());

        // MSE loss - need to broadcast correctly
        let diff = current_q - target_q.detach();
        let squared = diff.powf(Tensor::full([1], 2.0_f32, &self.device));
        let loss = squared.mean();

        loss.into_data().convert::<f32>().as_slice().unwrap()[0]
    }

    /// Update target network
    fn update_target_network(&mut self) {
        self.target_model = self.model.clone();
    }
}

impl<B: AutodiffBackend> CachePolicy for MetisPolicy<B> {
    fn select_action(&self, state: &State) -> Action {
        self.select_action(state)
    }

    fn update(&mut self, _transition: &Transition) -> f32 {
        // Updates happen in train_step with batches
        // This method is for online policies
        0.0
    }

    fn save(&self, _path: &Path) -> Result<(), Box<dyn Error>> {
        // Save model using Burn's recorder
        // TODO: Implement using ModelRecorder
        Err("Save not yet implemented for MetisPolicy".into())
    }

    fn load(&mut self, _path: &Path) -> Result<(), Box<dyn Error>> {
        // Load model using Burn's recorder
        // TODO: Implement using ModelRecorder
        Err("Load not yet implemented for MetisPolicy".into())
    }

    fn policy_type(&self) -> PolicyType {
        PolicyType::Metis
    }

    fn action_dim(&self) -> usize {
        self.config.action_dim
    }
}

impl<B: AutodiffBackend> ReplayPolicy for MetisPolicy<B> {
    fn train_step(&mut self, batch: &[Transition]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        // Extract batch components using utility
        let (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor) =
            batch_to_tensors(batch, self.config.state_dim, &self.device);

        // Train DQN
        let loss = self.train_dqn_step(
            &states_tensor,
            &actions_tensor,
            &rewards_tensor,
            &next_states_tensor,
            &dones_tensor,
        );

        // Decay exploration parameters
        self.explorer.decay();

        // Update target network periodically
        self.step_count += 1;
        if self.step_count % self.config.target_update_freq == 0 {
            self.update_target_network();
        }

        loss
    }

    fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    fn update_target(&mut self) {
        self.update_target_network();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn test_metis_config_default() {
        let config = MetisConfig::default();
        assert_eq!(
            config.state_dim, 32,
            "Default state_dim should be warp-aligned (32)"
        );
        assert_eq!(config.action_dim, 10);
        assert_eq!(config.epsilon_start, 1.0);

        // Verify exploration config is set
        match config.exploration {
            ExplorationConfig::EpsilonGreedy { epsilon_start, .. } => {
                assert_eq!(epsilon_start, 1.0);
            }
            _ => panic!("Expected EpsilonGreedy exploration config"),
        }
    }

    #[test]
    fn test_state_to_tensor() {
        type TestBackend = Autodiff<NdArray>;
        let device = <NdArray as burn::prelude::Backend>::Device::default();
        let config = MetisConfig::default();

        // Test state_to_tensor utility directly
        let state = State::Features(vec![1.0; 32]);
        let tensor = state_to_tensor::<TestBackend>(&state, config.state_dim, &device);

        assert_eq!(tensor.shape().dims, [1, 32]);
    }

    #[test]
    fn test_select_action_exploration() {
        type TestBackend = Autodiff<NdArray>;
        let device = <NdArray as burn::prelude::Backend>::Device::default();
        let mut config = MetisConfig::default();

        // Use epsilon-greedy with high exploration
        config.exploration = ExplorationConfig::EpsilonGreedy {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        #[allow(deprecated)]
        let model_config = crate::config::CombinedBanditDQNConfig::builder()
            .bandit(
                crate::config::BanditConfig::builder()
                    .input_dim(config.state_dim)
                    .hidden_layers(vec![64])
                    .feature_dim(config.feature_dim)
                    .build()
                    .expect("Valid bandit config"),
            )
            .dqn(
                crate::config::DQNConfig::builder()
                    .input_dim(config.feature_dim)
                    .hidden_layers(vec![128])
                    .action_dim(config.action_dim)
                    .build()
                    .expect("Valid DQN config"),
            )
            .build()
            .expect("Valid combined config");

        let policy = MetisPolicy::<TestBackend>::new(config, model_config, device);

        // With epsilon = 1.0, should explore (random actions)
        let state = State::Features(vec![1.0; 32]);
        let action = policy.select_action(&state);

        match action {
            Action::Discrete(a) => assert!(a < 10),
            _ => panic!("Expected discrete action"),
        }
    }

    #[test]
    fn test_policy_type() {
        type TestBackend = Autodiff<NdArray>;
        let device = <NdArray as burn::prelude::Backend>::Device::default();
        let config = MetisConfig::default();

        #[allow(deprecated)]
        let model_config = crate::config::CombinedBanditDQNConfig::builder()
            .bandit(
                crate::config::BanditConfig::builder()
                    .input_dim(config.state_dim)
                    .hidden_layers(vec![64])
                    .feature_dim(config.feature_dim)
                    .build()
                    .expect("Valid bandit config"),
            )
            .dqn(
                crate::config::DQNConfig::builder()
                    .input_dim(config.feature_dim)
                    .hidden_layers(vec![128])
                    .action_dim(config.action_dim)
                    .build()
                    .expect("Valid DQN config"),
            )
            .build()
            .expect("Valid combined config");

        let policy = MetisPolicy::<TestBackend>::new(config, model_config, device);

        assert_eq!(policy.policy_type(), PolicyType::Metis);
        assert_eq!(policy.action_dim(), 10);
    }

    #[test]
    fn test_new_epsilon_greedy_backward_compat() {
        type TestBackend = Autodiff<NdArray>;
        let device = <NdArray as burn::prelude::Backend>::Device::default();
        let config = MetisConfig::default();

        #[allow(deprecated)]
        let model_config = crate::config::CombinedBanditDQNConfig::builder()
            .bandit(
                crate::config::BanditConfig::builder()
                    .input_dim(config.state_dim)
                    .hidden_layers(vec![64])
                    .feature_dim(config.feature_dim)
                    .build()
                    .expect("Valid bandit config"),
            )
            .dqn(
                crate::config::DQNConfig::builder()
                    .input_dim(config.feature_dim)
                    .hidden_layers(vec![128])
                    .action_dim(config.action_dim)
                    .build()
                    .expect("Valid DQN config"),
            )
            .build()
            .expect("Valid combined config");

        // Test backward compatibility method
        let policy = MetisPolicy::<TestBackend>::new_epsilon_greedy(config, model_config, device);

        assert_eq!(policy.policy_type(), PolicyType::Metis);
        assert_eq!(policy.action_dim(), 10);

        // Verify exploration parameter
        let param = policy.explorer.get_param();
        assert!((param - 1.0).abs() < 1e-6); // epsilon_start should be 1.0
    }
}
