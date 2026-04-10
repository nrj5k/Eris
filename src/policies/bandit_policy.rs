//! # Standalone Bandit Policy - Neural Contextual Bandit
//!
//! This module provides a standalone contextual bandit policy for cache tier
//! optimization. Unlike combined approaches (e.g., METIS), this uses only the
//! bandit network for importance scoring and tier selection.
//!
//! ## Architecture
//!
//! The bandit policy consists of:
//! 1. **Feature Extraction Network**: Processes state features
//! 2. **Importance Scoring**: Outputs importance scores in [0, 1]
//! 3. **Tier Mapping**: Converts importance → tier index
//! 4. **Exploration Strategy**: Applies ThompsonSampling/UCB/EGreedy
//!
//! ## Key Differences from DQN
//!
//! | Aspect          | Bandit Policy      | DQN Policy         |
//! |-----------------|--------------------|--------------------|
//! | Learning        | Online (no replay) | Offline (replay)   |
//! | Q-function      | No                 | Yes                |
//! | Target network  | No                 | Yes                |
//! | Memory usage    | Low                | High               |
//! | Training speed  | Fast               | Medium             |
//! | Exploration     | Thompson/UCB       | Epsilon-greedy     |
//!
//! ## Usage Example
//!
//! ### Basic Setup
//!
//! ```rust,ignore
//! use eris::policies::{BanditPolicy, BanditPolicyConfig};
//! use eris::config::BanditConfig;
//! use eris::policies::exploration::ExplorationConfig;
//! use burn::backend::NdArray;
//! use burn::tensor::backend::Backend;
//!
//! // Create bandit network configuration
//! let bandit_config = BanditConfig::builder()
//!     .input_dim(15)              // State dimension
//!     .hidden_layers(vec![64, 128])  // Hidden layers
//!     .feature_dim(20)            // Feature representation size
//!     .build()
//!     .expect("Valid bandit config");
//!
//! // Thompson Sampling works best for bandits
//! let exploration = ExplorationConfig::ThompsonSampling {
//!     prior_mean: 0.0,
//!     prior_std: 1.0,
//! };
//!
//! // Create policy configuration
//! let config = BanditPolicyConfig::new(
//!     bandit_config,
//!     exploration,
//!     0.01,   // Higher learning rate for online learning
//!     5,      // Number of cache tiers
//! );
//!
//! // Initialize policy
//! let device = <NdArray as Backend>::Device::default();
//! let policy = BanditPolicy::new(config, &device);
//! ```
//!
//! ### Action Selection and Importance Scoring
//!
//! ```rust,ignore
//! use eris::policies::policy::{CachePolicy, State};
//!
//! // Create state from features
//! let state = State::Features(vec![
//!     1.0,   // access_frequency
//!     0.5,   // blob_size_normalized
//!     0.8,   // tier_capacity_fraction,
//!     tier_utilization_0,
//!     tier_utilization_1,
//!     tier_utilization_2,
//!     tier_utilization_3,
//!     tier_utilization_4,
//!     hotness_score,
//!     // ... more features
//! ]);
//!
//! // Get importance score (0.0 to 1.0)
//! let importance = policy.get_importance(&state);
//! println!("Importance: {:.3}", importance);
//!
//! // Select action with exploration
//! let action = policy.select_action(&state);
//! match action {
//!     Action::Discrete(idx) => {
//!         let tier = idx / 2;          // 0-4
//!         let operation = idx % 2;      // 0=read, 1=write
//!         println!("Tier: {}, Operation: {}", tier, operation);
//!     }
//! }
//! ```
//!
//! ### Online Learning
//!
//! ```rust,ignore
//! use eris::policies::policy::{CachePolicy, OnlinePolicy, Transition, Action};
//!
//! // Bandit learns from each transition immediately (online learning)
//! for step in 0..1000 {
//!     let state = env.get_state();
//!     let action = policy.select_action(&state);
//!     
//!     // Take action in environment
//!     let (next_state, reward, done) = env.step(&action);
//!     
//!     // Update policy (online learning, no replay buffer)
//!     let transition = Transition {
//!         state: state.clone(),
//!         action,
//!         reward,
//!         next_state,
//!         done,
//!     };
//!     
//!     let loss = policy.update(&transition);
//!     
//!     // Can also adjust learning rate
//!     if step % 100 == 0 {
//!         policy.set_learning_rate(0.01 * (1.0 - step as f32 / 1000.0));
//!     }
//! }
//! ```
//!
//! ### Exploration Strategies
//!
//! ```rust,ignore
//! // Thompson Sampling (recommended for bandits)
//! let thompson = ExplorationConfig::ThompsonSampling {
//!     prior_mean: 0.0,
//!     prior_std: 1.0,  // Higher = more exploration initially
//! };
//!
//! // UCB (theoretically optimal for stationary bandits)
//! let ucb = ExplorationConfig::UCB {
//!     c: 2.0,  // Exploration constant
//! };
//!
//! // Epsilon-Greedy (baseline, but suboptimal for bandits)
//! let egreedy = ExplorationConfig::EpsilonGreedy {
//!     epsilon_start: 1.0,
//!     epsilon_end: 0.01,
//!     epsilon_decay: 0.995,
//! };
//!
//! // Create policies with different strategies
//! let policy_ts = BanditPolicy::new(
//!     BanditPolicyConfig::new(bandit_config.clone(), thompson, 0.01, 5),
//!     &device
//! );
//! ```
//!
//! ## Importance Score Interpretation
//!
//! The bandit network outputs importance scores in [0, 1]:
//! - **0.0**: Least important, place on slowest tier (Tapes)
//! - **0.5**: Moderately important, middle tiers (SSD/HDD)
//! - **1.0**: Most important, place on fastest tier (Memory/NVMe)
//!
//! This is mapped to tiers by:
//! ```text
//! tier_index = floor(importance * num_tiers)
//! ```
//!
//! For example, with 5 tiers and importance = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
//! ```text
//! importance = 0.0  → tier = 0  (Memory)
//! importance = 0.2  → tier = 1  (NVMe)
//! importance = 0.4  → tier = 2  (SSD)
//! importance = 0.6  → tier = 3  (HDD)
//! importance = 0.8  → tier = 4  (Tapes)
//! importance = 1.0  → tier = 4  (Tapes, clamped)
//! ```
//!
//! ## When to Use Bandit vs DQN/METIS
//!
//! **Use Bandit when:**
//! - ✓ You need fast online adaptation
//! - ✓ Memory is constrained
//! - ✓ Real-time tier selection required
//! - ✓ Non-stationary workloads (importance changes over time)
//!
//! **Use DQN when:**
//! - ✓ You want to learn optimal Q-values
//! - ✓ You have enough memory for replay buffer
//! - ✓ Environment is stationary
//!
//! **Use METIS when:**
//! - ✓ You want best overall performance
//! - ✓ You can afford bandit + DQN memory
//! - ✓ You need both feature extraction and Q-learning
//!
//! ## Training Tips
//!
//! 1. **Higher learning rate**: Use 0.01-0.001 (vs 0.0001 for DQN)
//! 2. **Thompson Sampling or UCB**: Better than epsilon-greedy for bandits
//! 3. **Smaller network**: 64-128 hidden units suffice (vs 256+ for DQN)
//! 4. **Monitor importance scores**: Check tier distribution during training
//! 5. **Feature normalization**: Ensure state features are normalized [0, 1]
//!
//! ## Performance Characteristics
//!
//! - **Convergence**: Fast (100-500 episodes)
//! - **Memory**: O(state_dim * hidden) - no replay buffer
//! - **Inference speed**: Fast (single forward pass)
//! - **Sample efficiency**: Lower than DQN/METIS
//! - **Adaptability**: High (online learning)
//!
//! ## References
//!
//! - [Li et al., 2010] - A contextual-bandit approach to personalized news article recommendation
//! - [Agrawal & Goyal, 2013] - Thompson Sampling for contextual bandits

use super::exploration::ExplorationConfig;
use super::policy::{Action, CachePolicy, OnlinePolicy, PolicyType, State, Transition};
use super::tensor_utils::state_to_tensor;
use crate::config::BanditConfig;
use crate::models::ContextualBandit;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::Path;

/// Configuration for BanditPolicy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BanditPolicyConfig {
    /// Bandit network configuration (as serializable format)
    pub bandit_config: BanditConfig,
    /// Exploration strategy configuration
    pub exploration: ExplorationConfig,
    /// Learning rate for online updates
    pub learning_rate: f32,
    /// Number of tiers for action mapping
    pub num_tiers: usize,
}

impl BanditPolicyConfig {
    /// Create a new BanditPolicyConfig
    pub fn new(
        bandit_config: BanditConfig,
        exploration: ExplorationConfig,
        learning_rate: f32,
        num_tiers: usize,
    ) -> Self {
        Self {
            bandit_config,
            exploration,
            learning_rate,
            num_tiers,
        }
    }
}

/// Standalone contextual bandit policy for tier selection
///
/// This policy uses only the bandit network (no DQN) to:
/// 1. Extract features from state
/// 2. Compute importance score for tier selection
/// 3. Select tier based on importance * num_tiers
/// 4. Apply exploration strategy
///
/// # Type Parameters
///
/// * `B` - Burn autodiff backend (e.g., `NdArray`, `Wgpu`)
///
/// # Examples
///
/// ```rust,ignore
/// use eris::policies::{BanditPolicy, BanditPolicyConfig};
/// use eris::policies::policy::{State, Action};
/// use burn::backend::NdArray;
///
/// let device = NdArray::Device::default();
/// let policy = BanditPolicy::<NdArray>::new(config, &device);
///
/// // Select action for a state
/// let state = State::Features(vec![1.0, 2.0, 3.0]);
/// let action = policy.select_action(&state);
///
/// // Update policy with transition
/// let transition = Transition {
///     state: state.clone(),
///     action: action.clone(),
///     reward: 1.0,
///     next_state: state,
///     done: false,
/// };
/// let loss = policy.update(&transition);
/// ```
pub struct BanditPolicy<B: AutodiffBackend> {
    /// Contextual bandit network
    bandit: ContextualBandit<B>,
    /// Exploration strategy
    explorer: Box<dyn super::exploration::ExplorationStrategy<B>>,
    /// Configuration
    config: BanditPolicyConfig,
    /// Device for tensor operations
    device: B::Device,
    /// Learning rate for updates
    learning_rate: f32,
    /// Optimizer configuration
    optimizer_config: AdamConfig,
}

impl<B: AutodiffBackend> BanditPolicy<B> {
    /// Create a new BanditPolicy
    ///
    /// # Arguments
    /// * `config` - Policy configuration
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    /// Initialized BanditPolicy with random weights
    pub fn new(config: BanditPolicyConfig, device: &B::Device) -> Self {
        // Initialize bandit network
        let bandit = config.bandit_config.init(device);

        // Build exploration strategy
        // For standalone bandit, action_dim equals num_tiers
        let explorer = config.exploration.build(config.num_tiers);

        // Initialize optimizer configuration
        let optimizer_config = AdamConfig::new();

        Self {
            bandit,
            explorer,
            config: config.clone(),
            device: device.clone(),
            learning_rate: config.learning_rate,
            optimizer_config,
        }
    }

    /// Forward pass with gradient tracking
    ///
    /// # Arguments
    /// * `state` - Input state tensor [batch_size, state_dim] with gradients
    ///
    /// # Returns
    /// Importance score tensor [batch_size, 1] in range [0, 1]
    pub fn forward_train(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let (_features, importance) = self.bandit.forward(state);
        importance
    }

    // REMOVED: state_to_tensor method - now use tensor_utils::state_to_tensor

    /// Map importance score to tier action
    ///
    /// Converts importance [0, 1] to discrete tier index by scaling.
    ///
    /// # Arguments
    /// * `importance` - Importance score in range [0, 1]
    ///
    /// # Returns
    /// Tier index (0 to num_tiers-1)
    pub fn importance_to_tier(&self, importance: f32) -> usize {
        // Clamp importance to [0, 1]
        let clamped = importance.clamp(0.0, 1.0);

        // Map to tier: importance * num_tiers gives [0, num_tiers]
        // Use floor to get discrete tier
        let tier = (clamped * self.config.num_tiers as f32).floor() as usize;

        // Ensure tier is in valid range [0, num_tiers - 1]
        tier.min(self.config.num_tiers - 1)
    }

    /// Get importance score for a state (inference mode, no gradients)
    ///
    /// # Arguments
    /// * `state` - State to evaluate
    ///
    /// # Returns
    /// Importance score in range [0, 1]
    pub fn get_importance(&self, state: &State) -> f32 {
        let state_tensor =
            state_to_tensor(state, self.config.bandit_config.input_dim, &self.device);

        // Forward pass for inference - detach gradients
        let importance = {
            let (_features, imp) = self.bandit.forward(state_tensor.detach());
            imp
        };

        // Extract scalar importance
        importance
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.5)
    }
}

impl<B: AutodiffBackend> CachePolicy for BanditPolicy<B> {
    fn select_action(&self, state: &State) -> Action {
        // Convert state to tensor
        let state_tensor =
            state_to_tensor(state, self.config.bandit_config.input_dim, &self.device);

        // Get importance score from bandit network (detach for inference)
        let importance = {
            let (_features, imp) = self.bandit.forward(state_tensor.detach());
            imp
        };

        // Create Q-values tensor for exploration strategy
        // For bandit, we use importance scores as pseudo-Q-values
        // Shape: [batch_size=1, action_dim=num_tiers]
        let importance_value = importance
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default()
            .first()
            .copied()
            .unwrap_or(0.5);

        // Create pseudo-Q-values based on distance from each tier's ideal importance
        let mut q_values = Vec::with_capacity(self.config.num_tiers);
        for tier in 0..self.config.num_tiers {
            // Ideal importance for this tier
            let ideal_importance = (tier as f32 + 0.5) / self.config.num_tiers as f32;
            // Q-value is higher when current importance is closer to ideal
            let q_value = 1.0 - (importance_value - ideal_importance).abs();
            q_values.push(q_value);
        }

        let q_tensor_data = TensorData::new(q_values, [1, self.config.num_tiers]);
        let q_tensor = Tensor::from_data(q_tensor_data.convert::<f32>(), &self.device);

        // Use exploration strategy to select tier
        let exploration_action = self
            .explorer
            .select_action(&q_tensor, self.config.num_tiers);

        // Extract the tier from exploration result
        let tier_action: Vec<i32> = exploration_action
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap_or_default();

        let tier = tier_action.first().copied().unwrap_or(0) as usize;

        // Default operation: read (multiply tier by 2)
        // Actions are: tier * 2 for read, tier * 2 + 1 for write
        // We default to read operations in bandit policy
        let action = tier * 2;

        Action::Discrete(action)
    }

    fn update(&mut self, transition: &Transition) -> f32 {
        // Convert state to tensor with gradient tracking
        let state_tensor = state_to_tensor(
            &transition.state,
            self.config.bandit_config.input_dim,
            &self.device,
        );
        let state_with_grad = state_tensor.detach().require_grad();

        // Forward pass through bandit to get importance score
        let importance = self.forward_train(state_with_grad);

        // Compute target importance from reward
        // Map reward [-1, 1] to importance [0, 1]
        // High reward → high importance (important data)
        // Low reward → low importance (unimportant data)
        let target_importance = (transition.reward + 1.0) / 2.0;
        let target_data = TensorData::new(vec![target_importance as f32], [1, 1]);
        let target = Tensor::<B, 2>::from_data(target_data.convert::<f32>(), &self.device);

        // Compute MSE loss: (predicted - target)^2
        let loss = (importance.clone() - target).powf_scalar(2.0).mean();

        // Get loss value before backward pass
        let loss_value = loss.clone().into_data().to_vec::<f32>().unwrap_or_default()[0];

        // Backward pass to compute gradients
        let grads = loss.backward();

        // Convert gradients to parameter gradients
        let grads_params = GradientsParams::from_grads(grads, &self.bandit);

        // Update bandit weights using optimizer
        let learning_rate = self.learning_rate as f64;
        let mut optimizer = self.optimizer_config.init();
        self.bandit = optimizer.step(learning_rate, self.bandit.clone(), grads_params);

        // Update exploration strategy with reward
        // Extract action tier
        if let Action::Discrete(action_idx) = transition.action {
            let tier = action_idx / 2;
            self.explorer.update(tier, transition.reward);
        }

        // Return loss value
        loss_value
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        // For now, save configuration only
        // ContextualBandit doesn't have a direct save method, so we use serde
        let config_json = serde_json::to_string_pretty(&self.config)?;
        fs::write(path.with_extension("json"), config_json)?;

        // Note: Full model save would require using burn's Recorder trait
        // For now, we save the configuration and reinitialize the model on load
        log::info!("BanditPolicy configuration saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        // Load configuration
        let config_path = path.with_extension("json");
        if config_path.exists() {
            let config_json = fs::read_to_string(&config_path)?;
            let loaded_config: BanditPolicyConfig = serde_json::from_str(&config_json)?;
            self.config = loaded_config.clone();

            // Reinitialize bandit with loaded config
            self.bandit = self.config.bandit_config.init(&self.device);
            self.explorer = self.config.exploration.build(self.config.num_tiers);

            log::info!("BanditPolicy configuration loaded from {:?}", path);
        } else {
            log::warn!(
                "BanditPolicy config file not found at {:?}, using current config",
                config_path
            );
        }

        Ok(())
    }

    fn policy_type(&self) -> PolicyType {
        PolicyType::Bandit
    }

    fn action_dim(&self) -> usize {
        // Total actions = num_tiers * 2 (read/write operations)
        self.config.num_tiers * 2
    }
}

impl<B: AutodiffBackend> OnlinePolicy for BanditPolicy<B> {
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Activation;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::backend::Backend;

    type TestBackend = Autodiff<NdArray>;

    fn create_test_config() -> BanditPolicyConfig {
        let bandit_config = BanditConfig::builder()
            .input_dim(15)
            .hidden_layers(vec![64])
            .feature_dim(20)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Valid bandit config");

        BanditPolicyConfig::new(
            bandit_config,
            ExplorationConfig::EpsilonGreedy {
                epsilon_start: 0.5,
                epsilon_end: 0.01,
                epsilon_decay: 0.995,
            },
            0.001,
            5, // num_tiers
        )
    }

    #[test]
    fn test_bandit_policy_creation() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();
        let policy = BanditPolicy::<TestBackend>::new(config, &device);

        assert_eq!(policy.action_dim(), 10); // 5 tiers * 2 operations
        assert_eq!(policy.learning_rate(), 0.001);
    }

    #[test]
    fn test_importance_to_tier_mapping() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();
        let policy = BanditPolicy::<TestBackend>::new(config, &device);

        // Test boundary conditions
        assert_eq!(policy.importance_to_tier(0.0), 0);
        assert_eq!(policy.importance_to_tier(0.1), 0);
        assert_eq!(policy.importance_to_tier(0.2), 1);
        assert_eq!(policy.importance_to_tier(0.5), 2);
        assert_eq!(policy.importance_to_tier(0.8), 4);
        assert_eq!(policy.importance_to_tier(1.0), 4);
    }

    #[test]
    fn test_state_to_tensor_features() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();

        let state = State::Features(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let tensor =
            state_to_tensor::<TestBackend>(&state, config.bandit_config.input_dim, &device);

        let dims = tensor.dims();
        assert_eq!(dims[0], 1); // batch size
        assert!(dims[1] > 0); // state dimension
    }

    #[test]
    fn test_state_to_tensor_raw() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();

        let state = State::Raw(vec![1.0, 2.0, 3.0]);
        let tensor =
            state_to_tensor::<TestBackend>(&state, config.bandit_config.input_dim, &device);

        let dims = tensor.dims();
        assert_eq!(dims[0], 1); // batch size
        assert!(dims[1] > 0); // state dimension
    }

    #[test]
    fn test_state_to_tensor_empty() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();

        let state = State::Empty;
        let tensor =
            state_to_tensor::<TestBackend>(&state, config.bandit_config.input_dim, &device);

        // Should return zeros with input_dim shape
        let dims = tensor.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], config.bandit_config.input_dim);
    }

    #[test]
    fn test_select_action() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();
        let policy = BanditPolicy::<TestBackend>::new(config, &device);

        let state = State::Features(vec![0.5; 15]);
        let action = policy.select_action(&state);

        // Action should be discrete
        match action {
            Action::Discrete(idx) => {
                // Action should be in valid range [0, num_tiers * 2)
                assert!(idx < 10);
            }
            _ => panic!("Expected discrete action"),
        }
    }

    #[test]
    fn test_policy_type() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();
        let policy = BanditPolicy::<TestBackend>::new(config, &device);

        assert_eq!(policy.policy_type(), PolicyType::Bandit);
    }

    #[test]
    fn test_learning_rate_management() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();
        let mut policy = BanditPolicy::<TestBackend>::new(config, &device);

        assert_eq!(policy.learning_rate(), 0.001);

        policy.set_learning_rate(0.01);
        assert_eq!(policy.learning_rate(), 0.01);

        policy.set_learning_rate(0.0001);
        assert_eq!(policy.learning_rate(), 0.0001);
    }

    #[test]
    fn test_importance_extraction() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();
        let policy = BanditPolicy::<TestBackend>::new(config, &device);

        let state = State::Features(vec![0.5; 15]);
        let importance = policy.get_importance(&state);

        // Importance should be in range [0, 1] due to Sigmoid activation
        assert!((0.0..=1.0).contains(&importance));
    }

    #[test]
    fn test_update_returns_loss() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();
        let mut policy = BanditPolicy::<TestBackend>::new(config, &device);

        let state = State::Features(vec![0.5; 15]);
        let transition = Transition {
            state: state.clone(),
            action: Action::Discrete(0), // tier 0, read operation
            reward: 1.0,
            next_state: state,
            done: false,
        };

        let loss = policy.update(&transition);

        // Loss should be non-negative (MSE)
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_forward_pass() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();
        let policy = BanditPolicy::<TestBackend>::new(config, &device);

        // Create batch of states
        let state_data: Vec<f32> = vec![0.5; 15];
        let tensor_data = TensorData::new(state_data, [1, 15]);
        let state_tensor =
            Tensor::<TestBackend, 2>::from_data(tensor_data.convert::<f32>(), &device);

        let importance = policy.forward_train(state_tensor);

        // Importance should have shape [batch_size, 1]
        let dims = importance.dims();
        assert_eq!(dims, [1, 1]);

        // Values should be in [0, 1] range
        let values: Vec<f32> = importance.into_data().to_vec().unwrap_or_default();
        for v in values {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_multiple_tier_selections() {
        let config = create_test_config();
        let device = <TestBackend as Backend>::Device::default();
        let policy = BanditPolicy::<TestBackend>::new(config, &device);

        // Create different states and check actions are valid
        for i in 0..5 {
            let state_vec: Vec<f32> = (0..15).map(|j| (i + j) as f32 / 20.0).collect();
            let state = State::Features(state_vec);
            let action = policy.select_action(&state);

            match action {
                Action::Discrete(idx) => {
                    assert!(idx < 10, "Action {} should be < 10", idx);
                }
                _ => panic!("Expected discrete action"),
            }
        }
    }
}
