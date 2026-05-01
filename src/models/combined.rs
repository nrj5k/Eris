//! Combined model for Metis (baseline version).
//!
//! This is the **metis-baseline** model — a hardcoded chain of
//! ContextualBandit → QNetwork with end-to-end gradient flow.
//!
//! For the newer generic model using SequentialCompose (stop-gradient,
//! independent training), see `MetisModel` below.

use burn::{config::Config, module::Module};

use burn::tensor::{backend::Backend, Tensor};

use crate::models::{
    compose_adapter::{BanditAdapter, DQNAdapter},
    ContextualBandit, ContextualBanditConfig, QNetwork, QNetworkConfig,
};
use crate::tier::TierSelector;
use crate::training::checkpoint::{CheckpointMetadata, Checkpointable};

/// Combined model integrating contextual bandit and Q-network
///
/// This model combines:
/// 1. Contextual bandit - Extracts enhanced features and computes importance scores
/// 2. Q-network - Predicts Q-values for actions based on enhanced features
///
/// The combined model allows for end-to-end training where:
/// - Bandit learns to extract useful features and compute importance
/// - Q-network learns to estimate action values
/// - Importance scores guide tier selection
#[derive(Module, Debug)]
pub struct CombinedModel<B: Backend> {
    pub bandit: ContextualBandit<B>,
    pub qnetwork: QNetwork<B>,
}

#[derive(Config, Debug)]
pub struct CombinedModelConfig {
    /// Input state dimension
    pub state_dim: usize,
    /// Enhanced feature dimension
    pub feature_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Action dimension
    pub action_dim: usize,
}

impl CombinedModelConfig {
    /// Initialize the combined model
    ///
    /// # Arguments
    /// * `device` - Device to initialize the model on
    ///
    /// # Returns
    /// Initialized CombinedModel with random weights
    ///
    /// # Deprecation Notice
    ///
    /// This config type is deprecated. Use `eris::config::CombinedBanditDQNConfig` instead:
    /// ```rust,ignore
    /// use eris::model::ErisDefaults;
    /// use eris::training::mock_env::MockEnv;
    ///
    /// // Option 1: Use defaults with dynamic dimensions
    /// let env = MockEnv::new_with_dims(100, 50, 20);
    /// let state_dim = env.observation_space().dim();
    /// let action_dim = env.action_space().n;
    /// let config = ErisDefaults::storage_tier_model(state_dim, action_dim);
    ///
    /// // Option 2: Build manually with dynamic dimensions
    /// use eris::config::{BanditConfig, DQNConfig, CombinedBanditDQNConfig};
    ///
    /// let feature_dim = 20;
    /// let config = CombinedBanditDQNConfig::builder()
    ///     .bandit(BanditConfig::builder().input_dim(state_dim).hidden_layers(vec![64, 128]).feature_dim(feature_dim).build()?)
    ///     .dqn(DQNConfig::builder().input_dim(feature_dim).hidden_layers(vec![128, 128]).action_dim(action_dim).build()?)
    ///     .build()?;
    /// ```
    #[deprecated(
        since = "0.2.0",
        note = "Use `eris::config::CombinedBanditDQNConfig` or `eris::model::ErisDefaults` instead"
    )]
    pub fn init<B: Backend>(&self, device: &B::Device) -> CombinedModel<B> {
        log::warn!(
            "CombinedModelConfig is deprecated. Use eris::config::CombinedBanditDQNConfig or eris::model::ErisDefaults"
        );

        CombinedModel {
            // TODO: Migrate to eris::config::BanditConfig builder pattern
            #[allow(deprecated)]
            bandit: ContextualBanditConfig::new(
                self.state_dim,
                self.state_dim, // Use state_dim for bandit, not hidden_dim/2
                self.feature_dim,
            )
            .init(device),

            // TODO: Migrate to eris::config::DQNConfig builder pattern
            #[allow(deprecated)]
            qnetwork: QNetworkConfig::new(self.feature_dim, self.hidden_dim, self.action_dim)
                .init(device),
        }
    }
}

impl<B: Backend> CombinedModel<B> {
    /// Forward pass combining bandit and Q-network
    ///
    /// # Arguments
    /// * `state` - Input tensor of shape [batch_size, state_dim]
    ///
    /// # Returns
    /// * (features, importance_score, q_values) where:
    ///   - features: [batch_size, feature_dim]
    ///   - importance_score: [batch_size, 1]
    ///   - q_values: [batch_size, action_dim]
    pub fn forward(&self, state: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // Step 1: Bandit extracts features + importance score
        let (features, importance) = self.bandit.forward(state.clone());

        // Step 2: Q-network predicts Q-values from enhanced features
        let q_values = self.qnetwork.forward(features.clone());

        (features, importance, q_values)
    }

    /// Select action based on model output and epsilon-greedy policy
    ///
    /// # Arguments
    /// * `state` - Input tensor [1, state_dim]
    /// * `tier_selector` - Tier selector for mapping importance to tier
    /// * `epsilon` - Exploration rate [0, 1]
    ///
    /// # Returns
    /// * Action encoded as usize (0-9): tier_idx * 2 + op_type
    ///   where op_type: 0 = read, 1 = write
    pub fn select_action(
        &self,
        state: Tensor<B, 2>,
        tier_selector: &TierSelector,
        epsilon: f32,
    ) -> usize {
        use rand::prelude::*;
        use rand::rng;

        let mut rng = rng();

        if rng.random_range(0.0f32..1.0) < epsilon {
            // Exploration: random action
            rng.random_range(0..10)
        } else {
            // Exploitation: use model predictions
            let (_, importance, q_values) = self.forward(state);

            // Get importance score as scalar (batch_size=1, output_dim=1)
            let importance_vec: Vec<f32> = importance
                .into_data()
                .convert::<f32>()
                .to_vec()
                .expect("Failed to convert importance to vec");
            let importance_val = importance_vec[0];

            // Map importance to tier via capacity-weighted distribution
            let tier_idx = tier_selector.select_tier(importance_val);

            // Get Q-values for this tier
            let tier_start = tier_idx * 2;
            let tier_end = tier_start + 2;

            // Slice Q-values for this tier: [batch=1, 2]
            let tier_q_values = q_values.slice([0..1, tier_start..tier_end]);

            // Find argmax to select best operation
            let argmax_idx = tier_q_values.argmax(1);

            // Convert to scalar
            let op_idx_vec: Vec<i32> = argmax_idx
                .into_data()
                .convert::<i32>()
                .to_vec()
                .expect("Failed to convert argmax to vec");
            let op_idx = op_idx_vec[0];

            // Encode action: tier * 2 + op
            tier_idx * 2 + op_idx as usize
        }
    }
}

impl<B: Backend> Checkpointable<B> for CombinedModel<B> {
    fn checkpoint_name(&self) -> &str {
        "combined_model"
    }

    fn checkpoint_metadata(&self) -> CheckpointMetadata {
        // Delegate to sub-models for their config info
        let bandit_metadata = self.bandit.checkpoint_metadata();
        let qnetwork_metadata = self.qnetwork.checkpoint_metadata();

        // Merge dimensions - use the larger of the two for state/action
        let state_dim = bandit_metadata.state_dim.max(qnetwork_metadata.state_dim);
        let action_dim = bandit_metadata.action_dim.max(qnetwork_metadata.action_dim);
        let feature_dim = bandit_metadata
            .feature_dim
            .max(qnetwork_metadata.feature_dim);

        let mut metadata = CheckpointMetadata::new_with_dims(
            "CombinedModel".to_string(),
            0, // epoch - will be updated by training loop
            state_dim.unwrap_or(0),
            action_dim.unwrap_or(0),
            feature_dim.unwrap_or(0),
        );

        // Store sub-model configs as JSON
        metadata.model_config = Some(serde_json::json!({
            "bandit": bandit_metadata.model_config,
            "qnetwork": qnetwork_metadata.model_config,
        }));

        metadata
    }

    fn model(&self) -> &impl Module<B> {
        self
    }
}

/// Decode action index to (tier_idx, op_type)
///
/// # Arguments
/// * `action_idx` - Action index in range [0, 9]
///
/// # Returns
/// * (tier_idx, op_type) where:
///   - tier_idx: [0, 4] indicating which tier
///   - op_type: 0 = read, 1 = write
pub fn decode_action(action_idx: usize) -> (usize, usize) {
    let tier_idx = action_idx / 2;
    let op_type = action_idx % 2;
    (tier_idx, op_type)
}

/// Encode (tier_idx, op_type) to action index
///
/// # Arguments
/// * `tier_idx` - Tier index in range [0, 4]
/// * `op_type` - Operation type: 0 = read, 1 = write
///
/// # Returns
/// * Encoded action index in range [0, 9]
pub fn encode_action(tier_idx: usize, op_type: usize) -> usize {
    tier_idx * 2 + op_type
}

/// New Metis model using generic SequentialCompose.
///
/// This chains BanditAdapter → DQNAdapter with stop-gradient between them.
/// The bandit learns features independently, the DQN decides actions,
/// and they cooperate through shared reward.
///
/// For the baseline comparison, use `CombinedModel` (now "metis-baseline").
pub type MetisModel<B> = burnme_rly::models::SequentialCompose<B, BanditAdapter<B>, DQNAdapter<B>>;

/// Helper methods specific to MetisModel.
pub trait MetisModelExt<B: Backend> {
    /// Get features + importance from the bandit (for bandit loss).
    fn forward_bandit(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>);

    /// Get Q-values from the DQN (for DQN loss, without detach).
    fn forward_dqn(&self, features: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Get the composed forward pass (with detach, for action selection).
    fn forward_composed(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;
}

impl<B: Backend> MetisModelExt<B> for MetisModel<B> {
    fn forward_bandit(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        self.model_a.bandit.forward(input)
    }

    fn forward_dqn(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward_b(features)
    }

    fn forward_composed(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.forward(input)
    }
}

// Note: MetisModel<B> is a type alias for SequentialCompose from burnme-rly.
// It already implements Module<B> via the manual impl in burnme-rly.
// Checkpointing is done via the generic save_checkpoint function with the model name.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_encoding_decoding() {
        // Test all valid actions
        for tier_idx in 0..5 {
            for op_type in 0..2 {
                let action = encode_action(tier_idx, op_type);
                assert!(action < 10, "Action {} should be < 10", action);

                let (decoded_tier, decoded_op) = decode_action(action);
                assert_eq!(
                    decoded_tier, tier_idx,
                    "Decoded tier {} should match original {}",
                    decoded_tier, tier_idx
                );
                assert_eq!(
                    decoded_op, op_type,
                    "Decoded op {} should match original {}",
                    decoded_op, op_type
                );
            }
        }
    }

    #[test]
    fn test_action_bounds() {
        // Test encoding produces valid ranges
        for tier in 0..5 {
            for op in 0..2 {
                let action = encode_action(tier, op);
                assert!(action < 10, "Action should be in [0, 10)");
                if tier < 4 {
                    let next_tier_action = encode_action(tier + 1, op);
                    assert!(
                        next_tier_action > action,
                        "Larger tier should produce larger action"
                    );
                }
            }
        }
    }
}
