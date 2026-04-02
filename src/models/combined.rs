use burn::{config::Config, module::Module, prelude::*};
use rand::Rng;

use crate::models::{ContextualBandit, ContextualBanditConfig, QNetwork, QNetworkConfig};
use crate::tier::TierSelector;

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
    pub fn init<B: Backend>(&self, device: &B::Device) -> CombinedModel<B> {
        CombinedModel {
            bandit: ContextualBanditConfig::new(
                self.state_dim,
                self.hidden_dim / 2,
                self.feature_dim,
            )
            .init(device),
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
        let (features, importance) = self.bandit.forward(state);

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
        let mut rng = rand::rng();

        if rng.random::<f32>() < epsilon {
            // Exploration: random action
            rng.random_range(0..10)
        } else {
            // Exploitation: use model predictions
            let (_, importance, q_values) = self.forward(state);

            // Get importance score as scalar (batch_size=1, output_dim=1)
            let importance_val: f32 = importance.into_data().convert::<f32>().value[0];

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
            let op_idx: i64 = argmax_idx.into_data().convert::<i64>().value[0];

            // Encode action: tier * 2 + op
            tier_idx * 2 + op_idx as usize
        }
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
