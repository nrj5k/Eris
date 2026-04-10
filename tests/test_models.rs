use burn::backend::NdArray;
use burn::tensor::Tensor;

use eris::models::{
    decode_action, encode_action, CombinedModel, CombinedModelConfig, ContextualBandit,
    ContextualBanditConfig, QNetwork, QNetworkConfig,
};
use eris::tier::Tier;
use eris::TierConfig; // Old TierConfig from config_old

type Backend = NdArray;

fn create_test_tier(id: u32, capacity: f64, name: &str) -> Tier {
    Tier::new(TierConfig {
        name: name.into(),
        tier_id: id,
        capacity,
        access_latency: 0.01,
        description: String::new(),
    })
}

#[test]
fn test_qnetwork_forward() {
    let device = Default::default();
    let config = QNetworkConfig::new(20, 128, 10);
    let qnetwork = config.init::<Backend>(&device);

    // Input: [batch_size=1, input_dim=20]
    let input = Tensor::<Backend, 2>::zeros([1, 20], &device);

    let output = qnetwork.forward(input);

    // Output: [batch_size=1, action_dim=10]
    assert_eq!(output.shape().dims, [1, 10]);

    // Check values are finite
    let values: Vec<f32> = output
        .into_data()
        .to_vec()
        .expect("Failed to convert to vec");
    for v in &values {
        assert!(v.is_finite(), "Value {} should be finite", v);
    }
}

#[test]
fn test_contextual_bandit_forward() {
    let device = Default::default();
    let config = ContextualBanditConfig::new(10, 64, 20);
    let bandit = config.init::<Backend>(&device);

    // Input: [batch_size=1, state_dim=10]
    let input = Tensor::<Backend, 2>::zeros([1, 10], &device);

    let (features, importance) = bandit.forward(input);

    // Features: [batch_size=1, feature_dim=20]
    assert_eq!(features.shape().dims, [1, 20]);

    // Importance: [batch_size=1, 1]
    assert_eq!(importance.shape().dims, [1, 1]);

    // Importance should be in [0, 1] (sigmoid)
    let importance_vec: Vec<f32> = importance
        .into_data()
        .to_vec()
        .expect("Failed to convert importance to vec");
    let importance_val = importance_vec[0];
    assert!(
        importance_val >= 0.0 && importance_val <= 1.0,
        "Importance {} should be in [0, 1]",
        importance_val
    );
}

#[test]
fn test_combined_model_forward() {
    let device = Default::default();
    let config = CombinedModelConfig::new(10, 20, 128, 10);
    let combined = config.init::<Backend>(&device);

    // Input: [batch_size=1, state_dim=10]
    let input = Tensor::<Backend, 2>::zeros([1, 10], &device);

    let (features, importance, q_values) = combined.forward(input);

    // Check shapes
    assert_eq!(features.shape().dims, [1, 20]);
    assert_eq!(importance.shape().dims, [1, 1]);
    assert_eq!(q_values.shape().dims, [1, 10]);
}

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
fn test_qnetwork_dueling_architecture() {
    let device = Default::default();
    let config = QNetworkConfig::new(20, 128, 10);
    let qnetwork = config.init::<Backend>(&device);

    // Same input should produce different Q-values for different actions
    let input = Tensor::<Backend, 2>::ones([1, 20], &device);
    let q_values = qnetwork.forward(input);

    // Convert to vector
    let values: Vec<f32> = q_values
        .into_data()
        .to_vec()
        .expect("Failed to convert to vec");

    // At least some values should be different (dueling architecture should create differences)
    let first = values[0];
    let has_different = values.iter().any(|&v| v != first);
    assert!(
        has_different,
        "All Q-values are identical, dueling architecture may not be working"
    );
}

#[test]
fn test_qnetwork_batch_processing() {
    let device = Default::default();
    let config = QNetworkConfig::new(20, 128, 10);
    let qnetwork = config.init::<Backend>(&device);

    // Test with batch size of 3
    let input = Tensor::<Backend, 2>::zeros([3, 20], &device);
    let output = qnetwork.forward(input);

    // Output: [batch_size=3, action_dim=10]
    assert_eq!(output.shape().dims, [3, 10]);
}

#[test]
fn test_bandit_batch_processing() {
    let device = Default::default();
    let config = ContextualBanditConfig::new(10, 64, 20);
    let bandit = config.init::<Backend>(&device);

    // Test with batch size of 4
    let input = Tensor::<Backend, 2>::zeros([4, 10], &device);
    let (features, importance) = bandit.forward(input);

    assert_eq!(features.shape().dims, [4, 20]);
    assert_eq!(importance.shape().dims, [4, 1]);
}

#[test]
fn test_combined_model_batch_processing() {
    let device = Default::default();
    let config = CombinedModelConfig::new(10, 20, 128, 10);
    let combined = config.init::<Backend>(&device);

    // Test with batch size of 5
    let input = Tensor::<Backend, 2>::zeros([5, 10], &device);
    let (features, importance, q_values) = combined.forward(input);

    assert_eq!(features.shape().dims, [5, 20]);
    assert_eq!(importance.shape().dims, [5, 1]);
    assert_eq!(q_values.shape().dims, [5, 10]);
}
