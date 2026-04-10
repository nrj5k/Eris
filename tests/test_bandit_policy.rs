//! Comprehensive tests for Bandit Policy
//!
//! Tests cover creation, action selection, online learning, and trait implementations

use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;
use burn::tensor::Tensor;
use eris::config::BanditConfig;
use eris::model::Activation;
use eris::policies::{
    Action, BanditPolicy, BanditPolicyConfig, CachePolicy, ExplorationConfig, OnlinePolicy,
    PolicyType, State, Transition,
};

type TestBackend = Autodiff<NdArray>;

fn create_test_bandit_config() -> BanditPolicyConfig {
    let bandit_config = BanditConfig::builder()
        .input_dim(15)
        .hidden_layers(vec![64, 128])
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

// ============================================================================
// Creation Tests
// ============================================================================

#[test]
fn test_bandit_creation() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let policy = BanditPolicy::<TestBackend>::new(config, &device);

    assert_eq!(
        policy.action_dim(),
        10,
        "Action dimension should be num_tiers * 2 = 10"
    );
    assert_eq!(
        policy.learning_rate(),
        0.001,
        "Learning rate should be 0.001"
    );
    assert_eq!(
        policy.policy_type(),
        PolicyType::Bandit,
        "Policy type should be Bandit"
    );
}

#[test]
fn test_bandit_with_different_tiers() {
    let device = <NdArray as Backend>::Device::default();

    // Test with 3 tiers
    let bandit_config = BanditConfig::builder()
        .input_dim(10)
        .hidden_layers(vec![64])
        .feature_dim(20)
        .activation(Activation::Sigmoid)
        .build()
        .expect("Valid config");

    let config_3 = BanditPolicyConfig::new(
        bandit_config.clone(),
        ExplorationConfig::EpsilonGreedy {
            epsilon_start: 0.5,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        },
        0.001,
        3,
    );
    let policy_3 = BanditPolicy::<TestBackend>::new(config_3, &device);
    assert_eq!(policy_3.action_dim(), 6, "3 tiers should have 6 actions");

    // Test with 7 tiers
    let config_7 = BanditPolicyConfig::new(
        bandit_config,
        ExplorationConfig::EpsilonGreedy {
            epsilon_start: 0.5,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        },
        0.001,
        7,
    );
    let policy_7 = BanditPolicy::<TestBackend>::new(config_7, &device);
    assert_eq!(policy_7.action_dim(), 14, "7 tiers should have 14 actions");
}

#[test]
fn test_importance_mapping() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let policy = BanditPolicy::<TestBackend>::new(config, &device);

    // Test importance to tier mapping
    assert_eq!(
        policy.importance_to_tier(0.0),
        0,
        "Importance 0.0 should map to tier 0"
    );
    assert_eq!(
        policy.importance_to_tier(0.1),
        0,
        "Importance 0.1 should map to tier 0"
    );
    assert_eq!(
        policy.importance_to_tier(0.2),
        1,
        "Importance 0.2 should map to tier 1"
    );
    assert_eq!(
        policy.importance_to_tier(0.5),
        2,
        "Importance 0.5 should map to tier 2"
    );
    assert_eq!(
        policy.importance_to_tier(0.8),
        4,
        "Importance 0.8 should map to tier 4"
    );
    assert_eq!(
        policy.importance_to_tier(1.0),
        4,
        "Importance 1.0 should map to tier 4"
    );

    // Test edge cases
    assert_eq!(
        policy.importance_to_tier(-0.5),
        0,
        "Negative importance should clamp to tier 0"
    );
    assert_eq!(
        policy.importance_to_tier(1.5),
        4,
        "Importance > 1 should clamp to last tier"
    );
}

// ============================================================================
// Action Selection Tests
// ============================================================================

#[test]
fn test_bandit_select_action() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let policy = BanditPolicy::<TestBackend>::new(config, &device);

    let state = State::Features(vec![1.0; 15]);
    let action = policy.select_action(&state);

    match action {
        Action::Discrete(idx) => {
            assert!(idx < 10, "Action {} should be < 10", idx);
        }
        _ => panic!("Expected discrete action from bandit policy"),
    }
}

#[test]
fn test_bandit_online_update() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let mut policy = BanditPolicy::<TestBackend>::new(config, &device);

    // Create transition
    let state = State::Features(vec![0.5; 15]);
    let transition = Transition {
        state: state.clone(),
        action: Action::Discrete(0), // Tier 0 read operation
        reward: 1.0,
        next_state: state,
        done: false,
    };

    // Update policy
    let loss = policy.update(&transition);

    // Loss should be non-negative (MSE)
    assert!(loss >= 0.0, "Loss should be non-negative");
    assert!(loss.is_finite(), "Loss should be finite");
}

#[test]
fn test_bandit_update_with_different_rewards() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let mut policy = BanditPolicy::<TestBackend>::new(config, &device);

    let state = State::Features(vec![0.5; 15]);

    // Test with positive reward
    let transition_pos = Transition {
        state: state.clone(),
        action: Action::Discrete(0),
        reward: 1.0,
        next_state: state.clone(),
        done: false,
    };
    let loss_pos = policy.update(&transition_pos);
    assert!(
        loss_pos >= 0.0,
        "Loss with positive reward should be non-negative"
    );

    // Test with negative reward
    let transition_neg = Transition {
        state: state.clone(),
        action: Action::Discrete(2),
        reward: -1.0,
        next_state: state.clone(),
        done: false,
    };
    let loss_neg = policy.update(&transition_neg);
    assert!(
        loss_neg >= 0.0,
        "Loss with negative reward should be non-negative"
    );

    // Test with zero reward
    let transition_zero = Transition {
        state: state.clone(),
        action: Action::Discrete(4),
        reward: 0.0,
        next_state: state,
        done: false,
    };
    let loss_zero = policy.update(&transition_zero);
    assert!(
        loss_zero >= 0.0,
        "Loss with zero reward should be non-negative"
    );
}

// ============================================================================
// Trait Implementation Tests
// ============================================================================

#[test]
fn test_cache_policy_impl() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let mut policy = BanditPolicy::<TestBackend>::new(config, &device);

    // Test select_action
    let state = State::Features(vec![1.0; 15]);
    let action = policy.select_action(&state);
    match action {
        Action::Discrete(idx) => assert!(idx < 10),
        _ => panic!("Expected discrete action"),
    }

    // Test update
    let transition = Transition {
        state: State::Features(vec![1.0; 15]),
        action: Action::Discrete(0),
        reward: 1.0,
        next_state: State::Features(vec![2.0; 15]),
        done: false,
    };
    let loss = policy.update(&transition);
    assert!(loss >= 0.0, "Loss should be non-negative");

    // Test policy_type
    assert_eq!(policy.policy_type(), PolicyType::Bandit);

    // Test action_dim
    assert_eq!(
        policy.action_dim(),
        10,
        "action_dim should be num_tiers * 2"
    );
}

#[test]
fn test_online_policy_impl() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let mut policy = BanditPolicy::<TestBackend>::new(config, &device);

    // Test learning_rate
    assert_eq!(
        policy.learning_rate(),
        0.001,
        "Initial learning rate should be 0.001"
    );

    // Test set_learning_rate
    policy.set_learning_rate(0.01);
    assert_eq!(
        policy.learning_rate(),
        0.01,
        "Learning rate should update to 0.01"
    );

    policy.set_learning_rate(0.0001);
    assert_eq!(
        policy.learning_rate(),
        0.0001,
        "Learning rate should update to 0.0001"
    );
}

#[test]
fn test_policy_type() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let policy = BanditPolicy::<TestBackend>::new(config, &device);

    assert_eq!(policy.policy_type(), PolicyType::Bandit);
}

// ============================================================================
// Importance Score Tests
// ============================================================================

#[test]
fn test_get_importance() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let policy = BanditPolicy::<TestBackend>::new(config, &device);

    let state = State::Features(vec![0.5; 15]);
    let importance = policy.get_importance(&state);

    // Importance should be in range [0, 1] due to Sigmoid activation
    assert!(
        importance >= 0.0 && importance <= 1.0,
        "Importance should be in [0, 1], got {}",
        importance
    );
}

#[test]
fn test_forward_pass() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let policy = BanditPolicy::<TestBackend>::new(config, &device);

    use burn::tensor::TensorData;

    // Create input tensor
    let state_data: Vec<f32> = vec![0.5; 15];
    let tensor_data = TensorData::new(state_data, [1, 15]);
    let state_tensor = Tensor::<TestBackend, 2>::from_data(tensor_data.convert::<f32>(), &device);

    let importance = policy.forward_train(state_tensor);

    // Check output shape
    let dims = importance.dims();
    assert_eq!(dims, [1, 1], "Importance should have shape [1, 1]");

    // Check values are in [0, 1]
    let values: Vec<f32> = importance.into_data().to_vec().unwrap_or_default();
    for v in &values {
        assert!(
            *v >= 0.0 && *v <= 1.0,
            "Importance value {} should be in [0, 1]",
            v
        );
    }
}

// ============================================================================
// Multiple Tier Configurations
// ============================================================================

#[test]
fn test_different_num_tiers() {
    let device = <NdArray as Backend>::Device::default();

    for num_tiers in [3, 5, 7].iter() {
        let bandit_config = BanditConfig::builder()
            .input_dim(15)
            .hidden_layers(vec![64])
            .feature_dim(20)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Valid config");

        let config = BanditPolicyConfig::new(
            bandit_config,
            ExplorationConfig::EpsilonGreedy {
                epsilon_start: 0.5,
                epsilon_end: 0.01,
                epsilon_decay: 0.995,
            },
            0.001,
            *num_tiers,
        );

        let policy = BanditPolicy::<TestBackend>::new(config, &device);

        // Verify action dimension
        assert_eq!(
            policy.action_dim(),
            num_tiers * 2,
            "Action dim should be num_tiers * 2"
        );

        // Verify valid actions
        let state = State::Features(vec![1.0; 15]);
        let action = policy.select_action(&state);

        if let Action::Discrete(idx) = action {
            assert!(
                idx < num_tiers * 2,
                "Action {} should be < {}",
                idx,
                num_tiers * 2
            );
        }
    }
}

// ============================================================================
// Training Progression Tests
// ============================================================================

#[test]
fn test_training_progression() {
    let config = create_test_bandit_config();
    let device = <NdArray as Backend>::Device::default();
    let mut policy = BanditPolicy::<TestBackend>::new(config, &device);

    let mut losses = Vec::new();

    // Simulate training progression
    for episode in 0..10 {
        let state = State::Features(vec![episode as f32 / 10.0; 15]);
        let action = policy.select_action(&state);

        let transition = Transition {
            state: state.clone(),
            action,
            reward: 1.0 - (episode as f32 / 20.0), // Decreasing reward
            next_state: state,
            done: false,
        };

        let loss = policy.update(&transition);
        losses.push(loss);
    }

    // All losses should be non-negative and finite
    for (i, loss) in losses.iter().enumerate() {
        assert!(loss.is_finite(), "Loss {} should be finite: {}", i, loss);
        assert!(*loss >= 0.0, "Loss {} should be non-negative: {}", i, loss);
    }
}
