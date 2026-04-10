//! Comprehensive tests for DQN Policy
//!
//! Tests cover creation, action selection, training, and trait implementations

use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;
use eris::config::DQNConfig;
use eris::policies::{
    Action, CachePolicy, DQNExplorerConfig, DQNPolicy, ExplorationConfig, PolicyType, ReplayPolicy,
    State, Transition,
};

type TestBackend = Autodiff<NdArray>;

fn create_test_dqn_config() -> DQNConfig {
    DQNConfig::builder()
        .input_dim(15)
        .hidden_layers(vec![128, 128])
        .action_dim(10)
        .build()
        .expect("Valid DQN config")
}

fn create_test_exploration_epsilon() -> ExplorationConfig {
    ExplorationConfig::EpsilonGreedy {
        epsilon_start: 1.0,
        epsilon_end: 0.01,
        epsilon_decay: 0.995,
    }
}

// ============================================================================
// Creation Tests
// ============================================================================

#[test]
fn test_dqn_creation() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = create_test_exploration_epsilon();
    let config = DQNExplorerConfig::new(dqn_config, exploration);

    let policy = DQNPolicy::<TestBackend>::new(config, device);

    assert_eq!(policy.action_dim(), 10, "Action dimension should be 10");
    assert_eq!(
        policy.policy_type(),
        PolicyType::Dqn,
        "Policy type should be Dqn"
    );
    assert_eq!(
        policy.batch_size(),
        2048,
        "Default batch size should be 2048"
    );
}

#[test]
fn test_dqn_with_config_builders() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = create_test_exploration_epsilon();

    let config = DQNExplorerConfig::new(dqn_config, exploration)
        .with_learning_rate(0.001)
        .with_gamma(0.95)
        .with_target_update_freq(500)
        .with_batch_size(256)
        .with_buffer_capacity(5000);

    let policy = DQNPolicy::<TestBackend>::new(config, device);

    assert_eq!(policy.batch_size(), 256, "Batch size should be customized");
}

#[test]
fn test_dqn_with_exploration_strategies() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();

    // Test with EpsilonGreedy
    let exploration_epsilon = create_test_exploration_epsilon();
    let config = DQNExplorerConfig::new(dqn_config.clone(), exploration_epsilon);
    let policy = DQNPolicy::<TestBackend>::new(config, device.clone());
    assert_eq!(policy.policy_type(), PolicyType::Dqn);

    // Test with ThompsonSampling
    let exploration_thompson = ExplorationConfig::ThompsonSampling {
        prior_mean: 0.0,
        prior_std: 1.0,
    };
    let config = DQNExplorerConfig::new(dqn_config.clone(), exploration_thompson);
    let policy = DQNPolicy::<TestBackend>::new(config, device.clone());
    assert_eq!(policy.policy_type(), PolicyType::Dqn);

    // Test with UCB
    let exploration_ucb = ExplorationConfig::UCB { c: 2.0 };
    let config = DQNExplorerConfig::new(dqn_config, exploration_ucb);
    let policy = DQNPolicy::<TestBackend>::new(config, device);
    assert_eq!(policy.policy_type(), PolicyType::Dqn);
}

// ============================================================================
// Action Selection Tests
// ============================================================================

#[test]
fn test_dqn_select_action() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = create_test_exploration_epsilon();
    let config = DQNExplorerConfig::new(dqn_config, exploration);
    let policy = DQNPolicy::<TestBackend>::new(config, device);

    let state = State::Features(vec![1.0; 15]);
    let action = policy.select_action(&state);

    match action {
        Action::Discrete(idx) => {
            assert!(idx < 10, "Action {} should be < 10", idx);
        }
        _ => panic!("Expected discrete action from DQN policy"),
    }
}

#[test]
fn test_dqn_epsilon_greedy() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();

    // Create policy with epsilon=0 (pure exploitation)
    let exploration = ExplorationConfig::EpsilonGreedy {
        epsilon_start: 0.0,
        epsilon_end: 0.0,
        epsilon_decay: 0.99,
    };
    let config = DQNExplorerConfig::new(dqn_config, exploration);
    let policy = DQNPolicy::<TestBackend>::new(config, device);

    // Same state should produce same action (greedy)
    let state = State::Features(vec![1.0; 15]);

    let mut actions = Vec::new();
    for _ in 0..10 {
        let action = policy.select_action(&state);
        if let Action::Discrete(idx) = action {
            actions.push(idx);
        }
    }

    // All actions should be the same (greedy selection)
    let first = actions[0];
    assert!(
        actions.iter().all(|&a| a == first),
        "Epsilon=0 should always select greedy action"
    );
}

#[test]
fn test_dqn_thompson_sampling() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = ExplorationConfig::ThompsonSampling {
        prior_mean: 0.0,
        prior_std: 1.0,
    };
    let config = DQNExplorerConfig::new(dqn_config, exploration);
    let policy = DQNPolicy::<TestBackend>::new(config, device);

    let state = State::Features(vec![0.5; 15]);

    // Thompson sampling should produce different actions over time
    let mut action_counts = vec![0i32; 10];
    for _ in 0..50 {
        let action = policy.select_action(&state);
        if let Action::Discrete(idx) = action {
            action_counts[idx] += 1;
        }
    }

    // At least some actions should be selected
    assert!(
        action_counts.iter().any(|&c| c > 0),
        "Thompson sampling should select some actions"
    );
}

// ============================================================================
// Training Tests
// ============================================================================

#[test]
fn test_dqn_train_step() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = create_test_exploration_epsilon();
    let config = DQNExplorerConfig::new(dqn_config, exploration).with_batch_size(4);
    let mut policy = DQNPolicy::<TestBackend>::new(config, device);

    // Create training batch
    let batch: Vec<Transition> = (0..4)
        .map(|i| Transition {
            state: State::Features(vec![i as f32; 15]),
            action: Action::Discrete(i % 10),
            reward: (i as f32) * 0.1,
            next_state: State::Features(vec![(i + 1) as f32; 15]),
            done: false,
        })
        .collect();

    let loss = policy.train_step(&batch);

    assert!(loss >= 0.0, "Loss should be non-negative");
    assert!(loss.is_finite(), "Loss should be finite");
}

#[test]
fn test_dqn_multiple_train_steps() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = create_test_exploration_epsilon();
    let config = DQNExplorerConfig::new(dqn_config, exploration).with_batch_size(4);
    let mut policy = DQNPolicy::<TestBackend>::new(config, device);

    let mut losses = Vec::new();

    // Run multiple training steps
    for episode in 0..10 {
        let batch: Vec<Transition> = (0..4)
            .map(|i| Transition {
                state: State::Features(vec![(episode + i) as f32 / 100.0; 15]),
                action: Action::Discrete((episode + i) % 10),
                reward: (episode as f32) * 0.1,
                next_state: State::Features(vec![(episode + i + 1) as f32 / 100.0; 15]),
                done: i == 3,
            })
            .collect();

        let loss = policy.train_step(&batch);
        assert!(
            loss.is_finite(),
            "Loss should be finite at episode {}",
            episode
        );
        assert!(
            loss >= 0.0,
            "Loss should be non-negative at episode {}",
            episode
        );

        losses.push(loss);
    }

    // All losses should be finite and non-negative
    assert!(
        losses.iter().all(|&l| l.is_finite()),
        "All losses should be finite"
    );
}

// ============================================================================
// Trait Implementation Tests
// ============================================================================

#[test]
fn test_cache_policy_impl() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = create_test_exploration_epsilon();
    let config = DQNExplorerConfig::new(dqn_config, exploration);
    let mut policy = DQNPolicy::<TestBackend>::new(config, device);

    // Test select_action
    let state = State::Features(vec![1.0; 15]);
    let action = policy.select_action(&state);
    match action {
        Action::Discrete(idx) => assert!(idx < 10),
        _ => panic!("Expected discrete action"),
    }

    // Test policy_type
    assert_eq!(policy.policy_type(), PolicyType::Dqn);

    // Test action_dim
    assert_eq!(policy.action_dim(), 10);
}

#[test]
fn test_replay_policy_impl() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = create_test_exploration_epsilon();
    let config = DQNExplorerConfig::new(dqn_config, exploration).with_batch_size(4);
    let mut policy = DQNPolicy::<TestBackend>::new(config, device);

    // Test train_step
    let batch: Vec<Transition> = (0..4)
        .map(|i| Transition {
            state: State::Features(vec![i as f32; 15]),
            action: Action::Discrete(i % 10),
            reward: (i as f32) * 0.1,
            next_state: State::Features(vec![(i + 1) as f32; 15]),
            done: false,
        })
        .collect();

    let loss = policy.train_step(&batch);
    assert!(loss >= 0.0, "Loss should be non-negative");

    // Test batch_size
    assert_eq!(policy.batch_size(), 4);

    // Test update_target
    policy.update_target();
}

#[test]
fn test_policy_type() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = create_test_exploration_epsilon();
    let config = DQNExplorerConfig::new(dqn_config, exploration);
    let policy = DQNPolicy::<TestBackend>::new(config, device);

    assert_eq!(policy.policy_type(), PolicyType::Dqn);
}

// ============================================================================
// Different Network Architectures
// ============================================================================

#[test]
fn test_dqn_different_hidden_layers() {
    let device = <NdArray as Backend>::Device::default();

    // Test with single hidden layer
    let dqn_config_single = DQNConfig::builder()
        .input_dim(15)
        .hidden_layers(vec![64])
        .action_dim(10)
        .build()
        .expect("Valid config");

    let exploration = create_test_exploration_epsilon();
    let config = DQNExplorerConfig::new(dqn_config_single, exploration.clone());
    let policy = DQNPolicy::<TestBackend>::new(config, device.clone());

    let state = State::Features(vec![1.0; 15]);
    let action = policy.select_action(&state);
    assert!(matches!(action, Action::Discrete(_)));

    // Test with deep network
    let dqn_config_deep = DQNConfig::builder()
        .input_dim(15)
        .hidden_layers(vec![256, 256, 128, 64])
        .action_dim(10)
        .build()
        .expect("Valid config");

    let config_deep = DQNExplorerConfig::new(dqn_config_deep, exploration);
    let policy_deep = DQNPolicy::<TestBackend>::new(config_deep, device);

    let action_deep = policy_deep.select_action(&state);
    assert!(matches!(action_deep, Action::Discrete(_)));
}

// ============================================================================
// Exploration Parameter Tests
// ============================================================================

#[test]
fn test_dqn_exploration_param_management() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = create_test_dqn_config();
    let exploration = ExplorationConfig::EpsilonGreedy {
        epsilon_start: 0.8,
        epsilon_end: 0.01,
        epsilon_decay: 0.99,
    };
    let config = DQNExplorerConfig::new(dqn_config, exploration);
    let mut policy = DQNPolicy::<TestBackend>::new(config, device);

    // Check initial exploration parameter
    let initial = policy.get_exploration_param();
    assert!(
        (initial - 0.8).abs() < 1e-6,
        "Initial exploration should be 0.8"
    );

    // Modify exploration parameter
    policy.set_exploration_param(0.5);
    let updated = policy.get_exploration_param();
    assert!(
        (updated - 0.5).abs() < 1e-6,
        "Exploration should update to 0.5"
    );

    // Verify parameter is within bounds
    policy.set_exploration_param(0.001);
    let clamped = policy.get_exploration_param();
    assert!(clamped >= 0.01, "Exploration should clamp to minimum 0.01");
}
