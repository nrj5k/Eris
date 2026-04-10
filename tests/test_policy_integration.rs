//! Integration tests for policies with different configurations
//!
//! Tests cover multiple tier counts, exploration strategy swapping, and end-to-end training

use burn::backend::{Autodiff, NdArray};
use burn::prelude::Backend;
use eris::config::{BanditConfig, DQNConfig};
use eris::model::Activation;
use eris::policies::{
    Action, BanditPolicy, BanditPolicyConfig, CachePolicy, DQNExplorerConfig, DQNPolicy,
    ExplorationConfig, PolicyType, ReplayPolicy, State, Transition,
};

type TestBackend = Autodiff<NdArray>;

// ============================================================================
// Different Tier Counts Tests
// ============================================================================

#[test]
fn test_dqn_different_tier_counts() {
    let device = <NdArray as Backend>::Device::default();

    for num_tiers in [3, 5, 7].iter() {
        // DQN action_dim = num_tiers * 2 (read/write operations)
        let action_dim = num_tiers * 2;

        // Create config with appropriate action dimensions
        let dqn_config = DQNConfig::builder()
            .input_dim(15)
            .hidden_layers(vec![128])
            .action_dim(action_dim)
            .build()
            .expect("Valid config");

        let exploration = ExplorationConfig::EpsilonGreedy {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        let config = DQNExplorerConfig::new(dqn_config, exploration);
        let policy = DQNPolicy::<TestBackend>::new(config, device.clone());

        // Verify policy creation
        assert_eq!(
            policy.action_dim(),
            action_dim,
            "DQN action dim should be {}",
            action_dim
        );
        assert_eq!(policy.policy_type(), PolicyType::Dqn);

        // Verify valid action selection
        let state = State::Features(vec![1.0; 15]);
        let action = policy.select_action(&state);

        match action {
            Action::Discrete(idx) => {
                assert!(
                    idx < action_dim,
                    "Action {} should be < {}",
                    idx,
                    action_dim
                );
            }
            _ => panic!("Expected discrete action"),
        }
    }
}

#[test]
fn test_bandit_different_tier_counts() {
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

        // Bandit action_dim = num_tiers * 2
        let action_dim = num_tiers * 2;
        assert_eq!(
            policy.action_dim(),
            action_dim,
            "Bandit action dim should be {}",
            action_dim
        );
        assert_eq!(policy.policy_type(), PolicyType::Bandit);

        // Verify valid action selection
        let state = State::Features(vec![1.0; 15]);
        let action = policy.select_action(&state);

        match action {
            Action::Discrete(idx) => {
                assert!(
                    idx < action_dim,
                    "Action {} should be < {}",
                    idx,
                    action_dim
                );
            }
            _ => panic!("Expected discrete action"),
        }
    }
}

#[test]
fn test_action_dimensions_match_tiers() {
    let device = <NdArray as Backend>::Device::default();

    // Test that action dimensions correctly scale with tier count
    for num_tiers in [1, 2, 3, 5, 10].iter() {
        // DQN
        let dqn_config = DQNConfig::builder()
            .input_dim(10)
            .hidden_layers(vec![64])
            .action_dim(num_tiers * 2)
            .build()
            .expect("Valid config");

        let dqn_policy = DQNPolicy::<TestBackend>::new(
            DQNExplorerConfig::new(
                dqn_config,
                ExplorationConfig::EpsilonGreedy {
                    epsilon_start: 0.5,
                    epsilon_end: 0.01,
                    epsilon_decay: 0.99,
                },
            ),
            device.clone(),
        );

        assert_eq!(
            dqn_policy.action_dim(),
            num_tiers * 2,
            "DQN action dim should match tier count * 2"
        );

        // Bandit
        let bandit_config = BanditConfig::builder()
            .input_dim(10)
            .hidden_layers(vec![64])
            .feature_dim(20)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Valid config");

        let bandit_policy = BanditPolicy::<TestBackend>::new(
            BanditPolicyConfig::new(
                bandit_config,
                ExplorationConfig::EpsilonGreedy {
                    epsilon_start: 0.5,
                    epsilon_end: 0.01,
                    epsilon_decay: 0.99,
                },
                0.001,
                *num_tiers,
            ),
            &device,
        );

        assert_eq!(
            bandit_policy.action_dim(),
            num_tiers * 2,
            "Bandit action dim should match tier count * 2"
        );
    }
}

// ============================================================================
// Exploration Swapping Tests
// ============================================================================

#[test]
fn test_dqn_different_exploration_strategies() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = DQNConfig::builder()
        .input_dim(10)
        .hidden_layers(vec![64])
        .action_dim(6)
        .build()
        .expect("Valid config");

    // Train with each exploration strategy
    let explorations = vec![
        (
            "EpsilonGreedy",
            ExplorationConfig::EpsilonGreedy {
                epsilon_start: 1.0,
                epsilon_end: 0.01,
                epsilon_decay: 0.99,
            },
        ),
        (
            "ThompsonSampling",
            ExplorationConfig::ThompsonSampling {
                prior_mean: 0.0,
                prior_std: 1.0,
            },
        ),
        ("UCB", ExplorationConfig::UCB { c: 2.0 }),
    ];

    for (name, exploration) in explorations {
        let config = DQNExplorerConfig::new(dqn_config.clone(), exploration).with_batch_size(4);
        let mut policy = DQNPolicy::<TestBackend>::new(config, device.clone());

        // Run short training
        for episode in 0..5 {
            let batch: Vec<Transition> = (0..4)
                .map(|i| Transition {
                    state: State::Features(vec![i as f32; 10]),
                    action: Action::Discrete(i % 6),
                    reward: (episode + i) as f32 * 0.1,
                    next_state: State::Features(vec![(i + 1) as f32; 10]),
                    done: false,
                })
                .collect();

            let loss = policy.train_step(&batch);
            assert!(
                loss.is_finite() && loss >= 0.0,
                "{}: Loss should be valid at episode {}",
                name,
                episode
            );
        }
    }
}

#[test]
fn test_bandit_different_exploration_strategies() {
    let device = <NdArray as Backend>::Device::default();

    let bandit_config = BanditConfig::builder()
        .input_dim(10)
        .hidden_layers(vec![32])
        .feature_dim(16)
        .activation(Activation::Sigmoid)
        .build()
        .expect("Valid config");

    // Train with EpsilonGreedy and ThompsonSampling (UCB requires initialization)
    let explorations = vec![
        (
            "EpsilonGreedy",
            ExplorationConfig::EpsilonGreedy {
                epsilon_start: 0.5,
                epsilon_end: 0.01,
                epsilon_decay: 0.995,
            },
        ),
        (
            "ThompsonSampling",
            ExplorationConfig::ThompsonSampling {
                prior_mean: 0.0,
                prior_std: 1.0,
            },
        ),
    ];

    for (name, exploration) in explorations {
        let config = BanditPolicyConfig::new(bandit_config.clone(), exploration, 0.001, 3);
        let mut policy = BanditPolicy::<TestBackend>::new(config, &device);

        // Run short training
        for episode in 0..5 {
            let state = State::Features(vec![episode as f32 / 5.0; 10]);
            let action = policy.select_action(&state);

            let transition = Transition {
                state: state.clone(),
                action,
                reward: (episode as f32) * 0.2,
                next_state: state,
                done: false,
            };

            let loss = policy.update(&transition);
            assert!(
                loss.is_finite() && loss >= 0.0,
                "{}: Loss should be valid at episode {}",
                name,
                episode
            );
        }
    }
}

// ============================================================================
// End-to-End Training Tests
// ============================================================================

#[test]
fn test_dqn_short_training_run() {
    let device = <NdArray as Backend>::Device::default();
    let dqn_config = DQNConfig::builder()
        .input_dim(15)
        .hidden_layers(vec![64])
        .action_dim(10)
        .build()
        .expect("Valid config");

    let exploration = ExplorationConfig::EpsilonGreedy {
        epsilon_start: 0.5,
        epsilon_end: 0.01,
        epsilon_decay: 0.99,
    };

    let config = DQNExplorerConfig::new(dqn_config, exploration).with_batch_size(4);
    let mut policy = DQNPolicy::<TestBackend>::new(config, device);

    // Run 10 episodes
    let mut losses = Vec::new();

    for episode in 0..10 {
        // Create batch of transitions
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

    // Verify training progressed
    assert!(
        losses.iter().all(|&l| l.is_finite()),
        "All losses should be finite"
    );
}

#[test]
fn test_bandit_short_training_run() {
    let device = <NdArray as Backend>::Device::default();

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
        5,
    );

    let mut policy = BanditPolicy::<TestBackend>::new(config, &device);

    // Run 10 episodes
    let mut losses = Vec::new();

    for episode in 0..10 {
        let state = State::Features(vec![episode as f32 / 10.0; 15]);
        let action = policy.select_action(&state);

        let transition = Transition {
            state: state.clone(),
            action,
            reward: 1.0 / (1.0 + episode as f32), // Decreasing reward
            next_state: state,
            done: episode == 9,
        };

        let loss = policy.update(&transition);
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

    assert!(
        losses.iter().all(|&l| l.is_finite()),
        "All losses should be finite"
    );
}

// ============================================================================
// Policy Type Verification
// ============================================================================

#[test]
fn test_policy_type_distinction() {
    let device = <NdArray as Backend>::Device::default();

    // Create DQN
    let dqn_config = DQNConfig::builder()
        .input_dim(10)
        .hidden_layers(vec![64])
        .action_dim(6)
        .build()
        .expect("Valid config");

    let dqn = DQNPolicy::<TestBackend>::new(
        DQNExplorerConfig::new(
            dqn_config,
            ExplorationConfig::EpsilonGreedy {
                epsilon_start: 0.5,
                epsilon_end: 0.01,
                epsilon_decay: 0.99,
            },
        ),
        device.clone(),
    );

    // Create Bandit
    let bandit_config = BanditConfig::builder()
        .input_dim(10)
        .hidden_layers(vec![64])
        .feature_dim(20)
        .activation(Activation::Sigmoid)
        .build()
        .expect("Valid config");

    let bandit = BanditPolicy::<TestBackend>::new(
        BanditPolicyConfig::new(
            bandit_config,
            ExplorationConfig::EpsilonGreedy {
                epsilon_start: 0.5,
                epsilon_end: 0.01,
                epsilon_decay: 0.99,
            },
            0.001,
            3,
        ),
        &device,
    );

    // Verify policy types are distinct
    assert_eq!(dqn.policy_type(), PolicyType::Dqn);
    assert_eq!(bandit.policy_type(), PolicyType::Bandit);
    assert_ne!(dqn.policy_type(), bandit.policy_type());
}
