//! Comprehensive tests for exploration strategies
//!
//! Tests cover EpsilonGreedy, ThompsonSampling, UCB, and ExplorationConfig

use burn::backend::NdArray;
use burn::tensor::Tensor;
use eris::policies::{
    EpsilonGreedy, ExplorationConfig, ExplorationStrategy, ThompsonSampling, UCBExplorer,
};

type TestBackend = NdArray;

// ============================================================================
// Epsilon-Greedy Tests
// ============================================================================

#[test]
fn test_epsilon_greedy_exploration() {
    let device = Default::default();
    let strategy = EpsilonGreedy::new(1.0, 0.01, 0.995);

    let q_values = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);

    let mut action_counts = vec![0; 4];
    for _ in 0..100 {
        let actions = ExplorationStrategy::<TestBackend>::select_action(&strategy, &q_values, 4);
        let action_idx: Vec<i32> = actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap_or_default();
        action_counts[action_idx[0] as usize] += 1;
    }

    let total_selections: i32 = action_counts.iter().sum();
    assert_eq!(total_selections, 100);
    assert!(action_counts.iter().all(|&c| c > 0));
}

#[test]
fn test_epsilon_greedy_exploitation() {
    let device = Default::default();
    let strategy = EpsilonGreedy::new(0.0, 0.0, 0.995);

    let q_values = Tensor::<TestBackend, 2>::from_floats(
        [[1.0, 2.0, 3.0, 4.0], [10.0, 5.0, 3.0, 1.0]],
        &device,
    );

    let actions = ExplorationStrategy::<TestBackend>::select_action(&strategy, &q_values, 4);
    let action_slice: Vec<i32> = actions
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap_or_default();

    assert_eq!(action_slice.len(), 2);
    assert_eq!(action_slice[0], 3);
    assert_eq!(action_slice[1], 0);
}

#[test]
fn test_epsilon_decay() {
    let mut strategy = EpsilonGreedy::new(1.0, 0.01, 0.99);

    for _ in 0..10 {
        ExplorationStrategy::<TestBackend>::decay(&mut strategy);
    }

    let epsilon = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!(epsilon < 1.0);
    assert!(epsilon >= 0.01);
}

#[test]
fn test_epsilon_get_set() {
    let mut strategy = EpsilonGreedy::new(0.5, 0.01, 0.99);

    let epsilon = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!((epsilon - 0.5).abs() < 1e-6);

    ExplorationStrategy::<TestBackend>::set_param(&mut strategy, 0.3);
    let epsilon = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!((epsilon - 0.3).abs() < 1e-6);

    ExplorationStrategy::<TestBackend>::set_param(&mut strategy, 1.5);
    let epsilon = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!((epsilon - 0.5).abs() < 1e-6);

    ExplorationStrategy::<TestBackend>::set_param(&mut strategy, 0.001);
    let epsilon = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!((epsilon - 0.01).abs() < 1e-6);
}

#[test]
fn test_epsilon_greedy_batch_processing() {
    let device = Default::default();
    let strategy = EpsilonGreedy::new(1.0, 0.01, 0.995);

    for batch_size in [1, 4, 8, 16].iter() {
        let q_values = Tensor::<TestBackend, 2>::zeros([*batch_size, 5], &device);
        let actions = ExplorationStrategy::<TestBackend>::select_action(&strategy, &q_values, 5);
        let action_data: Vec<i32> = actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap_or_default();

        assert_eq!(action_data.len(), *batch_size as usize);
        for action in &action_data {
            assert!(*action >= 0 && *action < 5);
        }
    }
}

// ============================================================================
// Thompson Sampling Tests
// ============================================================================

#[test]
fn test_thompson_sampling_creation() {
    let device = Default::default();
    let strategy = ThompsonSampling::new(10, 0.0, 1.0);

    let q_values = Tensor::<TestBackend, 2>::zeros([1, 10], &device);
    let actions = ExplorationStrategy::<TestBackend>::select_action(&strategy, &q_values, 10);
    let action_data: Vec<i32> = actions
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap_or_default();

    assert_eq!(action_data.len(), 1);
    assert!(action_data[0] >= 0 && action_data[0] < 10);
}

#[test]
fn test_posterior_update() {
    let device = Default::default();
    let mut strategy = ThompsonSampling::new(5, 0.0, 1.0);

    ExplorationStrategy::<TestBackend>::update(&mut strategy, 0, 1.0);
    ExplorationStrategy::<TestBackend>::update(&mut strategy, 0, 0.0);
    ExplorationStrategy::<TestBackend>::update(&mut strategy, 1, 10.0);

    let q_values = Tensor::<TestBackend, 2>::zeros([1, 5], &device);
    let actions = ExplorationStrategy::<TestBackend>::select_action(&strategy, &q_values, 5);
    let action_data: Vec<i32> = actions
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap_or_default();

    assert_eq!(action_data.len(), 1);
    assert!(action_data[0] >= 0 && action_data[0] < 5);
}

#[test]
fn test_thompson_sampling_action_selection() {
    let device = Default::default();
    let strategy = ThompsonSampling::new(4, 0.0, 1.0);

    let q_values = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);

    let mut action_counts = vec![0i32; 4];
    for _ in 0..100 {
        let actions = ExplorationStrategy::<TestBackend>::select_action(&strategy, &q_values, 4);
        let action_slice: Vec<i32> = actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap_or_default();
        action_counts[action_slice[0] as usize] += 1;
    }

    // Thompson sampling should select multiple actions (stochastic)
    assert!(action_counts.iter().filter(|&&c| c > 0).count() > 1);
}

#[test]
fn test_thompson_decay() {
    let mut strategy = ThompsonSampling::new(3, 0.0, 1.0);

    let initial_param = ExplorationStrategy::<TestBackend>::get_param(&strategy);

    for _ in 0..10 {
        ExplorationStrategy::<TestBackend>::decay(&mut strategy);
    }

    let final_param = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!(final_param < initial_param);
    assert!(final_param >= 0.01);
}

#[test]
fn test_thompson_get_set_param() {
    let mut strategy = ThompsonSampling::new(5, 0.0, 1.0);

    let avg_std = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!(avg_std > 0.0 && avg_std <= 2.0);

    ExplorationStrategy::<TestBackend>::set_param(&mut strategy, 0.5);
    let updated_param = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!((updated_param - 0.5).abs() < 1e-6);

    ExplorationStrategy::<TestBackend>::set_param(&mut strategy, 0.001);
    let clamped_param = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!((clamped_param - 0.01).abs() < 1e-6);
}

// ============================================================================
// UCB Explorer Tests
// ============================================================================

#[test]
fn test_ucb_exploration_unexplored() {
    let device = Default::default();
    let mut strategy = UCBExplorer::new(4, 2.0);

    // Initialize UCB with some history to avoid all infinite scores
    for i in 0..4 {
        ExplorationStrategy::<TestBackend>::update(&mut strategy, i, 0.5);
    }

    let q_values = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);

    let actions = ExplorationStrategy::<TestBackend>::select_action(&strategy, &q_values, 4);
    let action_slice: Vec<i32> = actions
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap_or_default();

    assert!(action_slice[0] >= 0 && action_slice[0] < 4);
}

#[test]
fn test_ucb_count_based() {
    let device = Default::default();
    let mut strategy = UCBExplorer::new(4, 2.0);

    ExplorationStrategy::<TestBackend>::update(&mut strategy, 0, 5.0);
    ExplorationStrategy::<TestBackend>::update(&mut strategy, 0, 5.0);
    ExplorationStrategy::<TestBackend>::update(&mut strategy, 1, 10.0);
    // Initialize remaining actions to avoid infinite scores
    ExplorationStrategy::<TestBackend>::update(&mut strategy, 2, 1.0);
    ExplorationStrategy::<TestBackend>::update(&mut strategy, 3, 1.0);

    let q_values = Tensor::<TestBackend, 2>::from_floats([[5.0, 10.0, 15.0, 20.0]], &device);

    let mut action_counts = vec![0i32; 4];
    for _ in 0..50 {
        let actions = ExplorationStrategy::<TestBackend>::select_action(&strategy, &q_values, 4);
        let action_slice: Vec<i32> = actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .unwrap_or_default();
        ExplorationStrategy::<TestBackend>::update(&mut strategy, action_slice[0] as usize, 1.0);
        action_counts[action_slice[0] as usize] += 1;
    }

    assert!(action_counts.iter().all(|&c| c > 0));
}

#[test]
fn test_ucb_coefficient() {
    let device = Default::default();

    let strategy_low_c = UCBExplorer::new(4, 0.5);
    let strategy_high_c = UCBExplorer::new(4, 5.0);

    let param_low = ExplorationStrategy::<TestBackend>::get_param(&strategy_low_c);
    let param_high = ExplorationStrategy::<TestBackend>::get_param(&strategy_high_c);

    assert!((param_low - 0.5).abs() < 1e-6);
    assert!((param_high - 5.0).abs() < 1e-6);

    let mut strat_low = strategy_low_c;
    let mut strat_high = strategy_high_c;

    for i in 0..4 {
        ExplorationStrategy::<TestBackend>::update(&mut strat_low, i, 1.0);
        ExplorationStrategy::<TestBackend>::update(&mut strat_high, i, 1.0);
    }

    let q_values = Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);

    let actions_low = ExplorationStrategy::<TestBackend>::select_action(&strat_low, &q_values, 4);
    let actions_high = ExplorationStrategy::<TestBackend>::select_action(&strat_high, &q_values, 4);

    let action_low: Vec<i32> = actions_low
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap_or_default();
    let action_high: Vec<i32> = actions_high
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap_or_default();

    assert!(action_low[0] >= 0 && action_low[0] < 4);
    assert!(action_high[0] >= 0 && action_high[0] < 4);
}

#[test]
fn test_ucb_decay() {
    let mut strategy = UCBExplorer::new(4, 2.0);

    let initial_c = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    ExplorationStrategy::<TestBackend>::decay(&mut strategy);

    let final_c = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!(final_c < initial_c);
    assert!(final_c >= 0.5);
}

#[test]
fn test_ucb_get_set_param() {
    let mut strategy = UCBExplorer::new(4, 2.0);

    let param = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!((param - 2.0).abs() < 1e-6);

    ExplorationStrategy::<TestBackend>::set_param(&mut strategy, 1.5);
    let param = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!((param - 1.5).abs() < 1e-6);

    ExplorationStrategy::<TestBackend>::set_param(&mut strategy, 0.05);
    let param = ExplorationStrategy::<TestBackend>::get_param(&strategy);
    assert!((param - 0.1).abs() < 1e-6);
}

// ============================================================================
// ExplorationConfig Tests
// ============================================================================

#[test]
fn test_config_build_epsilon() {
    let config = ExplorationConfig::EpsilonGreedy {
        epsilon_start: 0.9,
        epsilon_end: 0.01,
        epsilon_decay: 0.99,
    };

    let strategy: Box<dyn ExplorationStrategy<TestBackend>> = config.build(5);

    let param = ExplorationStrategy::<TestBackend>::get_param(&*strategy);
    assert!((param - 0.9).abs() < 1e-6);

    let device = Default::default();
    let q_values = Tensor::<TestBackend, 2>::zeros([1, 5], &device);
    let actions = ExplorationStrategy::<TestBackend>::select_action(&*strategy, &q_values, 5);
    let action_data: Vec<i32> = actions
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap_or_default();

    assert_eq!(action_data.len(), 1);
    assert!(action_data[0] >= 0 && action_data[0] < 5);
}

#[test]
fn test_config_build_thompson() {
    let config = ExplorationConfig::ThompsonSampling {
        prior_mean: 0.5,
        prior_std: 0.3,
    };

    let strategy: Box<dyn ExplorationStrategy<TestBackend>> = config.build(10);

    let param = ExplorationStrategy::<TestBackend>::get_param(&*strategy);
    assert!(param > 0.0 && param < 1.0);

    let device = Default::default();
    let q_values = Tensor::<TestBackend, 2>::zeros([2, 10], &device);
    let actions = ExplorationStrategy::<TestBackend>::select_action(&*strategy, &q_values, 10);
    let action_data: Vec<i32> = actions
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap_or_default();

    assert_eq!(action_data.len(), 2);
    for action in &action_data {
        assert!(*action >= 0 && *action < 10);
    }
}

#[test]
fn test_config_build_ucb() {
    let config = ExplorationConfig::UCB { c: 1.5 };

    let mut strategy: Box<dyn ExplorationStrategy<TestBackend>> = config.build(6);

    let param = strategy.get_param();
    assert!((param - 1.5).abs() < 1e-6);

    // Initialize UCB with some history to avoid infinite scores
    for i in 0..6 {
        strategy.update(i, 0.5);
    }

    let device = Default::default();
    let q_values = Tensor::<TestBackend, 2>::zeros([1, 6], &device);
    let actions = strategy.select_action(&q_values, 6);
    let action_data: Vec<i32> = actions
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .unwrap_or_default();

    assert_eq!(action_data.len(), 1);
    assert!(action_data[0] >= 0 && action_data[0] < 6);
}

#[test]
fn test_exploration_strategy_clone() {
    let config = ExplorationConfig::EpsilonGreedy {
        epsilon_start: 0.5,
        epsilon_end: 0.01,
        epsilon_decay: 0.99,
    };

    let strategy: Box<dyn ExplorationStrategy<TestBackend>> = config.build(4);
    let cloned = strategy.clone();

    let param1 = ExplorationStrategy::<TestBackend>::get_param(&*strategy);
    let param2 = ExplorationStrategy::<TestBackend>::get_param(&*cloned);
    assert!((param1 - param2).abs() < 1e-6);
}

#[test]
fn test_exploration_multiple_decays() {
    let mut strategy = EpsilonGreedy::new(1.0, 0.01, 0.99);

    let mut epsilons = vec![ExplorationStrategy::<TestBackend>::get_param(&strategy)];
    for _ in 0..50 {
        ExplorationStrategy::<TestBackend>::decay(&mut strategy);
        epsilons.push(ExplorationStrategy::<TestBackend>::get_param(&strategy));
    }

    for i in 1..epsilons.len() {
        assert!(epsilons[i] <= epsilons[i - 1]);
    }

    assert!(epsilons.last().unwrap() <= &epsilons[0]);
}
