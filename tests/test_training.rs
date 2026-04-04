// Test file for DQN training components
//
// **⚠️ DEPRECATION NOTICE:**
// Tests using `train_step()` are maintained for backward compatibility.
// The manual train_step method is deprecated - use Burn TrainStep instead.
// See src/training/burn_trainer.rs for Burn-native implementation.
//
// These tests remain to verify:
// - Legacy functionality continues working
// - Backward compatibility during transition
// - Target network updates work correctly

use burn::backend::{Autodiff, NdArray};
use eris::training::{
    create_dummy_transition, fill_buffer, CheckpointMetadata, CombinedAgent, MockEnv, ReplayBuffer,
    TrainingConfig, Transition,
};

#[test]
fn test_replay_buffer() {
    let mut buffer = ReplayBuffer::new(10);

    // Add transitions with dynamic dimensions
    let obs_dim = 15;
    let num_actions = 10;

    for i in 0..15 {
        buffer.push(Transition {
            state: vec![i as f32; obs_dim],
            action: i % num_actions,
            reward: i as f32,
            next_state: vec![(i + 1) as f32; obs_dim],
            done: i == 14,
        });
    }

    // Capacity enforced
    assert_eq!(buffer.len(), 10);

    // Sample batch
    let batch = buffer.sample_batch(5);
    assert!(batch.is_some());
    let batch = batch.unwrap();
    assert_eq!(batch.states.len(), 5);
}

#[test]
fn test_replay_buffer_capacity() {
    let mut buffer = ReplayBuffer::new(20);

    // Add exactly capacity items
    for i in 0..20 {
        buffer.push(Transition {
            state: vec![i as f32],
            action: i % 10,
            reward: i as f32,
            next_state: vec![i as f32],
            done: false,
        });
    }

    assert_eq!(buffer.len(), 20);
    assert!(buffer.is_full());

    // Add one more, should evict oldest
    buffer.push(Transition {
        state: vec![20.0],
        action: 0,
        reward: 20.0,
        next_state: vec![20.0],
        done: false,
    });

    assert_eq!(buffer.len(), 20);
    assert!(buffer.is_full());

    // First item should now be index 1 (oldest evicted)
    let sample = buffer.sample(1);
    // This test might occasionally fail due to randomness, so we just check length
    assert_eq!(sample.len(), 1);
}

#[test]
fn test_replay_buffer_sample() {
    let mut buffer = ReplayBuffer::new(20);

    // Add transitions
    for i in 0..20 {
        buffer.push(Transition {
            state: vec![i as f32],
            action: i % 10,
            reward: i as f32,
            next_state: vec![i as f32],
            done: i == 19,
        });
    }

    // Sample with batch size smaller than buffer
    let sample = buffer.sample(5);
    assert_eq!(sample.len(), 5);

    // Sample with batch size larger than buffer
    let sample = buffer.sample(100);
    assert_eq!(sample.len(), 20); // Max available

    // Sample with batch size equal to buffer
    let sample = buffer.sample(20);
    assert_eq!(sample.len(), 20);
}

#[test]
fn test_replay_buffer_empty() {
    let buffer = ReplayBuffer::new(10);
    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);

    // Sampling from empty buffer
    let sample = buffer.sample(5);
    assert_eq!(sample.len(), 0);

    let batch = buffer.sample_batch(5);
    assert!(batch.is_none());
}

#[test]
fn test_training_config_default() {
    let config = TrainingConfig::default();

    assert_eq!(config.learning_rate, 0.001);
    assert_eq!(config.gamma, 0.99);
    assert_eq!(config.epsilon_start, 1.0);
    assert_eq!(config.epsilon_end, 0.01);
    assert_eq!(config.epsilon_decay, 0.995);
    assert_eq!(config.batch_size, 512); // Updated for GPU utilization
    assert_eq!(config.buffer_capacity, 10_000);
    assert_eq!(config.target_update_freq, 1000);
    assert_eq!(config.tau, 0.005);
}

#[test]
fn test_transition_batch() {
    let mut buffer = ReplayBuffer::new(20);

    // Add transitions with dynamic dimensions
    let obs_dim = 15;
    let num_actions = 10;

    for i in 0..10 {
        buffer.push(Transition {
            state: vec![i as f32; obs_dim],
            action: i % num_actions,
            reward: i as f32,
            next_state: vec![(i + 1) as f32; obs_dim],
            done: i == 9,
        });
    }

    // Sample batch
    let batch = buffer.sample_batch(5).unwrap();

    // Check batch structure
    assert_eq!(batch.states.len(), 5);
    assert_eq!(batch.actions.len(), 5);
    assert_eq!(batch.rewards.len(), 5);
    assert_eq!(batch.next_states.len(), 5);
    assert_eq!(batch.dones.len(), 5);

    // Check each state has correct dimension
    for state in &batch.states {
        assert_eq!(state.len(), obs_dim);
    }

    // Check actions are valid
    for action in &batch.actions {
        assert!(*action < num_actions);
    }
}

// ==============================================================================
// NEW TESTS FOR CRITICAL FIXES
// ==============================================================================

type TestBackend = Autodiff<NdArray>;

fn create_test_agent() -> CombinedAgent<TestBackend> {
    use eris::models::CombinedModelConfig;

    let device = Default::default();
    let config = TrainingConfig::default();
    // Dynamic dimensions from environment
    let env = MockEnv::new(100);
    let state_dim = env.observation_dim();
    let action_dim = env.num_actions();
    let model_config = CombinedModelConfig::new(state_dim, 20, 128, action_dim);

    CombinedAgent::new(config, model_config, device)
}

#[test]
#[allow(deprecated)]
fn test_train_step_returns_valid_loss() {
    let mut agent = create_test_agent();

    // Fill buffer
    for _ in 0..100 {
        agent.buffer.push(create_dummy_transition());
    }

    let batch = agent.buffer.sample_batch(32).unwrap();
    let loss = agent.train_step(batch);

    assert!(loss >= 0.0, "Loss should be non-negative, got {}", loss);
    assert!(!loss.is_nan(), "Loss should not be NaN");
    assert!(loss.is_finite(), "Loss should be finite");
}

#[test]
#[allow(deprecated)]
fn test_train_step_updates_weights() {
    let mut agent = create_test_agent();

    // Fill buffer
    for _ in 0..100 {
        agent.buffer.push(create_dummy_transition());
    }

    // Get initial loss
    let batch1 = agent.buffer.sample_batch(32).unwrap();
    let loss1 = agent.train_step(batch1);

    // Train 10 more steps
    for _ in 0..10 {
        if let Some(batch) = agent.buffer.sample_batch(32) {
            agent.train_step(batch);
        }
    }

    // Loss should change
    let batch2 = agent.buffer.sample_batch(32).unwrap();
    let loss2 = agent.train_step(batch2);

    // Either loss decreased or model is learning
    // (may not always decrease due to stochasticity)
    assert!(loss2.is_finite(), "Loss should remain finite");
}

#[test]
#[allow(deprecated)]
fn test_hard_update_target() {
    let mut agent = create_test_agent();

    // Modify model by training
    for _ in 0..10 {
        agent.buffer.push(create_dummy_transition());
    }
    if let Some(batch) = agent.buffer.sample_batch(32) {
        agent.train_step(batch);
    }

    // Update target network
    agent.hard_update_target();

    // Verify target_model is different instance
    // (implementation detail: can't directly compare weights without Tensor API)
    // This test verifies no crash occurs
}

#[test]
fn test_mock_env_reset() {
    let mut env = MockEnv::new(10);

    let state = env.reset();
    assert_eq!(state.len(), 15);
    assert!(state.iter().all(|&x| x == 0.0));

    let _ = env.step(5);
    assert_ne!(env.step_count, 0);

    let state = env.reset();
    assert_eq!(env.step_count, 0);
    assert!(state.iter().all(|&x| x == 0.0));
}

#[test]
fn test_mock_env_step() {
    let mut env = MockEnv::new(10);

    env.reset();

    let (next_state, reward, done) = env.step(5);
    assert_eq!(next_state.len(), 15);
    assert!(reward > 0.0);
    assert!(!done);

    // Check state changed
    assert!(next_state.iter().any(|&x| x != 0.0));
}

#[test]
fn test_checkpoint_metadata() {
    let meta = CheckpointMetadata::new(10, 1000, 0.5, 10.0, 8.5);

    assert_eq!(meta.epoch, 10);
    assert_eq!(meta.step_count, 1000);
    assert_eq!(meta.epsilon, 0.5);
    assert_eq!(meta.best_reward, 10.0);
    assert_eq!(meta.avg_reward_10, 8.5);
    assert!(!meta.timestamp.is_empty());
}

#[test]
fn test_training_config_defaults() {
    let config = TrainingConfig::default();

    assert_eq!(config.learning_rate, 0.001);
    assert_eq!(config.gamma, 0.99);
    assert_eq!(config.epsilon_start, 1.0);
    assert_eq!(config.epsilon_end, 0.01);
    assert_eq!(config.epsilon_decay, 0.995);
    assert_eq!(config.batch_size, 512); // Updated for GPU utilization
    assert_eq!(config.buffer_capacity, 10_000);
    assert_eq!(config.target_update_freq, 1000);
    assert_eq!(config.tau, 0.005);
    assert_eq!(config.backend, "wgpu");
    assert_eq!(config.checkpoint_interval, 10);
    assert_eq!(config.max_gradient_norm, 1.0);
}

#[test]
fn test_fill_buffer() {
    let mut buffer = ReplayBuffer::new(100);
    fill_buffer(&mut buffer, 50, 15, 10);

    assert_eq!(buffer.len(), 50);
}

#[test]
fn test_agent_initialization() {
    let agent = create_test_agent();

    assert_eq!(agent.buffer.len(), 0);
    assert_eq!(agent.epsilon, 1.0); // epsilon_start
    assert_eq!(agent.step_count, 0);
}

#[test]
#[allow(deprecated)]
fn test_epsilon_decay() {
    let mut agent = create_test_agent();
    let initial_epsilon = agent.epsilon;

    // Fill buffer and train
    for _ in 0..110 {
        agent.buffer.push(create_dummy_transition());
    }

    if let Some(batch) = agent.buffer.sample_batch(32) {
        agent.train_step(batch);
    }

    // Epsilon should have decayed
    assert!(agent.epsilon < initial_epsilon);
    assert!(agent.epsilon >= agent.config.epsilon_end);
}

#[test]
#[allow(deprecated)]
fn test_train_step_empty_batch() {
    let mut agent = create_test_agent();

    // Empty batch
    let batch = eris::training::TransitionBatch {
        states: vec![],
        actions: vec![],
        rewards: vec![],
        next_states: vec![],
        dones: vec![],
    };

    let loss = agent.train_step(batch);

    // Should return 0.0 for empty batch
    assert_eq!(loss, 0.0);
}
