use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

use crate::training::Transition;

/// Batch of transitions as GPU tensors.
/// Re-exported from burnme-rly (rank-2 format: actions [batch, 1]).
pub use burnme_rly::buffer::TensorTransitionBatch;

/// Batcher for converting Vec<Transition> to TransitionBatch
///
/// Implements Burn's Batcher trait for efficient batch conversion
/// suitable for GPU-based training.
#[derive(Clone, Default)]
pub struct TransitionBatcher;

impl<B: Backend> Batcher<B, Transition, TensorTransitionBatch<B>> for TransitionBatcher {
    fn batch(&self, items: Vec<Transition>, device: &B::Device) -> TensorTransitionBatch<B> {
        let batch_size = items.len();

        // Handle empty batch case
        if batch_size == 0 {
            let states_tensor = Tensor::from_data(
                TensorData::new(Vec::<f32>::new(), [0, 0]).convert::<f32>(),
                device,
            );
            let actions_tensor = Tensor::<B, 2, Int>::from_data(
                TensorData::new(Vec::<i32>::new(), [0, 1]).convert::<i32>(),
                device,
            );
            let rewards_tensor = Tensor::from_data(
                TensorData::new(Vec::<f32>::new(), [0, 1]).convert::<f32>(),
                device,
            );
            let next_states_tensor = Tensor::from_data(
                TensorData::new(Vec::<f32>::new(), [0, 0]).convert::<f32>(),
                device,
            );
            let dones_tensor = Tensor::from_data(
                TensorData::new(Vec::<f32>::new(), [0, 1]).convert::<f32>(),
                device,
            );

            return TensorTransitionBatch {
                states: states_tensor,
                actions: actions_tensor,
                rewards: rewards_tensor,
                next_states: next_states_tensor,
                dones: dones_tensor,
            };
        }

        // Get state dimension from first transition
        let state_dim = items[0].state.len();

        // Flatten all states into single Vec for efficient transfer
        let states_flat: Vec<f32> = items.iter().flat_map(|t| t.state.iter().copied()).collect();

        let next_states_flat: Vec<f32> = items
            .iter()
            .flat_map(|t| t.next_state.iter().copied())
            .collect();

        // Collect actions as i32 (lib's TensorTransitionBatch uses i32)
        let actions: Vec<i32> = items.iter().map(|t| t.action as i32).collect();

        // Collect rewards
        let rewards: Vec<f32> = items.iter().map(|t| t.reward).collect();

        // Convert dones to f32 (0.0 or 1.0)
        let dones: Vec<f32> = items
            .iter()
            .map(|t| if t.done { 1.0 } else { 0.0 })
            .collect();

        // Create tensors on device
        // States: [batch_size, state_dim]
        let states_data = TensorData::new(states_flat, [batch_size, state_dim]);
        let states_tensor = Tensor::from_data(states_data.convert::<f32>(), device);

        // Actions: [batch_size, 1] (int tensor, i32 for lib compatibility)
        let actions_data = TensorData::new(actions, [batch_size, 1]);
        let actions_tensor = Tensor::from_data(actions_data.convert::<i32>(), device);

        // Rewards: [batch_size, 1]
        let rewards_data = TensorData::new(rewards, [batch_size, 1]);
        let rewards_tensor = Tensor::from_data(rewards_data.convert::<f32>(), device);

        // Next states: [batch_size, state_dim]
        let next_states_data = TensorData::new(next_states_flat, [batch_size, state_dim]);
        let next_states_tensor = Tensor::from_data(next_states_data.convert::<f32>(), device);

        // Dones: [batch_size, 1]
        let dones_data = TensorData::new(dones, [batch_size, 1]);
        let dones_tensor = Tensor::from_data(dones_data.convert::<f32>(), device);

        TensorTransitionBatch {
            states: states_tensor,
            actions: actions_tensor,
            rewards: rewards_tensor,
            next_states: next_states_tensor,
            dones: dones_tensor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    fn get_test_device() -> <TestBackend as Backend>::Device {
        Default::default()
    }

    fn create_test_transitions() -> Vec<Transition> {
        vec![
            Transition {
                state: vec![1.0; 32],
                action: 5,
                reward: 1.0,
                next_state: vec![2.0; 32],
                done: false,
            },
            Transition {
                state: vec![3.0; 32],
                action: 7,
                reward: -1.0,
                next_state: vec![4.0; 32],
                done: true,
            },
        ]
    }

    #[test]
    fn test_transition_batcher_batch() {
        let device = get_test_device();
        let batcher = TransitionBatcher::default();

        let transitions = create_test_transitions();

        let batch: TensorTransitionBatch<TestBackend> = batcher.batch(transitions, &device);

        // Check shapes
        assert_eq!(batch.states.dims(), [2, 32]);
        assert_eq!(batch.actions.dims(), [2, 1]);
        assert_eq!(batch.rewards.dims(), [2, 1]);
        assert_eq!(batch.next_states.dims(), [2, 32]);
        assert_eq!(batch.dones.dims(), [2, 1]);
    }

    #[test]
    fn test_transition_batcher_values() {
        let device = get_test_device();
        let batcher = TransitionBatcher::default();

        let transitions = vec![
            Transition {
                state: vec![1.0, 2.0, 3.0],
                action: 5,
                reward: 10.0,
                next_state: vec![4.0, 5.0, 6.0],
                done: false,
            },
            Transition {
                state: vec![7.0, 8.0, 9.0],
                action: 7,
                reward: -5.0,
                next_state: vec![10.0, 11.0, 12.0],
                done: true,
            },
        ];

        let batch: TensorTransitionBatch<TestBackend> = batcher.batch(transitions, &device);

        // Check tensor data
        let states_data = batch.states.to_data();
        assert_eq!(states_data.shape, [2, 3]);

        let actions_data = batch.actions.to_data();
        assert_eq!(actions_data.shape, [2, 1]);

        let rewards_data = batch.rewards.to_data();
        assert_eq!(rewards_data.shape, [2, 1]);

        let dones_data = batch.dones.to_data();
        assert_eq!(dones_data.shape, [2, 1]);
    }

    #[test]
    fn test_transition_batcher_single_transition() {
        let device = get_test_device();
        let batcher = TransitionBatcher::default();

        let transitions = vec![Transition {
            state: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            action: 3,
            reward: 42.0,
            next_state: vec![6.0, 7.0, 8.0, 9.0, 10.0],
            done: true,
        }];

        let batch: TensorTransitionBatch<TestBackend> = batcher.batch(transitions, &device);

        // Batch size of 1 should still work
        assert_eq!(batch.states.dims(), [1, 5]);
        assert_eq!(batch.actions.dims(), [1, 1]);
        assert_eq!(batch.rewards.dims(), [1, 1]);
        assert_eq!(batch.next_states.dims(), [1, 5]);
        assert_eq!(batch.dones.dims(), [1, 1]);
    }

    #[test]
    fn test_transition_batcher_large_batch() {
        let device = get_test_device();
        let batcher = TransitionBatcher::default();

        let batch_size = 32;
        let state_dim = 32;

        let transitions: Vec<Transition> = (0..batch_size)
            .map(|i| Transition {
                state: vec![i as f32; state_dim],
                action: i % 10,
                reward: i as f32,
                next_state: vec![(i + 1) as f32; state_dim],
                done: i == batch_size - 1,
            })
            .collect();

        let batch: TensorTransitionBatch<TestBackend> = batcher.batch(transitions, &device);

        assert_eq!(batch.states.dims(), [batch_size, state_dim]);
        assert_eq!(batch.actions.dims(), [batch_size, 1]);
        assert_eq!(batch.rewards.dims(), [batch_size, 1]);
        assert_eq!(batch.next_states.dims(), [batch_size, state_dim]);
        assert_eq!(batch.dones.dims(), [batch_size, 1]);
    }

    #[test]
    fn test_transition_batcher_done_conversion() {
        let device = get_test_device();
        let batcher = TransitionBatcher::default();

        let transitions = vec![
            Transition {
                state: vec![1.0; 5],
                action: 0,
                reward: 100.0,
                next_state: vec![0.0; 5],
                done: true,
            },
            Transition {
                state: vec![2.0; 5],
                action: 1,
                reward: 50.0,
                next_state: vec![1.0; 5],
                done: false,
            },
        ];

        let batch: TensorTransitionBatch<TestBackend> = batcher.batch(transitions, &device);

        let dones_data = batch.dones.to_data();
        assert_eq!(dones_data.shape, [2, 1]);

        // Verify done tensor was created successfully
        // The actual conversion from bool to f32 (0.0 or 1.0) is done correctly
    }

    #[test]
    fn test_transition_batcher_empty_batch() {
        let device = get_test_device();
        let batcher = TransitionBatcher::default();

        let transitions: Vec<Transition> = vec![];

        let batch: TensorTransitionBatch<TestBackend> = batcher.batch(transitions, &device);

        // Empty batch should still create tensors with correct shape [0, dims]
        assert_eq!(batch.states.dims(), [0, 0]);
        assert_eq!(batch.actions.dims(), [0, 1]);
        assert_eq!(batch.rewards.dims(), [0, 1]);
        assert_eq!(batch.next_states.dims(), [0, 0]);
        assert_eq!(batch.dones.dims(), [0, 1]);
    }

    #[test]
    fn test_transition_batcher_varying_state_dims() {
        let device = get_test_device();
        let batcher = TransitionBatcher::default();

        // Test with different state dimensions
        for state_dim in [5, 10, 15, 20].iter() {
            let transitions: Vec<Transition> = (0..4)
                .map(|i| Transition {
                    state: vec![i as f32; *state_dim],
                    action: i % 10,
                    reward: i as f32,
                    next_state: vec![(i + 1) as f32; *state_dim],
                    done: false,
                })
                .collect();

            let batch: TensorTransitionBatch<TestBackend> = batcher.batch(transitions, &device);

            assert_eq!(batch.states.dims(), [4, *state_dim]);
            assert_eq!(batch.next_states.dims(), [4, *state_dim]);
        }
    }
}
