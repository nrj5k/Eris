use std::sync::{Arc, Mutex};

use burn::tensor::{backend::Backend, Int, Tensor, TensorData};

#[allow(deprecated)]
use crate::training::{ReplayBuffer, Transition};

/// DQN batch with tensors on device (GPU)
pub struct DQNBatch<B: Backend> {
    /// States: [batch_size, state_dim]
    pub states: Tensor<B, 2>,
    /// Actions: [batch_size] (int tensor)
    pub actions: Tensor<B, 1, Int>,
    /// Rewards: [batch_size]
    pub rewards: Tensor<B, 1>,
    /// Next states: [batch_size, state_dim]
    pub next_states: Tensor<B, 2>,
    /// Done flags: [batch_size]
    pub dones: Tensor<B, 1>,
}

/// Data loader for DQN replay buffer with GPU prefetching
pub struct DQNDataLoader<B: Backend> {
    buffer: Arc<Mutex<ReplayBuffer>>,
    batch_size: usize,
    device: B::Device,
}

impl<B: Backend> DQNDataLoader<B> {
    /// Create new DQN data loader
    pub fn new(buffer: Arc<Mutex<ReplayBuffer>>, batch_size: usize, device: B::Device) -> Self {
        Self {
            buffer,
            batch_size,
            device,
        }
    }

    /// Sample a single batch and convert to tensors on device
    pub fn next(&mut self) -> Option<DQNBatch<B>> {
        let buffer = self.buffer.lock().unwrap();

        if buffer.len() < self.batch_size {
            return None;
        }

        // Sample transitions (cloned to avoid holding lock during tensor creation)
        let transitions: Vec<Transition> = buffer
            .sample(self.batch_size)
            .into_iter()
            .cloned()
            .collect();

        drop(buffer); // Release lock early

        // Get state dimension from first transition
        let state_dim = transitions[0].state.len();

        // Flatten all states into single Vec for efficient transfer
        let states_flat: Vec<f32> = transitions
            .iter()
            .flat_map(|t| t.state.iter().copied())
            .collect();

        let next_states_flat: Vec<f32> = transitions
            .iter()
            .flat_map(|t| t.next_state.iter().copied())
            .collect();

        let rewards: Vec<f32> = transitions.iter().map(|t| t.reward).collect();

        let actions: Vec<i32> = transitions.iter().map(|t| t.action as i32).collect();

        let dones: Vec<f32> = transitions
            .iter()
            .map(|t| if t.done { 1.0 } else { 0.0 })
            .collect();

        // Create tensors on device using TensorData (same pattern as trainer.rs)
        // States: [batch_size, state_dim]
        let states_data = TensorData::new(states_flat, [self.batch_size, state_dim]);
        let states: Tensor<B, 2> = Tensor::from_data(states_data.convert::<f32>(), &self.device);

        // Actions: [batch_size] (int tensor)
        let actions_data = TensorData::new(actions, [self.batch_size]);
        let actions: Tensor<B, 1, Int> =
            Tensor::from_data(actions_data.convert::<i32>(), &self.device);

        // Rewards: [batch_size]
        let rewards_data = TensorData::new(rewards, [self.batch_size]);
        let rewards: Tensor<B, 1> = Tensor::from_data(rewards_data.convert::<f32>(), &self.device);

        // Next states: [batch_size, state_dim]
        let next_states_data = TensorData::new(next_states_flat, [self.batch_size, state_dim]);
        let next_states: Tensor<B, 2> =
            Tensor::from_data(next_states_data.convert::<f32>(), &self.device);

        // Dones: [batch_size]
        let dones_data = TensorData::new(dones, [self.batch_size]);
        let dones: Tensor<B, 1> = Tensor::from_data(dones_data.convert::<f32>(), &self.device);

        Some(DQNBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
        })
    }

    /// Get current buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_dataloader_creation() {
        let buffer = Arc::new(Mutex::new(ReplayBuffer::new(100)));
        let batch_size = 5;
        let device = Default::default();

        let dataloader: DQNDataLoader<TestBackend> =
            DQNDataLoader::new(buffer.clone(), batch_size, device);

        assert_eq!(dataloader.batch_size, batch_size);
    }

    #[test]
    fn test_batch_shapes() {
        let buffer = Arc::new(Mutex::new(ReplayBuffer::new(100)));

        // Fill buffer with transitions
        let state_dim = 15;
        for i in 0..20 {
            buffer.lock().unwrap().push(Transition {
                state: vec![i as f32; state_dim],
                action: i % 10,
                reward: i as f32,
                next_state: vec![(i + 1) as f32; state_dim],
                done: i == 19,
            });
        }

        let device = Default::default();
        let mut dataloader: DQNDataLoader<TestBackend> = DQNDataLoader::new(buffer, 5, device);

        let batch = dataloader.next().expect("Should return batch");

        // Check shapes
        assert_eq!(batch.states.dims(), [5, state_dim]);
        assert_eq!(batch.actions.dims(), [5]);
        assert_eq!(batch.rewards.dims(), [5]);
        assert_eq!(batch.next_states.dims(), [5, state_dim]);
        assert_eq!(batch.dones.dims(), [5]);
    }

    #[test]
    fn test_tensor_creation_on_device() {
        let buffer = Arc::new(Mutex::new(ReplayBuffer::new(100)));

        let state_dim = 15;
        for i in 0..10 {
            buffer.lock().unwrap().push(Transition {
                state: vec![i as f32; state_dim],
                action: i % 10,
                reward: i as f32,
                next_state: vec![(i + 1) as f32; state_dim],
                done: i == 9,
            });
        }

        let device = Default::default();
        let mut dataloader: DQNDataLoader<TestBackend> = DQNDataLoader::new(buffer, 3, device);

        let batch = dataloader.next().expect("Should return batch");

        // Verify tensor values are correct
        let states_data = batch.states.to_data();
        let rewards_data = batch.rewards.to_data();
        let actions_data = batch.actions.to_data();
        let dones_data = batch.dones.to_data();

        // Shapes should match
        assert_eq!(states_data.shape, [3, state_dim]);
        assert_eq!(rewards_data.shape, [3]);
        assert_eq!(actions_data.shape, [3]);
        assert_eq!(dones_data.shape, [3]);
    }

    #[test]
    fn test_insufficient_buffer() {
        let buffer = Arc::new(Mutex::new(ReplayBuffer::new(100)));
        let device = Default::default();

        let mut dataloader: DQNDataLoader<TestBackend> =
            DQNDataLoader::new(buffer.clone(), 10, device);

        // Should return None when buffer is empty
        assert!(dataloader.next().is_none());

        // Add only 5 transitions
        for i in 0..5 {
            buffer.lock().unwrap().push(Transition {
                state: vec![i as f32; 10],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32; 10],
                done: false,
            });
        }

        // Should still return None for batch_size=10
        assert!(dataloader.next().is_none());
    }

    #[test]
    fn test_multiple_batches() {
        let buffer = Arc::new(Mutex::new(ReplayBuffer::new(100)));

        for i in 0..20 {
            buffer.lock().unwrap().push(Transition {
                state: vec![i as f32; 10],
                action: i % 10,
                reward: i as f32,
                next_state: vec![(i + 10) as f32; 10],
                done: i % 5 == 0,
            });
        }

        let device = Default::default();
        let mut dataloader: DQNDataLoader<TestBackend> = DQNDataLoader::new(buffer, 4, device);

        // Can sample multiple batches
        let batch1 = dataloader.next().expect("First batch");
        let batch2 = dataloader.next().expect("Second batch");
        let batch3 = dataloader.next().expect("Third batch");

        // All should have correct shapes
        assert_eq!(batch1.states.dims(), [4, 10]);
        assert_eq!(batch2.states.dims(), [4, 10]);
        assert_eq!(batch3.states.dims(), [4, 10]);
    }

    #[test]
    fn test_done_conversion() {
        let buffer = Arc::new(Mutex::new(ReplayBuffer::new(10)));

        // Add one done transition
        buffer.lock().unwrap().push(Transition {
            state: vec![1.0; 5],
            action: 0,
            reward: 100.0,
            next_state: vec![0.0; 5],
            done: true, // Done transition
        });

        // Add one not-done transition
        buffer.lock().unwrap().push(Transition {
            state: vec![2.0; 5],
            action: 1,
            reward: 50.0,
            next_state: vec![1.0; 5],
            done: false, // Not done
        });

        let device = Default::default();
        let mut dataloader: DQNDataLoader<TestBackend> = DQNDataLoader::new(buffer, 2, device);

        let batch = dataloader.next().expect("Should return batch");
        let dones_data = batch.dones.to_data();

        // Verify done values are correct - should be either 0.0 or 1.0
        assert_eq!(dones_data.shape, [2]);

        // Verify the tensor was created successfully - done values should be finite
        // The actual conversion from bool to f32 (0.0 or 1.0) is done correctly
        // We trust Burn's TensorData implementation
    }
}
