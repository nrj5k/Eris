//! Ring buffer with pre-allocated GPU tensors for zero-allocation experience replay
//!
//! This module provides `TensorRingBuffer`, a high-performance ring buffer that uses
//! pre-allocated GPU tensors instead of `Vec<Transition>`. This eliminates memory
//! allocations during push operations and enables zero-copy sampling.

use burn::data::dataset::Dataset;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use rand::prelude::*;
use rand::rng;
use std::marker::PhantomData;

/// Transition data for experience replay
#[derive(Debug, Clone)]
pub struct Transition {
    /// Current state observation [state_dim]
    pub state: Vec<f32>,
    /// Action taken (discrete index)
    pub action: usize,
    /// Reward received
    pub reward: f32,
    /// Next state observation [state_dim]
    pub next_state: Vec<f32>,
    /// Whether episode terminated
    pub done: bool,
}

/// Ring buffer with pre-allocated GPU tensors for zero-allocation experience replay
pub struct TensorRingBuffer<B: Backend> {
    // Pre-allocated tensors [capacity, state_dim]
    states: Tensor<B, 2>,
    next_states: Tensor<B, 2>,

    // Pre-allocated tensors [capacity]
    actions: Tensor<B, 1, burn::tensor::Int>,
    rewards: Tensor<B, 1>,
    dones: Tensor<B, 1>,

    // Circular buffer state
    head: usize,
    size: usize,
    capacity: usize,
    state_dim: usize,

    // Backend marker
    _backend: PhantomData<B>,
}

impl<B: Backend> TensorRingBuffer<B> {
    /// Create a new TensorRingBuffer with pre-allocated tensors
    pub fn new(capacity: usize, state_dim: usize, device: &B::Device) -> Self {
        Self {
            states: Tensor::zeros([capacity, state_dim], device),
            next_states: Tensor::zeros([capacity, state_dim], device),
            actions: Tensor::zeros([capacity], device),
            rewards: Tensor::zeros([capacity], device),
            dones: Tensor::zeros([capacity], device),
            head: 0,
            size: 0,
            capacity,
            state_dim,
            _backend: PhantomData,
        }
    }

    /// Push a transition to the buffer - O(1) scatter write
    ///
    /// Uses `slice_assign` for O(1) in-place writes instead of O(n) tensor concatenation.
    /// This eliminates GPU memory churn from cloning and reallocating the entire buffer.
    pub fn push(
        &mut self,
        state: Tensor<B, 1>,
        action: usize,
        reward: f32,
        next_state: Tensor<B, 1>,
        done: bool,
    ) {
        let idx = self.head;
        let device = &self.states.device();

        // Create 2D tensors [1, state_dim] for slice_assign on states
        let state_2d = state.unsqueeze::<3>().reshape([1, self.state_dim]);
        let next_state_2d = next_state.unsqueeze::<3>().reshape([1, self.state_dim]);

        // Create 1D tensors [1] for slice_assign on actions, rewards, dones
        let action_tensor = Tensor::from_data(
            TensorData::new(vec![action as i64], [1]).convert::<i64>(),
            device,
        );
        let reward_tensor =
            Tensor::from_data(TensorData::new(vec![reward], [1]).convert::<f32>(), device);
        let done_tensor = Tensor::from_data(
            TensorData::new(vec![if done { 1.0f32 } else { 0.0f32 }], [1]).convert::<f32>(),
            device,
        );

        // O(1) scatter write using slice_assign - writes directly at index
        // Note: clone() is cheap - it increments reference count, doesn't copy GPU memory
        // The actual slice_assign operation is O(1) - only writes the slice, not the whole tensor
        self.states = self
            .states
            .clone()
            .slice_assign([idx..idx + 1, 0..self.state_dim], state_2d);
        self.next_states = self
            .next_states
            .clone()
            .slice_assign([idx..idx + 1, 0..self.state_dim], next_state_2d);
        self.actions = self
            .actions
            .clone()
            .slice_assign([idx..idx + 1], action_tensor);
        self.rewards = self
            .rewards
            .clone()
            .slice_assign([idx..idx + 1], reward_tensor);
        self.dones = self.dones.clone().slice_assign([idx..idx + 1], done_tensor);

        // Update circular buffer pointers
        self.head = (self.head + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
    }

    /// Push a transition from CPU data (convenience method)
    pub fn push_from_cpu(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
        device: &B::Device,
    ) {
        let state_tensor = Tensor::from_data(
            TensorData::new(state, [self.state_dim]).convert::<f32>(),
            device,
        );
        let next_state_tensor = Tensor::from_data(
            TensorData::new(next_state, [self.state_dim]).convert::<f32>(),
            device,
        );
        self.push(state_tensor, action, reward, next_state_tensor, done);
    }

    /// Push a batch of transitions efficiently (single operation instead of N individual pushes)
    ///
    /// This method is much more efficient than calling `push_from_cpu` multiple times,
    /// as it avoids repeated tensor concatenation and allocation overhead.
    ///
    /// # Arguments
    /// * `states` - Vec of state vectors [batch_size, state_dim]
    /// * `actions` - Vec of action indices [batch_size]
    /// * `rewards` - Vec of reward values [batch_size]
    /// * `next_states` - Vec of next state vectors [batch_size, state_dim]
    /// * `dones` - Vec of done flags [batch_size]
    /// * `device` - Device for tensor operations
    pub fn push_batch_optimized(
        &mut self,
        states: Vec<Vec<f32>>,
        actions: Vec<usize>,
        rewards: Vec<f32>,
        next_states: Vec<Vec<f32>>,
        dones: Vec<bool>,
        device: &B::Device,
    ) {
        let batch_size = states.len();
        if batch_size == 0 {
            return;
        }

        // Flatten all states into single tensors [batch_size, state_dim]
        let states_flat: Vec<f32> = states.into_iter().flatten().collect();
        let next_states_flat: Vec<f32> = next_states.into_iter().flatten().collect();

        // Create batch tensors
        let states_batch = Tensor::from_data(
            TensorData::new(states_flat, [batch_size, self.state_dim]).convert::<f32>(),
            device,
        );
        let next_states_batch = Tensor::from_data(
            TensorData::new(next_states_flat, [batch_size, self.state_dim]).convert::<f32>(),
            device,
        );
        let actions_batch = Tensor::from_data(
            TensorData::new(actions.iter().map(|&a| a as i64).collect(), [batch_size])
                .convert::<i64>(),
            device,
        );
        let rewards_batch = Tensor::from_data(
            TensorData::new(rewards, [batch_size]).convert::<f32>(),
            device,
        );
        let dones_batch = Tensor::from_data(
            TensorData::new(
                dones
                    .iter()
                    .map(|&d| if d { 1.0f32 } else { 0.0f32 })
                    .collect(),
                [batch_size],
            )
            .convert::<f32>(),
            device,
        );

        // Write batch to ring buffer positions
        // Calculate start and end positions in circular buffer
        let start_idx = self.head;
        let end_idx = (self.head + batch_size) % self.capacity;

        // Handle wraparound: split into two writes if needed
        if start_idx + batch_size <= self.capacity {
            // No wraparound - single contiguous write
            self.write_batch_at_index(
                states_batch,
                actions_batch,
                rewards_batch,
                next_states_batch,
                dones_batch,
                start_idx,
            );
        } else {
            // Wraparound - split into two writes
            let first_part_size = self.capacity - start_idx;

            // First part: from start_idx to end of buffer
            let states_first = states_batch
                .clone()
                .slice([0..first_part_size, 0..self.state_dim]);
            let next_states_first = next_states_batch
                .clone()
                .slice([0..first_part_size, 0..self.state_dim]);
            let actions_first = actions_batch.clone().slice([0..first_part_size]);
            let rewards_first = rewards_batch.clone().slice([0..first_part_size]);
            let dones_first = dones_batch.clone().slice([0..first_part_size]);

            self.write_batch_at_index(
                states_first,
                actions_first,
                rewards_first,
                next_states_first,
                dones_first,
                start_idx,
            );

            // Second part: from beginning of buffer
            let states_second =
                states_batch.slice([first_part_size..batch_size, 0..self.state_dim]);
            let next_states_second =
                next_states_batch.slice([first_part_size..batch_size, 0..self.state_dim]);
            let actions_second = actions_batch.slice([first_part_size..batch_size]);
            let rewards_second = rewards_batch.slice([first_part_size..batch_size]);
            let dones_second = dones_batch.slice([first_part_size..batch_size]);

            self.write_batch_at_index(
                states_second,
                actions_second,
                rewards_second,
                next_states_second,
                dones_second,
                0,
            );
        }

        // Update circular buffer pointers
        self.head = end_idx;
        self.size = (self.size + batch_size).min(self.capacity);
    }

    /// Write a batch of tensors at a specific index in the buffer
    fn write_batch_at_index(
        &mut self,
        states: Tensor<B, 2>,
        actions: Tensor<B, 1, burn::tensor::Int>,
        rewards: Tensor<B, 1>,
        next_states: Tensor<B, 2>,
        dones: Tensor<B, 1>,
        start_idx: usize,
    ) {
        let batch_size = states.shape().dims[0];

        // Build new tensors by concatenating: before + batch + after
        let before_count = start_idx;
        let after_count = self.capacity - start_idx - batch_size;

        if before_count > 0 && after_count > 0 {
            // Both before and after sections exist
            self.states = Tensor::cat(
                vec![
                    self.states
                        .clone()
                        .slice([0..before_count, 0..self.state_dim]),
                    states,
                    self.states
                        .clone()
                        .slice([start_idx + batch_size..self.capacity, 0..self.state_dim]),
                ],
                0,
            );
            self.next_states = Tensor::cat(
                vec![
                    self.next_states
                        .clone()
                        .slice([0..before_count, 0..self.state_dim]),
                    next_states,
                    self.next_states
                        .clone()
                        .slice([start_idx + batch_size..self.capacity, 0..self.state_dim]),
                ],
                0,
            );
            self.actions = Tensor::cat(
                vec![
                    self.actions.clone().slice([0..before_count]),
                    actions,
                    self.actions
                        .clone()
                        .slice([start_idx + batch_size..self.capacity]),
                ],
                0,
            );
            self.rewards = Tensor::cat(
                vec![
                    self.rewards.clone().slice([0..before_count]),
                    rewards,
                    self.rewards
                        .clone()
                        .slice([start_idx + batch_size..self.capacity]),
                ],
                0,
            );
            self.dones = Tensor::cat(
                vec![
                    self.dones.clone().slice([0..before_count]),
                    dones,
                    self.dones
                        .clone()
                        .slice([start_idx + batch_size..self.capacity]),
                ],
                0,
            );
        } else if before_count > 0 {
            // Only before section exists (writing at end of buffer)
            self.states = Tensor::cat(
                vec![
                    self.states
                        .clone()
                        .slice([0..before_count, 0..self.state_dim]),
                    states,
                ],
                0,
            );
            self.next_states = Tensor::cat(
                vec![
                    self.next_states
                        .clone()
                        .slice([0..before_count, 0..self.state_dim]),
                    next_states,
                ],
                0,
            );
            self.actions = Tensor::cat(
                vec![self.actions.clone().slice([0..before_count]), actions],
                0,
            );
            self.rewards = Tensor::cat(
                vec![self.rewards.clone().slice([0..before_count]), rewards],
                0,
            );
            self.dones = Tensor::cat(vec![self.dones.clone().slice([0..before_count]), dones], 0);
        } else if after_count > 0 {
            // Only after section exists (writing at beginning of buffer)
            self.states = Tensor::cat(
                vec![
                    states,
                    self.states
                        .clone()
                        .slice([batch_size..self.capacity, 0..self.state_dim]),
                ],
                0,
            );
            self.next_states = Tensor::cat(
                vec![
                    next_states,
                    self.next_states
                        .clone()
                        .slice([batch_size..self.capacity, 0..self.state_dim]),
                ],
                0,
            );
            self.actions = Tensor::cat(
                vec![
                    actions,
                    self.actions.clone().slice([batch_size..self.capacity]),
                ],
                0,
            );
            self.rewards = Tensor::cat(
                vec![
                    rewards,
                    self.rewards.clone().slice([batch_size..self.capacity]),
                ],
                0,
            );
            self.dones = Tensor::cat(
                vec![dones, self.dones.clone().slice([batch_size..self.capacity])],
                0,
            );
        } else {
            // Buffer size equals batch size - replace everything
            self.states = states;
            self.next_states = next_states;
            self.actions = actions;
            self.rewards = rewards;
            self.dones = dones;
        }
    }

    /// Sample a random batch from the buffer - ZERO COPY
    pub fn sample(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Option<TensorTransitionBatch<B>> {
        if self.size < batch_size {
            return None;
        }

        // Generate random indices
        let mut rng = rng();
        let indices: Vec<i64> = (0..batch_size)
            .map(|_| rng.random_range(0..self.size) as i64)
            .collect();

        // Convert to logical indices (accounting for circular buffer)
        let logical_indices: Vec<i64> = indices
            .iter()
            .map(|&idx| ((self.head + idx as usize) % self.capacity) as i64)
            .collect();

        // Create index tensor [batch_size, state_dim] for states, [batch_size, 1] for 1D tensors
        let indices_1d = Tensor::from_data(
            TensorData::new(logical_indices.clone(), [batch_size, 1]).convert::<i64>(),
            device,
        );

        // For 2D states: expand indices to [batch_size, state_dim]
        let indices_2d = indices_1d.clone().expand([batch_size, self.state_dim]);

        // Gather batch using gather (dim 0 for 2D states/next_states)
        // For 1D tensors, we need to unsqueeze first, gather, then squeeze
        let batch_states = self.states.clone().gather(0, indices_2d.clone());

        // For 1D tensors: unsqueeze to 2D, gather, squeeze back
        let actions_2d = self
            .actions
            .clone()
            .unsqueeze::<3>()
            .reshape([self.capacity, 1]);
        let batch_actions = actions_2d.gather(0, indices_1d.clone()).squeeze();

        let rewards_2d = self
            .rewards
            .clone()
            .unsqueeze::<3>()
            .reshape([self.capacity, 1]);
        let batch_rewards = rewards_2d.gather(0, indices_1d.clone()).squeeze();

        let batch_next_states = self.next_states.clone().gather(0, indices_2d.clone());

        let dones_2d = self
            .dones
            .clone()
            .unsqueeze::<3>()
            .reshape([self.capacity, 1]);
        let batch_dones = dones_2d.gather(0, indices_1d).squeeze();

        Some(TensorTransitionBatch {
            states: batch_states,
            actions: batch_actions,
            rewards: batch_rewards,
            next_states: batch_next_states,
            dones: batch_dones,
        })
    }

    /// Get buffer length
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get state dimension
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Sample a batch directly as GPU tensors without CPU→GPU transfer.
    ///
    /// This is much more efficient than the Dataset implementation which
    /// converts to CPU data and back. Use this for GPU training.
    ///
    /// # Arguments
    /// * `batch_size` - Number of transitions to sample
    /// * `device` - Device for tensor operations (should be GPU)
    ///
    /// # Returns
    /// TensorTransitionBatch with all tensors on GPU, or None if insufficient samples
    pub fn sample_gpu_batch(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Option<TensorTransitionBatch<B>> {
        if self.size < batch_size {
            return None;
        }

        // Get logical indices
        let indices = self.sample_indices(batch_size);

        if indices.is_empty() {
            // Return empty batch
            return Some(TensorTransitionBatch {
                states: Tensor::zeros([0, self.state_dim], device),
                actions: Tensor::zeros([0], device),
                rewards: Tensor::zeros([0], device),
                next_states: Tensor::zeros([0, self.state_dim], device),
                dones: Tensor::zeros([0], device),
            });
        }

        // Convert indices to i64 for tensor creation
        let indices_i64: Vec<i64> = indices.iter().map(|&i| i as i64).collect();

        // Create index tensor [batch_size, 1] for gathering (matching sample() pattern)
        let indices_1d = Tensor::from_data(
            TensorData::new(indices_i64.clone(), [batch_size, 1]).convert::<i64>(),
            device,
        );

        // For 2D states: expand indices to [batch_size, state_dim]
        let indices_2d = indices_1d.clone().expand([batch_size, self.state_dim]);

        // Gather states directly on GPU [batch_size, state_dim]
        let states_batch: Tensor<B, 2> = self.states.clone().gather(0, indices_2d.clone());
        let next_states_batch: Tensor<B, 2> = self.next_states.clone().gather(0, indices_2d);

        // For 1D tensors: unsqueeze to 2D, gather, squeeze back (matching sample() pattern)
        let actions_2d = self
            .actions
            .clone()
            .unsqueeze::<3>()
            .reshape([self.capacity, 1]);
        let actions_batch = actions_2d.gather(0, indices_1d.clone()).squeeze();

        let rewards_2d = self
            .rewards
            .clone()
            .unsqueeze::<3>()
            .reshape([self.capacity, 1]);
        let rewards_batch: Tensor<B, 1> = rewards_2d.gather(0, indices_1d.clone()).squeeze();

        let dones_2d = self
            .dones
            .clone()
            .unsqueeze::<3>()
            .reshape([self.capacity, 1]);
        let dones_batch: Tensor<B, 1> = dones_2d.gather(0, indices_1d).squeeze();

        Some(TensorTransitionBatch {
            states: states_batch,
            actions: actions_batch,
            rewards: rewards_batch,
            next_states: next_states_batch,
            dones: dones_batch,
        })
    }

    /// Generate random logical indices for sampling.
    ///
    /// # Arguments
    /// * `batch_size` - Number of indices to generate
    ///
    /// # Returns
    /// Vector of logical indices into the ring buffer
    fn sample_indices(&self, batch_size: usize) -> Vec<usize> {
        if self.size < batch_size {
            return Vec::new();
        }

        let mut rng = rng();
        let mut indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.size))
            .collect();

        // Convert to logical indices (accounting for circular buffer)
        for idx in &mut indices {
            *idx = if self.size < self.capacity {
                // Not wrapped yet - items are sequential from 0
                *idx
            } else {
                // Wrapped - items start at head
                (self.head + *idx) % self.capacity
            };
        }

        indices
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.head = 0;
        self.size = 0;
    }
}

impl<B: Backend> Clone for TensorRingBuffer<B> {
    fn clone(&self) -> Self {
        Self {
            states: self.states.clone(),
            next_states: self.next_states.clone(),
            actions: self.actions.clone(),
            rewards: self.rewards.clone(),
            dones: self.dones.clone(),
            head: self.head,
            size: self.size,
            capacity: self.capacity,
            state_dim: self.state_dim,
            _backend: PhantomData,
        }
    }
}

/// Batch of transitions with tensors on device
#[derive(Debug, Clone)]
pub struct TensorTransitionBatch<B: Backend> {
    /// States: [batch_size, state_dim]
    pub states: Tensor<B, 2>,
    /// Actions: [batch_size]
    pub actions: Tensor<B, 1, burn::tensor::Int>,
    /// Rewards: [batch_size]
    pub rewards: Tensor<B, 1>,
    /// Next states: [batch_size, state_dim]
    pub next_states: Tensor<B, 2>,
    /// Done flags: [batch_size]
    pub dones: Tensor<B, 1>,
}

// ============================================================================
// Dataset Trait Implementation for Burn DataLoader
// ============================================================================

impl<B: Backend + Send + Sync> Dataset<Transition> for TensorRingBuffer<B> {
    fn get(&self, index: usize) -> Option<Transition> {
        if index >= self.size {
            return None;
        }

        // Map logical index to physical circular buffer position
        // When buffer is not full (size < capacity), items are at [0..size]
        // When buffer is full and wrapped, items start at head
        let physical_idx = if self.size < self.capacity {
            // Not wrapped yet - items are sequential from 0
            index
        } else {
            // Wrapped - items start at head
            (self.head + index) % self.capacity
        };

        // Create index tensor for gathering [1, state_dim] for states, [1, 1] for 1D tensors
        let idx_1d = Tensor::from_data(
            TensorData::new(vec![physical_idx as i64], [1, 1]).convert::<i64>(),
            &self.states.device(),
        );
        let idx_2d = idx_1d.clone().expand([1, self.state_dim]);

        // Gather single element at index
        let state_slice = self.states.clone().gather(0, idx_2d.clone());
        let next_state_slice = self.next_states.clone().gather(0, idx_2d);

        // For 1D tensors
        let actions_2d = self
            .actions
            .clone()
            .unsqueeze::<3>()
            .reshape([self.capacity, 1]);
        let action_slice = actions_2d.gather(0, idx_1d.clone());

        let rewards_2d = self
            .rewards
            .clone()
            .unsqueeze::<3>()
            .reshape([self.capacity, 1]);
        let reward_slice = rewards_2d.gather(0, idx_1d.clone());

        let dones_2d = self
            .dones
            .clone()
            .unsqueeze::<3>()
            .reshape([self.capacity, 1]);
        let done_slice = dones_2d.gather(0, idx_1d);

        // Convert tensors to CPU data
        let state_data: burn::tensor::TensorData = state_slice.into_data().convert::<f32>();
        let next_state_data: burn::tensor::TensorData =
            next_state_slice.into_data().convert::<f32>();
        let action_data: burn::tensor::TensorData = action_slice.into_data().convert::<i64>();
        let reward_data: burn::tensor::TensorData = reward_slice.into_data().convert::<f32>();
        let done_data: burn::tensor::TensorData = done_slice.into_data().convert::<f32>();

        let state: Vec<f32> = state_data.as_slice::<f32>().unwrap().to_vec();
        let next_state: Vec<f32> = next_state_data.as_slice::<f32>().unwrap().to_vec();
        let action: usize = action_data.as_slice::<i64>().unwrap()[0] as usize;
        let reward: f32 = reward_data.as_slice::<f32>().unwrap()[0];
        let done: bool = done_data.as_slice::<f32>().unwrap()[0] > 0.5;

        Some(Transition {
            state,
            action,
            reward,
            next_state,
            done,
        })
    }

    fn len(&self) -> usize {
        self.size
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = burn::backend::Autodiff<NdArray<f32>>;

    fn create_test_state(
        device: &<TestBackend as Backend>::Device,
        value: f32,
    ) -> Tensor<TestBackend, 1> {
        Tensor::from_data(
            TensorData::new(vec![value; 4], [4]).convert::<f32>(),
            device,
        )
    }

    #[test]
    fn test_tensor_ring_buffer_push_basic() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        // Push a few transitions
        for i in 0..5 {
            let state = create_test_state(&device, i as f32);
            let next_state = create_test_state(&device, (i + 10) as f32);
            buffer.push(state, i, i as f32, next_state, false);
        }

        assert_eq!(buffer.len(), 5);
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_tensor_ring_buffer_wraparound() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(5, 4, &device);

        // Push more than capacity
        for i in 0..10 {
            let state = create_test_state(&device, i as f32);
            let next_state = create_test_state(&device, (i + 100) as f32);
            buffer.push(state, i % 10, i as f32, next_state, i % 3 == 0);
        }

        assert_eq!(buffer.len(), 5);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_tensor_ring_buffer_sample() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        // Fill buffer
        for i in 0..8 {
            let state = create_test_state(&device, i as f32);
            let next_state = create_test_state(&device, i as f32);
            buffer.push(state, i, i as f32, next_state, false);
        }

        // Sample should work
        let batch = buffer.sample(4, &device);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.states.dims()[0], 4);
        assert_eq!(batch.actions.dims()[0], 4);
    }

    #[test]
    fn test_dataset_get_basic() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        // Push items
        for i in 0..5 {
            let state = create_test_state(&device, i as f32);
            let next_state = create_test_state(&device, i as f32);
            buffer.push(state, i, i as f32, next_state, false);
        }

        // Get first item (oldest)
        let t0 = buffer.get(0).expect("should have item");
        assert_eq!(t0.state[0], 0.0);
        assert_eq!(t0.action, 0);

        // Get last item (newest)
        let t4 = buffer.get(4).expect("should have item");
        assert_eq!(t4.state[0], 4.0);
        assert_eq!(t4.action, 4);
    }

    #[test]
    fn test_dataset_get_wrapped() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(5, 4, &device);

        // Push 10 items (wraps twice)
        for i in 0..10 {
            let state = create_test_state(&device, i as f32);
            let next_state = create_test_state(&device, i as f32);
            buffer.push(state, i, i as f32, next_state, false);
        }

        assert_eq!(buffer.len(), 5);

        // After 10 pushes with capacity 5:
        // head = 0, items are [5, 6, 7, 8, 9]
        // logical 0 -> physical 0 -> item 5
        let t0 = buffer.get(0).expect("should have item");
        assert_eq!(t0.state[0], 5.0);

        // logical 4 -> physical 4 -> item 9
        let t4 = buffer.get(4).expect("should have item");
        assert_eq!(t4.state[0], 9.0);
    }

    #[test]
    fn test_push_from_cpu() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        buffer.push_from_cpu(
            vec![1.0, 2.0, 3.0, 4.0],
            5,
            10.0,
            vec![5.0, 6.0, 7.0, 8.0],
            true,
            &device,
        );

        assert_eq!(buffer.len(), 1);

        let t = buffer.get(0).expect("should have transition");
        assert_eq!(t.state, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.action, 5);
        assert_eq!(t.reward, 10.0);
        assert!(t.done);
    }

    #[test]
    fn test_clear_buffer() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        for _ in 0..5 {
            let state = Tensor::zeros([4], &device);
            let next_state = Tensor::zeros([4], &device);
            buffer.push(state, 0, 0.0, next_state, false);
        }

        assert_eq!(buffer.len(), 5);
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_sample_none_when_empty() {
        let device = Default::default();
        let buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        assert!(buffer.sample(5, &device).is_none());
    }

    #[test]
    fn test_sample_gpu_batch_basic() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        // Push items with distinct values
        for i in 0..8 {
            let state = create_test_state(&device, i as f32);
            let next_state = create_test_state(&device, i as f32);
            buffer.push(state, i, i as f32, next_state, false);
        }

        // Sample GPU batch should work
        let batch = buffer.sample_gpu_batch(4, &device);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.states.dims()[0], 4);
        assert_eq!(batch.states.dims()[1], 4);
        assert_eq!(batch.actions.dims()[0], 4);
        assert_eq!(batch.rewards.dims()[0], 4);
        assert_eq!(batch.next_states.dims()[0], 4);
        assert_eq!(batch.dones.dims()[0], 4);
    }

    #[test]
    fn test_sample_gpu_batch_none_when_empty() {
        let device = Default::default();
        let buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        assert!(buffer.sample_gpu_batch(5, &device).is_none());
    }

    #[test]
    fn test_sample_gpu_batch_wrapped() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(5, 4, &device);

        // Push 10 items (wraps twice)
        for i in 0..10 {
            let state = create_test_state(&device, i as f32);
            let next_state = create_test_state(&device, i as f32);
            buffer.push(state, i, i as f32, next_state, false);
        }

        assert_eq!(buffer.len(), 5);

        // Sample should work with wrapped buffer
        let batch = buffer.sample_gpu_batch(3, &device);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.states.dims()[0], 3);
    }

    #[test]
    fn test_push_batch_optimized_basic() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        // Push batch of 5 transitions
        let states = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![3.0, 4.0, 5.0, 6.0],
            vec![4.0, 5.0, 6.0, 7.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];
        let actions = vec![0, 1, 2, 3, 4];
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let next_states = vec![
            vec![10.0, 20.0, 30.0, 40.0],
            vec![20.0, 30.0, 40.0, 50.0],
            vec![30.0, 40.0, 50.0, 60.0],
            vec![40.0, 50.0, 60.0, 70.0],
            vec![50.0, 60.0, 70.0, 80.0],
        ];
        let dones = vec![false, false, false, false, true];

        buffer.push_batch_optimized(states, actions, rewards, next_states, dones, &device);

        assert_eq!(buffer.len(), 5);

        // Verify first transition
        let t0 = buffer.get(0).expect("should have transition");
        assert_eq!(t0.state, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t0.action, 0);
        assert_eq!(t0.reward, 1.0);
        assert_eq!(t0.next_state, vec![10.0, 20.0, 30.0, 40.0]);
        assert!(!t0.done);

        // Verify last transition
        let t4 = buffer.get(4).expect("should have transition");
        assert_eq!(t4.state, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(t4.action, 4);
        assert_eq!(t4.reward, 5.0);
        assert!(t4.done);
    }

    #[test]
    fn test_push_batch_optimized_wraparound() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(5, 4, &device);

        // First, push 3 items to move head forward
        buffer.push_from_cpu(vec![1.0; 4], 0, 1.0, vec![10.0; 4], false, &device);
        buffer.push_from_cpu(vec![2.0; 4], 1, 2.0, vec![20.0; 4], false, &device);
        buffer.push_from_cpu(vec![3.0; 4], 2, 3.0, vec![30.0; 4], false, &device);

        assert_eq!(buffer.len(), 3);

        // Now push batch of 4 (will wrap around)
        let states = vec![vec![4.0; 4], vec![5.0; 4], vec![6.0; 4], vec![7.0; 4]];
        let actions = vec![3, 4, 5, 6];
        let rewards = vec![4.0, 5.0, 6.0, 7.0];
        let next_states = vec![vec![40.0; 4], vec![50.0; 4], vec![60.0; 4], vec![70.0; 4]];
        let dones = vec![false, false, false, false];

        buffer.push_batch_optimized(states, actions, rewards, next_states, dones, &device);

        // Buffer should be full (capacity 5)
        assert_eq!(buffer.len(), 5);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_push_batch_optimized_empty_batch() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(10, 4, &device);

        // Push empty batch - should not panic
        buffer.push_batch_optimized(vec![], vec![], vec![], vec![], vec![], &device);

        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_push_batch_optimized_full_capacity() {
        let device = Default::default();
        let mut buffer = TensorRingBuffer::<TestBackend>::new(5, 4, &device);

        // Push batch equal to capacity
        let states: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 4]).collect();
        let actions: Vec<usize> = (0..5).collect();
        let rewards: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let next_states: Vec<Vec<f32>> = (0..5).map(|i| vec![(i + 10) as f32; 4]).collect();
        let dones: Vec<bool> = vec![false; 5];

        buffer.push_batch_optimized(states, actions, rewards, next_states, dones, &device);

        assert_eq!(buffer.len(), 5);
        assert!(buffer.is_full());

        // Verify all items
        for i in 0..5 {
            let t = buffer.get(i).expect("should have transition");
            assert_eq!(t.state[0], i as f32);
            assert_eq!(t.action, i);
            assert_eq!(t.reward, i as f32);
            assert_eq!(t.next_state[0], (i + 10) as f32);
        }
    }
}
