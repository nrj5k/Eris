//! Hybrid CPU/GPU Ring Buffer for Experience Replay
//!
//! This module provides `HybridRingBuffer`, which stores transitions on CPU
//! and converts to GPU tensors only during sampling. This matches Metis's
//! proven pattern and eliminates VRAM leaks from `slice_assign` operations.
//!
//! ## Design
//!
//! - **CPU Storage**: Transitions stored as `Vec<f32>` (no GPU memory)
//! - **GPU Conversion**: Only happens during `sample_batch()` (once per batch)
//! - **O(1) Push**: No GPU allocations during push operations
//! - **No VRAM Leak**: Memory stays constant during training

use crate::utils::backend_diagnostics::{detect_backend, log_backend_info};
use crate::utils::timing::OneTimeDiag;
use burn::tensor::{backend::Backend, Tensor, TensorData};
use burnme_rly::buffer::TensorTransitionBatch;
use tracing;

/// Hybrid buffer: stores transitions on CPU, converts to GPU only on sampling
/// This matches Metis's proven pattern - no VRAM leaks!
pub struct HybridRingBuffer<B: Backend> {
    // CPU-side storage (no VRAM leak!)
    states: Vec<Vec<f32>>,
    actions: Vec<usize>,
    rewards: Vec<f32>,
    next_states: Vec<Vec<f32>>,
    dones: Vec<bool>,

    head: usize,
    size: usize,
    capacity: usize,
    state_dim: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> HybridRingBuffer<B> {
    /// Create a new hybrid ring buffer with CPU storage
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of transitions to store
    /// * `state_dim` - Dimension of state vectors
    pub fn new(capacity: usize, state_dim: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            next_states: Vec::with_capacity(capacity),
            dones: Vec::with_capacity(capacity),
            head: 0,
            size: 0,
            capacity,
            state_dim,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Push transition - truly O(1), NO GPU allocations!
    ///
    /// This is the key difference from `TensorRingBuffer`:
    /// - No tensor creation
    /// - No GPU memory allocation
    /// - No slice_assign operations
    ///
    /// # Arguments
    /// * `state` - Current state vector
    /// * `action` - Action index
    /// * `reward` - Reward value
    /// * `next_state` - Next state vector
    /// * `done` - Episode termination flag
    pub fn push(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) {
        if self.states.len() < self.capacity {
            // Buffer not full yet, just append
            self.states.push(state);
            self.actions.push(action);
            self.rewards.push(reward);
            self.next_states.push(next_state);
            self.dones.push(done);
        } else {
            // Circular buffer - overwrite at head
            self.states[self.head] = state;
            self.actions[self.head] = action;
            self.rewards[self.head] = reward;
            self.next_states[self.head] = next_state;
            self.dones[self.head] = done;
        }

        self.head = (self.head + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
    }

    /// Push a batch of transitions efficiently.
    ///
    /// Since push() is O(1) with CPU Vecs, this is O(batch_size) with zero GPU allocations.
    ///
    /// # Arguments
    /// * `states` - Vector of state vectors
    /// * `actions` - Vector of action indices
    /// * `rewards` - Vector of reward values
    /// * `next_states` - Vector of next state vectors
    /// * `dones` - Vector of episode termination flags
    pub fn push_batch(
        &mut self,
        states: Vec<Vec<f32>>,
        actions: Vec<usize>,
        rewards: Vec<f32>,
        next_states: Vec<Vec<f32>>,
        dones: Vec<bool>,
    ) {
        let batch_size = states.len();
        for i in 0..batch_size {
            self.push(
                states[i].clone(),
                actions[i],
                rewards[i],
                next_states[i].clone(),
                dones[i],
            );
        }
    }

    /// Sample batch - convert to GPU tensors ONLY HERE
    ///
    /// This is where GPU allocation happens - once per batch, not per push.
    /// The conversion cost is amortized over the entire batch.
    ///
    /// # Arguments
    /// * `batch_size` - Number of transitions to sample
    /// * `device` - GPU device for tensor creation
    ///
    /// # Returns
    /// `Some(TensorTransitionBatch)` if enough samples, `None` otherwise
    pub fn sample_batch(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Option<TensorTransitionBatch<B>> {
        tracing::debug!(
            "sample_batch() called: batch_size={}, buffer_len={}",
            batch_size,
            self.size
        );
        tracing::debug!("sample_batch backend: {:?}", detect_backend::<B>());

        // GPU DIAGNOSTIC: One-time device verification using utility
        static DIAG: OneTimeDiag = OneTimeDiag::new();

        if DIAG.should_print() {
            log_backend_info::<B>("HybridRingBuffer::sample_batch", device);
        }

        if self.size < batch_size {
            tracing::debug!(
                "sample_batch returning None: buffer size ({}) < batch_size ({})",
                self.size,
                batch_size
            );
            return None;
        }

        tracing::trace!("Generating random indices for {} samples", batch_size);

        // Random indices using rand crate
        use rand::prelude::IteratorRandom;
        let mut rng = rand::rng();
        let indices: Vec<usize> = (0..self.size).sample(&mut rng, batch_size);

        tracing::trace!("Got {} random indices, gathering samples", indices.len());

        // Gather samples from CPU storage
        let mut batch_states = Vec::with_capacity(batch_size * self.state_dim);
        let mut batch_actions = Vec::with_capacity(batch_size);
        let mut batch_rewards = Vec::with_capacity(batch_size);
        let mut batch_next_states = Vec::with_capacity(batch_size * self.state_dim);
        let mut batch_dones = Vec::with_capacity(batch_size);

        for &idx in &indices {
            batch_states.extend_from_slice(&self.states[idx]);
            batch_actions.push(self.actions[idx] as i32);
            batch_rewards.push(self.rewards[idx]);
            batch_next_states.extend_from_slice(&self.next_states[idx]);
            batch_dones.push(self.dones[idx]);
        }

        tracing::debug!(
            "Converting to GPU tensors: batch_size={}, state_dim={}",
            batch_size,
            self.state_dim
        );

        // Convert to GPU tensors ONCE per sample (not per push!)
        // Using rank-2 format: [batch_size, 1] for actions/rewards/dones
        let states_tensor = Tensor::from_data(
            TensorData::new(batch_states, [batch_size, self.state_dim]).convert::<f32>(),
            device,
        );
        let actions_tensor = Tensor::from_data(
            TensorData::new(batch_actions, [batch_size, 1]).convert::<i32>(),
            device,
        );
        let rewards_tensor = Tensor::from_data(
            TensorData::new(batch_rewards, [batch_size, 1]).convert::<f32>(),
            device,
        );
        let next_states_tensor = Tensor::from_data(
            TensorData::new(batch_next_states, [batch_size, self.state_dim]).convert::<f32>(),
            device,
        );
        // Convert dones to f32 (1.0 for true, 0.0 for false), shape [batch_size, 1]
        let batch_dones_f32: Vec<f32> = batch_dones
            .iter()
            .map(|&d| if d { 1.0f32 } else { 0.0f32 })
            .collect();
        let dones_tensor = Tensor::from_data(
            TensorData::new(batch_dones_f32, [batch_size, 1]).convert::<f32>(),
            device,
        );

        tracing::debug!(
            "sample_batch SUCCESS: states shape [{}, {}]",
            batch_size,
            self.state_dim
        );

        Some(TensorTransitionBatch {
            states: states_tensor,
            actions: actions_tensor,
            rewards: rewards_tensor,
            next_states: next_states_tensor,
            dones: dones_tensor,
        })
    }

    /// Get buffer length (number of stored transitions)
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Check if buffer has enough samples for a batch
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.size >= batch_size
    }

    /// Get state dimension
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.head = 0;
        self.size = 0;
        self.states.clear();
        self.actions.clear();
        self.rewards.clear();
        self.next_states.clear();
        self.dones.clear();
    }

    /// Fill the buffer with random transitions for warmup initialization.
    ///
    /// This allows the training loop to start with a pre-filled buffer,
    /// avoiding the slow cold-start phase where the GPU sits idle waiting
    /// for enough samples.
    ///
    /// # Arguments
    /// * `num_transitions` - Number of random transitions to generate
    /// * `action_dim` - Number of possible actions (actions sampled from [0, action_dim))
    /// * `state_dim` - Dimension of state/observation vectors
    ///
    /// # Note
    /// Random transitions have zero reward and are not terminal (done=false).
    /// States and next_states are filled with small random values.
    /// This is standard DQN practice (Mnih et al., 2015).
    pub fn fill_random(&mut self, num_transitions: usize, action_dim: usize, state_dim: usize) {
        use rand::RngExt;
        let mut rng = rand::rng();
        let count = num_transitions.min(self.capacity - self.size);
        for _ in 0..count {
            let state: Vec<f32> = (0..state_dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let action = rng.random_range(0..action_dim);
            let reward = 0.0; // Zero reward for random warmup
            let next_state: Vec<f32> = (0..state_dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            let done = false;
            self.push(state, action, reward, next_state, done);
        }
        tracing::info!(
            "[STAGE:WARMUP] Pre-filled buffer with {} random transitions (action_dim={}, state_dim={})",
            count, action_dim, state_dim
        );
    }
}

impl<B: Backend> Clone for HybridRingBuffer<B> {
    fn clone(&self) -> Self {
        Self {
            states: self.states.clone(),
            actions: self.actions.clone(),
            rewards: self.rewards.clone(),
            next_states: self.next_states.clone(),
            dones: self.dones.clone(),
            head: self.head,
            size: self.size,
            capacity: self.capacity,
            state_dim: self.state_dim,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::prelude::Backend;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_hybrid_buffer_push_basic() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(100, 32);

        for i in 0..50 {
            buffer.push(
                vec![i as f32; 32],
                i % 10,
                i as f32,
                vec![(i + 1) as f32; 32],
                false,
            );
        }

        assert_eq!(buffer.len(), 50);
        assert!(!buffer.is_full());
        assert_eq!(buffer.capacity(), 100);
    }

    #[test]
    fn test_hybrid_buffer_wraparound() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(10, 4);

        // Push more than capacity
        for i in 0..20 {
            buffer.push(
                vec![i as f32; 4],
                i % 10,
                i as f32,
                vec![(i + 100) as f32; 4],
                i % 3 == 0,
            );
        }

        assert_eq!(buffer.len(), 10);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_hybrid_buffer_sample_batch() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = HybridRingBuffer::<TestBackend>::new(100, 8);

        // Fill buffer
        for i in 0..50 {
            buffer.push(
                vec![i as f32; 8],
                i % 10,
                i as f32,
                vec![(i + 1) as f32; 8],
                false,
            );
        }

        // Sample batch should work
        let batch = buffer.sample_batch(16, &device);
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.states.dims()[0], 16);
        assert_eq!(batch.states.dims()[1], 8);
        assert_eq!(batch.actions.dims()[0], 16);
        assert_eq!(batch.rewards.dims()[0], 16);
        assert_eq!(batch.next_states.dims()[0], 16);
        assert_eq!(batch.dones.dims()[0], 16);
    }

    #[test]
    fn test_hybrid_buffer_sample_none_when_empty() {
        let device = <NdArray as Backend>::Device::default();
        let buffer = HybridRingBuffer::<TestBackend>::new(100, 8);

        assert!(buffer.sample_batch(10, &device).is_none());
    }

    #[test]
    fn test_hybrid_buffer_sample_none_when_insufficient() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = HybridRingBuffer::<TestBackend>::new(100, 8);

        // Push only 5 items
        for i in 0..5 {
            buffer.push(
                vec![i as f32; 8],
                i,
                i as f32,
                vec![(i + 1) as f32; 8],
                false,
            );
        }

        // Try to sample 10 - should return None
        assert!(buffer.sample_batch(10, &device).is_none());

        // Sample 3 - should work
        let batch = buffer.sample_batch(3, &device);
        assert!(batch.is_some());
    }

    #[test]
    fn test_hybrid_buffer_clear() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(100, 8);

        for _ in 0..50 {
            buffer.push(vec![1.0; 8], 0, 1.0, vec![2.0; 8], false);
        }

        assert_eq!(buffer.len(), 50);
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_hybrid_buffer_o1_push() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(10000, 32);

        // Push 50000 transitions - should be O(n), not O(n²)
        for i in 0..50000 {
            buffer.push(
                vec![i as f32; 32],
                i % 10,
                i as f32,
                vec![(i + 1) as f32; 32],
                false,
            );
        }

        // Should only have capacity items
        assert_eq!(buffer.len(), 10000);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_hybrid_buffer_clone() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(10, 4);

        for i in 0..5 {
            buffer.push(
                vec![i as f32; 4],
                i,
                i as f32,
                vec![(i + 1) as f32; 4],
                false,
            );
        }

        let cloned = buffer.clone();
        assert_eq!(cloned.len(), 5);
        assert_eq!(cloned.capacity(), 10);
        assert_eq!(cloned.state_dim(), 4);
    }

    #[test]
    fn test_hybrid_buffer_sample_batch_values() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = HybridRingBuffer::<TestBackend>::new(10, 4);

        // Push known values
        for i in 0..10 {
            buffer.push(
                vec![i as f32; 4],
                i * 10,
                i as f32 * 0.1,
                vec![(i + 100) as f32; 4],
                i == 5,
            );
        }

        // Sample batch
        let batch = buffer.sample_batch(5, &device).unwrap();

        // Verify tensor shapes (rank-2 format for actions/rewards/dones)
        assert_eq!(batch.states.dims(), [5, 4]);
        assert_eq!(batch.actions.dims(), [5, 1]);
        assert_eq!(batch.rewards.dims(), [5, 1]);
        assert_eq!(batch.next_states.dims(), [5, 4]);
        assert_eq!(batch.dones.dims(), [5, 1]);
    }

    #[test]
    fn test_hybrid_buffer_fill_random() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(100, 4);
        buffer.fill_random(50, 10, 4);
        assert_eq!(buffer.len(), 50);
        assert!(buffer.can_sample(32));
    }

    #[test]
    fn test_hybrid_buffer_fill_random_caps_at_capacity() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(10, 4);
        buffer.fill_random(50, 10, 4);
        assert_eq!(buffer.len(), 10); // Capped at capacity
        assert!(buffer.is_full());
    }
}
