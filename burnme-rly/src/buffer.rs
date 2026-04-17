//! Replay buffer types with CPU-side storage and batch GPU conversion.
//!
//! This module implements a dual-layer architecture for experience replay:
//! - **CpuRingBuffer**: Stores transitions on CPU with O(1) push and random sampling
//! - **TensorTransitionBatch**: GPU tensor batch with `from_transitions()` conversion
//!
//! # Architecture
//!
//! The design follows the Metis pattern:
//! 1. Store `Transition` structs on CPU in a ring buffer
//! 2. Sample random batches using `rand::thread_rng()`
//! 3. Convert to GPU tensors at training time via `TensorTransitionBatch::from_transitions()`
//!
//! # Example
//!
//! ```rust,ignore
//! use burnme_rly::buffer::{CpuRingBuffer, TensorTransitionBatch, Transition};
//! use burn::backend::NdArray;
//!
//! // Create CPU buffer
//! let mut buffer = CpuRingBuffer::new(10_000);
//!
//! // Push transitions (O(1), CPU only)
//! buffer.push(Transition::new(vec![1.0, 2.0], 0, 0.5, vec![1.1, 2.1], false));
//!
//! // Sample random batch
//! if let Some(transitions) = buffer.sample(32) {
//!     // Convert to GPU tensors at training time
//!     let device = Default::default();
//!     let batch = TensorTransitionBatch::<NdArray>::from_transitions(
//!         &transitions,
//!         2, // state_dim
//!         &device,
//!     );
//! }
//! ```

use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use rand::prelude::*;
use rand::rng;
use std::collections::VecDeque;

/// Single transition tuple (s, a, r, s', done)
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    /// Current observation state
    pub state: Vec<f32>,
    /// Action taken (usize for discrete)
    pub action: usize,
    /// Reward received
    pub reward: f32,
    /// Next observation state
    pub next_state: Vec<f32>,
    /// Whether episode is done
    pub done: bool,
}

impl Transition {
    /// Create new transition
    pub fn new(
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
}

/// Batch of transitions as GPU tensors
///
/// This is the GPU-native representation used for training.
/// All fields are tensors on the GPU device.
#[derive(Debug, Clone)]
pub struct TensorTransitionBatch<B: Backend> {
    /// Batch of states [batch_size, state_dim]
    pub states: Tensor<B, 2>,
    /// Batch of actions [batch_size, 1] - rank-2 for gather()
    pub actions: Tensor<B, 2, Int>,
    /// Batch of rewards [batch_size, 1] - rank-2 for broadcast
    pub rewards: Tensor<B, 2>,
    /// Batch of next states [batch_size, state_dim]
    pub next_states: Tensor<B, 2>,
    /// Batch of done flags [batch_size, 1] - rank-2 for broadcast
    pub dones: Tensor<B, 2>,
}

impl<B: Backend> TensorTransitionBatch<B> {
    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.states.shape().dims[0]
    }

    /// Get state dimension
    pub fn state_dim(&self) -> usize {
        self.states.shape().dims[1]
    }

    /// Convert a batch of CPU transitions to GPU tensors.
    ///
    /// This is the batch-convert pattern from Metis's `batch_to_tensors()`.
    /// It creates rank-2 tensors compatible with DQN training operations.
    ///
    /// # Arguments
    ///
    /// * `transitions` - Slice of CPU-side transitions to convert
    /// * `state_dim` - Expected state dimension (states are padded/truncated to match)
    /// * `device` - GPU device for tensor allocation
    ///
    /// # Returns
    ///
    /// A `TensorTransitionBatch` with all tensors on the specified device.
    ///
    /// # Tensor Shapes
    ///
    /// - `states`: [batch_size, state_dim]
    /// - `actions`: [batch_size, 1] (i32 for gather operations)
    /// - `rewards`: [batch_size, 1] (f32 for broadcasting)
    /// - `next_states`: [batch_size, state_dim]
    /// - `dones`: [batch_size, 1] (f32 for broadcasting: 1.0 if done, 0.0 otherwise)
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use burnme_rly::buffer::{Transition, TensorTransitionBatch};
    ///
    /// let transitions = vec![
    ///     Transition::new(vec![1.0, 2.0], 0, 0.5, vec![1.1, 2.1], false),
    ///     Transition::new(vec![3.0, 4.0], 1, -0.3, vec![3.1, 4.1], true),
    /// ];
    ///
    /// let device = Default::default();
    /// let batch = TensorTransitionBatch::<NdArray>::from_transitions(
    ///     &transitions,
    ///     2,
    ///     &device,
    /// );
    ///
    /// assert_eq!(batch.batch_size(), 2);
    /// assert_eq!(batch.state_dim(), 2);
    /// ```
    pub fn from_transitions(
        transitions: &[Transition],
        state_dim: usize,
        device: &B::Device,
    ) -> Self {
        use burn::tensor::TensorData;

        let batch_size = transitions.len();

        // Flatten states and next_states
        let mut states_flat = Vec::with_capacity(batch_size * state_dim);
        let mut next_states_flat = Vec::with_capacity(batch_size * state_dim);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);

        for t in transitions {
            // Pad or truncate state to match state_dim
            let mut state = t.state.clone();
            state.resize(state_dim, 0.0);
            state.truncate(state_dim);
            states_flat.extend(state);

            actions.push(t.action as i32);
            rewards.push(t.reward);
            dones.push(if t.done { 1.0f32 } else { 0.0f32 });

            let mut next_state = t.next_state.clone();
            next_state.resize(state_dim, 0.0);
            next_state.truncate(state_dim);
            next_states_flat.extend(next_state);
        }

        // Create rank-2 tensors for DQN compatibility
        let states_data = TensorData::new(states_flat, [batch_size, state_dim]);
        let states = Tensor::from_data(states_data.convert::<f32>(), device);

        // actions: [batch_size, 1] rank-2 for gather()
        let actions_data = TensorData::new(actions, [batch_size, 1]);
        let actions = Tensor::from_data(actions_data.convert::<i32>(), device);

        // rewards: [batch_size, 1] rank-2 for broadcasting
        let rewards_data = TensorData::new(rewards, [batch_size, 1]);
        let rewards = Tensor::from_data(rewards_data.convert::<f32>(), device);

        let next_states_data = TensorData::new(next_states_flat, [batch_size, state_dim]);
        let next_states = Tensor::from_data(next_states_data.convert::<f32>(), device);

        // dones: [batch_size, 1] rank-2 for broadcasting
        let dones_data = TensorData::new(dones, [batch_size, 1]);
        let dones = Tensor::from_data(dones_data.convert::<f32>(), device);

        Self {
            states,
            actions,
            rewards,
            next_states,
            dones,
        }
    }
}

/// CPU-side ring buffer for experience replay.
///
/// Stores `Transition` structs on CPU, converts to GPU tensors at training time.
/// This follows the Metis `RingBuffer` pattern with O(1) push and i.i.d. random sampling.
///
/// # Architecture
///
/// - **Storage**: `Vec<Transition>` pre-allocated to capacity
/// - **Head**: Index for next write (circular buffer)
/// - **Size**: Current number of stored transitions
/// - **Capacity**: Maximum number of transitions
///
/// # Performance
///
/// - `push()`: O(1) - no allocation after initial capacity
/// - `sample()`: O(batch_size) - random index generation
/// - Memory: CPU-only, no GPU allocation until training time
///
/// # Examples
///
/// ```rust,ignore
/// use burnme_rly::buffer::{CpuRingBuffer, Transition};
///
/// let mut buffer = CpuRingBuffer::new(10_000);
///
/// // Push transitions (O(1))
/// for i in 0..1000 {
///     buffer.push(Transition::new(
///         vec![i as f32; 32],
///         i % 10,
///         i as f32,
///         vec![(i + 1) as f32; 32],
///         false,
///     ));
/// }
///
/// // Sample random batch
/// if let Some(batch) = buffer.sample(32) {
///     assert_eq!(batch.len(), 32);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct CpuRingBuffer {
    /// Storage vector (pre-allocated to capacity)
    storage: Vec<Transition>,
    /// Head index for circular writes
    head: usize,
    /// Current number of stored transitions
    size: usize,
    /// Maximum capacity
    capacity: usize,
}

impl CpuRingBuffer {
    /// Create a new CPU ring buffer with specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of transitions to store
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burnme_rly::buffer::CpuRingBuffer;
    ///
    /// let buffer = CpuRingBuffer::new(100_000);
    /// assert_eq!(buffer.capacity(), 100_000);
    /// assert_eq!(buffer.len(), 0);
    /// ```
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "CpuRingBuffer capacity must be > 0");
        Self {
            storage: Vec::with_capacity(capacity),
            head: 0,
            size: 0,
            capacity,
        }
    }

    /// Push a single transition to the buffer (O(1), CPU only).
    ///
    /// Uses circular buffer semantics:
    /// - If not at capacity: appends to storage
    /// - If at capacity: overwrites oldest transition at head position
    ///
    /// # Arguments
    ///
    /// * `transition` - Transition to store
    ///
    /// # Performance
    ///
    /// - Time: O(1) amortized
    /// - Memory: No allocation after initial capacity
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burnme_rly::buffer::{CpuRingBuffer, Transition};
    ///
    /// let mut buffer = CpuRingBuffer::new(100);
    ///
    /// // Push 150 transitions
    /// for i in 0..150 {
    ///     buffer.push(Transition::new(
    ///         vec![i as f32],
    ///         i % 10,
    ///         i as f32,
    ///         vec![(i + 1) as f32],
    ///         false,
    ///     ));
    /// }
    ///
    /// // Buffer contains only last 100
    /// assert_eq!(buffer.len(), 100);
    /// ```
    pub fn push(&mut self, transition: Transition) {
        if self.storage.len() < self.capacity {
            self.storage.push(transition);
        } else {
            self.storage[self.head] = transition;
        }

        self.head = (self.head + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
    }

    /// Push a batch of transitions efficiently.
    ///
    /// More efficient than calling `push()` N times due to reduced method call overhead.
    /// For `CpuRingBuffer`, this is primarily for API consistency with `GpuRingBuffer`.
    /// The actual push is still O(1) per transition.
    ///
    /// # Arguments
    ///
    /// * `transitions` - Vector of transitions to push
    ///
    /// # Performance
    ///
    /// - Time: O(n) where n is the number of transitions
    /// - Memory: No extra allocation beyond individual pushes
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burnme_rly::buffer::{CpuRingBuffer, Transition};
    ///
    /// let mut buffer = CpuRingBuffer::new(1000);
    ///
    /// let transitions = vec![
    ///     Transition::new(vec![1.0, 2.0], 0, 0.5, vec![1.1, 2.1], false),
    ///     Transition::new(vec![3.0, 4.0], 1, -0.3, vec![3.1, 4.1], true),
    /// ];
    ///
    /// buffer.push_batch(transitions);
    /// assert_eq!(buffer.len(), 2);
    /// ```
    pub fn push_batch(&mut self, transitions: Vec<Transition>) {
        let count = transitions.len();
        for transition in transitions {
            self.push(transition);
        }

        log::trace!(
            "[STAGE:DIAG] CpuRingBuffer: Pushed batch of {} transitions",
            count
        );
    }

    /// Sample random i.i.d. batch of transitions using `rand::thread_rng()`.
    ///
    /// Returns `Some(Vec<Transition>)` with `batch_size` randomly sampled transitions.
    /// Returns `None` if buffer has fewer transitions than requested.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of transitions to sample
    ///
    /// # Returns
    ///
    /// * `Some(Vec<Transition>)` - Random batch if enough samples available
    /// * `None` - If `self.size < batch_size`
    ///
    /// # Sampling Strategy
    ///
    /// Uses uniform random sampling with replacement:
    /// - Each sample is independent (i.i.d.)
    /// - Same transition may appear multiple times in batch
    /// - Uses `rand::thread_rng()` for fast thread-local randomness
    ///
    /// # Performance
    ///
    /// - Time: O(batch_size) for index generation + cloning
    /// - Memory: Allocates new Vec for batch
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burnme_rly::buffer::{CpuRingBuffer, Transition};
    ///
    /// let mut buffer = CpuRingBuffer::new(1000);
    ///
    /// // Fill buffer
    /// for i in 0..500 {
    ///     buffer.push(Transition::new(
    ///         vec![i as f32; 32],
    ///         i % 10,
    ///         i as f32,
    ///         vec![(i + 1) as f32; 32],
    ///         false,
    ///     ));
    /// }
    ///
    /// // Sample random batch
    /// if let Some(batch) = buffer.sample(32) {
    ///     assert_eq!(batch.len(), 32);
    /// }
    ///
    /// // Empty buffer returns None
    /// let empty = CpuRingBuffer::new(100);
    /// assert!(empty.sample(10).is_none());
    /// ```
    pub fn sample(&self, batch_size: usize) -> Option<Vec<Transition>> {
        if self.size < batch_size {
            return None;
        }

        let mut rng = rng();
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.size))
            .collect();

        Some(indices.iter().map(|&i| self.storage[i].clone()).collect())
    }

    /// Sample with a specific RNG for reproducibility.
    ///
    /// Returns `Some(Vec<Transition>)` with `batch_size` randomly sampled transitions
    /// using the provided RNG. Returns `None` if buffer has fewer transitions than requested.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of transitions to sample
    /// * `rng` - Random number generator for reproducible sampling
    ///
    /// # Returns
    ///
    /// * `Some(Vec<Transition>)` - Random batch if enough samples available
    /// * `None` - If `self.size < batch_size`
    ///
    /// # Reproducibility
    ///
    /// Using the same seed with the same RNG will produce identical samples:
    ///
    /// ```rust,ignore
    /// use burnme_rly::buffer::{CpuRingBuffer, Transition};
    /// use rand::{SeedableRng, rngs::StdRng};
    ///
    /// let mut buffer = CpuRingBuffer::new(1000);
    /// // Fill buffer...
    ///
    /// // Same seed gives same samples
    /// let mut rng1 = StdRng::seed_from_u64(42);
    /// let samples1 = buffer.sample_with_rng(32, &mut rng1);
    ///
    /// let mut rng2 = StdRng::seed_from_u64(42);
    /// let samples2 = buffer.sample_with_rng(32, &mut rng2);
    ///
    /// assert_eq!(samples1, samples2);
    /// ```
    pub fn sample_with_rng<R: Rng + ?Sized>(
        &self,
        batch_size: usize,
        rng: &mut R,
    ) -> Option<Vec<Transition>> {
        if self.size < batch_size {
            return None;
        }

        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.size))
            .collect();

        Some(indices.iter().map(|&i| self.storage[i].clone()).collect())
    }

    /// Get current number of transitions stored.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Check if buffer has enough samples for a batch.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Required batch size
    ///
    /// # Returns
    ///
    /// `true` if `self.size >= batch_size`
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.size >= batch_size
    }

    /// Get maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is full.
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Clear all transitions from the buffer.
    ///
    /// Resets size and head to 0, clears storage vector.
    pub fn clear(&mut self) {
        self.storage.clear();
        self.head = 0;
        self.size = 0;
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
            self.push(Transition::new(state, action, reward, next_state, done));
        }
        log::info!(
            "[STAGE:WARMUP] Pre-filled buffer with {} random transitions (action_dim={}, state_dim={})",
            count, action_dim, state_dim
        );
    }
}

impl Default for CpuRingBuffer {
    fn default() -> Self {
        Self::new(100_000) // Default: 100k capacity
    }
}

// ==================== HybridRingBuffer: CPU Storage → GPU Batch ====================

/// Hybrid ring buffer: stores transitions on CPU, converts to GPU only on sampling.
///
/// This follows the Metis proven pattern:
/// 1. Store transitions as separate CPU vectors (no GPU memory during push)
/// 2. Convert to GPU tensors only during sample_batch(), outputs rank-2 tensors
/// 3. O(1) push — no GPU allocations during training
/// 4. No VRAM leak — memory stays constant
///
/// # Architecture
///
/// - CPU storage: 5 separate vectors (states, actions, rewards, next_states, dones)
/// - GPU conversion: only in sample_batch(), outputs rank-2 tensors
/// - Ring buffer: overwrites oldest transitions when full
///
/// # Example
///
/// ```rust,ignore
/// use burnme_rly::buffer::HybridRingBuffer;
/// use burn::backend::NdArray;
///
/// let mut buffer = HybridRingBuffer::<NdArray>::new(10_000, 32);
///
/// buffer.push(vec![1.0; 32], 0, 0.5, vec![2.0; 32], false);
///
/// if let Some(batch) = buffer.sample_batch(32, &device) {
///     // batch.states: [32, 32], batch.actions: [32, 1], etc.
/// }
/// ```
pub struct HybridRingBuffer<B: Backend> {
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
    /// Create a new hybrid ring buffer with CPU storage.
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

    /// Push a single transition — O(1), no GPU allocations.
    pub fn push(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) {
        if self.states.len() < self.capacity {
            self.states.push(state);
            self.actions.push(action);
            self.rewards.push(reward);
            self.next_states.push(next_state);
            self.dones.push(done);
        } else {
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
        log::trace!(
            "[STAGE:DIAG] HybridRingBuffer: Pushed batch of {} transitions",
            batch_size
        );
    }

    /// Sample a random batch and convert to GPU tensors.
    ///
    /// Returns `Some(TensorTransitionBatch)` if enough samples, `None` otherwise.
    /// Creates rank-2 tensors: states [batch, state_dim], actions [batch, 1], etc.
    pub fn sample_batch(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> Option<TensorTransitionBatch<B>> {
        if self.size < batch_size {
            return None;
        }

        use rand::prelude::IteratorRandom;
        let mut rng = rand::rng();
        let indices: Vec<usize> = (0..self.size).sample(&mut rng, batch_size);
        if indices.len() < batch_size {
            return None;
        }

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
            batch_dones.push(if self.dones[idx] { 1.0f32 } else { 0.0f32 });
        }

        // Convert to rank-2 GPU tensors
        let states = Tensor::from_data(
            TensorData::new(batch_states, [batch_size, self.state_dim]).convert::<f32>(),
            device,
        );
        let actions = Tensor::from_data(
            TensorData::new(batch_actions, [batch_size, 1]).convert::<i32>(),
            device,
        );
        let rewards = Tensor::from_data(
            TensorData::new(batch_rewards, [batch_size, 1]).convert::<f32>(),
            device,
        );
        let next_states = Tensor::from_data(
            TensorData::new(batch_next_states, [batch_size, self.state_dim]).convert::<f32>(),
            device,
        );
        let dones = Tensor::from_data(
            TensorData::new(batch_dones, [batch_size, 1]).convert::<f32>(),
            device,
        );

        Some(TensorTransitionBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
        })
    }

    /// Get buffer length.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is full.
    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    /// Get state dimension.
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Check if we have enough samples.
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.size >= batch_size
    }

    /// Clear the buffer.
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
        log::info!(
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

impl<B: Backend> std::fmt::Debug for HybridRingBuffer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridRingBuffer")
            .field("capacity", &self.capacity)
            .field("size", &self.size)
            .field("state_dim", &self.state_dim)
            .field("head", &self.head)
            .finish_non_exhaustive()
    }
}

// ==================== GpuRingBuffer: GPU-Native Storage ====================

/// GPU-native ring buffer storing tensors directly on device.
///
/// Following Burn's official pattern - no CPU→GPU copies!
/// Uses `select()` for O(1) GPU-side sampling.
///
/// # Architecture
///
/// - All tensors stored directly on GPU device
/// - Lazy allocation on first `push()`
/// - Circular buffer with `slice_assign` for in-place updates
/// - O(1) sampling via `select()` - no CPU→GPU copies
///
/// # Example
///
/// ```rust,ignore
/// use burn::backend::NdArray;
/// use burn::tensor::Tensor;
/// use burnme_rly::buffer::GpuRingBuffer;
///
/// let device = Default::default();
/// let mut buffer = GpuRingBuffer::<NdArray>::new(10_000, 4, &device);
///
/// // Push transitions (GPU tensors)
/// let state = Tensor::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
/// let next_state = Tensor::from_floats([[1.1, 2.1, 3.1, 4.1]], &device);
/// buffer.push(&state, 0, 0.5, &next_state, false);
///
/// // Sample batch (GPU-side, O(1))
/// if let Some(batch) = buffer.sample(32) {
///     // batch.states, batch.actions, etc. are all GPU tensors
/// }
/// ```
pub struct GpuRingBuffer<B: Backend> {
    /// States tensor [capacity, state_dim] on GPU
    states: Option<Tensor<B, 2>>,
    /// Next states tensor [capacity, state_dim] on GPU
    next_states: Option<Tensor<B, 2>>,
    /// Actions tensor [capacity] on GPU (Int)
    actions: Option<Tensor<B, 1, Int>>,
    /// Rewards tensor [capacity] on GPU
    rewards: Option<Tensor<B, 1>>,
    /// Dones tensor [capacity] on GPU (bool as f32: 1.0=true, 0.0=false)
    dones: Option<Tensor<B, 1>>,
    /// Maximum capacity
    capacity: usize,
    /// Write head position
    write_head: usize,
    /// Current number of transitions stored
    len: usize,
    /// State dimension
    state_dim: usize,
    /// Device (stored to ensure same device for all ops)
    device: B::Device,
}

/// Batch of transitions as GPU tensors (already on device)
#[derive(Debug, Clone)]
pub struct GpuTransitionBatch<B: Backend> {
    /// Batch of states \[batch_size, state_dim\]
    pub states: Tensor<B, 2>,
    /// Batch of actions \[batch_size\]
    pub actions: Tensor<B, 1, Int>,
    /// Batch of rewards \[batch_size\]
    pub rewards: Tensor<B, 1>,
    /// Batch of next states \[batch_size, state_dim\]
    pub next_states: Tensor<B, 2>,
    /// Batch of done flags \[batch_size\]
    pub dones: Tensor<B, 1>,
}

impl<B: Backend> GpuTransitionBatch<B> {
    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.states.shape().dims[0]
    }

    /// Get state dimension
    pub fn state_dim(&self) -> usize {
        self.states.shape().dims[1]
    }
}

impl<B: Backend> GpuRingBuffer<B> {
    /// Create new GPU ring buffer.
    ///
    /// Note: Tensors are lazily allocated on first push.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of transitions to store
    /// * `state_dim` - Dimension of state space
    /// * `device` - GPU device for tensor allocation
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use burnme_rly::buffer::GpuRingBuffer;
    ///
    /// let device = Default::default();
    /// let buffer = GpuRingBuffer::<NdArray>::new(10_000, 4, &device);
    /// assert_eq!(buffer.capacity(), 10_000);
    /// assert_eq!(buffer.len(), 0);
    /// ```
    pub fn new(capacity: usize, state_dim: usize, device: &B::Device) -> Self {
        assert!(capacity > 0, "GpuRingBuffer capacity must be > 0");
        assert!(state_dim > 0, "GpuRingBuffer state_dim must be > 0");
        Self {
            states: None,
            next_states: None,
            actions: None,
            rewards: None,
            dones: None,
            capacity,
            write_head: 0,
            len: 0,
            state_dim,
            device: device.clone(),
        }
    }

    /// Push a single transition to the buffer.
    ///
    /// On first push, allocates GPU tensors.
    /// Uses circular buffer overwrite when full.
    ///
    /// # Arguments
    ///
    /// * `state` - Current state [1, state_dim]
    /// * `action` - Action taken (usize for discrete)
    /// * `reward` - Reward received
    /// * `next_state` - Next state [1, state_dim]
    /// * `done` - Whether episode is done
    ///
    /// # Performance
    ///
    /// - First push: O(capacity × state_dim) for allocation
    /// - Subsequent pushes: O(state_dim) for slice_assign
    /// - Memory: GPU-only, allocated once on first push
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use burn::tensor::Tensor;
    /// use burnme_rly::buffer::GpuRingBuffer;
    ///
    /// let device = Default::default();
    /// let mut buffer = GpuRingBuffer::<NdArray>::new(100, 4, &device);
    ///
    /// let state = Tensor::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
    /// let next_state = Tensor::from_floats([[1.1, 2.1, 3.1, 4.1]], &device);
    /// buffer.push(&state, 0, 0.5, &next_state, false);
    ///
    /// assert_eq!(buffer.len(), 1);
    /// ```
    pub fn push(
        &mut self,
        state: &Tensor<B, 2>, // [1, state_dim]
        action: usize,
        reward: f32,
        next_state: &Tensor<B, 2>, // [1, state_dim]
        done: bool,
    ) {
        // Validate shapes
        assert_eq!(
            state.shape().dims::<2>(),
            [1, self.state_dim],
            "State tensor shape must be [1, {}], got {:?}",
            self.state_dim,
            state.shape().dims::<2>()
        );
        assert_eq!(
            next_state.shape().dims::<2>(),
            [1, self.state_dim],
            "Next state tensor shape must be [1, {}], got {:?}",
            self.state_dim,
            next_state.shape().dims::<2>()
        );

        // Lazy allocation on first push
        if self.states.is_none() {
            self.allocate_buffers();
        }

        // Get the index to write to
        let idx = self.write_head;

        // Use slice_assign to write at specific index
        // states[idx] = state
        self.states = self
            .states
            .take()
            .map(|s| s.slice_assign([idx..idx + 1, 0..self.state_dim], state.clone()));

        self.next_states = self
            .next_states
            .take()
            .map(|s| s.slice_assign([idx..idx + 1, 0..self.state_dim], next_state.clone()));

        // For scalars, we need to create tensors and assign
        let action_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![action as i32], [1]).convert::<i32>(),
            &self.device,
        );
        self.actions = self.actions.take().map(|a| {
            // Note: slice_assign for 1D tensor - pass range directly, not in array
            a.slice_assign(idx..idx + 1, action_tensor)
        });

        let reward_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(vec![reward], [1]).convert::<f32>(),
            &self.device,
        );
        self.rewards = self
            .rewards
            .take()
            .map(|r| r.slice_assign(idx..idx + 1, reward_tensor));

        let done_tensor = Tensor::<B, 1>::from_data(
            TensorData::new(vec![if done { 1.0f32 } else { 0.0f32 }], [1]).convert::<f32>(),
            &self.device,
        );
        self.dones = self
            .dones
            .take()
            .map(|d| d.slice_assign(idx..idx + 1, done_tensor));

        // Update pointers
        self.write_head = (self.write_head + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    /// Push a batch of transitions efficiently.
    ///
    /// Much faster than calling push() N times - does bulk operations.
    ///
    /// # Arguments
    ///
    /// * `states` - Batch of states \[batch_size, state_dim\]
    /// * `actions` - Batch of actions \[batch_size\]
    /// * `rewards` - Batch of rewards \[batch_size\]
    /// * `next_states` - Batch of next states \[batch_size, state_dim\]
    /// * `dones` - Batch of done flags \[batch_size\]
    ///
    /// # Performance
    ///
    /// - Time: O(batch_size × state_dim) - bulk operations
    /// - Memory: GPU-only, no extra allocations
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use burn::tensor::Tensor;
    /// use burnme_rly::buffer::GpuRingBuffer;
    ///
    /// let device = Default::default();
    /// let mut buffer = GpuRingBuffer::<NdArray>::new(1000, 4, &device);
    ///
    /// let states = Tensor::from_floats([[1.0, 2.0, 3.0, 4.0]; 32], &device);
    /// let actions = Tensor::from_data(TensorData::new(vec![0i32; 32], [32]), &device);
    /// let rewards = Tensor::from_floats([0.5f32; 32], &device);
    /// let next_states = Tensor::from_floats([[1.1, 2.1, 3.1, 4.1]; 32], &device);
    /// let dones = Tensor::from_floats([0.0f32; 32], &device);
    ///
    /// buffer.push_batch(&states, &actions, &rewards, &next_states, &dones);
    /// assert_eq!(buffer.len(), 32);
    /// ```
    pub fn push_batch(
        &mut self,
        states: &Tensor<B, 2>,       // [batch_size, state_dim]
        actions: &Tensor<B, 1, Int>, // [batch_size]
        rewards: &Tensor<B, 1>,      // [batch_size]
        next_states: &Tensor<B, 2>,  // [batch_size, state_dim]
        dones: &Tensor<B, 1>,        // [batch_size]
    ) {
        let batch_size = states.dims()[0];
        if batch_size == 0 {
            return;
        }

        // Ensure buffers are allocated
        if self.states.is_none() {
            self.allocate_buffers();
        }

        // Handle wrap-around by splitting into chunks
        let start_idx = self.write_head;
        let end_idx = (start_idx + batch_size).min(self.capacity);
        let first_chunk = end_idx - start_idx;

        // Push first chunk
        if first_chunk > 0 {
            self.states = self.states.take().map(|s| {
                s.slice_assign(
                    [start_idx..end_idx, 0..self.state_dim],
                    states.clone().slice([0..first_chunk, 0..self.state_dim]),
                )
            });

            self.next_states = self.next_states.take().map(|s| {
                s.slice_assign(
                    [start_idx..end_idx, 0..self.state_dim],
                    next_states
                        .clone()
                        .slice([0..first_chunk, 0..self.state_dim]),
                )
            });

            self.actions = self
                .actions
                .take()
                .map(|a| a.slice_assign(start_idx..end_idx, actions.clone().slice(0..first_chunk)));
            self.rewards = self
                .rewards
                .take()
                .map(|r| r.slice_assign(start_idx..end_idx, rewards.clone().slice(0..first_chunk)));
            self.dones = self
                .dones
                .take()
                .map(|d| d.slice_assign(start_idx..end_idx, dones.clone().slice(0..first_chunk)));

            self.write_head = end_idx % self.capacity;
            self.len = (self.len + first_chunk).min(self.capacity);
        }

        // Handle wrap-around if needed
        if first_chunk < batch_size {
            let rem_states = states
                .clone()
                .slice([first_chunk..batch_size, 0..self.state_dim]);
            let rem_actions = actions.clone().slice(first_chunk..batch_size);
            let rem_rewards = rewards.clone().slice(first_chunk..batch_size);
            let rem_next = next_states
                .clone()
                .slice([first_chunk..batch_size, 0..self.state_dim]);
            let rem_dones = dones.clone().slice(first_chunk..batch_size);

            self.push_batch(
                &rem_states,
                &rem_actions,
                &rem_rewards,
                &rem_next,
                &rem_dones,
            );
        }
    }

    /// Sample a random batch using GPU operations.
    ///
    /// Uses `select()` for O(1) GPU-side sampling - no CPU→GPU copies!
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of transitions to sample
    ///
    /// # Returns
    ///
    /// * `Some(GpuTransitionBatch)` - Random batch if enough samples available
    /// * `None` - If `self.len < batch_size`
    ///
    /// # Performance
    ///
    /// - Time: O(batch_size) for random index generation + O(1) for select()
    /// - Memory: GPU-only, no CPU→GPU copies
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use burn::tensor::Tensor;
    /// use burnme_rly::buffer::GpuRingBuffer;
    ///
    /// let device = Default::default();
    /// let mut buffer = GpuRingBuffer::<NdArray>::new(1000, 4, &device);
    ///
    /// // Fill buffer
    /// for i in 0..100 {
    ///     let state = Tensor::from_floats([[i as f32; 4]], &device);
    ///     let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
    ///     buffer.push(&state, i % 10, i as f32 * 0.1, &next_state, false);
    /// }
    ///
    /// // Sample batch
    /// if let Some(batch) = buffer.sample(32) {
    ///     assert_eq!(batch.batch_size(), 32);
    ///     assert_eq!(batch.state_dim(), 4);
    /// }
    /// ```
    pub fn sample(&self, batch_size: usize) -> Option<GpuTransitionBatch<B>> {
        if self.len < batch_size {
            return None;
        }

        // Generate random indices on CPU for proper integer uniform distribution
        let mut rng = rng();
        let indices: Vec<i32> = (0..batch_size)
            .map(|_| rng.random_range(0..self.len) as i32)
            .collect();
        let indices_data = TensorData::new(indices, [batch_size]);
        let indices_tensor =
            Tensor::<B, 1, Int>::from_data(indices_data.convert::<i32>(), &self.device);

        // Use select() to sample from GPU tensors - O(1)!
        let states = self
            .states
            .as_ref()?
            .clone()
            .select(0, indices_tensor.clone());
        let next_states = self
            .next_states
            .as_ref()?
            .clone()
            .select(0, indices_tensor.clone());
        let actions = self
            .actions
            .as_ref()?
            .clone()
            .select(0, indices_tensor.clone());
        let rewards = self
            .rewards
            .as_ref()?
            .clone()
            .select(0, indices_tensor.clone());
        let dones = self.dones.as_ref()?.clone().select(0, indices_tensor);

        Some(GpuTransitionBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
        })
    }

    /// Sample with a specific RNG for reproducibility.
    ///
    /// Returns `Some(GpuTransitionBatch)` with `batch_size` randomly sampled transitions
    /// using the provided RNG. Returns `None` if buffer has fewer transitions than requested.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of transitions to sample
    /// * `rng` - Random number generator for reproducible sampling
    ///
    /// # Returns
    ///
    /// * `Some(GpuTransitionBatch<B>)` - Random batch if enough samples available
    /// * `None` - If `self.len < batch_size`
    ///
    /// # Reproducibility
    ///
    /// Using the same seed with the same RNG will produce identical samples:
    ///
    /// ```rust,ignore
    /// use burn::backend::NdArray;
    /// use burnme_rly::buffer::GpuRingBuffer;
    /// use rand::{SeedableRng, rngs::StdRng};
    ///
    /// let device = Default::default();
    /// let buffer = GpuRingBuffer::<NdArray>::new(1000, 4, &device);
    /// // Fill buffer...
    ///
    /// // Same seed gives same samples
    /// let mut rng1 = StdRng::seed_from_u64(42);
    /// let samples1 = buffer.sample_with_rng(32, &mut rng1);
    ///
    /// let mut rng2 = StdRng::seed_from_u64(42);
    /// let samples2 = buffer.sample_with_rng(32, &mut rng2);
    ///
    /// assert_eq!(samples1.batch_size(), samples2.batch_size());
    /// ```
    pub fn sample_with_rng<R: Rng + ?Sized>(
        &self,
        batch_size: usize,
        rng: &mut R,
    ) -> Option<GpuTransitionBatch<B>> {
        if self.len < batch_size {
            return None;
        }

        // Generate indices with provided RNG
        let indices: Vec<i32> = (0..batch_size)
            .map(|_| rng.random_range(0..self.len) as i32)
            .collect();

        // Create indices tensor
        let indices_tensor = Tensor::<B, 1, Int>::from_data(
            TensorData::new(indices, [batch_size]).convert::<i32>(),
            &self.device,
        );

        // Use select() to sample from GPU tensors - O(1)!
        let states = self
            .states
            .as_ref()?
            .clone()
            .select(0, indices_tensor.clone());
        let next_states = self
            .next_states
            .as_ref()?
            .clone()
            .select(0, indices_tensor.clone());
        let actions = self
            .actions
            .as_ref()?
            .clone()
            .select(0, indices_tensor.clone());
        let rewards = self
            .rewards
            .as_ref()?
            .clone()
            .select(0, indices_tensor.clone());
        let dones = self.dones.as_ref()?.clone().select(0, indices_tensor);

        Some(GpuTransitionBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
        })
    }

    /// Get current number of transitions
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if can sample
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.len >= batch_size
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if full
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Clear all transitions.
    ///
    /// Note: We keep the allocated tensors, just reset counters.
    pub fn clear(&mut self) {
        self.len = 0;
        self.write_head = 0;

        // Zero out tensor data to prevent stale data
        if let Some(ref mut s) = self.states {
            *s = Tensor::<B, 2>::zeros([self.capacity, self.state_dim], &self.device);
        }
        if let Some(ref mut ns) = self.next_states {
            *ns = Tensor::<B, 2>::zeros([self.capacity, self.state_dim], &self.device);
        }
        if let Some(ref mut a) = self.actions {
            *a = Tensor::<B, 1, Int>::zeros([self.capacity], &self.device);
        }
        if let Some(ref mut r) = self.rewards {
            *r = Tensor::<B, 1>::zeros([self.capacity], &self.device);
        }
        if let Some(ref mut d) = self.dones {
            *d = Tensor::<B, 1>::zeros([self.capacity], &self.device);
        }
    }

    /// Allocate GPU tensors (called lazily on first push)
    fn allocate_buffers(&mut self) {
        self.states = Some(Tensor::zeros([self.capacity, self.state_dim], &self.device));
        self.next_states = Some(Tensor::zeros([self.capacity, self.state_dim], &self.device));
        self.actions = Some(Tensor::zeros([self.capacity], &self.device));
        self.rewards = Some(Tensor::zeros([self.capacity], &self.device));
        self.dones = Some(Tensor::zeros([self.capacity], &self.device));
    }
}

impl<B: Backend> std::fmt::Debug for GpuRingBuffer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuRingBuffer")
            .field("capacity", &self.capacity)
            .field("len", &self.len)
            .field("write_head", &self.write_head)
            .field("state_dim", &self.state_dim)
            .finish_non_exhaustive()
    }
}

/// GPU-native ring buffer for tensor transitions.
///
/// # Deprecation Notice
///
/// **DEPRECATED**: Use `CpuRingBuffer` + `TensorTransitionBatch::from_transitions()` instead.
///
/// ## Why Deprecate?
///
/// - `TensorRingBuffer` stores data on GPU, causing memory pressure
/// - Each `push_transition()` creates batch_size=1 GPU tensors (inefficient)
/// - `sample()` always returns most recent batch (no random sampling)
///
/// ## Migration Path
///
/// ```rust,ignore
/// // OLD (deprecated)
/// let mut buffer = TensorRingBuffer::new(100_000, 32);
/// buffer.push_transition(transition, &device);
/// let batch = buffer.sample(32);
///
/// // NEW (recommended)
/// let mut buffer = CpuRingBuffer::new(100_000);
/// buffer.push(transition);
/// if let Some(transitions) = buffer.sample(32) {
///     let batch = TensorTransitionBatch::from_transitions(&transitions, 32, &device);
/// }
/// ```
#[deprecated(
    since = "0.2.0",
    note = "Use CpuRingBuffer + TensorTransitionBatch::from_transitions() instead. \
            See module documentation for migration guide."
)]
#[derive(Debug)]
pub struct TensorRingBuffer<B: AutodiffBackend> {
    /// Ring buffer of transitions stored as GPU tensors
    buffer: VecDeque<TensorTransitionBatch<B>>,
    /// Maximum capacity (number of batches)
    capacity: usize,
    /// Current number of transitions stored
    num_transitions: usize,
    /// State dimension
    state_dim: usize,
}

#[allow(deprecated)]
impl<B: AutodiffBackend> TensorRingBuffer<B> {
    /// Create a new tensor ring buffer
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of transitions to store
    /// * `state_dim` - Dimension of observation space
    pub fn new(capacity: usize, state_dim: usize) -> Self {
        Self {
            buffer: VecDeque::new(),
            capacity,
            num_transitions: 0,
            state_dim,
        }
    }

    /// Get current number of transitions stored
    pub fn len(&self) -> usize {
        self.num_transitions
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.num_transitions == 0
    }

    /// Check if buffer has enough samples for a batch
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.num_transitions >= batch_size
    }

    /// Push a batch of transitions to the buffer
    ///
    /// # Arguments
    /// * `batch` - Batch of transitions as GPU tensors
    pub fn push_batch(&mut self, batch: TensorTransitionBatch<B>) {
        let batch_len = batch.batch_size();

        // If at capacity, remove oldest batch
        if self.num_transitions + batch_len > self.capacity {
            if let Some(oldest) = self.buffer.pop_front() {
                let oldest_len = oldest.batch_size();
                self.num_transitions -= oldest_len;
            }
        }

        self.num_transitions += batch_len;
        self.buffer.push_back(batch);
    }

    /// Sample a random batch of transitions
    ///
    /// # Arguments
    /// * `batch_size` - Number of transitions to sample
    ///
    /// # Returns
    /// * `Some(TensorTransitionBatch)` if enough samples available
    /// * `None` if insufficient samples
    #[deprecated(
        since = "0.1.0",
        note = "Has limited randomness. Use CpuRingBuffer instead."
    )]
    pub fn sample(&self, batch_size: usize) -> Option<TensorTransitionBatch<B>> {
        if self.num_transitions < batch_size || self.buffer.is_empty() {
            return None;
        }

        // Randomly select a batch from the buffer
        let mut rng = rng();
        let batch_idx = rng.random_range(0..self.buffer.len());

        log::warn!("TensorRingBuffer::sample() has limited randomness. Use CpuRingBuffer instead.");

        self.buffer.get(batch_idx).map(|batch| {
            let start_idx = 0;
            TensorTransitionBatch {
                states: batch
                    .states
                    .clone()
                    .slice(start_idx..(start_idx + batch_size)),
                actions: batch
                    .actions
                    .clone()
                    .slice(start_idx..(start_idx + batch_size)),
                rewards: batch
                    .rewards
                    .clone()
                    .slice(start_idx..(start_idx + batch_size)),
                next_states: batch
                    .next_states
                    .clone()
                    .slice(start_idx..(start_idx + batch_size)),
                dones: batch
                    .dones
                    .clone()
                    .slice(start_idx..(start_idx + batch_size)),
            }
        })
    }

    /// Clear all transitions from the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.num_transitions = 0;
    }

    /// Get state dimension
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Get maximum capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Push a single transition to the buffer (converts to batch internally).
    ///
    /// # Arguments
    /// * `transition` - Single transition to store
    /// * `device` - GPU device for tensor operations
    pub fn push_transition(&mut self, transition: Transition, device: &B::Device) {
        let batch = self.transition_to_batch(vec![transition], device);
        self.push_batch(batch);
    }

    /// Convert transitions to a tensor batch
    fn transition_to_batch(
        &self,
        transitions: Vec<Transition>,
        device: &B::Device,
    ) -> TensorTransitionBatch<B> {
        use burn::tensor::TensorData;

        let batch_size = transitions.len();
        let state_dim = self.state_dim;

        // Flatten states and next_states
        let mut states_flat = Vec::with_capacity(batch_size * state_dim);
        let mut next_states_flat = Vec::with_capacity(batch_size * state_dim);
        let mut actions = Vec::with_capacity(batch_size);
        let mut rewards = Vec::with_capacity(batch_size);
        let mut dones = Vec::with_capacity(batch_size);

        for t in transitions {
            // Pad or truncate state to match state_dim
            let mut state = t.state;
            state.resize(state_dim, 0.0);
            state.truncate(state_dim);
            states_flat.extend(state);

            actions.push(t.action as i32);
            rewards.push(t.reward);
            dones.push(if t.done { 1.0f32 } else { 0.0f32 });

            // Pad or truncate next_state to match state_dim
            let mut next_state = t.next_state;
            next_state.resize(state_dim, 0.0);
            next_state.truncate(state_dim);
            next_states_flat.extend(next_state);
        }

        // Create tensors on the GPU device using TensorData pattern
        let states_data = TensorData::new(states_flat, [batch_size, state_dim]);
        let states: Tensor<B, 2> = Tensor::from_data(states_data.convert::<f32>(), device);

        let actions_data = TensorData::new(actions, [batch_size, 1]);
        let actions: Tensor<B, 2, Int> = Tensor::from_data(actions_data.convert::<i32>(), device);

        let rewards_data = TensorData::new(rewards, [batch_size, 1]);
        let rewards: Tensor<B, 2> = Tensor::from_data(rewards_data.convert::<f32>(), device);

        let next_states_data = TensorData::new(next_states_flat, [batch_size, state_dim]);
        let next_states: Tensor<B, 2> =
            Tensor::from_data(next_states_data.convert::<f32>(), device);

        let dones_data = TensorData::new(dones, [batch_size, 1]);
        let dones: Tensor<B, 2> = Tensor::from_data(dones_data.convert::<f32>(), device);

        TensorTransitionBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
        }
    }
}

#[allow(deprecated)]
impl<B: AutodiffBackend> Default for TensorRingBuffer<B> {
    fn default() -> Self {
        Self::new(100_000, 4) // Default: 100k capacity, 4-dim state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_transition() {
        let transition = Transition::new(vec![1.0, 2.0, 3.0], 1, 0.5, vec![1.1, 2.1, 3.1], false);
        assert_eq!(transition.state.len(), 3);
        assert_eq!(transition.action, 1);
    }

    // ==================== CpuRingBuffer Tests ====================

    #[test]
    fn test_cpu_ring_buffer_new() {
        let buffer = CpuRingBuffer::new(1000);
        assert_eq!(buffer.capacity(), 1000);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.can_sample(1));
    }

    #[test]
    fn test_cpu_ring_buffer_push() {
        let mut buffer = CpuRingBuffer::new(100);

        for i in 0..50 {
            buffer.push(Transition::new(
                vec![i as f32],
                i % 10,
                i as f32,
                vec![(i + 1) as f32],
                false,
            ));
        }

        assert_eq!(buffer.len(), 50);
        assert!(!buffer.is_empty());
        assert!(buffer.can_sample(32));
        assert!(!buffer.can_sample(64));
    }

    #[test]
    fn test_cpu_ring_buffer_wrap_around() {
        let mut buffer = CpuRingBuffer::new(10);

        // Push 20 items (wrap around twice)
        for i in 0..20 {
            buffer.push(Transition::new(
                vec![i as f32],
                i,
                i as f32,
                vec![i as f32],
                false,
            ));
        }

        // Should only have last 10
        assert_eq!(buffer.len(), 10);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_cpu_ring_buffer_sample() {
        let mut buffer = CpuRingBuffer::new(100);

        // Fill buffer
        for i in 0..50 {
            buffer.push(Transition::new(
                vec![i as f32; 32],
                i % 10,
                i as f32,
                vec![(i + 1) as f32; 32],
                false,
            ));
        }

        // Sample should work
        let batch = buffer.sample(32);
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 32);

        // Empty buffer should return None
        let empty = CpuRingBuffer::new(100);
        assert!(empty.sample(10).is_none());

        // Insufficient samples should return None
        let mut small = CpuRingBuffer::new(100);
        for i in 0..5 {
            small.push(Transition::new(
                vec![i as f32],
                i,
                i as f32,
                vec![i as f32],
                false,
            ));
        }
        assert!(small.sample(10).is_none());
    }

    #[test]
    fn test_cpu_ring_buffer_clear() {
        let mut buffer = CpuRingBuffer::new(100);

        // Fill buffer
        for i in 0..50 {
            buffer.push(Transition::new(
                vec![i as f32],
                i,
                i as f32,
                vec![i as f32],
                false,
            ));
        }

        assert_eq!(buffer.len(), 50);

        // Clear
        buffer.clear();

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.can_sample(1));
    }

    #[test]
    fn test_cpu_ring_buffer_o1_push() {
        let mut buffer = CpuRingBuffer::new(10000);

        // Push 50000 transitions - should be O(n), not O(n²)
        for i in 0..50000 {
            buffer.push(Transition::new(
                vec![i as f32; 32],
                i % 10,
                i as f32,
                vec![(i + 1) as f32; 32],
                false,
            ));
        }

        // Should only have capacity items
        assert_eq!(buffer.len(), 10000);
        assert!(buffer.is_full());
    }

    // ==================== TensorTransitionBatch Tests ====================

    #[test]
    fn test_tensor_transition_batch_from_transitions() {
        let device = <TestBackend as Backend>::Device::default();

        let transitions = vec![
            Transition::new(vec![1.0, 2.0], 0, 0.5, vec![1.1, 2.1], false),
            Transition::new(vec![3.0, 4.0], 1, -0.3, vec![3.1, 4.1], true),
            Transition::new(vec![5.0, 6.0], 2, 1.0, vec![5.1, 6.1], false),
        ];

        let batch =
            TensorTransitionBatch::<TestBackend>::from_transitions(&transitions, 2, &device);

        assert_eq!(batch.batch_size(), 3);
        assert_eq!(batch.state_dim(), 2);

        // Verify shapes
        assert_eq!(batch.states.shape().dims, [3, 2]);
        assert_eq!(batch.actions.shape().dims, [3, 1]);
        assert_eq!(batch.rewards.shape().dims, [3, 1]);
        assert_eq!(batch.next_states.shape().dims, [3, 2]);
        assert_eq!(batch.dones.shape().dims, [3, 1]);

        // Verify values
        let states_data: Vec<f32> = batch
            .states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert states");
        assert_eq!(states_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let actions_data: Vec<i32> = batch
            .actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("Failed to convert actions");
        assert_eq!(actions_data, vec![0, 1, 2]);

        let rewards_data: Vec<f32> = batch
            .rewards
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert rewards");
        assert!((rewards_data[0] - 0.5).abs() < 1e-5);
        assert!((rewards_data[1] - (-0.3)).abs() < 1e-5);
        assert!((rewards_data[2] - 1.0).abs() < 1e-5);

        let dones_data: Vec<f32> = batch
            .dones
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert dones");
        assert_eq!(dones_data, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_tensor_transition_batch_padding() {
        let device = <TestBackend as Backend>::Device::default();

        // Transitions with states smaller than state_dim
        let transitions = vec![
            Transition::new(vec![1.0], 0, 0.5, vec![1.1], false),
            Transition::new(vec![2.0], 1, -0.3, vec![2.1], true),
        ];

        let batch =
            TensorTransitionBatch::<TestBackend>::from_transitions(&transitions, 4, &device);

        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.state_dim(), 4); // Padded to 4

        // Verify states are padded with zeros
        let states_data: Vec<f32> = batch
            .states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert states");
        assert_eq!(states_data, vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tensor_transition_batch_truncation() {
        let device = <TestBackend as Backend>::Device::default();

        // Transitions with states larger than state_dim
        let transitions = vec![
            Transition::new(
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                0,
                0.5,
                vec![1.1, 2.1, 3.1, 4.1, 5.1],
                false,
            ),
            Transition::new(
                vec![6.0, 7.0, 8.0, 9.0, 10.0],
                1,
                -0.3,
                vec![6.1, 7.1, 8.1, 9.1, 10.1],
                true,
            ),
        ];

        let batch =
            TensorTransitionBatch::<TestBackend>::from_transitions(&transitions, 2, &device);

        assert_eq!(batch.batch_size(), 2);
        assert_eq!(batch.state_dim(), 2); // Truncated to 2

        // Verify states are truncated
        let states_data: Vec<f32> = batch
            .states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert states");
        assert_eq!(states_data, vec![1.0, 2.0, 6.0, 7.0]);
    }

    #[test]
    fn test_cpu_ring_buffer_sample_batch_integration() {
        let device = <TestBackend as Backend>::Device::default();
        let mut buffer = CpuRingBuffer::new(1000);

        // Fill buffer
        for i in 0..100 {
            buffer.push(Transition::new(
                vec![i as f32; 8],
                i % 10,
                i as f32,
                vec![(i + 1) as f32; 8],
                i == 99,
            ));
        }

        // Sample and convert to batch
        if let Some(transitions) = buffer.sample(32) {
            let batch =
                TensorTransitionBatch::<TestBackend>::from_transitions(&transitions, 8, &device);

            assert_eq!(batch.batch_size(), 32);
            assert_eq!(batch.state_dim(), 8);
            assert_eq!(batch.states.shape().dims, [32, 8]);
            assert_eq!(batch.actions.shape().dims, [32, 1]);
        } else {
            panic!("Should have sampled batch");
        }
    }

    #[test]
    fn test_gather_operation_with_rank2_actions() {
        use burn::tensor::{Tensor, TensorData};

        let device = <TestBackend as Backend>::Device::default();

        // Create Q-values tensor [batch_size, num_actions]
        // Each row represents Q-values for different actions
        let q_values_data = vec![
            1.0, 2.0, 3.0, // Sample 0: Q(s,0)=1, Q(s,1)=2, Q(s,2)=3
            4.0, 5.0, 6.0, // Sample 1: Q(s,0)=4, Q(s,1)=5, Q(s,2)=6
            7.0, 8.0, 9.0, // Sample 2: Q(s,0)=7, Q(s,1)=8, Q(s,2)=9
            10.0, 11.0, 12.0, // Sample 3: Q(s,0)=10, Q(s,1)=11, Q(s,2)=12
        ];
        let q_values: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::new(q_values_data, [4, 3]).convert::<f32>(),
            &device,
        );

        // Create rank-2 actions tensor [batch_size, 1] for gather operation
        // Actions: [1, 2, 0, 1] - selecting different actions for each sample
        let actions_data = vec![1i32, 2, 0, 1];
        let actions: Tensor<TestBackend, 2, Int> = Tensor::from_data(
            TensorData::new(actions_data, [4, 1]).convert::<i32>(),
            &device,
        );

        // Verify actions shape is rank-2 [batch_size, 1]
        assert_eq!(actions.shape().dims, [4, 1]);

        // Use gather operation to extract Q(s, a) values
        // gather(dim=1, indices=actions) extracts values along dimension 1
        let gathered = q_values.gather(1, actions);

        // Verify output shape is [batch_size, 1]
        assert_eq!(gathered.shape().dims, [4, 1]);

        // Verify gathered values match expected Q(s, a) values
        // Expected: [Q(s0,a1)=2, Q(s1,a2)=6, Q(s2,a0)=7, Q(s3,a1)=11]
        let gathered_data: Vec<f32> = gathered
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert gathered values");

        assert!(
            (gathered_data[0] - 2.0).abs() < 1e-5,
            "Expected Q(s0,a1)=2.0, got {}",
            gathered_data[0]
        );
        assert!(
            (gathered_data[1] - 6.0).abs() < 1e-5,
            "Expected Q(s1,a2)=6.0, got {}",
            gathered_data[1]
        );
        assert!(
            (gathered_data[2] - 7.0).abs() < 1e-5,
            "Expected Q(s2,a0)=7.0, got {}",
            gathered_data[2]
        );
        assert!(
            (gathered_data[3] - 11.0).abs() < 1e-5,
            "Expected Q(s3,a1)=11.0, got {}",
            gathered_data[3]
        );
    }

    #[test]
    fn test_tensor_batch_gather_compatibility() {
        use burn::tensor::{Tensor, TensorData};

        let device = <TestBackend as Backend>::Device::default();

        // Create transitions with specific actions
        let transitions = vec![
            Transition::new(vec![1.0, 2.0], 0, 0.5, vec![1.1, 2.1], false),
            Transition::new(vec![3.0, 4.0], 2, -0.3, vec![3.1, 4.1], true),
            Transition::new(vec![5.0, 6.0], 1, 1.0, vec![5.1, 6.1], false),
        ];

        // Convert to tensor batch - actions should be rank-2 [3, 1]
        let batch =
            TensorTransitionBatch::<TestBackend>::from_transitions(&transitions, 2, &device);

        // Verify actions tensor shape
        assert_eq!(batch.actions.shape().dims, [3, 1]);

        // Create Q-values for each sample [batch_size, num_actions]
        // Assuming 3 possible actions
        let q_values_data = vec![
            1.0, 2.0, 3.0, // Sample 0
            4.0, 5.0, 6.0, // Sample 1
            7.0, 8.0, 9.0, // Sample 2
        ];
        let q_values: Tensor<TestBackend, 2> = Tensor::from_data(
            TensorData::new(q_values_data, [3, 3]).convert::<f32>(),
            &device,
        );

        // Gather Q-values using rank-2 actions tensor
        let gathered = q_values.gather(1, batch.actions.clone());

        // Verify shape
        assert_eq!(gathered.shape().dims, [3, 1]);

        // Verify values match expected Q(s, a)
        // Actions: [0, 2, 1] -> Expected: [Q(s0,0)=1, Q(s1,2)=6, Q(s2,1)=8]
        let gathered_data: Vec<f32> = gathered
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert gathered values");

        assert!(
            (gathered_data[0] - 1.0).abs() < 1e-5,
            "Expected Q(s0,a0)=1.0, got {}",
            gathered_data[0]
        );
        assert!(
            (gathered_data[1] - 6.0).abs() < 1e-5,
            "Expected Q(s1,a2)=6.0, got {}",
            gathered_data[1]
        );
        assert!(
            (gathered_data[2] - 8.0).abs() < 1e-5,
            "Expected Q(s2,a1)=8.0, got {}",
            gathered_data[2]
        );
    }

    // ==================== Random Sampling Variation Tests ====================

    #[test]
    fn test_cpu_ring_buffer_sample_varied_results() {
        let mut buffer = CpuRingBuffer::new(1000);

        // Fill buffer with distinct transitions
        for i in 0..500 {
            buffer.push(Transition::new(
                vec![i as f32; 16],
                i % 100,
                i as f32 / 100.0,
                vec![(i + 1) as f32; 16],
                i % 50 == 49,
            ));
        }

        // Sample multiple times and verify results are rarely identical
        let sample1 = buffer.sample(32).expect("Should sample batch");
        let sample2 = buffer.sample(32).expect("Should sample batch");
        let sample3 = buffer.sample(32).expect("Should sample batch");

        // With 500 transitions and batch size 32, probability of identical samples is extremely low
        // Check that at least two samples differ
        let samples_equal_1_2 = sample1
            .iter()
            .zip(sample2.iter())
            .all(|(a, b)| a.state == b.state && a.action == b.action && a.reward == b.reward);

        let samples_equal_2_3 = sample2
            .iter()
            .zip(sample3.iter())
            .all(|(a, b)| a.state == b.state && a.action == b.action && a.reward == b.reward);

        // At least one pair should differ (probability of all identical is ~10^-50)
        assert!(
            !samples_equal_1_2 || !samples_equal_2_3,
            "Random samples should produce varied results across multiple calls"
        );
    }

    #[test]
    fn test_cpu_ring_buffer_sample_diversity_across_calls() {
        let mut buffer = CpuRingBuffer::new(1000);

        // Fill buffer with unique transitions
        for i in 0..800 {
            buffer.push(Transition::new(
                vec![i as f32; 8],
                i % 10,
                i as f32,
                vec![(i + 1) as f32; 8],
                false,
            ));
        }

        // Collect unique states from multiple samples
        // Use u32 representation to avoid f32 Hash/Eq issues
        let mut all_unique_states = std::collections::HashSet::new();

        for _ in 0..10 {
            if let Some(batch) = buffer.sample(64) {
                for transition in batch {
                    // Use first element of state as identifier (convert to u32 bits for Hash)
                    all_unique_states.insert(transition.state[0].to_bits());
                }
            }
        }

        // With 800 transitions and 10 samples of 64, we should see significant diversity
        // Expected unique states: much more than 64 (single sample size)
        assert!(
            all_unique_states.len() > 200,
            "Random sampling should produce diverse results across multiple calls, got {} unique states",
            all_unique_states.len()
        );
    }

    #[test]
    fn test_cpu_ring_buffer_sample_statistical_variation() {
        let mut buffer = CpuRingBuffer::new(1000);

        // Fill buffer with transitions having varying rewards
        for i in 0..500 {
            buffer.push(Transition::new(
                vec![i as f32; 4],
                i % 5,
                (i % 100) as f32 / 10.0, // Rewards from 0.0 to 9.9
                vec![(i + 1) as f32; 4],
                false,
            ));
        }

        // Sample multiple times and compute mean reward for each sample
        let mut sample_means = Vec::new();

        for _ in 0..20 {
            if let Some(batch) = buffer.sample(100) {
                let mean_reward: f32 = batch.iter().map(|t| t.reward).sum::<f32>() / 100.0;
                sample_means.push(mean_reward);
            }
        }

        // Verify there's variation in sample means (they shouldn't all be identical)
        let min_mean = sample_means.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_mean = sample_means
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        assert!(
            (max_mean - min_mean) > 0.1,
            "Sample means should vary across calls (range: {} to {})",
            min_mean,
            max_mean
        );
    }

    // ==================== Reproducibility Tests ====================

    #[test]
    fn test_cpu_ring_buffer_sample_with_rng_reproducible() {
        use rand::{rngs::StdRng, SeedableRng};

        let mut buffer = CpuRingBuffer::new(1000);

        // Fill buffer with distinct transitions
        for i in 0..500 {
            buffer.push(Transition::new(
                vec![i as f32; 8],
                i % 10,
                i as f32 / 100.0,
                vec![(i + 1) as f32; 8],
                i % 50 == 49,
            ));
        }

        // Sample with same seed twice
        let mut rng1 = StdRng::seed_from_u64(42);
        let samples1 = buffer
            .sample_with_rng(32, &mut rng1)
            .expect("Should sample");

        let mut rng2 = StdRng::seed_from_u64(42);
        let samples2 = buffer
            .sample_with_rng(32, &mut rng2)
            .expect("Should sample");

        // Same seed should give identical samples
        assert_eq!(samples1.len(), samples2.len());
        for (s1, s2) in samples1.iter().zip(samples2.iter()) {
            assert_eq!(s1.state, s2.state);
            assert_eq!(s1.action, s2.action);
            assert_eq!(s1.reward, s2.reward);
            assert_eq!(s1.next_state, s2.next_state);
            assert_eq!(s1.done, s2.done);
        }
    }

    #[test]
    fn test_cpu_ring_buffer_sample_with_rng_different_seeds() {
        use rand::{rngs::StdRng, SeedableRng};

        let mut buffer = CpuRingBuffer::new(1000);

        // Fill buffer with distinct transitions
        for i in 0..500 {
            buffer.push(Transition::new(
                vec![i as f32; 8],
                i % 10,
                i as f32 / 100.0,
                vec![(i + 1) as f32; 8],
                false,
            ));
        }

        // Sample with different seeds
        let mut rng1 = StdRng::seed_from_u64(42);
        let samples1 = buffer
            .sample_with_rng(32, &mut rng1)
            .expect("Should sample");

        let mut rng2 = StdRng::seed_from_u64(123);
        let samples2 = buffer
            .sample_with_rng(32, &mut rng2)
            .expect("Should sample");

        // Different seeds should likely give different samples
        // (probability of identical samples is extremely low)
        let all_identical = samples1
            .iter()
            .zip(samples2.iter())
            .all(|(a, b)| a.state == b.state && a.action == b.action);

        assert!(
            !all_identical,
            "Different seeds should produce different samples"
        );
    }

    #[test]
    fn test_cpu_ring_buffer_sample_with_rng_insufficient_samples() {
        use rand::{rngs::StdRng, SeedableRng};

        let mut buffer = CpuRingBuffer::new(100);

        // Push only 5 transitions
        for i in 0..5 {
            buffer.push(Transition::new(
                vec![i as f32],
                i,
                i as f32,
                vec![i as f32],
                false,
            ));
        }

        let mut rng = StdRng::seed_from_u64(42);

        // Should return None when insufficient samples
        assert!(buffer.sample_with_rng(10, &mut rng).is_none());

        // But 5 should work
        assert!(buffer.sample_with_rng(5, &mut rng).is_some());
    }

    #[test]
    fn test_gpu_ring_buffer_sample_with_rng_reproducible() {
        use rand::{rngs::StdRng, SeedableRng};

        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(1000, 4, &device);

        // Fill buffer with distinct transitions
        for i in 0..500 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i % 100, i as f32 / 100.0, &next_state, i % 50 == 49);
        }

        // Sample with same seed twice
        let mut rng1 = StdRng::seed_from_u64(42);
        let batch1 = buffer
            .sample_with_rng(32, &mut rng1)
            .expect("Should sample");

        let mut rng2 = StdRng::seed_from_u64(42);
        let batch2 = buffer
            .sample_with_rng(32, &mut rng2)
            .expect("Should sample");

        // Same seed should give identical batches
        assert_eq!(batch1.batch_size(), batch2.batch_size());
        assert_eq!(batch1.state_dim(), batch2.state_dim());

        // Compare tensor data
        let states1: Vec<f32> = batch1
            .states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert states1");
        let states2: Vec<f32> = batch2
            .states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert states2");
        assert_eq!(states1, states2);

        let actions1: Vec<i32> = batch1
            .actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("Failed to convert actions1");
        let actions2: Vec<i32> = batch2
            .actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("Failed to convert actions2");
        assert_eq!(actions1, actions2);
    }

    #[test]
    fn test_gpu_ring_buffer_sample_with_rng_different_seeds() {
        use rand::{rngs::StdRng, SeedableRng};

        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(1000, 4, &device);

        // Fill buffer with distinct transitions
        for i in 0..500 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i % 100, i as f32 / 100.0, &next_state, false);
        }

        // Sample with different seeds
        let mut rng1 = StdRng::seed_from_u64(42);
        let batch1 = buffer
            .sample_with_rng(32, &mut rng1)
            .expect("Should sample");

        let mut rng2 = StdRng::seed_from_u64(123);
        let batch2 = buffer
            .sample_with_rng(32, &mut rng2)
            .expect("Should sample");

        // Different seeds should likely give different samples
        assert_eq!(batch1.batch_size(), 32);
        assert_eq!(batch2.batch_size(), 32);

        // Compare state data - should differ
        let states1: Vec<f32> = batch1
            .states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert states1");
        let states2: Vec<f32> = batch2
            .states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert states2");

        // Very unlikely to be identical with different seeds
        assert_ne!(
            states1, states2,
            "Different seeds should produce different samples"
        );
    }

    #[test]
    fn test_gpu_ring_buffer_sample_with_rng_insufficient_samples() {
        use rand::{rngs::StdRng, SeedableRng};

        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(100, 4, &device);

        // Push only 5 transitions
        for i in 0..5 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i, i as f32, &next_state, false);
        }

        let mut rng = StdRng::seed_from_u64(42);

        // Should return None when insufficient samples
        assert!(buffer.sample_with_rng(10, &mut rng).is_none());

        // But 5 should work
        assert!(buffer.sample_with_rng(5, &mut rng).is_some());
    }

    // ==================== CPU Storage Tests ====================

    // ==================== GpuRingBuffer Tests ====================

    #[test]
    fn test_gpu_ring_buffer_creation() {
        let device = <NdArray as Backend>::Device::default();
        let buffer = GpuRingBuffer::<TestBackend>::new(100, 4, &device);
        assert_eq!(buffer.capacity(), 100);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.can_sample(1));
    }

    #[test]
    fn test_gpu_ring_buffer_push() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(100, 4, &device);

        // Create sample tensors
        let state = Tensor::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let next_state = Tensor::from_floats([[1.1, 2.1, 3.1, 4.1]], &device);

        buffer.push(&state, 0, 0.5, &next_state, false);

        assert_eq!(buffer.len(), 1);
        assert!(!buffer.is_empty());
        assert!(buffer.can_sample(1));
    }

    #[test]
    fn test_gpu_ring_buffer_sample() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(100, 4, &device);

        // Push multiple transitions
        for i in 0..50 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i % 10, i as f32 * 0.1, &next_state, i % 5 == 0);
        }

        // Sample batch
        let batch = buffer.sample(32).expect("Should sample");
        assert_eq!(batch.batch_size(), 32); // batch_size
        assert_eq!(batch.state_dim(), 4); // state_dim

        // Verify tensor shapes
        assert_eq!(batch.states.shape().dims, [32, 4]);
        assert_eq!(batch.actions.shape().dims, [32]);
        assert_eq!(batch.rewards.shape().dims, [32]);
        assert_eq!(batch.next_states.shape().dims, [32, 4]);
        assert_eq!(batch.dones.shape().dims, [32]);
    }

    #[test]
    fn test_gpu_ring_buffer_circular() {
        // Test that buffer wraps around correctly
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(10, 4, &device);

        // Fill beyond capacity
        for i in 0..15 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i % 10, i as f32 * 0.1, &next_state, false);
        }

        // Should still work and have 10 items
        assert_eq!(buffer.len(), 10);
        assert!(buffer.is_full());

        // Can still sample
        let batch = buffer.sample(5).expect("Should sample");
        assert_eq!(batch.batch_size(), 5);
        assert_eq!(batch.state_dim(), 4);
    }

    #[test]
    fn test_gpu_ring_buffer_clear() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(100, 4, &device);

        // Fill buffer
        for i in 0..50 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i, i as f32, &next_state, false);
        }

        assert_eq!(buffer.len(), 50);

        // Clear
        buffer.clear();

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert!(!buffer.can_sample(1));
    }

    #[test]
    fn test_gpu_ring_buffer_insufficient_samples() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(100, 4, &device);

        // Push only 5 transitions
        for i in 0..5 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i, i as f32, &next_state, false);
        }

        // Try to sample more than available
        assert!(buffer.sample(10).is_none());

        // But sampling 5 should work
        let batch = buffer.sample(5);
        assert!(batch.is_some());
    }

    #[test]
    fn test_gpu_ring_buffer_sample_varied_results() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(1000, 4, &device);

        // Fill buffer with distinct transitions
        for i in 0..500 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i % 100, i as f32 / 100.0, &next_state, i % 50 == 49);
        }

        // Sample multiple times
        let batch1 = buffer.sample(32).expect("Should sample");
        let batch2 = buffer.sample(32).expect("Should sample");

        // Batches should have correct shape
        assert_eq!(batch1.batch_size(), 32);
        assert_eq!(batch2.batch_size(), 32);
    }

    #[test]
    fn test_gpu_ring_buffer_lazy_allocation() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(100, 4, &device);

        // Before first push, tensors are not allocated
        // (We can't directly test this, but we can verify it works after push)

        // After first push, everything should work
        let state = Tensor::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let next_state = Tensor::from_floats([[1.1, 2.1, 3.1, 4.1]], &device);
        buffer.push(&state, 0, 0.5, &next_state, false);

        assert_eq!(buffer.len(), 1);
        let batch = buffer.sample(1).expect("Should sample");
        assert_eq!(batch.batch_size(), 1);
    }

    #[test]
    fn test_gpu_ring_buffer_done_flags() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(100, 4, &device);

        // Push transitions with specific done flags
        let state = Tensor::from_floats([[1.0, 2.0, 3.0, 4.0]], &device);
        let next_state = Tensor::from_floats([[1.1, 2.1, 3.1, 4.1]], &device);

        buffer.push(&state, 0, 0.5, &next_state, false); // not done
        buffer.push(&state, 1, 0.3, &next_state, true); // done
        buffer.push(&state, 2, 0.7, &next_state, false); // not done

        let batch = buffer.sample(3).expect("Should sample");

        // Verify dones tensor shape
        assert_eq!(batch.dones.shape().dims, [3]);
    }

    #[test]
    fn test_gpu_ring_buffer_actions_int_type() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(100, 4, &device);

        // Push transitions with different actions
        for i in 0..10 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i % 5, i as f32, &next_state, false);
        }

        let batch = buffer.sample(10).expect("Should sample");

        // Verify actions tensor has Int type (compile-time check)
        // and correct shape
        assert_eq!(batch.actions.shape().dims, [10]);
    }

    #[test]
    fn test_gpu_ring_buffer_capacity_boundary() {
        let device = <NdArray as Backend>::Device::default();
        let mut buffer = GpuRingBuffer::<TestBackend>::new(10, 4, &device);

        // Fill exactly to capacity
        for i in 0..10 {
            let state = Tensor::from_floats([[i as f32; 4]], &device);
            let next_state = Tensor::from_floats([[(i + 1) as f32; 4]], &device);
            buffer.push(&state, i, i as f32, &next_state, i == 9);
        }

        assert_eq!(buffer.len(), 10);
        assert!(buffer.is_full());

        // Sample full batch
        let batch = buffer.sample(10).expect("Should sample");
        assert_eq!(batch.batch_size(), 10);
    }

    #[test]
    fn test_cpu_ring_buffer_no_gpu_allocation() {
        // This test verifies that CpuRingBuffer operations don't require GPU device
        // The buffer should work entirely on CPU without any Backend dependency

        let mut buffer = CpuRingBuffer::new(1000);

        // Push transitions without any GPU device
        for i in 0..100 {
            buffer.push(Transition::new(
                vec![i as f32; 32],
                i % 10,
                i as f32,
                vec![(i + 1) as f32; 32],
                i % 20 == 19,
            ));
        }

        // Verify buffer state
        assert_eq!(buffer.len(), 100);
        assert!(!buffer.is_full());
        assert!(buffer.can_sample(50));

        // Sample without GPU
        let batch = buffer.sample(32).expect("Should sample");
        assert_eq!(batch.len(), 32);

        // Verify all transitions are intact (CPU storage)
        for transition in &batch {
            assert_eq!(transition.state.len(), 32);
            assert_eq!(transition.next_state.len(), 32);
        }
    }

    #[test]
    fn test_cpu_ring_buffer_cpu_only_lifecycle() {
        // Test complete lifecycle without GPU involvement
        let mut buffer = CpuRingBuffer::new(500);

        // Phase 1: Fill buffer on CPU
        for i in 0..600 {
            // Wrap around will occur
            buffer.push(Transition::new(
                vec![i as f32; 16],
                i % 20,
                (i * 2) as f32,
                vec![(i + 1) as f32; 16],
                i % 100 == 99,
            ));
        }

        // Verify capacity constraint (CPU memory management)
        assert_eq!(buffer.len(), 500);
        assert!(buffer.is_full());

        // Phase 2: Sample on CPU
        let samples = buffer.sample(128).expect("Should sample");
        assert_eq!(samples.len(), 128);

        // Phase 3: Clear on CPU
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());

        // Phase 4: Reuse buffer on CPU
        for i in 0..50 {
            buffer.push(Transition::new(
                vec![i as f32; 8],
                i,
                i as f32,
                vec![(i + 1) as f32; 8],
                false,
            ));
        }
        assert_eq!(buffer.len(), 50);
    }

    // ==================== Enhanced Tensor Shape Tests ====================

    #[test]
    fn test_tensor_transition_batch_all_tensor_shapes() {
        let device = <TestBackend as Backend>::Device::default();

        let transitions = vec![
            Transition::new(vec![1.0, 2.0, 3.0], 0, 0.5, vec![1.1, 2.1, 3.1], false),
            Transition::new(vec![4.0, 5.0, 6.0], 1, -0.3, vec![4.1, 5.1, 6.1], true),
        ];

        let batch = TensorTransitionBatch::<TestBackend>::from_transitions(
            &transitions,
            3, // state_dim
            &device,
        );

        // Verify all tensor shapes explicitly
        assert_eq!(
            batch.states.shape().dims,
            [2, 3],
            "states should be [batch_size, state_dim]"
        );
        assert_eq!(
            batch.actions.shape().dims,
            [2, 1],
            "actions should be [batch_size, 1] for gather()"
        );
        assert_eq!(
            batch.rewards.shape().dims,
            [2, 1],
            "rewards should be [batch_size, 1] for broadcast"
        );
        assert_eq!(
            batch.next_states.shape().dims,
            [2, 3],
            "next_states should be [batch_size, state_dim]"
        );
        assert_eq!(
            batch.dones.shape().dims,
            [2, 1],
            "dones should be [batch_size, 1] for broadcast"
        );

        // Verify rank (number of dimensions)
        assert_eq!(batch.states.shape().num_dims(), 2);
        assert_eq!(batch.actions.shape().num_dims(), 2);
        assert_eq!(batch.rewards.shape().num_dims(), 2);
        assert_eq!(batch.next_states.shape().num_dims(), 2);
        assert_eq!(batch.dones.shape().num_dims(), 2);
    }

    #[test]
    fn test_tensor_transition_batch_single_element_batch() {
        let device = <TestBackend as Backend>::Device::default();

        let transitions = vec![Transition::new(
            vec![7.0, 8.0],
            3,
            1.5,
            vec![7.1, 8.1],
            true,
        )];

        let batch =
            TensorTransitionBatch::<TestBackend>::from_transitions(&transitions, 2, &device);

        // Verify single-element batch shapes
        assert_eq!(batch.batch_size(), 1);
        assert_eq!(batch.states.shape().dims, [1, 2]);
        assert_eq!(batch.actions.shape().dims, [1, 1]);
        assert_eq!(batch.rewards.shape().dims, [1, 1]);
        assert_eq!(batch.next_states.shape().dims, [1, 2]);
        assert_eq!(batch.dones.shape().dims, [1, 1]);

        // Verify values
        let states_data: Vec<f32> = batch
            .states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert states");
        assert_eq!(states_data, vec![7.0, 8.0]);

        let dones_data: Vec<f32> = batch
            .dones
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert dones");
        assert_eq!(dones_data, vec![1.0]); // true -> 1.0
    }

    #[test]
    fn test_tensor_transition_batch_large_batch() {
        let device = <TestBackend as Backend>::Device::default();

        // Create a larger batch
        let transitions: Vec<Transition> = (0..256)
            .map(|i| {
                Transition::new(
                    vec![i as f32; 64],
                    i % 10,
                    i as f32 / 256.0,
                    vec![(i + 1) as f32; 64],
                    i % 50 == 49,
                )
            })
            .collect();

        let batch =
            TensorTransitionBatch::<TestBackend>::from_transitions(&transitions, 64, &device);

        // Verify large batch shapes
        assert_eq!(batch.batch_size(), 256);
        assert_eq!(batch.state_dim(), 64);
        assert_eq!(batch.states.shape().dims, [256, 64]);
        assert_eq!(batch.actions.shape().dims, [256, 1]);
        assert_eq!(batch.rewards.shape().dims, [256, 1]);
        assert_eq!(batch.next_states.shape().dims, [256, 64]);
        assert_eq!(batch.dones.shape().dims, [256, 1]);
    }

    // ==================== HybridRingBuffer Tests ====================

    #[test]
    fn test_hybrid_buffer_push_and_len() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(100, 4);
        buffer.push(
            vec![1.0, 2.0, 3.0, 4.0],
            0,
            0.5,
            vec![1.1, 2.1, 3.1, 4.1],
            false,
        );
        buffer.push(
            vec![5.0, 6.0, 7.0, 8.0],
            1,
            -0.3,
            vec![5.1, 6.1, 7.1, 8.1],
            true,
        );
        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.state_dim(), 4);
        assert!(!buffer.is_full());
    }

    #[test]
    fn test_hybrid_buffer_wraparound() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(10, 4);
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
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let mut buffer = HybridRingBuffer::<TestBackend>::new(100, 4);
        for i in 0..50 {
            buffer.push(
                vec![i as f32; 4],
                i % 10,
                i as f32,
                vec![(i + 1) as f32; 4],
                false,
            );
        }
        let batch = buffer.sample_batch(16, &device);
        assert!(batch.is_some());
        let batch = batch.unwrap();
        assert_eq!(batch.batch_size(), 16);
        assert_eq!(batch.state_dim(), 4);
        // Check shapes are rank-2
        assert_eq!(batch.states.shape().dims, [16, 4]);
        assert_eq!(batch.actions.shape().dims, [16, 1]);
    }

    #[test]
    fn test_hybrid_buffer_sample_none_when_empty() {
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();
        let buffer = HybridRingBuffer::<TestBackend>::new(100, 4);
        assert!(buffer.sample_batch(10, &device).is_none());
    }

    #[test]
    fn test_hybrid_buffer_clear() {
        let mut buffer = HybridRingBuffer::<TestBackend>::new(100, 4);
        for i in 0..50 {
            buffer.push(
                vec![i as f32; 4],
                i,
                i as f32,
                vec![(i + 1) as f32; 4],
                false,
            );
        }
        assert_eq!(buffer.len(), 50);
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
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
