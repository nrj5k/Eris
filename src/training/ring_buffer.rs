//! Ring buffer with O(1) index-based sampling for experience replay

use crate::training::Transition;
use burn::data::dataset::Dataset;
use rand::prelude::*;
use rand::rng;

/// Ring buffer with fixed capacity and index-based sampling
#[derive(Clone)]
pub struct RingBuffer {
    storage: Vec<Transition>,
    head: usize,
    size: usize,
    capacity: usize,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            storage: Vec::with_capacity(capacity),
            head: 0,
            size: 0,
            capacity,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.storage.len() < self.capacity {
            self.storage.push(transition);
        } else {
            self.storage[self.head] = transition;
        }

        self.head = (self.head + 1) % self.capacity;
        self.size = (self.size + 1).min(self.capacity);
    }

    /// Sample random batch of transitions
    /// Returns references WITHOUT cloning for O(batch_size) performance
    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        if self.size < batch_size {
            return Vec::new();
        }

        // Create fresh RNG each call (thread-local, very fast)
        let mut rng = rng();

        // Generate random indices - O(batch_size), not O(n)
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| rng.random_range(0..self.size))
            .collect();

        // Return references without cloning - O(batch_size)
        indices.iter().map(|&i| &self.storage[i]).collect()
    }

    /// Sample batch with conversion to TransitionBatch (for burn)
    /// This clones data, so use sample() when you only need references
    pub fn sample_batch(&self, batch_size: usize) -> Option<super::TransitionBatch> {
        if self.size < batch_size {
            return None;
        }

        let transitions = self.sample(batch_size);

        let states: Vec<Vec<f32>> = transitions.iter().map(|t| t.state.clone()).collect();
        let actions: Vec<usize> = transitions.iter().map(|t| t.action).collect();
        let rewards: Vec<f32> = transitions.iter().map(|t| t.reward).collect();
        let next_states: Vec<Vec<f32>> = transitions.iter().map(|t| t.next_state.clone()).collect();
        let dones: Vec<bool> = transitions.iter().map(|t| t.done).collect();

        Some(super::TransitionBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
        })
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Dataset<Transition> for RingBuffer {
    fn get(&self, index: usize) -> Option<Transition> {
        if index >= self.size {
            return None;
        }
        // Map logical index to physical circular buffer position
        // Logical index 0 = oldest transition (at self.head)
        // Logical index size-1 = newest transition
        let physical_idx = (self.head + index) % self.capacity;
        self.storage.get(physical_idx).cloned()
    }

    fn len(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let mut buffer = RingBuffer::new(100);

        for i in 0..150 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i % 10,
                reward: i as f32,
                next_state: vec![(i + 1) as f32],
                done: false,
            });
        }

        assert_eq!(buffer.len(), 100);

        let batch = buffer.sample(32);
        assert_eq!(batch.len(), 32);
    }

    #[test]
    fn test_ring_buffer_wrap_around() {
        let mut buffer = RingBuffer::new(10);

        // Add 20 items
        for i in 0..20 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }

        // Should only have last 10
        assert_eq!(buffer.len(), 10);
        assert!(buffer.is_full());

        // Sample should work
        let batch = buffer.sample(5);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_ring_buffer_no_sample_when_empty() {
        let buffer = RingBuffer::new(100);

        // Empty buffer should return empty vec
        assert!(buffer.sample(10).is_empty());

        // Add fewer than batch_size
        let mut buffer = RingBuffer::new(100);
        for i in 0..5 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }

        // Should still return empty vec
        assert!(buffer.sample(10).is_empty());
    }

    #[test]
    fn test_ring_buffer_sample_batch() {
        let mut buffer = RingBuffer::new(100);
        let obs_dim = 32;

        for i in 0..50 {
            buffer.push(Transition {
                state: vec![i as f32; obs_dim],
                action: i % 10,
                reward: i as f32,
                next_state: vec![(i + 1) as f32; obs_dim],
                done: i == 49,
            });
        }

        let batch = buffer.sample_batch(10);
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.states.len(), 10);
        assert_eq!(batch.states[0].len(), obs_dim);
        assert_eq!(batch.actions.len(), 10);
        assert_eq!(batch.rewards.len(), 10);
        assert_eq!(batch.next_states.len(), 10);
        assert_eq!(batch.dones.len(), 10);
    }

    #[test]
    fn test_ring_buffer_o1_push() {
        let mut buffer = RingBuffer::new(10000);

        // Push 50000 transitions - should be O(n), not O(n²)
        for i in 0..50000 {
            buffer.push(Transition {
                state: vec![i as f32; 32],
                action: i % 10,
                reward: i as f32,
                next_state: vec![(i + 1) as f32; 32],
                done: false,
            });
        }

        // Should only have capacity items
        assert_eq!(buffer.len(), 10000);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_ring_buffer_sample_references() {
        let mut buffer = RingBuffer::new(100);

        // Fill buffer
        for i in 0..50 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }

        // Sample should return references (no cloning)
        let refs = buffer.sample(10);
        assert_eq!(refs.len(), 10);

        // Verify references are valid
        for transition_ref in &refs {
            assert!(transition_ref.state.len() == 1);
        }
    }

    #[test]
    fn test_dataset_get_returns_oldest_transition() {
        let mut buffer = RingBuffer::new(10);

        // Push 10 items (fill buffer exactly)
        for i in 0..10 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }

        // Dataset::get(0) should return oldest (first pushed)
        let oldest: Transition = buffer.get(0).expect("should have oldest");
        assert_eq!(oldest.state[0], 0.0, "oldest should be item 0");
        assert_eq!(oldest.action, 0);
        assert_eq!(oldest.reward, 0.0);
    }

    #[test]
    fn test_dataset_get_returns_newest_transition() {
        let mut buffer = RingBuffer::new(10);

        // Push 10 items
        for i in 0..10 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }

        // Dataset::get(size-1) should return newest (last pushed)
        let newest: Transition = buffer.get(9).expect("should have newest");
        assert_eq!(newest.state[0], 9.0, "newest should be item 9");
        assert_eq!(newest.action, 9);
        assert_eq!(newest.reward, 9.0);
    }

    #[test]
    fn test_dataset_get_out_of_bounds() {
        let mut buffer = RingBuffer::new(10);

        for i in 0..5 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }

        // Out of bounds should return None
        let result: Option<Transition> = buffer.get(5);
        assert!(result.is_none(), "index 5 should return None");

        let result: Option<Transition> = buffer.get(10);
        assert!(result.is_none(), "index 10 should return None");

        let result: Option<Transition> = buffer.get(100);
        assert!(result.is_none(), "index 100 should return None");
    }

    #[test]
    fn test_dataset_get_with_wrapped_buffer() {
        let mut buffer = RingBuffer::new(10);

        // Push 15 items - buffer wraps
        // After pushing 15 items:
        // - head = 5 (next write position)
        // - storage[5] contains item 5 (oldest still in buffer)
        // - storage[4] contains item 14 (newest)
        for i in 0..15 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }

        assert_eq!(buffer.len(), 10);

        // logical 0 -> physical (5+0)%10 = 5 -> item 5 (oldest)
        let oldest: Transition = buffer.get(0).expect("should have oldest");
        assert_eq!(oldest.state[0], 5.0, "oldest should be item 5");
        assert_eq!(oldest.action, 5);

        // logical 9 -> physical (5+9)%10 = 4 -> item 14 (newest)
        let newest: Transition = buffer.get(9).expect("should have newest");
        assert_eq!(newest.state[0], 14.0, "newest should be item 14");
        assert_eq!(newest.action, 14);

        // Test middle elements
        // logical 5 -> physical (5+5)%10 = 0 -> item 10
        let middle: Transition = buffer.get(5).expect("should have middle");
        assert_eq!(middle.state[0], 10.0, "middle should be item 10");
        assert_eq!(middle.action, 10);
    }

    #[test]
    fn test_dataset_len() {
        let mut buffer = RingBuffer::new(100);

        // Empty buffer
        assert_eq!(Dataset::<Transition>::len(&buffer), 0);

        // Partially filled
        for i in 0..50 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }
        assert_eq!(Dataset::<Transition>::len(&buffer), 50);

        // Fully filled
        for i in 50..100 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }
        assert_eq!(Dataset::<Transition>::len(&buffer), 100);

        // Over capacity - stays at capacity
        for i in 100..150 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }
        assert_eq!(Dataset::<Transition>::len(&buffer), 100);
    }
}
