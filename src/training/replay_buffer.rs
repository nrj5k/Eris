use std::collections::VecDeque;

/// Transition for experience replay
#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f32>, // [15] - 5 tier sizes + 10 features
    pub action: usize,   // 0-9
    pub reward: f32,
    pub next_state: Vec<f32>, // [15]
    pub done: bool,
}

/// Experience replay buffer for DQN training
pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add transition to buffer
    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }

    /// Sample random batch
    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        use rand::prelude::*;
        use rand::rng;

        let batch_size = batch_size.min(self.buffer.len());

        let items: Vec<&Transition> = self.buffer.iter().collect();

        let mut rng = rng();
        items
            .sample(&mut rng, batch_size)
            .into_iter()
            .map(|&t| t)
            .collect()
    }

    /// Sample batch with conversion to tensors (for burn)
    pub fn sample_batch(&self, batch_size: usize) -> Option<TransitionBatch> {
        if self.buffer.len() < batch_size {
            return None;
        }

        let transitions = self.sample(batch_size);

        let states: Vec<Vec<f32>> = transitions.iter().map(|t| t.state.clone()).collect();

        let actions: Vec<usize> = transitions.iter().map(|t| t.action).collect();

        let rewards: Vec<f32> = transitions.iter().map(|t| t.reward).collect();

        let next_states: Vec<Vec<f32>> = transitions.iter().map(|t| t.next_state.clone()).collect();

        let dones: Vec<bool> = transitions.iter().map(|t| t.done).collect();

        Some(TransitionBatch {
            states,
            actions,
            rewards,
            next_states,
            dones,
        })
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.capacity
    }
}

/// Batch of transitions for training
#[derive(Debug, Clone)]
pub struct TransitionBatch {
    pub states: Vec<Vec<f32>>,
    pub actions: Vec<usize>,
    pub rewards: Vec<f32>,
    pub next_states: Vec<Vec<f32>>,
    pub dones: Vec<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_push_pop() {
        let mut buffer = ReplayBuffer::new(10);

        for i in 0..15 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i % 10,
                reward: i as f32,
                next_state: vec![i as f32],
                done: false,
            });
        }

        assert_eq!(buffer.len(), 10); // Capacity enforced
    }

    #[test]
    fn test_buffer_sample() {
        let mut buffer = ReplayBuffer::new(20);

        for i in 0..20 {
            buffer.push(Transition {
                state: vec![i as f32],
                action: i % 10,
                reward: i as f32,
                next_state: vec![i as f32],
                done: i == 19,
            });
        }

        let sample = buffer.sample(5);
        assert_eq!(sample.len(), 5);
    }

    #[test]
    fn test_sample_batch() {
        let mut buffer = ReplayBuffer::new(20);

        let obs_dim = 15;
        let num_actions = 10;

        for i in 0..20 {
            buffer.push(Transition {
                state: vec![i as f32; obs_dim],
                action: i % num_actions,
                reward: i as f32,
                next_state: vec![(i + 1) as f32; obs_dim],
                done: i == 19,
            });
        }

        let batch = buffer.sample_batch(5);
        assert!(batch.is_some());

        let batch = batch.unwrap();
        assert_eq!(batch.states.len(), 5);
        assert_eq!(batch.actions.len(), 5);
        assert_eq!(batch.rewards.len(), 5);
        assert_eq!(batch.next_states.len(), 5);
        assert_eq!(batch.dones.len(), 5);
    }
}
