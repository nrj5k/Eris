//! Tensor conversion utilities for policy state/action handling.
//!
//! This module provides shared utility functions for converting between
//! policy domain types (State, Action) and Burn tensors. These utilities
//! are used by all reinforcement learning policies (Bandit, DQN, Metis, Catcher).
//!
//! # Overview
//!
//! The conversion utilities handle:
//! - **State to Tensor**: Convert `State` enum variants to `Tensor<B, 2>`
//! - **Batch States**: Convert batches of state vectors to tensors
//! - **Transition Batching**: Parse transition batches into tensor components
//!
//! # State Variants
//!
//! The `State` enum has three variants:
//! - `Features(Vec<f32>)`: Normalized feature vector (used by DQN, Metis, Bandit)
//! - `Raw(Vec<f64>)`: Raw address/observation data (used by Catcher)
//! - `Empty`: No state information (fallback)
//!
//! # Example
//!
//! ```rust,ignore
//! use eris::policies::tensor_utils::{state_to_tensor, states_to_tensor, batch_to_tensors};
//! use eris::policies::policy::{State, Transition, Action};
//! use burn::backend::NdArray;
//! use burn::tensor::backend::Backend;
//!
//! let device = <NdArray as Backend>::Device::default();
//!
//! // Single state conversion
//! let state = State::Features(vec![1.0, 0.5, 0.8]);
//! let tensor = state_to_tensor(&state, 3, &device); // [1, 3]
//!
//! // Batch conversion
//! let states = vec![
//!     vec![1.0, 0.5, 0.8],
//!     vec![0.2, 0.7, 0.3],
//! ];
//! let batch_tensor = states_to_tensor(&states, &device); // [2, 3]
//! ```

use super::policy::{Action, State, Transition};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};

/// Convert a single `State` enum to a tensor.
///
/// Handles all three state variants:
/// - `State::Features`: Direct conversion to f32 tensor
/// - `State::Raw`: Convert f64 to f32, then to tensor
/// - `State::Empty`: Return zero tensor of specified dimension
///
/// # Arguments
///
/// * `state` - State enum to convert
/// * `state_dim` - Expected state dimension (used for Empty variant)
/// * `device` - Target device for tensor placement
///
/// # Returns
///
/// Tensor with shape `[1, state_dim]` containing the state features.
///
/// # Examples
///
/// ```rust,ignore
/// use eris::policies::policy::State;
/// use burn::backend::NdArray;
/// use burn::tensor::backend::Backend;
///
/// let device = <NdArray as Backend>::Device::default();
///
/// // Features variant
/// let state = State::Features(vec![1.0, 0.5, 0.8]);
/// let tensor = state_to_tensor(&state, 3, &device);
/// assert_eq!(tensor.shape().dims, [1, 3]);
///
/// // Empty variant returns zeros
/// let empty = State::Empty;
/// let empty_tensor = state_to_tensor(&empty, 10, &device);
/// assert_eq!(empty_tensor.shape().dims, [1, 10]);
/// ```
pub fn state_to_tensor<B: Backend>(
    state: &State,
    state_dim: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    match state {
        State::Features(features) => {
            let data = TensorData::new(features.clone(), [1, features.len()]);
            Tensor::from_data(data, device)
        }
        State::Raw(raw) => {
            let features: Vec<f32> = raw.iter().map(|&x| x as f32).collect();
            let data = TensorData::new(features.clone(), [1, features.len()]);
            Tensor::from_data(data, device)
        }
        State::Empty => Tensor::zeros([1, state_dim], device),
    }
}

/// Convert a batch of state vectors to a tensor.
///
/// Takes a slice of feature vectors and converts them into a single batched
/// tensor suitable for neural network forward passes.
///
/// # Arguments
///
/// * `states` - Slice of state feature vectors
/// * `device` - Target device for tensor placement
///
/// # Returns
///
/// Tensor with shape `[batch_size, state_dim]` where `batch_size = states.len()`.
/// Returns empty tensor `[0, state_dim]` if input is empty.
///
/// # Panics
///
/// Panics if states have inconsistent dimensions.
///
/// # Examples
///
/// ```rust,ignore
/// use burn::backend::NdArray;
/// use burn::tensor::backend::Backend;
///
/// let device = <NdArray as Backend>::Device::default();
///
/// // Batch of states
/// let states = vec![
///     vec![1.0, 2.0, 3.0],
///     vec![4.0, 5.0, 0.6],
///     vec![7.0, 8.0, 9.0],
/// ];
/// let tensor = states_to_tensor(&states, &device);
/// assert_eq!(tensor.shape().dims, [3, 3]);
///
/// // Empty batch
/// let empty: Vec<Vec<f32>> = vec![];
/// let empty_tensor = states_to_tensor(&empty, &device);
/// assert_eq!(empty_tensor.shape().dims[0], 0);
/// ```
pub fn states_to_tensor<B: Backend>(states: &[Vec<f32>], device: &B::Device) -> Tensor<B, 2> {
    if states.is_empty() {
        return Tensor::zeros([0, 0], device);
    }

    let batch_size = states.len();
    let state_dim = states[0].len();

    // Flatten all states into a single vector
    let flat: Vec<f32> = states.iter().flatten().copied().collect();

    let data = TensorData::new(flat, [batch_size, state_dim]);
    Tensor::from_data(data, device)
}

/// Parse a batch of transitions into separate tensor components.
///
/// Extracts all components from a batch of Transition structs and converts
/// them to tensors suitable for batch training operations.
///
/// # Arguments
///
/// * `batch` - Slice of Transition structs
/// * `state_dim` - Expected state dimension
/// * `device` - Target device for tensor placement
///
/// # Returns
///
/// Tuple containing:
/// - `states`: Tensor of shape `[batch_size, state_dim]`
/// - `actions`: Tensor of shape `[batch_size, 1]` (Int tensor for discrete actions)
/// - `rewards`: Tensor of shape `[batch_size, 1]`
/// - `next_states`: Tensor of shape `[batch_size, state_dim]`
/// - `dones`: Tensor of shape `[batch_size, 1]`
///
/// For empty batches, returns appropriately sized empty tensors.
///
/// # Action Handling
///
/// For `Action::Discrete(idx)`, extracts the action index.
/// For `Action::Continuous(_)`, uses action index 0 (fallback for compatible tensor shapes).
///
/// # Examples
///
/// ```rust,ignore
/// use eris::policies::policy::{State, Action, Transition};
/// use burn::backend::NdArray;
/// use burn::tensor::backend::Backend;
///
/// let device = <NdArray as Backend>::Device::default();
///
/// // Create batch of transitions
/// let batch = vec![
///     Transition {
///         state: State::Features(vec![1.0, 2.0]),
///         action: Action::Discrete(1),
///         reward: 0.5,
///         next_state: State::Features(vec![1.1, 2.1]),
///         done: false,
///     },
///     Transition {
///         state: State::Features(vec![3.0, 4.0]),
///         action: Action::Discrete(2),
///         reward: -0.3,
///         next_state: State::Features(vec![3.1, 4.1]),
///         done: true,
///     },
/// ];
///
/// let (states, actions, rewards, next_states, dones) =
///     batch_to_tensors::<NdArray>(&batch, 2, &device);
///
/// assert_eq!(states.shape().dims, [2, 2]);
/// assert_eq!(actions.shape().dims, [2, 1]);
/// assert_eq!(rewards.shape().dims, [2, 1]);
/// ```
pub fn batch_to_tensors<B: Backend>(
    batch: &[Transition],
    state_dim: usize,
    device: &B::Device,
) -> (
    Tensor<B, 2>,      // states
    Tensor<B, 2, Int>, // actions
    Tensor<B, 2>,      // rewards
    Tensor<B, 2>,      // next_states
    Tensor<B, 2>,      // dones
) {
    if batch.is_empty() {
        return (
            Tensor::zeros([0, state_dim], device),
            Tensor::zeros([0, 1], device),
            Tensor::zeros([0, 1], device),
            Tensor::zeros([0, state_dim], device),
            Tensor::zeros([0, 1], device),
        );
    }

    // Extract states
    let states: Vec<Vec<f32>> = batch
        .iter()
        .map(|t| match &t.state {
            State::Features(f) => f.clone(),
            State::Raw(r) => r.iter().map(|&x| x as f32).collect(),
            State::Empty => vec![0.0; state_dim],
        })
        .collect();

    // Extract actions (discrete action indices)
    let actions: Vec<i32> = batch
        .iter()
        .map(|t| match t.action {
            Action::Discrete(a) => a as i32,
            Action::Continuous(_) => 0, // Fallback for continuous actions
        })
        .collect();

    // Extract rewards
    let rewards: Vec<f32> = batch.iter().map(|t| t.reward).collect();

    // Extract next states
    let next_states: Vec<Vec<f32>> = batch
        .iter()
        .map(|t| match &t.next_state {
            State::Features(f) => f.clone(),
            State::Raw(r) => r.iter().map(|&x| x as f32).collect(),
            State::Empty => vec![0.0; state_dim],
        })
        .collect();

    // Extract done flags
    let dones: Vec<f32> = batch
        .iter()
        .map(|t| if t.done { 1.0 } else { 0.0 })
        .collect();

    // Convert to tensors
    let states_tensor = states_to_tensor(&states, device);
    let actions_data = TensorData::new(actions.clone(), [actions.len(), 1]);
    let actions_tensor = Tensor::<B, 2, Int>::from_data(actions_data.convert::<i32>(), device);
    let rewards_data = TensorData::new(rewards.clone(), [rewards.len(), 1]);
    let rewards_tensor = Tensor::from_data(rewards_data, device);
    let next_states_tensor = states_to_tensor(&next_states, device);
    let dones_data = TensorData::new(dones.clone(), [dones.len(), 1]);
    let dones_tensor = Tensor::from_data(dones_data, device);

    (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Backend;

    type TestBackend = NdArray;

    #[test]
    fn test_state_to_tensor_features() {
        let device = <TestBackend as Backend>::Device::default();
        let state = State::Features(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let tensor = state_to_tensor::<TestBackend>(&state, 5, &device);

        assert_eq!(tensor.shape().dims, [1, 5]);

        let values: Vec<f32> = tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert tensor");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_state_to_tensor_raw() {
        let device = <TestBackend as Backend>::Device::default();
        let state = State::Raw(vec![1.5, 2.5, 3.5]);
        let tensor = state_to_tensor::<TestBackend>(&state, 3, &device);

        assert_eq!(tensor.shape().dims, [1, 3]);

        let values: Vec<f32> = tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert tensor");
        // Raw is Vec<f64>, converted to f32
        assert!((values[0] - 1.5).abs() < 1e-5);
        assert!((values[1] - 2.5).abs() < 1e-5);
        assert!((values[2] - 3.5).abs() < 1e-5);
    }

    #[test]
    fn test_state_to_tensor_empty() {
        let device = <TestBackend as Backend>::Device::default();
        let state = State::Empty;
        let tensor = state_to_tensor::<TestBackend>(&state, 10, &device);

        assert_eq!(tensor.shape().dims, [1, 10]);

        let values: Vec<f32> = tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert tensor");
        assert!(values.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_states_to_tensor_batch() {
        let device = <TestBackend as Backend>::Device::default();
        let states = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let tensor = states_to_tensor::<TestBackend>(&states, &device);

        assert_eq!(tensor.shape().dims, [3, 3]);

        let values: Vec<f32> = tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert tensor");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_states_to_tensor_empty() {
        let device = <TestBackend as Backend>::Device::default();
        let states: Vec<Vec<f32>> = vec![];
        let tensor = states_to_tensor::<TestBackend>(&states, &device);

        assert_eq!(tensor.shape().dims, [0, 0]);
    }

    #[test]
    fn test_states_to_tensor_single() {
        let device = <TestBackend as Backend>::Device::default();
        let states = vec![vec![1.0, 2.0]];
        let tensor = states_to_tensor::<TestBackend>(&states, &device);

        assert_eq!(tensor.shape().dims, [1, 2]);

        let values: Vec<f32> = tensor
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert tensor");
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_batch_to_tensors_discrete_actions() {
        let device = <TestBackend as Backend>::Device::default();
        let batch = vec![
            Transition {
                state: State::Features(vec![1.0, 2.0]),
                action: Action::Discrete(1),
                reward: 0.5,
                next_state: State::Features(vec![1.1, 2.1]),
                done: false,
            },
            Transition {
                state: State::Features(vec![3.0, 4.0]),
                action: Action::Discrete(2),
                reward: -0.3,
                next_state: State::Features(vec![3.1, 4.1]),
                done: true,
            },
        ];

        let (states, actions, rewards, next_states, dones) =
            batch_to_tensors::<TestBackend>(&batch, 2, &device);

        assert_eq!(states.shape().dims, [2, 2]);
        assert_eq!(actions.shape().dims, [2, 1]);
        assert_eq!(rewards.shape().dims, [2, 1]);
        assert_eq!(next_states.shape().dims, [2, 2]);
        assert_eq!(dones.shape().dims, [2, 1]);

        // Verify state values
        let state_values: Vec<f32> = states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert");
        assert_eq!(state_values, vec![1.0, 2.0, 3.0, 4.0]);

        // Verify action values
        let action_values: Vec<i32> = actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("Failed to convert");
        assert_eq!(action_values, vec![1, 2]);

        // Verify reward values
        let reward_values: Vec<f32> = rewards
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert");
        assert!((reward_values[0] - 0.5).abs() < 1e-5);
        assert!((reward_values[1] - (-0.3)).abs() < 1e-5);

        // Verify done values
        let done_values: Vec<f32> = dones
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert");
        assert!((done_values[0] - 0.0).abs() < 1e-5);
        assert!((done_values[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_to_tensors_continuous_actions() {
        let device = <TestBackend as Backend>::Device::default();
        let batch = vec![Transition {
            state: State::Features(vec![1.0]),
            action: Action::Continuous(vec![0.5, 0.3]),
            reward: 1.0,
            next_state: State::Features(vec![1.1]),
            done: false,
        }];

        let (states, actions, rewards, next_states, dones) =
            batch_to_tensors::<TestBackend>(&batch, 1, &device);

        assert_eq!(states.shape().dims, [1, 1]);

        // Continuous actions should use fallback index 0
        let action_values: Vec<i32> = actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("Failed to convert");
        assert_eq!(action_values, vec![0]);
    }

    #[test]
    fn test_batch_to_tensors_empty_batch() {
        let device = <TestBackend as Backend>::Device::default();
        let batch: Vec<Transition> = vec![];

        let (states, actions, rewards, next_states, dones) =
            batch_to_tensors::<TestBackend>(&batch, 10, &device);

        assert_eq!(states.shape().dims, [0, 10]);
        assert_eq!(actions.shape().dims, [0, 1]);
        assert_eq!(rewards.shape().dims, [0, 1]);
        assert_eq!(next_states.shape().dims, [0, 10]);
        assert_eq!(dones.shape().dims, [0, 1]);
    }

    #[test]
    fn test_batch_to_tensors_raw_states() {
        let device = <TestBackend as Backend>::Device::default();
        let batch = vec![Transition {
            state: State::Raw(vec![1.5, 2.5]),
            action: Action::Discrete(0),
            reward: 0.0,
            next_state: State::Raw(vec![1.6, 2.6]),
            done: false,
        }];

        let (states, _actions, _rewards, next_states, _dones) =
            batch_to_tensors::<TestBackend>(&batch, 2, &device);

        // Raw states should be converted to f32
        let state_values: Vec<f32> = states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert");
        assert!((state_values[0] - 1.5).abs() < 1e-5);
        assert!((state_values[1] - 2.5).abs() < 1e-5);

        let next_state_values: Vec<f32> = next_states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert");
        assert!((next_state_values[0] - 1.6).abs() < 1e-5);
        assert!((next_state_values[1] - 2.6).abs() < 1e-5);
    }

    #[test]
    fn test_batch_to_tensors_empty_states() {
        let device = <TestBackend as Backend>::Device::default();
        let batch = vec![Transition {
            state: State::Empty,
            action: Action::Discrete(1),
            reward: 1.0,
            next_state: State::Empty,
            done: true,
        }];

        let (states, _actions, _rewards, next_states, _dones) =
            batch_to_tensors::<TestBackend>(&batch, 5, &device);

        // Empty states should produce zero tensors
        let state_values: Vec<f32> = states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert");
        assert_eq!(state_values.len(), 5);
        assert!(state_values.iter().all(|&x| x == 0.0));

        let next_state_values: Vec<f32> = next_states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert");
        assert_eq!(next_state_values.len(), 5);
        assert!(next_state_values.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_batch_to_tensors_mixed_states() {
        let device = <TestBackend as Backend>::Device::default();
        let batch = vec![
            Transition {
                state: State::Features(vec![1.0, 2.0]),
                action: Action::Discrete(0),
                reward: 0.5,
                next_state: State::Raw(vec![1.5, 2.5]),
                done: false,
            },
            Transition {
                state: State::Empty,
                action: Action::Discrete(1),
                reward: -0.2,
                next_state: State::Features(vec![2.0, 3.0]),
                done: true,
            },
        ];

        let (states, actions, rewards, next_states, dones) =
            batch_to_tensors::<TestBackend>(&batch, 2, &device);

        assert_eq!(states.shape().dims, [2, 2]);
        assert_eq!(actions.shape().dims, [2, 1]);
        assert_eq!(rewards.shape().dims, [2, 1]);
        assert_eq!(next_states.shape().dims, [2, 2]);
        assert_eq!(dones.shape().dims, [2, 1]);

        // First state is Features, should retain values
        let state_values: Vec<f32> = states
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("Failed to convert");
        assert_eq!(state_values[0], 1.0);
        assert_eq!(state_values[1], 2.0);
        // Second state is Empty, should be zeros
        assert_eq!(state_values[2], 0.0);
        assert_eq!(state_values[3], 0.0);
    }
}
