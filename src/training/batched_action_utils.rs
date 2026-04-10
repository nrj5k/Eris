//! Shared utilities for batched action selection
//!
//! This module provides GPU-optimized helper functions that are used
//! across multiple policies (DQN, Catcher, etc.) to avoid code duplication.

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Distribution, Int, Tensor, TensorData};

/// Convert observations to batched tensor on GPU device.
///
/// # Arguments
///
/// * `observations` - Batch of observations as Vec<Vec<f64>>
/// * `device` - GPU device for tensor allocation
///
/// # Returns
///
/// Tensor of shape [batch_size, state_dim] on GPU
///
/// # Example
///
/// ```rust,ignore
/// let observations = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
/// let tensor = observations_to_tensor(&observations, &device);
/// assert_eq!(tensor.shape().dims, [2, 2]);
/// ```
#[inline]
pub fn observations_to_tensor<B: AutodiffBackend>(
    observations: &[Vec<f64>],
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch_size = observations.len();
    if batch_size == 0 {
        return Tensor::zeros([0, 0], device);
    }

    let state_dim = observations[0].len();
    let states_flat: Vec<f32> = observations
        .iter()
        .flat_map(|obs| obs.iter().map(|x| *x as f32))
        .collect();

    let states_data = TensorData::new(states_flat, [batch_size, state_dim]);
    Tensor::from_data(states_data.convert::<f32>(), device)
}

/// Apply epsilon-greedy action selection on GPU.
///
/// Selects actions using epsilon-greedy policy:
/// - With probability epsilon: random action
/// - With probability (1-epsilon): argmax(Q-values)
///
/// # Arguments
///
/// * `q_values` - Q-values from network, shape [batch_size, action_dim]
/// * `action_dim` - Number of possible actions
/// * `epsilon` - Exploration rate (0.0 to 1.0)
/// * `device` - GPU device for random generation
///
/// # Returns
///
/// Vector of selected actions (one per batch element)
///
/// # Example
///
/// ```rust,ignore
/// let q_values = model.forward(states);  // [batch_size, action_dim]
/// let actions = epsilon_greedy_select(q_values, 10, 0.1, &device);
/// ```
#[inline]
pub fn epsilon_greedy_select<B: AutodiffBackend>(
    q_values: Tensor<B, 2>,
    action_dim: usize,
    epsilon: f32,
    device: &B::Device,
) -> Vec<usize> {
    let batch_size = q_values.shape().dims[0];

    // Generate random actions for exploration [batch_size]
    let random_float = Tensor::<B, 1>::random(
        [batch_size],
        Distribution::Uniform(0.0, action_dim as f64),
        device,
    );
    let random_actions: Tensor<B, 1, Int> = random_float.int();

    // Get greedy actions (argmax Q-values) - shape [batch_size, 1]
    let greedy_actions_2d: Tensor<B, 2, Int> = q_values.argmax(1);

    // Reshape to [batch_size]
    let greedy_actions: Tensor<B, 1, Int> = greedy_actions_2d.reshape([batch_size]);

    // Generate random values for epsilon-greedy decision
    let random_vals: Tensor<B, 1> =
        Tensor::random([batch_size], Distribution::Uniform(0.0, 1.0), device);

    // Create explore mask: random_vals < epsilon (Bool tensor)
    let explore_mask = random_vals.lower_elem(epsilon as f64);

    // Select: where explore_mask == false (exploit), use greedy; where true (explore), use random
    // mask_where(condition, replacement): where condition == false, use replacement
    let selected = greedy_actions.mask_where(explore_mask, random_actions);

    // Convert to Vec<usize>
    selected
        .into_data()
        .convert::<i64>()
        .as_slice()
        .unwrap()
        .iter()
        .map(|x: &i64| *x as usize)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn test_observations_to_tensor() {
        type TestBackend = Autodiff<NdArray>;
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        let observations = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let tensor = observations_to_tensor::<TestBackend>(&observations, &device);

        assert_eq!(tensor.shape().dims, [2, 3]);
    }

    #[test]
    fn test_epsilon_greedy_select() {
        type TestBackend = Autodiff<NdArray>;
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        // Create Q-values that strongly favor action 2
        let q_data = TensorData::new(vec![0.1f32, 0.2, 0.9, 0.1, 0.3, 0.8], [2, 3]);
        let q_values: Tensor<TestBackend, 2> = Tensor::from_data(q_data.convert::<f32>(), &device);

        // With epsilon=0.0, should always select greedy (action 2)
        let actions = epsilon_greedy_select::<TestBackend>(q_values, 3, 0.0, &device);

        assert_eq!(actions.len(), 2);
        // Both should select action 2 (index 2) with epsilon=0.0
        assert_eq!(actions[0], 2);
        assert_eq!(actions[1], 2);
    }

    #[test]
    fn test_observations_to_tensor_empty() {
        type TestBackend = Autodiff<NdArray>;
        let device = <TestBackend as burn::tensor::backend::Backend>::Device::default();

        let observations: Vec<Vec<f64>> = vec![];
        let tensor = observations_to_tensor::<TestBackend>(&observations, &device);

        assert_eq!(tensor.shape().dims, [0, 0]);
    }
}
