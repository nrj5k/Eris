use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor};

/// Compute TD target: r + γ * max_next_q * (1 - done)
pub fn compute_td_target<B: AutodiffBackend>(
    rewards: &Tensor<B, 1>,
    max_next_q: &Tensor<B, 1>,
    dones: &Tensor<B, 1>,
    gamma: f32,
) -> Tensor<B, 1> {
    let ones = Tensor::<B, 1>::ones_like(rewards);
    let not_done = ones - dones.clone();
    let gamma_t = Tensor::<B, 1>::full_like(rewards, gamma);
    rewards.clone() + gamma_t * max_next_q.clone() * not_done
}

/// Compute Double DQN loss with MSE
pub fn compute_double_dqn_loss<B: AutodiffBackend>(
    current_q: &Tensor<B, 1>,
    target_q: &Tensor<B, 1>,
) -> Tensor<B, 1> {
    let diff = current_q.clone() - target_q.clone();
    diff.powf_scalar(2.0).mean()
}

/// Gather Q-values for taken actions using gather
pub fn gather_q_values<B: AutodiffBackend>(
    q_values: &Tensor<B, 2>,
    actions: &Tensor<B, 1, Int>,
) -> Tensor<B, 1> {
    let batch_size = q_values.dims()[0];
    let actions_2d = actions.clone().reshape([batch_size, 1]);
    q_values.clone().gather(1, actions_2d).squeeze()
}

/// Safely extract scalar value from loss tensor
///
/// # Arguments
/// * `tensor` - Loss tensor (should be scalar)
///
/// # Returns
/// Scalar f32 value, or 0.0 if tensor is empty
pub fn loss_to_scalar<B: AutodiffBackend>(tensor: Tensor<B, 1>) -> f32 {
    tensor
        .into_data()
        .convert::<f32>()
        .as_slice()
        .ok()
        .and_then(|s| s.first())
        .copied()
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::TensorData;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_compute_td_target() {
        let device = Default::default();
        let rewards = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let max_next_q = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let dones = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let target = compute_td_target(&rewards, &max_next_q, &dones, 0.99);
        let target_val: f32 = target.into_data().convert::<f32>().as_slice().unwrap()[0];
        assert!(
            (target_val - 1.495).abs() < 1e-3,
            "Expected ~1.495, got {}",
            target_val
        );
    }

    #[test]
    fn test_compute_double_dqn_loss() {
        let device = Default::default();
        let current_q = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let target_q = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let loss = compute_double_dqn_loss(&current_q, &target_q);
        let loss_val: f32 = loss.into_data().convert::<f32>().as_slice().unwrap()[0];
        assert!(
            (loss_val - 0.25).abs() < 1e-3,
            "Expected ~0.25, got {}",
            loss_val
        );
    }

    #[test]
    fn test_gather_q_values() {
        let device = Default::default();
        // Use batch size 2 to avoid squeeze edge case
        let q_values =
            Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let actions = Tensor::<TestBackend, 1, Int>::from_data(
            TensorData::new(vec![1i32, 2i32], [2]).convert::<i32>(),
            &device,
        );
        let gathered = gather_q_values(&q_values, &actions);
        let gathered_slice: Vec<f32> = gathered
            .into_data()
            .convert::<f32>()
            .as_slice()
            .unwrap()
            .to_vec();
        assert!(
            (gathered_slice[0] - 2.0).abs() < 1e-3,
            "Expected 2.0, got {}",
            gathered_slice[0]
        );
        assert!(
            (gathered_slice[1] - 6.0).abs() < 1e-3,
            "Expected 6.0, got {}",
            gathered_slice[1]
        );
    }
}
