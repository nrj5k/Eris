use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor};

/// Async loss accumulator for GPU-native training
///
/// Accumulates loss on GPU to avoid frequent GPU→CPU synchronization.
/// Only syncs to CPU every `sync_freq` steps for better performance.
pub struct LossAccumulator<B: AutodiffBackend> {
    /// Accumulated loss tensor on GPU
    accumulated_loss: Tensor<B, 1>,
    /// Count of accumulated steps
    accumulated_count: usize,
    /// Frequency of sync to CPU
    sync_freq: usize,
    /// Device for tensor operations
    device: B::Device,
}

impl<B: AutodiffBackend> LossAccumulator<B> {
    /// Create new loss accumulator
    ///
    /// # Arguments
    /// * `sync_freq` - How often to sync loss to CPU (e.g., 100 = every 100 steps)
    /// * `device` - Backend device for tensor operations
    pub fn new(sync_freq: usize, device: &B::Device) -> Self {
        Self {
            accumulated_loss: Tensor::<B, 1>::zeros([1], device),
            accumulated_count: 0,
            sync_freq,
            device: device.clone(),
        }
    }

    /// Accumulate a loss value
    ///
    /// # Arguments
    /// * `loss` - Loss tensor to accumulate
    pub fn accumulate(&mut self, loss: Tensor<B, 1>) {
        self.accumulated_loss = self.accumulated_loss.clone() + loss;
        self.accumulated_count += 1;
    }

    /// Try to sync accumulated loss to CPU
    ///
    /// Returns averaged loss value if sync_freq threshold is reached, None otherwise.
    /// Resets accumulator after sync.
    ///
    /// # Returns
    /// * `Some(f32)` - Averaged loss value if sync threshold reached
    /// * `None` - Not yet time to sync
    pub fn try_sync(&mut self) -> Option<f32> {
        if self.accumulated_count.is_multiple_of(self.sync_freq) {
            let avg_loss = self.accumulated_loss.clone() / self.accumulated_count as f32;
            let loss_value = loss_to_scalar(avg_loss);

            // Reset accumulator
            self.accumulated_loss = Tensor::<B, 1>::zeros([1], &self.device);
            self.accumulated_count = 0;

            Some(loss_value)
        } else {
            None
        }
    }

    /// Force sync accumulated loss regardless of threshold
    ///
    /// Use this at end of training to get final loss value.
    ///
    /// # Returns
    /// * `Some(f32)` - Averaged loss value if any loss was accumulated
    /// * `None` - No loss accumulated
    pub fn force_sync(&mut self) -> Option<f32> {
        if self.accumulated_count > 0 {
            let avg_loss = self.accumulated_loss.clone() / self.accumulated_count as f32;
            let loss_value = loss_to_scalar(avg_loss);

            // Reset accumulator
            self.accumulated_loss = Tensor::<B, 1>::zeros([1], &self.device);
            self.accumulated_count = 0;

            Some(loss_value)
        } else {
            None
        }
    }

    /// Get current accumulated count
    pub fn count(&self) -> usize {
        self.accumulated_count
    }

    /// Check if accumulator is empty
    pub fn is_empty(&self) -> bool {
        self.accumulated_count == 0
    }
}

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

/// Compute standard DQN temporal difference loss (rank-2 version).
///
/// Uses rank-2 tensor layout: Q-values [batch_size, action_dim],
/// actions [batch_size, 1], rewards [batch_size, 1], dones [batch_size, 1].
/// This is the standard layout for DQN training.
///
/// Loss = mean((r + γ * max_a' Q_target(s', a') * (1 - done) - Q(s, a))^2)
pub fn compute_td_loss<B: AutodiffBackend>(
    q_values: Tensor<B, 2>,
    target_q_values: Tensor<B, 2>,
    actions: &Tensor<B, 2, Int>,
    rewards: &Tensor<B, 2>,
    dones: &Tensor<B, 2>,
    gamma: f32,
) -> Tensor<B, 1> {
    let device = q_values.device();
    let batch_size = rewards.shape().dims[0];

    // Gather Q-values for taken actions
    let current_q = q_values.gather(1, actions.clone());

    // Max Q from target network
    let max_next_q = target_q_values.max_dim(1);

    // TD target = r + gamma * max_next_q * (1 - done)
    let target_q = rewards.clone()
        + Tensor::full([batch_size, 1], gamma, &device)
            * max_next_q
            * (Tensor::ones_like(dones) - dones.clone());

    // MSE loss
    let diff = current_q - target_q.detach();
    diff.powf_scalar(2.0).mean()
}

/// Compute Double DQN temporal difference loss (rank-2 version).
///
/// Uses rank-2 tensor layout. Policy network selects actions, target evaluates.
///
/// Loss = mean((r + γ * Q_target(s', argmax Q_policy(s', a')) * (1 - done) - Q(s, a))^2)
pub fn compute_double_dqn_loss_rank2<B: AutodiffBackend>(
    q_values: Tensor<B, 2>,
    next_q_policy: Tensor<B, 2>,
    next_q_target: Tensor<B, 2>,
    actions: &Tensor<B, 2, Int>,
    rewards: &Tensor<B, 2>,
    dones: &Tensor<B, 2>,
    gamma: f32,
) -> Tensor<B, 1> {
    let device = q_values.device();
    let batch_size = rewards.shape().dims[0];

    // Gather Q-values for taken actions
    let current_q = q_values.gather(1, actions.clone());

    // Policy selects best actions
    let best_actions = next_q_policy.argmax(1);

    // Target evaluates selected actions
    let max_next_q = next_q_target.gather(1, best_actions);

    // TD target
    let target_q = rewards.clone()
        + Tensor::full([batch_size, 1], gamma, &device)
            * max_next_q
            * (Tensor::ones_like(dones) - dones.clone());

    // MSE loss
    let diff = current_q - target_q.detach();
    diff.powf_scalar(2.0).mean()
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

    #[test]
    fn test_loss_accumulator_creation() {
        let device = Default::default();
        let accumulator = LossAccumulator::<TestBackend>::new(100, &device);
        assert_eq!(accumulator.count(), 0);
        assert!(accumulator.is_empty());
    }

    #[test]
    fn test_loss_accumulator_accumulate() {
        let device = Default::default();
        let mut accumulator = LossAccumulator::<TestBackend>::new(100, &device);

        let loss = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        accumulator.accumulate(loss);

        assert_eq!(accumulator.count(), 1);
        assert!(!accumulator.is_empty());
    }

    #[test]
    fn test_loss_accumulator_try_sync() {
        let device = Default::default();
        let mut accumulator = LossAccumulator::<TestBackend>::new(3, &device);

        // Accumulate 3 losses (should sync on 3rd)
        for i in 1..=3 {
            let loss = Tensor::<TestBackend, 1>::from_floats([0.3], &device);
            accumulator.accumulate(loss);

            if i < 3 {
                assert!(accumulator.try_sync().is_none());
            }
        }

        // 3rd accumulation should sync
        let result = accumulator.try_sync();
        assert!(result.is_some());
        assert!((result.unwrap() - 0.3).abs() < 1e-3);
        assert_eq!(accumulator.count(), 0);
        assert!(accumulator.is_empty());
    }

    #[test]
    fn test_loss_accumulator_force_sync() {
        let device = Default::default();
        let mut accumulator = LossAccumulator::<TestBackend>::new(100, &device);

        // Accumulate 5 losses (not enough for sync)
        for _ in 0..5 {
            let loss = Tensor::<TestBackend, 1>::from_floats([0.4], &device);
            accumulator.accumulate(loss);
        }

        // Force sync should still return value
        let result = accumulator.force_sync();
        assert!(result.is_some());
        assert!((result.unwrap() - 0.4).abs() < 1e-3);
        assert!(accumulator.is_empty());
    }

    #[test]
    fn test_loss_accumulator_force_sync_empty() {
        let device = Default::default();
        let mut accumulator = LossAccumulator::<TestBackend>::new(100, &device);

        // Force sync on empty accumulator
        let result = accumulator.force_sync();
        assert!(result.is_none());
    }

    #[test]
    fn test_loss_accumulator_new() {
        let device = Default::default();
        let acc = LossAccumulator::<TestBackend>::new(100, &device);

        assert_eq!(acc.count(), 0);
        assert!(acc.is_empty());
    }

    #[test]
    fn test_accumulate_returns_none_until_sync_freq() {
        let device = Default::default();
        let mut acc = LossAccumulator::<TestBackend>::new(100, &device);

        // Accumulate 99 times - should return None each time
        for _ in 0..99 {
            let loss = Tensor::<TestBackend, 1>::from_floats([1.0f32], &device);
            acc.accumulate(loss);
            let result = acc.try_sync();
            assert!(result.is_none(), "Should return None before sync_freq");
        }

        assert_eq!(acc.count(), 99);
    }

    #[test]
    fn test_accumulate_returns_some_on_sync_freq() {
        let device = Default::default();
        let mut acc = LossAccumulator::<TestBackend>::new(100, &device);

        // Accumulate 99 times
        for _ in 0..99 {
            let loss = Tensor::<TestBackend, 1>::from_floats([1.0f32], &device);
            acc.accumulate(loss);
        }

        // 100th accumulation
        let loss = Tensor::<TestBackend, 1>::from_floats([1.0f32], &device);
        acc.accumulate(loss);

        // try_sync should return Some on sync_freq
        let result = acc.try_sync();

        assert!(result.is_some(), "Should return Some on sync_freq");
        // Average of 100 ones = 1.0
        assert!((result.unwrap() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_force_sync_flushes_remaining() {
        let device = Default::default();
        let mut acc = LossAccumulator::<TestBackend>::new(100, &device);

        // Accumulate only 50 times
        for _ in 0..50 {
            let loss = Tensor::<TestBackend, 1>::from_floats([2.0f32], &device);
            acc.accumulate(loss);
        }

        // Force sync should return average of 50 accumulated losses
        let result = acc.force_sync();
        assert!(result.is_some());
        assert!((result.unwrap() - 2.0).abs() < 0.001);

        // Count should be reset
        assert_eq!(acc.count(), 0);
        assert!(acc.is_empty());
    }

    #[test]
    fn test_reset_clears_accumulator() {
        let device = Default::default();
        let mut acc = LossAccumulator::<TestBackend>::new(100, &device);

        // Accumulate some losses
        for _ in 0..10 {
            let loss = Tensor::<TestBackend, 1>::from_floats([1.0f32], &device);
            acc.accumulate(loss);
        }

        assert_eq!(acc.count(), 10);

        // Reset via force_sync (only reset mechanism available)
        acc.force_sync();

        assert_eq!(acc.count(), 0);
        assert!(acc.is_empty());
    }

    #[test]
    fn test_average_calculation_correct() {
        let device = Default::default();
        let mut acc = LossAccumulator::<TestBackend>::new(3, &device); // Sync every 3

        // Accumulate losses: 1.0, 2.0, 3.0
        let loss1 = Tensor::<TestBackend, 1>::from_floats([1.0f32], &device);
        acc.accumulate(loss1);

        let loss2 = Tensor::<TestBackend, 1>::from_floats([2.0f32], &device);
        acc.accumulate(loss2);

        let loss3 = Tensor::<TestBackend, 1>::from_floats([3.0f32], &device);
        acc.accumulate(loss3);

        let result = acc.try_sync().unwrap();

        // Average should be 2.0
        assert!((result - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_sync_freq_1_syncs_every_step() {
        let device = Default::default();
        let mut acc = LossAccumulator::<TestBackend>::new(1, &device); // Sync every step

        // Every accumulation should return Some
        for _ in 0..5 {
            let loss = Tensor::<TestBackend, 1>::from_floats([1.0f32], &device);
            acc.accumulate(loss);
            let result = acc.try_sync();
            assert!(result.is_some(), "Should sync every step with freq=1");
        }
    }

    #[test]
    fn test_multiple_sync_cycles() {
        let device = Default::default();
        let mut acc = LossAccumulator::<TestBackend>::new(5, &device);

        // First cycle: 5 losses of 1.0
        for _ in 0..5 {
            let loss = Tensor::<TestBackend, 1>::from_floats([1.0f32], &device);
            acc.accumulate(loss);
        }
        let result1 = acc.try_sync().unwrap();
        assert!((result1 - 1.0).abs() < 0.001);

        // Second cycle: 5 losses of 3.0
        for _ in 0..5 {
            let loss = Tensor::<TestBackend, 1>::from_floats([3.0f32], &device);
            acc.accumulate(loss);
        }
        let result2 = acc.try_sync().unwrap();
        assert!((result2 - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_is_empty_after_operations() {
        let device = Default::default();
        let mut acc = LossAccumulator::<TestBackend>::new(10, &device);

        // Initially empty
        assert!(acc.is_empty());

        // After accumulate, not empty
        let loss = Tensor::<TestBackend, 1>::from_floats([1.0f32], &device);
        acc.accumulate(loss);
        assert!(!acc.is_empty());

        // After force_sync, empty again
        acc.force_sync();
        assert!(acc.is_empty());
    }

    #[test]
    fn test_compute_td_loss_rank2() {
        let device = Default::default();
        let q_values = Tensor::<TestBackend, 2>::zeros([4, 10], &device);
        let target_q = Tensor::<TestBackend, 2>::zeros([4, 10], &device);
        let actions = Tensor::<TestBackend, 2, Int>::zeros([4, 1], &device);
        let rewards = Tensor::<TestBackend, 2>::zeros([4, 1], &device);
        let dones: Tensor<TestBackend, 2> = Tensor::ones([4, 1], &device);

        let loss = compute_td_loss(q_values, target_q, &actions, &rewards, &dones, 0.99);
        let loss_val: f32 = loss.into_data().convert::<f32>().as_slice().unwrap()[0];
        // With all dones=true and zero rewards, TD target = reward, loss = (0-0)^2 = 0
        assert!(
            (loss_val - 0.0).abs() < 1e-5,
            "Expected ~0, got {}",
            loss_val
        );
    }

    #[test]
    fn test_compute_double_dqn_loss_rank2() {
        let device = Default::default();
        let q_values = Tensor::<TestBackend, 2>::ones([2, 3], &device);
        let next_q_policy = Tensor::<TestBackend, 2>::zeros([2, 3], &device);
        let next_q_target = Tensor::<TestBackend, 2>::zeros([2, 3], &device);
        let actions = Tensor::<TestBackend, 2, Int>::zeros([2, 1], &device);
        let rewards = Tensor::<TestBackend, 2>::zeros([2, 1], &device);
        let dones = Tensor::<TestBackend, 2>::ones([2, 1], &device);

        let loss = compute_double_dqn_loss_rank2(
            q_values,
            next_q_policy,
            next_q_target,
            &actions,
            &rewards,
            &dones,
            0.99,
        );
        let loss_val: f32 = loss.into_data().convert::<f32>().as_slice().unwrap()[0];
        assert!(
            loss_val.is_finite(),
            "Loss should be finite, got {}",
            loss_val
        );
    }
}
