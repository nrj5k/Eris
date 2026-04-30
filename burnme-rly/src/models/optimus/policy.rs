//! Optimus policy wrapper for eris integration with GPU support
//!
//! This module provides the OptimusPolicy struct which wraps the iTransformer model
//! for cache prefetching predictions. It implements the GpuTrainable, BatchedActionSelector,
//! and Checkpointable traits for integration with the eris training pipeline.

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

use crate::buffer::{CpuRingBuffer, TensorTransitionBatch};
use crate::checkpoint::Checkpointable;
use crate::traits::{BatchedActionSelector, GpuTrainable};

use super::bridge::{
    burn_device_to_candle, burn_to_candle, candle_to_burn, device_name, select_device,
};
use super::{OptimusConfig, OptimusModel};
use candle_core::Device as CandleDevice;

/// Optimus policy for cache prefetching using iTransformer
///
/// This policy wraps the iTransformer model for time-series forecasting of cache accesses.
/// It operates on time windows rather than standard RL transitions.
///
/// # Type Parameters
/// * `B` - The Burn backend type (must implement AutodiffBackend for training)
///
/// # Architecture
/// - Input: [batch, num_variates, lookback_len] - historical cache access patterns
/// - Output: [batch, pred_len, num_variates] - predicted future cache accesses
/// - Action selection: argmax over predicted values to select prefetch candidates
///
/// # Device Selection
/// The Candle device is automatically derived from the Burn backend device.
/// No separate device specification is needed.
pub struct OptimusPolicy<B: AutodiffBackend> {
    model: OptimusModel,
    config: OptimusConfig,
    burn_device: B::Device,
    step_count: usize,
    warmup_complete: bool,
    // For action selection
    action_dim: usize,
    // Dummy buffer for trait compliance (Optimus uses time windows, not standard replay)
    _buffer: CpuRingBuffer,
}

impl<B: AutodiffBackend> OptimusPolicy<B> {
    /// Create new policy from config with optional device override.
    ///
    /// # Arguments
    /// * `config` - Optimus model configuration
    /// * `burn_device` - Burn backend device for output tensors
    /// * `action_dim` - Number of possible actions (cache line buckets to prefetch)
    /// * `device_override` - Optional device string override ("cpu", "cuda", "cuda:0", etc.)
    ///
    /// # Returns
    /// A new OptimusPolicy instance with initialized model
    ///
    /// # Device Selection
    /// If `device_override` is None, the Candle device is automatically derived from the Burn backend.
    /// If `device_override` is Some, uses the specified device:
    /// - "cpu" forces CPU
    /// - "cuda" or "cuda:0" uses first GPU
    /// - "cuda:N" uses specific GPU index N
    /// - Unknown values warn and fall back to auto-detection
    ///
    /// # Panics
    /// Panics if the iTransformer model fails to initialize
    pub fn new(
        config: OptimusConfig,
        burn_device: B::Device,
        action_dim: usize,
        device_override: Option<&str>,
    ) -> Self {
        // Select Candle device with optional override
        let candle_device =
            select_device::<B>(&burn_device, device_override).unwrap_or(CandleDevice::Cpu);

        let model =
            OptimusModel::new(&config, &candle_device).expect("Failed to create Optimus model");

        log::info!(
            "[OptimusPolicy] Created on {}",
            device_name::<B>(&burn_device)
        );

        Self {
            model,
            config,
            burn_device,
            step_count: 0,
            warmup_complete: false,
            action_dim,
            _buffer: CpuRingBuffer::new(1000),
        }
    }

    /// Predict future cache accesses
    ///
    /// # Arguments
    /// * `history` - Historical cache access patterns with shape [batch, num_variates, lookback_len]
    ///
    /// # Returns
    /// * `Some(Tensor<B, 3>)` - Predictions with shape [batch, pred_len, num_variates]
    /// * `None` - If conversion or forward pass fails
    ///
    /// # Input/Output Shapes
    /// - Input: [batch, num_variates, lookback_len] - history of cache accesses
    /// - Output: [batch, pred_len, num_variates] - predicted future accesses
    ///
    /// # Example
    /// ```ignore
    /// let history = Tensor::<B, 3>::random([1, 128, 96], Distribution::Uniform(0.0, 1.0), &device);
    /// let predictions = policy.predict(&history);
    /// // predictions shape: [1, 48, 128]
    /// ```
    pub fn predict(&self, history: &Tensor<B, 3>) -> Option<Tensor<B, 3>> {
        // Auto-detect Candle device from Burn device
        let candle_dev = burn_device_to_candle::<B>(&self.burn_device).ok()?;

        // Convert Burn tensor to Candle (moves to GPU if needed)
        let candle_input = burn_to_candle(history, &candle_dev).ok()?;

        // Run iTransformer forward pass (on GPU if configured)
        let candle_output = self.model.forward(&candle_input).ok()?;

        // Convert back to Burn tensor
        candle_to_burn(&candle_output, &self.burn_device).ok()
    }
}

// Manual clone for non-AutodiffBackend
impl<B: AutodiffBackend> Clone for OptimusPolicy<B> {
    fn clone(&self) -> Self {
        // Note: This creates a new model instance with same config
        // Model weights are NOT copied - this is a shallow clone
        // For training with weight preservation, use checkpoint save/load
        // Auto-detect Candle device from Burn device (same as new())
        let candle_device =
            select_device::<B>(&self.burn_device, None).unwrap_or(CandleDevice::Cpu);

        Self {
            model: OptimusModel::new(&self.config, &candle_device).expect("Failed to clone model"),
            config: self.config.clone(),
            burn_device: self.burn_device.clone(),
            step_count: self.step_count,
            warmup_complete: self.warmup_complete,
            action_dim: self.action_dim,
            _buffer: CpuRingBuffer::new(1000),
        }
    }
}

impl<B: AutodiffBackend> GpuTrainable<B, CpuRingBuffer> for OptimusPolicy<B> {
    fn buffer_mut(&mut self) -> &mut CpuRingBuffer {
        // Optimus doesn't use standard buffer - operates on time windows
        // This is a no-op for trait compliance
        &mut self._buffer
    }

    fn buffer(&self) -> &CpuRingBuffer {
        &self._buffer
    }

    fn train_step_gpu(&mut self, _batch: &TensorTransitionBatch<B>) -> f32 {
        // Training via Candle requires implementing backward pass through Candle
        // This is not yet implemented - iTransformer is used for inference only
        // For training, one would need to:
        // 1. Convert batch to Candle tensors
        // 2. Run forward pass through Candle model
        // 3. Compute loss in Candle
        // 4. Run backward pass in Candle
        // 5. Update weights via Candle optimizer
        // 6. Sync weights back if needed
        log::warn!("[Optimus] train_step_gpu not yet implemented - iTransformer is inference-only");
        0.0
    }

    fn train_step_gpu_native(
        &mut self,
        _steps_since_last_train: usize,
        _device: &B::Device,
    ) -> Option<f32> {
        // Training not implemented - iTransformer used for inference only
        log::warn!("[Optimus] train_step_gpu_native not yet implemented");
        self.step_count += 1;
        Some(0.0)
    }

    fn device(&self) -> &B::Device {
        &self.burn_device
    }

    fn state_dim(&self) -> usize {
        // State dimension is num_variates (cache line buckets)
        self.config.num_variates
    }

    fn buffer_len(&self) -> usize {
        // Return lookback_len as effective buffer size for time window
        self.config.lookback_len
    }

    fn warmup_batch_size(&self) -> usize {
        // Warmup batch size matches lookback window
        self.config.lookback_len
    }

    fn is_warmup_complete(&self) -> bool {
        self.warmup_complete
    }

    fn set_warmup_complete(&mut self, complete: bool) {
        self.warmup_complete = complete;
    }

    fn epsilon(&self) -> f32 {
        // iTransformer is deterministic, no exploration needed
        0.0
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn increment_step_count(&mut self) {
        self.step_count += 1;
    }

    fn batch_size(&self) -> usize {
        // Default batch size for training (when implemented)
        32
    }

    fn target_update_freq(&self) -> usize {
        // iTransformer doesn't have target network like DQN
        usize::MAX
    }

    fn learning_rate(&self) -> f32 {
        // Default learning rate (when training is implemented)
        1e-4
    }

    fn gamma(&self) -> f32 {
        // Discount factor - not used for pure forecasting
        0.99
    }

    fn decay_exploration(&mut self) {
        // No exploration to decay for deterministic iTransformer
    }

    fn update_target_network(&mut self) {
        // No target network for iTransformer
    }

    fn save_checkpoint(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Save Candle model weights via VarMap
        // This requires accessing the internal VarMap from OptimusModel
        // and saving it to disk
        log::warn!("[Optimus] save_checkpoint not yet implemented - Candle weights not saved");
        // For now, just create a metadata file
        let metadata = crate::checkpoint::CheckpointMetadata::new_with_dims(
            "OptimusPolicy".to_string(),
            self.step_count,
            self.config.num_variates,
            self.action_dim,
            self.config.d_model,
        );
        let meta_path = format!("{}.json", path);
        std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)?;
        Ok(())
    }

    fn load_checkpoint(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Load Candle model weights via VarMap
        // This requires accessing the internal VarMap from OptimusModel
        // and loading weights from disk
        log::warn!("[Optimus] load_checkpoint not yet implemented - Candle weights not loaded");
        let meta_path = format!("{}.json", path);
        if std::path::Path::new(&meta_path).exists() {
            let _metadata: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
        }
        Ok(())
    }
}

impl<B: AutodiffBackend> BatchedActionSelector<B> for OptimusPolicy<B> {
    fn select_actions_batched(
        &self,
        observations: &[Vec<f64>],
        _device: &B::Device,
        _action_dim: usize,
        _epsilon: f32,
    ) -> Vec<usize> {
        // observations: Vec of [num_variates] vectors
        // For iTransformer, we need to aggregate into time windows
        // This is a simplified implementation - proper time window aggregation needed

        let batch_size = observations.len();
        if batch_size == 0 {
            return vec![];
        }

        // For simplicity: just return first action for each
        // TODO: Implement proper time window aggregation and prediction
        // This would require:
        // 1. Accumulating observations into time windows
        // 2. Converting to [batch, num_variates, lookback_len] tensor
        // 3. Running predict()
        // 4. Selecting actions from predictions
        vec![0usize; batch_size]
    }
}

impl<B: AutodiffBackend> Checkpointable<B> for OptimusPolicy<B> {
    fn checkpoint_name(&self) -> &str {
        "optimus_policy"
    }

    fn checkpoint_metadata(&self) -> crate::checkpoint::CheckpointMetadata {
        crate::checkpoint::CheckpointMetadata::new_with_dims(
            "OptimusPolicy".to_string(),
            self.step_count,
            self.config.num_variates,
            self.action_dim,
            self.config.d_model,
        )
    }

    fn model(&self) -> &impl burn::module::Module<B> {
        // OptimusPolicy uses Candle internally, not Burn modules.
        // This method should never be called.
        // Use save_checkpoint()/load_checkpoint() for persistence.
        // Dummy type to satisfy trait bounds - panic ensures it's never used.
        #[derive(Debug, Clone)]
        struct Dummy<B: AutodiffBackend>(std::marker::PhantomData<B>);
        impl<B: AutodiffBackend> burn::module::Module<B> for Dummy<B> {
            type Record = ();
            fn collect_devices(&self, _: Vec<B::Device>) -> Vec<B::Device> {
                vec![]
            }
            fn fork(self, _: &B::Device) -> Self {
                self
            }
            fn to_device(self, _: &B::Device) -> Self {
                self
            }
            fn visit<V: burn::module::ModuleVisitor<B>>(&self, _: &mut V) {}
            fn map<M: burn::module::ModuleMapper<B>>(self, _: &mut M) -> Self {
                self
            }
            fn load_record(self, _: Self::Record) -> Self {
                self
            }
            fn into_record(self) -> Self::Record {}
        }
        // Create dummy to satisfy return type, then panic
        let _dummy: Dummy<B> = Dummy(std::marker::PhantomData);
        panic!(
            "OptimusPolicy::model() should never be called. \
             Use save_checkpoint()/load_checkpoint() for persistence."
        );
        #[allow(unreachable_code)]
        &_dummy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = burn::backend::Autodiff<NdArray>;

    #[test]
    fn test_optimus_policy_checkpoint_metadata() {
        let config = OptimusConfig::default();
        let device = Default::default();
        let mut policy = OptimusPolicy::<TestBackend>::new(config.clone(), device, 10, None);

        policy.increment_step_count();
        policy.increment_step_count();

        let metadata = policy.checkpoint_metadata();
        assert_eq!(metadata.policy_type, "OptimusPolicy");
        assert_eq!(metadata.epoch, 2);
        assert_eq!(metadata.state_dim, Some(config.num_variates));
        assert_eq!(metadata.action_dim, Some(10));
        assert_eq!(metadata.feature_dim, Some(config.d_model));
    }

    #[test]
    fn test_optimus_policy_trait_methods() {
        let config = OptimusConfig::default();
        let device = Default::default();
        let mut policy = OptimusPolicy::<TestBackend>::new(config.clone(), device, 10, None);

        // Test GpuTrainable methods
        assert_eq!(policy.batch_size(), 32);
        assert_eq!(policy.target_update_freq(), usize::MAX);
        assert!((policy.learning_rate() - 1e-4).abs() < f32::EPSILON);
        assert!((policy.gamma() - 0.99).abs() < f32::EPSILON);

        // Test no-op methods
        policy.decay_exploration();
        policy.update_target_network();

        // Test buffer access (dummy buffer)
        assert!(policy.buffer().is_empty());
    }

    #[test]
    fn test_batched_action_selector() {
        let config = OptimusConfig::default();
        let device = Default::default();
        let policy = OptimusPolicy::<TestBackend>::new(config, device, 10, None);

        let observations = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let actions = policy.select_actions_batched(&observations, &device, 10, 0.0);

        assert_eq!(actions.len(), 3);
        // Current implementation returns 0 for all
        assert_eq!(actions, vec![0, 0, 0]);
    }
}
