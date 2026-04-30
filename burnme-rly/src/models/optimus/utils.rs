//! Shared utilities for Optimus iTransformer
//!
//! Provides common functions used across binaries and library code
//! to avoid DRY violations.
//!
//! # Device Selection
//! Device selection is now automatic based on the Burn backend device.
//! The `resolve_device` function is deprecated and kept only for backward compatibility.

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

/// Parse device string with auto-detection fallback.
///
/// DEPRECATED: Device selection is now automatic based on the Burn backend.
/// This function is kept for backward compatibility but should not be used in new code.
///
/// # Examples
/// ```ignore
/// // OLD (deprecated):
/// let device = resolve_device("cuda:0");
///
/// // NEW (automatic):
/// let device = <NdArray as Backend>::Device::default();
/// // Candle device is auto-detected from Burn device
/// ```
#[deprecated(
    since = "0.1.0",
    note = "Device selection is now automatic. Use Burn backend device directly."
)]
pub fn resolve_device(device_str: &str) -> crate::models::optimus::bridge::BridgeDevice {
    use crate::models::optimus::bridge::BridgeDevice;

    #[allow(deprecated)]
    match device_str.to_lowercase().as_str() {
        "auto" => BridgeDevice::auto(),
        s => crate::models::optimus::bridge::parse_device_str(s).unwrap_or_else(|| {
            log::warn!("Unknown device '{}', using auto", s);
            BridgeDevice::auto()
        }),
    }
}

/// Convert history data to tensor format.
///
/// Input: `[num_variates][lookback_len]` - one Vec per variate
/// Output: `[1, num_variates, lookback_len]` - tensor for iTransformer
///
/// # Arguments
/// * `history` - History data as Vec of Vecs (one per variate)
/// * `device` - Burn backend for output tensor
///
/// # Returns
/// Tensor of shape [1, num_variates, lookback_len]
pub fn history_to_tensor<B: Backend>(history: &[Vec<f32>], device: &B::Device) -> Tensor<B, 3> {
    let num_variates = history.len();
    let lookback_len = history.first().map(|v| v.len()).unwrap_or(0);

    if num_variates == 0 || lookback_len == 0 {
        // Return empty tensor with correct shape
        return Tensor::zeros(Shape::new([1, num_variates, lookback_len]), device);
    }

    // Flatten: [num_variates * lookback_len]
    let flattened: Vec<f32> = history.iter().flatten().copied().collect();

    // Create tensor with shape [1, num_variates, lookback_len]
    let data = TensorData::new(flattened, Shape::new([num_variates * lookback_len]));

    let tensor = Tensor::<B, 1>::from_data(data.convert::<f32>(), device);
    tensor.reshape(Shape::new([1, num_variates, lookback_len]))
}

/// Format inference run summary for display.
///
/// Creates a nice formatted string showing total steps, successful predictions,
/// and success percentage.
///
/// # Examples
/// ```
/// let summary = format_inference_summary(1000, 950);
/// println!("{}", summary);  // Shows formatted summary
/// ```
pub fn format_inference_summary(total: usize, successful: usize) -> String {
    let pct = 100.0 * successful as f64 / total.max(1) as f64;
    format!(
        "{}\n[COMPLETE] Inference finished\n  Total steps: {}\n  Successful: {} ({:.1}%)\n{}",
        "=".repeat(50),
        total,
        successful,
        pct,
        "=".repeat(50)
    )
}

/// Generate synthetic cache access history for testing.
///
/// Creates random data simulating cache access patterns.
/// Useful for testing without real trace data.
///
/// # Arguments
/// * `num_variates` - Number of cache line buckets
/// * `lookback_len` - History window length
///
/// # Returns
/// Vec of Vecs: `[num_variates][lookback_len]` random values
pub fn generate_synthetic_history(num_variates: usize, lookback_len: usize) -> Vec<Vec<f32>> {
    use rand::prelude::*;
    use rand::rng;

    let mut rng = rng();

    (0..num_variates)
        .map(|_| {
            (0..lookback_len)
                .map(|_| rng.random_range(0.0..1.0f32))
                .collect()
        })
        .collect()
}

/// Generate action from predictions (simple argmax over last timestep).
///
/// Takes the last prediction step and selects the variate with highest activity.
///
/// # Arguments
/// * `predictions` - Tensor of shape [batch, pred_len, num_variates]
/// * `action_dim` - Number of possible actions
///
/// # Returns
/// Action index (0 to action_dim-1)
pub fn select_action_from_predictions<B: Backend>(
    predictions: &Tensor<B, 3>,
    action_dim: usize,
) -> usize {
    // Get last prediction step
    let data = predictions.to_data();
    let values: Vec<f32> = data.to_vec().unwrap_or_default();

    if values.is_empty() {
        return 0;
    }

    // Find index of maximum value
    let max_idx = values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    max_idx % action_dim
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    #[allow(deprecated)]
    fn test_resolve_device_auto() {
        let device = resolve_device("auto");
        // Should not panic
        #[allow(deprecated)]
        let _ = device.to_candle();
    }

    #[test]
    #[allow(deprecated)]
    fn test_resolve_device_cpu() {
        let device = resolve_device("cpu");
        #[allow(deprecated)]
        assert!(!device.is_cuda());
    }

    #[test]
    #[allow(deprecated)]
    fn test_resolve_device_unknown() {
        // Should fallback to auto
        let _ = resolve_device("unknown_device");
    }

    #[test]
    fn test_history_to_tensor() {
        let history = vec![vec![1.0f32, 2.0, 3.0], vec![4.0f32, 5.0, 6.0]];

        let device = <NdArray as Backend>::Device::default();
        let tensor = history_to_tensor::<NdArray>(&history, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims(), [1, 2, 3]);
    }

    #[test]
    fn test_history_to_tensor_empty() {
        let history: Vec<Vec<f32>> = vec![];

        let device = <NdArray as Backend>::Device::default();
        let tensor = history_to_tensor::<NdArray>(&history, &device);

        let shape = tensor.shape();
        assert_eq!(shape.dims(), [1, 0, 0]);
    }

    #[test]
    fn test_format_inference_summary() {
        let summary = format_inference_summary(100, 95);
        assert!(summary.contains("Total steps: 100"));
        assert!(summary.contains("Successful: 95"));
        assert!(summary.contains("95.0%"));
    }

    #[test]
    fn test_generate_synthetic_history() {
        let history = generate_synthetic_history(5, 10);
        assert_eq!(history.len(), 5);
        assert_eq!(history[0].len(), 10);
    }

    #[test]
    fn test_select_action_from_predictions() {
        let device = <NdArray as Backend>::Device::default();
        let predictions =
            Tensor::<NdArray, 3>::from_floats([[[1.0f32, 2.0, 3.0, 4.0, 5.0]]], &device);

        let action = select_action_from_predictions(&predictions, 5);
        assert!(action < 5);
    }
}
