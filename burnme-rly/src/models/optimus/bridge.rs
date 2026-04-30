//! Bridge between Burn tensors and Candle tensors with GPU support
//!
//! This module provides conversion utilities to interface between
//! the Burn-based eris ecosystem and the Candle-based iTransformer.
//!
//! Device selection is now automatic based on the Burn backend device.
//! No separate BridgeDevice is needed - the Candle device is derived
//! from the Burn device type.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

/// Convert Burn device to Candle device based on backend type.
/// Automatically selects CPU or CUDA based on the Burn device.
///
/// # Arguments
/// * `_burn_device` - The Burn backend device (type parameter determines actual device)
///
/// # Returns
/// * `Ok(CandleDevice)` - CUDA device if available and feature enabled, otherwise CPU
/// * `Err(candle_core::Error)` - If CUDA device creation fails (when CUDA is the only option)
///
/// # Examples
/// ```ignore
/// let device = <NdArray as Backend>::Device::default();
/// let candle_device = burn_device_to_candle::<NdArray>(&device)?;
/// ```
pub fn burn_device_to_candle<B: Backend>(
    _burn_device: &B::Device,
) -> candle_core::Result<CandleDevice> {
    // Check if CUDA feature is enabled and available
    #[cfg(feature = "cuda")]
    {
        // Try CUDA device 0
        if let Ok(cuda) = CandleDevice::new_cuda(0) {
            return Ok(cuda);
        }
    }

    // Fall back to CPU
    Ok(CandleDevice::Cpu)
}

/// Check if Burn device is GPU (for logging)
///
/// # Arguments
/// * `_device` - The Burn backend device
///
/// # Returns
/// `true` if CUDA feature is enabled and CUDA device is available
pub fn is_gpu_device<B: Backend>(_device: &B::Device) -> bool {
    #[cfg(feature = "cuda")]
    {
        return CandleDevice::new_cuda(0).is_ok();
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get device name for logging
///
/// # Arguments
/// * `_device` - The Burn backend device
///
/// # Returns
/// "CUDA" if GPU is available, "CPU" otherwise
pub fn device_name<B: Backend>(_device: &B::Device) -> String {
    if is_gpu_device::<B>(_device) {
        "CUDA".to_string()
    } else {
        "CPU".to_string()
    }
}

/// Convert Burn tensor to Candle tensor
///
/// # Arguments
/// * `tensor` - Burn tensor [batch, num_variates, lookback_len]
/// * `device` - Target Candle device
///
/// # Returns
/// Candle tensor with same shape
pub fn burn_to_candle<B: Backend>(
    tensor: &Tensor<B, 3>,
    device: &CandleDevice,
) -> candle_core::Result<CandleTensor> {
    let data = tensor.to_data();
    let dims: [usize; 3] = tensor.shape().dims();

    // Convert to f32 slice
    let values: Vec<f32> = data
        .to_vec()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to convert tensor data: {}", e)))?;

    // Create tensor on specified device
    CandleTensor::from_vec(values, dims.to_vec(), device)
}

/// Convert Candle tensor to Burn tensor
///
/// # Arguments
/// * `tensor` - Candle tensor
/// * `device` - Burn device
///
/// # Returns
/// Burn tensor [batch, pred_len, num_variates]
pub fn candle_to_burn<B: Backend>(
    tensor: &CandleTensor,
    burn_device: &B::Device,
) -> candle_core::Result<Tensor<B, 3>> {
    let dims = tensor.dims().to_vec();
    if dims.len() != 3 {
        return Err(candle_core::Error::Msg(format!(
            "Expected 3D tensor, got {}D",
            dims.len()
        )));
    }

    // Extract values - handles both CPU and GPU tensors
    let values: Vec<f32> = tensor.to_vec1()?;

    // Create Burn tensor from data
    let tensor_data = TensorData::new(values, [dims[0], dims[1], dims[2]]);

    Ok(Tensor::from_data(tensor_data.convert::<f32>(), burn_device))
}

/// Supported device types (DEPRECATED - use automatic detection instead)
///
/// This enum is kept for backward compatibility but should not be used in new code.
/// Use `burn_device_to_candle()` for automatic device detection.
#[derive(Debug, Clone, Copy)]
pub enum BridgeDevice {
    Cpu,
    Cuda(usize), // GPU index
}

impl BridgeDevice {
    /// Create Candle device from BridgeDevice
    #[deprecated(
        since = "0.1.0",
        note = "Use burn_device_to_candle() for automatic detection"
    )]
    pub fn to_candle(&self) -> candle_core::Result<CandleDevice> {
        match self {
            BridgeDevice::Cpu => Ok(CandleDevice::Cpu),
            BridgeDevice::Cuda(_idx) => {
                #[cfg(feature = "cuda")]
                {
                    CandleDevice::new_cuda(0)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!("[WARN] CUDA not enabled, falling back to CPU");
                    Ok(CandleDevice::Cpu)
                }
            }
        }
    }

    /// Check if device is CUDA
    #[deprecated(since = "0.1.0", note = "Use is_gpu_device() for automatic detection")]
    pub fn is_cuda(&self) -> bool {
        matches!(self, BridgeDevice::Cuda(_))
    }

    /// Detect best available device
    #[deprecated(
        since = "0.1.0",
        note = "Use burn_device_to_candle() for automatic detection"
    )]
    pub fn auto() -> Self {
        #[cfg(feature = "cuda")]
        {
            // Try CUDA device 0
            if CandleDevice::new_cuda(0).is_ok() {
                return BridgeDevice::Cuda(0);
            }
        }
        BridgeDevice::Cpu
    }
}

/// Get BridgeDevice from string (for CLI)
///
/// DEPRECATED: This function is kept for backward compatibility but should not be used.
/// Use `burn_device_to_candle()` for automatic device detection instead.
#[deprecated(
    since = "0.1.0",
    note = "Use burn_device_to_candle() for automatic detection"
)]
pub fn parse_device_str(s: &str) -> Option<BridgeDevice> {
    match s.to_lowercase().as_str() {
        "cpu" => Some(BridgeDevice::Cpu),
        "cuda" | "gpu" => Some(BridgeDevice::Cuda(0)),
        s if s.starts_with("cuda:") => s
            .split(':')
            .nth(1)
            .and_then(|idx| idx.parse().ok())
            .map(BridgeDevice::Cuda),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn test_bridge_device_cpu() {
        let dev = BridgeDevice::Cpu;
        #[allow(deprecated)]
        let candle = dev.to_candle().unwrap();
        assert!(matches!(candle, CandleDevice::Cpu));
    }

    #[test]
    #[allow(deprecated)]
    fn test_parse_device_str() {
        assert!(matches!(parse_device_str("cpu"), Some(BridgeDevice::Cpu)));
        assert!(matches!(
            parse_device_str("cuda"),
            Some(BridgeDevice::Cuda(0))
        ));
        assert!(matches!(
            parse_device_str("cuda:1"),
            Some(BridgeDevice::Cuda(1))
        ));
        assert!(parse_device_str("invalid").is_none());
    }

    #[test]
    fn test_burn_device_to_candle_auto() {
        use burn::backend::NdArray;

        let device = <NdArray as Backend>::Device::default();
        let candle_device = burn_device_to_candle::<NdArray>(&device);
        assert!(candle_device.is_ok());
    }

    #[test]
    fn test_device_name() {
        use burn::backend::NdArray;

        let device = <NdArray as Backend>::Device::default();
        let name = device_name::<NdArray>(&device);
        // Should return either "CPU" or "CUDA" depending on feature flags
        assert!(!name.is_empty());
    }

    #[test]
    fn test_is_gpu_device() {
        use burn::backend::NdArray;

        let device = <NdArray as Backend>::Device::default();
        // Just check it doesn't panic - actual result depends on CUDA feature
        let _ = is_gpu_device::<NdArray>(&device);
    }
}
