//! Bridge between Burn tensors and Candle tensors with GPU support
//!
//! This module provides conversion utilities to interface between
//! the Burn-based eris ecosystem and the Candle-based iTransformer.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

/// Supported device types
#[derive(Debug, Clone, Copy)]
pub enum BridgeDevice {
    Cpu,
    Cuda(usize), // GPU index
}

impl BridgeDevice {
    /// Create Candle device from BridgeDevice
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

    /// Detect best available device
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

/// Get BridgeDevice from string (for CLI)
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
    fn test_bridge_device_cpu() {
        let dev = BridgeDevice::Cpu;
        let candle = dev.to_candle().unwrap();
        assert!(matches!(candle, CandleDevice::Cpu));
    }

    #[test]
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
}
