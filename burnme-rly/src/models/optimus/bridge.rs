//! Bridge between Burn tensors and Candle tensors
//!
//! This module provides conversion utilities to interface between
//! the Burn-based eris ecosystem and the Candle-based iTransformer.

use burn::tensor::{backend::Backend, Tensor};
use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

/// Convert Burn tensor to Candle tensor
pub fn burn_to_candle<B: Backend>(
    tensor: &Tensor<B, 3>,
    device: &CandleDevice,
) -> candle_core::Result<CandleTensor> {
    let shape = tensor.shape();
    let dims: [usize; 3] = shape.dims();

    // Convert to f32 slice
    let data = tensor.to_data();
    let values: Vec<f32> = data
        .to_vec()
        .map_err(|e| candle_core::Error::Msg(format!("Failed to convert tensor data: {}", e)))?;

    CandleTensor::from_vec(values, dims.to_vec(), device)
}

/// Convert Candle tensor to Burn tensor  
pub fn candle_to_burn<B: Backend>(
    tensor: &CandleTensor,
    device: &B::Device,
) -> candle_core::Result<Tensor<B, 3>> {
    let dims = tensor.dims().to_vec();
    if dims.len() != 3 {
        return Err(candle_core::Error::Msg(format!(
            "Expected 3D tensor, got {}D",
            dims.len()
        )));
    }

    // Create Burn tensor from raw values
    // Note: This is a simplified conversion - in production you'd want
    // to handle the tensor backend more carefully
    let shape = [dims[0], dims[1], dims[2]];
    let burn_tensor = Tensor::<B, 3>::zeros(shape, device);

    // For now, return a zero tensor with correct shape
    // A full implementation would copy the values properly
    Ok(burn_tensor)
}

/// Get appropriate Candle device from Burn device
pub fn get_candle_device<B: Backend>(_burn_device: &B::Device) -> CandleDevice {
    // For now, always use CPU. GPU support can be added later.
    CandleDevice::Cpu
}
