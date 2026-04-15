//! Macro utilities for Eris
//!
//! This module provides macros for common patterns in the codebase,
//! particularly for backend dispatch and boilerplate reduction.

/// Dispatch a training function to the correct Burn backend based on Device enum.
///
/// This macro eliminates the repetitive match-on-Device pattern by providing
/// a single dispatch point that handles all backend variants with proper
/// feature gating.
///
/// # Usage
///
/// ```rust
/// dispatch_training!(device, |B, dev| run_catcher_training::<B>(args, dev));
/// ```
///
/// Where:
/// - `device`: A `Device` enum value
/// - `B`: The identifier to use for the backend type
/// - `dev`: The identifier for the device variable bound in each arm
/// - `$body`: The code to execute with the backend type and device
///
/// # Feature Gates
///
/// Each arm is gated behind its corresponding feature:
/// - `cpu` → `Device::Cpu` → `Autodiff<NdArray<f32>>`
/// - `cuda` → `Device::Cuda` → `Autodiff<Cuda<f32, i32>>`
/// - `wgpu` → `Device::Wgpu` → `Autodiff<Wgpu<f32, i32>>`
/// - `rocm` → `Device::Rocm` → `Autodiff<Rocm<f32, i32>>`
///
/// # Example
///
/// ```rust
/// # use eris::device::Device;
/// # fn run_training<B: burn::tensor::backend::AutodiffBackend>(device: &<B as burn::tensor::backend::Backend>::Device) {}
/// fn train(device: Device) {
///     dispatch_training!(device, |B, dev| {
///         run_training::<B>(dev)
///     });
/// }
/// ```
#[macro_export]
macro_rules! dispatch_training {
    ($device:expr, |$backend:ident, $dev:ident| $body:expr) => {
        match $device {
            #[cfg(feature = "cpu")]
            $crate::device::Device::Cpu($dev) => {
                use burn::backend::{Autodiff, NdArray};
                type $backend = Autodiff<NdArray<f32>>;
                $body
            }
            #[cfg(feature = "cuda")]
            $crate::device::Device::Cuda($dev) => {
                use burn::backend::{Autodiff, Cuda};
                type $backend = Autodiff<Cuda<f32, i32>>;
                $body
            }
            #[cfg(feature = "wgpu")]
            $crate::device::Device::Wgpu($dev) => {
                use burn::backend::{Autodiff, Wgpu};
                type $backend = Autodiff<Wgpu<f32, i32>>;
                $body
            }
            #[cfg(feature = "rocm")]
            $crate::device::Device::Rocm($dev) => {
                use burn::backend::{Autodiff, Rocm};
                type $backend = Autodiff<Rocm<f32, i32>>;
                $body
            }
        }
    };
}

// Re-export at module level for convenience
pub use dispatch_training;
