//! Backend detection and diagnostics utilities

use burn::tensor::backend::Backend;
use tracing;

/// Represents the detected backend kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Wgpu,
    Rocm,
    Unknown,
}

/// Detect the backend type from the type name
pub fn detect_backend<B: Backend>() -> BackendKind {
    let type_name = std::any::type_name::<B>();

    if type_name.contains("NdArray") {
        BackendKind::Cpu
    } else if type_name.contains("Cuda") {
        BackendKind::Cuda
    } else if type_name.contains("Wgpu") {
        BackendKind::Wgpu
    } else if type_name.contains("Rocm") {
        BackendKind::Rocm
    } else {
        BackendKind::Unknown
    }
}

/// Log backend information at appropriate levels
///
/// Call this once to log backend diagnostics. The backend kind is detected
/// automatically from the type parameter.
pub fn log_backend_info<B: Backend>(context: &str, device: &B::Device) {
    let backend_name = std::any::type_name::<B>();
    let kind = detect_backend::<B>();

    tracing::debug!("{} DIAGNOSTIC:", context);
    tracing::debug!("   Backend type: {}", backend_name);
    tracing::debug!("   Device: {:?}", device);

    match kind {
        BackendKind::Cpu => {
            tracing::warn!("Backend is NdArray (CPU) - training will be slow");
        }
        BackendKind::Cuda => {
            tracing::debug!("  Backend is CUDA. Training should use GPU.");
        }
        BackendKind::Wgpu => {
            tracing::warn!("  Backend is WGPU (WebGPU).");
        }
        BackendKind::Rocm => {
            tracing::debug!("  Backend is ROCm. Training should use GPU.");
        }
        BackendKind::Unknown => {
            tracing::warn!("  Unknown backend type: {}", backend_name);
        }
    }
}

/// Check if the current backend is a GPU backend
pub fn is_gpu_backend<B: Backend>() -> bool {
    matches!(
        detect_backend::<B>(),
        BackendKind::Cuda | BackendKind::Rocm | BackendKind::Wgpu
    )
}

/// Check if the current backend is CPU
pub fn is_cpu_backend<B: Backend>() -> bool {
    detect_backend::<B>() == BackendKind::Cpu
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_detect_backend_cpu() {
        type B = NdArray<f32>;
        let kind = detect_backend::<B>();
        assert_eq!(kind, BackendKind::Cpu);
        assert!(is_cpu_backend::<B>());
        assert!(!is_gpu_backend::<B>());
    }
}
