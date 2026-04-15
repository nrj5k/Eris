//! Device creation module for GPU/CPU backend selection

/// Runtime device selection enum
#[derive(Clone, Debug)]
pub enum Device {
    #[cfg(feature = "cpu")]
    Cpu(burn::backend::ndarray::NdArrayDevice),
    #[cfg(feature = "wgpu")]
    Wgpu(burn::backend::wgpu::WgpuDevice),
    #[cfg(feature = "cuda")]
    Cuda(burn::backend::cuda::CudaDevice),
    #[cfg(feature = "rocm")]
    Rocm(burn::backend::rocm::RocmDevice),
}

impl Device {
    /// Create device from string (for CLI argument parsing)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            #[cfg(feature = "cpu")]
            "cpu" => Some(Device::Cpu(burn::backend::ndarray::NdArrayDevice::default())),
            #[cfg(feature = "wgpu")]
            "wgpu" | "gpu" => Some(Device::Wgpu(burn::backend::wgpu::WgpuDevice::default())),
            #[cfg(feature = "cuda")]
            "cuda" => Some(Device::Cuda(burn::backend::cuda::CudaDevice::default())),
            #[cfg(feature = "rocm")]
            "rocm" => Some(Device::Rocm(burn::backend::rocm::RocmDevice::default())),
            _ => None,
        }
    }

    /// Get device name as string
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cpu")]
            Device::Cpu(_) => "cpu",
            #[cfg(feature = "wgpu")]
            Device::Wgpu(_) => "wgpu",
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => "cuda",
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => "rocm",
        }
    }
}

/// Get list of available backends at compile time
pub fn available_backends() -> Vec<&'static str> {
    let mut backends = Vec::new();
    #[cfg(feature = "cpu")]
    backends.push("cpu");
    #[cfg(feature = "wgpu")]
    backends.push("wgpu");
    #[cfg(feature = "cuda")]
    backends.push("cuda");
    #[cfg(feature = "rocm")]
    backends.push("rocm");
    backends
}

/// Check if a backend is available
pub fn is_backend_available(backend: &str) -> bool {
    match backend {
        #[cfg(feature = "cpu")]
        "cpu" | "ndarray" => true,

        #[cfg(feature = "wgpu")]
        "gpu" | "wgpu" => true,

        #[cfg(feature = "cuda")]
        "cuda" | "nvidia" => true,

        #[cfg(feature = "rocm")]
        "rocm" | "amd" => true,

        _ => false,
    }
}
