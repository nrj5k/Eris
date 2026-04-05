//! Device creation module for GPU/CPU backend selection

/// Create device based on backend string
pub fn create_device(backend: &str) -> Box<dyn std::any::Any> {
    match backend {
        #[cfg(feature = "cpu")]
        "cpu" | "ndarray" => {
            println!("Creating NdArray CPU device...");
            use burn::backend::NdArray;
            let device = burn::backend::ndarray::NdArrayDevice::Cpu;
            Box::new(device)
        }

        #[cfg(feature = "gpu")]
        "gpu" | "wgpu" => {
            println!("Creating Wgpu GPU device...");
            use burn::backend::Wgpu;
            let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);
            Box::new(device)
        }

        #[cfg(feature = "nvidia")]
        "cuda" | "nvidia" => {
            println!("Creating CUDA device...");
            use burn::backend::Cuda;
            let device = burn::backend::cuda::CudaDevice::new(0);
            Box::new(device)
        }

        #[cfg(feature = "amd")]
        "rocm" | "amd" => {
            println!("Creating ROCm device...");
            use burn::backend::Rocm;
            let device = burn::backend::rocm::RocmDevice::new(0);
            Box::new(device)
        }

        _ => {
            eprintln!("Backend '{}' not compiled. Available: cpu", backend);
            std::process::exit(1);
        }
    }
}

/// Check if a backend is available
pub fn is_backend_available(backend: &str) -> bool {
    match backend {
        #[cfg(feature = "cpu")]
        "cpu" | "ndarray" => true,

        #[cfg(feature = "gpu")]
        "gpu" | "wgpu" => true,

        #[cfg(feature = "nvidia")]
        "cuda" | "nvidia" => true,

        #[cfg(feature = "amd")]
        "rocm" | "amd" => true,

        _ => false,
    }
}
