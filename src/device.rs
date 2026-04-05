//! Device creation module for GPU/CPU backend selection

/// Create device based on backend string
pub fn create_device(backend: &str) -> Box<dyn std::any::Any> {
    match backend {
        #[cfg(feature = "cpu-only")]
        "cpu" | "ndarray" => {
            println!("Creating NdArray CPU device...");
            use burn::backend::NdArray;
            let device = burn::backend::ndarray::NdArrayDevice::Cpu;
            Box::new(device)
        }

        #[cfg(feature = "wgpu-only")]
        "gpu" | "wgpu" => {
            println!("Creating Wgpu GPU device...");
            use burn::backend::Wgpu;
            let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);
            Box::new(device)
        }

        #[cfg(feature = "cuda-only")]
        "cuda" | "nvidia" => {
            println!("Creating CUDA device...");
            use burn::backend::Cuda;
            let device = burn::backend::cuda::CudaDevice::new(0);
            Box::new(device)
        }

        #[cfg(feature = "rocm-only")]
        "rocm" | "amd" => {
            println!("Creating ROCm device...");
            use burn::backend::Rocm;
            let device = burn::backend::rocm::RocmDevice::new(0);
            Box::new(device)
        }

        _ => {
            #[cfg(feature = "cpu-only")]
            let available = "cpu";
            #[cfg(feature = "wgpu-only")]
            let available = "wgpu";
            #[cfg(feature = "cuda-only")]
            let available = "cuda";
            #[cfg(feature = "rocm-only")]
            let available = "rocm";
            #[cfg(not(any(
                feature = "cpu-only",
                feature = "wgpu-only",
                feature = "cuda-only",
                feature = "rocm-only"
            )))]
            let available = "none";

            eprintln!(
                "Backend '{}' not compiled. Available: {}",
                backend, available
            );
            std::process::exit(1);
        }
    }
}

/// Check if a backend is available
pub fn is_backend_available(backend: &str) -> bool {
    match backend {
        #[cfg(feature = "cpu-only")]
        "cpu" | "ndarray" => true,

        #[cfg(feature = "wgpu-only")]
        "gpu" | "wgpu" => true,

        #[cfg(feature = "cuda-only")]
        "cuda" | "nvidia" => true,

        #[cfg(feature = "rocm-only")]
        "rocm" | "amd" => true,

        _ => false,
    }
}
