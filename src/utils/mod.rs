//! Utility modules for Eris

pub mod backend_diagnostics;
pub mod macros;
pub mod timing;

pub use backend_diagnostics::{
    detect_backend, is_cpu_backend, is_gpu_backend, log_backend_info, BackendKind,
};
pub use timing::{log_first_call, log_step_timing, OneTimeDiag};
