//! Profiling utilities using Tracy
//!
//! Enable with: cargo build --features profiling
//! Run Tracy profiler and connect to localhost:8086

#[cfg(feature = "profiling")]
pub use tracy_client::{span, Client};

#[cfg(feature = "profiling")]
/// Initialize Tracy client
pub fn init_tracy() {
    Client::start();
}

#[cfg(not(feature = "profiling"))]
/// No-op when profiling disabled
pub fn init_tracy() {}

/// Macro to create a profiling span
#[macro_export]
macro_rules! profile_span {
    ($name:expr) => {
        #[cfg(feature = "profiling")]
        {
            let _span = $crate::profiling::span!($name);
            _span
        }
        #[cfg(not(feature = "profiling"))]
        {
            // No-op when profiling disabled
        }
    };
}

/// Profile a function call
#[macro_export]
macro_rules! profile_function {
    () => {
        profile_span!(function_name!())
    };
}
