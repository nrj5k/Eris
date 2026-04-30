//! Optimus iTransformer model for time series forecasting
//!
//! This module provides an iTransformer-based architecture for cache
//! workload prediction. Enabled via the `optimus` feature flag.
//!
//! # Architecture
//! - Inverted attention across variates (not time steps)
//! - Reversible instance normalization
//! - GEGLU feed-forward networks
//! - Multi-horizon prediction heads

#[cfg(feature = "optimus")]
pub use self::{
    bridge::{burn_device_to_candle, burn_to_candle, candle_to_burn, device_name, is_gpu_device},
    config::OptimusConfig,
    model::OptimusModel,
    policy::OptimusPolicy,
    utils::{
        format_inference_summary, generate_synthetic_history, history_to_tensor,
        select_action_from_predictions,
    },
};

// Deprecated exports for backward compatibility
#[cfg(feature = "optimus")]
#[allow(deprecated)]
pub use self::bridge::{parse_device_str, BridgeDevice};

#[cfg(feature = "optimus")]
#[allow(deprecated)]
pub use self::utils::resolve_device;

#[cfg(feature = "optimus")]
mod bridge;
#[cfg(feature = "optimus")]
mod config;
#[cfg(feature = "optimus")]
mod model;
#[cfg(feature = "optimus")]
mod policy;
#[cfg(feature = "optimus")]
pub mod utils;

// Stub when feature disabled
#[cfg(not(feature = "optimus"))]
pub mod stub {
    //! Stub module when optimus feature is disabled
}
