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
    bridge::{burn_to_candle, candle_to_burn, parse_device_str, BridgeDevice},
    config::OptimusConfig,
    model::OptimusModel,
    policy::OptimusPolicy,
    utils::{
        format_inference_summary, generate_synthetic_history, history_to_tensor, resolve_device,
        select_action_from_predictions,
    },
};

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
