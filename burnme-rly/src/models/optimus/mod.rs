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
    bridge::{burn_to_candle, candle_to_burn},
    config::OptimusConfig,
    model::OptimusModel,
    policy::OptimusPolicy,
};

#[cfg(feature = "optimus")]
mod bridge;
#[cfg(feature = "optimus")]
mod config;
#[cfg(feature = "optimus")]
mod model;
#[cfg(feature = "optimus")]
mod policy;

// Stub when feature disabled
#[cfg(not(feature = "optimus"))]
pub mod stub {
    //! Stub module when optimus feature is disabled
}
