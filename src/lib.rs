//! HeirGym Enhanced Models - Multi-tier storage optimization RL environment
//!
//! This crate provides a reinforcement learning environment for optimizing
//! multi-tier storage hierarchies using contextual bandits and Q-learning.

pub mod config;
pub mod env;
pub mod error;
pub mod features;
pub mod models;
pub mod tier;
pub mod trace;
pub mod training;

// Re-exports for convenience
pub use config::{Config, TierConfig};
pub use error::{EnvError, Result};
pub use features::{hotness_score, AccessRecord, AccessTracker, BlobFeatures, HotnessConfig};
pub use tier::{BufferEnv, Tier, TierSelector};
pub use trace::{BlobData, IoOp, TraceReader};
// Note: The following re-exports will be enabled once implemented
// pub use models::{QNetwork, ContextualBandit, CombinedModel};
// pub use env::IOBufferEnv;
// pub use training::{ReplayBuffer, CombinedAgent};
