//! Cache policy implementations

pub mod bandit_policy;
pub mod cacheus;
pub mod catcher;
pub mod checkpoint;
pub mod dqn_policy;
pub mod exploration;
pub mod metis;
pub mod policy;
pub mod td_loss;
pub mod tensor_utils;

pub use crate::training::checkpoint::{CheckpointMetadata, Checkpointable};
pub use bandit_policy::{BanditPolicy, BanditPolicyConfig};
pub use cacheus::CacheusPolicy;
pub use catcher::CatcherPolicy;
pub use dqn_policy::{DQNExplorerConfig, DQNPolicy};
pub use exploration::{
    EpsilonGreedy, ExplorationConfig, ExplorationStrategy, ThompsonSampling, UCBExplorer,
};
pub use metis::MetisPolicy;
pub use policy::*;

// Re-export PolicyType variants for convenience
pub use policy::PolicyType;

// Re-export GpuTrainable trait for policies
pub use crate::training::GpuTrainable;
