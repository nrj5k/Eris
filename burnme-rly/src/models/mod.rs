//! Neural network models for reinforcement learning.

pub mod combined;
pub mod composable;
pub mod metis_v2;
pub mod ppo_model;

#[cfg(feature = "optimus")]
pub mod optimus;

pub use combined::{CombinedModel, CombinedModelConfig};
pub use composable::{ComposableModel, ComposeConfig, ParallelCompose, SequentialCompose};
pub use metis_v2::{MetisV2Config, MetisV2Policy};
pub use ppo_model::{PpoModel, PpoModelConfig};
