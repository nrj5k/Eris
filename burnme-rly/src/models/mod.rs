//! Neural network models for reinforcement learning.

pub mod combined;
pub mod composable;

pub use combined::{CombinedModel, CombinedModelConfig};
pub use composable::{ComposableModel, ComposeConfig, ParallelCompose, SequentialCompose};
