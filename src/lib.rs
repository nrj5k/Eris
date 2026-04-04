//! Eris Reinforcement Learning Library - Multi-tier Storage Optimization
//!
//! This crate provides a reinforcement learning environment for optimizing
//! multi-tier storage hierarchies using contextual bandits and Q-learning.
//!
//! # Overview
//!
//! Eris implements a flexible RL framework with:
//!
//! - **Dynamic Spaces**: `BoxSpace` for continuous observations and `DiscreteSpace` for actions
//! - **Environment Trait**: Generic interface for any RL environment
//! - **Training Module**: Complete training loop with DQN agents and experience replay
//!
//! # Architecture
//!
//! ## Spaces
//! - [`Space`] trait defines common interface for spaces
//! - [`BoxSpace`] represents bounded continuous observation spaces
//! - [`DiscreteSpace`] represents discrete action sets
//!
//! ## Environment
//! - [`Environment`] trait defines RL environment interface
//! - [`StepResult`] contains step outcome (observation, reward, done, info)
//! - Dynamic dimensions replace hardcoded constants
//!
//! ## Training
//! - [`train_agent`] function executes training loop
//! - [`TrainingResult`] contains training metrics
//! - [`CombinedAgent`] combines model and replay buffer
//!
//! # Example
//!
//! ```
//! use eris::training::MockEnv;
//! use eris::env::Environment;
//! use eris::space::Space;
//!
//! // Create environment with dynamic dimensions
//! let mut env = MockEnv::new_with_dims(100, 50, 20);
//!
//! // Get dimensions dynamically
//! let state_dim = env.observation_space().dim();
//! let action_dim = env.action_space().n;
//!
//! // Reset and step through environment
//! let observation = env.reset();
//! let result = env.step(0);
//! println!("Observation dim: {}", observation.len());
//! println!("Reward: {}", result.reward);
//! ```
//!
//! # Migration from v1.x
//!
//! The following breaking changes were introduced:
//!
//! - `OBSERVATION_DIM` constant removed - use dynamic dimensions via trait methods
//! - Environment trait now supports generic observation/action types
//! - Training module moved to `eris::training` with explicit parameters
//!
//! ## Old Code (v1.x)
//! ```rust,ignore
//! let env = MyEnv::new();
//! let state = env.reset(); // Fixed 15-dimensional
//! // OBSERVATION_DIM was a constant
//! ```
//!
//! ## New Code (v2.x)
//! ```rust,ignore
//! let mut env = MyEnv::new();
//! let state = env.reset();
//! let state_dim = env.observation_space().dim(); // Dynamic dimension
//! ```
//!
//! # Public API Exports
//!
//! ## Space Types
//! - [`Space`] - Trait for observation/action spaces
//! - [`BoxSpace`] - Bounded continuous space
//! - [`DiscreteSpace`] - Discrete action space
//!
//! ## Environment
//! - [`Environment`] - Main RL environment trait
//! - [`StepResult`] - Step outcome structure
//! - [`Info`] - Additional environment metrics
//!
//! ## Training
//! - [`train_agent`] - Main training function
//! - [`TrainingResult`] - Training metrics container
//! - [`CombinedAgent`] - Agent with model and replay buffer
//! - [`CheckpointMetadata`] - Training checkpoint info
//! - [`ReplayBuffer`] - Experience replay storage
//! - [`Transition`] - Single experience tuple
//! - [`TransitionBatch`] - Batch of experiences
//! - [`TrainingConfig`] - Training configuration
//!
//! ## Model
//! - [`Model`] - Trait for neural network models
//! - [`ErisDefaults`] - Default model configurations
//! - [`Activation`] - Neural network activation functions

// Public modules
pub mod config;
pub mod config_old;
pub mod env;
pub mod error;
pub mod features;
pub mod model;
pub mod models;
pub mod space;
pub mod tier;
pub mod trace;
pub mod training;

// Re-exports for convenience
// Config
pub use config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};
pub use config_old::{Config, TierConfig};

// Error handling
pub use error::{EnvError, Result};

// Features
pub use features::{hotness_score, AccessRecord, AccessTracker, BlobFeatures, HotnessConfig};

// Model
pub use model::{Activation, ErisDefaults, Model};
pub use models::{
    decode_action, encode_action, CombinedModel, CombinedModelConfig, ContextualBandit,
    ContextualBanditConfig, QNetwork, QNetworkConfig,
};

// Space types
pub use space::{BoxSpace, DiscreteSpace, Space};

// Environment trait
pub use env::{Environment, Info, StepResult};

// Tier
pub use tier::{BufferEnv, Tier, TierSelector};

// Trace
pub use trace::{BlobData, IoOp, TraceReader};

// Training
pub use training::{
    train_agent, CheckpointMetadata, CombinedAgent, MockEnv, ReplayBuffer, TrainingConfig,
    TrainingResult, Transition, TransitionBatch,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reexports_available() {
        // Test that all types are accessible from crate root
        // Space types
        let _box_space: BoxSpace = BoxSpace::uniform(3, -1.0, 1.0);
        let _discrete_space: DiscreteSpace = DiscreteSpace::new(5);

        // Verify Space trait is accessible
        fn check_space<S: Space>(_space: &S) {}
        check_space(&_box_space);
        check_space(&_discrete_space);

        // Verify Environment trait types are accessible
        let _step_result: Option<StepResult> = None;
        let _info: Option<Info> = None;
    }
}
