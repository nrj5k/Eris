//! Configuration APIs for Eris RL models
//!
//! This module provides a clean, three-tier configuration API:
//!
//! # Tier 1: Defaults (Immediate Usability)
//!
//! Use `ErisDefaults::storage_tier_model()` for pre-configured models optimized for
//! storage tier optimization. No configuration needed - just initialize and train.
//!
//! ```rust,ignore
//! use eris::model::ErisDefaults;
//! use eris::training::mock_env::MockEnv;
//! use burn::backend::wgpu::Wgpu;
//!
//! let env = MockEnv::new_with_dims(100, 50, 20);
//! let state_dim = env.observation_space().dim();
//! let action_dim = env.action_space().n;
//!
//! let config = ErisDefaults::storage_tier_model(state_dim, action_dim);
//! let device = Wgpu::default();
//! let model = config.init::<Wgpu>(&device);
//! ```
//!
//! # Tier 2: Builder Pattern (Clear Customization)
//!
//! Use the builder pattern for explicit configuration with compile-time validation.
//! Ideal for research and experimentation with different architectures.
//!
//! ```rust,ignore
//! use eris::config::{BanditConfig, DQNConfig, CombinedBanditDQNConfig};
//! use eris::model::Activation;
//! use eris::training::mock_env::MockEnv;
//!
//! let env = MockEnv::new_with_dims(100, 50, 20);
//! let obs_dim = env.observation_space().dim();
//! let action_dim = env.action_space().n;
//! let feature_dim = 20;
//!
//! let bandit_config = BanditConfig::builder()
//!     .input_dim(obs_dim)
//!     .hidden_layers(vec![64, 128])
//!     .feature_dim(feature_dim)
//!     .activation(Activation::Sigmoid)
//!     .build()?;
//!
//! let dqn_config = DQNConfig::builder()
//!     .input_dim(feature_dim)  // Must match bandit.feature_dim
//!     .hidden_layers(vec![128, 128])
//!     .action_dim(action_dim)
//!     .dueling(true)
//!     .build()?;
//!
//! let combined_config = CombinedBanditDQNConfig::builder()
//!     .bandit(bandit_config)
//!     .dqn(dqn_config)
//!     .build()?;
//! ```
//!
//! # Tier 3: Model Trait (Extensibility)
//!
//! Implement the `Model` trait for custom architectures. See `src/model.rs` for
//! the trait definition and examples.
//!
//! # Module Structure
//!
//! - `bandit_config`: Configuration for contextual bandit networks
//! - `dqn_config`: Configuration for Q-networks with dueling architecture
//! - `combined_config`: Combined bandit-DQN configuration

mod bandit_config;
mod combined_config;
mod dqn_config;

pub use bandit_config::{BanditConfig, BanditConfigBuilder};
pub use combined_config::{CombinedBanditDQNConfig, CombinedBanditDQNConfigBuilder};
pub use dqn_config::{DQNConfig, DQNConfigBuilder};

// Re-export the original config module for backwards compatibility
// but update references internally
pub use crate::config_old::{Config, TierConfig};
