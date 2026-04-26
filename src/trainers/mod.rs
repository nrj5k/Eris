//! Trainer configurations and base classes for RL algorithms.
//!
//! This module provides common configuration structures and base classes
//! for various reinforcement learning trainers including DQN, Metis, and PPO.
//!
//! # Modules
//!
//! - `base`: Base configuration shared across all trainers

pub mod base;
pub use base::{TrainerConfig, TrainerConfigBase};
