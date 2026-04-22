//! Trainers module for GPU-native RL algorithms
//!
//! This module provides trainer implementations that use GpuRingBuffer
//! for efficient GPU-native training.

pub mod base;
pub mod dqn_trainer;
pub mod metis_trainer;
pub mod ppo_trainer;
pub use base::TrainerConfig;
pub use dqn_trainer::{DQNTrainer, DQNTrainerConfig, QNetwork};
pub use metis_trainer::{MetisTrainer, MetisTrainerConfig};
pub use ppo_trainer::{PpoTrainer, PpoTrainerConfig};
