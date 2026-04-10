//! Unified interface for all cache policies (Metis, CACHEUS, Catcher)

use std::error::Error;
use std::path::Path;

/// State representation (unified for all policies)
#[derive(Clone, Debug, PartialEq)]
pub enum State {
    /// Discrete features (for Metis, CACHEUS)
    Features(Vec<f32>),
    /// Raw observation (for Catcher - address history)
    Raw(Vec<f64>),
    /// Empty state
    Empty,
}

/// Action types (discrete for Metis/CACHEUS, continuous for Catcher)
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Discrete action (arm selection)
    Discrete(usize),
    /// Continuous action (cache importance score)
    Continuous(Vec<f32>),
}

/// Policy type enumeration
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PolicyType {
    Metis,
    Cacheus,
    Catcher,
    Bandit,
    Dqn,
}

/// Transition for policy updates
#[derive(Clone, Debug)]
pub struct Transition {
    pub state: State,
    pub action: Action,
    pub reward: f32,
    pub next_state: State,
    pub done: bool,
}

/// Base trait for all cache policies
pub trait CachePolicy {
    /// Select action given current state
    fn select_action(&self, state: &State) -> Action;

    /// Update policy with transition, return loss/regret
    fn update(&mut self, transition: &Transition) -> f32;

    /// Save policy to path
    fn save(&self, path: &Path) -> Result<(), Box<dyn Error>>;

    /// Load policy from path
    fn load(&mut self, path: &Path) -> Result<(), Box<dyn Error>>;

    /// Get policy type
    fn policy_type(&self) -> PolicyType;

    /// Get number of actions (for discrete policies)
    fn action_dim(&self) -> usize;
}

/// Marker trait for online learning policies (no replay buffer)
pub trait OnlinePolicy: CachePolicy {
    /// Learning rate for online updates
    fn learning_rate(&self) -> f32;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f32);
}

/// Trait for replay-based policies
pub trait ReplayPolicy: CachePolicy {
    /// Train on batch from replay buffer
    fn train_step(&mut self, batch: &[Transition]) -> f32;

    /// Get batch size
    fn batch_size(&self) -> usize;

    /// Update target network (if applicable)
    fn update_target(&mut self);
}
