//! DQNConfig with builder pattern for Q-networks
//!
//! This module provides a clean API for configuring Q-networks
//! with optional dueling architecture support.

use crate::error::Result;
use std::fmt;

/// Configuration for Q-Network with dueling architecture
///
/// The Q-network takes enhanced features from the bandit and produces
/// Q-values for each possible action (tier × operation combinations).
///
/// # Dueling Architecture
///
/// When dueling is enabled (default), the network separates:
/// - V(s): Value of being in state s
/// - A(s, a): Advantage of taking action a in state s
///
/// Final Q-values: Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
///
/// This helps with value function approximation when action advantages
/// are similar across actions.
///
/// # Example
///
/// ```rust,ignore
/// use eris::config::DQNConfig;
/// use eris::training::mock_env::MockEnv;
///
/// // Get dimensions from environment dynamically
/// let env = MockEnv::new_with_dims(100, 50, 20);
/// let action_dim = env.action_space().n;
/// let feature_dim = 20; // From bandit network
///
/// let config = DQNConfig::builder()
///     .input_dim(feature_dim)             // Must match bandit.feature_dim
///     .hidden_layers(vec![128, 128])     // Shared hidden layers
///     .action_dim(action_dim)             // Dynamic action dimension
///     .dueling(true)                      // Enable dueling architecture
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct DQNConfig {
    /// Input dimension (must match bandit's feature_dim)
    pub input_dim: usize,

    /// Hidden layer dimensions for shared feature extraction
    /// Each layer refines the Q-value estimates
    pub hidden_layers: Vec<usize>,

    /// Action dimension (number of possible actions)
    /// For storage tier optimization: 5 tiers × 2 operations = 10
    pub action_dim: usize,

    /// Whether to use dueling architecture
    /// When enabled, separates value and advantage streams
    pub dueling: bool,

    /// Whether to include bias in linear layers
    pub bias: bool,
}

/// Builder for DQNConfig with validation
#[derive(Debug, Clone)]
pub struct DQNConfigBuilder {
    input_dim: Option<usize>,
    hidden_layers: Option<Vec<usize>>,
    action_dim: Option<usize>,
    dueling: Option<bool>,
    bias: Option<bool>,
}

impl DQNConfig {
    /// Create a new builder for DQNConfig
    ///
    /// # Returns
    /// A builder instance with all fields unset
    pub fn builder() -> DQNConfigBuilder {
        DQNConfigBuilder {
            input_dim: None,
            hidden_layers: None,
            action_dim: None,
            dueling: None,
            bias: None,
        }
    }

    /// Initialize a Q-network with this configuration
    ///
    /// # Arguments
    /// * `device` - Device to initialize the network on
    ///
    /// # Returns
    /// Initialized QNetwork with random weights
    pub fn init<B: burn::prelude::Backend>(
        &self,
        device: &B::Device,
    ) -> crate::models::QNetwork<B> {
        log::info!(
            "Initializing DQN with config: input={}, hidden={:?}, actions={}, dueling={}",
            self.input_dim,
            self.hidden_layers,
            self.action_dim,
            self.dueling
        );

        // Use old config for compatibility
        let hidden_dim = self.hidden_layers.get(0).copied().unwrap_or(128);

        let old_config =
            crate::models::QNetworkConfig::new(self.input_dim, hidden_dim, self.action_dim)
                .with_bias(self.bias);

        old_config.init(device)
    }
}

impl DQNConfigBuilder {
    /// Set the input dimension
    ///
    /// This MUST match the bandit network's feature_dim.
    /// Mismatched dimensions will cause runtime errors.
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.input_dim = Some(dim);
        self
    }

    /// Set the hidden layer dimensions
    ///
    /// Common patterns:
    /// - [128, 128] for moderate complexity (default)
    /// - [64] for simple models
    /// - [256, 256, 128] for complex action spaces
    pub fn hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = Some(layers);
        self
    }

    /// Set the action dimension
    ///
    /// For storage tier optimization:
    /// - 5 tiers (Memory, NVMe, SSD, HDD, Tapes)
    /// - 2 operations (Read, Write)
    /// - Total: 10 actions
    pub fn action_dim(mut self, dim: usize) -> Self {
        self.action_dim = Some(dim);
        self
    }

    /// Enable or disable dueling architecture
    ///
    /// Dueling architecture is recommended when:
    /// - Actions have similar advantages in many states
    /// - Value function estimation is important
    ///
    /// Default: true (enabled)
    pub fn dueling(mut self, enable: bool) -> Self {
        self.dueling = Some(enable);
        self
    }

    /// Set whether to include bias in linear layers
    ///
    /// Default: true
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Build the DQNConfig with validation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `input_dim` is not set
    /// - `action_dim` is not set
    /// - `hidden_layers` is empty
    ///
    /// # Returns
    /// Validated DQNConfig or error
    pub fn build(self) -> Result<DQNConfig> {
        let input_dim = self
            .input_dim
            .ok_or_else(|| crate::error::EnvError::ConfigError {
                message: "input_dim is required for DQNConfig".to_string(),
            })?;

        let hidden_layers =
            self.hidden_layers
                .ok_or_else(|| crate::error::EnvError::ConfigError {
                    message: "hidden_layers is required for DQNConfig".to_string(),
                })?;

        if hidden_layers.is_empty() {
            return Err(crate::error::EnvError::ConfigError {
                message: "hidden_layers must have at least one layer".to_string(),
            }
            .into());
        }

        let action_dim = self
            .action_dim
            .ok_or_else(|| crate::error::EnvError::ConfigError {
                message: "action_dim is required for DQNConfig".to_string(),
            })?;

        Ok(DQNConfig {
            input_dim,
            hidden_layers,
            action_dim,
            dueling: self.dueling.unwrap_or(true),
            bias: self.bias.unwrap_or(true),
        })
    }
}

impl fmt::Display for DQNConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DQNConfig(input={}, hidden={:?}, actions={}, dueling={})",
            self.input_dim, self.hidden_layers, self.action_dim, self.dueling
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dqn_builder_complete() {
        let input_dim = 30;
        let action_dim = 25;
        let config = DQNConfig::builder()
            .input_dim(input_dim)
            .hidden_layers(vec![128, 128])
            .action_dim(action_dim)
            .dueling(true)
            .bias(false)
            .build()
            .expect("Complete config should build");

        assert_eq!(config.input_dim, input_dim);
        assert_eq!(config.hidden_layers, vec![128, 128]);
        assert_eq!(config.action_dim, action_dim);
        assert!(config.dueling);
        assert!(!config.bias);
    }

    #[test]
    fn test_dqn_builder_defaults() {
        let action_dim = 32;
        let config = DQNConfig::builder()
            .input_dim(20)
            .hidden_layers(vec![128])
            .action_dim(action_dim)
            .build()
            .expect("Config with defaults should build");

        assert!(config.dueling, "Dueling should default to true");
        assert!(config.bias, "Bias should default to true");
    }

    #[test]
    fn test_dqn_builder_missing_input_dim() {
        let result = DQNConfig::builder()
            .hidden_layers(vec![128])
            .action_dim(20)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_dqn_builder_missing_action_dim() {
        let result = DQNConfig::builder()
            .input_dim(25)
            .hidden_layers(vec![128])
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_dqn_builder_empty_hidden_layers() {
        let result = DQNConfig::builder()
            .input_dim(35)
            .hidden_layers(vec![])
            .action_dim(20)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_dqn_config_display() {
        let input_dim = 40;
        let action_dim = 30;
        let config = DQNConfig::builder()
            .input_dim(input_dim)
            .hidden_layers(vec![128, 128])
            .action_dim(action_dim)
            .build()
            .unwrap();

        let display = format!("{}", config);
        assert!(display.contains(&format!("input={}", input_dim)));
        assert!(display.contains(&format!("actions={}", action_dim)));
    }
}
