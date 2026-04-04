//! BanditConfig with builder pattern for contextual bandit networks
//!
//! This module provides a clean API for configuring contextual bandit networks
//! with comprehensive validation and documentation.

use crate::error::Result;
use crate::model::Activation;
use std::fmt;

/// Configuration for contextual bandit network with enhanced features
///
/// The bandit network takes a state representation and produces:
/// 1. Enhanced features for the Q-network (feature_dim dimensional)
/// 2. Importance score for tier selection (single value in [0, 1])
///
/// # Architecture
///
/// The network consists of:
/// - Feature extraction layers: Linear(input_dim → hidden_layers[0] → ... → hidden_layers[n-1])
/// - Feature output: Linear(hidden_layers[n-1] → feature_dim)
/// - Importance score: Linear(hidden_layers[n-1] → 1) with Sigmoid activation
///
/// # Example
///
/// ```rust,ignore
/// use eris::config::BanditConfig;
/// use eris::model::Activation;
/// use eris::training::mock_env::MockEnv;
///
/// // Get dimensions from environment dynamically
/// let env = MockEnv::new_with_dims(100, 50, 20);
/// let obs_dim = env.observation_space().dim();
///
/// let config = BanditConfig::builder()
///     .input_dim(obs_dim)               // Dynamic state dimension
///     .hidden_layers(vec![64, 128])     // Architecture
///     .feature_dim(20)                  // Output for DQN
///     .activation(Activation::Sigmoid)  // For importance score
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct BanditConfig {
    /// Input dimension (state dimension)
    /// For storage tier optimization: 5 tier sizes + 10 blob features = 15
    pub input_dim: usize,

    /// Hidden layer dimensions
    /// Each layer progressively learns more abstract features
    pub hidden_layers: Vec<usize>,

    /// Output feature dimension (input to DQN)
    /// Must match DQN's input_dim
    pub feature_dim: usize,

    /// Activation function for hidden layers
    /// Sigmoid is recommended for importance score to constrain to [0, 1]
    pub activation: Activation,

    /// Whether to include bias in linear layers
    pub bias: bool,
}

/// Builder for BanditConfig with compile-time validation
#[derive(Debug, Clone)]
pub struct BanditConfigBuilder {
    input_dim: Option<usize>,
    hidden_layers: Option<Vec<usize>>,
    feature_dim: Option<usize>,
    activation: Option<Activation>,
    bias: Option<bool>,
}

impl BanditConfig {
    /// Create a new builder for BanditConfig
    ///
    /// # Returns
    /// A builder instance with all fields unset
    pub fn builder() -> BanditConfigBuilder {
        BanditConfigBuilder {
            input_dim: None,
            hidden_layers: None,
            feature_dim: None,
            activation: None,
            bias: None,
        }
    }

    /// Initialize a contextual bandit network with this configuration
    ///
    /// # Arguments
    /// * `device` - Device to initialize the network on
    ///
    /// # Returns
    /// Initialized ContextualBandit with random weights
    pub fn init<B: burn::prelude::Backend>(
        &self,
        device: &B::Device,
    ) -> crate::models::ContextualBandit<B> {
        log::info!(
            "Initializing Bandit with new config: input={}, hidden={:?}, feature={}",
            self.input_dim,
            self.hidden_layers,
            self.feature_dim
        );

        // Use the old config for now, but with parameters from new config
        // The old config expects: state_dim, hidden_dim, feature_dim
        // Our new config allows arbitrary hidden layers, but old only supports
        // a single hidden_dim that creates a 2-layer network
        let hidden_dim = self.hidden_layers.get(0).copied().unwrap_or(64);

        let old_config = crate::models::ContextualBanditConfig::new(
            self.input_dim,
            hidden_dim,
            self.feature_dim,
        )
        .with_bias(self.bias);

        old_config.init(device)
    }
}

impl BanditConfigBuilder {
    /// Set the input dimension (state dimension)
    ///
    /// For storage tier optimization:
    /// - 5 tier sizes (normalized capacities)
    /// - 10 blob features (size, access count, recency, etc.)
    /// - Total: 15 dimensions
    pub fn input_dim(mut self, dim: usize) -> Self {
        self.input_dim = Some(dim);
        self
    }

    /// Set the hidden layer dimensions
    ///
    /// Each layer should typically increase or maintain dimension
    /// for feature expansion before final reduction
    ///
    /// Common patterns:
    /// - [64, 128] for moderate complexity
    /// - [32] for simple/compact models
    /// - [128, 256, 128] for complex feature extraction
    pub fn hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = Some(layers);
        self
    }

    /// Set the output feature dimension
    ///
    /// This defines the dimension of the enhanced feature vector
    /// that will be used as input to the DQN.
    ///
    /// Typical values:
    /// - 10-20 for compact representations
    /// - 32-64 for rich feature spaces
    pub fn feature_dim(mut self, dim: usize) -> Self {
        self.feature_dim = Some(dim);
        self
    }

    /// Set the activation function
    ///
    /// Recommendations:
    /// - Sigmoid for importance score outputs (constrained to [0, 1])
    /// - ReLU for general feature extraction
    /// - Tanh for symmetric activation ranges
    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = Some(activation);
        self
    }

    /// Set whether to include bias in linear layers
    ///
    /// Default: true (bias is generally helpful)
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = Some(bias);
        self
    }

    /// Build the BanditConfig with validation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `input_dim` is not set
    /// - `feature_dim` is not set
    /// - `hidden_layers` is empty
    ///
    /// # Returns
    /// Validated BanditConfig or error
    pub fn build(self) -> Result<BanditConfig> {
        let input_dim = self
            .input_dim
            .ok_or_else(|| crate::error::EnvError::ConfigError {
                message: "input_dim is required for BanditConfig".to_string(),
            })?;

        let hidden_layers =
            self.hidden_layers
                .ok_or_else(|| crate::error::EnvError::ConfigError {
                    message: "hidden_layers is required for BanditConfig".to_string(),
                })?;

        if hidden_layers.is_empty() {
            return Err(crate::error::EnvError::ConfigError {
                message: "hidden_layers must have at least one layer".to_string(),
            }
            .into());
        }

        let feature_dim = self
            .feature_dim
            .ok_or_else(|| crate::error::EnvError::ConfigError {
                message: "feature_dim is required for BanditConfig".to_string(),
            })?;

        Ok(BanditConfig {
            input_dim,
            hidden_layers,
            feature_dim,
            activation: self.activation.unwrap_or(Activation::ReLU),
            bias: self.bias.unwrap_or(true),
        })
    }
}

impl fmt::Display for BanditConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BanditConfig(input={}, hidden={:?}, feature={}, activation={})",
            self.input_dim, self.hidden_layers, self.feature_dim, self.activation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bandit_builder_complete() {
        let input_dim = 50;
        let config = BanditConfig::builder()
            .input_dim(input_dim)
            .hidden_layers(vec![64, 128])
            .feature_dim(20)
            .activation(Activation::Sigmoid)
            .bias(false)
            .build()
            .expect("Complete config should build");

        assert_eq!(config.input_dim, input_dim);
        assert_eq!(config.hidden_layers, vec![64, 128]);
        assert_eq!(config.feature_dim, 20);
        assert_eq!(config.activation, Activation::Sigmoid);
        assert!(!config.bias);
    }

    #[test]
    fn test_bandit_builder_defaults() {
        let input_dim = 75;
        let config = BanditConfig::builder()
            .input_dim(input_dim)
            .hidden_layers(vec![64])
            .feature_dim(20)
            .build()
            .expect("Config with defaults should build");

        assert!(matches!(config.activation, Activation::ReLU));
        assert!(config.bias);
    }

    #[test]
    fn test_bandit_builder_missing_input_dim() {
        let result = BanditConfig::builder()
            .hidden_layers(vec![64])
            .feature_dim(20)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_bandit_builder_missing_hidden_layers() {
        let result = BanditConfig::builder()
            .input_dim(100)
            .feature_dim(20)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_bandit_builder_empty_hidden_layers() {
        let result = BanditConfig::builder()
            .input_dim(80)
            .hidden_layers(vec![])
            .feature_dim(20)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_bandit_config_display() {
        let input_dim = 90;
        let config = BanditConfig::builder()
            .input_dim(input_dim)
            .hidden_layers(vec![64, 128])
            .feature_dim(20)
            .build()
            .unwrap();

        let display = format!("{}", config);
        assert!(display.contains(&format!("input={}", input_dim)));
        assert!(display.contains("feature=20"));
    }
}
