//! Combined configuration for bandit-DQN models
//!
//! This module combines BanditConfig and DQNConfig into a unified
//! configuration for end-to-end training.

use crate::config::{BanditConfig, DQNConfig};
use crate::error::Result;
use std::fmt;

/// Combined configuration for bandit and DQN networks
///
/// This configuration ensures that the bandit's output dimension
/// matches the DQN's input dimension for seamless integration.
///
/// # Architecture Flow
///
/// ```text
/// State (input_dim)
///     ↓
/// [Bandit Network]
///     ↓
/// Enhanced Features (feature_dim)
///     ↓
/// [DQN Network]
///     ↓
/// Q-Values (action_dim)
/// ```
///
/// # Validation
///
/// The combined config validates that:
/// - `bandit.feature_dim == dqn.input_dim`
///
/// # Example
///
/// ```rust,ignore
/// use eris::config::{BanditConfig, DQNConfig, CombinedBanditDQNConfig};
/// use eris::model::ErisDefaults;
/// use eris::training::mock_env::MockEnv;
///
/// // Option 1: Use defaults with dynamic dimensions (Tier 1)
/// let env = MockEnv::new_with_dims(100, 50, 20);
/// let state_dim = env.observation_space().dim();
/// let action_dim = env.action_space().n;
/// let config = ErisDefaults::storage_tier_model(state_dim, action_dim);
///
/// // Option 2: Build manually with dynamic dimensions (Tier 2)
/// let obs_dim = env.observation_space().dim();
/// let action_dim = env.action_space().n;
/// let feature_dim = 20;
///
/// let bandit = BanditConfig::builder()
///     .input_dim(obs_dim)
///     .hidden_layers(vec![64, 128])
///     .feature_dim(feature_dim)
///     .build()?;
///
/// let dqn = DQNConfig::builder()
///     .input_dim(feature_dim)  // Must match bandit.feature_dim
///     .hidden_layers(vec![128, 128])
///     .action_dim(action_dim)
///     .build()?;
///
/// let combined = CombinedBanditDQNConfig::builder()
///     .bandit(bandit)
///     .dqn(dqn)
///     .build()?;
/// ```
#[derive(Debug, Clone)]
pub struct CombinedBanditDQNConfig {
    /// Bandit network configuration
    pub bandit: BanditConfig,

    /// Q-network configuration
    pub dqn: DQNConfig,
}

/// Builder for CombinedBanditDQNConfig with cross-validation
#[derive(Debug, Clone)]
pub struct CombinedBanditDQNConfigBuilder {
    bandit: Option<BanditConfig>,
    dqn: Option<DQNConfig>,
}

impl CombinedBanditDQNConfig {
    /// Create a new builder for CombinedBanditDQNConfig
    pub fn builder() -> CombinedBanditDQNConfigBuilder {
        CombinedBanditDQNConfigBuilder {
            bandit: None,
            dqn: None,
        }
    }

    /// Initialize the combined model with both bandit and DQN networks
    ///
    /// # Arguments
    /// * `device` - Device to initialize the model on
    ///
    /// # Returns
    /// Initialized CombinedModel ready for training
    pub fn init<B: burn::prelude::Backend>(
        &self,
        device: &B::Device,
    ) -> crate::models::CombinedModel<B> {
        log::info!(
            "Initializing combined model with bandit({}→{:?}→{}) and DQN({}→{:?}→{})",
            self.bandit.input_dim,
            self.bandit.hidden_layers,
            self.bandit.feature_dim,
            self.dqn.input_dim,
            self.dqn.hidden_layers,
            self.dqn.action_dim
        );

        let bandit = self.bandit.init(device);
        let qnetwork = self.dqn.init(device);

        crate::models::CombinedModel { bandit, qnetwork }
    }
}

impl CombinedBanditDQNConfigBuilder {
    /// Set the bandit network configuration
    ///
    /// The bandit network extracts features from the state and
    /// computes importance scores for tier selection.
    pub fn bandit(mut self, config: BanditConfig) -> Self {
        self.bandit = Some(config);
        self
    }

    /// Set the DQN network configuration
    ///
    /// The DQN network takes features from the bandit and
    /// estimates Q-values for each action.
    ///
    /// **Important**: `dqn.input_dim` must equal `bandit.feature_dim`
    pub fn dqn(mut self, config: DQNConfig) -> Self {
        self.dqn = Some(config);
        self
    }

    /// Build the CombinedBanditDQNConfig with cross-validation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `bandit` configuration is not set
    /// - `dqn` configuration is not set
    /// - `bandit.feature_dim != dqn.input_dim` (dimension mismatch)
    ///
    /// # Returns
    /// Validated combined configuration or error
    pub fn build(self) -> Result<CombinedBanditDQNConfig> {
        let bandit = self
            .bandit
            .ok_or_else(|| crate::error::EnvError::ConfigError {
                message: "bandit configuration is required for CombinedBanditDQNConfig".to_string(),
            })?;

        let dqn = self
            .dqn
            .ok_or_else(|| crate::error::EnvError::ConfigError {
                message: "dqn configuration is required for CombinedBanditDQNConfig".to_string(),
            })?;

        // Validate dimension compatibility
        if bandit.feature_dim != dqn.input_dim {
            return Err(crate::error::EnvError::ConfigError {
                message: format!(
                    "Dimension mismatch: bandit.feature_dim ({}) != dqn.input_dim ({}). \
                     The DQN input dimension must match the bandit feature dimension.",
                    bandit.feature_dim, dqn.input_dim
                ),
            }
            .into());
        }

        Ok(CombinedBanditDQNConfig { bandit, dqn })
    }
}

impl fmt::Display for CombinedBanditDQNConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CombinedBanditDQNConfig(bandit=[{}→{:?}→{}], dqn=[{}→{:?}→{}])",
            self.bandit.input_dim,
            self.bandit.hidden_layers,
            self.bandit.feature_dim,
            self.dqn.input_dim,
            self.dqn.hidden_layers,
            self.dqn.action_dim
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Activation;

    #[test]
    fn test_combined_config_valid_dimensions() {
        let state_dim = 50;
        let feature_dim = 30;
        let action_dim = 25;

        let bandit = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(vec![64, 128])
            .feature_dim(feature_dim)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Valid bandit config");

        let dqn = DQNConfig::builder()
            .input_dim(feature_dim) // Matches bandit.feature_dim
            .hidden_layers(vec![128, 128])
            .action_dim(action_dim)
            .build()
            .expect("Valid DQN config");

        let combined = CombinedBanditDQNConfig::builder()
            .bandit(bandit)
            .dqn(dqn)
            .build()
            .expect("Valid combined config");

        assert_eq!(combined.bandit.feature_dim, combined.dqn.input_dim);
    }

    #[test]
    fn test_combined_config_dimension_mismatch() {
        let state_dim = 75;
        let feature_dim = 30;

        let bandit = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(vec![64])
            .feature_dim(feature_dim)
            .build()
            .unwrap();

        let dqn = DQNConfig::builder()
            .input_dim(40) // Does NOT match bandit.feature_dim
            .hidden_layers(vec![128])
            .action_dim(20)
            .build()
            .unwrap();

        let result = CombinedBanditDQNConfig::builder()
            .bandit(bandit)
            .dqn(dqn)
            .build();

        assert!(result.is_err(), "Should fail with dimension mismatch");

        if let Err(e) = result {
            let msg = format!("{}", e);
            assert!(
                msg.contains("Dimension mismatch"),
                "Error message should mention dimension mismatch"
            );
            assert!(
                msg.contains("30") && msg.contains("40"),
                "Error should show both dimensions"
            );
        }
    }

    #[test]
    fn test_combined_config_missing_bandit() {
        let feature_dim = 25;
        let dqn = DQNConfig::builder()
            .input_dim(feature_dim)
            .hidden_layers(vec![128])
            .action_dim(30)
            .build()
            .unwrap();

        let result = CombinedBanditDQNConfig::builder().dqn(dqn).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_combined_config_missing_dqn() {
        let state_dim = 85;
        let bandit = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(vec![64])
            .feature_dim(25)
            .build()
            .unwrap();

        let result = CombinedBanditDQNConfig::builder().bandit(bandit).build();

        assert!(result.is_err());
    }

    #[test]
    fn test_combined_config_display() {
        let state_dim = 95;
        let feature_dim = 35;
        let action_dim = 40;

        let bandit = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(vec![64, 128])
            .feature_dim(feature_dim)
            .build()
            .unwrap();

        let dqn = DQNConfig::builder()
            .input_dim(feature_dim)
            .hidden_layers(vec![128, 128])
            .action_dim(action_dim)
            .build()
            .unwrap();

        let combined = CombinedBanditDQNConfig::builder()
            .bandit(bandit)
            .dqn(dqn)
            .build()
            .unwrap();

        let display = format!("{}", combined);
        assert!(display.contains("bandit=["));
        assert!(display.contains("dqn=["));
        assert!(display.contains("→"));
    }
}
