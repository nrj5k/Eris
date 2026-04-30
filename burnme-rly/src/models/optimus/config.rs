//! Optimus configuration

use serde::{Deserialize, Serialize};

/// Configuration for Optimus iTransformer model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimusConfig {
    /// Number of variates (cache line buckets)
    pub num_variates: usize,
    /// Lookback window length (history timesteps)
    pub lookback_len: usize,
    /// Prediction horizon (future timesteps)
    pub pred_len: usize,
    /// Embedding dimension
    pub d_model: usize,
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of encoder layers
    pub n_layers: usize,
    /// Feed-forward dimension
    pub d_ff: usize,
    /// Dropout rate
    pub dropout: f64,
    /// Use reversible instance normalization
    pub use_revin: bool,
}

impl Default for OptimusConfig {
    fn default() -> Self {
        Self {
            num_variates: 128,
            lookback_len: 96,
            pred_len: 48,
            d_model: 512,
            n_heads: 8,
            n_layers: 4,
            d_ff: 2048,
            dropout: 0.1,
            use_revin: true,
        }
    }
}

impl OptimusConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set num_variates
    pub fn with_num_variates(mut self, n: usize) -> Self {
        self.num_variates = n;
        self
    }

    /// Builder: set lookback_len
    pub fn with_lookback_len(mut self, n: usize) -> Self {
        self.lookback_len = n;
        self
    }

    /// Builder: set pred_len
    pub fn with_pred_len(mut self, n: usize) -> Self {
        self.pred_len = n;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.num_variates == 0 {
            return Err("num_variates must be > 0".to_string());
        }
        if self.lookback_len == 0 {
            return Err("lookback_len must be > 0".to_string());
        }
        if self.pred_len == 0 {
            return Err("pred_len must be > 0".to_string());
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err("d_model must be divisible by n_heads".to_string());
        }
        Ok(())
    }
}

impl std::fmt::Display for OptimusConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OptimusConfig {{\n\
             num_variates: {},\n\
             lookback_len: {},\n\
             pred_len: {},\n\
             d_model: {},\n\
             n_heads: {},\n\
             n_layers: {},\n\
             d_ff: {},\n\
             dropout: {:.2},\n\
             use_revin: {}\n\
             }}",
            self.num_variates,
            self.lookback_len,
            self.pred_len,
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.d_ff,
            self.dropout,
            self.use_revin
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_display() {
        let config = OptimusConfig::default();
        let output = format!("{}", config);

        assert!(output.contains("num_variates:"));
        assert!(output.contains("lookback_len:"));
        assert!(output.contains("d_model:"));
        assert!(output.contains("OptimusConfig"));
    }
}
