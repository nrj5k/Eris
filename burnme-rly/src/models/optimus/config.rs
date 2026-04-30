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
