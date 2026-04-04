use crate::error::{EnvError, Result};
use serde::Deserialize;
use std::path::Path;

/// Configuration for a storage tier
#[derive(Debug, Clone, Deserialize)]
pub struct TierConfig {
    /// Tier name (e.g., "Memory", "NVMe")
    pub name: String,
    /// Unique tier identifier
    pub tier_id: u32,
    /// Storage capacity in bytes
    pub capacity: f64,
    /// Access latency in milliseconds
    pub access_latency: f32,
    /// Human-readable description
    #[serde(default)]
    pub description: String,
}

/// Main configuration structure
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// List of tier configurations
    pub tier: Vec<TierConfig>,
}

impl Config {
    /// Load configuration from a TOML file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| EnvError::ConfigError {
            message: format!("Failed to read config file: {}", e),
        })?;

        toml::from_str(&content).map_err(|e| EnvError::ConfigError {
            message: format!("Failed to parse TOML: {}", e),
        })
    }

    /// Create default 5-tier configuration
    pub fn default_tiers() -> Self {
        Self {
            tier: vec![
                TierConfig {
                    name: "Memory".into(),
                    tier_id: 0,
                    capacity: 800_000.0,
                    access_latency: 0.01,
                    description: "Fastest tier - RAM".into(),
                },
                TierConfig {
                    name: "NVMe".into(),
                    tier_id: 1,
                    capacity: 2_000_000.0,
                    access_latency: 1.0,
                    description: "NVMe SSD tier".into(),
                },
                TierConfig {
                    name: "SSD".into(),
                    tier_id: 2,
                    capacity: 4_000_000.0,
                    access_latency: 10.0,
                    description: "Standard SSD tier".into(),
                },
                TierConfig {
                    name: "HDD".into(),
                    tier_id: 3,
                    capacity: 20_000_000.0,
                    access_latency: 10_000.0,
                    description: "Hard disk drive tier".into(),
                },
                TierConfig {
                    name: "Tapes".into(),
                    tier_id: 4,
                    capacity: 999_999_999_999.0,
                    access_latency: 1_000_000.0,
                    description: "Cold storage - tape archive".into(),
                },
            ],
        }
    }
}
