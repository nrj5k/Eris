//! Training configuration with full TOML support

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{EnvError, Result};

/// Model architecture types
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum ModelArchitecture {
    #[serde(rename = "dueling_dqn")]
    DuelingDQN,
    #[serde(rename = "bandit_dqn")]
    BanditDQN,
    #[serde(rename = "simple_dqn")]
    SimpleDQN,
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        Self::DuelingDQN
    }
}

impl std::fmt::Display for ModelArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelArchitecture::DuelingDQN => write!(f, "dueling_dqn"),
            ModelArchitecture::BanditDQN => write!(f, "bandit_dqn"),
            ModelArchitecture::SimpleDQN => write!(f, "simple_dqn"),
        }
    }
}

/// Backend types for training
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum BackendType {
    #[serde(rename = "cpu")]
    Cpu,
    #[serde(rename = "gpu")]
    Gpu,
    #[serde(rename = "torch")]
    Torch,
    #[serde(rename = "cuda")]
    Cuda,
    #[serde(rename = "rocm")]
    Rocm,
}

impl Default for BackendType {
    fn default() -> Self {
        Self::Cpu
    }
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Cpu => write!(f, "cpu"),
            BackendType::Gpu => write!(f, "gpu"),
            BackendType::Torch => write!(f, "torch"),
            BackendType::Cuda => write!(f, "cuda"),
            BackendType::Rocm => write!(f, "rocm"),
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    /// Model architecture type
    pub architecture: ModelArchitecture,
    /// Hidden layers for bandit network
    #[serde(default = "default_bandit_hidden")]
    pub bandit_hidden: Vec<usize>,
    /// Hidden layers for DQN network
    #[serde(default = "default_dqn_hidden")]
    pub dqn_hidden: Vec<usize>,
    /// Feature dimension (output of bandit, input to DQN)
    #[serde(default = "default_feature_dim")]
    pub feature_dim: usize,
}

fn default_bandit_hidden() -> Vec<usize> {
    vec![64, 128]
}

fn default_dqn_hidden() -> Vec<usize> {
    vec![128, 128]
}

fn default_feature_dim() -> usize {
    20
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architecture: ModelArchitecture::default(),
            bandit_hidden: default_bandit_hidden(),
            dqn_hidden: default_dqn_hidden(),
            feature_dim: default_feature_dim(),
        }
    }
}

/// Training hyperparameters
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingParams {
    /// Number of training episodes
    #[serde(default = "default_episodes")]
    pub episodes: usize,
    /// Maximum steps per episode
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// Batch size for training
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Learning rate
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f64,
    /// Discount factor (gamma)
    #[serde(default = "default_gamma")]
    pub gamma: f32,
    /// Starting exploration rate
    #[serde(default = "default_epsilon_start")]
    pub epsilon_start: f32,
    /// Final exploration rate
    #[serde(default = "default_epsilon_end")]
    pub epsilon_end: f32,
    /// Epsilon decay rate
    #[serde(default = "default_epsilon_decay")]
    pub epsilon_decay: f32,
    /// Target network update frequency
    #[serde(default = "default_target_update_freq")]
    pub target_update_freq: usize,
    /// Replay buffer size
    #[serde(default = "default_replay_buffer_size")]
    pub replay_buffer_size: usize,
    /// DataLoader worker threads (0 = single-threaded)
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,
}

fn default_episodes() -> usize {
    100
}
fn default_max_steps() -> usize {
    1000
}
fn default_batch_size() -> usize {
    2048 // Optimized for GPU utilization (multiple of 32 for warp alignment)
}
fn default_learning_rate() -> f64 {
    0.001
}
fn default_gamma() -> f32 {
    0.99
}
fn default_epsilon_start() -> f32 {
    1.0
}
fn default_epsilon_end() -> f32 {
    0.01
}
fn default_epsilon_decay() -> f32 {
    0.995
}
fn default_target_update_freq() -> usize {
    10
}
fn default_replay_buffer_size() -> usize {
    10000
}
fn default_num_workers() -> usize {
    2 // Phase 04: optimized for VecEnv (16 envs) with GPU training
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            episodes: default_episodes(),
            max_steps: default_max_steps(),
            batch_size: default_batch_size(),
            learning_rate: default_learning_rate(),
            gamma: default_gamma(),
            epsilon_start: default_epsilon_start(),
            epsilon_end: default_epsilon_end(),
            epsilon_decay: default_epsilon_decay(),
            target_update_freq: default_target_update_freq(),
            replay_buffer_size: default_replay_buffer_size(),
            num_workers: default_num_workers(),
        }
    }
}

/// Backend configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BackendConfig {
    /// Backend type
    #[serde(default)]
    pub backend_type: BackendType,
    /// Device ID for GPU/accelerator backends
    #[serde(default)]
    pub device_id: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::default(),
            device_id: 0,
        }
    }
}

/// Tier configuration
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
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

// Implement conversion from old TierConfig to new TierConfig
impl From<crate::config_old::TierConfig> for TierConfig {
    fn from(old: crate::config_old::TierConfig) -> Self {
        Self {
            name: old.name,
            tier_id: old.tier_id,
            capacity: old.capacity,
            access_latency: old.access_latency,
            description: old.description,
        }
    }
}

impl From<&crate::config_old::TierConfig> for TierConfig {
    fn from(old: &crate::config_old::TierConfig) -> Self {
        Self {
            name: old.name.clone(),
            tier_id: old.tier_id,
            capacity: old.capacity,
            access_latency: old.access_latency,
            description: old.description.clone(),
        }
    }
}

/// Complete training configuration (supports both old and new formats)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingConfig {
    /// Storage tier configurations (supports both [[tier]] and [[tiers]])
    #[serde(default, alias = "tier")]
    pub tiers: Vec<TierConfig>,

    /// Model configuration
    #[serde(default)]
    pub model: ModelConfig,

    /// Training parameters
    #[serde(default)]
    pub training: TrainingParams,

    /// Backend configuration
    #[serde(default)]
    pub backend: BackendConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            tiers: create_default_tiers(),
            model: ModelConfig::default(),
            training: TrainingParams::default(),
            backend: BackendConfig::default(),
        }
    }
}

fn create_default_tiers() -> Vec<TierConfig> {
    vec![
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
    ]
}

impl TrainingConfig {
    /// Load configuration from a TOML file
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| EnvError::ConfigError {
            message: format!("Failed to read config file: {}", e),
        })?;

        Self::from_toml_str(&content)
    }

    /// Parse configuration from TOML string
    pub fn from_toml_str(content: &str) -> Result<Self> {
        // First try to parse as new format
        if let Ok(mut config) = toml::from_str::<TrainingConfig>(content) {
            // Handle backward compatibility with old format
            // Old format uses [[tier]] instead of [[tiers]]
            if let Ok(old_format) = toml::from_str::<OldConfigFormat>(content) {
                if !old_format.tier.is_empty() {
                    config.tiers = old_format.tier;
                }
            }
            return Ok(config);
        }

        // If new format fails, try old format
        let old_format: OldConfigFormat =
            toml::from_str(content).map_err(|e| EnvError::ConfigError {
                message: format!("Failed to parse TOML: {}", e),
            })?;

        Ok(TrainingConfig {
            tiers: old_format.tier,
            ..Default::default()
        })
    }

    /// Create default configuration
    pub fn default_tiers() -> Self {
        Self::default()
    }

    /// Apply CLI overrides
    pub fn apply_overrides(
        &mut self,
        episodes: Option<usize>,
        max_steps: Option<usize>,
        batch_size: Option<usize>,
        learning_rate: Option<f64>,
        gamma: Option<f32>,
        backend: Option<String>,
        model: Option<String>,
        num_workers: Option<usize>,
    ) {
        if let Some(ep) = episodes {
            self.training.episodes = ep;
        }
        if let Some(ms) = max_steps {
            self.training.max_steps = ms;
        }
        if let Some(bs) = batch_size {
            self.training.batch_size = bs;
        }
        if let Some(lr) = learning_rate {
            self.training.learning_rate = lr;
        }
        if let Some(g) = gamma {
            self.training.gamma = g;
        }
        if let Some(b) = backend {
            self.backend.backend_type = match b.to_lowercase().as_str() {
                "cpu" | "ndarray" => BackendType::Cpu,
                "gpu" | "wgpu" => BackendType::Gpu,
                "torch" => BackendType::Torch,
                "cuda" => BackendType::Cuda,
                "rocm" => BackendType::Rocm,
                _ => BackendType::Cpu,
            };
        }
        if let Some(m) = model {
            self.model.architecture = match m.to_lowercase().as_str() {
                "dueling_dqn" => ModelArchitecture::DuelingDQN,
                "bandit_dqn" => ModelArchitecture::BanditDQN,
                "simple_dqn" => ModelArchitecture::SimpleDQN,
                _ => ModelArchitecture::DuelingDQN,
            };
        }
        if let Some(nw) = num_workers {
            self.training.num_workers = nw;
        }
    }
}

/// Old config format for backward compatibility
#[derive(Debug, Clone, Deserialize, Default)]
struct OldConfigFormat {
    #[serde(default)]
    tier: Vec<TierConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.training.episodes, 100);
        assert_eq!(config.model.feature_dim, 20);
        assert_eq!(config.tiers.len(), 5);
    }

    #[test]
    fn test_parse_old_format() {
        let old_format = r#"
[[tier]]
name = "Memory"
tier_id = 0
capacity = 800000.0
access_latency = 0.01
"#;
        let config = TrainingConfig::from_toml_str(old_format).unwrap();
        assert_eq!(config.tiers.len(), 1);
        assert_eq!(config.tiers[0].name, "Memory");
    }

    #[test]
    fn test_parse_new_format() {
        let new_format = r#"
[model]
architecture = "dueling_dqn"
feature_dim = 30

[training]
episodes = 200
batch_size = 256

[backend]
backend_type = "gpu"
device_id = 0
"#;
        let config = TrainingConfig::from_toml_str(new_format).unwrap();
        assert_eq!(config.training.episodes, 200);
        assert_eq!(config.training.batch_size, 256);
        assert_eq!(config.model.feature_dim, 30);
        assert_eq!(config.backend.backend_type, BackendType::Gpu);
    }

    #[test]
    fn test_apply_overrides() {
        let mut config = TrainingConfig::default();
        config.apply_overrides(
            Some(50),
            Some(500),
            Some(128),
            Some(0.01),
            Some(0.95),
            Some("gpu".to_string()),
            Some("bandit_dqn".to_string()),
            Some(4),
        );
        assert_eq!(config.training.episodes, 50);
        assert_eq!(config.training.max_steps, 500);
        assert_eq!(config.training.batch_size, 128);
        assert_eq!(config.training.learning_rate, 0.01);
        assert_eq!(config.training.gamma, 0.95);
        assert_eq!(config.backend.backend_type, BackendType::Gpu);
        assert_eq!(config.model.architecture, ModelArchitecture::BanditDQN);
        assert_eq!(config.training.num_workers, 4);
    }

    #[test]
    fn test_default_num_workers() {
        let config = TrainingConfig::default();
        assert_eq!(config.training.num_workers, 2);
    }
}
