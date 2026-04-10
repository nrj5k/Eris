//! # Unified Policy Checkpointing
//!
//! This module provides a standardized checkpointing framework for all cache policies.
//! It defines a common trait and utilities for saving and loading policy state,
//! metadata, and configuration.
//!
//! ## Overview
//!
//! The checkpoint system consists of:
//!
//! 1. **CheckpointMetadata**: Metadata about a saved checkpoint
//! 2. **Checkpoint trait**: Common interface for policy checkpointing
//! 3. **Helper functions**: Utilities for saving/loading metadata and config
//!
//! ## Design Philosophy
//!
//! - **Consistency**: All policies use the same checkpoint format
//! - **Versioning**: Metadata includes version for backward compatibility
//! - **Extensibility**: Easy to add new fields without breaking existing checkpoints
//! - **Safety**: Proper error handling and validation
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use eris::policies::checkpoint::{Checkpoint, CheckpointMetadata};
//! use std::path::Path;
//!
//! // Save a checkpoint
//! let checkpoint_dir = Path::new("checkpoints/bandit_policy_v1");
//! policy.save_checkpoint(checkpoint_dir)?;
//!
//! // Load a checkpoint
//! let mut policy = BanditPolicy::new(config, &device);
//! policy.load_checkpoint(checkpoint_dir)?;
//!
//! // Get metadata
//! let metadata = policy.checkpoint_metadata();
//! println!("Policy type: {}", metadata.policy_type);
//! println!("Steps: {}", metadata.step_count);
//! ```

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::Path;

/// Current checkpoint format version
///
/// Increment this when making breaking changes to the checkpoint format.
/// Old checkpoints may need migration code.
pub const CHECKPOINT_VERSION: u32 = 1;

/// Metadata for checkpoint serialization
///
/// This struct contains information about a policy checkpoint that enables
/// versioning, debugging, and policy identification.
///
/// # Fields
///
/// * `policy_type` - Human-readable policy type (e.g., "Bandit", "DQN")
/// * `version` - Checkpoint format version for backward compatibility
/// * `created_at` - ISO 8601 timestamp when checkpoint was created
/// * `step_count` - Number of training steps completed
/// * `episode_count` - Number of episodes completed (optional)
/// * `best_reward` - Best reward seen during training (optional)
/// * `model_config` - Model architecture configuration as JSON
///
/// # Example
///
/// ```rust,ignore
/// let metadata = CheckpointMetadata {
///     policy_type: "Bandit".to_string(),
///     version: 1,
///     created_at: "2024-01-15T10:30:00Z".to_string(),
///     step_count: 1000,
///     episode_count: Some(100),
///     best_reward: Some(0.85),
///     model_config: json!({"hidden_layers": [64, 128]}),
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Policy type (e.g., "Bandit", "DQN", "Metis", "Catcher", "Cacheus")
    pub policy_type: String,

    /// Checkpoint version for backward compatibility
    pub version: u32,

    /// Creation timestamp in ISO 8601 format
    pub created_at: String,

    /// Total training steps completed
    pub step_count: usize,

    /// Total episodes completed (if applicable to policy)
    pub episode_count: Option<usize>,

    /// Best reward achieved during training (if tracked)
    pub best_reward: Option<f32>,

    /// Model architecture configuration
    pub model_config: serde_json::Value,
}

impl CheckpointMetadata {
    /// Create new checkpoint metadata with current timestamp
    ///
    /// # Arguments
    ///
    /// * `policy_type` - Name of the policy type
    /// * `step_count` - Number of training steps completed
    /// * `model_config` - Model configuration as JSON
    ///
    /// # Returns
    ///
    /// New CheckpointMetadata with current timestamp and version 1
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let metadata = CheckpointMetadata::new(
    ///     "Bandit".to_string(),
    ///     1000,
    ///     json!({"layers": [64, 128]})
    /// );
    /// ```
    pub fn new(policy_type: String, step_count: usize, model_config: serde_json::Value) -> Self {
        Self {
            policy_type,
            version: CHECKPOINT_VERSION,
            created_at: chrono::Utc::now().to_rfc3339(),
            step_count,
            episode_count: None,
            best_reward: None,
            model_config,
        }
    }

    /// Create metadata builder for optional fields
    ///
    /// # Returns
    ///
    /// Builder instance for constructing metadata
    pub fn builder() -> CheckpointMetadataBuilder {
        CheckpointMetadataBuilder::default()
    }
}

/// Builder for CheckpointMetadata
///
/// Allows setting optional fields during construction.
///
/// # Example
///
/// ```rust,ignore
/// let metadata = CheckpointMetadata::builder()
///     .policy_type("Bandit")
///     .step_count(1000)
///     .model_config(json!({"layers": [64]}))
///     .episode_count(100)
///     .best_reward(0.85)
///     .build()?;
/// ```
#[derive(Debug, Default)]
pub struct CheckpointMetadataBuilder {
    policy_type: Option<String>,
    step_count: Option<usize>,
    episode_count: Option<usize>,
    best_reward: Option<f32>,
    model_config: Option<serde_json::Value>,
}

impl CheckpointMetadataBuilder {
    /// Set policy type
    pub fn policy_type(mut self, policy_type: impl Into<String>) -> Self {
        self.policy_type = Some(policy_type.into());
        self
    }

    /// Set step count
    pub fn step_count(mut self, step_count: usize) -> Self {
        self.step_count = Some(step_count);
        self
    }

    /// Set episode count
    pub fn episode_count(mut self, episode_count: usize) -> Self {
        self.episode_count = Some(episode_count);
        self
    }

    /// Set best reward
    pub fn best_reward(mut self, best_reward: f32) -> Self {
        self.best_reward = Some(best_reward);
        self
    }

    /// Set model configuration
    pub fn model_config(mut self, model_config: serde_json::Value) -> Self {
        self.model_config = Some(model_config);
        self
    }

    /// Build the final CheckpointMetadata
    ///
    /// # Errors
    ///
    /// Returns error if required fields are missing
    pub fn build(self) -> Result<CheckpointMetadata, Box<dyn Error>> {
        let policy_type = self.policy_type.ok_or("policy_type is required")?;
        let step_count = self.step_count.ok_or("step_count is required")?;
        let model_config = self.model_config.ok_or("model_config is required")?;

        Ok(CheckpointMetadata {
            policy_type,
            version: CHECKPOINT_VERSION,
            created_at: chrono::Utc::now().to_rfc3339(),
            step_count,
            episode_count: self.episode_count,
            best_reward: self.best_reward,
            model_config,
        })
    }
}

/// Trait for policies that support checkpointing
///
/// This trait defines a common interface for saving and loading policy state.
/// All policies (Bandit, DQN, Metis, Catcher, Cacheus) should implement this trait.
///
/// # Design Goals
///
/// - **Consistency**: All policies use the same checkpoint interface
/// - **Safety**: Errors are properly handled and reported
/// - **Extensibility**: New fields can be added without breaking compatibility
///
/// # Required Methods
///
/// - `save_checkpoint`: Save policy state to a directory
/// - `load_checkpoint`: Load policy state from a directory
/// - `checkpoint_metadata`: Get metadata about the current checkpoint
///
/// # Example Implementation
///
/// ```rust,ignore
/// impl<B: AutodiffBackend> Checkpoint for MyPolicy<B> {
///     fn save_checkpoint(&self, path: &Path) -> Result<(), Box<dyn Error>> {
///         // Create checkpoint directory
///         fs::create_dir_all(path)?;
///
///         // Save metadata
///         let metadata = self.checkpoint_metadata();
///         save_metadata(path, &metadata)?;
///
///         // Save model weights
///         let weights_path = path.join("weights.bin");
///         self.save_weights(&weights_path)?;
///
///         Ok(())
///     }
///
///     fn load_checkpoint(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
///         // Load metadata
///         let metadata = load_metadata(path)?;
///
///         // Validate policy type
///         if metadata.policy_type != "MyPolicy" {
///             return Err(format!("Invalid policy type: {}", metadata.policy_type).into());
///         }
///
///         // Load weights
///         let weights_path = path.join("weights.bin");
///         self.load_weights(&weights_path)?;
///
///         Ok(())
///     }
///
///     fn checkpoint_metadata(&self) -> CheckpointMetadata {
///         CheckpointMetadata::new(
///             "MyPolicy".to_string(),
///             self.step_count,
///             self.get_config_json(),
///         )
///     }
/// }
/// ```
pub trait Checkpoint {
    /// Save policy state to directory
    ///
    /// Creates a checkpoint at the specified path containing:
    /// - `metadata.json`: Checkpoint metadata
    /// - `config.json`: Model configuration
    /// - Policy-specific files (weights, replay buffer, etc.)
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path for checkpoint files
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error on failure
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Directory creation fails
    /// - File write fails
    /// - Serialization fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let checkpoint_dir = Path::new("checkpoints/policy_v1");
    /// policy.save_checkpoint(checkpoint_dir)?;
    /// ```
    fn save_checkpoint(&self, path: &Path) -> Result<(), Box<dyn Error>>;

    /// Load policy state from directory
    ///
    /// Loads a checkpoint from the specified path, restoring:
    /// - Model configuration
    /// - Model weights
    /// - Training state
    /// - Optimizer state (if applicable)
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path containing checkpoint files
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error on failure
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Checkpoint directory doesn't exist
    /// - Required files are missing
    /// - Deserialization fails
    /// - Version mismatch
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let checkpoint_dir = Path::new("checkpoints/policy_v1");
    /// let mut policy = MyPolicy::new(config, &device);
    /// policy.load_checkpoint(checkpoint_dir)?;
    /// ```
    fn load_checkpoint(&mut self, path: &Path) -> Result<(), Box<dyn Error>>;

    /// Get checkpoint metadata
    ///
    /// Returns metadata about the current policy state, including:
    /// - Policy type
    /// - Training progress
    /// - Model configuration summary
    ///
    /// # Returns
    ///
    /// CheckpointMetadata struct
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let metadata = policy.checkpoint_metadata();
    /// println!("Policy: {}", metadata.policy_type);
    /// println!("Steps: {}", metadata.step_count);
    /// ```
    fn checkpoint_metadata(&self) -> CheckpointMetadata;
}

/// Save checkpoint metadata to JSON file
///
/// Writes metadata to `metadata.json` in the specified directory.
///
/// # Arguments
///
/// * `path` - Directory path for metadata file
/// * `metadata` - Metadata to save
///
/// # Returns
///
/// `Ok(())` on success, error on failure
///
/// # Errors
///
/// Returns error if:
/// - JSON serialization fails
/// - File write fails
///
/// # Example
///
/// ```rust,ignore
/// let metadata = CheckpointMetadata::new(
///     "Bandit".to_string(),
///     1000,
///     json!({"layers": [64]})
/// );
/// save_metadata(Path::new("checkpoints/policy_v1"), &metadata)?;
/// ```
pub fn save_metadata(path: &Path, metadata: &CheckpointMetadata) -> Result<(), Box<dyn Error>> {
    // Create directory if it doesn't exist
    fs::create_dir_all(path)?;

    // Serialize metadata
    let metadata_path = path.join("metadata.json");
    let json = serde_json::to_string_pretty(metadata)?;

    // Write to file
    fs::write(&metadata_path, json)?;

    log::info!("Saved checkpoint metadata to {:?}", metadata_path);
    Ok(())
}

/// Load checkpoint metadata from JSON file
///
/// Reads metadata from `metadata.json` in the specified directory.
///
/// # Arguments
///
/// * `path` - Directory path containing metadata file
///
/// # Returns
///
/// `CheckpointMetadata` on success, error on failure
///
/// # Errors
///
/// Returns error if:
/// - Directory doesn't exist
/// - `metadata.json` file is missing
/// - JSON deserialization fails
///
/// # Example
///
/// ```rust,ignore
/// let metadata = load_metadata(Path::new("checkpoints/policy_v1"))?;
/// println!("Policy type: {}", metadata.policy_type);
/// ```
pub fn load_metadata(path: &Path) -> Result<CheckpointMetadata, Box<dyn Error>> {
    let metadata_path = path.join("metadata.json");

    // Check if metadata file exists
    if !metadata_path.exists() {
        return Err(format!("Metadata file not found at {:?}", metadata_path).into());
    }

    // Read and deserialize
    let json = fs::read_to_string(&metadata_path)?;
    let metadata: CheckpointMetadata = serde_json::from_str(&json)?;

    log::info!("Loaded checkpoint metadata from {:?}", metadata_path);
    Ok(metadata)
}

/// Save model configuration to JSON file
///
/// Writes configuration to `config.json` in the specified directory.
///
/// # Type Parameters
///
/// * `C` - Configuration type (must be serializable)
///
/// # Arguments
///
/// * `path` - Directory path for config file
/// * `config` - Configuration to save
///
/// # Returns
///
/// `Ok(())` on success, error on failure
///
/// # Errors
///
/// Returns error if:
/// - JSON serialization fails
/// - File write fails
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Serialize)]
/// struct MyConfig {
///     hidden_layers: Vec<usize>,
///     learning_rate: f32,
/// }
///
/// let config = MyConfig {
///     hidden_layers: vec![64, 128],
///     learning_rate: 0.001,
/// };
/// save_config(Path::new("checkpoints/policy_v1"), &config)?;
/// ```
pub fn save_config<C: serde::Serialize>(path: &Path, config: &C) -> Result<(), Box<dyn Error>> {
    // Create directory if it doesn't exist
    fs::create_dir_all(path)?;

    // Serialize config
    let config_path = path.join("config.json");
    let json = serde_json::to_string_pretty(config)?;

    // Write to file
    fs::write(&config_path, json)?;

    log::info!("Saved config to {:?}", config_path);
    Ok(())
}

/// Load model configuration from JSON file
///
/// Reads configuration from `config.json` in the specified directory.
///
/// # Type Parameters
///
/// * `C` - Configuration type (must be deserializable)
///
/// # Arguments
///
/// * `path` - Directory path containing config file
///
/// # Returns
///
/// Configuration struct on success, error on failure
///
/// # Errors
///
/// Returns error if:
/// - Directory doesn't exist
/// - `config.json` file is missing
/// - JSON deserialization fails
///
/// # Example
///
/// ```rust,ignore
/// #[derive(Deserialize)]
/// struct MyConfig {
///     hidden_layers: Vec<usize>,
///     learning_rate: f32,
/// }
///
/// let config: MyConfig = load_config(Path::new("checkpoints/policy_v1"))?;
/// println!("Learning rate: {}", config.learning_rate);
/// ```
pub fn load_config<C: serde::de::DeserializeOwned>(path: &Path) -> Result<C, Box<dyn Error>> {
    let config_path = path.join("config.json");

    // Check if config file exists
    if !config_path.exists() {
        return Err(format!("Config file not found at {:?}", config_path).into());
    }

    // Read and deserialize
    let json = fs::read_to_string(&config_path)?;
    let config: C = serde_json::from_str(&json)?;

    log::info!("Loaded config from {:?}", config_path);
    Ok(config)
}

/// Validate checkpoint directory exists and contains required files
///
/// # Arguments
///
/// * `path` - Directory path to validate
/// * `required_files` - List of required file names
///
/// # Returns
///
/// `Ok(())` if all required files exist, error otherwise
///
/// # Example
///
/// ```rust,ignore
/// validate_checkpoint_dir(
///     Path::new("checkpoints/policy_v1"),
///     &["metadata.json", "config.json", "weights.bin"]
/// )?;
/// ```
pub fn validate_checkpoint_dir(path: &Path, required_files: &[&str]) -> Result<(), Box<dyn Error>> {
    if !path.exists() {
        return Err(format!("Checkpoint directory does not exist: {:?}", path).into());
    }

    if !path.is_dir() {
        return Err(format!("Path is not a directory: {:?}", path).into());
    }

    for file_name in required_files {
        let file_path = path.join(file_name);
        if !file_path.exists() {
            return Err(format!("Required file missing: {:?}", file_path).into());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_metadata_creation() {
        let metadata =
            CheckpointMetadata::new("Bandit".to_string(), 1000, json!({"layers": [64, 128]}));

        assert_eq!(metadata.policy_type, "Bandit");
        assert_eq!(metadata.version, CHECKPOINT_VERSION);
        assert_eq!(metadata.step_count, 1000);
        assert!(metadata.episode_count.is_none());
        assert!(metadata.best_reward.is_none());
        assert!(!metadata.created_at.is_empty());
    }

    #[test]
    fn test_checkpoint_metadata_builder() {
        let metadata = CheckpointMetadata::builder()
            .policy_type("DQN")
            .step_count(5000)
            .episode_count(100)
            .best_reward(0.95)
            .model_config(json!({"hidden_layers": [256, 128]}))
            .build()
            .expect("Failed to build metadata");

        assert_eq!(metadata.policy_type, "DQN");
        assert_eq!(metadata.step_count, 5000);
        assert_eq!(metadata.episode_count, Some(100));
        assert_eq!(metadata.best_reward, Some(0.95));
    }

    #[test]
    fn test_checkpoint_metadata_builder_missing_fields() {
        let result = CheckpointMetadata::builder().policy_type("Test").build();

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("step_count"));
    }

    #[test]
    fn test_checkpoint_metadata_serialization() {
        let metadata = CheckpointMetadata {
            policy_type: "Bandit".to_string(),
            version: 1,
            created_at: "2024-01-15T10:30:00Z".to_string(),
            step_count: 1000,
            episode_count: Some(100),
            best_reward: Some(0.85),
            model_config: json!({"input_dim": 15, "hidden_layers": [64]}),
        };

        // Serialize to JSON
        let json = serde_json::to_string(&metadata).expect("Failed to serialize");
        assert!(json.contains("Bandit"));
        assert!(json.contains("1000"));

        // Deserialize back
        let deserialized: CheckpointMetadata =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized.policy_type, metadata.policy_type);
        assert_eq!(deserialized.step_count, metadata.step_count);
        assert_eq!(deserialized.episode_count, metadata.episode_count);
    }

    #[test]
    fn test_save_metadata() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        let metadata = CheckpointMetadata::new("Bandit".to_string(), 500, json!({"test": "value"}));

        // Save metadata
        save_metadata(checkpoint_path, &metadata).expect("Failed to save metadata");

        // Verify file exists
        let metadata_file = checkpoint_path.join("metadata.json");
        assert!(metadata_file.exists());

        // Verify content
        let content = fs::read_to_string(&metadata_file).expect("Failed to read metadata file");
        assert!(content.contains("Bandit"));
        assert!(content.contains("500"));
    }

    #[test]
    fn test_load_metadata() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        // Create metadata.json manually
        let metadata = CheckpointMetadata {
            policy_type: "DQN".to_string(),
            version: 1,
            created_at: "2024-01-15T10:30:00Z".to_string(),
            step_count: 2500,
            episode_count: None,
            best_reward: None,
            model_config: json!({}),
        };

        let json = serde_json::to_string_pretty(&metadata).expect("Failed to serialize");
        let metadata_path = checkpoint_path.join("metadata.json");
        fs::write(&metadata_path, json).expect("Failed to write metadata file");

        // Load metadata
        let loaded = load_metadata(checkpoint_path).expect("Failed to load metadata");

        assert_eq!(loaded.policy_type, "DQN");
        assert_eq!(loaded.step_count, 2500);
        assert_eq!(loaded.version, 1);
    }

    #[test]
    fn test_load_metadata_missing_file() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        // Try to load from directory without metadata.json
        let result = load_metadata(checkpoint_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_save_config() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        #[derive(Debug, Serialize, Deserialize)]
        struct TestConfig {
            hidden_layers: Vec<usize>,
            learning_rate: f32,
        }

        let config = TestConfig {
            hidden_layers: vec![64, 128, 256],
            learning_rate: 0.001,
        };

        // Save config
        save_config(checkpoint_path, &config).expect("Failed to save config");

        // Verify file exists
        let config_file = checkpoint_path.join("config.json");
        assert!(config_file.exists());

        // Verify content
        let content = fs::read_to_string(&config_file).expect("Failed to read config file");
        assert!(content.contains("hidden_layers"));
        assert!(content.contains("0.001"));
    }

    #[test]
    fn test_load_config() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestConfig {
            input_dim: usize,
            output_dim: usize,
            activation: String,
        }

        // Create config.json manually
        let config = TestConfig {
            input_dim: 128,
            output_dim: 10,
            activation: "relu".to_string(),
        };

        let config_path = checkpoint_path.join("config.json");
        let json = serde_json::to_string_pretty(&config).expect("Failed to serialize");
        fs::write(&config_path, json).expect("Failed to write config file");

        // Load config
        let loaded: TestConfig = load_config(checkpoint_path).expect("Failed to load config");

        assert_eq!(loaded.input_dim, 128);
        assert_eq!(loaded.output_dim, 10);
        assert_eq!(loaded.activation, "relu");
    }

    #[test]
    fn test_load_config_missing_file() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        #[derive(Debug, Deserialize)]
        struct TestConfig {
            value: i32,
        }

        // Try to load from directory without config.json
        let result: Result<TestConfig, Box<dyn Error>> = load_config(checkpoint_path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_roundtrip_metadata() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        // Create and save metadata
        let original =
            CheckpointMetadata::new("Metis".to_string(), 10000, json!({"feature_dim": 20}));

        save_metadata(checkpoint_path, &original).expect("Failed to save");

        // Load and verify
        let loaded = load_metadata(checkpoint_path).expect("Failed to load");

        assert_eq!(loaded.policy_type, original.policy_type);
        assert_eq!(loaded.version, original.version);
        assert_eq!(loaded.step_count, original.step_count);
        assert_eq!(loaded.created_at, original.created_at);
    }

    #[test]
    fn test_roundtrip_config() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        #[derive(Debug, Serialize, Deserialize, PartialEq)]
        struct TestConfig {
            layers: Vec<u8>,
            use_dropout: bool,
            dropout_rate: f64,
        }

        // Create and save config
        let original = TestConfig {
            layers: vec![32, 64, 128],
            use_dropout: true,
            dropout_rate: 0.2,
        };

        save_config(checkpoint_path, &original).expect("Failed to save");

        // Load and verify
        let loaded: TestConfig = load_config(checkpoint_path).expect("Failed to load");

        assert_eq!(loaded, original);
    }

    #[test]
    fn test_validate_checkpoint_dir() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        // Create required files
        fs::write(checkpoint_path.join("metadata.json"), "{}").expect("Failed to write");
        fs::write(checkpoint_path.join("config.json"), "{}").expect("Failed to write");
        fs::write(checkpoint_path.join("weights.bin"), vec![0u8, 1, 2]).expect("Failed to write");

        // Validate directory
        let result = validate_checkpoint_dir(
            checkpoint_path,
            &["metadata.json", "config.json", "weights.bin"],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_checkpoint_dir_missing_file() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        // Create only some files
        fs::write(checkpoint_path.join("metadata.json"), "{}").expect("Failed to write");

        // Validate directory
        let result = validate_checkpoint_dir(checkpoint_path, &["metadata.json", "config.json"]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("config.json"));
    }

    #[test]
    fn test_validate_checkpoint_dir_not_directory() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("not_a_directory.txt");

        // Create a file (not a directory)
        fs::write(&file_path, "test").expect("Failed to write");

        // Try to validate
        let result = validate_checkpoint_dir(&file_path, &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a directory"));
    }

    #[test]
    fn test_metadata_with_all_optional_fields() {
        let metadata = CheckpointMetadata {
            policy_type: "Catcher".to_string(),
            version: 1,
            created_at: "2024-01-15T12:00:00Z".to_string(),
            step_count: 5000,
            episode_count: Some(200),
            best_reward: Some(0.92),
            model_config: json!({"type": "catcher"}),
        };

        let json = serde_json::to_string(&metadata).expect("Failed to serialize");
        let loaded: CheckpointMetadata =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(loaded.episode_count, Some(200));
        assert_eq!(loaded.best_reward, Some(0.92));
    }

    #[test]
    fn test_checkpoint_version_constant() {
        // Ensure version is properly set
        assert_eq!(CHECKPOINT_VERSION, 1u32);
    }

    #[test]
    fn test_metadata_created_at_valid_iso8601() {
        let metadata = CheckpointMetadata::new("Test".to_string(), 100, json!({}));

        // Verify timestamp is valid ISO 8601 format
        let parsed = chrono::DateTime::parse_from_rfc3339(&metadata.created_at);
        assert!(parsed.is_ok(), "Created_at should be valid ISO 8601");
    }
}
