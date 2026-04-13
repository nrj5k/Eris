//! # Policy Checkpoint Utilities
//!
//! This module provides utility functions for saving and loading policy checkpoint
//! metadata and configuration files.
//!
//! ## Overview
//!
//! The checkpoint utilities provide:
//!
//! 1. **save_metadata/load_metadata**: Save and load checkpoint metadata
//! 2. **save_config/load_config**: Save and load model configuration
//! 3. **validate_checkpoint_dir**: Validate checkpoint directory structure
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
//! use eris::policies::checkpoint::{save_metadata, load_metadata};
//! use eris::training::checkpoint::CheckpointMetadata;
//! use std::path::Path;
//!
//! // Save metadata
//! let checkpoint_dir = Path::new("checkpoints/bandit_policy_v1");
//! let metadata = CheckpointMetadata::new("Bandit".to_string(), 1, 1000, json!({}));
//! save_metadata(checkpoint_dir, &metadata)?;
//!
//! // Load metadata
//! let metadata = load_metadata(checkpoint_dir)?;
//! println!("Policy type: {}", metadata.policy_type);
//! ```

use serde::Serialize;
use std::error::Error;
use std::fs;
use std::path::Path;

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
/// use eris::training::checkpoint::CheckpointMetadata;
/// let metadata = CheckpointMetadata::new("Bandit".to_string(), 1, 1000, json!({}));
/// save_metadata(Path::new("checkpoints/policy_v1"), &metadata)?;
/// ```
pub fn save_metadata(
    path: &Path,
    metadata: &crate::training::checkpoint::CheckpointMetadata,
) -> Result<(), Box<dyn Error>> {
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
pub fn load_metadata(
    path: &Path,
) -> Result<crate::training::checkpoint::CheckpointMetadata, Box<dyn Error>> {
    let metadata_path = path.join("metadata.json");

    // Check if metadata file exists
    if !metadata_path.exists() {
        return Err(format!("Metadata file not found at {:?}", metadata_path).into());
    }

    // Read and deserialize
    let json = fs::read_to_string(&metadata_path)?;
    let metadata: crate::training::checkpoint::CheckpointMetadata = serde_json::from_str(&json)?;

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
    use crate::training::checkpoint::CheckpointMetadata;
    use serde::{Deserialize, Serialize};
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn test_save_metadata() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        let metadata = CheckpointMetadata::new("Bandit".to_string(), 1, json!({"test": "value"}));

        // Save metadata
        save_metadata(checkpoint_path, &metadata).expect("Failed to save metadata");

        // Verify file exists
        let metadata_file = checkpoint_path.join("metadata.json");
        assert!(metadata_file.exists());

        // Verify content
        let content = fs::read_to_string(&metadata_file).expect("Failed to read metadata file");
        assert!(content.contains("Bandit"));
        assert!(content.contains("test"));
    }

    #[test]
    fn test_load_metadata() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let checkpoint_path = temp_dir.path();

        // Create metadata.json manually
        let metadata = CheckpointMetadata {
            policy_type: "DQN".to_string(),
            version: 2,
            created_at: "2024-01-15T10:30:00Z".to_string(),
            epoch: 1,
            step_count: 2500,
            episode_count: None,
            epsilon: 0.1,
            best_reward: None,
            avg_reward: None,
            state_dim: None,
            action_dim: None,
            feature_dim: None,
            model_config: None,
        };

        let json = serde_json::to_string_pretty(&metadata).expect("Failed to serialize");
        let metadata_path = checkpoint_path.join("metadata.json");
        fs::write(&metadata_path, json).expect("Failed to write metadata file");

        // Load metadata
        let loaded = load_metadata(checkpoint_path).expect("Failed to load metadata");

        assert_eq!(loaded.policy_type, "DQN");
        assert_eq!(loaded.step_count, 2500);
        assert_eq!(loaded.version, 2);
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
        let original = CheckpointMetadata::new("Metis".to_string(), 1, json!({"feature_dim": 20}));

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
}
