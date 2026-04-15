//! Checkpoint system for RL models with rich metadata.

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Current checkpoint format version
pub const CHECKPOINT_VERSION: u32 = 2;

/// Unified checkpoint metadata for all policy types.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointMetadata {
    /// Policy type identifier (e.g., "DQN", "Bandit", "Combined")
    pub policy_type: String,
    /// Checkpoint format version for backward compatibility
    pub version: u32,
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
    /// Training epoch
    pub epoch: usize,
    /// Total training steps completed
    pub step_count: usize,
    /// Total episodes completed
    pub episode_count: Option<usize>,
    /// Exploration parameter (epsilon for DQN, etc.)
    pub epsilon: f32,
    /// Best reward seen during training
    pub best_reward: Option<f32>,
    /// Average reward over last N episodes
    pub avg_reward: Option<f32>,
    /// Model architecture dimensions (for compatibility checking)
    pub state_dim: Option<usize>,
    pub action_dim: Option<usize>,
    pub feature_dim: Option<usize>,
    /// Model configuration as JSON (for reconstruction)
    pub model_config: Option<serde_json::Value>,
}

impl CheckpointMetadata {
    /// Create new metadata with defaults
    pub fn new(policy_type: String, epoch: usize, model_config: serde_json::Value) -> Self {
        Self {
            policy_type,
            version: CHECKPOINT_VERSION,
            created_at: chrono::Utc::now().to_rfc3339(),
            epoch,
            step_count: 0,
            episode_count: None,
            epsilon: 1.0,
            best_reward: None,
            avg_reward: None,
            state_dim: None,
            action_dim: None,
            feature_dim: None,
            model_config: Some(model_config),
        }
    }

    /// Create new metadata with dimension info
    pub fn new_with_dims(
        policy_type: String,
        epoch: usize,
        state_dim: usize,
        action_dim: usize,
        feature_dim: usize,
    ) -> Self {
        Self {
            policy_type,
            version: CHECKPOINT_VERSION,
            created_at: chrono::Utc::now().to_rfc3339(),
            epoch,
            step_count: 0,
            episode_count: None,
            epsilon: 1.0,
            best_reward: None,
            avg_reward: None,
            state_dim: Some(state_dim),
            action_dim: Some(action_dim),
            feature_dim: Some(feature_dim),
            model_config: None,
        }
    }

    /// Check if checkpoint dimensions match the expected model dimensions
    pub fn check_dimensions(
        &self,
        expected_state_dim: usize,
        expected_action_dim: usize,
        expected_feature_dim: usize,
    ) -> Result<(), String> {
        if self.state_dim.is_none() && self.action_dim.is_none() && self.feature_dim.is_none() {
            return Ok(());
        }
        if let Some(saved) = self.state_dim {
            if saved != expected_state_dim {
                return Err(format!(
                    "state_dim mismatch: checkpoint={}, expected={}",
                    saved, expected_state_dim
                ));
            }
        }
        if let Some(saved) = self.action_dim {
            if saved != expected_action_dim {
                return Err(format!(
                    "action_dim mismatch: checkpoint={}, expected={}",
                    saved, expected_action_dim
                ));
            }
        }
        if let Some(saved) = self.feature_dim {
            if saved != expected_feature_dim {
                return Err(format!(
                    "feature_dim mismatch: checkpoint={}, expected={}",
                    saved, expected_feature_dim
                ));
            }
        }
        Ok(())
    }
}

/// Trait for models that can be checkpointed.
pub trait Checkpointable<B: Backend> {
    /// Return the checkpoint name for this model type.
    fn checkpoint_name(&self) -> &str;

    /// Return metadata describing the current state.
    fn checkpoint_metadata(&self) -> CheckpointMetadata;

    /// Return a reference to the underlying model for checkpointing.
    fn model(&self) -> &impl Module<B>;
}

/// Extension trait for updating checkpoint metadata with training state.
pub trait CheckpointMetadataExt {
    /// Update metadata with actual training state.
    fn with_training_state(
        self,
        step_count: usize,
        episode_count: usize,
        epsilon: f32,
        best_reward: f32,
    ) -> Self;
}

impl CheckpointMetadataExt for CheckpointMetadata {
    fn with_training_state(
        mut self,
        step_count: usize,
        episode_count: usize,
        epsilon: f32,
        best_reward: f32,
    ) -> Self {
        self.step_count = step_count;
        self.epoch = episode_count;
        self.epsilon = epsilon;
        self.best_reward = Some(best_reward);
        self
    }
}

/// Validate checkpoint path - reject path traversal
fn validate_checkpoint_path(
    directory: &Path,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if directory
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err("Checkpoint directory contains '..' (path traversal not allowed)".into());
    }
    if name.contains("..") || name.contains('/') || name.contains('\\') {
        return Err("Checkpoint name contains invalid characters".into());
    }
    Ok(())
}

/// Save model checkpoint with rich metadata.
pub fn save_checkpoint<B, M>(
    model: &M,
    directory: impl AsRef<Path>,
    name: &str,
    epoch: usize,
    metadata: &CheckpointMetadata,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>>
where
    B: Backend,
    M: Module<B> + Clone,
{
    let directory = directory.as_ref();
    validate_checkpoint_path(directory, name)?;
    std::fs::create_dir_all(directory)?;

    let model_path = directory.join(format!("{}-{}.mpk", name, epoch));
    let meta_path = directory.join(format!("{}-{}.json", name, epoch));

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(&model_path, &recorder)?;
    std::fs::write(&meta_path, serde_json::to_string_pretty(metadata)?)?;

    if !model_path.exists() {
        return Err(format!("Model file not created: {}", model_path.display()).into());
    }
    if !meta_path.exists() {
        return Err(format!("Metadata file not created: {}", meta_path.display()).into());
    }

    Ok(model_path)
}

/// Load model checkpoint with rich metadata.
pub fn load_checkpoint<B, M>(
    directory: impl AsRef<Path>,
    name: &str,
    epoch: usize,
    device: &B::Device,
    config: impl FnOnce() -> M,
) -> Result<(M, CheckpointMetadata), Box<dyn std::error::Error>>
where
    B: Backend,
    M: Module<B>,
{
    let directory = directory.as_ref();
    validate_checkpoint_path(directory, name)?;

    let model_path = directory.join(format!("{}-{}.mpk", name, epoch));
    let meta_path = directory.join(format!("{}-{}.json", name, epoch));

    if !model_path.exists() {
        return Err(format!("Checkpoint not found: {}", model_path.display()).into());
    }

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = config().load_file(&model_path, &recorder, device)?;

    let metadata: CheckpointMetadata = if meta_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?
    } else {
        CheckpointMetadata::default()
    };

    Ok((model, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_metadata_new() {
        let meta = CheckpointMetadata::new("DQN".to_string(), 10, serde_json::json!({}));
        assert_eq!(meta.policy_type, "DQN");
        assert_eq!(meta.epoch, 10);
        assert_eq!(meta.version, CHECKPOINT_VERSION);
    }

    #[test]
    fn test_checkpoint_metadata_with_dims() {
        let meta = CheckpointMetadata::new_with_dims("DQN".to_string(), 0, 4, 10, 32);
        assert_eq!(meta.state_dim, Some(4));
        assert_eq!(meta.action_dim, Some(10));
        assert_eq!(meta.feature_dim, Some(32));
    }

    #[test]
    fn test_check_dimensions_match() {
        let meta = CheckpointMetadata::new_with_dims("DQN".to_string(), 0, 4, 10, 32);
        assert!(meta.check_dimensions(4, 10, 32).is_ok());
    }

    #[test]
    fn test_check_dimensions_mismatch() {
        let meta = CheckpointMetadata::new_with_dims("DQN".to_string(), 0, 4, 10, 32);
        assert!(meta.check_dimensions(8, 10, 32).is_err());
    }

    #[test]
    fn test_check_dimensions_no_dims_ok() {
        let meta = CheckpointMetadata::default();
        // No dims stored, should be ok
        assert!(meta.check_dimensions(4, 10, 32).is_ok());
    }

    #[test]
    fn test_metadata_ext() {
        let meta = CheckpointMetadata::new("DQN".to_string(), 0, serde_json::json!({}))
            .with_training_state(1000, 50, 0.1, 15.5);
        assert_eq!(meta.step_count, 1000);
        assert_eq!(meta.epoch, 50);
        assert!((meta.epsilon - 0.1).abs() < f32::EPSILON);
        assert_eq!(meta.best_reward, Some(15.5));
    }

    #[test]
    fn test_validate_path_rejects_parent_dir() {
        assert!(validate_checkpoint_path(std::path::Path::new("../etc"), "model").is_err());
        assert!(validate_checkpoint_path(std::path::Path::new("foo/../../bar"), "model").is_err());
    }

    #[test]
    fn test_validate_path_rejects_bad_name() {
        assert!(validate_checkpoint_path(std::path::Path::new("ok"), "../bad").is_err());
        assert!(validate_checkpoint_path(std::path::Path::new("ok"), "a/b").is_err());
    }

    #[test]
    fn test_validate_path_accepts_valid() {
        assert!(validate_checkpoint_path(std::path::Path::new("checkpoints"), "model").is_ok());
    }
}
