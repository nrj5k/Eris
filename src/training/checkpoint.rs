//! Burn-compatible checkpoint utilities for DQN models.
//!
//! Provides checkpoint functionality with DQN-specific metadata (epsilon, step_count).

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Current checkpoint format version
pub const CHECKPOINT_VERSION: u32 = 2;

/// Unified checkpoint metadata for all policy types.
/// Replaces the three divergent CheckpointMetadata structs.
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
        // If checkpoint doesn't have dimension info, skip check (old checkpoint)
        if self.state_dim.is_none() && self.action_dim.is_none() && self.feature_dim.is_none() {
            return Ok(());
        }

        // Check state dimension
        if let Some(saved_state_dim) = self.state_dim {
            if saved_state_dim != expected_state_dim {
                return Err(format!(
                    "Model dimension mismatch: checkpoint was trained with state_dim={}, \
                     but current model expects state_dim={}. \
                     Please delete old checkpoints and retrain, or use matching dimensions.",
                    saved_state_dim, expected_state_dim
                ));
            }
        }

        // Check action dimension
        if let Some(saved_action_dim) = self.action_dim {
            if saved_action_dim != expected_action_dim {
                return Err(format!(
                    "Model dimension mismatch: checkpoint was trained with action_dim={}, \
                     but current model expects action_dim={}. \
                     Please delete old checkpoints and retrain, or use matching dimensions.",
                    saved_action_dim, expected_action_dim
                ));
            }
        }

        // Check feature dimension
        if let Some(saved_feature_dim) = self.feature_dim {
            if saved_feature_dim != expected_feature_dim {
                return Err(format!(
                    "Model dimension mismatch: checkpoint was trained with feature_dim={}, \
                     but current model expects feature_dim={}. \
                     Please delete old checkpoints and retrain, or use matching dimensions.",
                    saved_feature_dim, expected_feature_dim
                ));
            }
        }

        Ok(())
    }
}

/// Trait for models and policies that can be checkpointed.
///
/// This trait provides metadata for checkpointing. The actual
/// save/load mechanics are handled by generic functions that work
/// with Burn modules.
///
/// For policies that wrap models, implement this trait to provide
/// metadata, and use the model's Module implementation for save/load.
pub trait Checkpointable<B: Backend> {
    /// Return the checkpoint name for this model type.
    /// Used to construct checkpoint file paths.
    ///
    /// Example: "q_network", "contextual_bandit", "combined_model"
    fn checkpoint_name(&self) -> &str;

    /// Return metadata describing the current state of this model.
    ///
    /// The training loop should update fields like step_count,
    /// episode_count, epsilon, best_reward with actual values.
    fn checkpoint_metadata(&self) -> CheckpointMetadata;

    /// Return a reference to the underlying model for checkpointing.
    ///
    /// For policies that wrap models, this returns the model.
    /// For models themselves, this returns self.
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

/// Generic save function for any Burn module.
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

    log::info!(
        "[STAGE:checkpoint_saved] {} (epoch {})",
        model_path.display(),
        epoch
    );

    Ok(model_path)
}

/// Validate checkpoint path for security.
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

/// Load a checkpoint for any Burn module.
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
        log::warn!(
            "[STAGE:checkpoint_warning] Metadata not found for epoch {}, using defaults",
            epoch
        );
        CheckpointMetadata::default()
    };

    log::info!(
        "[STAGE:checkpoint_loaded] {} (epoch {})",
        model_path.display(),
        epoch
    );

    Ok((model, metadata))
}

/// Helper functions for checkpoint management using Burn's recorders.
pub struct DQNCheckpointHelper;

impl DQNCheckpointHelper {
    /// Save model with metadata using Burn's recorder.
    #[deprecated(since = "0.1.0", note = "Use save_checkpoint() instead")]
    pub fn save<B: Backend, M: Module<B>>(
        model: &M,
        directory: impl AsRef<Path>,
        name: &str,
        epoch: usize,
        metadata: &CheckpointMetadata,
    ) -> Result<(), Box<dyn std::error::Error>> {
        save_checkpoint::<B, M>(model, directory, name, epoch, metadata)?;
        Ok(())
    }

    /// Load model with metadata using Burn's recorder.
    #[deprecated(since = "0.1.0", note = "Use load_checkpoint() instead")]
    pub fn load<B: Backend, M: Module<B>>(
        directory: impl AsRef<Path>,
        name: &str,
        epoch: usize,
        device: &B::Device,
        config: impl FnOnce() -> M,
    ) -> Result<(M, CheckpointMetadata), Box<dyn std::error::Error>> {
        load_checkpoint::<B, M>(directory, name, epoch, device, config)
    }

    /// Delete checkpoint files.
    pub fn delete(
        directory: impl AsRef<Path>,
        name: &str,
        epoch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let directory = directory.as_ref();

        let model_path = directory.join(format!("{}-{}.mpk", name, epoch));
        if model_path.exists() {
            std::fs::remove_file(&model_path)?;
        }

        let meta_path = directory.join(format!("{}-{}.json", name, epoch));
        if meta_path.exists() {
            std::fs::remove_file(&meta_path)?;
        }

        Ok(())
    }
}
