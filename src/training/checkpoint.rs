//! Burn-compatible checkpoint utilities for DQN models.
//!
//! Provides checkpoint functionality with DQN-specific metadata (epsilon, step_count).

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use std::path::Path;

/// Current checkpoint format version.
/// Re-exported from burnme-rly for DRY.
pub use burnme_rly::CHECKPOINT_VERSION;

/// Trait for models that can be checkpointed.
/// Re-exported from burnme-rly for DRY.
pub use burnme_rly::Checkpointable;

/// Extension trait for updating checkpoint metadata with training state.
/// Re-exported from burnme-rly for DRY.
pub use burnme_rly::CheckpointMetadataExt;

/// Unified checkpoint metadata for all policy types.
/// Re-exported from burnme-rly for DRY (struct definition is identical, serde-compatible).
pub use burnme_rly::CheckpointMetadata;

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
