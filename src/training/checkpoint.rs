//! Burn-compatible checkpoint utilities for DQN models.
//!
//! Provides checkpoint functionality with DQN-specific metadata (epsilon, step_count).

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Metadata saved alongside model checkpoints for DQN-specific state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CheckpointMetadata {
    pub epoch: usize,
    pub step_count: usize,
    pub epsilon: f32,
    pub best_reward: f32,
    pub avg_reward_10: f32,
    pub timestamp: String,
}

impl CheckpointMetadata {
    pub fn new(
        epoch: usize,
        step_count: usize,
        epsilon: f32,
        best_reward: f32,
        avg_reward_10: f32,
    ) -> Self {
        Self {
            epoch,
            step_count,
            epsilon,
            best_reward,
            avg_reward_10,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Helper functions for checkpoint management using Burn's recorders.
///
/// Uses Burn's `NamedMpkFileRecorder` for model persistence, wrapped with
/// DQN-specific metadata storage.
pub struct DQNCheckpointHelper;

impl DQNCheckpointHelper {
    /// Save model with metadata using Burn's recorder.
    ///
    /// # Arguments
    /// * `model` - Model to save (must implement Module)
    /// * `directory` - Save directory
    /// * `name` - Base name for checkpoint
    /// * `epoch` - Training epoch/episode
    /// * `metadata` - DQN-specific metadata
    pub fn save<B: Backend, M: Module<B>>(
        model: &M,
        directory: impl AsRef<Path>,
        name: &str,
        epoch: usize,
        metadata: &CheckpointMetadata,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let directory = directory.as_ref();
        std::fs::create_dir_all(directory)?;

        // Create fresh recorder instance for each save
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

        // Burn's NamedMpkFileRecorder expects path WITHOUT extension
        // It will append .mpk automatically
        // Format: "{name}-{epoch}" becomes "{name}-{epoch}.mpk"
        let file_path = directory.join(format!("{}-{}", name, epoch));

        tracing::debug!(
            "Attempting to save model to: {} (epoch {})",
            file_path.display(),
            epoch
        );

        // Save model using Burn's recorder
        model
            .clone()
            .save_file(&file_path, &recorder)
            .map_err(|e| format!("Failed to save model to {}: {:?}", file_path.display(), e))?;

        // Verify the .mpk file was created
        let mpk_path = directory.join(format!("{}-{}.mpk", name, epoch));
        if !mpk_path.exists() {
            // Try to list what files ARE in the directory for debugging
            let files: Vec<_> = std::fs::read_dir(directory)
                .map_err(|e| format!("Failed to read directory: {:?}", e))?
                .filter_map(|e| e.ok())
                .map(|e| e.path().display().to_string())
                .collect();

            return Err(format!(
                "Model file was not created at: {}. Files in directory: {:?}",
                mpk_path.display(),
                files
            )
            .into());
        }

        // Save metadata
        let meta_path = directory.join(format!("{}-{}.json", name, epoch));
        let json = serde_json::to_string_pretty(metadata)?;
        std::fs::write(&meta_path, json)?;

        // Verify metadata file was created
        if !meta_path.exists() {
            return Err(format!("Metadata file was not created: {}", meta_path.display()).into());
        }

        tracing::info!(
            "Checkpoint saved successfully: {} (epoch {}) - verified files exist",
            name,
            epoch
        );

        // Force sync to ensure files are visible
        if let Err(e) = std::fs::File::open(&mpk_path) {
            tracing::warn!("Could not re-open checkpoint file for verification: {}", e);
        }

        Ok(())
    }

    /// Load model with metadata using Burn's recorder.
    ///
    /// # Arguments
    /// * `directory` - Checkpoint directory
    /// * `name` - Base name for checkpoint
    /// * `epoch` - Checkpoint epoch to load
    /// * `device` - Device to load onto
    ///
    /// # Returns
    /// Tuple of (model, metadata)
    pub fn load<B: Backend, M: Module<B>>(
        directory: impl AsRef<Path>,
        name: &str,
        epoch: usize,
        device: &B::Device,
        config: impl FnOnce() -> M,
    ) -> Result<(M, CheckpointMetadata), Box<dyn std::error::Error>> {
        let directory = directory.as_ref();

        // Create recorder and path
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        let file_path = directory.join(format!("{}-{}", name, epoch));

        // Load model using Burn's recorder
        let model = config()
            .load_file(&file_path, &recorder, device)
            .map_err(|e| format!("Failed to load model: {:?}", e))?;

        // Load metadata
        let meta_path = directory.join(format!("{}-{}.json", name, epoch));
        let metadata = if meta_path.exists() {
            let json = std::fs::read_to_string(&meta_path)?;
            serde_json::from_str(&json)?
        } else {
            tracing::warn!("No metadata found for epoch {}, using defaults", epoch);
            CheckpointMetadata::default()
        };

        tracing::info!(
            "Checkpoint loaded: {} (epoch {})",
            file_path.display(),
            epoch
        );
        Ok((model, metadata))
    }

    /// Delete checkpoint files
    pub fn delete(
        directory: impl AsRef<Path>,
        name: &str,
        epoch: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let directory = directory.as_ref();

        // Delete model file
        let model_path = directory.join(format!("{}-{}.mpk", name, epoch));
        if model_path.exists() {
            std::fs::remove_file(&model_path)?;
        }

        // Delete metadata file
        let meta_path = directory.join(format!("{}-{}.json", name, epoch));
        if meta_path.exists() {
            std::fs::remove_file(&meta_path)?;
        }

        Ok(())
    }
}
