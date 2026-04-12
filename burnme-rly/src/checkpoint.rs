//! Simple checkpoint system wrapping Burn's recorder.

use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::path::Path;

/// Simple checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub step_count: usize,
    pub epsilon: f32,
    pub episode: usize,
}

/// Validate checkpoint path - reject path traversal
fn validate_checkpoint_path(path: &Path) -> Result<(), Box<dyn Error>> {
    // Reject paths with .. components
    if path
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err("Path contains '..' (path traversal not allowed)".into());
    }
    // Reject absolute paths
    if path.is_absolute() {
        return Err("Absolute paths not allowed".into());
    }
    Ok(())
}

/// Save model checkpoint atomically (temp files + rename)
pub fn save_checkpoint<B, M>(
    model: &M,
    metadata: &CheckpointMetadata,
    path: &Path,
) -> Result<(), Box<dyn Error>>
where
    B: Backend,
    M: Module<B>,
{
    validate_checkpoint_path(path)?;
    let temp_path = path.with_extension("mpk.tmp");
    let temp_meta = path.with_extension("json.tmp");

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model.clone().save_file(&temp_path, &recorder)?;
    std::fs::write(&temp_meta, serde_json::to_string(metadata)?)?;

    std::fs::rename(&temp_path, path)?;
    std::fs::rename(&temp_meta, path.with_extension("json"))?;

    Ok(())
}

/// Load model checkpoint
pub fn load_checkpoint<B, M>(
    path: &Path,
    device: &B::Device,
    config: impl FnOnce() -> M,
) -> Result<(M, CheckpointMetadata), Box<dyn Error>>
where
    B: Backend,
    M: Module<B>,
{
    validate_checkpoint_path(path)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = config()
        .load_file(path, &recorder, device)
        .map_err(|e| Box::new(e) as Box<dyn Error>)?;
    let meta_path = path.with_extension("json");
    let metadata: CheckpointMetadata = serde_json::from_str(&std::fs::read_to_string(meta_path)?)?;
    Ok((model, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_metadata() {
        let meta = CheckpointMetadata {
            step_count: 100,
            epsilon: 0.5,
            episode: 10,
        };
        assert_eq!(meta.step_count, 100);
    }

    #[test]
    fn test_validate_path_rejects_parent_dir() {
        assert!(validate_checkpoint_path(Path::new("../etc/passwd")).is_err());
        assert!(validate_checkpoint_path(Path::new("foo/../../bar")).is_err());
    }

    #[test]
    fn test_validate_path_rejects_absolute() {
        assert!(validate_checkpoint_path(Path::new("/etc/passwd")).is_err());
        assert!(validate_checkpoint_path(Path::new("/tmp/checkpoint.mpk")).is_err());
    }

    #[test]
    fn test_validate_path_accepts_relative() {
        assert!(validate_checkpoint_path(Path::new("checkpoint.mpk")).is_ok());
        assert!(validate_checkpoint_path(Path::new("checkpoints/model.mpk")).is_ok());
    }
}
