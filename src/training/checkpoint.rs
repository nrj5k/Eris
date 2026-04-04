use serde::{Deserialize, Serialize};

/// Metadata saved alongside model checkpoints
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
