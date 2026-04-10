// AsyncDQNDataLoader removed - using Burn's MultiThreadDataLoader instead
// pub mod async_dataloader;
pub mod batched_action_utils;
pub mod burn_callbacks;
pub mod burn_dataloader;
pub mod burn_metrics;
pub mod burn_trainer;
pub mod checkpoint;
pub mod coordinator;
pub mod gpu_coordinator;
pub mod gpu_trainable;
pub mod hybrid_buffer;
pub mod mock_env;
pub mod replay_buffer;
pub mod ring_buffer;
pub mod tensor_buffer;
pub mod trainer;
pub mod transition_batcher;

// pub use async_dataloader::AsyncDQNDataLoader;
pub use batched_action_utils::{epsilon_greedy_select, observations_to_tensor};
pub use burn_callbacks::{EpsilonDecayCallback, RewardTrackingCallback, TargetUpdateCallback};
pub use burn_dataloader::{DQNBatch, DQNDataLoader};
pub use burn_metrics::{
    EpsilonInput, EpsilonMetric, MeanQInput, MeanQMetric, RewardInput, RewardMetric, TierInput,
    TierMetric, TierUtilizationInput, TierUtilizationMetric,
};
pub use burn_trainer::DQNTrainingOutput;
pub use checkpoint::CheckpointMetadata;
pub use coordinator::{train_agent, train_agent_burn, train_agent_with_metrics, TrainingResult};
pub use gpu_coordinator::{
    BatchedActionSelector, GpuTrainingCoordinator, TrainingMetrics, VecEnvironment,
};
pub use gpu_trainable::{should_train, train_step_with_warmup, GpuTrainable};
pub use hybrid_buffer::HybridRingBuffer;
pub use mock_env::{create_dummy_transition, fill_buffer, MockEnv};
pub use replay_buffer::{ReplayBuffer, Transition, TransitionBatch};
pub use ring_buffer::RingBuffer;
pub use tensor_buffer::{TensorRingBuffer, TensorTransitionBatch as GpuTransitionBatch};
pub use trainer::{CombinedAgent, TrainingConfig};
pub use transition_batcher::{TensorTransitionBatch, TransitionBatcher};
