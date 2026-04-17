//! # burnme-rly
//!
//! Optimized GPU training pipeline for Burn-based reinforcement learning.
//!
//! ## Features
//!
//! - **GPU-native**: Zero-allocation experience replay with TensorRingBuffer
//! - **Backend-agnostic**: Works with CUDA, WGPU, CPU (NdArray)
//! - **Vectorized environments**: Train multiple environments in parallel
//! - **Warmup handling**: Automatic batch size ramp-up for stable training
//! - **Metis-based**: Extracted from proven high-performance training patterns
//!
//! ## Example
//!
//! ```rust,ignore
//! use burnme_rly::{
//!     GpuTrainingCoordinator, TrainingConfig, GpuTrainable,
//!     BatchedActionSelector, VecEnvironment
//! };
//! use burn::backend::Cuda;
//!
//! // Implement traits for your policy
//! impl<B: AutodiffBackend> GpuTrainable<B> for MyPolicy<B> { ... }
//! impl<B: AutodiffBackend> BatchedActionSelector<B> for MyPolicy<B> { ... }
//!
//! // Configure and run training
//! let config = TrainingConfig::new(1000, 500, 512)
//!     .with_warmup_batch_size(256);
//! let coordinator = GpuTrainingCoordinator::new(config);
//! let metrics = coordinator.run_training(
//!     &mut agent, &mut env, &device, "checkpoints"
//! )?;
//!
//! println!("Training complete! Avg reward: {}", metrics.avg_reward);
//! ```

pub mod buffer;
pub mod checkpoint;
pub mod coordinator;
pub mod diagnostics;
pub mod env;
pub mod loss;
pub mod models;
pub mod prefetch;
pub mod space;
pub mod trainers;
pub mod traits;
pub mod warmup;

// Re-export main types for convenient use
// TensorRingBuffer is deprecated but re-exported for backward compatibility
#[allow(deprecated)]
pub use buffer::{
    CpuRingBuffer, GpuRingBuffer, GpuTransitionBatch, HybridRingBuffer, TensorRingBuffer,
    TensorTransitionBatch, Transition,
};
pub use checkpoint::{
    load_checkpoint, save_checkpoint, CheckpointMetadata, CheckpointMetadataExt, Checkpointable,
    CHECKPOINT_VERSION,
};
pub use coordinator::{GpuTrainingCoordinator, TrainingConfig, TrainingMetrics};
pub use diagnostics::{log_backend_info, SimpleTimer};
pub use env::{Info, StepResult};
pub use loss::{
    compute_double_dqn_loss, compute_double_dqn_loss_rank2, compute_td_loss, compute_td_target,
};
pub use models::{
    ComposableModel, ComposeConfig, MetisV2Config, MetisV2Policy, ParallelCompose,
    SequentialCompose,
};
pub use prefetch::PrefetchBuffer;
pub use space::DiscreteSpace;
pub use trainers::{DQNTrainer, DQNTrainerConfig, MetisTrainer, MetisTrainerConfig};
pub use traits::{BatchedActionSelector, GpuTrainable, GpuTrainableExt, VecEnvironment};
pub use warmup::{should_train, train_step_with_warmup, train_step_with_warmup_config};

// Version info
pub const VERSION: &str = "0.1.0";

/// Initialize the logging backend for burnme-rly.
///
/// Call this once at program start to enable `RUST_LOG`-based filtering.
///
/// # Examples
///
/// ```no_run
/// // In your main.rs:
/// burnme_rly::init_logging();
/// // ... rest of your code
/// ```
///
/// Then run with:
/// ```bash
/// RUST_LOG=debug cargo run
/// RUST_LOG=burnme_rly=trace cargo run
/// ```
///
/// If not called, `log::info!` / `log::debug!` etc. are silent no-ops
/// (the default `log` crate behavior without a backend).
pub fn init_logging() {
    env_logger::init();
}
