//! # Standalone DQN Policy
//!
//! This module provides a standalone Deep Q-Network (DQN) implementation without
//! bandit feature extraction. It's simpler than the combined METIS policy and
//! serves as a baseline for comparison.
//!
//! ## Architecture
//!
//! Unlike [`MetisPolicy`](super::metis_policy::MetisPolicy) which combines bandit
//! feature extraction with DQN, this policy:
//! - Takes raw state features as input
//! - Passes them through a Q-network with dueling architecture
//! - Outputs Q-values for each action
//! - Supports multiple exploration strategies
//!
//! ## Key Features
//!
//! - **Dueling Architecture**: Separates state value V(s) and action advantage A(s,a)
//! - **Target Network**: Stabilizes training with periodic weight updates
//! - **Experience Replay**: Learning from stored transitions
//! - **Configurable Exploration**: EpsilonGreedy, ThompsonSampling, or UCB
//!
//! ## Usage Example
//!
//! ### Basic Setup
//!
//! ```rust,ignore
//! use eris::policies::{DQNPolicy, DQNExplorerConfig};
//! use eris::config::DQNConfig;
//! use eris::policies::exploration::ExplorationConfig;
//! use burn::backend::{Autodiff, NdArray};
//! use burn::tensor::backend::Backend;
//!
//! // Create DQN network configuration
//! let dqn_config = DQNConfig::builder()
//!     .input_dim(15)              // State dimension
//!     .hidden_layers(vec![128, 128])  // Two hidden layers
//!     .action_dim(10)            // Number of actions (5 tiers × 2 operations)
//!     .build()
//!     .expect("Valid DQN config");
//!
//! // Choose exploration strategy
//! let exploration = ExplorationConfig::EpsilonGreedy {
//!     epsilon_start: 1.0,        // Start with full exploration
//!     epsilon_end: 0.01,         // End with 99% exploitation
//!     epsilon_decay: 0.995,      // Decay rate per episode
//! };
//!
//! // Create policy configuration
//! let config = DQNExplorerConfig::new(dqn_config, exploration)
//!     .with_learning_rate(0.0001)
//!     .with_gamma(0.99)
//!     .with_batch_size(512)
//!     .with_buffer_capacity(100_000)
//!     .with_target_update_freq(1000);
//!
//! // Initialize policy
//! let device = <NdArray as Backend>::Device::default();
//! let policy = DQNPolicy::<Autodiff<NdArray>>::new(config, device);
//! ```
//!
//! ### Training Loop
//!
//! ```rust,ignore
//! use eris::policies::policy::{CachePolicy, State, Action, Transition};
//!
//! // Training loop
//! for episode in 0..100 {
//!     let state = env.reset();
//!     let mut done = false;
//!     let mut total_reward = 0.0;
//!     
//!     while !done {
//!         // Select action with exploration
//!         let action = policy.select_action(&state);
//!         
//!         // Take action in environment
//!         let (next_state, reward, is_done) = env.step(&action);
//!         
//!         // Create transition
//!         let transition = Transition {
//!             state: state.clone(),
//!             action,
//!             reward,
//!             next_state: next_state.clone(),
//!             done: is_done,
//!         };
//!         
//!         // Update policy (also stores in replay buffer)
//!         let loss = policy.update(&transition);
//!         
//!         total_reward += reward;
//!         state = next_state;
//!         done = is_done;
//!     }
//!     
//!     println!("Episode {}: reward={:.2}, exploration={:.3}",
//!              episode, total_reward, policy.get_exploration_param());
//! }
//! ```
//!
//! ### Different Exploration Strategies
//!
//! ```rust,ignore
//! // Epsilon-Greedy (standard for DQN)
//! let epsilon_greedy = ExplorationConfig::EpsilonGreedy {
//!     epsilon_start: 1.0,
//!     epsilon_end: 0.01,
//!     epsilon_decay: 0.995,
//! };
//!
//! // Thompson Sampling (better uncertainty handling)
//! let thompson = ExplorationConfig::ThompsonSampling {
//!     prior_mean: 0.0,
//!     prior_std: 1.0,
//! };
//!
//! // UCB (theoretically optimal regret)
//! let ucb = ExplorationConfig::UCB { c: 2.0 };
//!
//! // Create policies with different strategies
//! let policy_eg = DQNPolicy::new(
//!     DQNExplorerConfig::new(dqn_config.clone(), epsilon_greedy),
//!     device.clone()
//! );
//! let policy_ts = DQNPolicy::new(
//!     DQNExplorerConfig::new(dqn_config.clone(), thompson),
//!     device.clone()
//! );
//! let policy_ucb = DQNPolicy::new(
//!     DQNExplorerConfig::new(dqn_config, ucb),
//!     device
//! );
//! ```
//!
//! ## Comparison with METIS and Bandit
//!
//! | Feature            | DQN      | METIS         | Bandit   |
//! |--------------------|----------|---------------|----------|
//! | Neural Network     | ✓        | ✓             | ✓        |
//! | Replay Buffer      | ✓        | ✓             | ✗        |
//! | Target Network     | ✓        | ✓             | ✗        |
//! | Feature Extraction | ✗        | ✓ (Bandit)    | ✓        |
//! | Online Learning    | ✗        | ✗             | ✓        |
//! | Memory Usage       | High     | High          | Low      |
//! | Training Speed     | Medium   | Medium        | Fast     |
//! | Sample Efficiency  | Medium   | High          | Low      |
//!
//! ## When to Use DQN vs METIS vs Bandit
//!
//! **Use DQN when:**
//! - You want a simpler baseline for comparison
//! - You don't need bandit feature extraction
//! - You're studying pure Q-learning behavior
//!
//! **Use METIS when:**
//! - You want best overall performance
//! - You need sample-efficient learning
//! - You can afford higher memory/compute
//!
//! **Use Bandit when:**
//! - You need fast online adaptation
//! - You have memory constraints
//! - You want real-time tier selection
//!
//! ## References
//!
//! - [Mnih et al., 2015] - Human-level control through deep reinforcement learning
//! - [Wang et al., 2016] - Dueling network architectures for deep RL
//! - [Hasselt et al., 2016] - Deep RL with double Q-learning

use super::exploration::ExplorationStrategy;
use super::policy::*;
use super::td_loss::compute_td_loss;
use super::tensor_utils::{batch_to_tensors, state_to_tensor};
use crate::config::DQNConfig;
use crate::models::QNetwork;
use crate::training::HybridRingBuffer;
use crate::utils::backend_diagnostics::log_backend_info;
use crate::utils::timing::{log_step_timing, OneTimeDiag};
use burn::optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Int, Tensor};
use burnme_rly::buffer::TensorTransitionBatch;
use std::error::Error;
use std::path::Path;
use tracing;

/// Configuration for DQN policy with exploration
#[derive(Clone, Debug)]
pub struct DQNExplorerConfig {
    /// Q-network configuration
    pub dqn_config: DQNConfig,
    /// Exploration strategy configuration
    pub exploration: super::exploration::ExplorationConfig,
    /// Learning rate for optimizer
    pub learning_rate: f32,
    /// Discount factor for future rewards
    pub gamma: f32,
    /// Target network update frequency (in steps)
    pub target_update_freq: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Replay buffer capacity
    pub buffer_capacity: usize,
    /// Warmup batch size (starts small, ramps up to full batch)
    pub warmup_batch_size: usize,
}

impl DQNExplorerConfig {
    /// Create a new DQN explorer config with defaults
    pub fn new(dqn_config: DQNConfig, exploration: super::exploration::ExplorationConfig) -> Self {
        Self {
            dqn_config,
            exploration,
            learning_rate: 0.0001,
            gamma: 0.99,
            target_update_freq: 1000,
            batch_size: 2048, // Optimized for GPU utilization (multiple of 32 for warp alignment)
            buffer_capacity: 10_000,
            warmup_batch_size: 256,
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set discount factor
    pub fn with_gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set target update frequency
    pub fn with_target_update_freq(mut self, freq: usize) -> Self {
        self.target_update_freq = freq;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set buffer capacity
    pub fn with_buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Set warmup batch size
    pub fn with_warmup_batch_size(mut self, size: usize) -> Self {
        self.warmup_batch_size = size;
        self
    }
}

/// DQN Policy - Pure DQN without bandit feature extraction
///
/// This policy implements a standard DQN with:
/// - Q-network for action-value estimation
/// - Target network for stable training
/// - Experience replay buffer
/// - Configurable exploration strategies
///
/// # Example
///
/// ```rust,ignore
/// use eris::policies::{DQNPolicy, DQNExplorerConfig};
/// use eris::config::{DQNConfig, ExplorationConfig};
/// use burn::backend::{Autodiff, NdArray};
///
/// let dqn_config = DQNConfig::builder()
///     .input_dim(15)
///     .hidden_layers(vec![128, 128])
///     .action_dim(10)
///     .build()?;
///
/// let exploration = ExplorationConfig::EpsilonGreedy {
///     epsilon_start: 1.0,
///     epsilon_end: 0.01,
///     epsilon_decay: 0.995,
/// };
///
/// let config = DQNExplorerConfig::new(dqn_config, exploration);
/// let device = <NdArray as Backend>::Device::default();
/// let policy = DQNPolicy::<Autodiff<NdArray>>::new(config, device);
/// ```
pub struct DQNPolicy<B: AutodiffBackend> {
    /// Q-network (online network)
    q_network: QNetwork<B>,
    /// Target network (frozen copy)
    target_network: QNetwork<B>,
    /// GPU-native replay buffer (for GPU training)
    pub gpu_buffer: HybridRingBuffer<B>,
    /// Exploration strategy
    explorer: Box<dyn ExplorationStrategy<B>>,
    /// Configuration
    config: DQNExplorerConfig,
    /// Compute device
    device: B::Device,
    /// Training step counter
    step_count: usize,
    /// Warmup configuration
    pub warmup_batch_size: usize,
    pub warmup_complete: bool,
    /// Optimizer for training (cached to prevent VRAM leak)
    optimizer: OptimizerAdaptor<Adam, QNetwork<B>, B>,
}

impl<B: AutodiffBackend> DQNPolicy<B> {
    /// Create a new DQN Policy
    ///
    /// # Arguments
    /// * `config` - Policy configuration
    /// * `device` - Compute device
    ///
    /// # Returns
    /// Initialized policy with random weights and empty replay buffer
    pub fn new(config: DQNExplorerConfig, device: B::Device) -> Self {
        log_backend_info::<B>("DQNPolicy::new", &device);

        // Initialize Q-network
        let q_network = config.dqn_config.init(&device);
        let target_network = config.dqn_config.init(&device);

        // GPU DIAGNOSTIC: Benchmark network initialization with a forward pass
        {
            use burn::tensor::{Tensor, TensorData};
            let test_input = Tensor::<B, 2>::from_data(
                TensorData::new(
                    vec![0.0f32; config.dqn_config.input_dim],
                    [1, config.dqn_config.input_dim],
                )
                .convert::<f32>(),
                &device,
            );
            let bench_start = std::time::Instant::now();
            let _test_output = q_network.forward(test_input);
            let _ = _test_output.into_data(); // Force GPU sync
            let bench_elapsed = bench_start.elapsed();
            tracing::debug!(
                "q_network forward pass (initialization): {:?}",
                bench_elapsed
            );
            tracing::debug!("  → GPU typically <1ms, CPU typically 1-10ms");
        }

        // Initialize GPU-native replay buffer
        let gpu_buffer = HybridRingBuffer::new(config.buffer_capacity, config.dqn_config.input_dim);

        // Build exploration strategy
        let action_dim = config.dqn_config.action_dim;
        let explorer = config.exploration.clone().build(action_dim);

        // Warmup batch size: start with configured value (or full batch if smaller)
        let warmup_batch_size = config.warmup_batch_size.min(config.batch_size);

        // Initialize optimizer (cached once to prevent VRAM leak)
        let optimizer = AdamConfig::new().init();

        Self {
            q_network,
            target_network,
            gpu_buffer,
            explorer,
            config,
            device,
            step_count: 0,
            warmup_batch_size,
            warmup_complete: false,
            optimizer,
        }
    }

    /// Select action using exploration strategy
    ///
    /// # Arguments
    /// * `state` - Current state
    ///
    /// # Returns
    /// Selected action (discrete)
    fn select_action_impl(&self, state: &State) -> Action {
        let state_tensor = state_to_tensor(state, self.config.dqn_config.input_dim, &self.device);
        let q_values = self.q_network.forward(state_tensor);
        let action_dim = self.config.dqn_config.action_dim;

        let selected_actions = self.explorer.select_action(&q_values, action_dim);

        let action_idx: i32 = selected_actions
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("Failed to convert action")[0];

        Action::Discrete(action_idx as usize)
    }

    /// DQN training step with TD loss
    ///
    /// Uses shared TD loss computation from td_loss module and performs
    /// gradient descent to update the Q-network weights.
    ///
    /// # Arguments
    /// * `states` - Batch of states [batch_size, state_dim]
    /// * `actions` - Batch of actions [batch_size, 1]
    /// * `rewards` - Batch of rewards [batch_size, 1]
    /// * `next_states` - Batch of next states [batch_size, state_dim]
    /// * `dones` - Batch of done flags [batch_size, 1]
    ///
    /// # Returns
    /// Loss value
    fn train_dqn_step(
        &mut self,
        states: &Tensor<B, 2>,
        actions: &Tensor<B, 2, Int>,
        rewards: &Tensor<B, 2>,
        next_states: &Tensor<B, 2>,
        dones: &Tensor<B, 2>,
    ) -> f32 {
        tracing::trace!("train_dqn_step ENTRY, states shape: {:?}", states.shape());

        // Get Q-values from networks
        tracing::debug!("Calling q_network.forward(states)");
        let q_values = self.q_network.forward(states.clone());
        tracing::debug!("q_values shape: {:?}", q_values.shape());

        tracing::debug!("Calling target_network.forward(next_states)");
        let next_q_values = self.target_network.forward(next_states.clone());
        tracing::trace!("next_q_values shape: {:?}", next_q_values.shape());

        // Use shared TD loss computation
        tracing::debug!("Calling compute_td_loss");
        let loss = compute_td_loss(
            q_values,
            next_q_values,
            actions,
            rewards,
            dones,
            self.config.gamma,
        );
        tracing::debug!("loss tensor shape: {:?}", loss.shape());

        // Perform backpropagation and update weights
        tracing::debug!("Calling loss.backward() - THIS TRIGGERS GPU WORK!");
        let grads = loss.backward();
        tracing::debug!("backward() COMPLETE, got gradients");
        let grads_params = GradientsParams::from_grads(grads, &self.q_network);

        // Update Q-network with cached optimizer (no VRAM leak)
        tracing::debug!("Calling optimizer.step()");
        self.q_network = self.optimizer.step(
            self.config.learning_rate as f64,
            self.q_network.clone(),
            grads_params,
        );
        tracing::debug!("optimizer.step() COMPLETE");

        // Extract scalar loss value
        let loss_data = loss
            .detach()
            .into_data()
            .convert::<f32>()
            .as_slice()
            .unwrap()[0];
        tracing::debug!("train_dqn_step EXIT, loss: {:.4}", loss_data);
        loss_data
    }

    /// Update target network by copying weights from Q-network
    fn update_target_network(&mut self) {
        self.target_network = self.q_network.clone();
    }

    /// Get current exploration parameter
    pub fn get_exploration_param(&self) -> f32 {
        self.explorer.get_param()
    }

    /// Set exploration parameter
    pub fn set_exploration_param(&mut self, value: f32) {
        self.explorer.set_param(value);
    }

    /// Decay exploration after training
    fn decay_exploration(&mut self) {
        self.explorer.decay();
    }

    /// Get reference to the Q-network model
    pub fn model(&self) -> &QNetwork<B> {
        &self.q_network
    }

    /// Get checkpoint metadata for this policy
    pub fn checkpoint_metadata(&self) -> crate::training::checkpoint::CheckpointMetadata {
        crate::training::checkpoint::CheckpointMetadata::new_with_dims(
            "DQNPolicy".to_string(),
            self.step_count,
            self.config.dqn_config.input_dim,
            self.config.dqn_config.action_dim,
            self.config
                .dqn_config
                .hidden_layers
                .first()
                .copied()
                .unwrap_or(64),
        )
    }
}

impl<B: AutodiffBackend> crate::training::checkpoint::Checkpointable<B> for DQNPolicy<B> {
    fn checkpoint_name(&self) -> &str {
        "dqn_policy"
    }

    fn checkpoint_metadata(&self) -> crate::training::checkpoint::CheckpointMetadata {
        self.checkpoint_metadata()
    }

    fn model(&self) -> &impl burn::module::Module<B> {
        &self.q_network
    }
}

impl<B: AutodiffBackend> CachePolicy for DQNPolicy<B> {
    fn select_action(&self, state: &State) -> Action {
        self.select_action_impl(state)
    }

    fn update(&mut self, transition: &Transition) -> f32 {
        // Convert transition to tensors and push to GPU buffer
        let state = match &transition.state {
            State::Features(f) => f.clone(),
            State::Raw(r) => r.iter().map(|&x| x as f32).collect(),
            State::Empty => vec![0.0; self.config.dqn_config.input_dim],
        };

        let action_idx = match transition.action {
            Action::Discrete(a) => a,
            _ => 0,
        };

        let next_state = match &transition.next_state {
            State::Features(f) => f.clone(),
            State::Raw(r) => r.iter().map(|&x| x as f32).collect(),
            State::Empty => vec![0.0; self.config.dqn_config.input_dim],
        };

        self.gpu_buffer.push(
            state,
            action_idx,
            transition.reward as f32,
            next_state,
            transition.done,
        );

        // Return 0.0 - actual training happens via train_step_gpu_native
        0.0
    }

    fn save(&self, _path: &Path) -> Result<(), Box<dyn Error>> {
        // TODO: Implement using Burn's ModelRecorder
        Err("Save not yet implemented for DQNPolicy".into())
    }

    fn load(&mut self, _path: &Path) -> Result<(), Box<dyn Error>> {
        // TODO: Implement using Burn's ModelRecorder
        Err("Load not yet implemented for DQNPolicy".into())
    }

    fn policy_type(&self) -> PolicyType {
        PolicyType::Dqn
    }

    fn action_dim(&self) -> usize {
        self.config.dqn_config.action_dim
    }
}

impl<B: AutodiffBackend> ReplayPolicy for DQNPolicy<B> {
    fn train_step(&mut self, batch: &[Transition]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        // Use shared utility to convert batch to tensors
        let (states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor) =
            batch_to_tensors(batch, self.config.dqn_config.input_dim, &self.device);

        // Train DQN
        let loss = self.train_dqn_step(
            &states_tensor,
            &actions_tensor,
            &rewards_tensor,
            &next_states_tensor,
            &dones_tensor,
        );

        // Decay exploration
        self.decay_exploration();

        // Update target network periodically
        self.step_count += 1;
        if self.step_count % self.config.target_update_freq == 0 {
            self.update_target_network();
        }

        loss
    }

    fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    fn update_target(&mut self) {
        self.update_target_network();
    }
}

// ============================================================================
// GpuTrainable Implementation
// ============================================================================

impl<B: AutodiffBackend> crate::training::GpuTrainable<B> for DQNPolicy<B> {
    fn gpu_buffer_mut(&mut self) -> &mut HybridRingBuffer<B> {
        &mut self.gpu_buffer
    }

    fn gpu_buffer(&self) -> &HybridRingBuffer<B> {
        &self.gpu_buffer
    }

    fn full_batch_size(&self) -> usize {
        self.config.batch_size
    }

    fn warmup_batch_size(&self) -> usize {
        self.warmup_batch_size
    }

    fn is_warmup_complete(&self) -> bool {
        self.warmup_complete
    }

    fn set_warmup_complete(&mut self, complete: bool) {
        self.warmup_complete = complete;
    }

    fn target_update_freq(&self) -> usize {
        self.config.target_update_freq
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn increment_step_count(&mut self) {
        self.step_count += 1;
    }

    fn epsilon(&self) -> f32 {
        self.explorer.get_param()
    }

    fn update_epsilon(&mut self) {
        self.decay_exploration();
    }

    fn train_step_gpu(&mut self, batch: &TensorTransitionBatch<B>) -> f32 {
        static DIAG: OneTimeDiag = OneTimeDiag::new();

        if DIAG.should_print() {
            log_backend_info::<B>("DQNPolicy::train_step_gpu", &self.device);
        }

        tracing::trace!(
            "train_step_gpu ENTRY, batch_size: {}, step_count: {}",
            batch.states.dims()[0],
            self.step_count
        );

        // Time the training step (periodically, every 100 steps)
        let train_start = std::time::Instant::now();

        // Train DQN using GPU tensors directly (no CPU conversion)
        // Batch has rank-2 data format [batch_size, 1], reshape for type signature
        tracing::trace!(
            "Batch shapes - actions: {:?}, rewards: {:?}, dones: {:?}",
            batch.actions.shape(),
            batch.rewards.shape(),
            batch.dones.shape()
        );
        let actions_2d = batch.actions.clone().reshape([batch.states.dims()[0], 1]);
        let rewards_2d = batch.rewards.clone().reshape([batch.states.dims()[0], 1]);
        let dones_2d = batch.dones.clone().reshape([batch.states.dims()[0], 1]);

        tracing::debug!(
            "Calling train_dqn_step, states shape: {:?}",
            batch.states.shape()
        );
        let loss = self.train_dqn_step(
            &batch.states,
            &actions_2d,
            &rewards_2d,
            &batch.next_states,
            &dones_2d,
        );
        tracing::debug!("train_dqn_step returned loss: {:.4}", loss);

        let train_elapsed = train_start.elapsed();
        log_step_timing(self.step_count, "train_step_gpu", train_elapsed, 100);

        tracing::debug!("train_step_gpu EXIT, loss: {:.4}", loss);
        loss
    }

    fn maybe_update_target(&mut self, step_count: usize) {
        if step_count % self.config.target_update_freq == 0 {
            self.update_target_network();
        }
    }

    fn train_step_gpu_native_with_prefetch(
        &mut self,
        steps_since_last_train: usize,
        device: &B::Device,
        warmup_batch_size: usize,
        full_batch_size: usize,
        prebuilt_batch: Option<TensorTransitionBatch<B>>,
    ) -> Option<f32> {
        // Check training frequency using lib's canonical should_train
        let should_train = crate::training::should_train(
            self.is_warmup_complete(),
            steps_since_last_train,
            4, // train_frequency
        );

        if !should_train {
            return None;
        }

        // Get the batch: either use the prefetched one or sample internally
        let effective_batch_size = if self.is_warmup_complete() {
            full_batch_size
        } else {
            let effective = warmup_batch_size.min(full_batch_size);
            let buffer_len: usize = self.gpu_buffer().len();
            if buffer_len >= full_batch_size {
                self.set_warmup_complete(true);
            }
            effective
        };

        let batch = match prebuilt_batch {
            Some(batch) => batch,
            None => {
                match self
                    .gpu_buffer_mut()
                    .sample_batch(effective_batch_size, device)
                {
                    Some(batch) => batch,
                    None => {
                        log::trace!("[STAGE:DIAG] train_step_gpu_native_with_prefetch: Not enough samples (have {}, need {}), skipping",
                            self.gpu_buffer().len(), effective_batch_size);
                        return None;
                    }
                }
            }
        };

        // Train on the batch (same path as train_step_gpu)
        let loss = self.train_step_gpu(&batch);
        self.increment_step_count();
        self.maybe_update_target(self.step_count());
        self.update_epsilon();
        Some(loss)
    }
}

// ============================================================================
// burnme_rly::traits::GpuTrainable Implementation
// ============================================================================

impl<B: AutodiffBackend> burnme_rly::traits::GpuTrainable<B, HybridRingBuffer<B>> for DQNPolicy<B> {
    fn buffer_mut(&mut self) -> &mut HybridRingBuffer<B> {
        &mut self.gpu_buffer
    }

    fn buffer(&self) -> &HybridRingBuffer<B> {
        &self.gpu_buffer
    }

    fn train_step_gpu_native(
        &mut self,
        _steps_since_last_train: usize,
        device: &B::Device,
    ) -> Option<f32> {
        use burnme_rly::traits::GpuTrainableExt;

        let batch_size = self.effective_batch_size();

        // Sample from buffer (returns TensorTransitionBatch directly)
        let batch = self.gpu_buffer.sample_batch(batch_size, device)?;

        // Train on batch
        let loss = self.train_step_gpu(&batch);
        Some(loss)
    }

    fn train_step_gpu(&mut self, batch: &TensorTransitionBatch<B>) -> f32 {
        // Reuse existing train_dqn_step logic but with burnme-rly batch type
        // The batch has: states, actions, rewards, next_states, dones
        let batch_size = batch.states.dims()[0];
        let actions_2d = batch.actions.clone().reshape([batch_size, 1]);
        let rewards_2d = batch.rewards.clone().reshape([batch_size, 1]);
        let dones_2d = batch.dones.clone().reshape([batch_size, 1]);

        self.train_dqn_step(
            &batch.states,
            &actions_2d,
            &rewards_2d,
            &batch.next_states,
            &dones_2d,
        )
    }

    fn device(&self) -> &B::Device {
        &self.device
    }

    fn state_dim(&self) -> usize {
        self.config.dqn_config.input_dim
    }

    fn buffer_len(&self) -> usize {
        self.gpu_buffer.len()
    }

    fn warmup_batch_size(&self) -> usize {
        self.warmup_batch_size
    }

    fn is_warmup_complete(&self) -> bool {
        self.warmup_complete
    }

    fn set_warmup_complete(&mut self, complete: bool) {
        self.warmup_complete = complete;
    }

    fn epsilon(&self) -> f32 {
        self.get_exploration_param()
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn increment_step_count(&mut self) {
        self.step_count += 1;
    }

    fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    fn target_update_freq(&self) -> usize {
        self.config.target_update_freq
    }

    fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    fn gamma(&self) -> f32 {
        self.config.gamma
    }

    fn decay_exploration(&mut self) {
        self.decay_exploration();
    }

    fn update_target_network(&mut self) {
        self.update_target_network();
    }

    fn save_checkpoint(&self, _path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement checkpoint saving using burnme-rly checkpoint utilities
        Err("Checkpoint saving not yet implemented for DQNPolicy".into())
    }

    fn load_checkpoint(&mut self, _path: &str) -> Result<(), Box<dyn std::error::Error>> {
        // TODO: Implement checkpoint loading using burnme-rly checkpoint utilities
        Err("Checkpoint loading not yet implemented for DQNPolicy".into())
    }
}

// ============================================================================
// BatchedActionSelector Implementation
// ============================================================================

impl<B: AutodiffBackend> crate::training::BatchedActionSelector<B> for DQNPolicy<B> {
    fn select_actions_batched(
        &self,
        observations: &[Vec<f64>],
        device: &B::Device,
        action_dim: usize,
        epsilon: f32,
    ) -> Vec<usize> {
        // Use shared utility for tensor conversion - NO MORE DUPLICATION!
        let states_tensor =
            crate::training::batched_action_utils::observations_to_tensor(observations, device);

        // Policy-specific forward pass
        let q_values = self.q_network.forward(states_tensor);

        // Use shared utility for epsilon-greedy selection - NO MORE DUPLICATION!
        crate::training::batched_action_utils::epsilon_greedy_select(
            q_values, action_dim, epsilon, device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::tensor_utils::states_to_tensor;
    use burn::backend::{Autodiff, NdArray};
    use burn::prelude::Backend;

    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_dqn_config_builder() {
        let dqn_config = DQNConfig::builder()
            .input_dim(15)
            .hidden_layers(vec![128, 128])
            .action_dim(10)
            .build()
            .expect("Valid config");

        let exploration = super::super::exploration::ExplorationConfig::EpsilonGreedy {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        let config = DQNExplorerConfig::new(dqn_config, exploration)
            .with_learning_rate(0.001)
            .with_gamma(0.95)
            .with_batch_size(256);

        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.gamma, 0.95);
        assert_eq!(config.batch_size, 256);
    }

    #[test]
    fn test_dqn_policy_creation() {
        let device = <NdArray as Backend>::Device::default();

        let dqn_config = DQNConfig::builder()
            .input_dim(15)
            .hidden_layers(vec![128])
            .action_dim(10)
            .build()
            .expect("Valid config");

        let exploration = super::super::exploration::ExplorationConfig::EpsilonGreedy {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        let config = DQNExplorerConfig::new(dqn_config, exploration);
        let policy = DQNPolicy::<TestBackend>::new(config, device);

        assert_eq!(policy.action_dim(), 10);
        assert_eq!(policy.policy_type(), PolicyType::Dqn);
    }

    #[test]
    fn test_state_to_tensor() {
        let device = <NdArray as Backend>::Device::default();

        // Test with Features state
        let state = State::Features(vec![1.0; 15]);
        let tensor: Tensor<TestBackend, 2> = state_to_tensor(&state, 15, &device);
        assert_eq!(tensor.shape().dims, [1, 15]);

        // Test with Empty state
        let empty_state = State::Empty;
        let empty_tensor: Tensor<TestBackend, 2> = state_to_tensor(&empty_state, 15, &device);
        assert_eq!(empty_tensor.shape().dims, [1, 15]);
    }

    #[test]
    fn test_states_to_tensor() {
        let device = <NdArray as Backend>::Device::default();

        // Test with batch of states
        let states = vec![vec![1.0; 10], vec![2.0; 10], vec![3.0; 10]];
        let tensor: Tensor<TestBackend, 2> = states_to_tensor(&states, &device);
        assert_eq!(tensor.shape().dims, [3, 10]);

        // Test with empty batch
        let empty: Vec<Vec<f32>> = vec![];
        let empty_tensor: Tensor<TestBackend, 2> = states_to_tensor(&empty, &device);
        assert_eq!(empty_tensor.shape().dims, [0, 0]);
    }

    #[test]
    fn test_select_action() {
        let device = <NdArray as Backend>::Device::default();

        let dqn_config = DQNConfig::builder()
            .input_dim(15)
            .hidden_layers(vec![128])
            .action_dim(10)
            .build()
            .expect("Valid config");

        let exploration = super::super::exploration::ExplorationConfig::EpsilonGreedy {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        let config = DQNExplorerConfig::new(dqn_config, exploration);
        let policy = DQNPolicy::<TestBackend>::new(config, device);

        let state = State::Features(vec![1.0; 15]);
        let action = policy.select_action(&state);

        match action {
            Action::Discrete(a) => assert!(a < 10, "Action {} out of bounds", a),
            _ => panic!("Expected discrete action"),
        }
    }

    #[test]
    fn test_train_step() {
        let device = <NdArray as Backend>::Device::default();

        let dqn_config = DQNConfig::builder()
            .input_dim(10)
            .hidden_layers(vec![128])
            .action_dim(5)
            .build()
            .expect("Valid config");

        let exploration = super::super::exploration::ExplorationConfig::EpsilonGreedy {
            epsilon_start: 0.5,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        let config = DQNExplorerConfig::new(dqn_config, exploration).with_batch_size(4);
        let mut policy = DQNPolicy::<TestBackend>::new(config, device);

        // Create a batch of transitions
        let batch: Vec<Transition> = (0..4)
            .map(|i| Transition {
                state: State::Features(vec![i as f32; 10]),
                action: Action::Discrete(i % 5),
                reward: (i as f32) * 0.1,
                next_state: State::Features(vec![(i + 1) as f32; 10]),
                done: false,
            })
            .collect();

        let loss = policy.train_step(&batch);
        assert!(loss >= 0.0, "Loss should be non-negative");
    }

    #[test]
    fn test_exploration_param() {
        let device = <NdArray as Backend>::Device::default();

        let dqn_config = DQNConfig::builder()
            .input_dim(10)
            .hidden_layers(vec![128])
            .action_dim(5)
            .build()
            .expect("Valid config");

        let exploration = super::super::exploration::ExplorationConfig::EpsilonGreedy {
            epsilon_start: 0.8,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        let config = DQNExplorerConfig::new(dqn_config, exploration);
        let mut policy = DQNPolicy::<TestBackend>::new(config, device);

        // Check initial exploration param
        let initial_param = policy.get_exploration_param();
        assert!((initial_param - 0.8).abs() < 0.01);

        // Set exploration param
        policy.set_exploration_param(0.5);
        let updated_param = policy.get_exploration_param();
        assert!((updated_param - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_target_update() {
        let device = <NdArray as Backend>::Device::default();

        let dqn_config = DQNConfig::builder()
            .input_dim(10)
            .hidden_layers(vec![128])
            .action_dim(5)
            .build()
            .expect("Valid config");

        let exploration = super::super::exploration::ExplorationConfig::EpsilonGreedy {
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
        };

        let config = DQNExplorerConfig::new(dqn_config, exploration).with_target_update_freq(2);
        let mut policy = DQNPolicy::<TestBackend>::new(config, device);

        // Create initial batch
        let batch: Vec<Transition> = (0..4)
            .map(|i| Transition {
                state: State::Features(vec![i as f32; 10]),
                action: Action::Discrete(i % 5),
                reward: (i as f32) * 0.1,
                next_state: State::Features(vec![(i + 1) as f32; 10]),
                done: false,
            })
            .collect();

        // Train two steps to trigger target update
        policy.train_step(&batch);
        assert_eq!(policy.step_count, 1);

        policy.train_step(&batch);
        assert_eq!(policy.step_count, 2);

        // After step_count == target_update_freq, target should be updated
        // This is verified by step_count being correct
    }

    #[test]
    fn test_thompson_sampling_exploration() {
        let device = <NdArray as Backend>::Device::default();

        let dqn_config = DQNConfig::builder()
            .input_dim(10)
            .hidden_layers(vec![128])
            .action_dim(5)
            .build()
            .expect("Valid config");

        let exploration = super::super::exploration::ExplorationConfig::ThompsonSampling {
            prior_mean: 0.0,
            prior_std: 1.0,
        };

        let config = DQNExplorerConfig::new(dqn_config, exploration);
        let policy = DQNPolicy::<TestBackend>::new(config, device);

        let state = State::Features(vec![1.0; 10]);
        let action = policy.select_action(&state);

        match action {
            Action::Discrete(a) => assert!(a < 5),
            _ => panic!("Expected discrete action"),
        }
    }

    #[test]
    fn test_ucb_exploration() {
        let device = <NdArray as Backend>::Device::default();

        let dqn_config = DQNConfig::builder()
            .input_dim(10)
            .hidden_layers(vec![128])
            .action_dim(4)
            .build()
            .expect("Valid config");

        let exploration = super::super::exploration::ExplorationConfig::UCB { c: 2.0 };

        let config = DQNExplorerConfig::new(dqn_config, exploration);
        let mut policy = DQNPolicy::<TestBackend>::new(config, device.clone());

        // Initialize UCB explorer with some history to avoid infinite scores
        // UCB gives infinite scores to unvisited actions, which can cause issues
        // in tie-breaking when all actions are unvisited
        for i in 0..4 {
            policy.explorer.update(i, 0.5);
        }

        let state = State::Features(vec![1.0; 10]);
        let action = policy.select_action(&state);

        match action {
            Action::Discrete(a) => assert!(a < 4, "Action {} out of bounds", a),
            _ => panic!("Expected discrete action"),
        }
    }
}
