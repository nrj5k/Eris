//! Catcher policy - DDPG Actor-Critic for continuous cache control
//!
//! Implements Deep Deterministic Policy Gradient (DDPG) with:
//! - Actor network: State → Continuous action (cache importance score in [-1, 1])
//! - Critic network: (State, Action) → Q-value
//! - Experience replay with target networks
//! - Soft target updates (τ parameter)

use super::policy::*;
use crate::training::checkpoint::{CheckpointMetadata, Checkpointable};
use crate::training::gpu_trainable::GpuTrainable;
use crate::training::tensor_buffer::TensorTransitionBatch;
use crate::training::HybridRingBuffer;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu, Tanh};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor, TensorData};
use std::error::Error;
use std::path::Path;

// ============================================================================
// Actor Network
// ============================================================================

/// Actor network: determines which data to cache
///
/// Maps state (128 address history) to continuous action (cache importance score)
/// Output is bounded to [-1, 1] using tanh activation
#[derive(Module, Debug)]
pub struct Actor<B: Backend> {
    layer_1: Linear<B>,
    layer_2: Linear<B>,
    layer_3: Linear<B>,
    activation: Relu,
    output_activation: Tanh,
}

#[derive(Config, Debug)]
pub struct ActorConfig {
    /// Input dimension (address history size)
    #[config(default = 128)]
    pub state_dim: usize,
    /// Hidden layer 1 dimension
    #[config(default = 256)]
    pub hidden_dim_1: usize,
    /// Hidden layer 2 dimension
    #[config(default = 128)]
    pub hidden_dim_2: usize,
    /// Output dimension (always 1 for continuous action)
    #[config(default = 1)]
    pub action_dim: usize,
    #[config(default = true)]
    pub bias: bool,
}

impl ActorConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Actor<B> {
        Actor {
            layer_1: LinearConfig::new(self.state_dim, self.hidden_dim_1)
                .with_bias(self.bias)
                .init(device),
            layer_2: LinearConfig::new(self.hidden_dim_1, self.hidden_dim_2)
                .with_bias(self.bias)
                .init(device),
            layer_3: LinearConfig::new(self.hidden_dim_2, self.action_dim)
                .with_bias(self.bias)
                .init(device),
            activation: Relu::new(),
            output_activation: Tanh::new(),
        }
    }
}

impl<B: Backend> Actor<B> {
    /// Forward pass: state → action
    ///
    /// # Arguments
    /// * `state` - Input tensor [batch_size, state_dim]
    ///
    /// # Returns
    /// * Action tensor [batch_size, action_dim] in range [-1, 1]
    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_1.forward(state);
        let x = self.activation.forward(x);
        let x = self.layer_2.forward(x);
        let x = self.activation.forward(x);
        let x = self.layer_3.forward(x);
        self.output_activation.forward(x)
    }
}

// ============================================================================
// Critic Network
// ============================================================================

/// Critic network: evaluates state-action pairs
///
/// Estimates Q(s, a) = expected return for taking action a in state s
#[derive(Module, Debug)]
pub struct Critic<B: Backend> {
    state_layer: Linear<B>,
    action_layer: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}

#[derive(Config, Debug)]
pub struct CriticConfig {
    /// State dimension
    #[config(default = 128)]
    pub state_dim: usize,
    /// Action dimension
    #[config(default = 1)]
    pub action_dim: usize,
    /// Hidden layer dimension
    #[config(default = 128)]
    pub hidden_dim: usize,
    #[config(default = true)]
    pub bias: bool,
}

impl CriticConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Critic<B> {
        // State and action are processed separately then concatenated
        // Combined: state_features (128) + action_features (128) = 256
        Critic {
            state_layer: LinearConfig::new(self.state_dim, self.hidden_dim)
                .with_bias(self.bias)
                .init(device),
            action_layer: LinearConfig::new(self.action_dim, self.hidden_dim)
                .with_bias(self.bias)
                .init(device),
            hidden: LinearConfig::new(self.hidden_dim * 2, self.hidden_dim)
                .with_bias(self.bias)
                .init(device),
            output: LinearConfig::new(self.hidden_dim, 1)
                .with_bias(self.bias)
                .init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Critic<B> {
    /// Forward pass: (state, action) → Q-value
    ///
    /// # Arguments
    /// * `state` - State tensor [batch_size, state_dim]
    /// * `action` - Action tensor [batch_size, action_dim]
    ///
    /// # Returns
    /// * Q-value tensor [batch_size, 1]
    pub fn forward(&self, state: Tensor<B, 2>, action: Tensor<B, 2>) -> Tensor<B, 2> {
        // Process state and action separately, then combine
        let state_features = self.state_layer.forward(state);
        let action_features = self.action_layer.forward(action);

        // Concatenate along feature dimension
        let combined = Tensor::cat(vec![state_features, action_features], 1);

        let x = self.hidden.forward(combined);
        let x = self.activation.forward(x);
        self.output.forward(x)
    }
}

// ============================================================================
// CatcherPolicy
// ============================================================================

/// DDPG-based cache eviction policy
///
/// Uses actor-critic architecture with:
/// - Deterministic policy (Actor) for continuous actions
/// - Q-learning (Critic) for value estimation
/// - Experience replay for stability
/// - Soft target network updates
pub struct CatcherPolicy<B: AutodiffBackend> {
    // Networks
    actor: Actor<B>,
    critic: Critic<B>,
    target_actor: Actor<B>,
    target_critic: Critic<B>,

    // Hyperparameters
    /// Discount factor for future rewards
    gamma: f32,
    /// Soft update coefficient for target networks (τ)
    tau: f32,
    /// Batch size for training
    batch_size: usize,
    /// Learning rate for actor
    actor_lr: f64,
    /// Learning rate for critic
    critic_lr: f64,

    // Exploration
    /// Standard deviation for exploration noise
    noise_std: f64,

    // GPU Replay Buffer (NEW - replaces Vec<Transition>)
    /// GPU-native replay buffer for zero-copy training
    gpu_buffer: HybridRingBuffer<B>,

    // Warmup configuration (NEW)
    /// Warmup batch size (smaller batch during warmup phase)
    warmup_batch_size: usize,
    /// Whether warmup phase is complete
    warmup_complete: bool,

    // State
    device: B::Device,
    /// State dimension (address history size)
    state_dim: usize,
    /// Action dimension (always 1 for continuous)
    action_dim: usize,
    /// Step counter for delayed updates
    step_count: usize,
    /// Target network update frequency
    target_update_freq: usize,
}

impl<B: AutodiffBackend> CatcherPolicy<B> {
    /// Create new Catcher DDPG policy
    pub fn new(device: B::Device, state_dim: usize) -> Self {
        let actor_config = ActorConfig::new().with_state_dim(state_dim);
        let critic_config = CriticConfig::new().with_state_dim(state_dim);

        let actor = actor_config.init(&device);
        let critic = critic_config.init(&device);
        let target_actor = actor_config.init(&device);
        let target_critic = critic_config.init(&device);

        // Warmup batch size: start with 256 or 1/8 of full batch size
        let warmup_batch_size = 256;

        Self {
            actor,
            critic,
            target_actor,
            target_critic,
            gamma: 0.99,
            tau: 0.005,
            batch_size: 64,
            actor_lr: 1e-4,
            critic_lr: 1e-3,
            noise_std: 0.1,
            gpu_buffer: HybridRingBuffer::new(100_000, state_dim),
            warmup_batch_size,
            warmup_complete: false,
            device,
            state_dim,
            action_dim: 1,
            step_count: 0,
            target_update_freq: 100,
        }
    }

    /// Create new Catcher DDPG policy with custom configuration
    pub fn with_config(
        device: B::Device,
        buffer_capacity: usize,
        state_dim: usize,
        action_dim: usize,
        target_update_freq: usize,
    ) -> Self {
        let actor_config = ActorConfig::new().with_state_dim(state_dim);
        let critic_config = CriticConfig::new().with_state_dim(state_dim);

        let actor = actor_config.init(&device);
        let critic = critic_config.init(&device);
        let target_actor = actor_config.init(&device);
        let target_critic = critic_config.init(&device);

        // Warmup batch size: start with 256
        let warmup_batch_size = 256;

        Self {
            actor,
            critic,
            target_actor,
            target_critic,
            gamma: 0.99,
            tau: 0.005,
            batch_size: 64,
            actor_lr: 1e-4,
            critic_lr: 1e-3,
            noise_std: 0.1,
            gpu_buffer: HybridRingBuffer::new(buffer_capacity, state_dim),
            warmup_batch_size,
            warmup_complete: false,
            device,
            state_dim,
            action_dim,
            step_count: 0,
            target_update_freq,
        }
    }

    /// Get state dimension
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// Get action dimension
    pub fn action_dim(&self) -> usize {
        self.action_dim
    }

    /// Select action with optional exploration noise
    ///
    /// # Arguments
    /// * `state` - Current environment state
    /// * `add_noise` - Whether to add exploration noise
    ///
    /// # Returns
    /// * Continuous action in range [-1, 1]
    pub fn select_action_with_noise(&self, state: &State, add_noise: bool) -> Action {
        let state_tensor = state_to_tensor::<B>(state, &self.device);

        // Actor forward pass (deterministic)
        let action = self.actor.forward(state_tensor);

        let action = if add_noise {
            // Add Gaussian exploration noise
            let noise = Tensor::random(
                [1, self.action_dim],
                Distribution::Normal(0.0, self.noise_std),
                &self.device,
            );
            let noisy = action + noise;
            // Clamp to [-1, 1]
            noisy.clamp(-1.0, 1.0)
        } else {
            action
        };

        // Extract scalar value
        let action_data = action.into_data().convert::<f32>();
        let action_val = action_data.as_slice().unwrap()[0];
        Action::Continuous(vec![action_val])
    }

    /// DDPG training step
    ///
    /// # Arguments
    /// * `states` - Batch of states [batch_size, state_dim]
    /// * `actions` - Batch of actions [batch_size, action_dim]
    /// * `rewards` - Batch of rewards [batch_size]
    /// * `next_states` - Batch of next states [batch_size, state_dim]
    /// * `dones` - Batch of done flags [batch_size]
    ///
    /// # Returns
    /// * (critic_loss, actor_loss) tuple
    pub fn train_ddpg(
        &mut self,
        states: Tensor<B, 2>,
        actions: Tensor<B, 2>,
        rewards: Tensor<B, 1>,
        next_states: Tensor<B, 2>,
        dones: Tensor<B, 1>,
    ) -> (f32, f32) {
        // Update critic
        let critic_loss = self.update_critic(
            states.clone(),
            actions.clone(),
            rewards,
            next_states.clone(),
            dones,
        );

        // Update actor (every other step for stability)
        let actor_loss = if self.step_count % 2 == 0 {
            self.update_actor(states)
        } else {
            0.0
        };

        // Soft update target networks
        self.soft_update_targets();

        self.step_count += 1;

        (critic_loss, actor_loss)
    }

    /// Update critic network
    ///
    /// Minimizes TD error: (Q(s,a) - y)^2
    /// where y = r + γ * Q'(s', π'(s'))
    fn update_critic(
        &mut self,
        states: Tensor<B, 2>,
        actions: Tensor<B, 2>,
        rewards: Tensor<B, 1>,
        next_states: Tensor<B, 2>,
        dones: Tensor<B, 1>,
    ) -> f32 {
        // Compute target Q: y = r + γ * Q'(s', π'(s'))
        let next_actions = self.target_actor.forward(next_states.clone());
        let target_q = self.target_critic.forward(next_states, next_actions);
        let target_q_1d: Tensor<B, 1> = target_q.squeeze();

        // y = r + (1 - done) * gamma * Q'(s', a')
        let ones = Tensor::<B, 1>::ones([rewards.shape().dims[0]], &self.device);
        let not_dones = ones - dones;
        let target = rewards
            + not_dones * target_q_1d * Tensor::from_floats([self.gamma as f64], &self.device);

        // Current Q
        let current_q: Tensor<B, 1> = self.critic.forward(states, actions).squeeze();

        // Critic loss: MSE
        let loss = (current_q.clone() - target.detach())
            .powf_scalar(2.0)
            .mean();

        // Get loss value before backward
        let loss_val = loss
            .clone()
            .into_data()
            .convert::<f32>()
            .as_slice()
            .unwrap()[0];

        // Backprop
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.critic);
        let mut optimizer = AdamConfig::new().init();
        self.critic = optimizer.step(self.critic_lr, self.critic.clone(), grads_params);

        loss_val
    }

    /// Update actor network
    ///
    /// Maximizes expected Q-value: -Q(s, π(s))
    fn update_actor(&mut self, states: Tensor<B, 2>) -> f32 {
        // Actor loss: -mean(Q(s, actor(s)))
        let actions = self.actor.forward(states.clone());
        let q_values = self.critic.forward(states, actions);
        let loss = -q_values.mean();

        // Get loss value before backward
        let loss_val = loss
            .clone()
            .into_data()
            .convert::<f32>()
            .as_slice()
            .unwrap()[0];

        // Backprop
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.actor);
        let mut optimizer = AdamConfig::new().init();
        self.actor = optimizer.step(self.actor_lr, self.actor.clone(), grads_params);

        loss_val
    }

    /// Soft update target networks (polyak averaging)
    ///
    /// θ_target ← τθ + (1 - τ)θ_target
    ///
    /// Note: This is a simplified implementation. In production, you would
    /// implement proper soft updates by iterating over named parameters.
    /// For now, this uses periodic hard updates.
    fn soft_update_targets(&mut self) {
        // TODO: Implement proper soft updates using Burn's parameter API
        // For now, we use periodic hard updates for simplicity
        if self.step_count % 100 == 0 {
            // Hard update every 100 steps
            self.target_actor = self.actor.clone();
            self.target_critic = self.critic.clone();
        }
    }
}

// ============================================================================
// CachePolicy Implementation
// ============================================================================

impl<B: AutodiffBackend> CachePolicy for CatcherPolicy<B> {
    fn select_action(&self, state: &State) -> Action {
        // During training, add exploration noise
        self.select_action_with_noise(state, true)
    }

    fn update(&mut self, _transition: &Transition) -> f32 {
        // DDPG uses batch updates via train_step
        // This method is for single-step online updates
        // Return 0.0 and handle updates in train_step
        0.0
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        // Note: Burn doesn't have a simple save/load API yet
        // This is a placeholder - actual implementation would use
        // Burn's ModelRecorder trait or similar serialization
        let _ = path;
        Err("Save not yet implemented - requires Burn ModelRecorder".into())
    }

    fn load(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
        // Note: Burn doesn't have a simple load API yet
        let _ = path;
        Err("Load not yet implemented - requires Burn ModelRecorder".into())
    }

    fn policy_type(&self) -> PolicyType {
        PolicyType::Catcher
    }

    fn action_dim(&self) -> usize {
        self.action_dim
    }
}

// ============================================================================
// ReplayPolicy Implementation
// ============================================================================

impl<B: AutodiffBackend> ReplayPolicy for CatcherPolicy<B> {
    fn train_step(&mut self, batch: &[Transition]) -> f32 {
        if batch.is_empty() {
            return 0.0;
        }

        // Convert batch to tensors
        let (states, actions, rewards, next_states, dones) =
            batch_to_tensors::<B>(batch, &self.device);

        let (critic_loss, actor_loss) =
            self.train_ddpg(states, actions, rewards, next_states, dones);

        critic_loss + actor_loss
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn update_target(&mut self) {
        // Hard update target networks
        self.target_actor = self.actor.clone();
        self.target_critic = self.critic.clone();
    }
}

// ============================================================================
// GpuTrainable Implementation
// ============================================================================

impl<B: AutodiffBackend> GpuTrainable<B> for CatcherPolicy<B> {
    fn gpu_buffer_mut(&mut self) -> &mut HybridRingBuffer<B> {
        &mut self.gpu_buffer
    }

    fn gpu_buffer(&self) -> &HybridRingBuffer<B> {
        &self.gpu_buffer
    }

    fn full_batch_size(&self) -> usize {
        self.batch_size
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
        self.target_update_freq
    }

    fn step_count(&self) -> usize {
        self.step_count
    }

    fn increment_step_count(&mut self) {
        self.step_count += 1;
    }

    fn epsilon(&self) -> f32 {
        // Catcher uses deterministic policy with Gaussian noise
        // Return noise_std as a proxy for exploration
        self.noise_std as f32
    }

    fn update_epsilon(&mut self) {
        // Decay exploration noise (similar to epsilon decay in DQN)
        // Decay towards minimum noise of 0.01
        self.noise_std = (self.noise_std * 0.995).max(0.01);
    }

    fn train_step_gpu(&mut self, batch: &TensorTransitionBatch<B>) -> f32 {
        // Catcher-specific GPU training logic (DDPG)

        // Forward pass through actor and critic
        let features = batch.states.clone();

        // Convert actions from Int to Float for critic (actions are continuous in [-1, 1])
        // For Catcher, actions are stored as discrete tier indices (0-9)
        // We need to convert them back to continuous values in [-1, 1] for DDPG training
        // Map [0, 9] -> [-1, 1]: continuous = (index / 9.0) * 2.0 - 1.0
        let batch_size = batch.states.dims()[0];
        let actions_float: Vec<f32> = batch
            .actions
            .clone()
            .into_data()
            .convert::<i64>()
            .as_slice::<i64>()
            .unwrap()
            .iter()
            .map(|&idx| ((idx as f32 / 9.0) * 2.0) - 1.0)
            .collect();
        let actions_2d: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(actions_float, [batch_size, 1]).convert::<f32>(),
            &self.device,
        );

        let _q_values = self.critic.forward(features.clone(), actions_2d.clone());

        // Compute target Q-values
        let next_features = batch.next_states.clone();
        let next_actions = self.target_actor.forward(next_features.clone());
        let next_q_values = self.target_critic.forward(next_features, next_actions);
        let max_next_q: Tensor<B, 1> = next_q_values.squeeze();

        // TD target: y = r + (1 - done) * gamma * Q'(s', a')
        let not_done = batch.dones.clone().neg().add_scalar(1.0f32);
        let gamma_tensor = Tensor::from_data(
            TensorData::new(vec![self.gamma], [1]).convert::<f32>(),
            &self.device,
        );
        let target_q = batch
            .rewards
            .clone()
            .add(max_next_q.mul(not_done).mul(gamma_tensor))
            .detach();

        // Current Q-values for actions taken
        let current_q: Tensor<B, 1> = self.critic.forward(features, actions_2d).squeeze();

        // Compute critic loss (MSE)
        let critic_loss = current_q.sub(target_q).powf_scalar(2.0).mean();

        // Get critic loss value before backward
        let critic_loss_val: f32 = critic_loss
            .clone()
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()[0];

        // Backward pass and optimize critic
        let grads = critic_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.critic);
        let mut critic_optimizer = AdamConfig::new().init();
        self.critic = critic_optimizer.step(self.critic_lr, self.critic.clone(), grads_params);

        // Update actor (every other step for stability)
        let actor_loss = if self.step_count % 2 == 0 {
            // Actor loss: -mean(Q(s, actor(s)))
            let actor_actions = self.actor.forward(batch.states.clone());
            let actor_q_values = self.critic.forward(batch.states.clone(), actor_actions);
            let actor_loss = -actor_q_values.mean();

            let actor_loss_val = actor_loss
                .clone()
                .into_data()
                .convert::<f32>()
                .as_slice()
                .unwrap()[0];

            // Backward pass and optimize actor
            let grads = actor_loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &self.actor);
            let mut actor_optimizer = AdamConfig::new().init();
            self.actor = actor_optimizer.step(self.actor_lr, self.actor.clone(), grads_params);

            actor_loss_val
        } else {
            0.0
        };

        // Soft update target networks
        self.soft_update_targets();

        critic_loss_val + actor_loss
    }

    fn maybe_update_target(&mut self, step_count: usize) {
        if step_count % self.target_update_freq == 0 {
            // Hard update target networks
            self.target_actor = self.actor.clone();
            self.target_critic = self.critic.clone();
        }
    }
}

// ============================================================================
// Checkpointable Implementation
// ============================================================================

impl<B: AutodiffBackend> Checkpointable<B> for CatcherPolicy<B> {
    fn checkpoint_name(&self) -> &str {
        "catcher_policy"
    }

    fn checkpoint_metadata(&self) -> CheckpointMetadata {
        CheckpointMetadata::new_with_dims(
            "CatcherPolicy".to_string(),
            0, // epoch - will be updated by training loop
            self.state_dim,
            self.action_dim,
            self.state_dim, // feature_dim = state_dim for catcher
        )
    }

    fn model(&self) -> &impl Module<B> {
        &self.actor
    }
}

// ============================================================================
// BatchedActionSelector Implementation
// ============================================================================

impl<B: AutodiffBackend> crate::training::BatchedActionSelector<B> for CatcherPolicy<B> {
    fn select_actions_batched(
        &self,
        observations: &[Vec<f64>],
        device: &B::Device,
        action_dim: usize,
        epsilon: f32,
    ) -> Vec<usize> {
        use burn::tensor::{Distribution, Int, Tensor};

        // Use shared utility for tensor conversion - NO MORE DUPLICATION!
        let states_tensor =
            crate::training::batched_action_utils::observations_to_tensor(observations, device);

        // Policy-specific: Actor network forward pass (continuous actions)
        let continuous_actions = self.actor.forward(states_tensor);

        // Catcher-specific: Map continuous actions [-1, 1] to discrete tier indices [0, action_dim)
        // Formula: index = ((continuous + 1.0) / 2.0) * action_dim
        let actions_shifted = continuous_actions.clone().add_scalar(1.0);
        let actions_scaled = actions_shifted.div_scalar(2.0);
        let actions_final = actions_scaled.mul_scalar(action_dim as f64);

        // Add exploration noise during training
        if epsilon > 0.0 {
            let batch_size = observations.len();
            let noise = Tensor::<B, 2>::random(
                [batch_size, 1],
                Distribution::Uniform(-epsilon as f64, epsilon as f64),
                device,
            );
            let actions_noisy = actions_final + noise;
            let actions_clamped = actions_noisy.clamp(0.0, action_dim as f64 - 1.0);
            let actions_int: Tensor<B, 1, Int> = actions_clamped.int().squeeze();

            // Convert to Vec<usize>
            actions_int
                .into_data()
                .convert::<i64>()
                .as_slice::<i64>()
                .unwrap()
                .iter()
                .map(|&x| x as usize)
                .collect()
        } else {
            // No exploration - just convert to discrete
            let actions_int: Tensor<B, 1, Int> = actions_final.int().squeeze();

            actions_int
                .into_data()
                .convert::<i64>()
                .as_slice::<i64>()
                .unwrap()
                .iter()
                .map(|&x| x as usize)
                .collect()
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert State to tensor
///
/// # Arguments
/// * `state` - Policy state (Raw for address history)
/// * `device` - Compute device
///
/// # Returns
/// * Tensor of shape [1, state_dim]
fn state_to_tensor<B: Backend>(state: &State, device: &B::Device) -> Tensor<B, 2> {
    match state {
        State::Raw(data) => {
            // Pad or truncate to 128 elements
            let mut vec: Vec<f32> = data.iter().map(|&x| x as f32).collect();

            if vec.len() < 128 {
                vec.extend(std::iter::repeat(0.0).take(128 - vec.len()));
            } else if vec.len() > 128 {
                vec.truncate(128);
            }

            let data = TensorData::new(vec, [1, 128]).convert::<f32>();
            Tensor::from_data(data, device)
        }
        State::Features(feat) => {
            // Use features as state (pad/truncate to 128)
            let mut vec = feat.clone();
            if vec.len() < 128 {
                vec.extend(std::iter::repeat(0.0).take(128 - vec.len()));
            } else if vec.len() > 128 {
                vec.truncate(128);
            }

            let data = TensorData::new(vec, [1, 128]).convert::<f32>();
            Tensor::from_data(data, device)
        }
        State::Empty => Tensor::zeros([1, 128], device),
    }
}

/// Convert batch of transitions to tensors
///
/// # Arguments
/// * `batch` - Slice of transitions
/// * `device` - Compute device
///
/// # Returns
/// * (states, actions, rewards, next_states, dones) tensors
fn batch_to_tensors<B: Backend>(
    batch: &[Transition],
    device: &B::Device,
) -> (
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 1>,
    Tensor<B, 2>,
    Tensor<B, 1>,
) {
    let batch_size = batch.len();

    // Collect data
    let mut states_flat = Vec::with_capacity(batch_size * 128);
    let mut actions_flat = Vec::with_capacity(batch_size);
    let mut rewards_flat = Vec::with_capacity(batch_size);
    let mut next_states_flat = Vec::with_capacity(batch_size * 128);
    let mut dones_flat = Vec::with_capacity(batch_size);

    for transition in batch {
        // State
        let state_vec = match &transition.state {
            State::Raw(d) => d.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            State::Features(f) => f.clone(),
            State::Empty => vec![0.0; 128],
        };
        for i in 0..128 {
            states_flat.push(if i < state_vec.len() {
                state_vec[i]
            } else {
                0.0
            });
        }

        // Action (continuous, extract scalar)
        let action_val = match &transition.action {
            Action::Continuous(v) => v.get(0).copied().unwrap_or(0.0),
            Action::Discrete(_) => 0.0, // Should not happen for Catcher
        };
        actions_flat.push(action_val);

        // Reward
        rewards_flat.push(transition.reward);

        // Next state
        let next_state_vec = match &transition.next_state {
            State::Raw(d) => d.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            State::Features(f) => f.clone(),
            State::Empty => vec![0.0; 128],
        };
        for i in 0..128 {
            next_states_flat.push(if i < next_state_vec.len() {
                next_state_vec[i]
            } else {
                0.0
            });
        }

        // Done
        dones_flat.push(if transition.done { 1.0 } else { 0.0 });
    }

    // Create tensors
    let states = Tensor::from_data(
        TensorData::new(states_flat, [batch_size, 128]).convert::<f32>(),
        device,
    );
    let actions = Tensor::from_data(
        TensorData::new(actions_flat, [batch_size, 1]).convert::<f32>(),
        device,
    );
    let rewards = Tensor::from_data(
        TensorData::new(rewards_flat, [batch_size]).convert::<f32>(),
        device,
    );
    let next_states = Tensor::from_data(
        TensorData::new(next_states_flat, [batch_size, 128]).convert::<f32>(),
        device,
    );
    let dones = Tensor::from_data(
        TensorData::new(dones_flat, [batch_size]).convert::<f32>(),
        device,
    );

    (states, actions, rewards, next_states, dones)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    // Test backend with autodiff for policy tests
    type TestBackend = Autodiff<NdArray>;

    #[test]
    fn test_actor_forward() {
        let device = <NdArray as Backend>::Device::default();
        let config = ActorConfig::new();
        let actor: Actor<NdArray> = config.init(&device);

        // Create dummy state [batch=1, state_dim=128]
        let state = Tensor::zeros([1, 128], &device);
        let action = actor.forward(state);

        assert_eq!(action.shape().dims, [1, 1]);

        // Should be in [-1, 1] due to tanh
        let val: f32 = action
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()[0];
        assert!(val >= -1.0 && val <= 1.0, "Action {} not in [-1, 1]", val);
    }

    #[test]
    fn test_critic_forward() {
        let device = <NdArray as Backend>::Device::default();
        let config = CriticConfig::new();
        let critic: Critic<NdArray> = config.init(&device);

        // Create dummy inputs
        let state = Tensor::zeros([1, 128], &device);
        let action = Tensor::zeros([1, 1], &device);

        let q_value = critic.forward(state, action);

        assert_eq!(q_value.shape().dims, [1, 1]);

        // Q-value should be a finite scalar
        let val: f32 = q_value
            .into_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .unwrap()[0];
        assert!(val.is_finite());
    }

    #[test]
    fn test_catcher_policy_creation() {
        let device = <TestBackend as Backend>::Device::default();
        let policy: CatcherPolicy<TestBackend> = CatcherPolicy::new(device, 128);

        assert_eq!(policy.state_dim, 128);
        assert_eq!(policy.action_dim, 1);
        assert_eq!(policy.gamma, 0.99);
        assert_eq!(policy.tau, 0.005);
        assert_eq!(policy.batch_size, 64);
    }

    #[test]
    fn test_action_selection() {
        let device = <TestBackend as Backend>::Device::default();
        let policy: CatcherPolicy<TestBackend> = CatcherPolicy::new(device, 128);

        // Test with Features state
        let state = State::Features(vec![1.0; 128]);
        let action = policy.select_action(&state);

        match action {
            Action::Continuous(v) => {
                assert_eq!(v.len(), 1);
                assert!(
                    v[0] >= -1.0 && v[0] <= 1.0,
                    "Action {} not in [-1, 1]",
                    v[0]
                );
            }
            Action::Discrete(_) => panic!("Expected continuous action"),
        }

        // Test with Raw state (for address history)
        let raw_state = State::Raw(vec![100.0; 128]);
        let action = policy.select_action(&raw_state);

        match action {
            Action::Continuous(v) => {
                assert_eq!(v.len(), 1);
                assert!(
                    v[0] >= -1.0 && v[0] <= 1.0,
                    "Action {} not in [-1, 1]",
                    v[0]
                );
            }
            Action::Discrete(_) => panic!("Expected continuous action"),
        }
    }

    #[test]
    fn test_action_selection_no_noise() {
        let device = <TestBackend as Backend>::Device::default();
        let policy: CatcherPolicy<TestBackend> = CatcherPolicy::new(device, 128);

        let state = State::Features(vec![0.5; 128]);
        let action = policy.select_action_with_noise(&state, false);

        // Deterministic action should be consistent
        let action2 = policy.select_action_with_noise(&state, false);

        assert_eq!(action, action2);
    }

    #[test]
    fn test_state_to_tensor_conversion() {
        let device = <NdArray as Backend>::Device::default();

        // Test Raw state conversion
        let raw_state = State::Raw(vec![1.0; 100]); // Shorter than 128
        let tensor = state_to_tensor::<NdArray>(&raw_state, &device);
        assert_eq!(tensor.shape().dims, [1, 128]);

        // Test Features state conversion
        let feat_state = State::Features(vec![0.5; 200]); // Longer than 128
        let tensor = state_to_tensor::<NdArray>(&feat_state, &device);
        assert_eq!(tensor.shape().dims, [1, 128]);

        // Test Empty state
        let empty_state = State::Empty;
        let tensor = state_to_tensor::<NdArray>(&empty_state, &device);
        assert_eq!(tensor.shape().dims, [1, 128]);
    }

    #[test]
    fn test_policy_type() {
        let device = <TestBackend as Backend>::Device::default();
        let policy: CatcherPolicy<TestBackend> = CatcherPolicy::new(device, 128);

        assert_eq!(policy.policy_type(), PolicyType::Catcher);
        assert_eq!(policy.action_dim(), 1);
    }
}
