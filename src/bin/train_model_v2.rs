#![recursion_limit = "256"]

//! Generic model training binary with checkpoint support (V2 - Optimized).
//!
//! Usage:
//!   train_model_v2 --model dqn --episodes 100 --max-steps 100
//!
//! Backend selection is done via --backend flag (runtime):
//!   train_model_v2 --model dqn --backend cpu
//!   train_model_v2 --model dqn --backend wgpu
//!   train_model_v2 --model dqn --backend cuda
//!   train_model_v2 --model dqn --backend rocm
//!
//! Key optimizations from train_inference:
//! - 64MB thread stack (prevents stack overflow)
//! - Box::new() for heap-allocated models

use burn::tensor::TensorData;
use clap::{Parser, ValueEnum};
use eris::device::{available_backends, Device};
#[cfg(feature = "cuda")]
use eris::utils::is_gpu_backend;
use eris::utils::log_backend_info;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use eris::training::CombinedAgent;
use std::path::PathBuf;

#[derive(Clone, Debug, ValueEnum)]
enum ModelType {
    /// Metis: Combined DQN + Bandit (legacy)
    Metis,
    /// MetisV2: Joint Bandit + DQN with SequentialCompose (NEW)
    MetisV2,
    /// Cacheus: Contextual Multi-Armed Bandit
    Cacheus,
    /// Catcher: DDPG Actor-Critic
    Catcher,
    /// DQN: Pure Deep Q-Network
    Dqn,
    /// Bandit: Standalone Contextual Bandit
    Bandit,
}

/// Exploration strategy for action selection
#[derive(Clone, Debug, ValueEnum)]
enum ExplorationStrategy {
    /// Epsilon-greedy: random with probability epsilon
    EpsilonGreedy,
    /// Thompson Sampling: Bayesian posterior sampling
    ThompsonSampling,
    /// Upper Confidence Bound: theoretically optimal exploration
    Ucb,
}

/// Logging verbosity level
#[derive(Clone, Debug, ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Trace file format
#[derive(Clone, Debug, ValueEnum)]
enum TraceFormat {
    /// Auto-detect from file extension
    Autodetect,
    /// Recorder CSV format
    Recorder,
    /// DFTracer .pfw.gz format
    Dftracer,
}

/// Validate batch size is a multiple of 32 and within reasonable bounds
fn validate_batch_size(s: &str) -> Result<usize, String> {
    let size: usize = s.parse().map_err(|_| "Invalid number")?;
    if size % 32 != 0 {
        return Err(format!(
            "Batch size must be multiple of 32 for GPU warp alignment (got {})",
            size
        ));
    }
    // Allow smaller sizes for warmup_batch_size
    if size < 32 {
        return Err(format!("Batch size must be at least 32 (got {})", size));
    }
    // Increase max for large batch training
    if size > 65536 {
        return Err(format!("Batch size should not exceed 65536 (got {})", size));
    }
    Ok(size)
}

#[derive(Parser, Clone)]
#[command(name = "train_model_v2")]
#[command(about = "Train cache policy: metis, cacheus, or catcher (V2 - Optimized)")]
struct Args {
    /// Which policy to train
    #[arg(short, long, value_enum, default_value = "metis")]
    model: ModelType,

    /// Number of episodes
    #[arg(short, long, default_value = "100")]
    episodes: usize,

    /// Max steps per episode
    #[arg(short = 's', long, default_value = "100")]
    max_steps: usize,

    /// Batch size for training (must be multiple of 32 for GPU warp alignment)
    #[arg(short = 'B', long, default_value = "2048", value_parser = validate_batch_size)]
    batch_size: usize,

    /// Warmup batch size for training (smaller batches during initial steps)
    /// During warmup, uses this smaller batch size to stabilize training.
    /// After warmup_steps, switches to full batch_size.
    /// Must be <= batch_size and multiple of 32.
    #[arg(long, default_value = "256", value_parser = validate_batch_size)]
    warmup_batch_size: usize,

    /// Number of warmup steps before using full batch size
    /// During warmup, training uses warmup_batch_size and runs every step.
    /// After warmup, uses full batch_size and runs every train_freq steps.
    #[arg(long, default_value = "1000")]
    warmup_steps: usize,

    /// Learning rate
    #[arg(short, long, default_value = "0.0001")]
    learning_rate: f64,

    /// Backend selection at runtime: cpu, wgpu, cuda, rocm
    #[arg(long, default_value = "cpu")]
    backend: String,

    /// Path to configuration file
    #[arg(short, long, default_value = "config/tiers.toml")]
    config: PathBuf,

    /// Path to trace file (CSV or .pfw.gz)
    #[arg(long, default_value = "recorder-csv/NWChem-64_combined.csv")]
    trace_file: PathBuf,

    /// Trace format: recorder (CSV), dftracer (pfw.gz), or autodetect (by extension)
    #[arg(long, value_enum, default_value = "autodetect")]
    trace_format: TraceFormat,

    /// Number of parallel environments
    #[arg(long, default_value = "16")]
    num_envs: usize,

    /// Discount factor for future rewards
    #[arg(short, long, default_value = "0.99")]
    gamma: f64,

    /// Target network update frequency (in steps)
    #[arg(long, default_value = "1000")]
    target_update_freq: usize,

    /// Replay buffer capacity
    #[arg(long, default_value = "100000")]
    buffer_capacity: usize,

    /// Minimum replay buffer size before training starts
    #[arg(long, default_value = "10000")]
    min_buffer_size: usize,

    /// How often to train (in steps)
    #[arg(long, default_value = "4")]
    train_freq: usize,

    /// Gradient clipping norm (0.0 disables clipping)
    #[arg(long, default_value = "1.0")]
    grad_clip: f64,

    /// Exploration epsilon start
    #[arg(long, default_value = "1.0")]
    epsilon_start: f64,

    /// Exploration epsilon end
    #[arg(long, default_value = "0.01")]
    epsilon_end: f64,

    /// Exploration decay rate
    #[arg(long, default_value = "0.995")]
    epsilon_decay: f64,

    /// Exploration decay interval (steps)
    #[arg(long, default_value = "100")]
    epsilon_decay_interval: usize,

    /// Log level for tracing output
    #[arg(long, value_enum, default_value = "info")]
    log_level: LogLevel,

    /// Enable checkpoint saving
    #[arg(long, default_value = "true")]
    checkpoint: bool,

    /// Checkpoint directory
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: PathBuf,

    /// Load checkpoint from file
    #[arg(long)]
    load_checkpoint: Option<PathBuf>,

    /// Exploration strategy
    #[arg(long, value_enum, default_value = "epsilon-greedy")]
    exploration: ExplorationStrategy,

    /// Bandit exploration bonus weight (for UCB/Thompson)
    #[arg(long, default_value = "1.0")]
    bandit_bonus: f64,

    /// DQN loss type: huber or mse
    #[arg(long, default_value = "huber")]
    loss_type: String,

    /// Double DQN: use target network for action selection
    #[arg(long, default_value = "true")]
    double_dqn: bool,

    /// Dueling DQN: separate value and advantage streams
    #[arg(long, default_value = "true")]
    dueling_dqn: bool,

    /// Prioritized Experience Replay: sample important transitions more
    #[arg(long, default_value = "false")]
    per: bool,

    /// PER alpha: prioritization exponent
    #[arg(long, default_value = "0.6")]
    per_alpha: f64,

    /// PER beta: importance sampling exponent
    #[arg(long, default_value = "0.4")]
    per_beta: f64,

    /// PER epsilon: small constant for numerical stability
    #[arg(long, default_value = "0.00001")]
    per_epsilon: f64,

    /// N-step return: number of steps for n-step returns
    #[arg(long, default_value = "1")]
    n_step: usize,

    /// Noisy networks: use noisy layers for exploration
    #[arg(long, default_value = "false")]
    noisy: bool,

    /// Noisy sigma: noise standard deviation
    #[arg(long, default_value = "0.5")]
    noisy_sigma: f64,

    /// Categorical DQN: use distributional RL
    #[arg(long, default_value = "false")]
    categorical: bool,

    /// Categorical atoms: number of atoms for C51
    #[arg(long, default_value = "51")]
    categorical_atoms: usize,

    /// V-min: minimum value for categorical DQN
    #[arg(long, default_value = "-100.0")]
    categorical_v_min: f64,

    /// V-max: maximum value for categorical DQN
    #[arg(long, default_value = "100.0")]
    categorical_v_max: f64,

    /// Rainbow: combine all DQN improvements
    #[arg(long, default_value = "false")]
    rainbow: bool,

    /// Bandit context window: number of past observations for context
    #[arg(long, default_value = "10")]
    context_window: usize,

    /// Bandit decay: decay factor for bandit rewards
    #[arg(long, default_value = "0.99")]
    bandit_decay: f64,

    /// Bandit prior: prior count for bandit arms
    #[arg(long, default_value = "1.0")]
    bandit_prior: f64,

    /// Actor learning rate (for Catcher/DDPG)
    #[arg(long, default_value = "0.0001")]
    actor_lr: f64,

    /// Critic learning rate (for Catcher/DDPG)
    #[arg(long, default_value = "0.001")]
    critic_lr: f64,

    /// Actor update frequency (for Catcher/DDPG)
    #[arg(long, default_value = "2")]
    actor_update_freq: usize,

    /// Target network soft update tau (for Catcher/DDPG)
    #[arg(long, default_value = "0.005")]
    tau: f64,

    /// Ornstein-Uhlenbeck noise theta (for Catcher/DDPG)
    #[arg(long, default_value = "0.15")]
    ou_theta: f64,

    /// Ornstein-Uhlenbeck noise sigma (for Catcher/DDPG)
    #[arg(long, default_value = "0.2")]
    ou_sigma: f64,

    /// Ornstein-Uhlenbeck noise mu (for Catcher/DDPG)
    #[arg(long, default_value = "0.0")]
    ou_mu: f64,

    /// Feature dimension for model
    #[arg(long, default_value = "256")]
    feature_dim: usize,

    /// DQN hidden layer sizes (comma-separated)
    #[arg(long, default_value = "512,256")]
    dqn_hidden: String,

    /// Bandit hidden layer sizes (comma-separated)
    #[arg(long, default_value = "256,128")]
    bandit_hidden: String,

    /// Activation function: relu, gelu, silu
    #[arg(long, default_value = "relu")]
    activation: String,

    /// Layer normalization: use layer norm in model
    #[arg(long, default_value = "true")]
    layer_norm: bool,

    /// Dropout rate (0.0 disables dropout)
    #[arg(long, default_value = "0.0")]
    dropout: f64,

    /// Residual connections: use skip connections
    #[arg(long, default_value = "true")]
    residual: bool,

    /// Attention heads: number of attention heads (0 disables attention)
    #[arg(long, default_value = "0")]
    attention_heads: usize,

    /// Attention dim: attention dimension
    #[arg(long, default_value = "64")]
    attention_dim: usize,

    /// Use FlashAttention: optimized attention kernel
    #[arg(long, default_value = "false")]
    flash_attention: bool,

    /// Use Triton kernels: custom GPU kernels
    #[arg(long, default_value = "false")]
    triton_kernels: bool,

    /// Use mixed precision: FP16 training
    #[arg(long, default_value = "false")]
    mixed_precision: bool,

    /// Use gradient accumulation: accumulate gradients over multiple steps
    #[arg(long, default_value = "1")]
    grad_accum_steps: usize,

    /// Use learning rate scheduling: adjust LR during training
    #[arg(long, default_value = "true")]
    lr_schedule: bool,

    /// LR schedule type: linear, cosine, step
    #[arg(long, default_value = "cosine")]
    lr_schedule_type: String,

    /// LR warmup steps: linear warmup at start
    #[arg(long, default_value = "1000")]
    lr_warmup_steps: usize,

    /// LR decay steps: steps to decay LR
    #[arg(long, default_value = "100000")]
    lr_decay_steps: usize,

    /// LR min: minimum learning rate
    #[arg(long, default_value = "0.00001")]
    lr_min: f64,

    /// Use early stopping: stop if no improvement
    #[arg(long, default_value = "false")]
    early_stopping: bool,

    /// Early stopping patience: episodes to wait
    #[arg(long, default_value = "10")]
    patience: usize,

    /// Early stopping min delta: minimum improvement
    #[arg(long, default_value = "0.001")]
    min_delta: f64,

    /// Use curriculum learning: start with easy tasks
    #[arg(long, default_value = "false")]
    curriculum: bool,

    /// Curriculum start episode: when to start curriculum
    #[arg(long, default_value = "0")]
    curriculum_start: usize,

    /// Curriculum end episode: when to end curriculum
    #[arg(long, default_value = "10000")]
    curriculum_end: usize,

    /// Use self-play: train against previous versions
    #[arg(long, default_value = "false")]
    self_play: bool,

    /// Self-play update frequency: how often to update opponent
    #[arg(long, default_value = "100")]
    self_play_freq: usize,

    /// Use imitation learning: learn from expert demonstrations
    #[arg(long, default_value = "false")]
    imitation: bool,

    /// Imitation batch size: batch size for IL
    #[arg(long, default_value = "256")]
    imitation_batch_size: usize,

    /// Imitation weight: weight for IL loss
    #[arg(long, default_value = "0.5")]
    imitation_weight: f64,

    /// Use offline RL: train from fixed dataset
    #[arg(long, default_value = "false")]
    offline: bool,

    /// Offline dataset path: path to offline dataset
    #[arg(long, default_value = "")]
    offline_dataset: PathBuf,

    /// Use model-based RL: learn environment model
    #[arg(long, default_value = "false")]
    model_based: bool,

    /// Model horizon: planning horizon for model-based
    #[arg(long, default_value = "5")]
    model_horizon: usize,

    /// Model ensemble size: number of models in ensemble
    #[arg(long, default_value = "5")]
    model_ensemble_size: usize,

    /// Use multi-task learning: train on multiple tasks
    #[arg(long, default_value = "false")]
    multi_task: bool,

    /// Multi-task weights: task weights (comma-separated)
    #[arg(long, default_value = "1.0")]
    multi_task_weights: String,

    /// Use meta-learning: learn to learn
    #[arg(long, default_value = "false")]
    meta_learning: bool,

    /// Meta inner steps: inner loop steps for MAML
    #[arg(long, default_value = "5")]
    meta_inner_steps: usize,

    /// Meta outer lr: outer loop learning rate
    #[arg(long, default_value = "0.001")]
    meta_outer_lr: f64,

    /// Use distributed training: multi-GPU training
    #[arg(long, default_value = "false")]
    distributed: bool,

    /// Distributed world size: number of GPUs
    #[arg(long, default_value = "1")]
    world_size: usize,

    /// Distributed rank: GPU rank
    #[arg(long, default_value = "0")]
    rank: usize,

    /// Use async training: async data loading
    #[arg(long, default_value = "true")]
    async_training: bool,

    /// Async num workers: number of data loader workers
    #[arg(long, default_value = "4")]
    async_workers: usize,

    /// Use mixed backend: combine CPU and GPU
    #[arg(long, default_value = "false")]
    mixed_backend: bool,

    /// Mixed backend ratio: GPU ratio (0.0-1.0)
    #[arg(long, default_value = "0.8")]
    mixed_ratio: f64,

    /// Use memory compression: compress replay buffer
    #[arg(long, default_value = "false")]
    memory_compression: bool,

    /// Memory compression ratio: compression ratio
    #[arg(long, default_value = "0.5")]
    compression_ratio: f64,

    /// Use quantization: quantize model weights
    #[arg(long, default_value = "false")]
    quantization: bool,

    /// Quantization bits: number of bits for quantization
    #[arg(long, default_value = "8")]
    quantization_bits: usize,

    /// Use pruning: prune model weights
    #[arg(long, default_value = "false")]
    pruning: bool,

    /// Pruning ratio: ratio of weights to prune
    #[arg(long, default_value = "0.5")]
    pruning_ratio: f64,

    /// Use knowledge distillation: distill from larger model
    #[arg(long, default_value = "false")]
    distillation: bool,

    /// Distillation temperature: temperature for soft targets
    #[arg(long, default_value = "1.0")]
    distillation_temp: f64,

    /// Distillation weight: weight for distillation loss
    #[arg(long, default_value = "0.5")]
    distillation_weight: f64,

    /// Teacher model path: path to teacher model
    #[arg(long, default_value = "")]
    teacher_model: PathBuf,

    /// Use adversarial training: add adversarial perturbations
    #[arg(long, default_value = "false")]
    adversarial: bool,

    /// Adversarial epsilon: perturbation magnitude
    #[arg(long, default_value = "0.1")]
    adversarial_epsilon: f64,

    /// Use robust training: robust to distribution shift
    #[arg(long, default_value = "false")]
    robust: bool,

    /// Robust radius: radius for robust optimization
    #[arg(long, default_value = "0.1")]
    robust_radius: f64,

    /// Use safe RL: enforce safety constraints
    #[arg(long, default_value = "false")]
    safe_rl: bool,

    /// Safe constraint threshold: safety threshold
    #[arg(long, default_value = "0.0")]
    safe_threshold: f64,

    /// Use constrained RL: optimize with constraints
    #[arg(long, default_value = "false")]
    constrained: bool,

    /// Constraint weights: constraint weights (comma-separated)
    #[arg(long, default_value = "1.0")]
    constraint_weights: String,

    /// Use hierarchical RL: hierarchical policy
    #[arg(long, default_value = "false")]
    hierarchical: bool,

    /// Hierarchy levels: number of hierarchy levels
    #[arg(long, default_value = "2")]
    hierarchy_levels: usize,

    /// Use option framework: options as temporally extended actions
    #[arg(long, default_value = "false")]
    options: bool,

    /// Number of options: number of options to learn
    #[arg(long, default_value = "10")]
    num_options: usize,

    /// Option horizon: option duration
    #[arg(long, default_value = "10")]
    option_horizon: usize,

    /// Use skill discovery: discover reusable skills
    #[arg(long, default_value = "false")]
    skill_discovery: bool,

    /// Number of skills: number of skills to discover
    #[arg(long, default_value = "20")]
    num_skills: usize,

    /// Skill diversity: diversity bonus for skills
    #[arg(long, default_value = "0.1")]
    skill_diversity: f64,

    /// Use transfer learning: transfer from source task
    #[arg(long, default_value = "false")]
    transfer: bool,

    /// Transfer source path: path to source model
    #[arg(long, default_value = "")]
    transfer_source: PathBuf,

    /// Transfer freeze layers: number of layers to freeze
    #[arg(long, default_value = "0")]
    transfer_freeze: usize,

    /// Use few-shot learning: learn from few examples
    #[arg(long, default_value = "false")]
    few_shot: bool,

    /// Few-shot ways: number of ways for few-shot
    #[arg(long, default_value = "5")]
    few_shot_ways: usize,

    /// Few-shot shots: number of shots per way
    #[arg(long, default_value = "1")]
    few_shot_shots: usize,

    /// Use zero-shot learning: generalize without examples
    #[arg(long, default_value = "false")]
    zero_shot: bool,

    /// Zero-shot prompts: prompts for zero-shot (comma-separated)
    #[arg(long, default_value = "")]
    zero_shot_prompts: String,

    /// Use contrastive learning: contrastive representation
    #[arg(long, default_value = "false")]
    contrastive: bool,

    /// Contrastive temperature: temperature for contrastive loss
    #[arg(long, default_value = "0.07")]
    contrastive_temp: f64,

    /// Contrastive queue size: queue size for MoCo
    #[arg(long, default_value = "65536")]
    contrastive_queue: usize,

    /// Use augmentation: data augmentation
    #[arg(long, default_value = "false")]
    augmentation: bool,

    /// Augmentation types: augmentation types (comma-separated)
    #[arg(long, default_value = "noise,dropout")]
    augmentation_types: String,

    /// Augmentation probability: augmentation probability
    #[arg(long, default_value = "0.5")]
    augmentation_prob: f64,

    /// Use normalization: input normalization
    #[arg(long, default_value = "true")]
    normalization: bool,

    /// Normalization type: batch, layer, instance
    #[arg(long, default_value = "layer")]
    normalization_type: String,

    /// Use standardization: standardize inputs
    #[arg(long, default_value = "true")]
    standardization: bool,

    /// Use clipping: clip input values
    #[arg(long, default_value = "true")]
    clipping: bool,

    /// Clip value: clipping threshold
    #[arg(long, default_value = "1.0")]
    clip_value: f64,

    /// Use scaling: scale input values
    #[arg(long, default_value = "true")]
    scaling: bool,

    /// Scale factor: scaling factor
    #[arg(long, default_value = "1.0")]
    scale_factor: f64,

    /// Use bias correction: correct bias in estimates
    #[arg(long, default_value = "true")]
    bias_correction: bool,

    /// Use reward scaling: scale rewards
    #[arg(long, default_value = "true")]
    reward_scaling: bool,

    /// Reward scale factor: reward scaling factor
    #[arg(long, default_value = "1.0")]
    reward_scale: f64,

    /// Use reward clipping: clip rewards
    #[arg(long, default_value = "false")]
    reward_clipping: bool,

    /// Reward clip value: reward clipping threshold
    #[arg(long, default_value = "1.0")]
    reward_clip: f64,

    /// Use reward normalization: normalize rewards
    #[arg(long, default_value = "false")]
    reward_normalization: bool,

    /// Use reward shaping: shape rewards
    #[arg(long, default_value = "false")]
    reward_shaping: bool,

    /// Shaping potential: potential function for shaping
    #[arg(long, default_value = "")]
    shaping_potential: String,

    /// Use intrinsic rewards: add intrinsic motivation
    #[arg(long, default_value = "false")]
    intrinsic_rewards: bool,

    /// Intrinsic weight: weight for intrinsic rewards
    #[arg(long, default_value = "0.1")]
    intrinsic_weight: f64,

    /// Intrinsic type: curiosity, novelty, information
    #[arg(long, default_value = "curiosity")]
    intrinsic_type: String,

    /// Use curiosity: curiosity-driven exploration
    #[arg(long, default_value = "false")]
    curiosity: bool,

    /// Curiosity weight: weight for curiosity bonus
    #[arg(long, default_value = "0.1")]
    curiosity_weight: f64,

    /// Use novelty: novelty-based exploration
    #[arg(long, default_value = "false")]
    novelty: bool,

    /// Novelty weight: weight for novelty bonus
    #[arg(long, default_value = "0.1")]
    novelty_weight: f64,

    /// Use information gain: information-based exploration
    #[arg(long, default_value = "false")]
    information_gain: bool,

    /// Information weight: weight for information bonus
    #[arg(long, default_value = "0.1")]
    information_weight: f64,

    /// Use empowerment: maximize empowerment
    #[arg(long, default_value = "false")]
    empowerment: bool,

    /// Empowerment weight: weight for empowerment
    #[arg(long, default_value = "0.1")]
    empowerment_weight: f64,

    /// Use causal discovery: discover causal structure
    #[arg(long, default_value = "false")]
    causal_discovery: bool,

    /// Causal horizon: horizon for causal discovery
    #[arg(long, default_value = "10")]
    causal_horizon: usize,

    /// Use world models: learn world model
    #[arg(long, default_value = "false")]
    world_model: bool,

    /// World model type: rssm, dreamer, planet
    #[arg(long, default_value = "rssm")]
    world_model_type: String,

    /// World model horizon: planning horizon
    #[arg(long, default_value = "15")]
    world_model_horizon: usize,

    /// Use latent dynamics: latent space dynamics
    #[arg(long, default_value = "false")]
    latent_dynamics: bool,

    /// Latent dim: latent space dimension
    #[arg(long, default_value = "32")]
    latent_dim: usize,

    /// Use discrete latent: discrete latent variables
    #[arg(long, default_value = "false")]
    discrete_latent: bool,

    /// Discrete categories: number of categories
    #[arg(long, default_value = "32")]
    discrete_categories: usize,

    /// Use continuous latent: continuous latent variables
    #[arg(long, default_value = "true")]
    continuous_latent: bool,

    /// Continuous dim: continuous latent dimension
    #[arg(long, default_value = "32")]
    continuous_dim: usize,

    /// Use stochastic latent: stochastic latent variables
    #[arg(long, default_value = "true")]
    stochastic_latent: bool,

    /// Stochastic dim: stochastic latent dimension
    #[arg(long, default_value = "32")]
    stochastic_dim: usize,

    /// Use deterministic latent: deterministic latent variables
    #[arg(long, default_value = "false")]
    deterministic_latent: bool,

    /// Deterministic dim: deterministic latent dimension
    #[arg(long, default_value = "32")]
    deterministic_dim: usize,

    /// Use recurrent latent: recurrent latent variables
    #[arg(long, default_value = "true")]
    recurrent_latent: bool,

    /// Recurrent type: gru, lstm, simple
    #[arg(long, default_value = "gru")]
    recurrent_type: String,

    /// Use attention latent: attention in latent space
    #[arg(long, default_value = "false")]
    attention_latent: bool,

    /// Attention latent heads: number of attention heads
    #[arg(long, default_value = "4")]
    attention_latent_heads: usize,

    /// Use transformer latent: transformer in latent space
    #[arg(long, default_value = "false")]
    transformer_latent: bool,

    /// Transformer layers: number of transformer layers
    #[arg(long, default_value = "2")]
    transformer_layers: usize,

    /// Use convolutional latent: convolutional in latent space
    #[arg(long, default_value = "false")]
    convolutional_latent: bool,

    /// Convolutional kernel: kernel size for conv
    #[arg(long, default_value = "3")]
    convolutional_kernel: usize,

    /// Use graph latent: graph neural network in latent space
    #[arg(long, default_value = "false")]
    graph_latent: bool,

    /// Graph nodes: number of graph nodes
    #[arg(long, default_value = "10")]
    graph_nodes: usize,

    /// Graph edges: number of graph edges
    #[arg(long, default_value = "20")]
    graph_edges: usize,

    /// Use memory latent: memory augmented latent space
    #[arg(long, default_value = "false")]
    memory_latent: bool,

    /// Memory size: memory size for memory augmented
    #[arg(long, default_value = "100")]
    memory_size: usize,

    /// Memory dim: memory dimension
    #[arg(long, default_value = "32")]
    memory_dim: usize,

    /// Use external memory: external memory access
    #[arg(long, default_value = "false")]
    external_memory: bool,

    /// External memory type: dnc, ntmm, simple
    #[arg(long, default_value = "dnc")]
    external_memory_type: String,

    /// Use internal memory: internal memory access
    #[arg(long, default_value = "true")]
    internal_memory: bool,

    /// Internal memory type: lstm, gru, simple
    #[arg(long, default_value = "lstm")]
    internal_memory_type: String,

    /// Use working memory: working memory module
    #[arg(long, default_value = "false")]
    working_memory: bool,

    /// Working memory size: working memory size
    #[arg(long, default_value = "10")]
    working_memory_size: usize,

    /// Use episodic memory: episodic memory module
    #[arg(long, default_value = "false")]
    episodic_memory: bool,

    /// Episodic memory size: episodic memory size
    #[arg(long, default_value = "1000")]
    episodic_memory_size: usize,

    /// Use semantic memory: semantic memory module
    #[arg(long, default_value = "false")]
    semantic_memory: bool,

    /// Semantic memory size: semantic memory size
    #[arg(long, default_value = "100")]
    semantic_memory_size: usize,

    /// Use procedural memory: procedural memory module
    #[arg(long, default_value = "false")]
    procedural_memory: bool,

    /// Procedural memory size: procedural memory size
    #[arg(long, default_value = "100")]
    procedural_memory_size: usize,

    /// Use sensory memory: sensory memory module
    #[arg(long, default_value = "false")]
    sensory_memory: bool,

    /// Sensory memory size: sensory memory size
    #[arg(long, default_value = "10")]
    sensory_memory_size: usize,

    /// Use short-term memory: short-term memory module
    #[arg(long, default_value = "true")]
    short_term_memory: bool,

    /// Short-term memory size: short-term memory size
    #[arg(long, default_value = "100")]
    short_term_memory_size: usize,

    /// Use long-term memory: long-term memory module
    #[arg(long, default_value = "true")]
    long_term_memory: bool,

    /// Long-term memory size: long-term memory size
    #[arg(long, default_value = "10000")]
    long_term_memory_size: usize,

    /// Use memory consolidation: consolidate memories
    #[arg(long, default_value = "false")]
    memory_consolidation: bool,

    /// Consolidation frequency: consolidation frequency
    #[arg(long, default_value = "1000")]
    consolidation_freq: usize,

    /// Use memory replay: replay memories
    #[arg(long, default_value = "true")]
    memory_replay: bool,

    /// Replay frequency: replay frequency
    #[arg(long, default_value = "100")]
    replay_freq: usize,

    /// Use memory prioritization: prioritize memories
    #[arg(long, default_value = "false")]
    memory_prioritization: bool,

    /// Prioritization type: recency, importance, surprise
    #[arg(long, default_value = "importance")]
    prioritization_type: String,

    /// Use memory compression: compress memories
    #[arg(long, default_value = "false")]
    memory_compression_flag: bool,

    /// Compression ratio: compression ratio for memories
    #[arg(long, default_value = "0.5")]
    memory_compression_ratio: f64,

    /// Use memory retrieval: retrieve memories
    #[arg(long, default_value = "true")]
    memory_retrieval: bool,

    /// Retrieval type: exact, approximate, semantic
    #[arg(long, default_value = "approximate")]
    retrieval_type: String,

    /// Use memory writing: write to memories
    #[arg(long, default_value = "true")]
    memory_writing: bool,

    /// Writing type: append, overwrite, merge
    #[arg(long, default_value = "append")]
    writing_type: String,

    /// Use memory reading: read from memories
    #[arg(long, default_value = "true")]
    memory_reading: bool,

    /// Reading type: sequential, random, attention
    #[arg(long, default_value = "attention")]
    reading_type: String,

    /// Use memory addressing: address memories
    #[arg(long, default_value = "true")]
    memory_addressing: bool,

    /// Addressing type: content, location, hybrid
    #[arg(long, default_value = "content")]
    addressing_type: String,

    /// Use memory gating: gate memory access
    #[arg(long, default_value = "true")]
    memory_gating: bool,

    /// Gating type: hard, soft, learned
    #[arg(long, default_value = "learned")]
    gating_type: String,

    /// Use memory binding: bind memories
    #[arg(long, default_value = "false")]
    memory_binding: bool,

    /// Binding type: temporal, spatial, semantic
    #[arg(long, default_value = "temporal")]
    binding_type: String,

    /// Use memory unbinding: unbind memories
    #[arg(long, default_value = "false")]
    memory_unbinding: bool,

    /// Unbinding type: temporal, spatial, semantic
    #[arg(long, default_value = "temporal")]
    unbinding_type: String,

    /// Use memory association: associate memories
    #[arg(long, default_value = "true")]
    memory_association: bool,

    /// Association type: hebbian, correlation, attention
    #[arg(long, default_value = "hebbian")]
    association_type: String,

    /// Use memory disassociation: disassociate memories
    #[arg(long, default_value = "false")]
    memory_disassociation: bool,

    /// Disassociation type: hebbian, correlation, attention
    #[arg(long, default_value = "hebbian")]
    disassociation_type: String,

    /// Use memory integration: integrate memories
    #[arg(long, default_value = "false")]
    memory_integration: bool,

    /// Integration type: additive, multiplicative, attention
    #[arg(long, default_value = "additive")]
    integration_type: String,

    /// Use memory segregation: segregate memories
    #[arg(long, default_value = "false")]
    memory_segregation: bool,

    /// Segregation type: temporal, spatial, semantic
    #[arg(long, default_value = "temporal")]
    segregation_type: String,

    /// Use memory generalization: generalize memories
    #[arg(long, default_value = "false")]
    memory_generalization: bool,

    /// Generalization type: abstraction, induction, deduction
    #[arg(long, default_value = "abstraction")]
    generalization_type: String,

    /// Use memory specialization: specialize memories
    #[arg(long, default_value = "false")]
    memory_specialization: bool,

    /// Specialization type: differentiation, refinement, adaptation
    #[arg(long, default_value = "differentiation")]
    specialization_type: String,

    /// Use memory abstraction: abstract memories
    #[arg(long, default_value = "false")]
    memory_abstraction: bool,

    /// Abstraction level: level of abstraction
    #[arg(long, default_value = "1")]
    abstraction_level: usize,

    /// Use memory concretization: concretize memories
    #[arg(long, default_value = "false")]
    memory_concretization: bool,

    /// Concretization level: level of concretization
    #[arg(long, default_value = "1")]
    concretization_level: usize,

    /// Use memory synthesis: synthesize memories
    #[arg(long, default_value = "false")]
    memory_synthesis: bool,

    /// Synthesis type: combination, transformation, generation
    #[arg(long, default_value = "combination")]
    synthesis_type: String,

    /// Use memory analysis: analyze memories
    #[arg(long, default_value = "false")]
    memory_analysis: bool,

    /// Analysis type: decomposition, comparison, evaluation
    #[arg(long, default_value = "decomposition")]
    analysis_type: String,

    /// Use memory evaluation: evaluate memories
    #[arg(long, default_value = "false")]
    memory_evaluation: bool,

    /// Evaluation type: accuracy, relevance, utility
    #[arg(long, default_value = "accuracy")]
    evaluation_type: String,

    /// Use memory creation: create memories
    #[arg(long, default_value = "true")]
    memory_creation: bool,

    /// Creation type: encoding, learning, inference
    #[arg(long, default_value = "encoding")]
    creation_type: String,

    /// Use memory destruction: destroy memories
    #[arg(long, default_value = "false")]
    memory_destruction: bool,

    /// Destruction type: forgetting, pruning, decay
    #[arg(long, default_value = "forgetting")]
    destruction_type: String,

    /// Use memory modification: modify memories
    #[arg(long, default_value = "true")]
    memory_modification: bool,

    /// Modification type: updating, editing, transforming
    #[arg(long, default_value = "updating")]
    modification_type: String,

    /// Use memory preservation: preserve memories
    #[arg(long, default_value = "false")]
    memory_preservation: bool,

    /// Preservation type: consolidation, rehearsal, protection
    #[arg(long, default_value = "consolidation")]
    preservation_type: String,

    /// Use memory enhancement: enhance memories
    #[arg(long, default_value = "false")]
    memory_enhancement: bool,

    /// Enhancement type: strengthening, amplification, refinement
    #[arg(long, default_value = "strengthening")]
    enhancement_type: String,

    /// Use memory impairment: impair memories
    #[arg(long, default_value = "false")]
    memory_impairment: bool,

    /// Impairment type: weakening, suppression, interference
    #[arg(long, default_value = "weakening")]
    impairment_type: String,

    /// Use memory recovery: recover memories
    #[arg(long, default_value = "false")]
    memory_recovery: bool,

    /// Recovery type: retrieval, reconstruction, restoration
    #[arg(long, default_value = "retrieval")]
    recovery_type: String,

    /// Use memory loss: lose memories
    #[arg(long, default_value = "false")]
    memory_loss: bool,

    /// Loss type: decay, interference, suppression
    #[arg(long, default_value = "decay")]
    loss_type_flag: String,

    /// Use memory gain: gain memories
    #[arg(long, default_value = "true")]
    memory_gain: bool,

    /// Gain type: learning, encoding, inference
    #[arg(long, default_value = "learning")]
    gain_type: String,

    /// Use memory transfer: transfer memories
    #[arg(long, default_value = "false")]
    memory_transfer_flag: bool,

    /// Transfer type: forward, backward, lateral
    #[arg(long, default_value = "forward")]
    transfer_type: String,

    /// Use memory interference: interfere with memories
    #[arg(long, default_value = "false")]
    memory_interference: bool,

    /// Interference type: proactive, retroactive, lateral
    #[arg(long, default_value = "proactive")]
    interference_type: String,

    /// Use memory facilitation: facilitate memories
    #[arg(long, default_value = "false")]
    memory_facilitation: bool,

    /// Facilitation type: priming, cueing, context
    #[arg(long, default_value = "priming")]
    facilitation_type: String,

    /// Use memory inhibition: inhibit memories
    #[arg(long, default_value = "false")]
    memory_inhibition: bool,

    /// Inhibition type: suppression, blocking, interference
    #[arg(long, default_value = "suppression")]
    inhibition_type: String,

    /// Use memory activation: activate memories
    #[arg(long, default_value = "true")]
    memory_activation: bool,

    /// Activation type: priming, cueing, context
    #[arg(long, default_value = "priming")]
    activation_type: String,

    /// Use memory deactivation: deactivate memories
    #[arg(long, default_value = "false")]
    memory_deactivation: bool,

    /// Deactivation type: suppression, blocking, interference
    #[arg(long, default_value = "suppression")]
    deactivation_type: String,

    /// Use memory accessibility: access memories
    #[arg(long, default_value = "true")]
    memory_accessibility: bool,

    /// Accessibility type: availability, retrievability, usability
    #[arg(long, default_value = "availability")]
    accessibility_type: String,

    /// Use memory inaccessibility: make memories inaccessible
    #[arg(long, default_value = "false")]
    memory_inaccessibility: bool,

    /// Inaccessibility type: unavailability, unretrievability, unusability
    #[arg(long, default_value = "unavailability")]
    inaccessibility_type: String,

    /// Use memory availability: make memories available
    #[arg(long, default_value = "true")]
    memory_availability: bool,

    /// Availability type: presence, accessibility, retrievability
    #[arg(long, default_value = "presence")]
    availability_type: String,

    /// Use memory unavailability: make memories unavailable
    #[arg(long, default_value = "false")]
    memory_unavailability: bool,

    /// Unavailability type: absence, inaccessibility, unretrievability
    #[arg(long, default_value = "absence")]
    unavailability_type: String,
}

fn to_trace_format(format: &TraceFormat) -> eris::TraceFormat {
    match format {
        TraceFormat::Autodetect => eris::TraceFormat::Autodetect,
        TraceFormat::Recorder => eris::TraceFormat::Recorder,
        TraceFormat::Dftracer => eris::TraceFormat::Dftracer,
    }
}

fn to_log_level(level: &LogLevel) -> Level {
    match level {
        LogLevel::Trace => Level::TRACE,
        LogLevel::Debug => Level::DEBUG,
        LogLevel::Info => Level::INFO,
        LogLevel::Warn => Level::WARN,
        LogLevel::Error => Level::ERROR,
    }
}

fn main() {
    // Use 64MB thread stack to prevent stack overflow
    // Burn's deeply nested generic types require large stack frames
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024) // 64 MB stack
        .spawn(|| {
            let args = Args::parse();

            // Initialize logging with user-specified level
            let level = to_log_level(&args.log_level);
            let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");

            println!("=== Eris Model Training V2 (Optimized) ===");
            println!("Stack size: 64 MB");
            println!("Model allocation: Heap (Box::new)");
            println!();
            println!("Configuration:");
            println!("  Model: {:?}", args.model);
            println!("  Episodes: {}", args.episodes);
            println!("  Max steps: {}", args.max_steps);
            println!("  Batch size: {}", args.batch_size);
            println!("  Backend: {}", args.backend);
            println!("  Trace file: {:?}", args.trace_file);
            println!();

            // TODO: Implement training dispatch
            // The actual training loop will be implemented by a future Meeseeks
            // For now, this skeleton provides:
            // - 64MB thread stack (prevents stack overflow)
            // - Box::new() for heap-allocated models
            // - Full argument parsing
            // - Logging setup
            //
            // Next steps:
            // 1. Add device selection based on --backend
            // 2. Add model initialization with Box::new()
            // 3. Add training loop dispatch
            // 4. Add checkpoint loading/saving
            // 5. Add progress reporting

            tracing::info!("Training skeleton ready - dispatch not yet implemented");

            // Placeholder to satisfy compiler
            let _backend = &args.backend;
            let _config_path = &args.config;
            let _trace_path = &args.trace_file;
            let _trace_format = to_trace_format(&args.trace_format);

            // TODO: Add actual training dispatch here
            // Example pattern from train_inference:
            // let model = Box::new(
            //     eris::models::CombinedModelConfig::new(...)
            //         .init(&device)
            // );

            println!("Training complete (skeleton)");
        })
        .unwrap()
        .join()
        .unwrap();
}
