//! Model abstraction and default configurations for the Eris RL training system
//!
//! This module provides a three-tier configuration API:
//!
//! ## TIER 1: Defaults (Immediate Usability)
//! Use `ErisDefaults::storage_tier_model()` for out-of-the-box working configurations.
//!
//! ```rust,ignore
//! let config = ErisDefaults::storage_tier_model(state_dim, action_dim);
//! let device = burn::backend::wgpu::Wgpu::default();
//! let model = config.init::<Wgpu>(&device);
//! ```
//!
//! ## TIER 2: Builder Pattern (Clear Customization)
//! Use builder pattern for explicit configuration with validation.
//!
//! ```rust,ignore
//! // Get dimensions from environment
//! let env = MockEnv::new_with_dims(100, 50, 20);
//! let obs_dim = env.observation_space().dim();
//! let action_dim = env.action_space().n;
//!
//! let bandit_config = BanditConfig::builder()
//!     .input_dim(obs_dim)
//!     .hidden_layers(vec![64, 128])
//!     .feature_dim(20)
//!     .activation(Activation::Sigmoid)
//!     .build()?;
//!
//! let dqn_config = DQNConfig::builder()
//!     .input_dim(20)
//!     .hidden_layers(vec![128, 128])
//!     .action_dim(action_dim)
//!     .dueling(true)
//!     .build()?;
//! ```
//!
//! ## TIER 3: Model Trait (Extensibility)
//! Implement the `Model` trait for custom model architectures.
//!
//! # Architecture for Storage Tier Optimization
//!
//! The default architecture is optimized for multi-tier storage decision making:
//!
//! **Bandit Network: Linear(15→64→128→20) + Sigmoid**
//! - Input: State dimension (5 tier sizes + 10 blob features = 15)
//! - Output: Feature dimension (20D feature vector) + Importance score [0,1]
//! - Purpose: Compress state into meaningful features and compute importance
//!
//! **DQN Network: Linear(20→128→128→10) with Dueling Architecture**
//! - Input: Feature dimension from bandit (20D)
//! - Hidden: 128 units per layer
//! - Output: Q-values for actions (10 = 5 tiers × 2 operations)
//! - Dueling: Value(128→1) + Advantage(128→10)
//! - Purpose: Estimate action values for storage optimization

use crate::error::Result;
use burn::prelude::*;
use std::path::Path;

/// Activation function variants for neural networks
///
/// Supports common activation functions used in deep learning:
/// - ReLU: Rectified Linear Unit (fast, effective)
/// - Sigmoid: Output in [0,1] range (for importance scores)
/// - Tanh: Output in [-1,1] range (zero-centered)
/// - LeakyReLU: ReLU variant with small slope for negative values
///
/// # Example
///
/// ```
/// use eris::model::Activation;
///
/// let relu = Activation::ReLU;
/// let sigmoid = Activation::Sigmoid;
/// let leaky_relu = Activation::LeakyReLU(0.01);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU(f32),
}

impl Default for Activation {
    fn default() -> Self {
        Activation::ReLU
    }
}

impl std::fmt::Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Activation::ReLU => write!(f, "ReLU"),
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Tanh => write!(f, "Tanh"),
            Activation::LeakyReLU(alpha) => write!(f, "LeakyReLU({})", alpha),
        }
    }
}

/// Model trait for extensible neural network architectures.
///
/// This trait defines the interface for models that can:
/// - Process states and produce outputs
/// - Select actions with exploration strategies
/// - Persist and load from disk
///
/// # Type Parameters
///
/// - `B`: Burn backend type (e.g., `Wgpu`, `NdArray`)
///
/// # Purpose
///
/// The `Model` trait enables:
/// - Custom neural network architectures
/// - Backend agnostic model code
/// - Model serialization/deserialization
///
/// # Example
///
/// ```rust,ignore
/// use eris::model::Model;
/// use burn::prelude::*;
///
/// struct CustomModel<B: Backend> {
///     // Custom fields
/// }
///
/// impl<B: Backend> Model<B> for CustomModel<B> {
///     type Config = CustomModelConfig;
///     type Action = usize;
///
///     fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
///         // Forward pass implementation
///     }
///
///     fn select_action(&self, state: Tensor<B, 2>, epsilon: f32) -> Self::Action {
///         // Action selection with epsilon-greedy
///     }
///
///     fn save(&self, path: &Path) -> Result<()> {
///         // Save model weights
///     }
///
///     fn load(path: &Path, config: &Self::Config) -> Result<Self> {
///         // Load model weights
///     }
/// }
/// ```
pub trait Model<B: Backend>: Send + Sync {
    /// Configuration type for initializing the model
    type Config: Send + Sync;

    /// Action type produced by the model
    type Action: Send + Sync;

    /// Forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `state` - Input state tensor [batch_size, state_dim]
    ///
    /// # Returns
    ///
    /// Output tensor (shape depends on model architecture)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let output = model.forward(state_tensor);
    /// ```
    fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Select an action using epsilon-greedy exploration.
    ///
    /// # Arguments
    ///
    /// * `state` - Input state tensor [1, state_dim]
    /// * `epsilon` - Exploration rate [0, 1]
    ///
    /// # Returns
    ///
    /// Selected action
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let action = model.select_action(state_tensor, 0.1);
    /// ```
    fn select_action(&self, state: Tensor<B, 2>, epsilon: f32) -> Self::Action;

    /// Save model weights to disk.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to save the model
    ///
    /// # Returns
    ///
    /// Result indicating success or failure
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// model.save(Path::new("model.bin"))?;
    /// ```
    fn save(&self, path: &Path) -> Result<()>;

    /// Load model weights from disk.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to load the model from
    /// * `config` - Model configuration
    ///
    /// # Returns
    ///
    /// Result containing loaded model or error
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = CustomModel::load(Path::new("model.bin"), &config)?;
    /// ```
    fn load(path: &Path, config: &Self::Config) -> Result<Self>
    where
        Self: Sized;
}

/// Default configurations for Eris RL models.
///
/// Provides pre-configured model architectures optimized for
/// specific use cases in storage tier optimization.
///
/// # Key Features
///
/// - **Storage Tier Model**: Optimized for multi-tier storage optimization
/// - **Compact Model**: Smaller architecture for faster training
/// - **Dynamic Dimensions**: Both configurations support runtime dimensions
///
/// # Example
///
/// ```
/// use eris::model::ErisDefaults;
/// use eris::training::MockEnv;
/// use eris::env::Environment;
/// use eris::space::Space;
///
/// // Create environment with dynamic dimensions
/// let env = MockEnv::new_with_dims(100, 50, 20);
/// let state_dim = env.observation_space().dim();
/// let action_dim = env.action_space().n;
///
/// // Create default config with dynamic dimensions
/// let config = ErisDefaults::storage_tier_model(state_dim, action_dim);
/// ```
pub struct ErisDefaults;

impl ErisDefaults {
    /// Create a combined bandit-DQN model configuration for storage tier optimization.
    ///
    /// This is a "just works" configuration optimized for:
    /// - Multi-tier storage systems (5 tiers)
    /// - Read/write operation decisions (2 operations)
    /// - State representation with tier capacities and blob features
    ///
    /// # Architecture Details
    ///
    /// **Bandit Network:**
    /// - Input: Dynamic (configured via dimensions parameter)
    /// - Hidden: [64, 128] units with ReLU activation
    /// - Feature output: 20 dimensions
    /// - Importance score: Sigmoid activation for [0, 1] range
    ///
    /// **DQN Network:**
    /// - Input: 20 dimensions (feature output from bandit)
    /// - Hidden: [128, 128] units
    /// - Output: Dynamic (configured via dimensions parameter)
    /// - Dueling architecture for better value estimation
    ///
    /// # Arguments
    ///
    /// * `state_dim` - State dimensionality (input to bandit network)
    /// * `action_dim` - Number of actions (output from DQN network)
    ///
    /// # Returns
    ///
    /// Combined configuration ready for initialization
    ///
    /// # Example
    /// ```rust,ignore
    /// use eris::model::ErisDefaults;
    /// use eris::training::MockEnv;
    /// use burn::backend::wgpu::Wgpu;
    ///
    /// let env = MockEnv::new_with_dims(100, 50, 20);
    /// let state_dim = env.observation_space().dim();
    /// let action_dim = env.action_space().n;
    ///
    /// let config = ErisDefaults::storage_tier_model(state_dim, action_dim);
    /// let device = Wgpu::default();
    /// let model = config.init::<Wgpu>(&device);
    /// ```
    pub fn storage_tier_model(
        state_dim: usize,
        action_dim: usize,
    ) -> crate::config::CombinedBanditDQNConfig {
        use crate::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};

        // Feature representation dimension
        const FEATURE_DIM: usize = 20; // Enhanced feature representation

        // Architecture optimized for storage optimization
        const BANDIT_HIDDEN: &[usize] = &[64, 128];
        const DQN_HIDDEN: &[usize] = &[128, 128];

        let bandit_config = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(BANDIT_HIDDEN.to_vec())
            .feature_dim(FEATURE_DIM)
            .activation(Activation::Sigmoid) // For importance score [0, 1]
            .build()
            .expect("Default bandit config should be valid");

        let dqn_config = DQNConfig::builder()
            .input_dim(FEATURE_DIM)
            .hidden_layers(DQN_HIDDEN.to_vec())
            .action_dim(action_dim)
            .dueling(true) // Enable dueling DQN architecture
            .build()
            .expect("Default DQN config should be valid");

        CombinedBanditDQNConfig::builder()
            .bandit(bandit_config)
            .dqn(dqn_config)
            .build()
            .expect("Default combined config should be valid")
    }

    /// Create a compact model for testing or smaller deployments.
    ///
    /// Uses smaller hidden layers for faster inference and training.
    ///
    /// # Arguments
    ///
    /// * `state_dim` - State dimensionality (input to bandit network)
    /// * `action_dim` - Number of actions (output from DQN network)
    ///
    /// # Returns
    ///
    /// Compact combined configuration
    ///
    /// # Example
    /// ```rust,ignore
    /// use eris::model::ErisDefaults;
    /// use eris::training::MockEnv;
    ///
    /// let env = MockEnv::new_with_dims(100, 50, 20);
    /// let state_dim = env.observation_space().dim();
    /// let action_dim = env.action_space().n;
    ///
    /// let config = ErisDefaults::compact_model(state_dim, action_dim);
    /// ```
    pub fn compact_model(
        state_dim: usize,
        action_dim: usize,
    ) -> crate::config::CombinedBanditDQNConfig {
        use crate::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};

        // Compact feature dimension
        const FEATURE_DIM: usize = 10;

        let bandit_config = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(vec![32])
            .feature_dim(FEATURE_DIM)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Compact bandit config should be valid");

        let dqn_config = DQNConfig::builder()
            .input_dim(FEATURE_DIM)
            .hidden_layers(vec![32])
            .action_dim(action_dim)
            .dueling(false) // Simpler architecture for compact model
            .build()
            .expect("Compact DQN config should be valid");

        CombinedBanditDQNConfig::builder()
            .bandit(bandit_config)
            .dqn(dqn_config)
            .build()
            .expect("Compact combined config should be valid")
    }

    /// Create a large capacity model for complex storage behaviors.
    ///
    /// Optimized for high-compute production environments with:
    /// - Complex storage tier patterns
    /// - Large-scale blob management
    /// - High-accuracy decision making
    ///
    /// # Use Case
    ///
    /// Ideal for production systems where:
    /// - Accuracy is critical, latency is acceptable
    /// - Training time is not a constraint
    /// - Storage decisions have high business impact
    /// - GPUs with >8GB VRAM are available
    ///
    /// # Architecture
    ///
    /// ```text
    /// Bandit: [state_dim] → [128, 256] → [32]
    /// DQN:    [32] → [256, 256] → [action_dim]
    /// ```
    ///
    /// **Bandit Network:**
    /// - Input: state_dim (dynamic)
    /// - Hidden: [128, 256] units with ReLU
    /// - Feature output: 32 dimensions
    /// - Importance score: Sigmoid [0, 1]
    ///
    /// **DQN Network:**
    /// - Input: 32 features
    /// - Hidden: [256, 256] units
    /// - Output: action_dim (dynamic)
    /// - Dueling: Enabled
    ///
    /// # When to Use
    ///
    /// - Production systems with high compute budget
    /// - Complex tier interactions (5+ tiers)
    /// - Historical data available for training
    /// - Batch processing (not real-time)
    ///
    /// # When NOT to Use
    ///
    /// - Real-time inference requirements (<10ms)
    /// - Memory-constrained environments (<4GB RAM)
    /// - Simple storage patterns
    /// - Limited training data
    ///
    /// # Arguments
    ///
    /// * `state_dim` - State dimensionality (input to bandit network)
    /// * `action_dim` - Number of actions (output from DQN network)
    ///
    /// # Returns
    ///
    /// Large capacity configuration
    ///
    /// # Example
    ///
    /// ```
    /// use eris::model::ErisDefaults;
    ///
    /// let config = ErisDefaults::large_model(15, 10);
    /// ```
    pub fn large_model(
        state_dim: usize,
        action_dim: usize,
    ) -> crate::config::CombinedBanditDQNConfig {
        use crate::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};

        const FEATURE_DIM: usize = 32;
        const BANDIT_HIDDEN: &[usize] = &[128, 256];
        const DQN_HIDDEN: &[usize] = &[256, 256];

        let bandit_config = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(BANDIT_HIDDEN.to_vec())
            .feature_dim(FEATURE_DIM)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Large bandit config should be valid");

        let dqn_config = DQNConfig::builder()
            .input_dim(FEATURE_DIM)
            .hidden_layers(DQN_HIDDEN.to_vec())
            .action_dim(action_dim)
            .dueling(true)
            .build()
            .expect("Large DQN config should be valid");

        CombinedBanditDQNConfig::builder()
            .bandit(bandit_config)
            .dqn(dqn_config)
            .build()
            .expect("Large combined config should be valid")
    }

    /// Create a fast inference model for real-time decisions.
    ///
    /// Optimized for minimal latency in production systems with:
    /// - Sub-millisecond inference time
    /// - Real-time decision requirements
    /// - Low memory footprint
    /// - Edge deployment scenarios
    ///
    /// # Use Case
    ///
    /// Ideal for systems where:
    /// - Latency is critical (<5ms per decision)
    /// - Real-time responsiveness required
    /// - Memory is limited (<2GB)
    /// - High throughput needed (>10K decisions/sec)
    ///
    /// # Architecture
    ///
    /// ```text
    /// Bandit: [state_dim] → [64] → [16]
    /// DQN:    [16] → [64] → [action_dim]
    /// ```
    ///
    /// **Bandit Network:**
    /// - Input: state_dim (dynamic)
    /// - Hidden: [64] units with ReLU
    /// - Feature output: 16 dimensions
    /// - Importance score: Sigmoid [0, 1]
    ///
    /// **DQN Network:**
    /// - Input: 16 features
    /// - Hidden: [64] units
    /// - Output: action_dim (dynamic)
    /// - Dueling: Disabled (reduces compute)
    ///
    /// # When to Use
    ///
    /// - Real-time storage tier decisions
    /// - Edge devices or embedded systems
    /// - High-frequency workloads (>100 ops/sec)
    /// - Latency-sensitive applications
    ///
    /// # When NOT to Use
    ///
    /// - Complex tier interactions requiring deep reasoning
    /// - Training phase (use larger model)
    /// - Systems with ample compute resources
    /// - Accuracy-critical decisions
    ///
    /// # Arguments
    ///
    /// * `state_dim` - State dimensionality (input to bandit network)
    /// * `action_dim` - Number of actions (output from DQN network)
    ///
    /// # Returns
    ///
    /// Fast inference configuration
    ///
    /// # Example
    ///
    /// ```
    /// use eris::model::ErisDefaults;
    ///
    /// let config = ErisDefaults::fast_inference(15, 10);
    /// ```
    pub fn fast_inference(
        state_dim: usize,
        action_dim: usize,
    ) -> crate::config::CombinedBanditDQNConfig {
        use crate::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};

        const FEATURE_DIM: usize = 16;
        const BANDIT_HIDDEN: &[usize] = &[64];
        const DQN_HIDDEN: &[usize] = &[64];

        let bandit_config = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(BANDIT_HIDDEN.to_vec())
            .feature_dim(FEATURE_DIM)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Fast inference bandit config should be valid");

        let dqn_config = DQNConfig::builder()
            .input_dim(FEATURE_DIM)
            .hidden_layers(DQN_HIDDEN.to_vec())
            .action_dim(action_dim)
            .dueling(false)
            .build()
            .expect("Fast inference DQN config should be valid");

        CombinedBanditDQNConfig::builder()
            .bandit(bandit_config)
            .dqn(dqn_config)
            .build()
            .expect("Fast inference combined config should be valid")
    }

    /// Create a 3-tier storage system configuration.
    ///
    /// Optimized for standard 3-tier storage architectures:
    /// - Tier 0: Memory (fastest)
    /// - Tier 1: SSD (medium)
    /// - Tier 2: HDD (slowest)
    ///
    /// # Use Case
    ///
    /// Ideal for:
    /// - Standard cache hierarchy (Memory/SSD/HDD)
    /// - Hot/warm/cold data classification
    /// - Most common production deployments
    ///
    /// # Architecture
    ///
    /// ```text
    /// State:  [3 tiers + 10 blob features] = 13 dimensions
    /// Bandit: [13] → [64, 64] → [16]
    /// DQN:    [16] → [128, 64] → [6 actions]
    /// ```
    ///
    /// **State Representation:**
    /// - 3 tier capacity features (normalized 0-1)
    /// - 10 blob features (size, age, access patterns)
    /// - Total: 13 dimensions
    ///
    /// **Action Space:**
    /// - 3 tiers × 2 operations (promote/demote) = 6 actions
    ///
    /// # When to Use
    ///
    /// - Standard 3-tier storage systems
    /// - Memory-SSD-HDD architectures
    /// - Production deployments
    /// - Balanced latency/accuracy requirements
    ///
    /// # Returns
    ///
    /// Optimized configuration for 3-tier system
    ///
    /// # Example
    ///
    /// ```
    /// use eris::model::ErisDefaults;
    ///
    /// let config = ErisDefaults::three_tier_system();
    /// ```
    pub fn three_tier_system() -> crate::config::CombinedBanditDQNConfig {
        use crate::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};

        // Fixed dimensions for 3-tier system
        const STATE_DIM: usize = 13; // 3 tiers + 10 blob features
        const ACTION_DIM: usize = 6; // 3 tiers × 2 operations
        const FEATURE_DIM: usize = 16;
        const BANDIT_HIDDEN: &[usize] = &[64, 64];
        const DQN_HIDDEN: &[usize] = &[128, 64];

        let bandit_config = BanditConfig::builder()
            .input_dim(STATE_DIM)
            .hidden_layers(BANDIT_HIDDEN.to_vec())
            .feature_dim(FEATURE_DIM)
            .activation(Activation::Sigmoid)
            .build()
            .expect("3-tier bandit config should be valid");

        let dqn_config = DQNConfig::builder()
            .input_dim(FEATURE_DIM)
            .hidden_layers(DQN_HIDDEN.to_vec())
            .action_dim(ACTION_DIM)
            .dueling(true)
            .build()
            .expect("3-tier DQN config should be valid");

        CombinedBanditDQNConfig::builder()
            .bandit(bandit_config)
            .dqn(dqn_config)
            .build()
            .expect("3-tier combined config should be valid")
    }

    /// Create a 7-tier storage system configuration.
    ///
    /// Optimized for comprehensive storage hierarchies:
    /// - Tier 0: Memory
    /// - Tier 1: NVMe SSD
    /// - Tier 2: SATA SSD
    /// - Tier 3: HDD
    /// - Tier 4: Network Storage
    /// - Tier 5: Cold Storage
    /// - Tier 6: Glacier/Archive
    ///
    /// # Use Case
    ///
    /// Ideal for:
    /// - Enterprise storage systems
    /// - Cloud storage architectures
    /// - Multi-location data management
    /// - Compliance-driven data lifecycle
    ///
    /// # Architecture
    ///
    /// ```text
    /// State:  [7 tiers + 10 blob features] = 17 dimensions
    /// Bandit: [17] → [128, 128, 64] → [24]
    /// DQN:    [24] → [256, 128] → [14 actions]
    /// ```
    ///
    /// **State Representation:**
    /// - 7 tier capacity features (normalized 0-1)
    /// - 10 blob features (size, age, access, compliance)
    /// - Total: 17 dimensions
    ///
    /// **Action Space:**
    /// - 7 tiers × 2 operations (promote/demote) = 14 actions
    ///
    /// # When to Use
    ///
    /// - Enterprise-grade storage systems
    /// - Multi-tier cloud architectures
    /// - Data lifecycle management
    /// - Compliance requirements (data retention policies)
    ///
    /// # Returns
    ///
    /// Optimized configuration for 7-tier system
    ///
    /// # Example
    ///
    /// ```
    /// use eris::model::ErisDefaults;
    ///
    /// let config = ErisDefaults::seven_tier_system();
    /// ```
    pub fn seven_tier_system() -> crate::config::CombinedBanditDQNConfig {
        use crate::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};

        // Fixed dimensions for 7-tier system
        const STATE_DIM: usize = 17; // 7 tiers + 10 blob features
        const ACTION_DIM: usize = 14; // 7 tiers × 2 operations
        const FEATURE_DIM: usize = 24;
        const BANDIT_HIDDEN: &[usize] = &[128, 128, 64];
        const DQN_HIDDEN: &[usize] = &[256, 128];

        let bandit_config = BanditConfig::builder()
            .input_dim(STATE_DIM)
            .hidden_layers(BANDIT_HIDDEN.to_vec())
            .feature_dim(FEATURE_DIM)
            .activation(Activation::Sigmoid)
            .build()
            .expect("7-tier bandit config should be valid");

        let dqn_config = DQNConfig::builder()
            .input_dim(FEATURE_DIM)
            .hidden_layers(DQN_HIDDEN.to_vec())
            .action_dim(ACTION_DIM)
            .dueling(true)
            .build()
            .expect("7-tier DQN config should be valid");

        CombinedBanditDQNConfig::builder()
            .bandit(bandit_config)
            .dqn(dqn_config)
            .build()
            .expect("7-tier combined config should be valid")
    }

    /// Create a research model with fully configurable architecture.
    ///
    /// Provides maximum flexibility for experimentation:
    /// - Custom hidden layer sizes
    /// - Adjustable feature dimensions
    /// - Research and hyperparameter tuning
    ///
    /// # Use Case
    ///
    /// Ideal for:
    /// - Academic research
    /// - Hyperparameter optimization
    /// - Architecture search
    /// - Custom storage systems
    ///
    /// # Architecture
    ///
    /// ```text
    /// Bandit: [state_dim] → bandit_layers → [feature_dim]
    /// DQN:    [feature_dim] → dqn_layers → [action_dim]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `state_dim` - Input state dimensionality
    /// * `action_dim` - Number of output actions
    /// * `bandit_layers` - Hidden layer sizes for bandit network
    /// * `feature_dim` - Feature representation dimension
    /// * `dqn_layers` - Hidden layer sizes for DQN network
    ///
    /// # Returns
    ///
    /// Configurable research model
    ///
    /// # Example
    ///
    /// ```
    /// use eris::model::ErisDefaults;
    ///
    /// let config = ErisDefaults::research_model(
    ///     15,                      // state_dim
    ///     10,                      // action_dim
    ///     vec![64, 128, 64],       // bandit_layers
    ///     32,                      // feature_dim
    ///     vec![128, 128],          // dqn_layers
    /// );
    /// ```
    pub fn research_model(
        state_dim: usize,
        action_dim: usize,
        bandit_layers: Vec<usize>,
        feature_dim: usize,
        dqn_layers: Vec<usize>,
    ) -> crate::config::CombinedBanditDQNConfig {
        use crate::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};

        let bandit_config = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(bandit_layers)
            .feature_dim(feature_dim)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Research bandit config should be valid");

        let dqn_config = DQNConfig::builder()
            .input_dim(feature_dim)
            .hidden_layers(dqn_layers)
            .action_dim(action_dim)
            .dueling(true)
            .build()
            .expect("Research DQN config should be valid");

        CombinedBanditDQNConfig::builder()
            .bandit(bandit_config)
            .dqn(dqn_config)
            .build()
            .expect("Research combined config should be valid")
    }

    /// Create a wide model for learning more patterns.
    ///
    /// Uses wide layers to capture diverse feature interactions:
    /// - Broad pattern recognition
    /// - High-dimensional feature spaces
    /// - Complex state representations
    ///
    /// # Use Case
    ///
    /// Ideal for:
    /// - Diverse workloads with many patterns
    /// - Large state representations (>20 dims)
    /// - Systems with many blob features
    /// - When deep networks overfit
    ///
    /// # Architecture
    ///
    /// ```text
    /// Bandit: [state_dim] → [256] → [32]
    /// DQN:    [32] → [256, 128] → [action_dim]
    /// ```
    ///
    /// **Bandit Network:**
    /// - Input: state_dim (dynamic)
    /// - Hidden: [256] units (wide)
    /// - Feature output: 32 dimensions
    /// - Importance score: Sigmoid [0, 1]
    ///
    /// **DQN Network:**
    /// - Input: 32 features
    /// - Hidden: [256, 128] units
    /// - Output: action_dim (dynamic)
    /// - Dueling: Enabled
    ///
    /// # When to Use
    ///
    /// - High-dimensional state spaces (>15 dims)
    /// - Complex feature interactions
    /// - When deep networks converge slowly
    /// - Need to learn many patterns simultaneously
    ///
    /// # When NOT to Use
    ///
    /// - Simple storage patterns
    /// - Low-dimensional states
    /// - Memory-constrained environments
    ///
    /// # Arguments
    ///
    /// * `state_dim` - State dimensionality (input to bandit network)
    /// * `action_dim` - Number of actions (output from DQN network)
    ///
    /// # Returns
    ///
    /// Wide network configuration
    ///
    /// # Example
    ///
    /// ```
    /// use eris::model::ErisDefaults;
    ///
    /// let config = ErisDefaults::wide_model(20, 8);
    /// ```
    pub fn wide_model(
        state_dim: usize,
        action_dim: usize,
    ) -> crate::config::CombinedBanditDQNConfig {
        use crate::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};

        const FEATURE_DIM: usize = 32;
        const BANDIT_HIDDEN: &[usize] = &[256];
        const DQN_HIDDEN: &[usize] = &[256, 128];

        let bandit_config = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(BANDIT_HIDDEN.to_vec())
            .feature_dim(FEATURE_DIM)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Wide bandit config should be valid");

        let dqn_config = DQNConfig::builder()
            .input_dim(FEATURE_DIM)
            .hidden_layers(DQN_HIDDEN.to_vec())
            .action_dim(action_dim)
            .dueling(true)
            .build()
            .expect("Wide DQN config should be valid");

        CombinedBanditDQNConfig::builder()
            .bandit(bandit_config)
            .dqn(dqn_config)
            .build()
            .expect("Wide combined config should be valid")
    }

    /// Create a deep model for hierarchical feature extraction.
    ///
    /// Uses deep architecture for learning complex feature hierarchies:
    /// - Multi-level abstractions
    /// - Complex decision boundaries
    /// - Hierarchical pattern recognition
    ///
    /// # Use Case
    ///
    /// Ideal for:
    /// - Complex tier interaction patterns
    /// - Non-linear decision boundaries
    /// - Systems requiring multi-step reasoning
    /// - Advanced feature extraction needs
    ///
    /// # Architecture
    ///
    /// ```text
    /// Bandit: [state_dim] → [64, 128, 256] → [40]
    /// DQN:    [40] → [128, 128, 128] → [action_dim]
    /// ```
    ///
    /// **Bandit Network:**
    /// - Input: state_dim (dynamic)
    /// - Hidden: [64, 128, 256] units (progressive expansion)
    /// - Feature output: 40 dimensions
    /// - Importance score: Sigmoid [0, 1]
    ///
    /// **DQN Network:**
    /// - Input: 40 features
    /// - Hidden: [128, 128, 128] units
    /// - Output: action_dim (dynamic)
    /// - Dueling: Enabled
    ///
    /// # When to Use
    ///
    /// - Complex storage tier interactions
    /// - Need for hierarchical feature learning
    /// - Sufficient training data available
    /// - Longer training times acceptable
    ///
    /// # When NOT to Use
    ///
    /// - Simple linear decision patterns
    /// - Limited training data
    /// - Need for fast training
    /// - Real-time inference requirements
    ///
    /// # Arguments
    ///
    /// * `state_dim` - State dimensionality (input to bandit network)
    /// * `action_dim` - Number of actions (output from DQN network)
    ///
    /// # Returns
    ///
    /// Deep network configuration
    ///
    /// # Example
    ///
    /// ```
    /// use eris::model::ErisDefaults;
    ///
    /// let config = ErisDefaults::deep_model(15, 10);
    /// ```
    pub fn deep_model(
        state_dim: usize,
        action_dim: usize,
    ) -> crate::config::CombinedBanditDQNConfig {
        use crate::config::{BanditConfig, CombinedBanditDQNConfig, DQNConfig};

        const FEATURE_DIM: usize = 40;
        const BANDIT_HIDDEN: &[usize] = &[64, 128, 256];
        const DQN_HIDDEN: &[usize] = &[128, 128, 128];

        let bandit_config = BanditConfig::builder()
            .input_dim(state_dim)
            .hidden_layers(BANDIT_HIDDEN.to_vec())
            .feature_dim(FEATURE_DIM)
            .activation(Activation::Sigmoid)
            .build()
            .expect("Deep bandit config should be valid");

        let dqn_config = DQNConfig::builder()
            .input_dim(FEATURE_DIM)
            .hidden_layers(DQN_HIDDEN.to_vec())
            .action_dim(action_dim)
            .dueling(true)
            .build()
            .expect("Deep DQN config should be valid");

        CombinedBanditDQNConfig::builder()
            .bandit(bandit_config)
            .dqn(dqn_config)
            .build()
            .expect("Deep combined config should be valid")
    }
}

/// Configuration validation error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid dimensions: {0}")]
    InvalidDimensions(String),

    #[error("Layer configuration error: {0}")]
    LayerError(String),

    #[error("Missing required field: {0}")]
    MissingField(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that all model configurations can be created successfully
    #[test]
    fn test_all_configurations_create_successfully() {
        // Existing configurations
        let _ = ErisDefaults::storage_tier_model(15, 10);
        let _ = ErisDefaults::compact_model(15, 10);

        // New configurations
        let _ = ErisDefaults::large_model(15, 10);
        let _ = ErisDefaults::fast_inference(15, 10);
        let _ = ErisDefaults::three_tier_system();
        let _ = ErisDefaults::seven_tier_system();
        let _ = ErisDefaults::wide_model(15, 10);
        let _ = ErisDefaults::deep_model(15, 10);

        // Research model with custom parameters
        let _ = ErisDefaults::research_model(20, 12, vec![64, 128, 64], 32, vec![128, 256]);
    }

    /// Test storage_tier_model architecture
    #[test]
    fn test_storage_tier_model_architecture() {
        let config = ErisDefaults::storage_tier_model(15, 10);

        // Verify bandit network dimensions
        assert_eq!(config.bandit.input_dim, 15);
        assert_eq!(config.bandit.hidden_layers, vec![64, 128]);
        assert_eq!(config.bandit.feature_dim, 20);

        // Verify DQN network dimensions
        assert_eq!(config.dqn.input_dim, 20);
        assert_eq!(config.dqn.hidden_layers, vec![128, 128]);
        assert_eq!(config.dqn.action_dim, 10);
        assert!(config.dqn.dueling);
    }

    /// Test compact_model architecture
    #[test]
    fn test_compact_model_architecture() {
        let config = ErisDefaults::compact_model(15, 10);

        assert_eq!(config.bandit.input_dim, 15);
        assert_eq!(config.bandit.hidden_layers, vec![32]);
        assert_eq!(config.bandit.feature_dim, 10);

        assert_eq!(config.dqn.input_dim, 10);
        assert_eq!(config.dqn.hidden_layers, vec![32]);
        assert_eq!(config.dqn.action_dim, 10);
        assert!(!config.dqn.dueling);
    }

    /// Test large_model architecture
    #[test]
    fn test_large_model_architecture() {
        let config = ErisDefaults::large_model(20, 14);

        assert_eq!(config.bandit.input_dim, 20);
        assert_eq!(config.bandit.hidden_layers, vec![128, 256]);
        assert_eq!(config.bandit.feature_dim, 32);

        assert_eq!(config.dqn.input_dim, 32);
        assert_eq!(config.dqn.hidden_layers, vec![256, 256]);
        assert_eq!(config.dqn.action_dim, 14);
        assert!(config.dqn.dueling);
    }

    /// Test fast_inference architecture
    #[test]
    fn test_fast_inference_architecture() {
        let config = ErisDefaults::fast_inference(12, 6);

        assert_eq!(config.bandit.input_dim, 12);
        assert_eq!(config.bandit.hidden_layers, vec![64]);
        assert_eq!(config.bandit.feature_dim, 16);

        assert_eq!(config.dqn.input_dim, 16);
        assert_eq!(config.dqn.hidden_layers, vec![64]);
        assert_eq!(config.dqn.action_dim, 6);
        assert!(!config.dqn.dueling);
    }

    /// Test three_tier_system fixed dimensions
    #[test]
    fn test_three_tier_system_architecture() {
        let config = ErisDefaults::three_tier_system();

        // Fixed dimensions for 3-tier
        assert_eq!(config.bandit.input_dim, 13); // 3 tiers + 10 features
        assert_eq!(config.bandit.hidden_layers, vec![64, 64]);
        assert_eq!(config.bandit.feature_dim, 16);

        assert_eq!(config.dqn.input_dim, 16);
        assert_eq!(config.dqn.hidden_layers, vec![128, 64]);
        assert_eq!(config.dqn.action_dim, 6); // 3 tiers × 2 ops
        assert!(config.dqn.dueling);
    }

    /// Test seven_tier_system fixed dimensions
    #[test]
    fn test_seven_tier_system_architecture() {
        let config = ErisDefaults::seven_tier_system();

        // Fixed dimensions for 7-tier
        assert_eq!(config.bandit.input_dim, 17); // 7 tiers + 10 features
        assert_eq!(config.bandit.hidden_layers, vec![128, 128, 64]);
        assert_eq!(config.bandit.feature_dim, 24);

        assert_eq!(config.dqn.input_dim, 24);
        assert_eq!(config.dqn.hidden_layers, vec![256, 128]);
        assert_eq!(config.dqn.action_dim, 14); // 7 tiers × 2 ops
        assert!(config.dqn.dueling);
    }

    /// Test research_model with custom configuration
    #[test]
    fn test_research_model_custom_architecture() {
        let config = ErisDefaults::research_model(
            25,                  // state_dim
            8,                   // action_dim
            vec![64, 128, 64],   // bandit_layers
            28,                  // feature_dim
            vec![128, 256, 128], // dqn_layers
        );

        assert_eq!(config.bandit.input_dim, 25);
        assert_eq!(config.bandit.hidden_layers, vec![64, 128, 64]);
        assert_eq!(config.bandit.feature_dim, 28);

        assert_eq!(config.dqn.input_dim, 28);
        assert_eq!(config.dqn.hidden_layers, vec![128, 256, 128]);
        assert_eq!(config.dqn.action_dim, 8);
        assert!(config.dqn.dueling);
    }

    /// Test wide_model architecture
    #[test]
    fn test_wide_model_architecture() {
        let config = ErisDefaults::wide_model(18, 8);

        assert_eq!(config.bandit.input_dim, 18);
        assert_eq!(config.bandit.hidden_layers, vec![256]);
        assert_eq!(config.bandit.feature_dim, 32);

        assert_eq!(config.dqn.input_dim, 32);
        assert_eq!(config.dqn.hidden_layers, vec![256, 128]);
        assert_eq!(config.dqn.action_dim, 8);
        assert!(config.dqn.dueling);
    }

    /// Test deep_model architecture
    #[test]
    fn test_deep_model_architecture() {
        let config = ErisDefaults::deep_model(20, 12);

        assert_eq!(config.bandit.input_dim, 20);
        assert_eq!(config.bandit.hidden_layers, vec![64, 128, 256]);
        assert_eq!(config.bandit.feature_dim, 40);

        assert_eq!(config.dqn.input_dim, 40);
        assert_eq!(config.dqn.hidden_layers, vec![128, 128, 128]);
        assert_eq!(config.dqn.action_dim, 12);
        assert!(config.dqn.dueling);
    }

    /// Test model comparison: large vs compact
    #[test]
    fn test_model_size_comparison() {
        let state_dim = 15;
        let action_dim = 10;

        let compact = ErisDefaults::compact_model(state_dim, action_dim);
        let standard = ErisDefaults::storage_tier_model(state_dim, action_dim);
        let large = ErisDefaults::large_model(state_dim, action_dim);

        // Compare feature dimensions: compact < standard < large
        assert!(compact.bandit.feature_dim < standard.bandit.feature_dim);
        assert!(standard.bandit.feature_dim < large.bandit.feature_dim);

        // Compare bandit layer sizes: compact < standard < large
        assert!(compact.bandit.hidden_layers.len() < standard.bandit.hidden_layers.len());
        assert!(
            compact.bandit.hidden_layers.iter().sum::<usize>()
                < standard.bandit.hidden_layers.iter().sum::<usize>()
        );

        // Compare DQN layer sizes
        assert!(
            compact.dqn.hidden_layers.iter().sum::<usize>()
                < standard.dqn.hidden_layers.iter().sum::<usize>()
        );
        assert!(
            standard.dqn.hidden_layers.iter().sum::<usize>()
                < large.dqn.hidden_layers.iter().sum::<usize>()
        );
    }

    /// Test inference speed comparison
    #[test]
    fn test_inference_complexity_comparison() {
        // Fast inference should have fewer layers but reasonable features
        let fast = ErisDefaults::fast_inference(15, 10);
        let compact = ErisDefaults::compact_model(15, 10);
        let standard = ErisDefaults::storage_tier_model(15, 10);

        // Fast has minimal layers (1 bandit layer, 1 DQN layer)
        assert_eq!(fast.bandit.hidden_layers.len(), 1);
        assert_eq!(fast.dqn.hidden_layers.len(), 1);

        // Compact also has minimal layers
        assert_eq!(compact.bandit.hidden_layers.len(), 1);
        assert_eq!(compact.dqn.hidden_layers.len(), 1);

        // Fast has fewer or equal total neurons than standard
        let fast_total: usize = fast.bandit.hidden_layers.iter().sum::<usize>()
            + fast.dqn.hidden_layers.iter().sum::<usize>();
        let standard_total: usize = standard.bandit.hidden_layers.iter().sum::<usize>()
            + standard.dqn.hidden_layers.iter().sum::<usize>();
        assert!(fast_total <= standard_total);
    }

    /// Test activation function defaults
    #[test]
    fn test_activation_function_defaults() {
        // All models use Sigmoid for bandit importance score
        let configs = [
            ErisDefaults::storage_tier_model(15, 10),
            ErisDefaults::compact_model(15, 10),
            ErisDefaults::large_model(15, 10),
            ErisDefaults::fast_inference(15, 10),
            ErisDefaults::three_tier_system(),
            ErisDefaults::seven_tier_system(),
            ErisDefaults::research_model(15, 10, vec![64], 20, vec![64]),
            ErisDefaults::wide_model(15, 10),
            ErisDefaults::deep_model(15, 10),
        ];

        for config in &configs {
            assert!(matches!(config.bandit.activation, Activation::Sigmoid));
        }
    }

    /// Test dueling architecture usage
    #[test]
    fn test_dueling_architecture_usage() {
        // Models with dueling enabled
        let with_dueling = [
            ErisDefaults::storage_tier_model(15, 10),
            ErisDefaults::large_model(15, 10),
            ErisDefaults::three_tier_system(),
            ErisDefaults::seven_tier_system(),
            ErisDefaults::research_model(15, 10, vec![64], 20, vec![64]),
            ErisDefaults::wide_model(15, 10),
            ErisDefaults::deep_model(15, 10),
        ];

        for config in &with_dueling {
            assert!(config.dqn.dueling);
        }

        // Models with dueling disabled
        let without_dueling = [
            ErisDefaults::compact_model(15, 10),
            ErisDefaults::fast_inference(15, 10),
        ];

        for config in &without_dueling {
            assert!(!config.dqn.dueling);
        }
    }

    /// Test edge case: minimum dimensions
    #[test]
    fn test_minimum_dimensions() {
        // Should work with minimal valid dimensions
        let config = ErisDefaults::storage_tier_model(1, 1);
        assert_eq!(config.bandit.input_dim, 1);
        assert_eq!(config.dqn.action_dim, 1);
    }

    /// Test edge case: large dimensions
    #[test]
    fn test_large_dimensions() {
        // Should work with large dimensions
        let state_dim = 1000;
        let action_dim = 500;

        let config = ErisDefaults::storage_tier_model(state_dim, action_dim);
        assert_eq!(config.bandit.input_dim, state_dim);
        assert_eq!(config.dqn.action_dim, action_dim);
    }

    /// Test research_model with various layer configurations
    #[test]
    fn test_research_model_various_layers() {
        // Single layer
        let config1 = ErisDefaults::research_model(10, 5, vec![64], 16, vec![32]);
        assert_eq!(config1.bandit.hidden_layers, vec![64]);
        assert_eq!(config1.dqn.hidden_layers, vec![32]);

        // Multiple layers
        let config2 =
            ErisDefaults::research_model(10, 5, vec![64, 128, 256, 128], 32, vec![128, 256, 128]);
        assert_eq!(config2.bandit.hidden_layers.len(), 4);
        assert_eq!(config2.dqn.hidden_layers.len(), 3);
    }

    /// Test tier-specific models have correct state dimensions
    #[test]
    fn test_tier_system_state_dimensions() {
        // 3-tier: 3 tier features + 10 blob features = 13
        let three_tier = ErisDefaults::three_tier_system();
        assert_eq!(three_tier.bandit.input_dim, 13);

        // 7-tier: 7 tier features + 10 blob features = 17
        let seven_tier = ErisDefaults::seven_tier_system();
        assert_eq!(seven_tier.bandit.input_dim, 17);
    }

    /// Test tier-specific models have correct action dimensions
    #[test]
    fn test_tier_system_action_dimensions() {
        // 3-tier × 2 operations = 6 actions
        let three_tier = ErisDefaults::three_tier_system();
        assert_eq!(three_tier.dqn.action_dim, 6);

        // 7-tier × 2 operations = 14 actions
        let seven_tier = ErisDefaults::seven_tier_system();
        assert_eq!(seven_tier.dqn.action_dim, 14);
    }
}
