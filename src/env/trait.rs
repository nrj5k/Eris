use crate::space::{BoxSpace, DiscreteSpace};
use std::collections::HashMap;

/// Result of a step in the environment.
///
/// Contains the outcome of taking an action in the environment,
/// similar to OpenAI Gym's step result format.
///
/// # Fields
///
/// * `observation` - The new state after taking the action
/// * `reward` - The reward received for the transition
/// * `done` - Whether the episode has ended
/// * `info` - Additional diagnostic information
///
/// # Example
///
/// ```
/// use eris::env::{StepResult, Info};
///
/// let result = StepResult {
///     observation: vec![1.0, 2.0, 3.0],
///     reward: 10.0,
///     done: false,
///     info: Info::new().with_metric("step", 10.0),
/// };
///
/// assert_eq!(result.observation.len(), 3);
/// assert_eq!(result.reward, 10.0);
/// assert!(!result.done);
/// ```
#[derive(Debug, Clone)]
pub struct StepResult {
    /// The observation after taking the action
    pub observation: Vec<f64>,
    /// The reward for taking the action
    pub reward: f64,
    /// Whether the episode is done
    pub done: bool,
    /// Additional information (metrics, diagnostics, etc.)
    pub info: Info,
}

/// Additional information returned by environment step.
///
/// Contains key-value pairs of metrics and diagnostic information
/// that don't fit in the standard observation/reward/done interface.
///
/// # Example
///
/// ```
/// use eris::env::Info;
///
/// // Create empty info
/// let info = Info::new();
///
/// // Add metrics using builder pattern
/// let info = Info::new()
///     .with_metric("latency_ms", 15.5)
///     .with_metric("throughput", 100.0);
///
/// assert_eq!(info.metrics.len(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Info {
    /// Key-value metrics from the environment
    pub metrics: HashMap<String, f64>,
}

impl Info {
    /// Create an empty Info struct
    ///
    /// # Example
    ///
    /// ```
    /// use eris::env::Info;
    /// let info = Info::new();
    /// assert!(info.metrics.is_empty());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a metric to the info
    ///
    /// # Arguments
    ///
    /// * `key` - The metric name
    /// * `value` - The metric value
    ///
    /// # Returns
    ///
    /// Self for method chaining
    ///
    /// # Example
    ///
    /// ```
    /// use eris::env::Info;
    ///
    /// let info = Info::new()
    ///     .with_metric("latency", 10.5)
    ///     .with_metric("throughput", 100.0);
    ///
    /// assert!(info.metrics.contains_key("latency"));
    /// assert_eq!(info.metrics.get("throughput"), Some(&100.0));
    /// ```
    pub fn with_metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(key.into(), value);
        self
    }
}

/// Trait for RL environments.
///
/// Defines the interface for reinforcement learning environments,
/// similar to OpenAI Gym's Env interface.
///
/// # Type Parameters
///
/// * `Observation` - The observation type (typically `Vec<f64>`)
/// * `Action` - The action type (typically `usize` for discrete actions)
///
/// # Environment Lifecycle
///
/// ```
/// use eris::env::Environment;
/// use eris::training::MockEnv;
///
/// let mut env = MockEnv::new_with_dims(10, 50, 20);
///
/// // Reset to start episode
/// let observation = env.reset();
///
/// // Take steps in environment
/// let action = 0;
/// let result = env.step(action);
/// println!("Reward: {}", result.reward);
///
/// // Episode ends when done is true
/// if result.done {
///     env.reset(); // Start new episode
/// }
/// ```
///
/// # Dynamic Dimensions
///
/// The trait supports dynamic dimensions instead of hardcoded constants:
///
/// ```
/// use eris::env::Environment;
/// use eris::space::Space;
///
/// fn get_environment_dims<E: Environment>(env: &E) -> (usize, usize) {
///     let obs_dim = env.observation_space().dim();
/// let action_dim = env.action_space().n;
///     (obs_dim, action_dim)
/// }
/// ```
pub trait Environment {
    /// The observation type
    type Observation;
    /// The action type
    type Action;

    /// Reset the environment and return the initial observation.
    ///
    /// # Returns
    ///
    /// The initial observation for a new episode.
    ///
    /// # Example
    ///
    /// ```
    /// use eris::env::Environment;
    /// use eris::training::MockEnv;
    ///
    /// let mut env = MockEnv::new(10);
    /// let observation = env.reset();
    /// ```
    fn reset(&mut self) -> Self::Observation;

    /// Get tier utilization states [0.0, 1.0] for visualization.
    ///
    /// # Returns
    ///
    /// Vector of tier utilization values, or empty vector if not applicable.
    fn get_tier_utilization(&self) -> Vec<f32> {
        Vec::new() // Default implementation returns empty vector
    }

    /// Take a step in the environment.
    ///
    /// # Arguments
    ///
    /// * `action` - The action to take
    ///
    /// # Returns
    ///
    /// A [`StepResult`] containing:
    /// - `observation`: The new observation
    /// - `reward`: The reward for taking the action
    /// - `done`: Whether the episode is finished
    /// - `info`: Additional information
    ///
    /// # Example
    ///
    /// ```
    /// use eris::env::{Environment, StepResult};
    /// use eris::training::MockEnv;
    ///
    /// let mut env = MockEnv::new(10);
    /// let _ = env.reset();
    ///
    /// let result: StepResult = env.step(0);
    /// println!("Observation: {:?}", result.observation);
    /// println!("Reward: {}", result.reward);
    /// ```
    fn step(&mut self, action: Self::Action) -> StepResult;

    /// Get the observation space.
    ///
    /// This defines the shape and bounds of observations.
    ///
    /// # Returns
    ///
    /// A [`BoxSpace`] defining valid observation ranges.
    ///
    /// # Example
    ///
    /// ```
    /// use eris::env::Environment;
    /// use eris::space::Space;
    /// use eris::training::MockEnv;
    ///
    /// let env = MockEnv::new_with_dims(10, 50, 20);
    /// let obs_space = env.observation_space();
    /// assert_eq!(obs_space.dim(), 50);
    /// ```
    fn observation_space(&self) -> BoxSpace;

    /// Get the action space.
    ///
    /// This defines the number and type of possible actions.
    ///
    /// # Returns
    ///
    /// A [`DiscreteSpace`] defining valid action indices.
    ///
    /// # Example
    ///
    /// ```
    /// use eris::env::Environment;
    /// use eris::space::Space;
    /// use eris::training::MockEnv;
    ///
    /// let env = MockEnv::new_with_dims(10, 50, 20);
    /// let action_space = env.action_space();
    /// assert_eq!(action_space.n, 20);
    /// ```
    fn action_space(&self) -> DiscreteSpace;

    /// Set the random seed for reproducibility.
    ///
    /// # Arguments
    ///
    /// * `seed` - The random seed
    ///
    /// # Example
    ///
    /// ```
    /// use eris::env::Environment;
    /// use eris::training::MockEnv;
    ///
    /// let mut env = MockEnv::new(10);
    /// env.seed(42);
    /// ```
    fn seed(&mut self, seed: u64);

    /// Render the environment (optional).
    ///
    /// This is optional and can be used for visualization.
    /// Default implementation is a no-op.
    fn render(&mut self) {
        // Default: no-op
    }

    /// Close the environment and cleanup resources (optional).
    /// Default implementation is a no-op.
    fn close(&mut self) {
        // Default: no-op
    }
}
