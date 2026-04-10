use crate::env::{Environment, Info, StepResult};
use crate::space::{BoxSpace, DiscreteSpace};
use crate::training::replay_buffer::Transition;

/// Mock environment for testing without external dependencies.
///
/// Provides a deterministic environment simulation for unit tests and development.
/// Supports configurable state and action spaces for flexible testing scenarios.
///
/// # Purpose
///
/// `MockEnv` is designed for:
/// - Unit testing without CSV file dependencies
/// - Development and prototyping
/// - Algorithm testing with controlled environments
/// - Performance benchmarking
///
/// # Dynamic Dimensions
///
/// Unlike older versions with hardcoded dimensions, `MockEnv` supports:
///
/// ```
/// use eris::training::MockEnv;
/// use eris::env::Environment;
/// use eris::space::Space;
///
/// // Create with custom dimensions
/// let mut env = MockEnv::new_with_dims(100, 50, 20);
///
/// // Get dimensions dynamically
/// let obs_dim = env.observation_space().dim();  // 50
/// let action_dim = env.action_space().n;       // 20
///
/// // Use Environment trait
/// let obs = env.reset();
/// assert_eq!(obs.len(), obs_dim);
///
/// let result = env.step(0);
/// println!("Reward: {}", result.reward);
/// ```
///
/// # API Compatibility
///
/// `MockEnv` supports both the legacy API and the Environment trait:
///
/// ```rust,ignore
/// // Legacy API (for backward compatibility)
/// let state = env.reset(); // Returns Vec<f64>
/// let (next_state, reward, done) = env.step(action);
/// let obs_dim = env.observation_space(); // Returns usize
/// let action_dim = env.action_space();   // Returns usize
///
/// // Environment trait API
/// let obs = <MockEnv as Environment>::reset(env);
/// let result: StepResult = <MockEnv as Environment>::step(env, action);
/// let obs_space = <MockEnv as Environment>::observation_space(env);
/// let action_space = <MockEnv as Environment>::action_space(env);
/// ```
pub struct MockEnv {
    state: Vec<f64>,
    pub step_count: usize,
    max_steps: usize,
    obs_dim: usize,
    num_actions: usize,
    seed: u64,
}

impl MockEnv {
    /// Create a new MockEnv with default dimensions (32 obs, 10 actions).
    ///
    /// # Arguments
    ///
    /// * `max_steps` - Maximum steps before episode ends
    ///
    /// # Backward Compatibility
    ///
    /// This maintains the original API with warp-aligned dimensions:
    /// - Observation dimension: 32 (warp size for GPU optimization)
    /// - Number of actions: 10
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::MockEnv;
    ///
    /// let env = MockEnv::new(10);
    /// assert_eq!(env.observation_dim(), 32);
    /// assert_eq!(env.num_actions(), 10);
    /// ```
    pub fn new(max_steps: usize) -> Self {
        Self::new_with_dims(max_steps, 32, 10)
    }

    /// Create a new MockEnv with custom dimensions.
    ///
    /// # Arguments
    ///
    /// * `max_steps` - Maximum steps before episode ends
    /// * `obs_dim` - Observation space dimensionality
    /// * `num_actions` - Number of discrete actions
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::MockEnv;
    /// use eris::env::Environment;
    /// use eris::space::Space;
    ///
    /// let mut env = MockEnv::new_with_dims(100, 50, 20);
    ///
    /// // Get dimensions
    /// let obs_dim = env.observation_space().dim();  // 50
    /// let action_dim = env.action_space().n;        // 20
    ///
    /// // Reset and step
    /// let obs = env.reset();
    /// assert_eq!(obs.len(), 50);
    ///
    /// let result = env.step(0);
    /// println!("Reward: {}", result.reward);
    /// ```
    pub fn new_with_dims(max_steps: usize, obs_dim: usize, num_actions: usize) -> Self {
        Self {
            state: vec![0.0; obs_dim],
            step_count: 0,
            max_steps,
            obs_dim,
            num_actions,
            seed: 42,
        }
    }

    /// Get observation dimension
    ///
    /// # Returns
    ///
    /// The number of dimensions in the observation space
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::MockEnv;
    /// let env = MockEnv::new_with_dims(10, 50, 20);
    /// assert_eq!(env.observation_dim(), 50);
    /// ```
    pub fn observation_dim(&self) -> usize {
        self.obs_dim
    }

    /// Get number of actions
    ///
    /// # Returns
    ///
    /// The number of discrete actions available
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::MockEnv;
    /// let env = MockEnv::new_with_dims(10, 50, 20);
    /// assert_eq!(env.num_actions(), 20);
    /// ```
    pub fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Legacy reset method for backward compatibility.
    ///
    /// Returns the observation vector directly (not via Environment trait).
    ///
    /// # Returns
    ///
    /// The initial observation vector
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::MockEnv;
    /// let mut env = MockEnv::new(10);
    /// let obs = env.reset();
    /// assert_eq!(obs.len(), 32);
    /// ```
    pub fn reset(&mut self) -> Vec<f64> {
        self.step_count = 0;
        self.state = vec![0.0; self.obs_dim];
        self.state.clone()
    }

    /// Legacy step method for backward compatibility.
    ///
    /// Returns (next_state, reward, done) tuple.
    ///
    /// # Arguments
    ///
    /// * `action` - Action index in range [0, num_actions)
    ///
    /// # Returns
    ///
    /// A tuple of (next_state, reward, done)
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::MockEnv;
    /// let mut env = MockEnv::new(10);
    /// let obs = env.reset();
    /// let (next_state, reward, done) = env.step(0);
    /// assert_eq!(next_state.len(), 32);
    /// assert!(reward > 0.0);
    /// ```
    pub fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool) {
        assert!(action < self.num_actions, "Invalid action: {}", action);

        self.step_count += 1;

        // Simple reward: reward decreases over time
        let reward = 10.0 - self.step_count as f64 * 0.1;

        // Update state (simple simulation)
        for i in 0..self.state.len() {
            self.state[i] += (action as f64) * 0.1;
        }

        let done = self.step_count >= self.max_steps;

        (self.state.clone(), reward, done)
    }

    /// Legacy action_space method for backward compatibility.
    ///
    /// # Returns
    ///
    /// The number of actions
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::MockEnv;
    /// let env = MockEnv::new(10);
    /// assert_eq!(env.action_space(), 10);
    /// ```
    pub fn action_space(&self) -> usize {
        self.num_actions
    }

    /// Legacy observation_space method for backward compatibility.
    ///
    /// # Returns
    ///
    /// The observation dimension
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::MockEnv;
    /// let env = MockEnv::new(10);
    /// assert_eq!(env.observation_space(), 32);
    /// ```
    pub fn observation_space(&self) -> usize {
        self.obs_dim
    }
}

/// Implement Environment trait for MockEnv
impl Environment for MockEnv {
    type Observation = Vec<f64>;
    type Action = usize;

    /// Reset the environment and return initial observation
    ///
    /// # Returns
    ///
    /// The initial observation vector
    ///
    /// # Example
    ///
    /// ```
    /// use eris::env::Environment;
    /// use eris::training::MockEnv;
    ///
    /// let mut env = MockEnv::new_with_dims(10, 50, 20);
    /// let obs = env.reset();
    /// assert_eq!(obs.len(), 50);
    /// ```
    fn reset(&mut self) -> Self::Observation {
        self.step_count = 0;
        self.state = vec![0.0; self.obs_dim];
        self.state.clone()
    }

    /// Take a step in the environment
    ///
    /// # Arguments
    ///
    /// * `action` - Action index in range [0, num_actions)
    ///
    /// # Returns
    ///
    /// StepResult containing observation, reward, done, and info
    ///
    /// # Example
    ///
    /// ```
    /// use eris::env::{Environment, StepResult};
    /// use eris::training::MockEnv;
    ///
    /// let mut env = MockEnv::new_with_dims(10, 50, 20);
    /// let _ = env.reset();
    ///
    /// let result: StepResult = env.step(0);
    /// println!("Observation: {:?}", result.observation);
    /// println!("Reward: {}", result.reward);
    /// println!("Info: {:?}", result.info.metrics);
    /// ```
    fn step(&mut self, action: Self::Action) -> StepResult {
        assert!(action < self.num_actions, "Invalid action: {}", action);

        self.step_count += 1;

        // Simple reward: reward decreases over time
        let reward = 10.0 - self.step_count as f64 * 0.1;

        // Update state (simple simulation)
        for i in 0..self.state.len() {
            self.state[i] += (action as f64) * 0.1;
        }

        let done = self.step_count >= self.max_steps;

        StepResult {
            observation: self.state.clone(),
            reward,
            done,
            info: Info::new()
                .with_metric("step_count", self.step_count as f64)
                .with_metric("max_steps", self.max_steps as f64),
        }
    }

    /// Get the observation space
    ///
    /// # Returns
    ///
    /// BoxSpace with the correct dimensionality
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
    fn observation_space(&self) -> BoxSpace {
        BoxSpace::uniform(self.obs_dim, 0.0, 100.0)
    }

    /// Get the action space
    ///
    /// # Returns
    ///
    /// DiscreteSpace with the correct number of actions
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
    fn action_space(&self) -> DiscreteSpace {
        DiscreteSpace::new(self.num_actions)
    }

    /// Set the random seed for reproducibility
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
    fn seed(&mut self, seed: u64) {
        self.seed = seed;
    }
}

/// Create dummy transition for testing with default dimensions (32 obs, 10 actions).
///
/// This function provides backward compatibility with hardcoded dimensions.
/// For custom dimensions, use [`create_dummy_transition_with_dims()`].
///
/// # Returns
///
/// A Transition with default dimensions
///
/// # Example
///
/// ```
/// use eris::training::create_dummy_transition;
///
/// let trans = create_dummy_transition();
/// assert_eq!(trans.state.len(), 32);
/// ```
pub fn create_dummy_transition() -> Transition {
    create_dummy_transition_with_dims(32, 10)
}

/// Create dummy transition with custom dimensions.
///
/// # Arguments
///
/// * `obs_dim` - Observation dimension
/// * `num_actions` - Number of actions
///
/// # Returns
///
/// Transition with appropriate dimensions
///
/// # Example
///
/// ```
/// use eris::training::create_dummy_transition_with_dims;
///
/// let trans = create_dummy_transition_with_dims(100, 20);
/// assert_eq!(trans.state.len(), 100);
/// assert_eq!(trans.next_state.len(), 100);
/// ```
pub fn create_dummy_transition_with_dims(obs_dim: usize, _num_actions: usize) -> Transition {
    Transition {
        state: vec![0.5; obs_dim],
        action: 0,
        reward: 1.0,
        next_state: vec![0.6; obs_dim],
        done: false,
    }
}

/// Fill buffer with dummy transitions.
///
/// # Arguments
///
/// * `buffer` - Replay buffer to fill
/// * `n` - Number of transitions to add
/// * `obs_dim` - Observation dimension
/// * `num_actions` - Number of actions
///
/// # Example
///
/// ```
/// use eris::training::replay_buffer::ReplayBuffer;
/// use eris::training::fill_buffer;
///
/// let mut buffer = ReplayBuffer::new(100);
/// fill_buffer(&mut buffer, 50, 32, 10);
/// assert_eq!(buffer.len(), 50);
/// ```
pub fn fill_buffer(
    buffer: &mut crate::training::replay_buffer::ReplayBuffer,
    n: usize,
    obs_dim: usize,
    num_actions: usize,
) {
    for i in 0..n {
        let mut trans = create_dummy_transition_with_dims(obs_dim, num_actions);
        trans.action = i % num_actions; // Vary actions
        trans.reward = (i as f32) * 0.1; // Vary rewards
        buffer.push(trans);
    }
}
