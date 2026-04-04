use super::Space;

/// A discrete space with n possible actions.
///
/// Similar to OpenAI Gym's Discrete space, this represents a set of n discrete actions
/// indexed from 0 to n-1.
///
/// # Purpose
///
/// `DiscreteSpace` is used to define:
/// - Action spaces with discrete choices
/// - Classification output spaces
/// - Discrete control problems
///
/// # Key Characteristics
///
/// - **Indexed**: Actions are 0-indexed integers
/// - **Fixed Count**: Exact number of available actions
/// - **Simple Sampling**: Uniform random selection
///
/// # Example
///
/// ```
/// use eris::space::DiscreteSpace;
///
/// // Create a discrete space with 5 actions
/// let space = DiscreteSpace::new(5);
///
/// // Sample random action
/// let action = space.sample_action();
/// assert!(action < 5);
///
/// // Validate action
/// assert!(space.contains_action(action));
///
/// // Check generic space validation
/// let action_vec = vec![action as f64];
/// assert!(space.contains(&action_vec));
/// ```
///
/// # Use Cases
///
/// - **Multi-tier storage**: [tier_0, tier_1, tier_2, tier_3, tier_4]
/// - **Operation selection**: [read, write, delete, copy, move]
/// - **Classification**: [class_0, class_1, ..., class_n-1]
///
/// # Panics
///
/// - No panics - all methods safely handle edge cases
/// - Contains validation rejects out-of-range values
pub struct DiscreteSpace {
    /// Number of possible actions
    pub n: usize,
}

impl DiscreteSpace {
    /// Creates a new DiscreteSpace with n actions.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of discrete actions (valid indices: 0 to n-1)
    ///
    /// # Example
    ///
    /// ```
    /// use eris::space::DiscreteSpace;
    ///
    /// let space = DiscreteSpace::new(3);
    /// assert_eq!(space.n, 3);
    /// ```
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Samples a random action and returns the action index.
    ///
    /// This method is more convenient when working with discrete actions
    /// as it returns a `usize` directly rather than a `Vec<f64>`.
    ///
    /// # Returns
    ///
    /// A random action index in the range [0, n).
    ///
    /// # Example
    ///
    /// ```
    /// use eris::space::DiscreteSpace;
    ///
    /// let space = DiscreteSpace::new(10);
    /// let action = space.sample_action();
    /// assert!(action < 10);
    /// ```
    pub fn sample_action(&self) -> usize {
        use rand::prelude::*;
        use rand::rng;

        let mut rng = rng();
        rng.random_range(0..self.n)
    }

    /// Checks if an action index is valid.
    ///
    /// # Arguments
    ///
    /// * `action` - The action index to check
    ///
    /// # Returns
    ///
    /// `true` if the action is in the range [0, n), `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use eris::space::DiscreteSpace;
    ///
    /// let space = DiscreteSpace::new(5);
    /// assert!(space.contains_action(0));
    /// assert!(space.contains_action(4));
    /// assert!(!space.contains_action(5));
    /// ```
    pub fn contains_action(&self, action: usize) -> bool {
        action < self.n
    }
}

impl Space for DiscreteSpace {
    fn dim(&self) -> usize {
        // Discrete space has dimension 1 (single action index)
        1
    }

    fn sample(&self) -> Vec<f64> {
        // Returns a single-element vector with the action index as f64
        vec![self.sample_action() as f64]
    }

    fn contains(&self, value: &[f64]) -> bool {
        if value.len() != 1 {
            return false;
        }

        // Check if the value is a valid integer within bounds
        let action = value[0];

        // Action must be non-negative
        if action < 0.0 {
            return false;
        }

        // Action must be an integer (with tolerance for floating point)
        let action_int = action as usize;
        if (action - action_int as f64).abs() > 1e-10 {
            return false;
        }

        // Action must be within bounds
        action_int < self.n
    }
}
