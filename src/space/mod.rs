mod box_space;
mod discrete_space;

pub use box_space::BoxSpace;
pub use discrete_space::DiscreteSpace;

/// Trait for observation/action spaces in RL environments.
///
/// This trait defines the common interface for all space types,
/// similar to OpenAI Gym's Space interface.
///
/// # Purpose
///
/// The `Space` trait provides a unified way to:
/// - Get dimension information for neural network inputs/outputs
/// - Sample valid random values from the space
/// - Validate that values belong to the space
///
/// # Type System
///
/// Space implementations use `Vec<f64>` for uniform representation:
/// - [`BoxSpace`] uses multi-dimensional float vectors
/// - [`DiscreteSpace`] uses single-element vectors with integer indices
///
/// # Example
///
/// ```
/// use eris::space::{BoxSpace, DiscreteSpace, Space};
///
/// // Continuous space sampling
/// let box_space = BoxSpace::uniform(4, -10.0, 10.0);
/// let sample = box_space.sample();
/// assert!(box_space.contains(&sample));
///
/// // Discrete space sampling
/// let discrete_space = DiscreteSpace::new(5);
/// let action = discrete_space.sample_action();
/// assert!(discrete_space.contains_action(action));
/// ```
///
/// # Implementing Custom Spaces
///
/// ```
/// use eris::space::Space;
///
/// struct CustomSpace {
///     pub values: Vec<f64>,
/// }
///
/// impl Space for CustomSpace {
///     fn dim(&self) -> usize {
///         self.values.len()
///     }
///
///     fn sample(&self) -> Vec<f64> {
///         // Custom sampling logic
///         self.values.clone()
///     }
///
///     fn contains(&self, value: &[f64]) -> bool {
///         // Custom validation logic
///         value.len() == self.dim()
///     }
/// }
/// ```
pub trait Space {
    /// Returns the dimension of the space.
    ///
    /// For [`BoxSpace`], this is the length of the observation vector.
    /// For [`DiscreteSpace`], this is always 1 (single action index).
    ///
    /// # Returns
    ///
    /// The number of dimensions in this space.
    ///
    /// # Example
    ///
    /// ```
    /// use eris::space::{BoxSpace, DiscreteSpace, Space};
    ///
    /// let box_space = BoxSpace::uniform(4, -1.0, 1.0);
    /// assert_eq!(box_space.dim(), 4);
    ///
    /// let discrete_space = DiscreteSpace::new(10);
    /// assert_eq!(discrete_space.dim(), 1);
    /// ```
    fn dim(&self) -> usize;

    /// Samples a random element from the space.
    ///
    /// Returns a vector of f64 values representing a valid point in the space.
    /// For [`BoxSpace`], returns values within [low, high] bounds.
    /// For [`DiscreteSpace`], returns a single-element vector with the action index.
    ///
    /// # Returns
    ///
    /// A `Vec<f64>` containing a valid random sample from this space.
    ///
    /// # Example
    ///
    /// ```
    /// use eris::space::{BoxSpace, DiscreteSpace, Space};
    ///
    /// let box_space = BoxSpace::uniform(3, 0.0, 1.0);
    /// let sample = box_space.sample();
    /// assert_eq!(sample.len(), 3);
    /// assert!(sample.iter().all(|&x| x >= 0.0 && x <= 1.0));
    /// ```
    fn sample(&self) -> Vec<f64>;

    /// Checks if a value belongs to the space.
    ///
    /// Validates that the given value is a valid member of this space.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to check
    ///
    /// # Returns
    ///
    /// `true` if the value is valid for this space, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use eris::space::{BoxSpace, DiscreteSpace, Space};
    ///
    /// let box_space = BoxSpace::uniform(3, 0.0, 1.0);
    /// assert!(box_space.contains(&[0.5, 0.3, 0.8]));
    /// assert!(!box_space.contains(&[1.5, 0.3, 0.8])); // Out of bounds
    ///
    /// let discrete_space = DiscreteSpace::new(5);
    /// assert!(discrete_space.contains(&[2.0])); // Valid action
    /// assert!(!discrete_space.contains(&[5.0])); // Out of bounds
    /// ```
    fn contains(&self, value: &[f64]) -> bool;
}
