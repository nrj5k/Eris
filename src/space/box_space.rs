use super::Space;

/// A continuous multidimensional space (box) with low and high bounds.
///
/// Similar to OpenAI Gym's Box space, this represents a bounded continuous space
/// where observations can take any value within [low, high] for each dimension.
///
/// # Purpose
///
/// `BoxSpace` is used to define:
/// - Observation spaces with bounded continuous values
/// - Action spaces for continuous control problems
/// - Validation of value ranges in RL applications
///
/// # Key Characteristics
///
/// - **High Dimensional**: Supports any number of dimensions
/// - **Bounded**: Each dimension has explicit min/max bounds
/// - **Flexible Shape**: Supports both uniform and non-uniform bounds
///
/// # Example
///
/// ```
/// use eris::space::{BoxSpace, Space};
///
/// // Create a 4D observation space (e.g., CartPole)
/// let space = BoxSpace::uniform(4, -10.0, 10.0);
///
/// // Get dimension
/// assert_eq!(space.dim(), 4);
///
/// // Sample random observation
/// let obs = space.sample();
/// assert!(space.contains(&obs));
///
/// // Check specific value
/// assert!(space.contains(&[5.0, -3.0, 0.0, 7.5]));
/// ```
///
/// # Use Cases
///
/// - **State observations**: [position, velocity, angle, angular_velocity]
/// - **Action spaces**: Continuous control outputs
/// - **Feature vectors**: Normalized input features [0,1] or [-1,1]
///
/// # Panics
///
/// - Panics in [`new()`](BoxSpace::new) if bounds are invalid (low > high)
/// - Panics in [`contains()`](Space::contains) if dimension mismatch
pub struct BoxSpace {
    /// Lower bounds for each dimension
    pub low: Vec<f64>,
    /// Upper bounds for each dimension
    pub high: Vec<f64>,
    /// Shape of the space (dimensions)
    pub shape: Vec<usize>,
}

impl BoxSpace {
    /// Creates a new BoxSpace with given low, high bounds, and shape.
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bounds for each dimension
    /// * `high` - Upper bounds for each dimension
    /// * `shape` - Shape of the space (must match the product of low.len())
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The length of `low` doesn't match the product of `shape`
    /// - The length of `high` doesn't match the length of `low`
    /// - Any `low[i] > high[i]` (invalid bounds)
    ///
    /// # Example
    ///
    /// ```
    /// use eris::space::{BoxSpace, Space};
    ///
    /// let space = BoxSpace::new(
    ///     vec![0.0, -1.0, 0.0],
    ///     vec![1.0, 1.0, 10.0],
    ///     vec![3]
    /// ).unwrap();
    /// assert_eq!(space.dim(), 3);
    /// ```
    pub fn new(low: Vec<f64>, high: Vec<f64>, shape: Vec<usize>) -> crate::error::Result<Self> {
        let expected_size = shape.iter().product::<usize>();

        if low.len() != expected_size {
            return Err(crate::error::EnvError::InvalidSpace(format!(
                "low vector length {} doesn't match shape size {}",
                low.len(),
                expected_size
            )));
        }

        if high.len() != low.len() {
            return Err(crate::error::EnvError::InvalidSpace(format!(
                "high length {} doesn't match low length {}",
                high.len(),
                low.len()
            )));
        }

        // Validate that low <= high for all dimensions
        for (i, (&lo, &hi)) in low.iter().zip(high.iter()).enumerate() {
            if lo > hi {
                return Err(crate::error::EnvError::InvalidSpace(format!(
                    "low bound {} is greater than high bound {} at dimension {}",
                    lo, hi, i
                )));
            }
        }

        Ok(Self { low, high, shape })
    }

    /// Creates a BoxSpace with uniform bounds for all dimensions.
    ///
    /// # Arguments
    ///
    /// * `dim` - Number of dimensions
    /// * `low` - Lower bound (same for all dimensions)
    /// * `high` - Upper bound (same for all dimensions)
    ///
    /// # Example
    ///
    /// ```
    /// use eris::space::{BoxSpace, Space};
    ///
    /// let space = BoxSpace::uniform(3, -1.0, 1.0);
    /// assert_eq!(space.dim(), 3);
    /// assert!(space.low.iter().all(|&x| x == -1.0));
    /// assert!(space.high.iter().all(|&x| x == 1.0));
    /// ```
    pub fn uniform(dim: usize, low: f64, high: f64) -> Self {
        Self {
            low: vec![low; dim],
            high: vec![high; dim],
            shape: vec![dim],
        }
    }

    /// Creates a BoxSpace from shape and bounds using broadcasting.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the space
    /// * `low` - Single low bound (broadcast to all dimensions)
    /// * `high` - Single high bound (broadcast to all dimensions)
    ///
    /// # Example
    ///
    /// ```
    /// use eris::space::BoxSpace;
    ///
    /// // Create a 2D space with shape (3, 4)
    /// let space = BoxSpace::from_shape(vec![3, 4], 0.0, 1.0);
    /// assert_eq!(space.dim(), 12);
    /// assert_eq!(space.shape, vec![3, 4]);
    /// ```
    pub fn from_shape(shape: Vec<usize>, low: f64, high: f64) -> Self {
        let total_size: usize = shape.iter().product();
        Self {
            low: vec![low; total_size],
            high: vec![high; total_size],
            shape,
        }
    }
}

impl Space for BoxSpace {
    fn dim(&self) -> usize {
        self.low.len()
    }

    fn sample(&self) -> Vec<f64> {
        use rand::prelude::*;
        use rand::rng;

        let mut rng = rng();

        self.low
            .iter()
            .zip(self.high.iter())
            .map(|(&lo, &hi)| rng.random_range(lo..=hi))
            .collect()
    }

    fn contains(&self, value: &[f64]) -> bool {
        if value.len() != self.dim() {
            return false;
        }

        value
            .iter()
            .zip(self.low.iter())
            .zip(self.high.iter())
            .all(|((&v, &lo), &hi)| v >= lo && v <= hi)
    }
}
