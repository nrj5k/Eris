//! Space definitions for environments

use rand::RngExt;

/// Discrete action space (e.g., number of tiers)
#[derive(Debug, Clone, Copy)]
pub struct DiscreteSpace {
    n: usize, // Number of discrete actions
}

impl DiscreteSpace {
    /// Create new discrete space with n actions
    pub fn new(n: usize) -> Self {
        Self { n }
    }

    /// Get number of actions
    pub fn n(&self) -> usize {
        self.n
    }

    /// Sample random action
    pub fn sample(&self) -> usize {
        let mut rng = rand::rng();
        rng.random_range(0..self.n)
    }
}

impl Default for DiscreteSpace {
    fn default() -> Self {
        Self { n: 10 } // Default: 10 actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_space() {
        let space = DiscreteSpace::new(5);
        assert_eq!(space.n(), 5);

        let action = space.sample();
        assert!(action < 5);
    }
}
