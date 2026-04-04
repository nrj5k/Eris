//! Batchable types for RL integration
//!
//! These types wrap tensors to implement the Batchable trait
//! from burn-rl, allowing them to be batched and unbatched.

use burn::prelude::*;
use burn_rl::Batchable;

/// Observation tensor that can be batched
///
/// Represents a batch of state observations as a 2D tensor
/// with shape [batch_size, feature_dim]
#[derive(Clone, Debug)]
pub struct Observation<B: Backend> {
    /// The observation tensor [batch_size, feature_dim]
    pub tensor: Tensor<B, 2>,
}

impl<B: Backend> Batchable for Observation<B> {
    fn batch(items: Vec<Self>) -> Self {
        if items.is_empty() {
            panic!("Cannot batch empty observation list");
        }
        let tensors: Vec<Tensor<B, 2>> = items.iter().map(|o| o.tensor.clone()).collect();
        Observation {
            tensor: Tensor::cat(tensors, 0),
        }
    }

    fn unbatch(self) -> Vec<Self> {
        let batch_size = self.tensor.dims()[0];
        if batch_size == 0 {
            return vec![];
        }

        self.tensor
            .split(1, 0)
            .into_iter()
            .map(|tensor| Observation { tensor })
            .collect()
    }
}

/// Action distribution (Q-values/logits) for discrete actions
///
/// Represents a batch of action distributions as a 2D tensor
/// with shape [batch_size, action_dim]
#[derive(Clone, Debug)]
pub struct ActionDistribution<B: Backend> {
    /// The Q-values/logits tensor [batch_size, action_dim]
    pub logits: Tensor<B, 2>,
}

impl<B: Backend> Batchable for ActionDistribution<B> {
    fn batch(items: Vec<Self>) -> Self {
        if items.is_empty() {
            panic!("Cannot batch empty distribution list");
        }
        let tensors: Vec<Tensor<B, 2>> = items.iter().map(|d| d.logits.clone()).collect();
        ActionDistribution {
            logits: Tensor::cat(tensors, 0),
        }
    }

    fn unbatch(self) -> Vec<Self> {
        self.logits
            .split(1, 0)
            .into_iter()
            .map(|logits| ActionDistribution { logits })
            .collect()
    }
}

/// Discrete action tensor
///
/// Represents a batch of discrete actions as a 2D tensor
/// with shape [batch_size, 1]
#[derive(Clone, Debug)]
pub struct Action<B: Backend> {
    /// The action indices [batch_size, 1]
    pub indices: Tensor<B, 2>,
}

impl<B: Backend> Batchable for Action<B> {
    fn batch(items: Vec<Self>) -> Self {
        if items.is_empty() {
            panic!("Cannot batch empty action list");
        }
        let tensors: Vec<Tensor<B, 2>> = items.iter().map(|a| a.indices.clone()).collect();
        Action {
            indices: Tensor::cat(tensors, 0),
        }
    }

    fn unbatch(self) -> Vec<Self> {
        self.indices
            .split(1, 0)
            .into_iter()
            .map(|indices| Action { indices })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_observation_batch_unbatch() {
        let device: <TestBackend as Backend>::Device = Default::default();

        // Create 3 observations of shape [1, 10]
        let obs1: Observation<TestBackend> = Observation {
            tensor: Tensor::zeros([1, 10], &device),
        };
        let obs2: Observation<TestBackend> = Observation {
            tensor: Tensor::ones([1, 10], &device),
        };
        let obs3: Observation<TestBackend> = Observation {
            tensor: Tensor::zeros([1, 10], &device),
        };

        // Batch them
        let batched = Observation::batch(vec![obs1.clone(), obs2.clone(), obs3.clone()]);
        assert_eq!(batched.tensor.dims(), [3, 10]);

        // Unbatch them
        let unbatched = batched.unbatch();
        assert_eq!(unbatched.len(), 3);
        assert_eq!(unbatched[0].tensor.dims(), [1, 10]);
    }

    #[test]
    fn test_action_distribution_batch() {
        let device: <TestBackend as Backend>::Device = Default::default();

        let dist1: ActionDistribution<TestBackend> = ActionDistribution {
            logits: Tensor::zeros([1, 10], &device),
        };
        let dist2: ActionDistribution<TestBackend> = ActionDistribution {
            logits: Tensor::ones([1, 10], &device),
        };

        let batched = ActionDistribution::batch(vec![dist1, dist2]);
        assert_eq!(batched.logits.dims(), [2, 10]);

        let unbatched = batched.unbatch();
        assert_eq!(unbatched.len(), 2);
    }

    #[test]
    fn test_action_batch() {
        let device: <TestBackend as Backend>::Device = Default::default();

        let action1: Action<TestBackend> = Action {
            indices: Tensor::from_floats([[0.0]], &device),
        };
        let action2: Action<TestBackend> = Action {
            indices: Tensor::from_floats([[1.0]], &device),
        };

        let batched = Action::batch(vec![action1, action2]);
        assert_eq!(batched.indices.dims(), [2, 1]);

        let unbatched = batched.unbatch();
        assert_eq!(unbatched.len(), 2);
    }
}
