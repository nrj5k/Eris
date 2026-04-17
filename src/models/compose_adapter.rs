//! Adapters for eris models to implement burnme-rly's ComposableModel trait.
//!
//! These adapters wrap ContextualBandit and QNetwork so they can be used
//! in SequentialCompose and ParallelCompose.

use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};
use burnme_rly::models::ComposableModel;

use crate::models::{ContextualBandit, QNetwork};

/// Adapter wrapping ContextualBandit to implement ComposableModel.
///
/// ContextualBandit normally returns `(features, importance)` from its
/// forward pass. This adapter returns ONLY the features tensor from
/// `forward_composable()`, making it compatible with SequentialCompose.
///
/// The importance score is still accessible via `importance()` method.
#[derive(Module, Debug)]
pub struct BanditAdapter<B: Backend> {
    pub bandit: ContextualBandit<B>,
    /// Feature dimension (output_dim for ComposableModel)
    feature_dim: usize,
}

impl<B: Backend> BanditAdapter<B> {
    /// Create a new BanditAdapter from an existing ContextualBandit.
    pub fn new(bandit: ContextualBandit<B>, feature_dim: usize) -> Self {
        Self {
            bandit,
            feature_dim,
        }
    }

    /// Get the importance score from the bandit.
    ///
    /// Call this after `forward_composable()` to get the importance tensor
    /// for computing the bandit's own loss.
    pub fn importance(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let (_, importance) = self.bandit.forward(input);
        importance
    }
}

impl<B: Backend> ComposableModel<B> for BanditAdapter<B> {
    fn forward_composable(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let (features, _) = self.bandit.forward(input);
        features
    }

    fn output_dim(&self) -> usize {
        self.feature_dim
    }
}

/// Adapter wrapping QNetwork to implement ComposableModel.
///
/// QNetwork already returns `Tensor<B, 2>` (Q-values), so this is
/// a trivial adapter that just delegates.
#[derive(Module, Debug)]
pub struct DQNAdapter<B: Backend> {
    pub qnetwork: QNetwork<B>,
    /// Action dimension (output_dim for ComposableModel)
    action_dim: usize,
}

impl<B: Backend> DQNAdapter<B> {
    /// Create a new DQNAdapter from an existing QNetwork.
    pub fn new(qnetwork: QNetwork<B>, action_dim: usize) -> Self {
        Self {
            qnetwork,
            action_dim,
        }
    }
}

impl<B: Backend> ComposableModel<B> for DQNAdapter<B> {
    fn forward_composable(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.qnetwork.forward(input)
    }

    fn output_dim(&self) -> usize {
        self.action_dim
    }
}
