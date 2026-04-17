//! Generic model composition primitives for multi-model RL architectures.
//!
//! Provides `SequentialCompose` and `ParallelCompose` for combining
//! any sub-models whose input/output dimensions align.
//!
//! # Design Philosophy
//!
//! Sub-models train INDEPENDENTLY on a common reward signal.
//! Stop-gradient (`.detach()`) prevents gradient interference between models.
//! The shared reward ensures emergent cooperation.

use burn::module::{Devices, Module, ModuleMapper, ModuleVisitor};
use burn::nn::LinearConfig;
use burn::tensor::{backend::Backend, Tensor};

/// Trait for models that can participate in composed architectures.
///
/// Any model implementing this trait can be used as a sub-model
/// in `SequentialCompose` or `ParallelCompose`.
///
/// # Contract
///
/// - Input: `[batch_size, input_dim]`
/// - Output: `[batch_size, output_dim]`
///
/// Models with multiple outputs (e.g., returning features + importance)
/// should implement this trait to return only the primary output
/// (features), and expose secondary outputs via separate methods.
pub trait ComposableModel<B: Backend>: Module<B> {
    /// Forward pass: input → output.
    ///
    /// # Arguments
    /// * `input` - Input tensor `[batch_size, input_dim]`
    ///
    /// # Returns
    /// Output tensor `[batch_size, output_dim]`
    fn forward_composable(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Get the output dimension of this model.
    fn output_dim(&self) -> usize;
}

/// Configuration for composed model training.
///
/// Controls how losses from sub-models are combined during training.
#[derive(Debug, Clone)]
pub struct ComposeConfig {
    /// Loss weight for each sub-model.
    /// Default: [1.0, 1.0] (equal weighting).
    /// For SequentialCompose: [model_a_weight, model_b_weight].
    /// For ParallelCompose: [model_a_weight, model_b_weight].
    pub loss_weights: Vec<f32>,
}

impl Default for ComposeConfig {
    fn default() -> Self {
        Self {
            loss_weights: vec![1.0, 1.0],
        }
    }
}

impl ComposeConfig {
    /// Create config with equal weights for N sub-models.
    pub fn equal_weights(n: usize) -> Self {
        Self {
            loss_weights: vec![1.0; n],
        }
    }

    /// Create config with custom weights.
    pub fn with_weights(weights: Vec<f32>) -> Self {
        Self {
            loss_weights: weights,
        }
    }

    /// Get the combined loss from per-model losses.
    ///
    /// # Arguments
    /// * `losses` - Per-model losses (must match loss_weights length)
    ///
    /// # Returns
    /// Weighted sum: Σ weight_i * loss_i
    pub fn combine_losses<B: Backend>(&self, losses: &[Tensor<B, 1>]) -> Tensor<B, 1> {
        assert_eq!(
            losses.len(),
            self.loss_weights.len(),
            "Number of losses ({}) must match number of weights ({})",
            losses.len(),
            self.loss_weights.len()
        );
        let mut total = losses[0].clone() * self.loss_weights[0];
        for (loss, weight) in losses[1..].iter().zip(&self.loss_weights[1..]) {
            total = total + loss.clone() * *weight;
        }
        total
    }
}

/// Sequential composition of two models: input → A → features.detach() → B → output.
///
/// Model A (perception) produces features. The features are detached
/// (stop-gradient) before being passed to Model B (decision). This means:
/// - Model A's gradients come from its own loss (perception loss)
/// - Model B's gradients come from its own loss (decision loss)
/// - No gradient flows from B back through A (independent training)
/// - Both models receive the same reward → emergent cooperation
///
/// # Type Parameters
/// - `B`: Backend type
/// - `A`: First model (perception) implementing `ComposableModel<B>`
/// - `M`: Second model (decision) implementing `ComposableModel<B>`
///
/// # Example
/// ```rust,ignore
/// use burnme_rly::models::{SequentialCompose, ComposableModel};
/// use burn::backend::NdArray;
///
/// // Bandit perceives → DQN decides
/// let model = SequentialCompose::new(bandit, dqn);
/// let features = model.model_a().forward_composable(input);  // For bandit loss
/// let decision = model.forward(input);                       // For DQN loss
/// ```
#[derive(Debug, Clone)]
pub struct SequentialCompose<B: Backend, A: ComposableModel<B>, M: ComposableModel<B>> {
    /// First model (perception)
    pub model_a: A,
    /// Second model (decision)
    pub model_b: M,
    _backend: std::marker::PhantomData<B>,
}

impl<B, A, M> Module<B> for SequentialCompose<B, A, M>
where
    B: Backend,
    A: ComposableModel<B>,
    M: ComposableModel<B>,
{
    type Record = (A::Record, M::Record);

    fn collect_devices(&self, _devices: Devices<B>) -> Devices<B> {
        let mut devices = self.model_a.devices();
        devices.extend(self.model_b.devices());
        devices
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            model_a: self.model_a.fork(device),
            model_b: self.model_b.fork(device),
            _backend: std::marker::PhantomData,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            model_a: self.model_a.to_device(device),
            model_b: self.model_b.to_device(device),
            _backend: std::marker::PhantomData,
        }
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.model_a.visit(visitor);
        self.model_b.visit(visitor);
    }

    fn map<M2: ModuleMapper<B>>(self, mapper: &mut M2) -> Self {
        Self {
            model_a: self.model_a.map(mapper),
            model_b: self.model_b.map(mapper),
            _backend: std::marker::PhantomData,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model_a: self.model_a.load_record(record.0),
            model_b: self.model_b.load_record(record.1),
            _backend: std::marker::PhantomData,
        }
    }

    fn into_record(self) -> Self::Record {
        (self.model_a.into_record(), self.model_b.into_record())
    }
}

impl<B, A, M> burn::module::AutodiffModule<B> for SequentialCompose<B, A, M>
where
    B: burn::tensor::backend::AutodiffBackend,
    A: ComposableModel<B> + burn::module::AutodiffModule<B> + Send + Clone + std::fmt::Debug,
    M: ComposableModel<B> + burn::module::AutodiffModule<B> + Send + Clone + std::fmt::Debug,
    A::InnerModule: ComposableModel<B::InnerBackend>,
    M::InnerModule: ComposableModel<B::InnerBackend>,
{
    type InnerModule = SequentialCompose<B::InnerBackend, A::InnerModule, M::InnerModule>;

    fn valid(&self) -> Self::InnerModule {
        SequentialCompose {
            model_a: self.model_a.valid(),
            model_b: self.model_b.valid(),
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B, A, M> SequentialCompose<B, A, M>
where
    B: Backend,
    A: ComposableModel<B>,
    M: ComposableModel<B>,
{
    /// Create a new sequential composition.
    pub fn new(model_a: A, model_b: M) -> Self {
        Self {
            model_a,
            model_b,
            _backend: std::marker::PhantomData,
        }
    }

    /// Forward pass through both models WITH stop-gradient between them.
    ///
    /// Flow: input → model_a → features.detach() → model_b → output
    ///
    /// The `.detach()` prevents gradient from model_b flowing back through model_a.
    /// Each model should compute its own loss independently.
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let features = self.model_a.forward_composable(input);
        let features_detached = features.detach();
        self.model_b.forward_composable(features_detached)
    }

    /// Forward pass through model_a only (for computing model_a's loss).
    pub fn forward_a(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.model_a.forward_composable(input)
    }

    /// Forward pass through model_b only (given pre-computed features).
    /// Does NOT detach — use this for model_b's loss computation.
    pub fn forward_b(&self, features: Tensor<B, 2>) -> Tensor<B, 2> {
        self.model_b.forward_composable(features)
    }
}

/// Parallel composition of two models: input → (A, B) → concat → merge → output.
///
/// Both models receive the same input independently. Their outputs are
/// concatenated and passed through a learned merge layer. Stop-gradient
/// is applied to both sub-model outputs before merge, so each model
/// trains independently on the common reward.
///
/// # Type Parameters
/// - `B`: Backend type
/// - `A`: First model implementing `ComposableModel<B>`
/// - `M`: Second model implementing `ComposableModel<B>`
///
/// # Example
/// ```rust,ignore
/// let model = ParallelCompose::new(model_a, model_b, a_dim + b_dim, output_dim);
/// let combined = model.forward(input);  // Combined prediction
/// ```
#[derive(Debug, Clone)]
pub struct ParallelCompose<B: Backend, A: ComposableModel<B>, M: ComposableModel<B>> {
    /// First model
    pub model_a: A,
    /// Second model
    pub model_b: M,
    /// Learned merge layer: combines outputs from both models
    pub merge: burn::nn::Linear<B>,
}

impl<B, A, M> Module<B> for ParallelCompose<B, A, M>
where
    B: Backend,
    A: ComposableModel<B>,
    M: ComposableModel<B>,
{
    type Record = (
        A::Record,
        M::Record,
        <burn::nn::Linear<B> as Module<B>>::Record,
    );

    fn collect_devices(&self, _devices: Devices<B>) -> Devices<B> {
        let mut devices = self.model_a.devices();
        devices.extend(self.model_b.devices());
        devices.extend(self.merge.devices());
        devices
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            model_a: self.model_a.fork(device),
            model_b: self.model_b.fork(device),
            merge: self.merge.fork(device),
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            model_a: self.model_a.to_device(device),
            model_b: self.model_b.to_device(device),
            merge: self.merge.to_device(device),
        }
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.model_a.visit(visitor);
        self.model_b.visit(visitor);
        self.merge.visit(visitor);
    }

    fn map<M2: ModuleMapper<B>>(self, mapper: &mut M2) -> Self {
        Self {
            model_a: self.model_a.map(mapper),
            model_b: self.model_b.map(mapper),
            merge: self.merge.map(mapper),
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model_a: self.model_a.load_record(record.0),
            model_b: self.model_b.load_record(record.1),
            merge: self.merge.load_record(record.2),
        }
    }

    fn into_record(self) -> Self::Record {
        (
            self.model_a.into_record(),
            self.model_b.into_record(),
            self.merge.into_record(),
        )
    }
}

impl<B, A, M> ParallelCompose<B, A, M>
where
    B: Backend,
    A: ComposableModel<B>,
    M: ComposableModel<B>,
{
    /// Create a new parallel composition.
    ///
    /// # Arguments
    /// * `model_a` - First model
    /// * `model_b` - Second model
    /// * `merge_input_dim` - Sum of both models' output dimensions (a.output_dim() + b.output_dim())
    /// * `merge_output_dim` - Final output dimension
    /// * `device` - Backend device for initializing the merge layer
    pub fn new(
        model_a: A,
        model_b: M,
        merge_input_dim: usize,
        merge_output_dim: usize,
        device: &B::Device,
    ) -> Self {
        let merge = LinearConfig::new(merge_input_dim, merge_output_dim).init(device);
        Self {
            model_a,
            model_b,
            merge,
        }
    }

    /// Forward pass through both models in parallel WITH stop-gradient before merge.
    ///
    /// Flow: input → model_a → pred_a.detach() ┐
    ///       input → model_b → pred_b.detach() ┤→ concat → merge → output
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let pred_a = self.model_a.forward_composable(input.clone()).detach();
        let pred_b = self.model_b.forward_composable(input).detach();
        let combined = Tensor::cat(vec![pred_a, pred_b], 1);
        self.merge.forward(combined)
    }

    /// Forward pass through model_a only (for computing model_a's loss).
    pub fn forward_a(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.model_a.forward_composable(input)
    }

    /// Forward pass through model_b only (for computing model_b's loss).
    pub fn forward_b(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.model_b.forward_composable(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::tensor::{backend::Backend, Tensor};

    type TestBackend = NdArray;

    /// Simple linear model for testing composition
    #[derive(Module, Debug)]
    struct SimpleModel<B: Backend> {
        linear: burn::nn::Linear<B>,
        out_dim: usize,
    }

    impl<B: Backend> SimpleModel<B> {
        fn new(in_dim: usize, out_dim: usize, device: &B::Device) -> Self {
            Self {
                linear: LinearConfig::new(in_dim, out_dim).init(device),
                out_dim,
            }
        }
    }

    impl<B: Backend> ComposableModel<B> for SimpleModel<B> {
        fn forward_composable(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
            self.linear.forward(input)
        }

        fn output_dim(&self) -> usize {
            self.out_dim
        }
    }

    #[test]
    fn test_sequential_compose_forward() {
        let device = Default::default();
        let model_a = SimpleModel::<TestBackend>::new(8, 16, &device);
        let model_b = SimpleModel::<TestBackend>::new(16, 4, &device);
        let compose = SequentialCompose::new(model_a, model_b);

        let input = Tensor::zeros([2, 8], &device);
        let output = compose.forward(input);
        assert_eq!(output.shape().dims, [2, 4]);
    }

    #[test]
    fn test_sequential_compose_forward_a() {
        let device = Default::default();
        let model_a = SimpleModel::<TestBackend>::new(8, 16, &device);
        let model_b = SimpleModel::<TestBackend>::new(16, 4, &device);
        let compose = SequentialCompose::new(model_a, model_b);

        let input = Tensor::zeros([2, 8], &device);
        let features = compose.forward_a(input);
        assert_eq!(features.shape().dims, [2, 16]);
    }

    #[test]
    fn test_parallel_compose_forward() {
        let device = Default::default();
        let model_a = SimpleModel::<TestBackend>::new(8, 6, &device);
        let model_b = SimpleModel::<TestBackend>::new(8, 4, &device);
        let compose = ParallelCompose::new(model_a, model_b, 10, 3, &device);

        let input = Tensor::zeros([2, 8], &device);
        let output = compose.forward(input);
        assert_eq!(output.shape().dims, [2, 3]);
    }

    #[test]
    fn test_parallel_compose_forward_individual() {
        let device = Default::default();
        let model_a = SimpleModel::<TestBackend>::new(8, 6, &device);
        let model_b = SimpleModel::<TestBackend>::new(8, 4, &device);
        let compose = ParallelCompose::new(model_a, model_b, 10, 3, &device);

        let input = Tensor::zeros([2, 8], &device);
        let pred_a = compose.forward_a(input.clone());
        let pred_b = compose.forward_b(input);
        assert_eq!(pred_a.shape().dims, [2, 6]);
        assert_eq!(pred_b.shape().dims, [2, 4]);
    }

    #[test]
    fn test_compose_config_combine_losses() {
        let device = Default::default();
        let config = ComposeConfig::with_weights(vec![1.0, 0.5]);
        let loss_a = Tensor::<TestBackend, 1>::from_data([1.0], &device);
        let loss_b = Tensor::<TestBackend, 1>::from_data([2.0], &device);
        let combined = config.combine_losses(&[loss_a, loss_b]);
        let val: f32 = combined.into_data().as_slice().unwrap()[0];
        assert!((val - 2.0).abs() < 1e-5, "Expected 2.0, got {}", val); // 1.0*1.0 + 0.5*2.0 = 2.0
    }
}
