//! Burn RL Policy Implementation for DQN
//!
//! Implements burn-rl traits: Policy, PolicyLearner, PolicyState
//! Based on burn-rl/examples/dqn-agent

use burn::module::AutodiffModule;
use burn::nn::loss::MseLoss;
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{Adaptor, LossInput};
use burn::train::ItemLazy;

use crate::models::CombinedModel;
use burn_rl::{
    Batchable, LearnerTransitionBatch, Policy, PolicyLearner, PolicyState, RLTrainOutput,
};

/// Observation tensor for DQN
#[derive(Clone, Debug)]
pub struct StateTensor<B: Backend> {
    pub state: Tensor<B, 2>,
}

impl<B: Backend> Batchable for StateTensor<B> {
    fn batch(values: Vec<Self>) -> Self {
        let tensors: Vec<Tensor<B, 2>> = values.iter().map(|v| v.state.clone()).collect();
        Self {
            state: Tensor::cat(tensors, 0),
        }
    }

    fn unbatch(self) -> Vec<Self> {
        self.state
            .split(1, 0)
            .into_iter()
            .map(|s| Self { state: s })
            .collect()
    }
}

impl<B: Backend> burn_rl::SliceAccess<B> for StateTensor<B> {
    fn zeros_like(sample: &Self, capacity: usize, device: &B::Device) -> Self {
        let dim = sample.state.dims()[1];
        Self {
            state: Tensor::zeros([capacity, dim], device),
        }
    }

    fn select(self, dim: usize, indices: Tensor<B, 1, burn::prelude::Int>) -> Self {
        Self {
            state: Tensor::select(self.state, dim, indices),
        }
    }

    fn slice_assign_inplace(&mut self, index: usize, value: Self) {
        self.state
            .inplace(|t| t.slice_assign(index..index + 1, value.state));
    }
}

/// Q-value distribution
#[derive(Clone, Debug)]
pub struct DiscreteLogits<B: Backend> {
    pub logits: Tensor<B, 2>,
}

impl<B: Backend> Batchable for DiscreteLogits<B> {
    fn batch(values: Vec<Self>) -> Self {
        let tensors: Vec<Tensor<B, 2>> = values.iter().map(|v| v.logits.clone()).collect();
        Self {
            logits: Tensor::cat(tensors, 0),
        }
    }

    fn unbatch(self) -> Vec<Self> {
        self.logits
            .split(1, 0)
            .into_iter()
            .map(|l| Self { logits: l })
            .collect()
    }
}

/// Discrete action
#[derive(Clone, Debug)]
pub struct ActionTensor<B: Backend> {
    pub actions: Tensor<B, 2, burn::prelude::Int>,
}

impl<B: Backend> Batchable for ActionTensor<B> {
    fn batch(values: Vec<Self>) -> Self {
        let tensors: Vec<Tensor<B, 2, burn::prelude::Int>> =
            values.iter().map(|v| v.actions.clone()).collect();
        Self {
            actions: Tensor::cat(tensors, 0),
        }
    }

    fn unbatch(self) -> Vec<Self> {
        self.actions
            .split(1, 0)
            .into_iter()
            .map(|a| Self { actions: a })
            .collect()
    }
}

/// Policy state wraps the model
#[derive(Clone)]
pub struct CombinedPolicyState<B: Backend>(pub CombinedModel<B>);

impl<B: Backend> PolicyState<B> for CombinedPolicyState<B> {
    type Record = <CombinedModel<B> as Module<B>>::Record;

    fn into_record(self) -> Self::Record {
        self.0.into_record()
    }

    fn load_record(&self, record: Self::Record) -> Self {
        Self(self.0.clone().load_record(record))
    }
}

/// DQN Policy implementing burn-rl::Policy
#[derive(Clone)]
pub struct DQNPolicy<B: Backend> {
    pub model: CombinedModel<B>,
}

impl<B: Backend> Policy<B> for DQNPolicy<B> {
    type Observation = StateTensor<B>;
    type ActionDistribution = DiscreteLogits<B>;
    type Action = ActionTensor<B>;
    type ActionContext = ();
    type PolicyState = CombinedPolicyState<B>;

    fn forward(&mut self, obs: Self::Observation) -> Self::ActionDistribution {
        let (_, _, q_values) = self.model.forward(obs.state);
        DiscreteLogits { logits: q_values }
    }

    fn action(
        &mut self,
        obs: Self::Observation,
        _deterministic: bool,
    ) -> (Self::Action, Vec<Self::ActionContext>) {
        let dist = self.forward(obs);
        let actions = dist.logits.argmax(1).unsqueeze();
        (ActionTensor { actions }, vec![])
    }

    fn update(&mut self, state: Self::PolicyState) {
        self.model = state.0;
    }

    fn state(&self) -> Self::PolicyState {
        CombinedPolicyState(self.model.clone())
    }

    fn load_record(self, record: <Self::PolicyState as PolicyState<B>>::Record) -> Self {
        let state = self.state().load_record(record);
        Self { model: state.0 }
    }
}

/// Training output
#[derive(Clone)]
pub struct DQNTrainOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
}

impl<B: Backend> ItemLazy for DQNTrainOutput<B> {
    type ItemSync = Self;
    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for DQNTrainOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

/// DQN Learner configuration
#[derive(Config, Debug)]
pub struct DQNConfig {
    #[config(default = 0.99)]
    pub gamma: f64,

    #[config(default = 3e-4)]
    pub learning_rate: f64,

    #[config(default = 0.005)]
    pub tau: f64,
}

/// DQN Learner implementing PolicyLearner
pub struct DQNLearner<B: AutodiffBackend, O>
where
    O: Optimizer<CombinedModel<B>, B>,
{
    policy_model: CombinedModel<B>,
    target_model: CombinedModel<B>,
    optimizer: O,
    config: DQNConfig,
    device: B::Device,
}

impl<B, O> DQNLearner<B, O>
where
    B: AutodiffBackend,
    O: Optimizer<CombinedModel<B>, B>,
{
    pub fn new(
        model: CombinedModel<B>,
        optimizer: O,
        config: DQNConfig,
        device: B::Device,
    ) -> Self {
        Self {
            target_model: model.clone(),
            policy_model: model,
            optimizer,
            config,
            device,
        }
    }
}

impl<B, O> PolicyLearner<B> for DQNLearner<B, O>
where
    B: AutodiffBackend,
    CombinedModel<B>: AutodiffModule<B>,
    O: Optimizer<CombinedModel<B>, B>,
{
    type TrainContext = DQNTrainOutput<B>;
    type InnerPolicy = DQNPolicy<B>;
    type Record = <CombinedModel<B> as Module<B>>::Record;

    fn train(
        &mut self,
        batch: LearnerTransitionBatch<B, Self::InnerPolicy>,
    ) -> RLTrainOutput<Self::TrainContext, <Self::InnerPolicy as Policy<B>>::PolicyState> {
        let states = batch.states.state;
        let next_states = batch.next_states.state;
        let actions = batch.actions.actions;
        let rewards = batch.rewards;
        let dones = batch.dones;

        let batch_size = actions.dims()[0];

        let (_, _, q_values) = self.policy_model.forward(states);
        let q_selected = q_values.gather(1, actions).squeeze::<1>();

        let (_, _, target_q) = self.target_model.forward(next_states);
        let max_next = target_q.max_dim(1).squeeze::<1>();

        let not_done = Tensor::ones_like(&dones) - dones;
        let rewards_flat = rewards.squeeze::<1>();
        let targets = rewards_flat
            + Tensor::full([batch_size], self.config.gamma, &self.device)
                * max_next
                * not_done.squeeze::<1>();

        let loss = MseLoss::new().forward(q_selected, targets, burn::nn::loss::Reduction::Mean);

        let grads = loss.backward();
        let params = GradientsParams::from_grads(grads, &self.policy_model);
        self.policy_model =
            self.optimizer
                .step(self.config.learning_rate, self.policy_model.clone(), params);

        self.target_model = self.policy_model.clone();

        RLTrainOutput {
            policy: CombinedPolicyState(self.policy_model.clone()),
            item: DQNTrainOutput { loss },
        }
    }

    fn policy(&self) -> Self::InnerPolicy {
        DQNPolicy {
            model: self.policy_model.clone(),
        }
    }

    fn update_policy(&mut self, policy: Self::InnerPolicy) {
        self.policy_model = policy.model;
    }

    fn record(&self) -> Self::Record {
        self.policy_model.clone().into_record()
    }

    fn load_record(self, record: Self::Record) -> Self {
        let policy_model = self.policy_model.load_record(record);
        Self {
            target_model: policy_model.clone(),
            policy_model,
            optimizer: self.optimizer,
            config: self.config,
            device: self.device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::CombinedModelConfig;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    #[test]
    fn test_state_tensor_batching() {
        type TestBackend = NdArray;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        let s1: StateTensor<TestBackend> = StateTensor {
            state: Tensor::from_data(TensorData::new(vec![1.0f32, 2.0, 3.0], [1, 3]), &device),
        };
        let s2: StateTensor<TestBackend> = StateTensor {
            state: Tensor::from_data(TensorData::new(vec![4.0f32, 5.0, 6.0], [1, 3]), &device),
        };

        let batched = StateTensor::batch(vec![s1, s2]);
        assert_eq!(
            batched.state.dims().into_iter().collect::<Vec<_>>(),
            vec![2, 3]
        );

        let unbatched = batched.unbatch();
        assert_eq!(unbatched.len(), 2);
    }

    #[test]
    fn test_dqn_policy_forward() {
        type TestBackend = NdArray;
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let config = CombinedModelConfig::new(10, 20, 64, 10);
        let model: CombinedModel<TestBackend> = config.init(&device);

        let mut policy = DQNPolicy { model };

        let obs: StateTensor<TestBackend> = StateTensor {
            state: Tensor::from_data(TensorData::new(vec![1.0f32; 10], [1, 10]), &device),
        };

        let output = policy.forward(obs);
        assert_eq!(
            output.logits.dims().into_iter().collect::<Vec<_>>(),
            vec![1, 10]
        );
    }
}
