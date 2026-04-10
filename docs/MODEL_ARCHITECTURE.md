# Eris Neural Network Architecture

This document provides detailed documentation of the neural network models used in the eris reinforcement learning system, including the Dueling DQN architecture, Contextual Bandit, and Combined Model.

## Table of Contents

1. [Overview](#overview)
2. [Dueling DQN Architecture](#dueling-dqn-architecture)
3. [Contextual Bandit](#contextual-bandit)
4. [Combined Model](#combined-model)
5. [Layer Dimensions](#layer-dimensions)
6. [Activation Functions](#activation-functions)
7. [Weight Initialization](#weight-initialization)
8. [Training Considerations](#training-considerations)

## Overview

Eris uses three neural network components working together:

```
┌─────────────────────────────────────────────────────────────┐
│                    Combined Model                            │
│                                                              │
│  ┌────────────────────┐     ┌─────────────────────────┐    │
│  │   Raw State (15)   │────▶│  Contextual Bandit      │    │
│  │                    │     │  - Feature extraction   │    │
│  │                    │     │  - Importance scoring   │    │
│  └────────────────────┘     └───────────┬─────────────┘    │
│                                        │                   │
│                                        ▼                   │
│  ┌─────────────────────────────────────▼───────────────┐  │
│  │            Dueling DQN Network                        │  │
│  │  ┌───────────────┐    ┌─────────────────────────┐   │  │
│  │  │  Value Stream │    │   Advantage Stream      │   │  │
│  │  │   V(s)        │    │   A(s,a)                │   │  │
│  │  └───────┬───────┘    └───────────┬─────────────┘   │  │
│  │          │                        │                 │  │
│  │          └───────────┬────────────┘                 │  │
│  │                      ▼                              │  │
│  │         Q(s,a) = V(s) + A(s,a) - mean(A)           │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │                                │
│                          ▼                                │
│               Q-values for 10 actions [0-9]               │
└─────────────────────────────────────────────────────────────┘
```

## Dueling DQN Architecture

### Concept

The Dueling DQN architecture separates the estimation of:
- **V(s)**: The value of being in state s (independent of actions)
- **A(s,a)**: The advantage of taking action a in state s

This separation allows the network to learn the value of states without needing to learn the effect of every action in every state.

### Architecture Diagram

```
Input: Features [batch, feature_dim]
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Shared Feature Extraction                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Fully Connected Layer 1                             │    │
│  │  - Input: feature_dim                                │    │
│  │  - Output: hidden_dim                                │    │
│  │  - Weights: [feature_dim × hidden_dim]               │    │
│  │  - Bias: [hidden_dim]                                │    │
│  │  - Activation: ReLU                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Fully Connected Layer 2                             │    │
│  │  - Input: hidden_dim                                 │    │
│  │  - Output: hidden_dim                                │    │
│  │  - Weights: [hidden_dim × hidden_dim]                │    │
│  │  - Bias: [hidden_dim]                                │    │
│  │  - Activation: ReLU                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
└─────────────────────────┼───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Value Stream   │ │ Advantage Stream│ │  Aggregation    │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│                 │ │                 │ │                 │
│ FC1: [hid→hid]  │ │ FC1: [hid→hid]  │ │ Q = V + A - mean│
│ FC2: [hid→1]    │ │ FC2: [hid→10]   │ │                 │
│                 │ │                 │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   ▼                   ▼
      V(s) [batch,1]    A(s,a) [batch,10]    Q(s,a) [batch,10]
```

### Implementation Details

**File**: `src/models/dqn.rs` (122 lines)

```rust
#[derive(Module, Debug)]
pub struct QNetwork<B: Backend> {
    // Shared feature extraction layers
    fc1: Linear<B>,
    fc2: Linear<B>,

    // Value stream: V(s)
    value_fc1: Linear<B>,
    value_fc2: Linear<B>,

    // Advantage stream: A(s, a)
    advantage_fc1: Linear<B>,
    advantage_fc2: Linear<B>,

    activation: Relu,

    // Store dimensions
    input_dim: usize,
    hidden_dim: usize,
    action_dim: usize,
}

#[derive(Config, Debug)]
pub struct QNetworkConfig {
    /// Input dimension (features from bandit)
    pub input_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Action dimension (5 tiers × 2 ops = 10 actions)
    pub action_dim: usize,
    #[config(default = true)]
    pub bias: bool,
}
```

### Forward Pass

```rust
impl<B: Backend> QNetwork<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        // Shared feature extraction
        let x = self.activation.forward(self.fc1.forward(x));
        let x = self.activation.forward(self.fc2.forward(x));

        // Value stream: V(s)
        let value = self.activation.forward(self.value_fc1.forward(x.clone()));
        let value = self.value_fc2.forward(value);  // [batch, 1]

        // Advantage stream: A(s, a)
        let advantage = self.activation.forward(self.advantage_fc1.forward(x));
        let advantage = self.advantage_fc2.forward(advantage);  // [batch, action_dim]

        // Combine: Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
        let mean_advantage = advantage.clone().mean_dim(1);
        let q_values = value + (advantage - mean_advantage);

        q_values
    }
}
```

### Aggregation Formula

The aggregation formula `Q(s,a) = V(s) + A(s,a) - mean(A)` is used because:

1. **Identifiability**: Without the mean subtraction, we couldn't recover V(s) and A(s,a) uniquely from Q(s,a)
2. **Stability**: The mean centering reduces variance in the advantage estimates
3. **Regularization**: Acts as a form of implicit regularization

## Contextual Bandit

### Concept

The Contextual Bandit component processes the raw state and:
1. **Extracts meaningful features** for the Q-network
2. **Computes importance scores** for tier selection

This provides the DQN with pre-processed, semantically meaningful inputs.

### Architecture Diagram

```
Input: Raw State [batch, state_dim] (15 features)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Extraction Network                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Fully Connected Layer 1                             │    │
│  │  - Input: state_dim (15)                             │    │
│  │  - Output: hidden_dim (128)                          │    │
│  │  - Weights: [15 × 128]                               │    │
│  │  - Bias: [128]                                        │    │
│  │  - Activation: ReLU                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Fully Connected Layer 2                             │    │
│  │  - Input: hidden_dim (128)                           │    │
│  │  - Output: hidden_dim * 2 (256)                      │    │
│  │  - Weights: [128 × 256]                              │    │
│  │  - Bias: [256]                                        │    │
│  │  - Activation: ReLU                                  │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
└─────────────────────────┼───────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
         ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Feature Head   │ │   Score Head    │ │     Outputs      │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│                 │ │                 │ │                 │
│ Input: [256]    │ │ Input: [256]    │ │ features: [64]  │
│ Output: [64]    │ │ Output: [1]     │ │ importance: [1] │
│ Activation:     │ │ Activation:     │ │                 │
│   None (linear) │ │   Sigmoid       │ │                 │
│                 │ │                 │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   ▼                   ▼
  Enhanced features    Importance score     For tier selection
  for Q-network        [0, 1]               and Q-value estimation
```

### Implementation Details

**File**: `src/models/bandit.rs` (97 lines)

```rust
#[derive(Module, Debug)]
pub struct ContextualBandit<B: Backend> {
    // Feature extraction layers
    fc1: Linear<B>,  // state_dim -> hidden_dim
    fc2: Linear<B>,  // hidden_dim -> hidden_dim * 2

    // Output heads
    feature_head: Linear<B>,  // hidden_dim * 2 -> feature_dim
    score_head: Linear<B>,    // hidden_dim * 2 -> 1

    activation: Relu,
    sigmoid: Sigmoid,
}

#[derive(Config, Debug)]
pub struct ContextualBanditConfig {
    /// Input state dimension
    pub state_dim: usize,
    /// Hidden layer dimension (fc1)
    pub hidden_dim: usize,
    /// Output feature dimension
    pub feature_dim: usize,
    #[config(default = true)]
    pub bias: bool,
}
```

### Forward Pass

```rust
impl<B: Backend> ContextualBandit<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Feature extraction
        let x = self.activation.forward(self.fc1.forward(x));
        let x = self.activation.forward(self.fc2.forward(x));

        // Feature output (for Q-network input)
        let features = self.feature_head.forward(x.clone());

        // Importance score output (for tier selector)
        let score = self.sigmoid.forward(self.score_head.forward(x));

        (features, score)
    }
}
```

### Importance Score Interpretation

The importance score is a value in [0, 1] representing:
- **Near 0**: Cold data (rarely accessed, suitable for slow tiers)
- **Near 1**: Hot data (frequently accessed, needs fast tiers)

This score is used by the TierSelector to make capacity-aware placement decisions.

## Combined Model

### Concept

The Combined Model integrates the Contextual Bandit with the Dueling DQN into a single neural network that:

1. Takes raw state as input
2. Processes through the contextual bandit
3. Passes enhanced features to the DQN
4. Outputs Q-values for all 10 actions

### Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────────┐
│                         CombinedModel                                  │
├───────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Raw State (15)                                                        │
│      │                                                                 │
│      ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    Contextual Bandit                             │  │
│  │  ┌─────────────────┐    ┌─────────────────────────┐            │  │
│  │  │ FC1 (15→128)    │    │ FC2 (128→256)           │            │  │
│  │  │ + ReLU          │    │ + ReLU                  │            │  │
│  │  └────────┬────────┘    └───────────┬─────────────┘            │  │
│  │           │                         │                           │  │
│  │           └────────────┬────────────┘                           │  │
│  │                        │                                        │  │
│  │           ┌────────────┴────────────┐                           │  │
│  │           ▼                         ▼                           │  │
│  │  ┌─────────────────┐    ┌─────────────────┐                    │  │
│  │  │ Feature Head    │    │ Score Head      │                    │  │
│  │  │ (256→64)        │    │ (256→1)         │                    │  │
│  │  │ Linear          │    │ + Sigmoid       │                    │  │
│  │  └────────┬────────┘    └────────┬────────┘                    │  │
│  │           │                      │                             │  │
│  │           ▼                      ▼                             │  │
│  │  Features (64)          Importance (1)                         │  │
│  │           │                      │                             │  │
│  └───────────┼──────────────────────┼─────────────────────────────┘  │
│              │                      │                                  │
│              └──────────┬───────────┘                                  │
│                         │                                              │
│                         ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    Dueling DQN                                  │  │
│  │                                                                 │  │
│  │  ┌─────────────────────────────────────────────────────────┐   │  │
│  │  │ Shared FC1 (64→128) + ReLU                              │   │  │
│  │  └─────────────────────────────────────────────────────────┘   │  │
│  │                            │                                   │  │
│  │                            ▼                                   │  │
│  │  ┌─────────────────────────────────────────────────────────┐   │  │
│  │  │ Shared FC2 (128→128) + ReLU                             │   │  │
│  │  └─────────────────────────────────────────────────────────┘   │  │
│  │                            │                                   │  │
│  │           ┌────────────────┼────────────────┐                  │  │
│  │           ▼                ▼                ▼                  │  │
│  │  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │  │
│  │  │ Value Stream    │ │ Advantage Stream│ │ Aggregation     │ │  │
│  │  │ (128→128→1)     │ │ (128→128→10)    │ │ Q = V + A - m   │ │  │
│  │  └─────────────────┘ └─────────────────┘ └─────────────────┘ │  │
│  │                                                                 │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                            │                                          │
│                            ▼                                          │
│                   Q-values [batch, 10]                                │
│                            │                                          │
│                            ▼                                          │
│                  Action Selection (argmax)                            │
└───────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

**File**: `src/models/combined.rs` (211 lines)

```rust
#[derive(Module, Debug)]
pub struct CombinedModel<B: Backend> {
    bandit: ContextualBandit<B>,
    dqn: QNetwork<B>,
}

#[derive(Config, Debug)]
pub struct CombinedModelConfig {
    /// Raw state input dimension (15)
    pub state_dim: usize,
    /// Hidden layer dimension (128)
    pub hidden_dim: usize,
    /// Action dimension (10)
    pub action_dim: usize,
    /// Bandit output feature dimension (64)
    pub feature_dim: usize,
}
```

### Forward Pass

```rust
impl<B: Backend> CombinedModel<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // Pass through contextual bandit
        let (bandit_features, importance) = self.bandit.forward(x);

        // Pass enhanced features to DQN
        let q_values = self.dqn.forward(bandit_features);

        // Return all outputs for training
        (bandit_features, importance, q_values)
    }

    /// Inference mode: only compute Q-values
    pub fn forward_inference(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let (_, _, q_values) = self.forward(x);
        q_values
    }
}
```

## Layer Dimensions

### Default Configuration

```rust
CombinedModelConfig {
    state_dim: 15,      // Raw state (5 tier sizes + 10 features)
    hidden_dim: 128,    // Hidden layer size
    action_dim: 10,     // 5 tiers × 2 operations
    feature_dim: 64,    // Bandit output features
}
```

### Complete Layer Sizes

| Layer | Input | Output | Parameters |
|-------|-------|--------|------------|
| Bandit FC1 | 15 | 128 | 15×128 + 128 = 2,048 |
| Bandit FC2 | 128 | 256 | 128×256 + 256 = 33,024 |
| Bandit Feature Head | 256 | 64 | 256×64 + 64 = 16,448 |
| Bandit Score Head | 256 | 1 | 256×1 + 1 = 257 |
| **Bandit Total** | | | **51,777** |
| DQN FC1 | 64 | 128 | 64×128 + 128 = 8,320 |
| DQN FC2 | 128 | 128 | 128×128 + 128 = 16,512 |
| Value Stream FC1 | 128 | 128 | 128×128 + 128 = 16,512 |
| Value Stream FC2 | 128 | 1 | 128×1 + 1 = 129 |
| Advantage Stream FC1 | 128 | 128 | 128×128 + 128 = 16,512 |
| Advantage Stream FC2 | 128 | 10 | 128×10 + 10 = 1,290 |
| **DQN Total** | | | **59,275** |
| **Grand Total** | | | **111,052** |

### Parameter Count Summary

| Component | Parameters | % of Total |
|-----------|------------|------------|
| Bandit | 51,777 | 46.6% |
| DQN | 59,275 | 53.4% |
| **Total** | **111,052** | **100%** |

## Activation Functions

### ReLU (Rectified Linear Unit)

Used in all hidden layers:

```rust
fn relu(x: f32) -> f32 {
    max(0.0, x)
}
```

Properties:
- **Non-linear**: Enables learning complex patterns
- **Sparsity**: Outputs can be zero, creating sparse representations
- **Stable**: No vanishing gradient problem for positive inputs

### Sigmoid

Used only for the importance score output:

```rust
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
}
```

Properties:
- **Bounded**: Output in [0, 1]
- **Interpretable**: Probability-like interpretation
- **Smooth gradient**: Continuous derivative

## Weight Initialization

The Burn framework uses Kaiming (He) initialization for linear layers with ReLU:

```rust
// Kaiming initialization for ReLU networks
fn kaiming_init(fan_in: usize, fan_out: usize) -> Tensor {
    let std = sqrt(2.0 / fan_in as f32);
    Tensor::randn_standard() * std
}
```

### Initialization Details

| Layer Type | Initialization | Scale |
|------------|---------------|-------|
| FC with ReLU | Kaiming (He) | √(2/fan_in) |
| Bias | Zero | 0.0 |
| Last linear (no activation) | Kaiming | √(2/fan_in) |

## Training Considerations

### Input Processing

The raw 15-dimensional state is processed as follows:

```rust
// State: [tier_sizes(5) + features(10)] = 15-dim
// Convert to tensor
let state_tensor: Tensor<B, 2> = Tensor::from_data(
    TensorData::new(state_vec, [1, 15]).convert::<f32>(),
    &device,
);

// Forward pass
let (features, importance, q_values) = model.forward(state_tensor);

// Q-values shape: [1, 10]
let q_values_array = q_values.clone().into_data().to_vec::<f32>().unwrap();

// Action selection: argmax
let action = q_values_array.iter()
    .enumerate()
    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    .map(|(idx, _)| idx)
    .unwrap();
```

### Output Interpretation

The 10 Q-values correspond to:

| Index | Action | Tier | Operation |
|-------|--------|------|-----------|
| 0 | Memory Read | 0 | Read |
| 1 | Memory Write | 0 | Write |
| 2 | NVMe Read | 1 | Read |
| 3 | NVMe Write | 1 | Write |
| 4 | SSD Read | 2 | Read |
| 5 | SSD Write | 2 | Write |
| 6 | HDD Read | 3 | Read |
| 7 | HDD Write | 3 | Write |
| 8 | Tapes Read | 4 | Read |
| 9 | Tapes Write | 4 | Write |

### Inference Optimization

For inference (no gradients needed), use `.valid()` mode:

```rust
// Training mode: gradients enabled
let q_values = model.forward(states);

// Inference mode: gradients disabled (faster)
let q_values = model.forward(states).valid();
```

### Gradient Flow

During training, gradients flow through:
1. Raw state → Bandit → Features → DQN → Q-values
2. Loss → Q-values → DQN → Bandit → Raw state

The target network receives no gradients (detached):

```rust
// Target network: no gradient flow
let (_, _, target_q) = self.target_model.forward(next_states.detach());
```

## Model Serialization

### Save Model

```rust
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
model.save_file(path, &recorder)?;
```

### Load Model

```rust
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
let model = model_config.init(&device);
let model = model.load_file(path, &recorder, &device)?;
```

## Next Steps

- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training hyperparameters
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system-level architecture
- See [TESTING.md](TESTING.md) for model testing strategies