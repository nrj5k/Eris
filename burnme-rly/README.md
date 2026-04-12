# burnme-rly

Optimized GPU training pipeline for Burn-based reinforcement learning.

## Overview

`burnme-rly` is a library providing efficient training infrastructure for reinforcement learning agents built with the [Burn](https://burn.rs/) deep learning framework.

## Features

- **DQNTrainer**: Double DQN with target network, gradient clipping, and async loss reporting
- **MetisTrainer**: Combined DQN + Contextual Bandit for cache tiering decisions
- **CombinedModel**: Dual-head architecture (Bandit + DQN) with importance-weighted action selection
- **GPU-Native Buffers**: CpuRingBuffer and GpuRingBuffer with batch operations
- **Builder Pattern**: Ergonomic configuration with `.with_*()` methods
- **Checkpointing**: Atomic save/load with metadata
- **Warmup Support**: Progressive batch sizing for stable training

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
burnme-rly = { path = "burnme-rly" }
```

## Feature Flags

- `cpu` (default): Enable CPU-only ndarray backend
- `cuda`: Enable CUDA backend support
- `wgpu`: Enable WebGPU backend support

## Quick Start

### DQN Training
```rust
use burnme_rly::{DQNTrainer, DQNTrainerConfig};
use burn::backend::{Autodiff, Wgpu};

let config = DQNTrainerConfig::default()
    .with_gamma(0.99)
    .with_batch_size(2048)
    .with_buffer_capacity(100_000);

let trainer = DQNTrainer::new(
    q_network,
    state_dim,
    config,
    device,
);

// Training loop
while training {
    if let Some(loss) = trainer.train_step() {
        println!("Loss: {}", loss);
    }
}
```

### Metis (Combined DQN + Bandit) Training
```rust
use burnme_rly::{MetisTrainer, MetisTrainerConfig, CombinedModel};

let config = MetisTrainerConfig::default()
    .with_bandit_loss_weight(0.5)
    .with_warmup_steps(1000);

let trainer = MetisTrainer::new(
    combined_model,
    state_dim,
    config,
    device,
);
```

## Architecture

### DQNTrainer
- Double DQN: Policy network selects, target network evaluates
- Epsilon-greedy exploration with decay
- Soft or hard target network updates
- Async loss accumulation (avoids GPU→CPU sync every step)

### MetisTrainer
- Joint loss: DQN loss + bandit_weight * bandit loss
- Contextual bandit for importance scoring
- Importance-weighted action selection
- Configurable warmup phase

### Buffers
- `CpuRingBuffer`: CPU storage, O(1) push, random sampling
- `GpuRingBuffer`: GPU storage, batch operations
- Both support reproducible sampling with seeded RNG

## Usage

```rust
use burnme_rly::{buffer, env, coordinator};

// Your RL training code here
```

## License

GPL-3.0. See the [LICENSE](../LICENSE) file in the parent project for details.
