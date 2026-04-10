# Eris Training Guide

This guide provides comprehensive instructions for training and using the eris reinforcement learning system for multi-tier storage optimization.

## Table of Contents

1. [Quick Start Training](#quick-start-training)
2. [Configuration Options](#configuration-options)
3. [Hyperparameters](#hyperparameters)
4. [Checkpoint Management](#checkpoint-management)
5. [Monitoring Training](#monitoring-training)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)
8. [Example Sessions](#example-sessions)

## Quick Start Training

### Basic Training Command

```bash
# Run the training binary with default settings
cargo run --bin train --release
```

### Command-Line Arguments

```bash
# Full options
cargo run --bin train --release -- \
    --config config/tiers.toml \
    --trace recorder-csv/NWChem-64_combined.csv \
    --episodes 500 \
    --steps-per-episode 1000 \
    --output checkpoints/ \
    --backend ndarray
```

### Programmatic Training

```rust
use eris::{
    IOBufferEnv, TrainingConfig, CombinedAgent, CombinedModelConfig,
    Config, TraceReader,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create environment
    let mut env = IOBufferEnv::new(
        Path::new("config/tiers.toml"),
        Path::new("recorder-csv/NWChem-64_combined.csv"),
        1000,  // max steps per episode
    )?;
    
    // Initialize training configuration
    let config = TrainingConfig {
        learning_rate: 0.001,
        gamma: 0.99,
        epsilon_start: 1.0,
        epsilon_end: 0.01,
        epsilon_decay: 0.995,
        batch_size: 32,
        buffer_capacity: 10_000,
        target_update_freq: 1000,
        checkpoint_interval: 10,
        max_gradient_norm: 1.0,
        backend: "ndarray".to_string(),
        tau: 0.005,
    };
    
    // Create model configuration
    let model_config = CombinedModelConfig::new(15, 128, 10);
    
    // Create agent
    let mut agent = CombinedAgent::new(
        config.clone(),
        model_config,
        &Default::default(),
    );
    
    // Training loop
    let num_episodes = 500;
    for episode in 0..num_episodes {
        let mut state = env.reset();
        let mut total_reward = 0.0;
        let mut steps = 0;
        
        loop {
            // Select action (epsilon-greedy)
            let action = agent.select_action(&state);
            
            // Take step in environment
            let (next_state, reward, done) = env.step(action);
            
            // Store transition
            agent.buffer.push(
                state.clone(),
                action,
                reward as f32,
                next_state.clone(),
                done,
            );
            
            // Train if buffer has enough samples
            if agent.buffer.len() >= config.batch_size {
                let batch = agent.buffer.sample(config.batch_size);
                let loss = agent.train_step(batch);
                println!("Episode {} Step {}: loss = {:.4}", episode, steps, loss);
            }
            
            state = next_state;
            total_reward += reward;
            steps += 1;
            
            if done || steps >= 1000 {
                break;
            }
        }
        
        println!("Episode {}: reward = {:.2}, epsilon = {:.3}", 
                 episode, total_reward, agent.epsilon);
        
        // Save checkpoint
        if episode % config.checkpoint_interval == 0 {
            agent.save_checkpoint(
                format!("checkpoints/model_ep{}", episode),
                episode,
                total_reward as f32,
            )?;
        }
    }
    
    Ok(())
}
```

## Configuration Options

### TrainingConfig Fields

```rust
pub struct TrainingConfig {
    /// Learning rate for the Adam optimizer
    /// Range: 0.0001 to 0.01
    /// Default: 0.001
    pub learning_rate: f64,
    
    /// Discount factor for future rewards (gamma)
    /// Range: 0.9 to 0.999
    /// Default: 0.99
    pub gamma: f32,
    
    /// Initial exploration rate (epsilon)
    /// Range: 0.0 to 1.0
    /// Default: 1.0
    pub epsilon_start: f32,
    
    /// Final exploration rate
    /// Range: 0.0 to 0.1
    /// Default: 0.01
    pub epsilon_end: f32,
    
    /// Exploration decay rate per step
    /// Range: 0.99 to 1.0
    /// Default: 0.995
    pub epsilon_decay: f32,
    
    /// Mini-batch size for training
    /// Range: 16 to 256
    /// Default: 32
    pub batch_size: usize,
    
    /// Maximum transitions in replay buffer
    /// Range: 1,000 to 100,000
    /// Default: 10,000
    pub buffer_capacity: usize,
    
    /// Steps between target network updates
    /// Range: 100 to 10,000
    /// Default: 1,000
    pub target_update_freq: usize,
    
    /// Soft update coefficient (for future use)
    /// Default: 0.005
    pub tau: f32,
    
    /// Backend: "ndarray" (CPU) or "wgpu" (GPU)
    /// Default: "wgpu"
    pub backend: String,
    
    /// Save checkpoint every N episodes
    /// Range: 1 to 100
    /// Default: 10
    pub checkpoint_interval: usize,
    
    /// Maximum gradient norm for clipping
    /// Range: 0.1 to 10.0
    /// Default: 1.0
    pub max_gradient_norm: f32,
}
```

### Default Configuration

```rust
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            batch_size: 32,
            buffer_capacity: 10_000,
            target_update_freq: 1000,
            tau: 0.005,
            backend: "wgpu".to_string(),
            checkpoint_interval: 10,
            max_gradient_norm: 1.0,
        }
    }
}
```

## Hyperparameters

### Learning Rate

The learning rate controls how much the model weights change during each training step.

| Value | Effect |
|-------|--------|
| 0.0001 | Slow learning, stable but may not converge |
| 0.001 | Balanced (recommended default) |
| 0.01 | Fast learning, may overshoot |
| 0.1 | Very fast, likely unstable |

**Recommendation**: Start with 0.001. Reduce if training oscillates. Increase if training is too slow.

### Discount Factor (Gamma)

Gamma controls how much future rewards are valued.

| Value | Effect |
|-------|--------|
| 0.90 | Short-sighted, cares about immediate rewards |
| 0.99 | Long-term planning (recommended) |
| 0.999 | Very long-term, may be unstable |

**Recommendation**: 0.99 for most storage optimization tasks.

### Epsilon (Exploration)

Epsilon controls the exploration vs. exploitation tradeoff.

```rust
// Epsilon decay formula
epsilon = max(epsilon_end, epsilon * epsilon_decay)

// Decay examples
epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995:
// Step 0:   1.000
// Step 100: 0.605
// Step 500: 0.082
// Step 1000: 0.013
// Step 2000: 0.002 (capped at 0.01)
```

### Batch Size

| Value | Effect |
|-------|--------|
| 16 | More variance, slower convergence |
| 32 | Balanced (recommended default) |
| 64 | Smoother gradients, more memory |
| 128 | Requires more memory |

### Target Network Update Frequency

| Value | Effect |
|-------|--------|
| 100 | Frequent updates, may be unstable |
| 1000 | Balanced (recommended) |
| 5000 | Stable but slow adaptation |

### Gradient Clipping

```rust
// Gradients are clipped to max_norm to prevent exploding gradients
let grad_norm = grads.to_tensor().norm(2.0);
if grad_norm > max_gradient_norm {
    // Scale down gradients
    grads = grads.scale(max_gradient_norm / grad_norm);
}
```

## Checkpoint Management

### Saving Checkpoints

```rust
// Save with metadata
agent.save_checkpoint(
    path: "checkpoints/model_final",
    episode: 500,
    avg_reward: -1500.0,
)?;
```

This creates three files:
- `model_final.mpk` - Policy network weights
- `model_final.target.mpk` - Target network weights
- `model_final.json` - Metadata

### Checkpoint Metadata

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub epoch: usize,
    pub step_count: usize,
    pub epsilon: f32,
    pub best_reward: f32,
    pub avg_reward_10: f32,
    pub timestamp: String,
}
```

### Loading Checkpoints

```rust
// Load trained model
let mut agent = CombinedAgent::load_checkpoint(
    path: "checkpoints/model_final",
    config: training_config,
    model_config: combined_model_config,
    device: &device,
)?;
```

### Checkpoint Best Practices

1. **Save Frequently**: Checkpoint every 10 episodes during training
2. **Save Best**: Track best reward and save separately
3. **Version Control**: Include episode number in filename
4. **Backup**: Keep multiple checkpoints during long training

## Monitoring Training

### Console Output

```bash
# Enable tracing for detailed logs
RUST_LOG=debug cargo run --bin train --release
```

Expected output format:
```bash
Episode 0: reward = -2500.50, epsilon = 0.995
Episode 1: reward = -2200.25, epsilon = 0.990
Episode 10: reward = -1850.00, epsilon = 0.951, loss = 0.5234
Episode 50: reward = -1200.75, epsilon = 0.779, loss = 0.3121
Episode 100: reward = -950.25, epsilon = 0.605, loss = 0.1987
```

### Key Metrics to Watch

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Reward | Increasing | Learning is occurring |
| Loss | Decreasing | Policy improving |
| Epsilon | Decaying | Moving to exploitation |
| Q-values | Stabilizing | Convergence |

### Convergence Indicators

1. **Reward Plateaus**: Reward stops improving after initial increase
2. **Epsilon Near Minimum**: Exploration nearly complete
3. **Loss Stabilizes**: Small fluctuations around constant value
4. **Action Distribution**: Actions concentrate on optimal choices

## Troubleshooting

### Common Issues

#### Issue: Training Doesn't Converge

**Symptoms**: Loss oscillates, reward doesn't improve

**Solutions**:
```rust
// Reduce learning rate
learning_rate: 0.0001

// Increase target update frequency
target_update_freq: 500

// Increase gradient clipping
max_gradient_norm: 0.5
```

#### Issue: Nan Loss

**Symptoms**: Loss becomes NaN during training

**Solutions**:
```rust
// Add gradient clipping
max_gradient_norm: 1.0

// Reduce learning rate
learning_rate: 0.0001

// Check for invalid rewards in environment
```

#### Issue: High Memory Usage

**Symptoms**: OOM errors, slow training

**Solutions**:
```rust
// Reduce buffer capacity
buffer_capacity: 5000

// Reduce batch size
batch_size: 16

// Use CPU backend (ndarray) instead of GPU
backend: "ndarray"
```

#### Issue: Model Doesn't Save

**Symptoms**: Checkpoint files not created

**Solutions**:
```rust
// Ensure output directory exists
std::fs::create_dir_all("checkpoints")?;

// Check file permissions
// Verify proper recorder type
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
```

#### Issue: Type Conversion Errors

**Symptoms**: Type mismatch between f32 and f64

**Solutions**:
```rust
// Explicit type conversion
let states: Tensor<B, 2> = Tensor::from_data(
    TensorData::new(states_flat, [batch_size, state_dim])
        .convert::<f32>(), 
    &device,
);
```

### Debug Mode

```bash
# Run with debug assertions
cargo run --bin train -- --debug

# Enable logging
RUST_LOG=trace cargo run --bin train 2>&1 | tee training.log
```

## Best Practices

### 1. Hyperparameter Tuning

Start with defaults and tune incrementally:

```bash
# Grid search example
for lr in 0.0001 0.001 0.01; do
    for gamma in 0.95 0.99; do
        echo "Testing lr=$lr gamma=$gamma"
        cargo run --bin train -- --learning-rate $lr --gamma $gamma
    done
done
```

### 2. Early Stopping

```rust
let mut best_reward = f32::MIN;
let patience = 50;
let mut patience_counter = 0;

for episode in 0..num_episodes {
    let reward = train_episode();
    
    if reward > best_reward {
        best_reward = reward;
        agent.save_checkpoint("checkpoints/best_model", episode, reward)?;
        patience_counter = 0;
    } else {
        patience_counter += 1;
        if patience_counter >= patience {
            println!("Early stopping at episode {}", episode);
            break;
        }
    }
}
```

### 3. Multiple Runs

```bash
# Run training multiple times for statistical significance
for seed in 1 2 3 4 5; do
    cargo run --bin train -- --seed $seed --output "results/run$seed/"
done
```

### 4. Resource Monitoring

```bash
# Monitor memory usage
watch -n 1 'ps aux | grep train | grep -v grep | awk "{print $6}"'

# Monitor GPU usage (if using wgpu)
nvidia-smi -l 1
```

### 5. Reproducibility

```rust
use rand::SeedableRng;

// Set random seeds
let mut rng = rand_pcg::Pcg64::seed_from_u64(42);

// In environment
self.random = rand_pcg::Pcg64::seed_from_u64(seed);

// In agent
self.rng = rand_pcg::Pcg64::seed_from_u64(seed);
```

## Example Sessions

### Example 1: Basic Training (100 Episodes)

```bash
# Training with default settings (100 episodes)
cargo run --bin train --release
```

Expected output:
```
Starting training...
Episode 0: steps=50, reward=-450.25, epsilon=0.995
Episode 1: steps=50, reward=-380.50, epsilon=0.990
...
Training complete!
Average reward: -375.42
```

### Example 2: Full Training (500 Episodes)

```bash
cargo run --bin train --release -- \
    --episodes 500 \
    --steps-per-episode 1000 \
    --learning-rate 0.001 \
    --gamma 0.99 \
    --checkpoint-interval 25 \
    --output checkpoints/full_training/
```

### Example 3: GPU Training

```bash
# Ensure wgpu backend is available
cargo run --bin train --release -- --backend wgpu
```

### Example 4: CPU Training (No GPU Required)

```bash
# Use ndarray backend
cargo run --bin train --release -- --backend ndarray
```

### Example 5: Hyperparameter Experiment

```rust
// Create experiment configuration
let experiments = vec![
    ("lr_0.0001", TrainingConfig { learning_rate: 0.0001, ..Default::default() }),
    ("lr_0.001", TrainingConfig { learning_rate: 0.001, ..Default::default() }),
    ("lr_0.01", TrainingConfig { learning_rate: 0.01, ..Default::default() }),
    ("gamma_095", TrainingConfig { gamma: 0.95, ..Default::default() }),
    ("gamma_099", TrainingConfig { gamma: 0.99, ..Default::default() }),
];

for (name, config) in experiments {
    println!("Running experiment: {}", name);
    let result = run_training(config)?;
    println!("Result: reward = {:.2}", result.final_reward);
}
```

### Example 6: Resume Training from Checkpoint

```rust
// Load checkpoint
let mut agent = CombinedAgent::load_checkpoint(
    "checkpoints/model_ep100",
    config,
    model_config,
    &device,
)?;

// Continue training from episode 101
for episode in 101..num_episodes {
    // ... training loop
}
```

## Next Steps

- See [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) for neural network details
- See [TESTING.md](TESTING.md) for testing strategies
- See [ARCHITECTURE.md](ARCHITECTURE.md) for system overview