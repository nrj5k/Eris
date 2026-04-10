# API Documentation Overview - Eris RL Library v2.0

This document provides comprehensive API documentation for the refactored Eris library, focusing on major changes introduced in version 2.0.

## Overview

The Eris library has been refactored to use dynamic dimensions instead of hardcoded constants, providing greater flexibility and better support for different problem sizes.

## Breaking Changes

### Removed: OBSERVATION_DIM Constant

**Version 1.x:**
```rust
const OBSERVATION_DIM: usize = 15;
let env = MyEnv::new(); // Uses hardcoded dimension
```

**Version 2.x:**
```rust
let env = MyEnv::new_with_dims(100, 50, 20); // Dynamic dimensions
let state_dim = env.observation_space().dim(); // Query dimension at runtime
```

## Public API Reference

### Space Types

#### [`Space`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/space/mod.rs)

Trait defining common interface for observation/action spaces.

**Key Methods:**
- `dim()` - Get space dimension
- `sample()` - Generate random valid value
- `contains(value)` - Validate value belongs to space

**Implementations:**
- [`BoxSpace`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/space/box_space.rs) - Bounded continuous space
- [`DiscreteSpace`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/space/discrete_space.rs) - Discrete action space

#### [`BoxSpace`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/space/box_space.rs)

Continuous multidimensional space with bounded values.

**Construction:**
```rust
// With uniform bounds
let space = BoxSpace::uniform(4, -10.0, 10.0);

// With custom bounds per dimension
let space = BoxSpace::new(
    vec![0.0, -1.0, 0.0],
    vec![1.0, 1.0, 10.0],
    vec![3]
).unwrap();
```

**Use Cases:**
- Observation vectors with bounded values
- Continuous action spaces
- Feature normalization

#### [`DiscreteSpace`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/space/discrete_space.rs)

Discrete action space with n possible actions.

**Construction:**
```rust
let space = DiscreteSpace::new(5); // Actions indexed 0-4
```

**Key Methods:**
- `sample_action()` - Return random action index
- `contains_action(action)` - Validate action index
- `sample()` - Return Vec<f64> with action index

### Environment Trait

#### [`Environment`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/env/trait.rs)

Main interface for RL environments.

**Methods:**
- `reset()` - Reset environment, return initial observation
- `step(action)` - Take action, return StepResult
- `observation_space()` - Get observation space definition
- `action_space()` - Get action space definition
- `seed(seed)` - Set random seed
- `render()` - Visualize environment (optional)
- `close()` - Cleanup resources (optional)

**StepResult Structure:**
```rust
struct StepResult {
    observation: Vec<f64>,
    reward: f64,
    done: bool,
    info: Info,
}
```

**Example Implementation:**
```rust
struct MyEnv {
    // ... fields
}

impl Environment for MyEnv {
    type Observation = Vec<f64>;
    type Action = usize;
    
    fn reset(&mut self) -> Self::Observation {
        // Return initial observation
    }
    
    fn step(&mut self, action: Self::Action) -> StepResult {
        // Execute action and return (obs, reward, done, info)
    }
    
    fn observation_space(&self) -> BoxSpace {
        BoxSpace::uniform(4, -10.0, 10.0)
    }
    
    fn action_space(&self) -> DiscreteSpace {
        DiscreteSpace::new(5)
    }
    
    fn seed(&mut self, seed: u64) {
        // Set random seed
    }
}
```

### Training Module

#### [`train_agent`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/training/coordinator.rs)

Main training function for DQN agents.

**Signature:**
```rust
fn train_agent<B, E>(
    env: &mut E,
    agent: &mut CombinedAgent<B>,
    num_episodes: usize,
    tier_selector: &TierSelector,
) -> TrainingResult
```

**Parameters:**
- `env` - Environment implementing Environment trait
- `agent` - Agent with model and replay buffer
- `num_episodes` - Number of episodes to train
- `tier_selector` - Tier selector for action selection

**Returns:** TrainingResult with episode rewards, losses, and final epsilon

**Training Process:**
1. Loop through episodes
2. Within each episode, take actions until done
3. Use epsilon-greedy action selection
4. Store transitions in replay buffer
5. Train when buffer has enough samples
6. Decay epsilon after each episode

#### [`TrainingResult`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/training/coordinator.rs)

Training metrics container.

**Fields:**
- `episode_rewards` - Rewards for each episode
- `losses` - Training losses per optimization step
- `final_epsilon` - Final exploration rate

**Usage:**
```rust
let result = train_agent(env, agent, 100, &tier_selector);

println!("Average reward: {}", 
    result.episode_rewards.iter().sum::<f32>() / result.episode_rewards.len() as f32);
println!("Final epsilon: {}", result.final_epsilon);
```

#### [`MockEnv`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/training/mock_env.rs)

Mock environment for testing without external dependencies.

**Construction:**
```rust
// With default dimensions (15 obs, 10 actions)
let env = MockEnv::new(100);

// With custom dimensions
let env = MockEnv::new_with_dims(100, 50, 20);
```

**Dynamic Dimensions:**
```rust
let mut env = MockEnv::new_with_dims(100, 50, 20);

// Get dimensions
let obs_dim = env.observation_space().dim();  // 50
let action_dim = env.action_space().n;        // 20

// Use Environment trait
let obs = env.reset();
let result = env.step(0);
```

**Legacy API Compatibility:**
MockEnv supports both legacy and Environment trait APIs:
- Legacy: `env.reset()`, `env.step()`, `env.observation_space()`, `env.action_space()`
- Trait: `<MockEnv as Environment>::reset()`, etc.

### Model Configuration

#### [`ErisDefaults`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/model.rs)

Default configurations for RL models.

**Key Methods:**
- `storage_tier_model(state_dim, action_dim)` - Full-featured configuration
- `compact_model(state_dim, action_dim)` - Smaller configuration for fast training

**Usage:**
```rust
use eris::model::ErisDefaults;
use eris::training::MockEnv;
use eris::env::Environment;
use eris::space::Space;

// Get dimensions dynamically
let env = MockEnv::new_with_dims(100, 50, 20);
let state_dim = env.observation_space().dim();
let action_dim = env.action_space().n;

// Create default configuration
let config = ErisDefaults::storage_tier_model(state_dim, action_dim);
```

## Migration Guide

### From v1.x to v2.x

**Before (v1.x):**
```rust
use eris::OBSERVATION_DIM; // Removed in v2.x

let env = MyEnv::new();
let obs = env.reset();
assert_eq!(obs.len(), OBSERVATION_DIM); // Hardcoded
```

**After (v2.x):**
```rust
let mut env = MyEnv::new_with_dims(100, 50, 20);
let obs = env.reset();
let state_dim = env.observation_space().dim(); // Dynamic
assert_eq!(obs.len(), state_dim);
```

### Benefits of v2.x

1. **Flexibility**: Support any state/action dimensions
2. **Discoverability**: Dynamic queries via trait methods
3. **Testing**: Easy creation of environments with custom sizes
4. **Maintainability**: No hardcoded constants to update
5. **Consistency**: OpenAI Gym-style interface

## Complete Example

```rust
use eris::training::{train_agent, CombinedAgent, TrainingConfig};
use eris::env::Environment;
use eris::tier::TierSelector;
use eris::training::MockEnv;
use eris::space::Space;

// Create environment
let mut env = MockEnv::new_with_dims(100, 50, 20);
let state_dim = env.observation_space().dim();
let action_dim = env.action_space().n;

// Create model configuration
let model_config = eris::models::CombinedModelConfig::new(
    state_dim, 
    20, 
    128, 
    action_dim
);
let training_config = TrainingConfig::default();
let device = burn::backend::ndarray::NdArrayDevice::Cpu;
let mut agent = CombinedAgent::new(training_config, model_config, device);

// Train
let tier_selector = TierSelector::new(vec![]);
let result = train_agent(&mut env, &mut agent, 10, &tier_selector);

// Analyze results
println!("Trained for {} episodes", result.episode_rewards.len());
println!("Average reward: {:.2}", 
    result.episode_rewards.iter().sum::<f32>() / result.episode_rewards.len() as f32);
```

## API Documentation Files

- [`lib.rs`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/lib.rs) - Main library exports
- [`space/mod.rs`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/space/mod.rs) - Space trait documentation
- [`space/box_space.rs`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/space/box_space.rs) - BoxSpace type documentation
- [`space/discrete_space.rs`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/space/discrete_space.rs) - DiscreteSpace type documentation
- [`env/trait.rs`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/env/trait.rs) - Environment trait documentation
- [`training/coordinator.rs`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/training/coordinator.rs) - Training functions documentation
- [`training/mock_env.rs`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/training/mock_env.rs) - MockEnv documentation
- [`model.rs`](/home/neeraj/syncthing/machine-sync-folder/sync-in-here/Git_repos/eris/src/model.rs) - Model configuration documentation
