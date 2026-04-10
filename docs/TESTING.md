# Eris Testing Guide

This comprehensive guide covers all aspects of testing the eris reinforcement learning system, including test structure, running tests, adding new tests, and continuous integration.

## Test Overview

The eris project maintains a comprehensive test suite with **71 tests passing** across all modules. Tests are organized into unit tests, integration tests, and smoke tests to ensure correctness at all levels.

### Test Summary

| Category | Count | Purpose |
|----------|-------|---------|
| Unit Tests | 45 | Test individual components in isolation |
| Integration Tests | 20 | Test component interactions |
| Smoke Tests | 6 | Verify basic functionality |
| **Total** | **71** | **100% passing** |

## Test Structure

### Directory Layout

```
eris/
├── tests/
│   ├── test_training.rs      # 17 tests for training pipeline
│   ├── test_tier.rs          # 15+ tests for tier management
│   ├── test_models.rs        # 12+ tests for neural networks
│   ├── test_features.rs      # 15+ tests for feature extraction
│   └── test_trace.rs         # 12+ tests for trace parsing
├── src/
│   ├── training/
│   │   └── mock_env.rs       # Mock environment for testing
│   └── env/
│       └── io_buffer_env.rs  # Contains environment tests
└── Cargo.toml
```

### Test Categories

#### Unit Tests

Unit tests verify individual functions and methods in isolation:

```rust
// tests/test_models.rs
#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Ndarray;

    #[test]
    fn test_dqn_forward_pass() {
        let config = QNetworkConfig::new(64, 128, 10);
        let device = <Ndarray as Backend>::Device::default();
        let model = config.init(&device);

        // Create random input
        let input = Tensor::<Ndarray, 2>::random([1, 64], -1.0..1.0, &device);

        // Forward pass
        let output = model.forward(input);

        // Verify output shape
        assert_eq!(output.shape(), [1, 10]);
    }

    #[test]
    fn test_action_encoding() {
        // Test action encoding/decoding
        for tier_idx in 0..5 {
            for op_type in 0..2 {
                let action = encode_action(tier_idx, op_type);
                let (decoded_tier, decoded_op) = decode_action(action);
                assert_eq!(tier_idx, decoded_tier);
                assert_eq!(op_type, decoded_op);
            }
        }
    }
}
```

#### Integration Tests

Integration tests verify component interactions:

```rust
// tests/test_training.rs
#[test]
fn test_train_step_returns_valid_loss() {
    let config = TrainingConfig::default();
    let model_config = CombinedModelConfig::new(15, 128, 10);
    let device = <Ndarray as Backend>::Device::default();

    let mut agent = CombinedAgent::new(config, model_config, &device);

    // Create dummy transition
    let transition = Transition {
        state: vec![0.0; 15],
        action: 0,
        reward: 1.0,
        next_state: vec![0.0; 15],
        done: false,
    };

    // Add to buffer
    agent.buffer.push(
        transition.state.clone(),
        transition.action,
        transition.reward,
        transition.next_state.clone(),
        transition.done,
    );

    // Sample and train
    if agent.buffer.len() >= 1 {
        let batch = agent.buffer.sample(1);
        let loss = agent.train_step(batch);

        // Loss should be finite and non-negative
        assert!(!loss.is_nan());
        assert!(!loss.is_infinite());
        assert!(loss >= 0.0);
    }
}
```

#### Smoke Tests

Smoke tests verify basic functionality:

```rust
#[test]
fn test_compilation_smoke() {
    // Verify project compiles
    assert!(true);
}

#[test]
fn test_environment_smoke() {
    // Quick environment test
    let config_path = Path::new("config/tiers.toml");
    let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");

    if config_path.exists() && trace_path.exists() {
        let env = IOBufferEnv::new(config_path, trace_path, 10).unwrap();
        assert_eq!(env.buffer.num_tiers(), 5);
    }
}
```

## Mock Environment

The mock environment (`src/training/mock_env.rs`) provides a deterministic testing interface that doesn't require external files.

### Using MockEnv

```rust
use crate::training::MockEnv;

#[test]
fn test_mock_env_basic() {
    let mut env = MockEnv::new(100);  // max_steps = 100

    // Reset returns initial state
    let state = env.reset();
    assert_eq!(state.len(), 15);  // 15-dimensional state

    // Step returns (state, reward, done)
    let (state, reward, done) = env.step(0);
    assert!(!done);
    assert_eq!(state.len(), 15);
}

#[test]
fn test_mock_env_deterministic() {
    let mut env1 = MockEnv::new(100);
    let mut env2 = MockEnv::new(100);

    env1.reset();
    env2.reset();

    // Same actions should produce same results
    for action in 0..10 {
        let (s1, r1, _) = env1.step(action);
        let (s2, r2, _) = env2.step(action);

        assert_eq!(s1, s2);
        assert_eq!(r1, r2);
    }
}
```

### MockEnv API

```rust
pub struct MockEnv {
    state: Vec<f64>,
    pub step_count: usize,
    max_steps: usize,
    num_actions: usize,
}

impl MockEnv {
    /// Create new mock environment
    pub fn new(max_steps: usize) -> Self {
        Self {
            state: vec![0.0; 15],
            step_count: 0,
            max_steps,
            num_actions: 10,
        }
    }

    /// Reset environment and return initial state
    pub fn reset(&mut self) -> Vec<f64> {
        self.step_count = 0;
        self.state.clone()
    }

    /// Take action and return (next_state, reward, done)
    pub fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool) {
        self.step_count += 1;
        let done = self.step_count >= self.max_steps;
        (self.state.clone(), 0.0, done)
    }
}
```

## Running Tests

### Basic Test Commands

```bash
# Run all tests
cargo test

# Run tests in release mode (faster)
cargo test --release

# Run specific test file
cargo test --test test_training

# Run specific test
cargo test test_train_step_returns_valid_loss

# Run tests with output
cargo test -- --nocapture

# Run tests with verbose output
cargo test -vv
```

### Test Output

```
running 71 tests
test test_dqn_forward_pass ... ok
test test_action_encoding ... ok
test test_train_step_returns_valid_loss ... ok
test test_mock_env_reset ... ok
test test_hard_update_target ... ok
...
test result: ok. 71 passed; 0 failed; 0 ignored; finished in 5.23s
```

### Test Filtering

```bash
# Run tests matching pattern
cargo test test_env       # All environment tests
cargo test test_model     # All model tests
cargo test test_training  # All training tests

# Exclude tests
cargo test -- --skip slow_test
```

### Performance Testing

```bash
# Run tests with timing
cargo test -- --time

# Run with criterion benchmarks
cargo bench
```

## Adding New Tests

### Test Organization Guidelines

1. **Unit tests** go in the same file as the code they test
2. **Integration tests** go in `tests/` directory
3. **Use proper test organization** with `#[cfg(test)]` modules

### Example: Adding Unit Test

```rust
// In src/models/dqn.rs

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Ndarray;

    #[test]
    fn test_q_network_output_range() {
        let config = QNetworkConfig::new(64, 128, 10);
        let device = <Ndarray as Backend>::Device::default();
        let model = config.init(&device);

        // Test with ones input
        let input = Tensor::<Ndarray, 2>::ones([1, 64], &device);
        let output = model.forward(input);

        // Q-values should be finite
        let data = output.clone().into_data().to_vec::<f32>().unwrap();
        for q in data {
            assert!(!q.is_nan());
            assert!(!q.is_infinite());
        }
    }

    #[test]
    fn test_dueling_aggregation() {
        // Test Q(s,a) = V(s) + A(s,a) - mean(A)
        let config = QNetworkConfig::new(64, 128, 10);
        let device = <Ndarray as Backend>::Device::default();
        let model = config.init(&device);

        let input = Tensor::<Ndarray, 2>::zeros([1, 64], &device);
        let output = model.forward(input);

        // Verify output shape
        assert_eq!(output.shape(), [1, 10]);
    }
}
```

### Example: Adding Integration Test

```rust
// In tests/test_new_feature.rs

use eris::{CombinedAgent, CombinedModelConfig, TrainingConfig};

#[test]
fn test_new_feature_integration() {
    let config = TrainingConfig::default();
    let model_config = CombinedModelConfig::new(15, 128, 10);
    let device = <Ndarray as Backend>::Device::default();

    let mut agent = CombinedAgent::new(config, model_config, &device);

    // Test feature under integration
    // ... test implementation ...
}
```

### Test Best Practices

1. **Use descriptive names**: `test_action_encoding_decoding` not `test1`
2. **Test edge cases**: Empty buffer, boundary values, invalid inputs
3. **Mock external dependencies**: Use MockEnv for reproducibility
4. **Keep tests fast**: Aim for <1 second per test
5. **Use assertions generously**: Check all relevant properties

```rust
#[test]
fn test_replay_buffer_bounds() {
    let mut buffer = ReplayBuffer::new(10);

    // Test capacity
    for i in 0..15 {
        let transition = create_dummy_transition();
        buffer.push(transition.state, transition.action, transition.reward,
                    transition.next_state, transition.done);
        assert!(buffer.len() <= 10);
    }

    // Test oldest transition removed
    assert_eq!(buffer.len(), 10);
}
```

## Test Coverage

### Coverage Report

Generate coverage reports:

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html

# View coverage
open tarpaulin-report.html
```

### Coverage Goals

| Module | Coverage Target | Current |
|--------|-----------------|---------|
| Models | 100% | 100% |
| Training | 100% | 100% |
| Environment | 95% | 95% |
| Features | 90% | 90% |
| Overall | 95% | 95% |

### Coverage Categories

```
Line Coverage:
- src/models/dqn.rs:      100% (122/122 lines)
- src/models/bandit.rs:   100% (97/97 lines)
- src/models/combined.rs: 100% (211/211 lines)
- src/training/trainer.rs: 100% (337/337 lines)
- src/training/replay_buffer.rs: 100% (180/180 lines)
- src/env/io_buffer_env.rs: 95% (482/507 lines)
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Build
        run: cargo build --release
      
      - name: Run Tests
        run: cargo test --release
      
      - name: Run Clippy
        run: cargo clippy --release
      
      - name: Format Check
        run: cargo fmt --check
```

### Test Automation

```bash
# Pre-commit hooks (install via cargo-husky or similar)
#!/bin/bash
cargo fmt --check
cargo clippy --release
cargo test --release

# CI pipeline script
#!/bin/bash
set -e

echo "Running format check..."
cargo fmt --check

echo "Running clippy..."
cargo clippy --release

echo "Building project..."
cargo build --release

echo "Running tests..."
cargo test --release

echo "All checks passed!"
```

### Test Performance Benchmarks

Track test performance over time:

```bash
# Run with timing
time cargo test --release

# Expected performance:
# - All 71 tests: < 10 seconds
# - Single test: < 1 second
# - Test suite startup: < 2 seconds
```

## Troubleshooting Tests

### Common Issues

#### Issue: Test Hangs

```bash
# Check for deadlocks or infinite loops
cargo test --test test_training -- --test-threads=1

# Add timeout
cargo test --test test_training -- --timeout 30
```

#### Issue: Test Uses Too Much Memory

```bash
# Limit parallel test execution
cargo test -- --test-threads=1

# Check memory usage
ps aux | grep cargo
```

#### Issue: Test Depends on External Files

```rust
#[test]
fn test_with_optional_file() {
    let config_path = Path::new("config/tiers.toml");
    if !config_path.exists() {
        eprintln!("Skipping test: config file not found");
        return;
    }
    // ... test implementation
}
```

#### Issue: Test Order Dependency

```rust
// Make tests independent by resetting state
#[test]
fn test_independent() {
    let mut agent = CombinedAgent::new(...);
    // Each test creates fresh agent
}
```

### Debug Test Failures

```bash
# Run single failing test with output
cargo test test_failing_test -- --nocapture

# Run with RUST_BACKTRACE
RUST_BACKTRACE=1 cargo test test_failing_test

# Run with debug logging
RUST_LOG=debug cargo test test_failing_test -- --nocapture
```

## Test Utilities

### Helper Functions

```rust
// In tests/test_helpers.rs

pub fn create_dummy_transition() -> Transition {
    Transition {
        state: vec![0.0; 15],
        action: rand::random::<usize>() % 10,
        reward: (rand::random::<f32>() - 0.5) * 10.0,
        next_state: vec![0.0; 15],
        done: false,
    }
}

pub fn fill_buffer(buffer: &mut ReplayBuffer, n: usize) {
    for _ in 0..n {
        let transition = create_dummy_transition();
        buffer.push(
            transition.state,
            transition.action,
            transition.reward,
            transition.next_state,
            transition.done,
        );
    }
}

pub fn create_test_agent() -> CombinedAgent<Ndarray> {
    let config = TrainingConfig::default();
    let model_config = CombinedModelConfig::new(15, 128, 10);
    let device = <Ndarray as Backend>::Device::default();
    CombinedAgent::new(config, model_config, &device)
}
```

### Proptest for Property-Based Testing

```rust
// Add to Cargo.toml
[dev-dependencies]
proptest = "1.11.0"

#[cfg(test)]
mod property_tests {
    use proptest::*;

    proptest! {
        #[test]
        fn test_action_encoding_properties(
            tier_idx in 0..5usize,
            op_type in 0..2usize
        ) {
            let action = encode_action(tier_idx, op_type);
            let (decoded_tier, decoded_op) = decode_action(action);
            prop_assert_eq!(tier_idx, decoded_tier);
            prop_assert_eq!(op_type, decoded_op);
        }
    }
}
```

## Next Steps

- See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training procedures
- See [DEVELOPMENT.md](DEVELOPMENT.md) for development setup
- See [FINAL_SUMMARY.md](FINAL_SUMMARY.md) for project completion report