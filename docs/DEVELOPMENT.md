# HeirGym Enhanced Models Developer Guide

This guide provides comprehensive instructions for developers working on the Eris project, covering setup, development workflows, testing, debugging, and common issues.

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Linux/macOS/Windows | Linux (Ubuntu 22.04+) |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 1 GB | 10+ GB |
| GPU | Optional | NVIDIA with 4+ GB VRAM |

### Required Software

**Rust Toolchain**
```bash
# Install Rust (requires 1.75 or later)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify installation
rustc --version  # Should be >= 1.75.0
cargo --version  # Should be >= 1.75.0

# Install additional components
rustup component add rustfmt clippy
rustup target add wasm32-unknown-unknown  # For web deployment
```

**Build Tools**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake pkg-config

# macOS (Xcode)
xcode-select --install

# Windows
# Install Visual Studio Build Tools
```

**GPU Support (Optional)**
```bash
# Ubuntu/Debian - NVIDIA drivers + Vulkan SDK
sudo apt-get install nvidia-driver-530 vulkan-tools

# Verify GPU detection
nvidia-smi
vulkaninfo | grep GPU
```

### Recommended Tools

```bash
# Rust development tools
cargo install cargo-expand         # View macro expansions
cargo install cargo-audit          # Security auditing
cargo install cargo-tree           # Dependency tree
cargo install cargo-nextest        # Faster test runner
cargo install grcov                # Code coverage

# Performance tools
sudo apt-get install valgrind perf linux-tools-common

# Editor/IDE support
# VSCode: rust-analyzer extension
# IntelliJ: Intellij Rust plugin
```

## Project Setup

### Clone and Initialize

```bash
# Clone the repository
git clone https://github.com/your-org/eris.git
cd eris

# Initialize submodules (if any)
git submodule update --init --recursive

# Verify project structure
tree -L 2 -I target
```

**Expected Structure**:
```
eris/
├── Cargo.toml
├── Cargo.lock
├── src/
│   ├── bin/
│   │   └── train.rs
│   ├── env/
│   ├── features/
│   ├── models/
│   ├── tier/
│   ├── trace/
│   ├── training/
│   ├── config.rs
│   ├── error.rs
│   └── lib.rs
├── config/
│   └── tiers.toml
├── recorder-csv/
│   └── NWChem-64_combined.csv
├── benchmarks/
├── tests/
├── docs/
└── README.md
```

### Dependency Installation

```bash
# Update dependencies
cargo update

# Build in debug mode (quick check)
cargo build

# Build in release mode (for testing)
cargo build --release

# Build with all features
cargo build --release --all-features
```

**Expected Build Time**:
- Debug: 2-5 minutes (first build), 30-60 seconds (incremental)
- Release: 5-10 minutes (first build), 1-2 minutes (incremental)

## Building

### Build Commands

```bash
# Debug build (development)
cargo build

# Release build (performance)
cargo build --release

# Build specific binary
cargo build --release --bin train

# Build all binaries and tests
cargo build --all

# Build with optimizations
cargo build --profile perf
```

### Build Options

```bash
# Disable default features
cargo build --no-default-features

# Enable specific backend
cargo build --features ndarray    # CPU only
cargo build --features wgpu       # GPU (if available)

# Enable debug features
cargo build --features debug-trace

# Enable all features
cargo build --all-features
```

### Build Verification

```bash
# Check code (without building)
cargo check
cargo check --all-targets

# Run clippy lints
cargo clippy
cargo clippy --all-targets

# Format code
cargo fmt
cargo fmt -- --check

# Generate documentation
cargo doc --no-deps
```

## Testing

### Running Tests

```bash
# Run all tests
cargo test

# Run specific tests
cargo test --lib           # Library tests only
cargo test --bin train     # Binary tests only
cargo test --tests         # Integration tests

# Run with nextest (faster)
cargo nextest run

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_feature_extraction
cargo test test_replay_buffer_push
```

### Test Coverage

```bash
# Install tarpaulin
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html
cargo tarpaulin --out Lcov

# View coverage
open tarpaulin-report.html
```

**Coverage Targets**:
- Line coverage: >90%
- Function coverage: >95%
- Branch coverage: >80%

### Benchmarking

```bash
# Run all benchmarks
cargo bench

# Run specific benchmarks
cargo bench --bench feature_extraction
cargo bench --bench environment_step
cargo bench --bench model_forward

# Compare benchmarks
cargo bench --bench feature_extraction -- --baseline compare

# Profile with flamegraph
cargo flamegraph --bin train -- --episodes 10
```

## Running Training

### Basic Training Command

```bash
cargo run --release --bin train -- \
    --config config/tiers.toml \
    --trace recorder-csv/NWChem-64_combined.csv \
    --episodes 1000 \
    --output output/model.postcard
```

### Full Command Options

```bash
cargo run --release --bin train --help
```

**Output**:
```
USAGE:
    train [OPTIONS] --config <CONFIG> --trace <TRACE>

OPTIONS:
    -c, --config <CONFIG>      Path to tier configuration file [default: config/tiers.toml]
    -t, --trace <TRACE>        Path to trace CSV file [default: recorder-csv/NWChem-64_combined.csv]
    -e, --episodes <EPISODES>  Number of training episodes [default: 1000]
    -o, --output <OUTPUT>      Path to save checkpoint [default: output/model.postcard]
    -b, --batch-size <SIZE>    Training batch size [default: 32]
    -r, --replay-size <SIZE>   Replay buffer capacity [default: 10000]
    -l, --learning-rate <LR>   Learning rate [default: 0.0001]
        --epsilon-start <E>    Initial exploration rate [default: 1.0]
        --epsilon-decay <D>    Exploration decay rate [default: 0.995]
        --epsilon-min <M>      Minimum exploration rate [default: 0.01]
    -v, --verbose              Enable verbose logging
    -h, --help                 Print help
```

### Training Examples

**Quick training (100 episodes)**:
```bash
cargo run --release --bin train -- \
    --config config/tiers.toml \
    --trace recorder-csv/NWChem-64_combined.csv \
    --episodes 100 \
    --output output/quick_model.postcard
```

**Full training (10000 episodes)**:
```bash
cargo run --release --bin train -- \
    --config config/tiers.toml \
    --trace recorder-csv/NWChem-64_combined.csv \
    --episodes 10000 \
    --output output/full_model.postcard \
    --learning-rate 0.0001 \
    --batch-size 64
```

**Continue training from checkpoint**:
```bash
cargo run --release --bin train -- \
    --config config/tiers.toml \
    --trace recorder-csv/NWChem-64_combined.csv \
    --episodes 500 \
    --resume output/model.postcard
```

### Training Output

**Console Output**:
```
2024-01-15T10:30:00Z INFO  train: Starting training
2024-01-15T10:30:00Z INFO  train: Episodes: 1000, Buffer: 10000
2024-01-15T10:30:01Z INFO  train::episode: Episode 0: reward=-1234.56, epsilon=0.990
2024-01-15T10:30:02Z INFO  train::episode: Episode 100: reward=-987.65, epsilon=0.904
2024-01-15T10:30:05Z INFO  train::episode: Episode 500: reward=-456.78, epsilon=0.606
2024-01-15T10:30:10Z INFO  train::episode: Episode 1000: reward=-123.45, epsilon=0.010
2024-01-15T10:30:10Z INFO  train: Training complete: 1000 episodes in 10.2s
```

**Metrics**:
- Episode number
- Total reward (higher is better, less negative)
- Epsilon value (exploration rate)
- Training time

## Running Benchmarks

### Feature Extraction Benchmark

```bash
cargo bench --bench feature_extraction
```

**Expected Output**:
```
feature_extraction_single      time:   [52.345 us 53.102 us 54.001 us]
feature_extraction_batch_100   time:   [4.231 ms 4.345 ms 4.512 ms]
```

### Environment Step Benchmark

```bash
cargo bench --bench environment_step
```

**Expected Output**:
```
env_step        time:   [78.234 us 79.567 us 81.123 us]
env_reset       time:   [45.678 us 46.890 us 48.234 us]
```

### Model Forward Pass Benchmark

```bash
cargo bench --bench model_forward
```

**Expected Output**:
```
model_forward               time:   [234.56 us 245.67 us 256.78 us]
model_forward_batch_32      time:   [5.234 ms 5.456 ms 5.678 ms]
```

## Debugging

### Common Issues

#### 1. Build Fails with Missing Dependencies

```bash
# Error: Could not find native library
sudo apt-get install pkg-config libssl-dev cmake
```

#### 2. GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Vulkan
vulkaninfo | grep GPU

# Use CPU backend instead
cargo run --features ndarray --release --bin train
```

#### 3. Out of Memory

```bash
# Reduce replay buffer size
cargo run --release --bin train -- --replay-size 5000

# Reduce batch size
cargo run --release --bin train -- --batch-size 16

# Use debug build (less memory intensive)
cargo run --bin train
```

#### 4. Training Doesn't Converge

```bash
# Check learning rate
# Too high: reduce learning rate
cargo run --release --bin train -- --learning-rate 0.00001

# Too low: increase learning rate
cargo run --release --bin train -- --learning-rate 0.001

# Adjust epsilon decay
# Faster decay
cargo run --release --bin train -- --epsilon-decay 0.990
# Slower decay
cargo run --release --bin train -- --epsilon-decay 0.999
```

### Debug Modes

```rust
// Enable debug logging
RUST_LOG=debug cargo run --release --bin train

// Enable trace logging
RUST_LOG=trace cargo run --release --bin train

// Enable specific module logging
RUST_LOG=eris::env=debug cargo run --release --bin train
```

### Debug Print Statements

```rust
// In your code
println!("State: {:?}", state);
println!("Features: {:?}", features);
println!("Action: {:?}", action);
println!("Reward: {:?}", reward);

// Use debug tracing
use tracing::{info, debug, error};

info!("Episode {} started", episode);
debug!("State shape: {:?}", state.shape());
error!("Failed to save checkpoint");
```

### Memory Debugging

```bash
# Valgrind massif
valgrind --tool=massif cargo run --release --bin train --episodes 10
ms_print massif.out.*

# DHAT profiler
cargo run --features dhat --release --bin train --episodes 10
```

### CPU Profiling

```bash
# Perf record
perf record -g cargo run --release --bin train --episodes 100
perf report

# Flamegraph
cargo flamegraph --bin train -- --episodes 100
```

## IDE Setup

### VSCode

**Extensions**:
- `rust-analyzer` - Rust language server
- `CodeLLDB` - Debugger
- `Even Better TOML` - TOML support

**Settings** (`~/.vscode/settings.json`):
```json
{
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.cargo.features": ["all"],
    "editor.formatOnSave": true,
    "files.watcherExclude": {
        "**/target": true
    }
}
```

### IntelliJ IDEA

**Plugins**:
- IntelliJ Rust
- TOML

**Configuration**:
- Set `cargo.features` to `all`
- Enable `proc macros` support

## Performance Tuning Guide

### 1. Backend Selection

```toml
# Cargo.toml
[dependencies]
burn = { version = "0.14", default-features = false, features = ["ndarray"] }

# For GPU support
burn = { version = "0.14", features = ["wgpu"] }
```

### 2. Compilation Optimizations

```toml
# Cargo.toml
[profile.release]
opt-level = 3          # Maximum optimization
lto = "fat"            # Link-time optimization
codegen-units = 1      # Single codegen unit for better optimization
panic = "abort"        # Smaller binary, no unwinding

[profile.perf]
inherits = "release"
opt-level = 3
lto = "fat"
codegen-units = 1
```

### 3. Runtime Settings

```bash
# Enable CPU features
export RUSTFLAGS="-C target-cpu=native"

# For NVIDIA GPU
export RUSTFLAGS="-C target-cpu=native -C link-arg=-Wl,--no-as-needed"

# Multi-threaded training
export TOKIO_WORKER_THREADS=8
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Build fails | Missing system deps | Install `build-essential cmake pkg-config` |
| GPU OOM | Too large batch | Reduce `--batch-size` |
| Training diverges | Learning rate too high | Reduce `--learning-rate` |
| Low throughput | CPU backend only | Install NVIDIA driver + Vulkan SDK |
| Memory leak | Unbounded buffer | Use bounded `VecDeque` |
| Slow compilation | Incremental cache | Run `cargo clean` |

## Release Process

### Version Bump

```bash
# Update version in Cargo.toml
# Semantic versioning: MAJOR.MINOR.PATCH

# Commit version change
git add Cargo.toml
git commit -m "Bump version to v0.1.0"
git tag v0.1.0
```

### Build Release Artifacts

```bash
# Build for current platform
cargo build --release

# Build for other platforms (cross-compilation)
cargo build --release --target x86_64-unknown-linux-gnu
cargo build --release --target x86_64-pc-windows-gnu
cargo build --release --target aarch64-unknown-linux-gnu
```

### Publish to Crates.io

```bash
# Login to crates.io
cargo login

# Dry run
cargo publish --dry-run

# Publish
cargo publish
```

## Contributing

### Pull Request Workflow

```bash
# Fork repository
# Create feature branch
git checkout -b feature/my-feature

# Make changes
# Add tests
cargo test

# Run linting
cargo fmt
cargo clippy

# Commit
git add .
git commit -m "Add feature: my feature"

# Push
git push origin feature/my-feature

# Create PR on GitHub
```

### Code Style

```bash
# Format code
cargo fmt

# Check formatting
cargo fmt -- --check

# Run linter
cargo clippy --all-targets -- -D warnings
```

### Testing Requirements

- All new features must have unit tests
- Integration tests for new components
- Documentation for public APIs
- Benchmarks for performance-critical code

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System design overview
- [API Reference](API.md) - Detailed API documentation
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Development roadmap
- [Performance](PERFORMANCE.md) - Performance targets and benchmarks
- [Data Formats](DATA_FORMATS.md) - Input/output specifications