# Eris System Architecture

This document provides a comprehensive overview of the eris system's architecture, covering all components, data flows, and design decisions.

## High-Level Architecture

The eris system follows a classic reinforcement learning pipeline adapted for storage optimization:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Trace CSV Files                                                         │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────────────┐                                                     │
│  │  TraceReader    │  Parses CSV → BlobData                              │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────┐     ┌───────────────────────────────────────────┐  │
│  │  IOBufferEnv    │────▶│  State: [5 tier sizes + 10 features]      │  │
│  │  (RL Env)       │◀────│  15-dimensional observation space         │  │
│  └────────┬────────┘     └───────────────────────────────────────────┘  │
│           │                                                              │
│           │ Action (0-9)                                                 │
│           │ 5 tiers × 2 ops = 10 actions                                 │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    CombinedAgent                                 │    │
│  │  ┌─────────────────┐    ┌────────────────────────────────────┐  │    │
│  │  │ Contextual      │───▶│ DQN (Dueling Architecture)         │  │    │
│  │  │ Bandit          │    │ ┌────────┐  ┌──────────────────┐  │  │    │
│  │  │ - Feature extr  │    │ │  V(s)  │  │  A(s,a)          │  │  │    │
│  │  │ - Importance    │    │ │ Stream │  │  Stream          │  │  │    │
│  │  └─────────────────┘    │ └────┬───┘  └─────┬────────────┘  │  │    │
│  │                          │      │            │               │  │    │
│  │                          │      ▼            ▼               │  │    │
│  │                          │  Q(s,a) = V(s) + A(s,a) - mean(A)  │  │    │
│  │                          └────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           │ Reward (negative latency)                                    │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Replay Buffer                                 │    │
│  │         Stores: (state, action, reward, next_state, done)        │    │
│  │              Capacity: 10,000 transitions                        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           │ Batch sampling (32)                                         │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    Training Loop                                 │    │
│  │  - Adam optimizer (lr=0.001, beta1=0.9, beta2=0.999)            │    │
│  │  - Target network updates (hard copy every 1000 steps)           │    │
│  │  - Epsilon-greedy exploration (decay 0.995→0.01)                 │    │
│  │  - Gradient clipping (max_norm=1.0)                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ReplayBuffer (VecDeque<Transition>)                │   │
│  │  - capacity: 10K                                     │   │
│  │  - sample(batch_size)                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  CombinedAgent                                       │   │
│  │  - ContextualBandit: state → features + importance │   │
│  │  - QNetwork: features → Q-values                    │   │
│  │  - TierSelector: importance → tier (capacity-weighted) │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  IOBufferEnv (gymnasia::core::Env)                  │   │
│  │  - 5 tiers (Memory, NVMe, SSD, HDD, Tapes)         │   │
│  │  - state: [tier_sizes(5) + features(10)] = 15-dim  │   │
│  │  - action: (tier_idx, op_type) → encoded as 0-9     │   │
│  │  - reward: -sum(latency × count)                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Features::AccessTracker                             │   │
│  │  - History: VecDeque<AccessRecord> (10K window)      │   │
│  │  - Extract: recency, frequency, intervals, etc      │   │
│  │  - Hotness: unified score for RL + eviction          │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  TraceReader                                         │   │
│  │  - CSV parsing (11 columns)                         │   │
│  │  - Streaming iterator                                │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Trace Reader (`src/trace/`)

The trace reader is responsible for loading and parsing the CSV trace data that drives the simulation. It implements streaming parsing to handle large trace files without loading them entirely into memory.

Key responsibilities:
- Parse CSV format with 11 columns
- Convert string values to typed structs (`BlobData`)
- Provide iterator-based access for streaming
- Handle missing values and type conversions

#### 2. Access Tracker (`src/features/tracker.rs`)

The access tracker maintains a sliding window of access records and computes real-time features for each blob. It uses a combination of in-memory and memory-mapped storage for efficiency.

Key responsibilities:
- Maintain access history window (10K records default)
- Extract features for any blob on demand
- Compute hotness scores for eviction decisions
- Support memory-mapped overflow for large datasets

#### 3. Feature Extractor (`src/features/extractor.rs`)

The feature extractor transforms raw access patterns into a normalized 10-dimensional feature vector used by the RL models.

Features extracted:
- **Recency**: Time since last access (normalized 0-1)
- **Frequency**: Access count relative to maximum
- **Mean Interval**: Average time between accesses
- **Std Interval**: Variability in access patterns
- **Sequential Flag**: Whether accesses are sequential
- **Reuse Distance**: Position of last access in history
- **Access Type**: Read vs write ratio
- **Blob Size**: Relative size classification
- **Next Access Prediction**: Predicted next access time
- **Overwrite Amount**: Write frequency ratio

#### 4. Tier Management (`src/tier/`)

The tier subsystem manages the actual storage tiers and implements capacity-aware selection.

Components:
- **Tier**: Individual storage tier with HashMap storage
- **TierSelector**: Capacity-weighted importance to tier mapping
- **TierManager**: Coordinates tier operations and demotion

#### 5. RL Environment (`src/env/io_buffer_env.rs`)

The IOBufferEnv implements the gymnasium environment interface, providing:
- **State Space**: 15-dimensional continuous vector
- **Action Space**: 10 discrete actions (5 tiers × 2 operations)
- **Reward Function**: Negative weighted latency

#### 6. Neural Network Models (`src/models/`)

The model subsystem contains the burn-based neural network implementations:
- **QNetwork**: 3-layer MLP for Q-value estimation
- **ContextualBandit**: Feature extraction + importance scoring
- **CombinedModel**: Integrated model with tier selection

#### 7. Training Infrastructure (`src/training/`)

The training subsystem provides:
- **ReplayBuffer**: Experience replay with priority sampling
- **CombinedAgent**: Main RL agent with training logic
- **Checkpoint**: Model persistence with postcard serialization

## Data Flow

### Complete Pipeline

The following diagram shows the complete data flow from CSV trace to model update:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Trace Data Flow                                 │
└──────────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────┐
   │ CSV Trace File  │
   │ (18K+ records)  │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐     ┌─────────────────┐
   │ TraceReader     │────▶│ BlobData (11F)  │
   │ - CSV Parser    │     │ - offset_id     │
   │ - Streaming     │     │ - offset_score  │
   │ - Type Convert  │     │ - freq, size    │
   └─────────────────┘     │ - is_sequence   │
                           │ - recency, op   │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ AccessTracker   │
                           │ - VecDeque (10K)│
                           │ - Mmap overflow │
                           │ - Index (BTree) │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ FeatureExtractor│
                           │ - 10-dim vector │
                           │ - Normalized    │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ State Vector    │
                           │ [5+10] = 15-dim │
                           └────────┬────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │ ContextualBandit│    │ QNetwork        │    │ TierSelector    │
   │ - 20-dim output │    │ - 10-dim Q-vals │    │ - Tier index    │
   │ - Importance(1) │    │ - per action    │    │ - Capacity adj  │
   └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Combined Action │
                           │ (tier_idx,      │
                           │  op_type) → 0-9 │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ IOBufferEnv     │
                           │ - Apply action  │
                           │ - Compute reward│
                           │ - Next state    │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Transition      │
                           │ (s, a, r, s', d)│
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ ReplayBuffer    │
                           │ - Store (10K)   │
                           │ - Sample batch  │
                           └────────┬────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │ Training Step   │
                           │ - TD loss       │
                           │ - Update weights│
                           └─────────────────┘
```

### State Encoding

The state vector is a 15-dimensional continuous vector:

```rust
struct EnvironmentState {
    tier_sizes: [f32; 5],    // Normalized to capacity (0.0-1.0)
    features: [f32; 10],     // Access pattern features
}

// Total: 5 + 10 = 15 dimensions
```

#### Tier Size Encoding

Each tier's current fill percentage:
```rust
fn encode_tier_sizes(tiers: &[Tier]) -> [f32; 5] {
    tiers.iter()
        .map(|t| (t.current_size / t.config.capacity) as f32)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
```

#### Feature Encoding

The 10 feature dimensions are computed from access history:
1. `recency`: `(now - last_access) / max_time_window`
2. `frequency`: `access_count / max_observed_count`
3. `mean_interval`: `sum(intervals) / count / max_interval`
4. `std_interval`: `stddev(intervals) / max_interval`
5. `is_sequence`: `1.0 if sequential else 0.0`
6. `reuse_distance`: `position / window_size`
7. `last_access_type`: `0.0 for read, 1.0 for write`
8. `size`: `log2(blob_size) / 40.0` (normalized)
9. `next_access_pred`: `predicted_interval / max_interval`
10. `overwrite_amount`: `write_count / total_count`

### Action Space

The action space is discrete with 10 possible actions (5 tiers × 2 operations):

```rust
// Encoding: action_idx = tier_idx * 2 + op_type
enum IoOp {
    Read = 0,
    Write = 1,
}

fn decode_action(action_idx: usize) -> (usize, IoOp) {
    let tier_idx = action_idx / 2;
    let op_type = match action_idx % 2 {
        0 => IoOp::Read,
        1 => IoOp::Write,
        _ => unreachable!(),
    };
    (tier_idx, op_type)
}
```

### Reward Function

The reward is designed to minimize total latency:

```rust
fn compute_reward(actions: &[Action], tier_configs: &[TierConfig]) -> f32 {
    let total_latency: f64 = actions.iter()
        .zip(tier_configs)
        .map(|(action, config)| {
            config.base_latency_ms * action.size_factor()
        })
        .sum();
    -(total_latency as f32)
}
```

## Key Design Decisions

### 1. Action Space Design

The discrete action space with 10 actions (5 tiers × 2 operations) was chosen for several reasons:

- **Simplicity**: Compared to hierarchical action spaces, flat encoding is easier to learn
- **Efficiency**: Single-step action selection without sub-decisions
- **Interpretability**: Each action has clear semantics (tier + operation type)

The tier selection is capacity-aware through the TierSelector component:
```rust
fn select_tier(&self, importance: f32) -> usize {
    // Normalize importance to [0, 1]
    let normalized = importance.clamp(0.0, 1.0);
    
    // Find first tier with sufficient capacity
    for (idx, tier) in self.tiers.iter().enumerate() {
        let fill_pct = tier.current_size / tier.config.capacity;
        let threshold = 1.0 - normalized;  // Higher importance = lower threshold
        
        if fill_pct < threshold {
            return idx;
        }
    }
    
    // Fallback: least full tier
    self.tiers.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            a.current_size.partial_cmp(&b.current_size).unwrap()
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
```

### 2. State Space Design

The 15-dimensional state space was designed to capture:
- **System State**: Current tier utilization (for capacity awareness)
- **Blob Features**: Access patterns and predictions

This split ensures the agent is aware of both the current system state and the characteristics of the blob being accessed.

### 3. Memory Management Strategy

The system uses a tiered memory approach:

1. **Hot Window**: VecDeque with 10K recent accesses for fast feature extraction
2. **Mmap Overflow**: Memory-mapped file for traces exceeding hot window
3. **Eviction Policy**: When >100K records, drop oldest or least important

```rust
const HOT_WINDOW_SIZE: usize = 10_000;
const MAX_RECORDS: usize = 100_000;

impl AccessTracker {
    fn evict_if_needed(&mut self) {
        while self.mmap.len() > MAX_RECORDS {
            // Evict least important old records
            if let Some(oldest) = self.history.pop_front() {
                self.mmap.append(&oldest);
            }
        }
    }
}
```

### 4. Burn Backend Selection

The Burn framework supports multiple backends. We prioritize:

1. **Wgpu**: GPU acceleration for neural network operations
2. **Ndarray**: Pure Rust CPU implementation (no external dependencies)

Auto-detection is handled:
```rust
fn select_device() -> Device<Wgpu> {
    Device::default()  // Automatically detects GPU or falls back to CPU
}
```

### 5. Checkpoint Format

Model checkpoints use the Postcard serialization format for:

- **Compactness**: Binary format is smaller than JSON
- **CRC Protection**: Detection of corrupted checkpoints
- **Full State**: Includes architecture + weights + metadata

```rust
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    timestamp: chrono::DateTime<Utc>,
    model_architecture: ModelArchitecture,
    weights: ModelWeights,
    training_metadata: TrainingMetadata,
    #[serde(with = "crc32")]
    checksum: u32,
}
```

## File Structure

```
eris/
├── Cargo.toml
├── src/
│   ├── bin/
│   │   └── train.rs              # Training entry point
│   ├── env/
│   │   ├── io_buffer_env.rs      # Gymnasium environment
│   │   └── mod.rs
│   ├── features/
│   │   ├── tracker.rs            # Access history
│   │   ├── extractor.rs          # Feature extraction
│   │   ├── hotness.rs            # Hotness scoring
│   │   └── mod.rs
│   ├── models/
│   │   ├── dqn.rs                # QNetwork
│   │   ├── bandit.rs             # ContextualBandit
│   │   ├── combined.rs           # CombinedModel
│   │   └── mod.rs
│   ├── tier/
│   │   ├── tier.rs               # Individual tier
│   │   ├── selector.rs           # Tier selection logic
│   │   ├── manager.rs            # Multi-tier coordination
│   │   └── mod.rs
│   ├── trace/
│   │   ├── blob.rs               # BlobData struct
│   │   ├── reader.rs             # CSV parser
│   │   └── mod.rs
│   ├── training/
│   │   ├── replay_buffer.rs      # Experience replay
│   │   ├── trainer.rs            # CombinedAgent
│   │   ├── checkpoint.rs         # Model persistence
│   │   └── mod.rs
│   ├── config.rs                 # TOML configuration
│   ├── error.rs                  # Error types
│   └── lib.rs
├── config/
│   └── tiers.toml                # Tier definitions
├── recorder-csv/
│   └── NWChem-64_combined.csv    # Trace data
├── docs/
│   ├── ARCHITECTURE.md           # This file
│   ├── API.md                    # API documentation
│   ├── IMPLEMENTATION_PLAN.md    # Development roadmap
│   ├── PERFORMANCE.md            # Performance targets
│   ├── DATA_FORMATS.md           # Data specifications
│   └── DEVELOPMENT.md            # Developer guide
└── tests/
    └── integration.rs
```

## Dependencies

### Core Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `burn` | 0.14 | Neural network framework |
| `burn-ndarray` | 0.14 | CPU backend |
| `burn-wgpu` | 0.14 | GPU backend |
| `gymnasium` | 0.4 | RL environment interface |
| `tokio` | 1.0 | Async I/O |
| `postcard` | 1.0 | Serialization |
| `csv` | 1.0 | CSV parsing |
| `toml` | 0.8 | Configuration |
| `serde` | 1.0 | Serialization |
| `memmap2` | 0.9 | Memory-mapped files |

### Development Dependencies

| Crate | Purpose |
|-------|---------|
| `cargo-nextest` | Extended test runner |
| `criterion` | Benchmarks |
| `valgrind` | Memory profiling |
| `rustfmt` | Code formatting |
| `clippy` | Linting |

## Related Documentation

- [API Reference](API.md) - Detailed API documentation
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Phased development roadmap
- [Performance Tuning](PERFORMANCE.md) - Optimization strategies
- [Data Formats](DATA_FORMATS.md) - Input/output specifications
- [Developer Guide](DEVELOPMENT.md) - Getting started guide