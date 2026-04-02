# HeirGym Enhanced Models Data Formats

This document specifies all data formats used by the Eris project, including trace files, configuration files, model checkpoints, and internal data structures.

## Trace Data Format

### CSV Trace File

The primary input data is a CSV file containing access traces with 11 columns.

**File Location**: `recorder-csv/NWChem-64_combined.csv`

**Expected Row Count**: 18,083 rows

**Column Format**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `offset_id` | String | Unique blob identifier | `(nil)_143360_3_1_8_0_NWChem` |
| `offset_score` | f32 | Computed access score | `32.0` |
| `offset_access_frequency` | u32 | Total access count | `64` |
| `access_offset` | Option<f64> | Byte offset (may be empty) | `143360.0` |
| `access_size` | f64 | Operation size in bytes | `143360.0` |
| `offset_size` | f64 | Total blob size in bytes | `143360.0` |
| `is_sequence` | bool | Sequential access flag | `False` |
| `first_seen` | bool | First occurrence flag | `False` |
| `overwrite_amount` | f32 | Overwrite percentage (0-1) | `0.0` |
| `recency` | f32 | Time since last access (may be "inf") | `inf` |
| `io_op` | String | Operation type | `read` |

**Sample Rows**:

```csv
offset_id,offset_score,offset_access_frequency,access_offset,access_size,offset_size,is_sequence,first_seen,overwrite_amount,recency,io_op
(nil)_143360_3_1_8_0_NWChem,32.0,64,,143360.0,143360.0,False,False,0.0,inf,read
(nil)_143360_3_1_8_0_NWChem,32.0,64,,143360.0,143360.0,False,False,0.0,5000.0,write
(nil)_143360_3_1_8_0_NWChem,32.0,64,,143360.0,143360.0,False,False,0.0,10000.0,read
```

**Parsing Rules**:

1. `offset_id`: Direct string parse, may contain special characters like `_` and `:`
2. `offset_score`: Parse as f32, handle missing as 0.0
3. `access_offset`: Parse as Option<f64>, empty string becomes None
4. `io_op`: Case-insensitive comparison ("read" or "write")
5. `recency`: Parse as f32, "inf" becomes f32::INFINITY

### BlobData Struct

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct BlobData {
    pub offset_id: String,
    pub offset_score: f32,
    pub offset_access_frequency: u32,
    pub access_offset: Option<f64>,
    pub access_size: f64,
    pub offset_size: f64,
    pub is_sequence: bool,
    pub first_seen: bool,
    pub overwrite_amount: f32,
    pub recency: f32,
    pub io_op: String,
}

impl BlobData {
    /// Returns true if this is a write operation.
    pub fn is_write(&self) -> bool {
        self.io_op.to_lowercase() == "write"
    }
    
    /// Returns a normalized timestamp from recency.
    pub fn timestamp(&self) -> u64 {
        if self.recency.is_infinite() {
            0  // First access
        } else {
            (self.recency * 1000.0) as u64  // Convert to milliseconds
        }
    }
    
    /// Returns the blob size in bytes.
    pub fn size(&self) -> f64 {
        self.offset_size
    }
}
```

## Configuration Format

### Tier Configuration (TOML)

**File**: `config/tiers.toml`

**Format**:

```toml
[tier.memory]
name = "memory"
type = "memory"
capacity_gb = 1.0
read_latency_ms = 0.1
write_latency_ms = 0.1
eviction_enabled = true
eviction_threshold = 0.9

[tier.nvme]
name = "nvme"
type = "nvme"
capacity_gb = 10.0
read_latency_ms = 0.5
write_latency_ms = 1.0
eviction_enabled = true
eviction_threshold = 0.85

[tier.ssd]
name = "ssd"
type = "ssd"
capacity_gb = 100.0
read_latency_ms = 2.0
write_latency_ms = 5.0
eviction_enabled = true
eviction_threshold = 0.85

[tier.hdd]
name = "hdd"
type = "hdd"
capacity_gb = 1000.0
read_latency_ms = 10.0
write_latency_ms = 20.0
eviction_enabled = false
eviction_threshold = 0.95

[tier.tape]
name = "tape"
type = "tape"
capacity_gb = 10000.0
read_latency_ms = 5000.0
write_latency_ms = 1000.0
eviction_enabled = false
eviction_threshold = 1.0

[training]
batch_size = 32
replay_capacity = 10000
learning_rate = 0.0001
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
gamma = 0.99
```

**Tier Configuration Struct**:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    pub name: String,
    pub tier_type: String,
    #[serde(alias = "capacity_gb")]
    pub capacity: f64,
    #[serde(alias = "read_latency_ms")]
    pub read_latency_ms: f64,
    #[serde(alias = "write_latency_ms")]
    pub write_latency_ms: f64,
    #[serde(alias = "eviction_enabled")]
    pub eviction_enabled: bool,
    #[serde(alias = "eviction_threshold")]
    pub eviction_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub learning_rate: f32,
    pub epsilon_start: f32,
    pub epsilon_decay: f32,
    pub epsilon_min: f32,
    pub gamma: f32,
}
```

**Tier Type Values**:

| Type | Latency (read/write) | Capacity | Use Case |
|------|---------------------|----------|----------|
| `memory` | 0.1ms / 0.1ms | 1-10 GB | Hot data |
| `nvme` | 0.5ms / 1.0ms | 10-100 GB | Warm data |
| `ssd` | 2ms / 5ms | 100-1000 GB | Cold data |
| `hdd` | 10ms / 20ms | 1-10 TB | Archive |
| `tape` | 5000ms / 1000ms | 10+ TB | Deep archive |

## State and Action Formats

### Environment State (15-dim)

```rust
/// Complete environment state vector (15 dimensions)
/// 
/// Structure:
/// - indices 0-4: Tier fill percentages (0.0 = empty, 1.0 = full)
/// - indices 5-14: Blob access features (10 dimensions)
struct EnvironmentState {
    tier_fill_pct: [f32; 5],  // Current utilization of each tier
    features: [f32; 10],       // Access pattern features
}

impl EnvironmentState {
    const DIM: usize = 15;
    
    /// Creates state from tier sizes and features
    fn from_components(tier_sizes: &[f32], features: &[f32]) -> Self {
        let mut tier_fill_pct = [0.0; 5];
        tier_fill_pct[..tier_sizes.len()].copy_from_slice(&tier_sizes[..5.min(tier_sizes.len())]);
        
        let mut feature_arr = [0.0; 10];
        feature_arr[..features.len()].copy_from_slice(&features[..10.min(features.len())]);
        
        Self { tier_fill_pct, features: feature_arr }
    }
    
    /// Converts to Vec<f32> for neural network input
    fn to_vec(&self) -> Vec<f32> {
        self.tier_fill_pct.iter()
            .chain(self.features.iter())
            .copied()
            .collect()
    }
}
```

### Action Encoding

Actions are encoded as integers 0-9:

```rust
/// Action encoding: tier_idx * 2 + op_type
/// 
/// Where:
/// - tier_idx: 0-4 (5 tiers)
/// - op_type: 0 = read, 1 = write
struct Action {
    tier_idx: usize,
    op_type: IoOp,
}

impl Action {
    /// Encodes action as single integer
    fn encode(&self) -> usize {
        self.tier_idx * 2 + self.op_type as usize
    }
    
    /// Decodes action from integer
    fn decode(action_idx: usize) -> Self {
        Self {
            tier_idx: action_idx / 2,
            op_type: match action_idx % 2 {
                0 => IoOp::Read,
                1 => IoOp::Write,
                _ => unreachable!(),
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum IoOp {
    Read = 0,
    Write = 1,
}

const ACTION_READ_TIER_0: usize = 0;
const ACTION_WRITE_TIER_0: usize = 1;
const ACTION_READ_TIER_1: usize = 2;
const ACTION_WRITE_TIER_1: usize = 3;
const ACTION_READ_TIER_2: usize = 4;
const ACTION_WRITE_TIER_2: usize = 5;
const ACTION_READ_TIER_3: usize = 6;
const ACTION_WRITE_TIER_3: usize = 7;
const ACTION_READ_TIER_4: usize = 8;
const ACTION_WRITE_TIER_4: usize = 9;
```

### Feature Vector (10-dim)

```rust
/// 10-dimensional feature vector for a blob
struct BlobFeatures {
    /// Time since last access (normalized 0-1, higher = older)
    recency: f32,
    
    /// Access count relative to max observed (normalized 0-1)
    frequency: f32,
    
    /// Mean time between accesses in ms (normalized 0-1)
    mean_interval: f32,
    
    /// Std dev of access intervals in ms (normalized 0-1)
    std_interval: f32,
    
    /// 1.0 if sequential pattern, 0.0 otherwise
    is_sequential: f32,
    
    /// Position in history window (normalized 0-1)
    reuse_distance: f32,
    
    /// 0.0 for read, 1.0 for write
    last_access_type: f32,
    
    /// Blob size (normalized by max size)
    size: f32,
    
    /// Predicted next access time (normalized 0-1)
    next_access_pred: f32,
    
    /// Write frequency ratio (0-1)
    overwrite_amount: f32,
}

impl BlobFeatures {
    const DIM: usize = 10;
    
    fn to_vec(&self) -> [f32; 10] {
        [
            self.recency,
            self.frequency,
            self.mean_interval,
            self.std_interval,
            self.is_sequential,
            self.reuse_distance,
            self.last_access_type,
            self.size,
            self.next_access_pred,
            self.overwrite_amount,
        ]
    }
}
```

## Model Checkpoint Format

### Checkpoint File Structure

Checkpoints use Postcard serialization for compact binary storage.

```rust
/// Complete model checkpoint
#[derive(Serialize, Deserialize)]
struct Checkpoint {
    /// Timestamp of checkpoint creation
    timestamp: chrono::DateTime<Utc>,
    
    /// Model architecture version
    version: String,
    
    /// Contextual bandit weights
    bandit_weights: BanditWeights,
    
    /// Q-network weights
    qnetwork_weights: QNetworkWeights,
    
    /// Tier configurations at save time
    tier_configs: Vec<TierConfig>,
    
    /// Training metadata
    training_metadata: TrainingMetadata,
    
    /// CRC32 checksum for corruption detection
    #[serde(with = "crc32_serializer")]
    checksum: u32,
}

/// Bandit network weights
#[derive(Serialize, Deserialize)]
struct BanditWeights {
    /// FC1: 15 -> 64
    fc1_weight: Vec<f32>,
    fc1_bias: Vec<f32>,
    
    /// FC2: 64 -> 128
    fc2_weight: Vec<f32>,
    fc2_bias: Vec<f32>,
    
    /// FC3: 128 -> 31
    fc3_weight: Vec<f32>,
    fc3_bias: Vec<f32>,
}

/// Q-network weights
#[derive(Serialize, Deserialize)]
struct QNetworkWeights {
    /// FC1: 20 -> 128
    fc1_weight: Vec<f32>,
    fc1_bias: Vec<f32>,
    
    /// FC2: 128 -> 128
    fc2_weight: Vec<f32>,
    fc2_bias: Vec<f32>,
    
    /// FC3: 128 -> 10
    fc3_weight: Vec<f32>,
    fc3_bias: Vec<f32>,
}

/// Training metadata
#[derive(Serialize, Deserialize)]
struct TrainingMetadata {
    /// Total training episodes completed
    episodes: usize,
    
    /// Total steps taken
    steps: usize,
    
    /// Current epsilon value
    epsilon: f32,
    
    /// Best reward achieved
    best_reward: f32,
    
    /// Average reward over last 100 episodes
    avg_reward_100: f32,
    
    /// Total training time in seconds
    training_time_sec: u64,
    
    /// Learning rate used
    learning_rate: f32,
}

/// CRC32 serializer helper
mod crc32_serializer {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};
    
    pub fn serialize<S>(value: &u32, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u32(*value)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<u32, D::Error>
    where
        D: Deserializer<'de>,
        u32: Deserialize<'de>,
    {
        let value = u32::deserialize(deserializer)?;
        Ok(value)
    }
}
```

### Checkpoint Binary Layout

```
┌──────────────────────────────────────────────────────────────┐
│                      Checkpoint Header                        │
├──────────────────────────────────────────────────────────────┤
│ magic_number: 0xDEADBEEF (4 bytes)                           │
│ version: u32 (4 bytes)                                       │
│ timestamp: i64 (8 bytes)                                     │
├──────────────────────────────────────────────────────────────┤
│                 Training Metadata (variable)                  │
├──────────────────────────────────────────────────────────────┤
│ episodes: u32                                                │
│ steps: u64                                                   │
│ epsilon: f32                                                 │
│ best_reward: f32                                             │
│ avg_reward_100: f32                                          │
│ training_time_sec: u64                                       │
│ learning_rate: f32                                           │
├──────────────────────────────────────────────────────────────┤
│                 Tier Configs (variable)                       │
├──────────────────────────────────────────────────────────────┤
│ num_tiers: u32                                               │
│ [TierConfig for each tier]                                   │
├──────────────────────────────────────────────────────────────┤
│              Bandit Weights (variable)                       │
├──────────────────────────────────────────────────────────────┤
│ fc1_weight: [f32; 15*64]                                     │
│ fc1_bias: [f32; 64]                                          │
│ fc2_weight: [f32; 64*128]                                    │
│ fc2_bias: [f32; 128]                                         │
│ fc3_weight: [f32; 128*31]                                    │
│ fc3_bias: [f32; 31]                                          │
├──────────────────────────────────────────────────────────────┤
│              QNetwork Weights (variable)                      │
├──────────────────────────────────────────────────────────────┤
│ fc1_weight: [f32; 20*128]                                    │
│ fc1_bias: [f32; 128]                                         │
│ fc2_weight: [f32; 128*128]                                   │
│ fc2_bias: [f32; 128]                                         │
│ fc3_weight: [f32; 128*10]                                    │
│ fc3_bias: [f32; 10]                                          │
├──────────────────────────────────────────────────────────────┤
│                    Footer                                     │
├──────────────────────────────────────────────────────────────┤
│ checksum: u32 (CRC32 of all preceding bytes)                 │
└──────────────────────────────────────────────────────────────┘
```

**Typical Checkpoint Size**:

- Bandit weights: ~48KB
- Q-network weights: ~130KB
- Metadata: ~64 bytes
- Total: ~180KB (uncompressed)

### Saving Checkpoints

```rust
impl CombinedAgent {
    pub fn save(&self, path: &Path) -> Result<(), EnvError> {
        let checkpoint = Checkpoint {
            timestamp: chrono::Utc::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            bandit_weights: self.extract_bandit_weights(),
            qnetwork_weights: self.extract_qnetwork_weights(),
            tier_configs: self.tier_configs.clone(),
            training_metadata: self.get_training_metadata(),
            checksum: 0,  // Will be computed
        };
        
        // Serialize to bytes
        let data = postcard::to_allocvec(&checkpoint)
            .map_err(|e| EnvError::Serialization(e.to_string()))?;
        
        // Compute checksum
        let checksum = crc32::crc32(&data);
        
        // Re-serialize with checksum
        let mut data_with_checksum = data;
        let checksum_bytes = checksum.to_le_bytes();
        data_with_checksum.extend_from_slice(&checksum_bytes);
        
        // Write to file
        std::fs::write(path, &data_with_checksum)
            .map_err(|e| EnvError::Io(e.to_string()))?;
        
        Ok(())
    }
}
```

## Replay Buffer Format

### Transition Struct

```rust
/// A single experience in the replay buffer
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Transition {
    /// Current state (15-dimensional)
    state: Vec<f32>,
    
    /// Action taken (0-9)
    action: usize,
    
    /// Reward received
    reward: f32,
    
    /// Next state (15-dimensional)
    next_state: Vec<f32>,
    
    /// Whether episode terminated
    done: bool,
}

impl Transition {
    fn new(
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
    ) -> Self {
        Self {
            state,
            action,
            reward,
            next_state,
            done,
        }
    }
    
    /// Serialized size in bytes
    fn serialized_size(&self) -> usize {
        // state: 15 * 4 = 60
        // action: 4
        // reward: 4
        // next_state: 60
        // done: 1
        // Total: ~129 bytes (plus serialization overhead)
        129
    }
}
```

### Replay Buffer Layout

```rust
struct ReplayBuffer {
    /// Circular buffer of transitions
    buffer: VecDeque<Transition>,
    
    /// Maximum capacity
    capacity: usize,
    
    /// Current write position
    write_ptr: usize,
    
    /// Total transitions added
    total_added: usize,
}

const DEFAULT_CAPACITY: usize = 10_000;
const MAX_CAPACITY: usize = 100_000;
```

## Access Record Format

### AccessRecord Struct

```rust
/// A single access event in history
#[derive(Debug, Clone)]
struct AccessRecord {
    /// Unique blob identifier
    blob_id: String,
    
    /// Timestamp in milliseconds since epoch
    timestamp: u64,
    
    /// Type of I/O operation
    access_type: IoOp,
    
    /// Size of the access in bytes
    size: f64,
}

impl AccessRecord {
    fn new(blob_id: String, timestamp: u64, access_type: &str, size: f64) -> Self {
        Self {
            blob_id,
            timestamp,
            access_type: match access_type.to_lowercase().as_str() {
                "write" => IoOp::Write,
                _ => IoOp::Read,
            },
            size,
        }
    }
}
```

## Error Codes

All errors are defined in `src/error.rs`:

```rust
#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Configuration error: {0}")]
    Config(#[from] toml::de::Error),
    
    #[error("CSV parsing error: {0}")]
    Csv(#[from] csv::Error),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] postcard::Error),
    
    #[error("Tier '{0}' is full")]
    TierFull(String),
    
    #[error("Insufficient space in tier '{tier}': needed {needed}, available {available}")]
    InsufficientSpace {
        tier: String,
        needed: f64,
        available: f64,
    },
    
    #[error("Invalid capacity for tier '{0}'")]
    InvalidCapacity(String),
    
    #[error("Trace parsing error: {0}")]
    TraceParse(String),
    
    #[error("Blob '{0}' not found")]
    BlobNotFound(String),
    
    #[error("Invalid action: {0}")]
    InvalidAction(usize),
    
    #[error("Model error: {0}")]
    Model(String),
}
```

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System design overview
- [API Reference](API.md) - Detailed API documentation
- [Performance](PERFORMANCE.md) - Performance targets
- [Developer Guide](DEVELOPMENT.md) - Getting started guide