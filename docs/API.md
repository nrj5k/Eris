# HeirGym Enhanced Models API Reference

This document provides comprehensive API documentation for all public interfaces in the Eris Rust project.

## Table of Contents

1. [Environment API](#environment-api)
2. [Storage API](#storage-api)
3. [Features API](#features-api)
4. [Models API](#models-api)
5. [Training API](#training-api)
6. [Configuration API](#configuration-api)
7. [Error Handling](#error-handling)

---

## Environment API

### IOBufferEnv

The main RL environment that implements the gymnasium interface.

```rust
use gymnasium::core::{Env, Reward, Termination};

pub struct IOBufferEnv {
    tiers: Vec<Tier>,
    access_tracker: AccessTracker,
    feature_extractor: FeatureExtractor,
    current_blob: Option<BlobData>,
    step_count: usize,
    max_steps: usize,
    config: EnvConfig,
}

impl IOBufferEnv {
    /// Creates a new environment from configuration and trace paths.
    pub fn new(config_path: &Path, trace_path: &Path, max_steps: usize) -> Result<Self, EnvError> {
        let config = EnvConfig::load(config_path)?;
        let trace_reader = TraceReader::new(trace_path)?;
        let tiers = TierManager::new(&config.tiers)?;
        
        Ok(Self {
            tiers,
            access_tracker: AccessTracker::new(10_000),
            feature_extractor: FeatureExtractor::new(),
            current_blob: None,
            step_count: 0,
            max_steps,
            config,
        })
    }

    /// Returns the current observation (15-dimensional state vector).
    pub fn get_state(&self) -> Vec<f32> {
        let tier_sizes: Vec<f32> = self.tiers
            .iter()
            .map(|t| (t.current_size / t.config.capacity) as f32)
            .collect();
        
        let features = self.current_blob
            .as_ref()
            .and_then(|blob| self.feature_extractor.extract(&blob, &self.access_tracker))
            .unwrap_or_else(|| vec![0.0; 10]);
        
        [tier_sizes, features].concat()
    }

    /// Returns the number of accesses per tier.
    pub fn get_tier_accesses(&self) -> Vec<u64> {
        self.tiers.iter().map(|t| t.access_count).collect()
    }

    /// Resets the environment to initial state.
    pub fn reset(&mut self) {
        self.tiers.iter_mut().for_each(Tier::clear);
        self.access_tracker.clear();
        self.step_count = 0;
        self.current_blob = None;
    }
}
```

#### gymnasium::core::Env Implementation

```rust
use gymnasium::core::{ActType, ObsType, Info};

impl Env for IOBufferEnv {
    /// Action type: Discrete action space 0-9
    type Action = usize;
    
    /// Observation type: 15-dimensional continuous vector
    type Observation = Vec<f32>;
    
    /// Additional info returned with reset/step
    type Info = EnvInfo;
    
    /// Executes one environment step.
    fn step(
        &mut self,
        action: Self::Action,
    ) -> ActReward<Self::Observation, Termination, Self::Info> {
        // Decode action: (tier_idx, op_type)
        let tier_idx = action / 2;
        let op_type = if action % 2 == 0 { IoOp::Read } else { IoOp::Write };
        
        // Validate action
        if tier_idx >= self.tiers.len() {
            return (
                self.get_state(),
                -1.0,  // Penalty for invalid action
                true,  // Truncate episode
                false, // Not done
                EnvInfo::error("Invalid tier index"),
            );
        }
        
        let blob = match &self.current_blob {
            Some(b) => b.clone(),
            None => {
                return (
                    self.get_state(),
                    0.0,
                    false,
                    false,
                    EnvInfo::empty(),
                );
            }
        };
        
        // Execute action
        let latency = self.execute_operation(tier_idx, &blob, op_type);
        let reward = -(latency as f32);
        
        // Get next blob
        self.current_blob = self.next_trace_record();
        self.step_count += 1;
        
        // Check termination conditions
        let done = self.step_count >= self.max_steps;
        let truncated = !done && self.access_tracker.len() > 100_000;
        
        let info = EnvInfo {
            tier_idx,
            op_type,
            latency,
            blob_id: blob.offset_id.clone(),
        };
        
        (self.get_state(), reward, done, truncated, info)
    }

    /// Resets the environment.
    fn reset(
        &mut self,
        seed: Option<u64>,
        _return_info: bool,
        _options: Option<ResetOptions>,
    ) -> (ObsType<Self::Observation>, Info<Self::Info>) {
        if let Some(seed) = seed {
            // Initialize random number generator
            self.rng = StdRng::seed_from_u64(seed);
        }
        
        self.reset();
        self.current_blob = self.next_trace_record();
        
        (self.get_state(), EnvInfo::empty())
    }

    /// Returns the action space.
    fn action_space(&self) -> &Space<Self::Action> {
        &Discrete::new(10)  // 5 tiers × 2 operations
    }

    /// Returns the observation space.
    fn observation_space(&self) -> &Space<Self::Observation> {
        let bounds = vec![
            0.0..=1.0_f32,  // tier_0_size
            0.0..=1.0_f32,  // tier_1_size
            0.0..=1.0_f32,  // tier_2_size
            0.0..=1.0_f32,  // tier_3_size
            0.0..=1.0_f32,  // tier_4_size
            0.0..=1.0_f32,  // recency
            0.0..=1.0_f32,  // frequency
            0.0..=1.0_f32,  // mean_interval
            0.0..=1.0_f32,  // std_interval
            0.0..=1.0_f32,  // is_sequence
            0.0..=1.0_f32,  // reuse_distance
            0.0..=1.0_f32,  // last_access_type
            0.0..=1.0_f32,  // size
            0.0..=1.0_f32,  // next_access_pred
            0.0..=1.0_f32,  // overwrite_amount
        ];
        &Box::new(Bound::new(bounds))
    }
}

impl IOBufferEnv {
    fn execute_operation(
        &mut self,
        tier_idx: usize,
        blob: &BlobData,
        op_type: IoOp,
    ) -> f64 {
        let tier = &mut self.tiers[tier_idx];
        let latency = match op_type {
            IoOp::Read => {
                let _ = tier.read(&blob.offset_id);
                tier.config.read_latency_ms
            }
            IoOp::Write => {
                let _ = tier.write(&blob.offset_id, blob.offset_size);
                tier.config.write_latency_ms
            }
        };
        
        tier.access_count += 1;
        
        // Record access for feature tracking
        self.access_tracker.record(AccessRecord {
            blob_id: blob.offset_id.clone(),
            timestamp: blob.timestamp,
            access_type: op_type,
            size: blob.offset_size,
        });
        
        latency
    }
}
```

### EnvInfo

Additional information returned by the environment.

```rust
#[derive(Debug, Clone)]
pub struct EnvInfo {
    pub tier_idx: usize,
    pub op_type: IoOp,
    pub latency: f64,
    pub blob_id: String,
    pub error: Option<String>,
}

impl EnvInfo {
    pub fn empty() -> Self {
        Self {
            tier_idx: 0,
            op_type: IoOp::Read,
            latency: 0.0,
            blob_id: String::new(),
            error: None,
        }
    }

    pub fn error(msg: &str) -> Self {
        Self {
            tier_idx: 0,
            op_type: IoOp::Read,
            latency: 0.0,
            blob_id: String::new(),
            error: Some(msg.to_string()),
        }
    }
}
```

---

## Storage API

### Tier

A single storage tier with HashMap-based storage.

```rust
pub struct Tier {
    /// Tier configuration
    pub config: TierConfig,
    /// Data storage: blob_id -> size
    storage: HashMap<String, f64>,
    /// Current utilization
    pub current_size: f64,
    /// Total access count
    pub access_count: u64,
    /// Random number generator for probabilistic eviction
    rng: StdRng,
}

impl Tier {
    /// Creates a new tier with the given configuration.
    pub fn new(config: TierConfig) -> Self {
        Self {
            config,
            storage: HashMap::new(),
            current_size: 0.0,
            access_count: 0,
            rng: StdRng::from_entropy(),
        }
    }

    /// Writes data to the tier.
    pub fn write(&mut self, blob_id: &str, size: f64) -> Result<(), EnvError> {
        let size = size.abs();  // Ensure non-negative
        
        // Check capacity
        if self.current_size + size > self.config.capacity {
            // Try to evict if enabled
            if self.config.eviction_enabled {
                self.evict_to_free(size)?;
            } else {
                return Err(EnvError::TierFull(self.config.name.clone()));
            }
        }
        
        // Update storage
        self.current_size += size;
        self.storage.insert(blob_id.to_string(), size);
        
        Ok(())
    }

    /// Reads data from the tier.
    pub fn read(&self, blob_id: &str) -> Option<f64> {
        self.storage.get(blob_id).copied()
    }

    /// Removes data from the tier.
    pub fn remove(&mut self, blob_id: &str) -> bool {
        if let Some(size) = self.storage.remove(blob_id) {
            self.current_size -= size;
            true
        } else {
            false
        }
    }

    /// Clears all data in the tier.
    pub fn clear(&mut self) {
        self.storage.clear();
        self.current_size = 0.0;
        self.access_count = 0;
    }

    /// Returns available capacity.
    pub fn available_capacity(&self) -> f64 {
        self.config.capacity - self.current_size
    }

    /// Returns current fill percentage.
    pub fn fill_percentage(&self) -> f32 {
        (self.current_size / self.config.capacity) as f32
    }

    fn evict_to_free(&mut self, needed_size: f64) -> Result<(), EnvError> {
        let mut freed = 0.0;
        
        // Collect all items with their "coldness" score
        let mut items: Vec<(&String, &f64, f32)> = self.storage.iter()
            .map(|(id, &size)| {
                let coldness = self.calculate_coldness(id);
                (id, size, coldness)
            })
            .collect();
        
        // Sort by coldness (coldest first)
        items.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
        
        // Evict until we have enough space
        for (id, size, _) in items {
            self.storage.remove(id);
            self.current_size -= *size;
            freed += *size;
            
            if freed >= needed_size {
                return Ok(());
            }
        }
        
        Err(EnvError::InsufficientSpace {
            tier: self.config.name.clone(),
            needed: needed_size,
            available: freed,
        })
    }

    fn calculate_coldness(&self, blob_id: &str) -> f32 {
        // Simple coldness based on last access time
        // In practice, this would use the access tracker
        0.5  // Placeholder
    }
}
```

### TierConfig

Configuration for a storage tier.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TierConfig {
    pub name: String,
    pub tier_type: TierType,
    pub capacity: f64,
    pub read_latency_ms: f64,
    pub write_latency_ms: f64,
    pub eviction_enabled: bool,
    pub eviction_threshold: f32,  // Fill percentage to trigger eviction
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TierType {
    Memory,  // DRAM
    Nvme,    // NVMe SSD
    Ssd,     // SATA SSD
    Hdd,     // Hard Disk Drive
    Tape,    // Tape storage
}
```

### TierSelector

Capacity-weighted tier selection based on importance score.

```rust
pub struct TierSelector {
    tiers: Vec<TierConfig>,
    capacity_weights: Vec<f32>,
}

impl TierSelector {
    /// Creates a new selector with tier configurations.
    pub fn new(tiers: Vec<TierConfig>) -> Self {
        let total_capacity: f64 = tiers.iter().map(|t| t.capacity).sum();
        let capacity_weights: Vec<f32> = tiers
            .iter()
            .map(|t| (t.capacity / total_capacity) as f32)
            .collect();
        
        Self { tiers, capacity_weights }
    }

    /// Selects a tier based on importance score.
    ///
    /// Higher importance scores map to faster tiers, respecting capacity constraints.
    pub fn select_tier(&self, importance: f32) -> usize {
        // Normalize importance to [0, 1]
        let normalized = importance.clamp(0.0, 1.0);
        
        // Compute adjusted score with capacity weighting
        let adjusted_scores: Vec<f32> = self.tiers.iter()
            .enumerate()
            .map(|(idx, tier)| {
                let tier_speed = match tier.tier_type {
                    TierType::Memory => 1.0,
                    TierType::Nvme => 0.8,
                    TierType::Ssd => 0.6,
                    TierType::Hdd => 0.3,
                    TierType::Tape => 0.1,
                };
                
                // Balance importance with capacity availability
                tier_speed * (1.0 - self.capacity_weights[idx]) + normalized * self.capacity_weights[idx]
            })
            .collect();
        
        // Select tier with highest adjusted score
        adjusted_scores
            .into_iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Returns the number of tiers.
    pub fn num_tiers(&self) -> usize {
        self.tiers.len()
    }

    /// Returns tier configuration by index.
    pub fn get_tier_config(&self, idx: usize) -> Option<&TierConfig> {
        self.tiers.get(idx)
    }
}
```

### TierManager

Coordinates multiple tiers and handles demotion logic.

```rust
pub struct TierManager {
    tiers: Vec<Tier>,
    selector: TierSelector,
    demotion_threshold: f32,
}

impl TierManager {
    /// Creates a new manager from tier configurations.
    pub fn new(configs: &[TierConfig]) -> Result<Self, EnvError> {
        let mut tiers = Vec::with_capacity(configs.len());
        
        for config in configs {
            if config.capacity <= 0.0 {
                return Err(EnvError::InvalidCapacity(config.name.clone()));
            }
            tiers.push(Tier::new(config.clone()));
        }
        
        let selector = TierSelector::new(configs.to_vec());
        
        Ok(Self {
            tiers,
            selector,
            demotion_threshold: 0.9,  // 90% fill
        })
    }

    /// Gets current state for the environment.
    pub fn get_state(&self) -> Vec<f32> {
        self.tiers
            .iter()
            .map(|t| t.fill_percentage())
            .collect()
    }

    /// Demotes data from faster to slower tiers.
    pub fn demote(&mut self) -> Result<usize, EnvError> {
        let mut demoted_count = 0;
        
        for tier_idx in 0..self.tiers.len().saturating_sub(1) {
            let fill_pct = self.tiers[tier_idx].fill_percentage();
            
            if fill_pct > self.demotion_threshold {
                // Find candidates to demote
                let candidates: Vec<String> = self.tiers[tier_idx]
                    .storage
                    .keys()
                    .filter(|id| self.calculate_coldness(id) > 0.5)
                    .cloned()
                    .collect();
                
                for blob_id in candidates {
                    if self.tiers[tier_idx].remove(&blob_id) {
                        let size = self.tiers[tier_idx + 1].available_capacity();
                        // Write to next tier
                        self.tiers[tier_idx + 1].write(&blob_id, size)?;
                        demoted_count += 1;
                    }
                }
            }
        }
        
        Ok(demoted_count)
    }

    fn calculate_coldness(&self, blob_id: &str) -> f32 {
        // Use access tracker to determine coldness
        0.5  // Placeholder
    }
}
```

---

## Features API

### BlobFeatures

The 10-dimensional feature vector for a blob.

```rust
#[derive(Debug, Clone, PartialDefault)]
pub struct BlobFeatures {
    /// Time since last access (normalized 0-1, higher = older)
    pub recency: f32,
    
    /// Access count / max_observed_count (normalized 0-1)
    pub frequency: f32,
    
    /// Mean time between accesses (ms, normalized 0-1)
    pub mean_interval: f32,
    
    /// Std dev of intervals (ms, normalized 0-1)
    pub std_interval: f32,
    
    /// 1.0 if sequential pattern detected, 0.0 otherwise
    pub is_sequential: f32,
    
    /// Position of last access in history window (normalized 0-1)
    pub reuse_distance: f32,
    
    /// 0.0 for read, 1.0 for write
    pub last_access_type: f32,
    
    /// Blob size (normalized by max size)
    pub size: f32,
    
    /// Predicted time until next access (normalized 0-1)
    pub next_access_pred: f32,
    
    /// Write frequency / total accesses (0-1)
    pub overwrite_amount: f32,
}

impl BlobFeatures {
    /// Converts features to a Vec<f32> for the neural network.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
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

    /// Returns the dimension of the feature vector.
    pub const fn dim() -> usize {
        10
    }
}
```

### AccessRecord

A single access event in the history.

```rust
#[derive(Debug, Clone)]
pub struct AccessRecord {
    /// Unique blob identifier
    pub blob_id: String,
    
    /// Timestamp in milliseconds since epoch
    pub timestamp: u64,
    
    /// Type of I/O operation
    pub access_type: IoOp,
    
    /// Size of the access in bytes
    pub size: f64,
}

impl AccessRecord {
    /// Creates a new access record.
    pub fn new(blob_id: String, timestamp: u64, access_type: IoOp, size: f64) -> Self {
        Self { blob_id, timestamp, access_type, size }
    }
}
```

### AccessTracker

Maintains access history and provides feature extraction.

```rust
pub struct AccessTracker {
    /// Hot window: recent accesses in memory
    memory_window: VecDeque<AccessRecord>,
    
    /// Cold storage: memory-mapped file for overflow
    mmap: Option<MmapMut>,
    
    /// Index for fast lookup by blob_id
    index: BTreeMap<String, VecDequeIndexEntry>,
    
    /// Maximum memory window size
    window_size: usize,
    
    /// Total records seen
    total_records: u64,
}

struct VecDequeIndexEntry {
    positions: Vec<usize>,  // Positions in memory_window
    last_updated: u64,
}

impl AccessTracker {
    /// Creates a new tracker with the specified window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            memory_window: VecDeque::with_capacity(window_size),
            mmap: None,
            index: BTreeMap::new(),
            window_size,
            total_records: 0,
        }
    }

    /// Records a new access event.
    pub fn record(&mut self, access: AccessRecord) {
        // Add to memory window
        self.memory_window.push_back(access.clone());
        
        // Update index
        let entry = self.index
            .entry(access.blob_id.clone())
            .or_insert_with(|| VecDequeIndexEntry {
                positions: Vec::new(),
                last_updated: self.total_records,
            });
        
        entry.positions.push(self.memory_window.len() - 1);
        entry.last_updated = self.total_records;
        
        self.total_records += 1;
        
        // Evict old records if window is full
        if self.memory_window.len() > self.window_size {
            if let Some(oldest) = self.memory_window.pop_front() {
                // Clean up index entries
                if let Some(entry) = self.index.get_mut(&oldest.blob_id) {
                    entry.positions.retain(|&p| p != 0);
                    entry.positions.iter_mut().for_each(|p| *p -= 1);
                }
            }
        }
    }

    /// Gets features for a specific blob.
    pub fn get_features(&self, blob_id: &str) -> Option<BlobFeatures> {
        let records = self.get_access_records(blob_id)?;
        Some(FeatureExtractor::compute_features(records))
    }

    /// Gets the hotness score for a blob.
    pub fn get_hotness(&self, blob_id: &str) -> Option<f32> {
        let features = self.get_features(blob_id)?;
        Some(hotness_score(
            features.recency,
            features.is_sequential,
            features.overwrite_amount,
            features.recency,  // Using recency as both metrics for now
            &HotnessConfig::default(),
        ))
    }

    fn get_access_records(&self, blob_id: &str) -> Option<Vec<&AccessRecord>> {
        let positions = self.index.get(blob_id)?.positions.clone();
        Some(positions
            .iter()
            .filter_map(|&pos| self.memory_window.get(pos))
            .collect())
    }

    /// Clears all tracked data.
    pub fn clear(&mut self) {
        self.memory_window.clear();
        self.index.clear();
        self.total_records = 0;
    }

    /// Returns the number of records in the hot window.
    pub fn len(&self) -> usize {
        self.memory_window.len()
    }

    /// Returns true if the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.memory_window.is_empty()
    }
}
```

### HotnessConfig

Configuration for hotness score calculation.

```rust
#[derive(Debug, Clone)]
pub struct HotnessConfig {
    /// Weight for recency score
    pub recency_weight: f32,
    /// Weight for sequential access bonus
    pub sequence_weight: f32,
    /// Weight for overwrite frequency
    pub overwrite_weight: f32,
    /// Penalty for old accesses
    pub age_penalty: f32,
    /// Default values
    pub fn default() -> Self {
        Self {
            recency_weight: 0.4,
            sequence_weight: 0.2,
            overwrite_weight: 0.3,
            age_penalty: 0.1,
        }
    }
}

/// Computes the hotness score for a blob.
///
/// The hotness score is a unified metric combining multiple factors:
/// - Recent accesses (higher = hotter)
/// - Sequential access patterns (bonus for sequential)
/// - Write frequency (higher = potentially hotter)
/// - Age penalty (older = cooler)
pub fn hotness_score(
    recency: f32,
    is_sequence: f32,
    overwrite_amount: f32,
    age: f32,
    config: &HotnessConfig,
) -> f32 {
    recency * config.recency_weight
        + is_sequence * config.sequence_weight
        + overwrite_amount * config.overwrite_weight
        - age * config.age_penalty
}
```

### FeatureExtractor

Extracts features from access history.

```rust
pub struct FeatureExtractor {
    max_observed_count: u32,
    max_interval: u64,
    max_blob_size: f64,
}

impl FeatureExtractor {
    /// Creates a new feature extractor.
    pub fn new() -> Self {
        Self {
            max_observed_count: 1,
            max_interval: 1,
            max_blob_size: 1.0,
        }
    }

    /// Extracts features for a blob from the access tracker.
    pub fn extract(
        &mut self,
        blob: &BlobData,
        tracker: &AccessTracker,
    ) -> Option<Vec<f32>> {
        let records = tracker.get_access_records(&blob.offset_id)?;
        let features = Self::compute_features(records);
        
        // Update max values for normalization
        self.max_observed_count = self.max_observed_count.max(records.len() as u32);
        
        Some(features.to_vec())
    }

    /// Computes features from a list of access records.
    pub fn compute_features(records: Vec<&AccessRecord>) -> BlobFeatures {
        if records.is_empty() {
            return BlobFeatures::default();
        }
        
        let now = records.last().unwrap().timestamp;
        
        // Sort by timestamp
        let mut sorted: Vec<_> = records.iter().collect();
        sorted.sort_by_key(|r| r.timestamp);
        
        let timestamps: Vec<u64> = sorted.iter().map(|r| r.timestamp).collect();
        let access_types: Vec<IoOp> = sorted.iter().map(|r| r.access_type).collect();
        let sizes: Vec<f64> = sorted.iter().map(|r| r.size).collect();
        
        let recency = if let Some(last) = timestamps.last() {
            ((now - last) as f32 / 1000.0)  // Convert to seconds
        } else {
            1.0
        }.clamp(0.0, 1.0);
        
        let frequency = (sorted.len() as f32 / 1000.0).clamp(0.0, 1.0);
        
        // Compute intervals
        let intervals: Vec<u64> = timestamps
            .windows(2)
            .map(|w| w[1].saturating_sub(w[0]))
            .collect();
        
        let mean_interval = if !intervals.is_empty() {
            (intervals.iter().sum::<u64>() / intervals.len() as u64) as f32 / 1000.0
        } else {
            0.0
        }.clamp(0.0, 1.0);
        
        let std_interval = if intervals.len() > 1 {
            let mean = intervals.iter().sum::<u64>() as f64 / intervals.len() as f64;
            let variance: f64 = intervals.iter()
                .map(|i| {
                    let d = *i as f64 - mean;
                    d * d
                })
                .sum::<f64>() / intervals.len() as f64;
            variance.sqrt() as f32 / 1000.0
        } else {
            0.0
        }.clamp(0.0, 1.0);
        
        // Detect sequential access pattern
        let is_sequential = if timestamps.len() > 1 {
            let mut sequential_count = 0;
            for w in timestamps.windows(2) {
                if w[1] > w[0] {
                    sequential_count += 1;
                }
            }
            (sequential_count as f32 / (timestamps.len() - 1) as f32 > 0.8) as i32 as f32
        } else {
            0.0
        };
        
        // Reuse distance (position in history window)
        let reuse_distance = if !timestamps.is_empty() {
            let newest_timestamp = *timestamps.last().unwrap();
            let position_in_window = timestamps
                .iter()
                .rev()
                .position(|&t| t == newest_timestamp)
                .unwrap_or(0) as f32;
            (position_in_window / 100.0).clamp(0.0, 1.0)  // Assume 100 recent accesses
        } else {
            1.0
        };
        
        // Last access type
        let last_access_type = match access_types.last() {
            Some(IoOp::Read) => 0.0,
            Some(IoOp::Write) => 1.0,
            None => 0.5,
        };
        
        // Blob size (average size from records)
        let size = if !sizes.is_empty() {
            let avg_size = sizes.iter().sum::<f64>() / sizes.len() as f64;
            (avg_size.log2() / 40.0).clamp(0.0, 1.0)  // Normalize assuming max 1TB
        } else {
            0.5
        };
        
        // Next access prediction (based on interval mean)
        let next_access_pred = mean_interval;
        
        // Overwrite amount (write frequency)
        let overwrite_amount = access_types
            .iter()
            .filter(|&&op| op == IoOp::Write)
            .count() as f32 / access_types.len() as f32;
        
        BlobFeatures {
            recency,
            frequency,
            mean_interval,
            std_interval,
            is_sequential,
            reuse_distance,
            last_access_type,
            size,
            next_access_pred,
            overwrite_amount,
        }
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}
```

---

## Models API

### Backend

The burn backend type used throughout the models.

```rust
// Primary backend: Wgpu for GPU acceleration
type Backend = Wgpu;

// Fallback backend: Ndarray for CPU-only systems
type CpuBackend = Ndarray;
```

### ContextualBandit

Extracts features and estimates importance from state.

```rust
use burn::nn::Linear;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct ContextualBandit<B: Backend> {
    /// First fully connected layer: 10 -> 64
    fc1: Linear<B>,
    /// Second fully connected layer: 64 -> 128
    fc2: Linear<B>,
    /// Third fully connected layer: 128 -> 31
    /// Outputs: 20 features + 1 importance score
    fc3: Linear<B>,
    /// Device for tensor operations
    device: Device<B>,
}

impl<B: Backend> ContextualBandit<B> {
    /// Creates a new contextual bandit model.
    pub fn new(state_dim: usize, device: &Device<B>) -> Self {
        let fc1 = Linear::new(64, true);
        let fc2 = Linear::new(128, true);
        let fc3 = Linear::new(31, true);  // 20 features + 1 score
        
        Self {
            fc1,
            fc2,
            fc3,
            device: device.clone(),
        }
    }

    /// Forward pass through the bandit network.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [state_dim]
    ///
    /// # Returns
    /// * `features` - Enhanced features [20]
    /// * `importance` - Importance score [1]
    pub fn forward(&self, x: Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let x = x.to_device(&self.device);
        
        // Forward through layers with activation
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x).relu();
        let x = self.fc3.forward(x);
        
        // Split into features and importance
        let features = x.clone().slice(0..20);
        let importance = x.slice(20..21).sigmoid();  // Normalize to [0, 1]
        
        (features, importance)
    }

    /// Returns the feature dimension output.
    pub const fn feature_dim(&self) -> usize {
        20
    }

    /// Returns the importance dimension.
    pub const fn importance_dim(&self) -> usize {
        1
    }
}
```

### QNetwork

Deep Q-Network for action-value estimation.

```rust
use burn::nn::Linear;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct QNetwork<B: Backend> {
    /// First fully connected layer
    fc1: Linear<B>,
    /// Second fully connected layer
    fc2: Linear<B>,
    /// Output layer (Q-values per action)
    fc3: Linear<B>,
    /// Device for tensor operations
    device: Device<B>,
}

impl<B: Backend> QNetwork<B> {
    /// Creates a new Q-network.
    pub fn new(input_dim: usize, action_dim: usize, device: &Device<B>) -> Self {
        let fc1 = Linear::new(128, true);
        let fc2 = Linear::new(128, true);
        let fc3 = Linear::new(action_dim, true);
        
        Self {
            fc1,
            fc2,
            fc3,
            device: device.clone(),
        }
    }

    /// Forward pass through the Q-network.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [input_dim]
    ///
    /// # Returns
    /// * `q_values` - Q-values for each action [action_dim]
    pub fn forward(&self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = x.to_device(&self.device);
        
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x).relu();
        let x = self.fc3.forward(x);
        
        x
    }

    /// Returns the action dimension.
    pub const fn action_dim(&self) -> usize {
        10
    }
}
```

### CombinedModel

Integrated model combining bandit and Q-network.

```rust
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct CombinedModel<B: Backend> {
    /// Contextual bandit for feature extraction and importance scoring
    pub bandit: ContextualBandit<B>,
    /// Q-network for action-value estimation
    pub qnetwork: QNetwork<B>,
    /// Tier selector for capacity-aware placement
    tier_selector: TierSelector,
    /// Device for tensor operations
    device: Device<B>,
}

impl<B: Backend> CombinedModel<B> {
    /// Creates a new combined model.
    pub fn new(state_dim: usize, num_tiers: usize, device: &Device<B>) -> Self {
        let bandit = ContextualBandit::new(state_dim, device);
        let qnetwork = QNetwork::new(20, 10, device);  // 20 features -> 10 actions
        let tier_selector = TierSelector::new(vec![]);  // Initialize with empty, populate later
        
        Self {
            bandit,
            qnetwork,
            tier_selector,
            device: device.clone(),
        }
    }

    /// Sets the tier selector (must be done after initialization).
    pub fn set_tier_selector(&mut self, selector: TierSelector) {
        self.tier_selector = selector;
    }

    /// Forward pass returning all components.
    ///
    /// # Arguments
    /// * `state` - Raw state vector [15]
    ///
    /// # Returns
    /// * `features` - Enhanced features [20]
    /// * `importance` - Importance score [1]
    /// * `q_values` - Q-values per action [10]
    pub fn forward(
        &self,
        state: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let (features, importance) = self.bandit.forward(state.clone());
        let q_values = self.qnetwork.forward(features.clone());
        
        (features, importance, q_values)
    }

    /// Selects an action using epsilon-greedy exploration.
    ///
    /// # Arguments
    /// * `state` - State tensor [15]
    /// * `epsilon` - Exploration rate [0, 1]
    ///
    /// # Returns
    /// * `action` - Selected action index [0-9]
    pub fn select_action(&self, state: Tensor<B, 1>, epsilon: f32) -> usize {
        if rand::random::<f32>() < epsilon {
            // Explore: random action
            rand::random::<usize>() % 10
        } else {
            // Exploit: select best action
            let (_, _, q_values) = self.forward(state);
            let q_values: Vec<f32> = q_values.to_vec();
            
            q_values
                .into_iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }

    /// Returns the model device.
    pub fn device(&self) -> &Device<B> {
        &self.device
    }
}
```

---

## Training API

### Transition

A single experience in the replay buffer.

```rust
#[derive(Debug, Clone)]
pub struct Transition {
    /// Current state [15]
    pub state: Vec<f32>,
    /// Action taken [0-9]
    pub action: usize,
    /// Reward received
    pub reward: f32,
    /// Next state [15]
    pub next_state: Vec<f32>,
    /// Whether episode terminated
    pub done: bool,
}

impl Transition {
    /// Creates a new transition.
    pub fn new(
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
}
```

### ReplayBuffer

Experience replay buffer with uniform sampling.

```rust
#[derive(Debug)]
pub struct ReplayBuffer {
    /// Circular buffer of transitions
    buffer: VecDeque<Transition>,
    /// Maximum capacity
    capacity: usize,
    /// Number of transitions added
    write_idx: usize,
}

impl ReplayBuffer {
    /// Creates a new replay buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            write_idx: 0,
        }
    }

    /// Adds a transition to the buffer.
    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }

    /// Samples a batch of transitions.
    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        if self.buffer.len() < batch_size {
            return self.buffer.iter().collect();
        }
        
        // Uniform random sampling
        let mut sample = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();
        
        for _ in 0..batch_size {
            let idx = rng.gen_range(0..self.buffer.len());
            if let Some(transition) = self.buffer.get(idx) {
                sample.push(transition);
            }
        }
        
        sample
    }

    /// Returns the number of transitions in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.write_idx = 0;
    }
}
```

### CombinedAgent

Main RL agent with training logic.

```rust
pub struct CombinedAgent<B: Backend> {
    /// Main model
    model: CombinedModel<B>,
    /// Target model for stable updates
    target_model: CombinedModel<B>,
    /// Experience replay buffer
    replay_buffer: ReplayBuffer,
    /// Current epsilon for exploration
    epsilon: f32,
    /// Decay rate for epsilon
    epsilon_decay: f32,
    /// Minimum epsilon value
    epsilon_min: f32,
    /// Discount factor
    gamma: f32,
    /// Learning rate
    learning_rate: f32,
    /// Batch size for training
    batch_size: usize,
    /// Optimizer
    optimizer: Adam<Wgpu>,
}

impl<B: Backend> CombinedAgent<B> {
    /// Creates a new agent with configuration.
    pub fn new(config: AgentConfig, device: &Device<B>) -> Self {
        let model = CombinedModel::new(
            config.state_dim,
            config.num_tiers,
            device,
        );
        
        let target_model = model.clone();
        
        let optimizer = Adam::new(
            AdamConfig::new()
                .with_lr(config.learning_rate),
        );
        
        Self {
            model,
            target_model,
            replay_buffer: ReplayBuffer::new(config.replay_capacity),
            epsilon: config.epsilon_start,
            epsilon_decay: config.epsilon_decay,
            epsilon_min: config.epsilon_min,
            gamma: config.gamma,
            learning_rate: config.learning_rate,
            batch_size: config.batch_size,
            optimizer,
        }
    }

    /// Selects an action for a given state.
    pub fn select_action(&self, state: &[f32], epsilon: Option<f32>) -> usize {
        let epsilon = epsilon.unwrap_or(self.epsilon);
        let state_tensor = Tensor::from_vec(state.to_vec(), &[state.len()], &self.model.device);
        self.model.select_action(state_tensor, epsilon)
    }

    /// Performs one training step.
    pub fn train_step(&mut self) -> Option<f32> {
        if self.replay_buffer.len() < self.batch_size {
            return None;
        }
        
        let batch = self.replay_buffer.sample(self.batch_size);
        self.update(batch)
    }

    fn update(&mut self, batch: Vec<&Transition>) -> Option<f32> {
        // Convert batch to tensors
        let states: Tensor<Wgpu, 1> = Tensor::from_vec(
            batch.iter().flat_map(|t| t.state.clone()).collect(),
            &[self.batch_size, 15],
            &self.model.device,
        );
        
        let actions: Vec<usize> = batch.iter().map(|t| t.action).collect();
        let rewards: Vec<f32> = batch.iter().map(|t| t.reward).collect();
        let next_states: Tensor<Wgpu, 1> = Tensor::from_vec(
            batch.iter().flat_map(|t| t.next_state.clone()).collect(),
            &[self.batch_size, 15],
            &self.model.device,
        );
        let dones: Vec<bool> = batch.iter().map(|t| t.done).collect();
        
        // Compute current Q-values
        let (_, _, q_values) = self.model.forward(states.clone());
        
        // Compute target Q-values
        let (_, _, next_q_values) = self.target_model.forward(next_states);
        
        // Compute TD loss
        let loss = self.compute_td_loss(&q_values, &actions, &rewards, &next_q_values, &dones);
        
        // Backpropagate
        self.optimizer.backward(&loss);
        
        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
        
        Some(loss.to_scalar())
    }

    fn compute_td_loss(
        &self,
        q_values: &Tensor<Wgpu, 2>,
        actions: &[usize],
        rewards: &[f32],
        next_q_values: &Tensor<Wgpu, 2>,
        dones: &[bool],
    ) -> Tensor<Wgpu, 1> {
        // TD loss implementation
        todo!()
    }

    /// Saves the model to a checkpoint file.
    pub fn save(&self, path: &Path) -> Result<(), EnvError> {
        let checkpoint = self.create_checkpoint()?;
        let data = postcard::to_allocvec(&checkpoint)
            .map_err(|e| EnvError::Serialization(e.to_string()))?;
        
        std::fs::write(path, data)
            .map_err(|e| EnvError::Io(e.to_string()))?;
        
        Ok(())
    }

    /// Loads a model from a checkpoint file.
    pub fn load(&mut self, path: &Path) -> Result<(), EnvError> {
        let data = std::fs::read(path)
            .map_err(|e| EnvError::Io(e.to_string()))?;
        
        let checkpoint: Checkpoint = postcard::from_bytes(&data)
            .map_err(|e| EnvError::Deserialization(e.to_string()))?;
        
        self.restore_checkpoint(checkpoint)
    }

    fn create_checkpoint(&self) -> Result<Checkpoint, EnvError> {
        todo!()
    }

    fn restore_checkpoint(&mut self, checkpoint: Checkpoint) -> Result<(), EnvError> {
        todo!()
    }

    /// Updates the target model with the main model weights.
    pub fn update_target(&mut self) {
        // Soft update: tau * main + (1 - tau) * target
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub state_dim: usize,
    pub num_tiers: usize,
    pub replay_capacity: usize,
    pub epsilon_start: f32,
    pub epsilon_decay: f32,
    pub epsilon_min: f32,
    pub gamma: f32,
    pub learning_rate: f32,
    pub batch_size: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            state_dim: 15,
            num_tiers: 5,
            replay_capacity: 10_000,
            epsilon_start: 1.0,
            epsilon_decay: 0.995,
            epsilon_min: 0.01,
            gamma: 0.99,
            learning_rate: 1e-4,
            batch_size: 32,
        }
    }
}
```

---

## Configuration API

### EnvConfig

Main environment configuration loaded from TOML.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvConfig {
    pub tiers: Vec<TierConfig>,
    pub max_steps: usize,
    pub reward_scale: f64,
    pub hotness_config: HotnessConfig,
}

impl EnvConfig {
    /// Loads configuration from a TOML file.
    pub fn load(path: &Path) -> Result<Self, EnvError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| EnvError::Io(e.to_string()))?;
        
        toml::from_str(&content)
            .map_err(|e| EnvError::Config(e.to_string()))
    }

    /// Creates default configuration.
    pub fn default() -> Self {
        Self {
            tiers: vec![
                TierConfig {
                    name: "memory".to_string(),
                    tier_type: TierType::Memory,
                    capacity: 1024.0 * 1024.0 * 1024.0,  // 1GB
                    read_latency_ms: 0.1,
                    write_latency_ms: 0.1,
                    eviction_enabled: true,
                    eviction_threshold: 0.9,
                },
                TierConfig {
                    name: "nvme".to_string(),
                    tier_type: TierType::Nvme,
                    capacity: 1024.0 * 1024.0 * 1024.0 * 10.0,  // 10GB
                    read_latency_ms: 0.5,
                    write_latency_ms: 1.0,
                    eviction_enabled: true,
                    eviction_threshold: 0.85,
                },
                TierConfig {
                    name: "ssd".to_string(),
                    tier_type: TierType::Ssd,
                    capacity: 1024.0 * 1024.0 * 1024.0 * 100.0,  // 100GB
                    read_latency_ms: 2.0,
                    write_latency_ms: 5.0,
                    eviction_enabled: true,
                    eviction_threshold: 0.85,
                },
                TierConfig {
                    name: "hdd".to_string(),
                    tier_type: TierType::Hdd,
                    capacity: 1024.0 * 1024.0 * 1024.0 * 1000.0,  // 1TB
                    read_latency_ms: 10.0,
                    write_latency_ms: 20.0,
                    eviction_enabled: false,
                    eviction_threshold: 0.95,
                },
                TierConfig {
                    name: "tape".to_string(),
                    tier_type: TierType::Tape,
                    capacity: 1024.0 * 1024.0 * 1024.0 * 10000.0,  // 10TB
                    read_latency_ms: 5000.0,  // 5 seconds
                    write_latency_ms: 1000.0,
                    eviction_enabled: false,
                    eviction_threshold: 1.0,
                },
            ],
            max_steps: 10_000,
            reward_scale: 1.0,
            hotness_config: HotnessConfig::default(),
        }
    }
}
```

---

## Error Handling

### EnvError

All errors returned by the environment and subsystems.

```rust
#[derive(Debug, thiserror::Error)]
pub enum EnvError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    /// Deserialization error
    #[error("Deserialization error: {0}")]
    Deserialization(String),
    
    /// Tier is full
    #[error("Tier '{0}' is full")]
    TierFull(String),
    
    /// Insufficient space for operation
    #[error("Insufficient space in tier '{tier}': needed {needed}, available {available}")]
    InsufficientSpace {
        tier: String,
        needed: f64,
        available: f64,
    },
    
    /// Invalid capacity configuration
    #[error("Invalid capacity for tier '{0}': must be positive")]
    InvalidCapacity(String),
    
    /// Trace parsing error
    #[error("Trace parsing error: {0}")]
    TraceParse(String),
    
    /// Blob not found
    #[error("Blob '{0}' not found")]
    BlobNotFound(String),
    
    /// Invalid action
    #[error("Invalid action: {0}")]
    InvalidAction(usize),
    
    /// Model error
    #[error("Model error: {0}")]
    Model(String),
}

impl EnvError {
    /// Returns true if this is a fatal error.
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            EnvError::InvalidCapacity(_) | EnvError::Deserialization(_)
        )
    }
}
```

---

## Cross-Reference

| Component | File | Module |
|-----------|------|--------|
| IOBufferEnv | `src/env/io_buffer_env.rs` | `crate::env` |
| Tier | `src/tier/tier.rs` | `crate::tier` |
| TierSelector | `src/tier/selector.rs` | `crate::tier` |
| AccessTracker | `src/features/tracker.rs` | `crate::features` |
| BlobFeatures | `src/features/extractor.rs` | `crate::features` |
| ContextualBandit | `src/models/bandit.rs` | `crate::models` |
| QNetwork | `src/models/dqn.rs` | `crate::models` |
| CombinedModel | `src/models/combined.rs` | `crate::models` |
| CombinedAgent | `src/training/trainer.rs` | `crate::training` |
| ReplayBuffer | `src/training/replay_buffer.rs` | `crate::training` |
| EnvConfig | `src/config.rs` | `crate` |