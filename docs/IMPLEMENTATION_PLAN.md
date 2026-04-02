# HeirGym Enhanced Models Implementation Plan

This document outlines the detailed implementation roadmap for the Eris Rust port, organized into 7 phases over 16 development days.

## Overview

| Phase | Duration | Focus | Deliverable |
|-------|----------|-------|-------------|
| Phase 1 | Days 1-3 | Foundation | Config + Trace loading |
| Phase 2 | Days 4-5 | Storage | Tier management |
| Phase 3 | Days 6-7 | Features | Feature extraction |
| Phase 4 | Days 8-10 | Models | Neural networks |
| Phase 5 | Days 11-12 | Environment | Gymnasium integration |
| Phase 6 | Days 13-14 | Training | Training loop |
| Phase 7 | Days 15-16 | Optimization | Performance tuning |

---

## Phase 1: Foundation (Days 1-3)

### Goal

Establish the project infrastructure with error handling, configuration parsing, and trace data loading.

### Tasks

#### Day 1: Project Setup and Error Types

**1.1 Create project structure**

```bash
cargo init --name eris
# Add dependencies to Cargo.toml
```

**1.2 Create error module** (`src/error.rs`)

```rust
// Deliverable: src/error.rs
use thiserror::Error;

#[derive(Debug, Error)]
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
    
    #[error("Insufficient space: needed {needed}, available {available}")]
    InsufficientSpace { needed: f64, available: f64 },
    
    #[error("Invalid capacity for tier '{0}'")]
    InvalidCapacity(String),
    
    #[error("Model error: {0}")]
    Model(String),
}

// Unit test: Error conversion
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_messages() {
        let err = EnvError::TierFull("memory".to_string());
        assert_eq!(format!("{}", err), "Tier 'memory' is full");
    }
}
```

**1.3 Create tier configuration** (`config/tiers.toml`)

```toml
# Deliverable: config/tiers.toml

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

#### Day 2: Configuration Parsing

**1.4 Create config module** (`src/config.rs`)

```rust
// Deliverable: src/config.rs
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub tier: Vec<TierConfig>,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TierConfig {
    pub name: String,
    pub tier_type: String,
    pub capacity_gb: f64,
    pub read_latency_ms: f64,
    pub write_latency_ms: f64,
    pub eviction_enabled: bool,
    pub eviction_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub replay_capacity: usize,
    pub learning_rate: f32,
    pub epsilon_start: f32,
    pub epsilon_decay: f32,
    pub epsilon_min: f32,
    pub gamma: f32,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, EnvError> {
        let content = std::fs::read_to_string(path)?;
        toml::from_str(&content).map_err(EnvError::Config)
    }
    
    pub fn num_tiers(&self) -> usize {
        self.tier.len()
    }
}
```

#### Day 3: Trace Data Structures

**1.5 Create blob data structure** (`src/trace/blob.rs`)

```rust
// Deliverable: src/trace/blob.rs
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
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
    
    /// Returns the timestamp for this access (derived from recency).
    pub fn timestamp(&self) -> u64 {
        // Convert recency to timestamp (simplified)
        (self.recency * 1000.0) as u64
    }
}
```

**1.6 Create trace reader** (`src/trace/reader.rs`)

```rust
// Deliverable: src/trace/reader.rs
use csv::{Reader, StringRecord};
use std::fs::File;
use std::path::Path;

pub struct TraceReader {
    reader: Reader<File>,
    headers: StringRecord,
}

impl TraceReader {
    pub fn new(path: &Path) -> Result<Self, EnvError> {
        let file = File::open(path)?;
        let mut reader = csv::Reader::from_reader(file);
        let headers = reader.headers()?.clone();
        
        Ok(Self { reader, headers })
    }
    
    /// Returns an iterator over all blob records.
    pub fn iter(&mut self) -> TraceIter {
        TraceIter {
            reader: &mut self.reader,
            phantom: std::marker::PhantomData,
        }
    }
}

pub struct TraceIter<'a> {
    reader: &'a mut Reader<File>,
    phantom: std::marker::PhantomData<BlobData>,
}

impl<'a> Iterator for TraceIter<'a> {
    type Item = Result<BlobData, EnvError>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let record = self.reader.records().next()?;
        
        match record {
            Ok(rec) => {
                let blob = BlobData {
                    offset_id: rec.get(0)?.to_string(),
                    offset_score: rec.get(1)?.parse().ok()?,
                    offset_access_frequency: rec.get(2)?.parse().ok()?,
                    access_offset: rec.get(3).and_then(|s| s.parse().ok()),
                    access_size: rec.get(4)?.parse().ok()?,
                    offset_size: rec.get(5)?.parse().ok()?,
                    is_sequence: rec.get(6)?.parse().ok()?,
                    first_seen: rec.get(7)?.parse().ok()?,
                    overwrite_amount: rec.get(8)?.parse().ok()?,
                    recency: rec.get(9)?.parse().ok()?,
                    io_op: rec.get(10)?.to_string(),
                };
                Some(Ok(blob))
            }
            Err(e) => Some(Err(EnvError::Csv(e))),
        }
    }
}

// Unit test: CSV parsing
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trace_parsing() {
        let path = Path::new("recorder-csv/NWChem-64_combined.csv");
        let mut reader = TraceReader::new(path).unwrap();
        
        let count = reader.iter().take(100).filter_map(|r| r.ok()).count();
        assert!(count > 50, "Should parse at least 50 records");
    }
}
```

### Phase 1 Deliverables

- [ ] `src/error.rs` - EnvError enum with all error variants
- [ ] `config/tiers.toml` - Tier definitions
- [ ] `src/config.rs` - TOML parser with Config struct
- [ ] `src/trace/blob.rs` - BlobData struct with 11 fields
- [ ] `src/trace/reader.rs` - CSV parser with streaming iterator
- [ ] Unit tests for CSV parsing (>90% coverage)
- [ ] **Working**: Load config + parse first 100 trace records

---

## Phase 2: Storage (Days 4-5)

### Goal

Implement tier management with capacity tracking and eviction.

### Tasks

#### Day 4: Tier Implementation

**2.1 Create tier struct** (`src/tier/tier.rs`)

```rust
// Deliverable: src/tier/tier.rs
use std::collections::HashMap;

pub struct Tier {
    pub config: TierConfig,
    storage: HashMap<String, f64>,
    pub current_size: f64,
    pub access_count: u64,
}

impl Tier {
    pub fn new(config: TierConfig) -> Self {
        Self {
            config,
            storage: HashMap::new(),
            current_size: 0.0,
            access_count: 0,
        }
    }
    
    pub fn write(&mut self, blob_id: &str, size: f64) -> Result<(), EnvError> {
        if self.current_size + size > self.config.capacity {
            return Err(EnvError::TierFull(self.config.name.clone()));
        }
        
        self.current_size += size;
        self.storage.insert(blob_id.to_string(), size);
        Ok(())
    }
    
    pub fn read(&self, blob_id: &str) -> Option<f64> {
        self.storage.get(blob_id).copied()
    }
    
    pub fn remove(&mut self, blob_id: &str) -> bool {
        if let Some(size) = self.storage.remove(blob_id) {
            self.current_size -= size;
            true
        } else {
            false
        }
    }
    
    pub fn clear(&mut self) {
        self.storage.clear();
        self.current_size = 0.0;
        self.access_count = 0;
    }
    
    pub fn available_capacity(&self) -> f64 {
        self.config.capacity - self.current_size
    }
    
    pub fn fill_percentage(&self) -> f32 {
        (self.current_size / self.config.capacity) as f32
    }
}

#[derive(Debug, Clone)]
pub struct TierConfig {
    pub name: String,
    pub tier_type: String,
    pub capacity: f64,
    pub read_latency_ms: f64,
    pub write_latency_ms: f64,
    pub eviction_enabled: bool,
    pub eviction_threshold: f32,
}
```

**2.2 Create tier selector** (`src/tier/selector.rs`)

```rust
// Deliverable: src/tier/selector.rs
pub struct TierSelector {
    tiers: Vec<TierConfig>,
    capacity_weights: Vec<f32>,
}

impl TierSelector {
    pub fn new(tiers: Vec<TierConfig>) -> Self {
        let total: f64 = tiers.iter().map(|t| t.capacity).sum();
        let weights: Vec<f32> = tiers
            .iter()
            .map(|t| (t.capacity / total) as f32)
            .collect();
        
        Self { tiers, capacity_weights: weights }
    }
    
    pub fn select_tier(&self, importance: f32) -> usize {
        // Capacity-weighted tier selection
        let normalized = importance.clamp(0.0, 1.0);
        
        self.tiers.iter()
            .enumerate()
            .map(|(idx, tier)| {
                let speed = tier_speed(&tier.tier_type);
                speed * (1.0 - self.capacity_weights[idx]) + normalized * self.capacity_weights[idx]
            })
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
    
    pub fn num_tiers(&self) -> usize {
        self.tiers.len()
    }
}

fn tier_speed(tier_type: &str) -> f32 {
    match tier_type {
        "memory" => 1.0,
        "nvme" => 0.8,
        "ssd" => 0.6,
        "hdd" => 0.3,
        "tape" => 0.1,
        _ => 0.5,
    }
}
```

#### Day 5: Tier Manager

**2.3 Create tier manager** (`src/tier/manager.rs`)

```rust
// Deliverable: src/tier/manager.rs
pub struct TierManager {
    tiers: Vec<Tier>,
    selector: TierSelector,
}

impl TierManager {
    pub fn new(configs: &[TierConfig]) -> Result<Self, EnvError> {
        let mut tiers = Vec::new();
        
        for config in configs {
            if config.capacity <= 0.0 {
                return Err(EnvError::InvalidCapacity(config.name.clone()));
            }
            tiers.push(Tier::new(config.clone()));
        }
        
        let configs: Vec<_> = tiers.iter().map(|t| t.config.clone()).collect();
        let selector = TierSelector::new(configs);
        
        Ok(Self { tiers, selector })
    }
    
    pub fn get_state(&self) -> Vec<f32> {
        self.tiers.iter().map(|t| t.fill_percentage()).collect()
    }
    
    pub fn select_tier(&self, importance: f32) -> usize {
        self.selector.select_tier(importance)
    }
    
    pub fn get_tier(&mut self, idx: usize) -> Option<&mut Tier> {
        self.tiers.get_mut(idx)
    }
}
```

**2.4 Create hotness scoring** (`src/tier/hotness.rs`)

```rust
// Deliverable: src/tier/hotness.rs
pub fn hotness_score(
    recency: f32,
    is_sequence: bool,
    overwrite_amount: f32,
    config: &HotnessConfig,
) -> f32 {
    let seq_bonus = if is_sequence { 1.0 } else { 0.0 };
    
    recency * config.recency_weight
        + seq_bonus * config.sequence_weight
        + overwrite_amount * config.overwrite_weight
        - recency * config.age_penalty
}

#[derive(Debug, Clone)]
pub struct HotnessConfig {
    pub recency_weight: f32,
    pub sequence_weight: f32,
    pub overwrite_weight: f32,
    pub age_penalty: f32,
}

impl Default for HotnessConfig {
    fn default() -> Self {
        Self {
            recency_weight: 0.4,
            sequence_weight: 0.2,
            overwrite_weight: 0.3,
            age_penalty: 0.1,
        }
    }
}
```

### Phase 2 Deliverables

- [ ] `src/tier/tier.rs` - Tier with HashMap storage
- [ ] `src/tier/selector.rs` - Capacity-weighted selection
- [ ] `src/tier/manager.rs` - Multi-tier coordinator
- [ ] `src/tier/hotness.rs` - Hotness scoring formula
- [ ] Unit tests for tier management (>90% coverage)
- [ ] **Working**: Write/remove data to tiers with capacity tracking

---

## Phase 3: Features (Days 6-7)

### Goal

Implement feature extraction from access history.

### Tasks

#### Day 6: Access Tracker

**3.1 Create access tracker** (`src/features/tracker.rs`)

```rust
// Deliverable: src/features/tracker.rs
use std::collections::{VecDeque, BTreeMap};

pub struct AccessTracker {
    history: VecDeque<AccessRecord>,
    index: BTreeMap<String, Vec<usize>>,
    window_size: usize,
}

struct AccessRecord {
    blob_id: String,
    timestamp: u64,
    access_type: String,
    size: f64,
}

impl AccessTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(window_size),
            index: BTreeMap::new(),
            window_size,
        }
    }
    
    pub fn record(&mut self, blob_id: &str, timestamp: u64, access_type: &str, size: f64) {
        let idx = self.history.len();
        
        self.history.push_back(AccessRecord {
            blob_id: blob_id.to_string(),
            timestamp,
            access_type: access_type.to_string(),
            size,
        });
        
        self.index
            .entry(blob_id.to_string())
            .or_insert_with(Vec::new)
            .push(idx);
        
        // Evict if needed
        if self.history.len() > self.window_size {
            if let Some(oldest) = self.history.pop_front() {
                if let Some(positions) = self.index.get_mut(&oldest.blob_id) {
                    positions.remove(0);
                    positions.iter_mut().for_each(|p| *p -= 1);
                }
            }
        }
    }
    
    pub fn get_records(&self, blob_id: &str) -> Option<Vec<&AccessRecord>> {
        let positions = self.index.get(blob_id)?;
        Some(positions.iter().filter_map(|&p| self.history.get(p)).collect())
    }
    
    pub fn len(&self) -> usize {
        self.history.len()
    }
    
    pub fn clear(&mut self) {
        self.history.clear();
        self.index.clear();
    }
}
```

**3.2 Create feature extractor** (`src/features/extractor.rs`)

```rust
// Deliverable: src/features/extractor.rs
use super::tracker::AccessTracker;

#[derive(Debug, Default)]
pub struct BlobFeatures {
    pub recency: f32,
    pub frequency: f32,
    pub mean_interval: f32,
    pub std_interval: f32,
    pub is_sequential: f32,
    pub reuse_distance: f32,
    pub last_access_type: f32,
    pub size: f32,
    pub next_access_pred: f32,
    pub overwrite_amount: f32,
}

impl BlobFeatures {
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.recency, self.frequency, self.mean_interval,
            self.std_interval, self.is_sequential, self.reuse_distance,
            self.last_access_type, self.size, self.next_access_pred,
            self.overwrite_amount,
        ]
    }
}

pub struct FeatureExtractor;

impl FeatureExtractor {
    pub fn extract(tracker: &AccessTracker, blob_id: &str, now: u64) -> Option<BlobFeatures> {
        let records = tracker.get_records(blob_id)?;
        Some(Self::compute_features(records, now))
    }
    
    fn compute_features(records: Vec<&AccessRecord>, now: u64) -> BlobFeatures {
        let mut sorted: Vec<_> = records.into_iter().collect();
        sorted.sort_by_key(|r| r.timestamp);
        
        let timestamps: Vec<u64> = sorted.iter().map(|r| r.timestamp).collect();
        let access_types: Vec<&str> = sorted.iter().map(|r| r.access_type.as_str()).collect();
        
        let recency = if let Some(&last) = timestamps.last() {
            ((now - last) as f32 / 1000.0).clamp(0.0, 1.0)
        } else {
            1.0
        };
        
        let frequency = (sorted.len() as f32 / 1000.0).clamp(0.0, 1.0);
        
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
                .map(|i| (*i as f64 - mean).powi(2))
                .sum::<f64>() / intervals.len() as f64;
            (variance.sqrt() as f32 / 1000.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        
        let is_sequential = if timestamps.len() > 1 {
            let seq_count = timestamps.windows(2)
                .filter(|w| w[1] > w[0])
                .count();
            (seq_count as f32 / (timestamps.len() - 1) as f32 > 0.8) as i32 as f32
        } else {
            0.0
        };
        
        BlobFeatures {
            recency,
            frequency,
            mean_interval,
            std_interval,
            is_sequential,
            reuse_distance: 0.5,  // TODO
            last_access_type: match access_types.last() {
                Some("write") => 1.0,
                _ => 0.0,
            },
            size: 0.5,  // TODO
            next_access_pred: mean_interval,
            overwrite_amount: access_types.iter()
                .filter(|&&t| t == "write")
                .count() as f32 / access_types.len() as f32,
        }
    }
}
```

#### Day 7: State Encoding

**3.3 Implement state encoding**

```rust
// Add to src/env/io_buffer_env.rs

fn get_state(&self, tier_sizes: &[f32], features: &[f32]) -> Vec<f32> {
    [tier_sizes.to_vec(), features.to_vec()].concat()
}

// State dimension: 5 tier sizes + 10 features = 15
```

**3.4 Set up Criterion benchmarks**

```rust
// benchmarks/feature_extraction.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn feature_extraction_benchmark(c: &mut Criterion) {
    c.bench_function("feature_extraction_100", |b| {
        b.iter(|| {
            // Extract features for 100 blobs
            for i in 0..100 {
                black_box(i);
            }
        })
    });
}

criterion_group!(benches, feature_extraction_benchmark);
criterion_main!(benches);
```

### Phase 3 Deliverables

- [ ] `src/features/tracker.rs` - Access history with window
- [ ] `src/features/extractor.rs` - 10-dim feature extraction
- [ ] State encoding (15-dim vector)
- [ ] `benches/feature_extraction.rs` - Criterion benchmark
- [ ] **Working**: Extract features for any blob <100μs

---

## Phase 4: Models (Days 8-10)

### Goal

Implement neural network models using Burn.

### Tasks

#### Day 8: Burn Setup and QNetwork

**4.1 Add Burn dependencies**

```toml
# Cargo.toml
burn = "0.14"
burn-ndarray = "0.14"  # CPU backend
burn-wgpu = "0.14"     # GPU backend (optional)
```

**4.2 Create QNetwork** (`src/models/dqn.rs`)

```rust
// Deliverable: src/models/dqn.rs
use burn::nn::Linear;
use burn::tensor::Tensor;
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct QNetwork<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B: Backend> QNetwork<B> {
    pub fn new(input_dim: usize, action_dim: usize, device: &Device<B>) -> Self {
        let fc1 = Linear::new(128, true);
        let fc2 = Linear::new(128, true);
        let fc3 = Linear::new(action_dim, true);
        
        Self { fc1, fc2, fc3 }
    }
    
    pub fn forward(&self, x: Tensor<B, 1>) -> Tensor<B, 1> {
        let x = x.to_device(self.device());
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x).relu();
        self.fc3.forward(x)
    }
    
    pub fn device(&self) -> &Device<B> {
        todo!()
    }
}
```

#### Day 9: ContextualBandit

**4.3 Create ContextualBandit** (`src/models/bandit.rs`)

```rust
// Deliverable: src/models/bandit.rs
use burn::nn::Linear;
use burn::tensor::Tensor;
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct ContextualBandit<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
}

impl<B: Backend> ContextualBandit<B> {
    pub fn new(state_dim: usize, device: &Device<B>) -> Self {
        let fc1 = Linear::new(64, true);
        let fc2 = Linear::new(128, true);
        let fc3 = Linear::new(31, true);  // 20 features + 1 importance
        
        Self { fc1, fc2, fc3 }
    }
    
    pub fn forward(&self, x: Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let x = x.to_device(self.device());
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(x).relu();
        let x = self.fc3.forward(x);
        
        let features = x.clone().slice(0..20);
        let importance = x.slice(20..21).sigmoid();
        
        (features, importance)
    }
    
    pub fn device(&self) -> &Device<B> {
        todo!()
    }
}
```

#### Day 10: CombinedModel

**4.4 Create CombinedModel** (`src/models/combined.rs`)

```rust
// Deliverable: src/models/combined.rs
use burn::tensor::Tensor;
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct CombinedModel<B: Backend> {
    pub bandit: ContextualBandit<B>,
    pub qnetwork: QNetwork<B>,
    tier_selector: TierSelector,
}

impl<B: Backend> CombinedModel<B> {
    pub fn new(state_dim: usize, num_tiers: usize, device: &Device<B>) -> Self {
        let bandit = ContextualBandit::new(state_dim, device);
        let qnetwork = QNetwork::new(20, 10, device);
        
        Self {
            bandit,
            qnetwork,
            tier_selector: TierSelector::new(vec![]),
        }
    }
    
    pub fn forward(&self, state: Tensor<B, 1>) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
        let (features, importance) = self.bandit.forward(state.clone());
        let q_values = self.qnetwork.forward(features.clone());
        
        (features, importance, q_values)
    }
    
    pub fn select_action(&self, state: Tensor<B, 1>, epsilon: f32) -> usize {
        if rand::random::<f32>() < epsilon {
            rand::random::<usize>() % 10
        } else {
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
}
```

### Phase 4 Deliverables

- [ ] Burn backend setup (Wgpu + Ndarray)
- [ ] `src/models/dqn.rs` - QNetwork (3-layer MLP)
- [ ] `src/models/bandit.rs` - ContextualBandit
- [ ] `src/models/combined.rs` - CombinedModel
- [ ] Shape validation tests
- [ ] **Working**: Forward pass produces correct shapes

---

## Phase 5: Environment (Days 11-12)

### Goal

Implement gymnasium environment interface.

### Tasks

#### Day 11: IOBufferEnv Core

**5.1 Create IOBufferEnv** (`src/env/io_buffer_env.rs`)

```rust
// Deliverable: src/env/io_buffer_env.rs
use gymnasium::core::{Env, Reward, Termination, ActType, ObsType};
use gymnasium::space::Discrete;

pub struct IOBufferEnv {
    tiers: Vec<Tier>,
    access_tracker: AccessTracker,
    current_blob: Option<BlobData>,
    step_count: usize,
    max_steps: usize,
    rng: StdRng,
}

impl IOBufferEnv {
    pub fn new(config_path: &Path, trace_path: &Path, max_steps: usize) -> Result<Self, EnvError> {
        let config = Config::load(config_path)?;
        let trace_reader = TraceReader::new(trace_path)?;
        
        Ok(Self {
            tiers: TierManager::new(&config.tier)?.into_tiers(),
            access_tracker: AccessTracker::new(10_000),
            current_blob: None,
            step_count: 0,
            max_steps,
            rng: StdRng::from_entropy(),
        })
    }
}

impl Env for IOBufferEnv {
    type Action = usize;  // 0-9 (5 tiers × 2 operations)
    type Observation = Vec<f32>;  // 15-dim
    type Info = ();
    
    fn step(&mut self, action: usize) -> ActReward<Self::Observation, Termination, Self::Info> {
        let tier_idx = action / 2;
        let op_type = if action % 2 == 0 { "read" } else { "write" };
        
        if tier_idx >= self.tiers.len() {
            return (self.get_state(), -1.0, Termination::Truncated, false, ());
        }
        
        // Execute action, compute reward
        let latency = self.execute_action(tier_idx, op_type);
        let reward = -(latency as f32);
        
        // Get next blob
        self.current_blob = self.next_trace_record();
        self.step_count += 1;
        
        let done = self.step_count >= self.max_steps;
        
        (self.get_state(), reward, Termination::Truncated, done, ())
    }
    
    fn reset(
        &mut self,
        seed: Option<u64>,
        _return_info: bool,
        _options: Option<ResetOptions>,
    ) -> ObsType<Self::Observation> {
        if let Some(seed) = seed {
            self.rng = StdRng::seed_from_u64(seed);
        }
        
        self.tiers.iter_mut().for_each(Tier::clear);
        self.access_tracker.clear();
        self.step_count = 0;
        self.current_blob = None;
        
        self.get_state()
    }
    
    fn action_space(&self) -> &Space<Self::Action> {
        &Discrete::new(10)
    }
    
    fn observation_space(&self) -> &Space<Self::Observation> {
        todo!()
    }
}

impl IOBufferEnv {
    fn get_state(&self) -> Vec<f32> {
        let tier_sizes: Vec<f32> = self.tiers.iter()
            .map(|t| t.fill_percentage())
            .collect();
        
        let features = self.current_blob
            .as_ref()
            .and_then(|b| FeatureExtractor::extract(&self.access_tracker, &b.offset_id, b.timestamp()))
            .unwrap_or_else(|| vec![0.0; 10]);
        
        [tier_sizes, features].concat()
    }
}
```

#### Day 12: Reward and Integration

**5.2 Implement reward calculation**

```rust
fn compute_reward(&self, action: usize, blob: &BlobData) -> f32 {
    let tier_idx = action / 2;
    let is_write = action % 2 == 1;
    
    let tier = &self.tiers[tier_idx];
    let base_latency = if is_write {
        tier.config.write_latency_ms
    } else {
        tier.config.read_latency_ms
    };
    
    // Add latency based on blob size and tier
    let size_factor = (blob.offset_size / 1024.0 / 1024.0).log2().max(0.0);
    let latency = base_latency * (1.0 + size_factor);
    
    -(latency as f32)
}
```

### Phase 5 Deliverables

- [ ] `src/env/io_buffer_env.rs` - Gymnasium environment
- [ ] Implement Env trait (step, reset, spaces)
- [ ] Reward calculation logic
- [ ] Unit tests for environment
- [ ] **Working**: Run full episodes

---

## Phase 6: Training (Days 13-14)

### Goal

Implement training loop with experience replay.

### Tasks

#### Day 13: ReplayBuffer and Agent

**6.1 Create ReplayBuffer** (`src/training/replay_buffer.rs`)

```rust
// Deliverable: src/training/replay_buffer.rs
use std::collections::VecDeque;

pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    
    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }
    
    pub fn sample(&self, batch_size: usize) -> Vec<&Transition> {
        if self.buffer.len() < batch_size {
            return self.buffer.iter().collect();
        }
        
        let mut rng = rand::thread_rng();
        let mut sample = Vec::with_capacity(batch_size);
        
        for _ in 0..batch_size {
            let idx = rng.gen_range(0..self.buffer.len());
            if let Some(t) = self.buffer.get(idx) {
                sample.push(t);
            }
        }
        
        sample
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}
```

**6.2 Create CombinedAgent** (`src/training/trainer.rs`)

```rust
// Deliverable: src/training/trainer.rs
pub struct CombinedAgent<B: Backend> {
    model: CombinedModel<B>,
    target_model: CombinedModel<B>,
    replay_buffer: ReplayBuffer,
    epsilon: f32,
    epsilon_decay: f32,
    epsilon_min: f32,
    gamma: f32,
    batch_size: usize,
}

impl<B: Backend> CombinedAgent<B> {
    pub fn new(config: AgentConfig, device: &Device<B>) -> Self {
        let model = CombinedModel::new(config.state_dim, config.num_tiers, device);
        let target_model = model.clone();
        
        Self {
            model,
            target_model,
            replay_buffer: ReplayBuffer::new(config.replay_capacity),
            epsilon: config.epsilon_start,
            epsilon_decay: config.epsilon_decay,
            epsilon_min: config.epsilon_min,
            gamma: config.gamma,
            batch_size: config.batch_size,
        }
    }
    
    pub fn select_action(&self, state: &[f32]) -> usize {
        let epsilon = self.epsilon;
        let state_tensor = Tensor::from_vec(state.to_vec(), &[state.len()], self.model.device());
        self.model.select_action(state_tensor, epsilon)
    }
    
    pub fn train_step(&mut self) -> Option<f32> {
        if self.replay_buffer.len() < self.batch_size {
            return None;
        }
        
        let batch = self.replay_buffer.sample(self.batch_size);
        self.update(batch)
    }
    
    fn update(&mut self, batch: Vec<&Transition>) -> Option<f32> {
        // TD learning implementation
        todo!()
    }
    
    pub fn save(&self, path: &Path) -> Result<(), EnvError> {
        todo!()
    }
    
    pub fn load(&mut self, path: &Path) -> Result<(), EnvError> {
        todo!()
    }
}
```

#### Day 14: Training Binary

**6.3 Create training binary** (`src/bin/train.rs`)

```rust
// Deliverable: src/bin/train.rs
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt)]
struct Args {
    #[structopt(short, long, default_value = "config/tiers.toml")]
    config: PathBuf,
    
    #[structopt(short, long, default_value = "recorder-csv/NWChem-64_combined.csv")]
    trace: PathBuf,
    
    #[structopt(short, long, default_value = "1000")]
    episodes: usize,
    
    #[structopt(short, long, default_value = "output/model.postcard")]
    checkpoint: PathBuf,
    
    #[structopt(short, long, default_value = "info")]
    log_level: String,
}

fn main() {
    let args = Args::from_args();
    
    // Initialize environment
    let env = IOBufferEnv::new(&args.config, &args.trace, 10_000).unwrap();
    
    // Initialize agent
    let config = AgentConfig::default();
    let device = Device::default();
    let mut agent = CombinedAgent::new(config, &device);
    
    // Training loop
    for episode in 0..args.episodes {
        let state = env.reset(None, false, None);
        
        loop {
            let action = agent.select_action(&state);
            let (next_state, reward, done, _, _) = env.step(action);
            
            agent.replay_buffer.push(Transition {
                state: state.clone(),
                action,
                reward,
                next_state: next_state.clone(),
                done,
            });
            
            agent.train_step();
            
            if done {
                break;
            }
            
            state = next_state;
        }
        
        println!("Episode {} complete", episode);
    }
    
    agent.save(&args.checkpoint).unwrap();
}
```

### Phase 6 Deliverables

- [ ] `src/training/replay_buffer.rs` - Experience replay
- [ ] `src/training/trainer.rs` - CombinedAgent with TD learning
- [ ] `src/training/checkpoint.rs` - Postcard save/load
- [ ] `src/bin/train.rs` - Training executable
- [ ] Integration tests
- [ ] **Working**: Train model from trace

---

## Phase 7: Optimization (Days 15-16)

### Goal

Optimize performance and meet targets.

### Tasks

#### Day 15: Profiling and Optimization

**7.1 Profile hot paths**

```bash
# Use perf to profile
perf record -g cargo run --release --bin train
perf report
```

**7.2 Apply optimizations**

```rust
// SIMD for hotness calculation
#[target_feature(enable = "avx2")]
fn compute_hotness_simd(...) {
    // AVX2 optimized implementation
}

// Use smallvec for VecDeque
use smallvec::SmallVec;

struct AccessTracker {
    history: VecDeque<AccessRecord, 1000>,  // Inline 1000 elements
}

// Use arrayvec for fixed-size buffers
use arrayvec::ArrayVec;

fn encode_state(state: &mut ArrayVec<f32, 15>) {
    state.extend(tier_sizes.iter().copied());
    state.extend(features.iter().copied());
}
```

#### Day 16: Final Benchmarks

**7.3 Run final benchmarks**

```bash
# Feature extraction benchmark
cargo bench --bench feature_extraction

# Environment step benchmark  
cargo bench --bench environment_step

# Model forward benchmark
cargo bench --bench model_forward

# Memory profiling
valgrind --tool=massif cargo run --release --bin train --episodes 10

# Training throughput
cargo run --release --bin train --episodes 100 --log-level info
```

### Phase 7 Deliverables

- [ ] Performance profiling results
- [ ] SIMD optimizations applied
- [ ] Memory optimization applied
- [ ] Final benchmark results
- [ ] **Working**: Meet performance targets (<100μs/step)

---

## Dependencies Between Phases

```
Phase 1 (Foundation)
    │
    ├─► Phase 2 (Storage)    - Uses Config from Phase 1
    │
    ├─► Phase 3 (Features)   - Uses AccessTracker from Phase 2
    │
    ├─► Phase 4 (Models)     - Independent
    │
    ├─► Phase 5 (Environment) - Uses Features from Phase 3 + Storage from Phase 2
    │
    └─► Phase 6 (Training)   - Uses Environment from Phase 5 + Models from Phase 4
```

## Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| 1-3 | Low | Standard Rust patterns |
| 4 | Medium | Burn API complexity |
| 5 | Low | Gymnasium interface is stable |
| 6 | Medium | Training stability |
| 7 | High | Performance targets aggressive |

## Success Criteria

- [ ] All unit tests pass (>90% coverage)
- [ ] Feature extraction <100μs per operation
- [ ] Environment step <100μs per operation
- [ ] Memory usage <20MB at rest
- [ ] Training achieves 5000+ episodes/hour
- [ ] Checkpoint save/load works correctly