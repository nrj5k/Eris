# HeirGym Enhanced Models Performance

This document details performance targets, benchmarks, and optimization strategies for the Eris Rust implementation.

## Performance Targets

### Comparison with Python Baseline

| Component | Python Baseline | Rust Target | Improvement |
|-----------|----------------|-------------|-------------|
| Feature extraction | ~1ms/op | <100μs/op | **10x** |
| Environment step | ~1ms | <100μs | **10x** |
| Model forward pass | ~10ms | <1ms | **10x** |
| Memory footprint | ~100MB | <20MB | **5x** |
| Training throughput | 100 eps/hr | 5000 eps/hr | **50x** |
| CSV load (18K rows) | ~2s | <100ms | **20x** |
| Per-step latency | ~1ms | <100μs | **10x** |

## Benchmark Suite

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench --all

# Run specific benchmarks
cargo bench --bench feature_extraction
cargo bench --bench environment_step
cargo bench --bench model_forward

# Run with flamegraph
cargo flamegraph --bin train
```

### Benchmark Definitions

#### Feature Extraction Benchmark

```rust
// benches/feature_extraction.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_feature_extraction(c: &mut Criterion) {
    let tracker = AccessTracker::new(10_000);
    let extractor = FeatureExtractor::default();
    let blob_id = "test_blob_123";
    
    // Add some test data
    for i in 0..1000 {
        tracker.record(blob_id, i * 100, "read", 1024.0);
    }
    
    c.bench_function("feature_extraction_single", |b| {
        b.iter(|| {
            black_box(extractor.extract(&tracker, blob_id, 100_000));
        })
    });
    
    c.bench_function("feature_extraction_batch_100", |b| {
        b.iter(|| {
            for i in 0..100 {
                let id = format!("blob_{}", i % 10);
                black_box(extractor.extract(&tracker, &id, 100_000));
            }
        })
    });
}

criterion_group!(benches, bench_feature_extraction);
criterion_main!(benches);
```

#### Environment Step Benchmark

```rust
// benches/environment_step.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_environment_step(c: &mut Criterion) {
    let mut env = IOBufferEnv::new(
        Path::new("config/tiers.toml"),
        Path::new("recorder-csv/NWChem-64_combined.csv"),
        10_000,
    ).unwrap();
    
    let _state = env.reset(None, false, None);
    
    c.bench_function("env_step", |b| {
        b.iter(|| {
            let action = black_box(rand::random::<usize>() % 10);
            let _ = env.step(action);
        })
    });
}

criterion_group!(benches, bench_environment_step);
criterion_main!(benches);
```

#### Model Forward Pass Benchmark

```rust
// benches/model_forward.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_model_forward(c: &mut Criterion) {
    let device = Device::default();
    let model = CombinedModel::new(15, 5, &device);
    let state = Tensor::randn([15], &device);
    
    c.bench_function("model_forward", |b| {
        b.iter(|| {
            black_box(model.forward(state.clone()));
        })
    });
    
    c.bench_function("model_forward_batch_32", |b| {
        let batch = Tensor::randn([32, 15], &device);
        b.iter(|| {
            // Process batch
            for i in 0..32 {
                let state_i = batch.clone().slice(i..i+1);
                black_box(model.forward(state_i));
            }
        })
    });
}

criterion_group!(benches, bench_model_forward);
criterion_main!(benches);
```

## Performance Profiling

### CPU Profiling

```bash
# Using perf
perf record -g -- cargo run --release --bin train --episodes 100
perf report

# Using callgrind
valgrind --tool=callgrind --dump-instr=yes cargo run --release --bin train --episodes 10
kcachegrind callgrind.out.*
```

### Memory Profiling

```bash
# Using valgrind massif
valgrind --tool=massif --pagesize=4096 cargo run --release --bin train --episodes 100

# Generate detailed heap profile
valgrind --tool=massif --detailed-peaks=yes --threshold=0.1 cargo run --release --bin train

# Analyze with ms_print
ms_print massif.out.*
```

### Allocation Tracking

```rust
// Enable in code for allocation tracking
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    let _profiler = dhat::Profiler::new_heap();
    // ... run training ...
}
```

## Optimization Strategies

### 1. Feature Extraction Optimization

**Current bottleneck**: Recomputing features for each blob on every access.

**Optimization**: Cache frequently accessed features.

```rust
// Pre-compute and cache features
impl FeatureExtractor {
    fn extract_cached(&mut self, tracker: &AccessTracker, blob_id: &str) -> Option<&BlobFeatures> {
        self.feature_cache.get(blob_id).or_else(|| {
            let features = self.extract(tracker, blob_id)?;
            self.feature_cache.insert(blob_id.to_string(), features);
            self.feature_cache.get(blob_id)
        })
    }
}

// Use smallvec for feature vector
type FeatureVec = SmallVec<[f32; 10]>;
```

### 2. Memory Optimization

**Current footprint**: ~100MB from Python, target <20MB.

**Optimizations**:

```rust
// Use SmallVec for VecDeque
use smallvec::SmallVec;

pub struct AccessTracker {
    history: VecDeque<AccessRecord, 1024>,  // Inline 1024 elements
    // ...
}

// Use arrayvec for fixed-size state
use arrayvec::ArrayVec;

pub struct EnvironmentState {
    tier_sizes: ArrayVec<f32, 5>,
    features: ArrayVec<f32, 10>,
}

// Enable LTO for smaller binary
// In Cargo.toml
[profile.release]
lto = "thin"
codegen-units = 1
opt-level = 3
```

### 3. SIMD Optimization

**Target functions**: Hotness score calculation, feature extraction.

```rust
// SIMD-accelerated hotness calculation
#[target_feature(enable = "avx2")]
unsafe fn compute_hotness_avx2(
    recency: &[f32],
    is_sequential: &[f32],
    overwrite: &[f32],
    output: &mut [f32],
) {
    // AVX2 implementation
    use std::arch::avx2::*;
    
    let zeros = _mm256_setzero_ps();
    let weights = _mm256_setr_ps(0.4, 0.2, 0.3, -0.1, 0.0, 0.0, 0.0, 0.0);
    
    for (r, o, ow, out) in recency.chunks(8)
        .zip(is_sequential.chunks(8))
        .zip(overwrite.chunks(8))
        .zip(output.chunks_mut(8))
    {
        let rec = _mm256_loadu_ps(r.as_ptr());
        let seq = _mm256_loadu_ps(is_sequential.as_ptr());
        let over = _mm256_loadu_ps(overwrite.as_ptr());
        
        // Compute: rec * 0.4 + seq * 0.2 + over * 0.3 - rec * 0.1
        let result = _mm256_fmadd_ps(rec, _mm256_set1_ps(0.4), zeros);
        let result = _mm256_fmadd_ps(seq, _mm256_set1_ps(0.2), result);
        let result = _mm256_fmadd_ps(over, _mm256_set1_ps(0.3), result);
        let result = _mm256_sub_ps(result, _mm256_mul_ps(rec, _mm256_set1_ps(0.1)));
        
        _mm256_storeu_ps(out.as_mut_ptr(), result);
    }
}
```

### 4. Neural Network Optimization

**Target**: Model forward pass <1ms.

```rust
// Use quantized inference for inference
// In Cargo.toml
burn = { version = "0.14", features = ["quantization"] }

// Quantize model after training
let quantized_model = model.quantize::<Q8>(QuantizationConfig::default());

// Enable TensorCore usage on NVIDIA GPUs
// The Wgpu backend automatically uses TensorCore when available
```

### 5. I/O Optimization

**Target**: CSV load <100ms.

```rust
// Use memmap for large CSV files
use memmap2::Mmap;

fn load_trace_mmap(path: &Path) -> Result<impl Iterator<Item = Result<BlobData, EnvError>>, EnvError> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    
    let cursor = Cursor::new(&mmap);
    let reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(cursor);
    
    Ok(reader.into_deserialize())
}

// Parallel parsing for large files
use rayon::prelude::*;

fn load_trace_parallel(path: &Path) -> Result<Vec<BlobData>, EnvError> {
    let content = std::fs::read(path)?;
    
    let records: Vec<BlobData> = content
        .par_lines()
        .skip(1)  // Skip header
        .filter_map(|line| parse_line(line).ok())
        .collect();
    
    Ok(records)
}
```

## Performance Targets by Component

### Feature Extraction

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Single blob extraction | <100μs | ~500μs | ❌ |
| Batch (100 blobs) | <5ms | ~50ms | ❌ |
| Memory per tracker | <1MB | ~2MB | ❌ |
| Throughput | 10K blobs/sec | 2K blobs/sec | ❌ |

### Environment

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Step (includes model) | <100μs | ~1ms | ❌ |
| Reset time | <50μs | ~100μs | ❌ |
| Memory per env | <5MB | ~10MB | ❌ |
| Episode time (10K steps) | <1s | ~10s | ❌ |

### Training

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Episode time | <720ms | ~36s | ❌ |
| Episodes/hour | 5000 | 100 | ❌ |
| Training throughput | 500 samples/sec | 10 samples/sec | ❌ |
| Memory (replay buffer) | <10MB | ~20MB | ❌ |

## Monitoring

### Logging Performance Metrics

```rust
use metrics::{gauge, histogram, counter};

pub struct PerformanceMonitor {
    step_latency:Histogram,
    feature_latency:Histogram,
    model_latency:Histogram,
}

impl PerformanceMonitor {
    pub fn record_step(&self, latency: std::time::Duration) {
        histogram!("eris_step_latency_ms").record(latency.as_secs_f64() * 1000.0);
    }
    
    pub fn record_feature_extraction(&self, blob_id: &str, latency: std::time::Duration) {
        histogram!("eris_feature_latency_ms").record(latency.as_secs_f64() * 1000.0);
        counter!("eris_features_extracted_total", 1);
    }
}
```

### Prometheus Integration

```rust
// In Cargo.toml
prometheus = "0.13"

pub fn start_metrics_server() {
    tokio::spawn(async {
        let registry = Registry::new();
        
        let step_latency = HistogramVec::new(
            HistogramOpts::new("eris_step_latency_ms", "Step latency in ms"),
            &["episode"],
        );
        
        registry.register(Box::new(step_latency.clone())).unwrap();
        
        // Start HTTP server
        let app = warp::path("metrics").map(move || {
            let mut output = String::new();
            for encoder in registry.gather() {
                output += &encode_to_string(encoder).unwrap();
            }
            output
        });
        
        warp::serve(app).run(([127, 0, 0, 1], 9090)).await;
    });
}
```

## Expected Final Performance

After all optimizations:

| Component | Expected Performance | Confidence |
|-----------|---------------------|------------|
| Feature extraction | 50-100μs/op | High |
| Environment step | 50-100μs | Medium |
| Model forward pass | 200-500μs | High |
| Memory footprint | 15-20MB | High |
| Training | 2000-5000 eps/hr | Medium |
| CSV load (18K rows) | 50-100ms | High |

## Related Documentation

- [Architecture](ARCHITECTURE.md) - System design overview
- [API Reference](API.md) - Detailed API documentation
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Development roadmap
- [Developer Guide](DEVELOPMENT.md) - Getting started guide