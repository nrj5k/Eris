# DataLoader Worker Tuning Guide

## Overview

The `num_workers` parameter controls the number of threads Burn's MultiThreadDataLoader uses for batching transitions. This guide explains optimal settings for different training configurations.

## Current Default

As of Phase 04 optimization, the default is **2 workers** (changed from 4).

### Why 2?

- **VecEnv**: With 16 parallel environments, the bottleneck is GPU computation, not CPU batching
- **TensorRingBuffer**: Data is already GPU-resident via pre-allocated tensors
- **Contention**: 4+ workers create threading overhead without improving throughput

## Worker Count Recommendations

| Scenario | num_workers | Rationale |
|----------|-------------|-----------|
| GPU + VecEnv (16 envs) | **2** (default) | Optimal balance, minimal contention |
| GPU + single env | 1-2 | GPU-bound, workers add minimal value |
| CPU + VecEnv | 4-8 | Benefit from parallel batching |
| CPU + single env | 4 | CPU preprocessing needs parallelism |

## CLI Override

Override the default via command line:

```bash
# GPU training (default is optimal)
cargo run --bin train_model -- --num-workers 2

# CPU-only training with more workers
cargo run --bin train_model -- --backend cpu --num-workers 4

# Single-threaded (for debugging)
cargo run --bin train_model -- --num-workers 0
```

## Performance Expectations

- **Workers=2 vs Workers=4**: 1.2-1.5x speedup (GPU training)
- **Workers=8**: Diminishing returns, possible slowdown due to contention
- **Workers=0**: Single-threaded, useful for debugging

## Benchmarking

Run the worker count benchmark to find optimal settings for your hardware:

```bash
cargo bench --bench worker_count_bench
```

Results are saved to `target/criterion/worker_count_comparison/`.

## Technical Details

### Threading Model

```
VecEnv (16 threads) → TensorRingBuffer (GPU) → DataLoader Workers → GPU Training
```

- **VecEnv threads**: One per environment, collecting experiences
- **TensorRingBuffer**: Pre-allocated GPU tensors (Phase 03)
- **DataLoader workers**: Only needed for batching (data already on GPU)

### When to Increase Workers

Increase workers if:
1. Using CPU backend (ndarray)
2. Heavy preprocessing is required on CPU
3. GPU utilization < 50% (CPU is bottleneck)

### When to Decrease Workers

Decrease to 1 or 0 if:
1. Debugging threading issues
2. On resource-constrained systems
3. GPU is already saturated

## References

- Phase 04 implementation: `benches/worker_count_bench.rs`
- Default changed in: `src/bin/train_model.rs`
- Related optimizations:
  - Phase 01: Shape alignment (state_dim = 32)
  - Phase 02: Batch size optimization (default = 2048)
  - Phase 03: TensorRingBuffer (GPU-resident data)
