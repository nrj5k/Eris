# Batch Size Optimization Guide

## Overview

Batch size is a critical hyperparameter for deep learning training performance, especially on GPU accelerators. This guide explains how to select optimal batch sizes for the Eris RL training system.

## Why Multiples of 32?

### GPU Warp Size

Modern GPUs execute threads in groups called **warps** (NVIDIA) or **wavefronts** (AMD). Key facts:

- **NVIDIA warp size**: 32 threads
- **AMD wavefront size**: 64 threads (still divisible by 32)
- Warps execute in SIMD (Single Instruction, Multiple Data) fashion
- Misaligned batch sizes cause **warp divergence** and reduced utilization

### Memory Alignment

- GPU memory transactions are aligned to 32-element boundaries
- Batch sizes that are multiples of 32 ensure:
  - Coalesced memory access patterns
  - Maximum memory bandwidth utilization
  - Reduced memory transaction overhead

### Kernel Launch Efficiency

```
Batch Size = 512  →  16 warps (100% utilization) ✓
Batch Size = 500  →  16 warps (80% utilization, 20% wasted) ✗
Batch Size = 2048 →  64 warps (100% utilization) ✓
```

## Recommended Batch Sizes

### By GPU Memory

| GPU Memory | Recommended Batch Size | Max Safe Batch Size |
|------------|----------------------|---------------------|
| 2 GB       | 256-512              | 1024                |
| 4 GB       | 512-1024             | 2048                |
| 8 GB       | 1024-2048            | 4096                |
| 12 GB      | 2048-4096            | 8192                |
| 16 GB      | 4096-8192            | 16384               |
| 24 GB      | 8192-16384           | 32768               |

### By Training Stage

| Stage | Batch Size | Rationale |
|-------|-----------|-----------|
| Initial exploration | 256-512 | Faster iteration, more frequent updates |
| Main training | 2048 | Balance of throughput and stability |
| Fine-tuning | 1024-2048 | More stable gradients |
| Large-scale training | 4096-8192 | Maximum GPU utilization |

## Memory Requirements

### Per-Sample Memory Estimate

For a typical DQN with state_dim=32:

```
Data per sample:
├── State tensor:        32 × 4 bytes = 128 bytes
├── Next state tensor:   32 × 4 bytes = 128 bytes
├── Action tensor:       1 × 8 bytes =   8 bytes
├── Reward tensor:       1 × 4 bytes =   4 bytes
├── Done tensor:         1 × 4 bytes =   4 bytes
└── Total (data):                    ~272 bytes

Training overhead:
├── Activations:         ~2× data =  544 bytes
├── Gradients:           ~2× data =  544 bytes
├── Optimizer state:     ~3× data =  816 bytes (Adam: momentum + variance)
└── Total (overhead):               ~1904 bytes

Total per sample:                   ~2200 bytes
```

### Batch Memory Calculation

```rust
// Example: batch_size = 2048
let memory_per_batch = 2048 * 2200;  // ~4.5 MB per batch
let memory_for_buffer = 10000 * 272; // ~2.7 MB for replay buffer
let total_memory = memory_per_batch + memory_for_buffer; // ~7.2 MB
```

**Note**: This is a simplified estimate. Actual memory usage depends on:
- Model architecture (number of parameters)
- Hidden layer sizes
- Sequence length (for RNNs)
- Additional buffers (gradient accumulation, etc.)

## Trade-offs: Throughput vs. Latency

### Large Batch Sizes (2048-8192)

**Pros:**
- ✓ Better GPU utilization (more parallelism)
- ✓ Fewer kernel launches per epoch
- ✓ More stable gradient estimates
- ✓ Higher throughput (samples/second)
- ✓ Better for distributed training

**Cons:**
- ✗ Higher memory usage
- ✗ Less frequent weight updates
- ✗ May converge to sharper minima
- ✗ Higher latency per update

### Small Batch Sizes (32-256)

**Pros:**
- ✓ Lower memory usage
- ✓ More frequent updates
- ✓ Better for online learning
- ✓ Lower latency per update
- ✓ May find flatter minima (better generalization)

**Cons:**
- ✗ Poor GPU utilization
- ✗ More kernel launch overhead
- ✗ Noisier gradient estimates
- ✗ Lower throughput

## Auto-Tuning

Eris provides automatic batch size tuning based on GPU memory:

```rust
use eris::config::BatchTuner;

// Create tuner for your state dimension
let tuner = BatchTuner::new(32); // state_dim = 32

// Get optimal batch size for your GPU
let gpu_memory_mb = 8192; // 8GB GPU
let optimal_batch_size = tuner.tune(gpu_memory_mb);
println!("Optimal batch size: {}", optimal_batch_size);
```

The tuner considers:
- Available GPU memory
- State and action dimensions
- Model architecture (hidden layers)
- Safety headroom (25% reserved)
- Warp alignment (rounds to multiple of 32)

## Validation

Batch size validation is enforced at runtime:

```bash
# Valid (multiple of 32)
cargo run --bin train_model --batch-size 2048  # ✓

# Invalid (not multiple of 32)
cargo run --bin train_model --batch-size 2000  # ✗ Error: must be multiple of 32
```

## Benchmarking

Run batch size benchmarks to find optimal size for your hardware:

```bash
# Run all batch size benchmarks
cargo bench --bench batch_size_bench

# Look for:
# - Highest throughput (samples/second)
# - Lowest time per batch
# - No OOM errors
```

## Common Issues

### Out of Memory (OOM)

**Symptoms:** Training crashes with CUDA out of memory error

**Solutions:**
1. Reduce batch size by half
2. Reduce replay buffer size
3. Use gradient accumulation instead of large batches
4. Close other GPU applications

### Poor GPU Utilization

**Symptoms:** GPU usage < 50%, slow training

**Solutions:**
1. Increase batch size (if memory allows)
2. Ensure batch size is multiple of 32
3. Use mixed precision training (if supported)
4. Increase number of DataLoader workers

### Unstable Training

**Symptoms:** Loss oscillates wildly, poor convergence

**Solutions:**
1. Reduce batch size (more frequent updates)
2. Reduce learning rate
3. Increase gradient clipping
4. Use learning rate warmup

## Best Practices

1. **Start with 2048** - Good default for most GPUs
2. **Validate alignment** - Always use multiples of 32
3. **Monitor memory** - Leave 25% headroom for stability
4. **Benchmark** - Test different sizes on your hardware
5. **Scale learning rate** - When doubling batch size, consider doubling learning rate (linear scaling rule)

## References

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Deep Learning Batch Size Best Practices](https://arxiv.org/abs/1706.02677)
- [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)
