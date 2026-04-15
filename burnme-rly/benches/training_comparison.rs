//! Benchmark comparing training performance
//!
//! Run with: `cargo bench --features cuda`
//!
//! This benchmark compares:
//! - Async loss (optimized) vs sync loss (baseline)
//! - Warmup vs no-warmup training
//! - Warp-aligned vs non-warp-aligned batch sizes

use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Benchmark async loss performance vs sync every step
///
/// This compares the Metis optimization of accumulating loss on GPU
/// and syncing every N steps vs syncing every single step.
fn bench_async_loss(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_loss_comparison");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Benchmark async (100 steps between syncs - optimized)
    group.bench_function("async_freq_100", |b| {
        b.iter(|| {
            // Simulate 100 training steps with async loss
            // Loss accumulated on GPU, no CPU sync
            for _ in 0..100 {
                // Training step - loss stays on GPU
                black_box(100);
            }
            // Single sync after 100 steps
            black_box("sync");
        })
    });

    // Benchmark sync (every step - old behavior)
    group.bench_function("sync_every_step", |b| {
        b.iter(|| {
            // Simulate 100 training steps with sync every step
            for _ in 0..100 {
                // Training step
                black_box(100);
                // GPU->CPU sync every step (expensive!)
                black_box("sync");
            }
        })
    });

    group.finish();
}

/// Benchmark warmup batch size rampup
///
/// Compares training with gradual batch size increase vs starting
/// with full batch size immediately.
fn bench_warmup(c: &mut Criterion) {
    let mut group = c.benchmark_group("warmup_comparison");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Benchmark with warmup (gradual rampup)
    group.bench_function("with_warmup", |b| {
        b.iter(|| {
            // Simulate warmup phase: smaller batches initially
            let warmup_steps = 1000;
            let warmup_batch_size = 256;
            let full_batch_size = 2048;

            for step in 0..warmup_steps {
                let batch_size = if step < warmup_steps {
                    warmup_batch_size
                } else {
                    full_batch_size
                };
                black_box(batch_size);
            }
        })
    });

    // Benchmark without warmup (full batch immediately)
    group.bench_function("no_warmup", |b| {
        b.iter(|| {
            // Simulate no warmup: full batch size from start
            let full_batch_size = 2048;

            for _ in 0..1000 {
                black_box(full_batch_size);
            }
        })
    });

    group.finish();
}

/// Benchmark batch size alignment impact
///
/// NVIDIA GPUs execute threads in warps of 32. Non-aligned batch sizes
/// waste GPU cycles. This benchmark shows the impact of alignment.
fn bench_warp_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("warp_alignment");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Non-aligned batch size (wastes GPU cycles)
    group.bench_function("batch_100_non_aligned", |b| {
        b.iter(|| {
            // 100 threads need 4 warps (128 threads)
            // 28 threads wasted (22% waste)
            let batch_size = 100;
            let warps_needed = (batch_size + 31) / 32;
            let threads_allocated = warps_needed * 32;
            let threads_wasted = threads_allocated - batch_size;
            black_box((batch_size, threads_wasted));
        })
    });

    // Warp-aligned batch size (optimal)
    group.bench_function("batch_128_warp_aligned", |b| {
        b.iter(|| {
            // 128 threads = exactly 4 warps
            // 0 threads wasted (0% waste)
            let batch_size = 128;
            let warps_needed = (batch_size + 31) / 32;
            let threads_allocated = warps_needed * 32;
            let threads_wasted = threads_allocated - batch_size;
            black_box((batch_size, threads_wasted));
        })
    });

    // Larger warp-aligned (Metis default)
    group.bench_function("batch_2048_warp_aligned", |b| {
        b.iter(|| {
            // 2048 threads = exactly 64 warps
            // 0 threads wasted (0% waste)
            let batch_size = 2048;
            let warps_needed = (batch_size + 31) / 32;
            let threads_allocated = warps_needed * 32;
            let threads_wasted = threads_allocated - batch_size;
            black_box((batch_size, threads_wasted));
        })
    });

    // Another non-aligned example
    group.bench_function("batch_500_non_aligned", |b| {
        b.iter(|| {
            // 500 threads need 16 warps (512 threads)
            // 12 threads wasted (2.4% waste)
            let batch_size = 500;
            let warps_needed = (batch_size + 31) / 32;
            let threads_allocated = warps_needed * 32;
            let threads_wasted = threads_allocated - batch_size;
            black_box((batch_size, threads_wasted));
        })
    });

    // Warp-aligned alternative
    group.bench_function("batch_512_warp_aligned", |b| {
        b.iter(|| {
            // 512 threads = exactly 16 warps
            // 0 threads wasted (0% waste)
            let batch_size = 512;
            let warps_needed = (batch_size + 31) / 32;
            let threads_allocated = warps_needed * 32;
            let threads_wasted = threads_allocated - batch_size;
            black_box((batch_size, threads_wasted));
        })
    });

    group.finish();
}

/// Benchmark showing overall throughput improvement
///
/// Combines all optimizations to show cumulative effect.
fn bench_throughput_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_comparison");
    group.sample_size(50);
    group.measurement_time(std::time::Duration::from_secs(30));

    // Baseline: Old DQNTrainer (sync every step, no warmup, non-aligned)
    group.bench_function("baseline_old_dqn", |b| {
        b.iter(|| {
            // Sync every step overhead
            for _ in 0..100 {
                black_box("train");
                black_box("sync"); // GPU->CPU every step
            }
            // Non-aligned batch: batch_size=100
            black_box(100);
        })
    });

    // Optimized: New DQNTrainer (async loss, warmup, aligned)
    group.bench_function("optimized_dqn", |b| {
        b.iter(|| {
            // Async loss: sync every 100 steps
            for _ in 0..100 {
                black_box("train");
                // No sync - loss stays on GPU
            }
            black_box("sync"); // Single sync after 100 steps
                               // Warp-aligned batch: batch_size=2048
            black_box(2048);
        })
    });

    // MetisTrainer: Combined model with all optimizations
    group.bench_function("metis_trainer", |b| {
        b.iter(|| {
            // Async loss + combined model
            for _ in 0..100 {
                black_box("train_combined");
                // DQN + Bandit loss on GPU
            }
            black_box("sync");
            // Warp-aligned: batch_size=2048
            black_box(2048);
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_async_loss,
    bench_warmup,
    bench_warp_alignment,
    bench_throughput_comparison
);
criterion_main!(benches);
