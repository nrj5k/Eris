//! Batch size benchmark for GPU utilization optimization
//!
//! This benchmark measures the performance impact of different batch sizes
//! on GPU kernel launches and memory throughput.
//!
//! # Usage
//! ```bash
//! cargo bench --bench batch_size_bench
//! ```

use burn::backend::NdArray;
use burn::tensor::backend::Backend;
use burnme_rly::buffer::{TensorTransitionBatch, Transition};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

type TestBackend = NdArray;

/// Create a dummy transition for benchmarking
fn create_dummy_transition(idx: usize) -> Transition {
    let state_dim = 32; // Typical state dimension
    Transition {
        state: vec![(idx % 100) as f32; state_dim],
        action: idx % 10,
        reward: (idx % 100) as f32 / 100.0,
        next_state: vec![((idx + 1) % 100) as f32; state_dim],
        done: idx % 100 == 99,
    }
}

fn get_test_device() -> <TestBackend as Backend>::Device {
    Default::default()
}

fn bench_batch_sizes(c: &mut Criterion) {
    let device = get_test_device();

    let mut group = c.benchmark_group("batch_size_comparison");

    // Test common batch sizes
    // Note: All sizes are multiples of 32 for GPU warp alignment
    for batch_size in [256, 512, 1024, 2048, 4096].iter() {
        let transitions: Vec<Transition> = (0..*batch_size)
            .map(|i| create_dummy_transition(i))
            .collect();

        group.bench_with_input(BenchmarkId::new("batch", batch_size), batch_size, |b, _| {
            b.iter(|| {
                let batch: TensorTransitionBatch<TestBackend> =
                    TensorTransitionBatch::from_transitions(&transitions, 32, &device);
                black_box(batch);
            });
        });
    }

    group.finish();
}

/// Benchmark memory throughput for different batch sizes
fn bench_memory_throughput(c: &mut Criterion) {
    let device = get_test_device();

    let mut group = c.benchmark_group("memory_throughput");
    group.measurement_time(std::time::Duration::from_secs(30));

    for batch_size in [512, 1024, 2048, 4096].iter() {
        let transitions: Vec<Transition> = (0..*batch_size)
            .map(|i| create_dummy_transition(i))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("throughput", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let batch: TensorTransitionBatch<TestBackend> =
                        TensorTransitionBatch::from_transitions(&transitions, 32, &device);
                    // Measure elements processed per second
                    black_box(batch.states.dims()[0]);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark batch size efficiency (samples per millisecond)
fn bench_batch_efficiency(c: &mut Criterion) {
    let device = get_test_device();

    let mut group = c.benchmark_group("batch_efficiency");
    group.throughput(criterion::Throughput::Elements(1));

    for batch_size in [256, 512, 1024, 2048].iter() {
        let transitions: Vec<Transition> = (0..*batch_size)
            .map(|i| create_dummy_transition(i))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("efficiency", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let batch: TensorTransitionBatch<TestBackend> =
                        TensorTransitionBatch::from_transitions(&transitions, 32, &device);
                    black_box(batch);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_batch_sizes,
    bench_memory_throughput,
    bench_batch_efficiency
);
criterion_main!(benches);
