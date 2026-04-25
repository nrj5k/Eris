//! Benchmark training speed at different batch sizes (16K, 32K, 64K, 128K)
//!
//! This benchmark finds the optimal batch size for GPU training on AMD 780M.
//! Measures throughput (ms/step) across different batch sizes.
//!
//! # Usage
//! ```bash
//! cargo bench --bench batch_size_sweep
//! ```

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::AutodiffBackend;
use burnme_rly::{
    buffer::Transition,
    trainers::{DQNTrainer, DQNTrainerConfig, QNetwork},
};
use rand::RngExt;
use std::time::Instant;

type TestBackend = Autodiff<NdArray>;

/// Simple Q-network for benchmarking
#[derive(Module, Debug)]
struct BenchmarkQNetwork<B: burn::tensor::backend::Backend> {
    layers: Vec<Linear<B>>,
    output: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> BenchmarkQNetwork<B> {
    fn new(state_dim: usize, action_dim: usize, device: &B::Device) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = state_dim;
        let hidden_dims = vec![64, 64];

        for &h in &hidden_dims {
            layers.push(LinearConfig::new(prev_dim, h).init(device));
            prev_dim = h;
        }

        let output = LinearConfig::new(prev_dim, action_dim).init(device);

        Self { layers, output }
    }
}

impl<B: burn::tensor::backend::Backend> BenchmarkQNetwork<B> {
    fn forward_inner(&self, input: burn::tensor::Tensor<B, 2>) -> burn::tensor::Tensor<B, 2> {
        let mut x = input;
        for layer in &self.layers {
            x = burn::tensor::activation::relu(layer.forward(x));
        }
        self.output.forward(x)
    }
}

impl<B: AutodiffBackend> QNetwork<B> for BenchmarkQNetwork<B> {
    fn forward_q(&self, states: burn::tensor::Tensor<B, 2>) -> burn::tensor::Tensor<B, 2> {
        self.forward_inner(states)
    }
}

/// Create synthetic transitions for benchmarking
fn create_test_transitions(count: usize, state_dim: usize, action_dim: usize) -> Vec<Transition> {
    let mut rng = rand::rng();

    (0..count)
        .map(|_| {
            let state: Vec<f32> = (0..state_dim).map(|_| rng.random()).collect();
            let action = rng.random_range(0..action_dim);
            let reward = rng.random_range(-1.0..1.0);
            let next_state: Vec<f32> = (0..state_dim).map(|_| rng.random()).collect();
            let done = rng.random_bool(0.1); // 10% episode ends

            Transition {
                state,
                action,
                reward,
                next_state,
                done,
            }
        })
        .collect()
}

/// Benchmark a specific batch size
/// Returns milliseconds per training step
fn benchmark_batch_size(batch_size: usize) -> f64 {
    let device = &<TestBackend as burn::tensor::backend::Backend>::Device::default();

    let state_dim = 32;
    let action_dim = 10;
    let buffer_capacity = batch_size * 4; // Need enough buffer for sampling

    // Create Q-network
    let q_network = BenchmarkQNetwork::<TestBackend>::new(state_dim, action_dim, device);

    // Create trainer
    let config = DQNTrainerConfig::default()
        .with_batch_size(batch_size)
        .with_buffer_capacity(buffer_capacity)
        .with_loss_sync_freq(1); // Sync every step for accurate timing

    let mut trainer = DQNTrainer::<TestBackend, BenchmarkQNetwork<TestBackend>>::new(
        q_network,
        state_dim,
        config,
        device.clone(),
    )
    .expect("Failed to create trainer");

    // Fill buffer with test data
    let transitions = create_test_transitions(buffer_capacity, state_dim, action_dim);

    // Convert to GPU tensors and fill buffer
    use burn::tensor::Tensor;
    for t in &transitions {
        let state_data: Vec<f32> = t.state.clone();
        let next_state_data: Vec<f32> = t.next_state.clone();
        let state_tensor: Tensor<TestBackend, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(state_data, [1, state_dim]).convert::<f32>(),
            device,
        );
        let next_state_tensor: Tensor<TestBackend, 2> = Tensor::from_data(
            burn::tensor::TensorData::new(next_state_data, [1, state_dim]).convert::<f32>(),
            device,
        );
        trainer.buffer.push(
            &state_tensor,
            t.action,
            t.reward,
            &next_state_tensor,
            t.done,
        );
    }

    // Warmup: run a few steps to stabilize
    for _ in 0..3 {
        trainer.train_step();
    }

    // Benchmark 10 training steps
    let steps = 10;
    let start = Instant::now();
    for _ in 0..steps {
        trainer.train_step();
    }
    let elapsed = start.elapsed().as_secs_f64();

    // Return ms per step
    (elapsed * 1000.0) / steps as f64
}

fn main() {
    println!("========================================");
    println!("  Batch Size Sweep Benchmark");
    println!("  Finding optimal batch size for GPU");
    println!("========================================\n");

    let batch_sizes = vec![16384, 32768, 65536, 131072];
    let mut results = Vec::new();

    println!("Testing batch sizes: {:?}\n", batch_sizes);
    println!(
        "{:<12} | {:>12} | {:>15}",
        "Batch Size", "ms/step", "Status"
    );
    println!("{}", "-".repeat(45));

    for &batch_size in &batch_sizes {
        print!("{:<12} | ", batch_size);

        match std::panic::catch_unwind(|| benchmark_batch_size(batch_size)) {
            Ok(ms_per_step) => {
                results.push((batch_size, ms_per_step));
                println!("{:>12.2} | {:>15}", ms_per_step, "✓");
            }
            Err(_) => {
                println!("{:>12} | {:>15}", "OOM/Error", "✗");
            }
        }
    }

    println!("\n========================================");
    println!("  Results Summary");
    println!("========================================");

    if results.is_empty() {
        println!("No successful benchmarks. Try smaller batch sizes.");
    } else {
        // Find best (lowest ms/step)
        let best = results
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        println!("\n{:<12} | {:>12}", "Batch Size", "ms/step");
        println!("{}", "-".repeat(28));
        for (size, ms) in &results {
            let marker = if size == &best.0 { " ← BEST" } else { "" };
            println!("{:<12} | {:>12.2}{}", size, ms, marker);
        }

        println!("\nOptimal batch size: {} ({} ms/step)", best.0, best.1);
        println!("\nRecommendation:");
        println!("- Use batch size {} for best throughput", best.0);
        println!("- Lower batch sizes = faster steps but less GPU utilization");
        println!("- Higher batch sizes = slower steps, may OOM");
    }
}
