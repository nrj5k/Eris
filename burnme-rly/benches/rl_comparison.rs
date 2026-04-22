//! Benchmark comparing DQN, Metis, and PPO on identical workloads
//!
//! This benchmark measures training throughput and loss convergence
//! across different RL algorithms using the same synthetic data.
//!
//! # Usage
//! ```bash
//! cargo bench --bench rl_comparison
//! ```

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::AutodiffBackend;
use burnme_rly::{
    DQNTrainer, DQNTrainerConfig, MetisTrainer, MetisTrainerConfig, PpoTrainer, PpoTrainerConfig,
    buffer::Transition,
    models::{CombinedModel, CombinedModelConfig},
    trainers::dqn_trainer::QNetwork,
};
use rand::RngExt;
use std::time::Instant;

type TestBackend = Autodiff<NdArray>;

/// Simple Q-network for DQN benchmarking
#[derive(Module, Debug)]
struct SimpleQNetwork<B: burn::tensor::backend::Backend> {
    layers: Vec<Linear<B>>,
    output: Linear<B>,
}

impl<B: burn::tensor::backend::Backend> SimpleQNetwork<B> {
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

impl<B: burn::tensor::backend::Backend> SimpleQNetwork<B> {
    fn forward_inner(&self, input: burn::tensor::Tensor<B, 2>) -> burn::tensor::Tensor<B, 2> {
        let mut x = input;
        for layer in &self.layers {
            x = burn::tensor::activation::relu(layer.forward(x));
        }
        self.output.forward(x)
    }
}

impl<B: AutodiffBackend> QNetwork<B> for SimpleQNetwork<B> {
    fn forward_q(&self, states: burn::tensor::Tensor<B, 2>) -> burn::tensor::Tensor<B, 2> {
        self.forward_inner(states)
    }
}

/// Create synthetic transitions for benchmarking
fn create_synthetic_transitions(
    num_transitions: usize,
    state_dim: usize,
    action_dim: usize,
) -> Vec<Transition> {
    let mut rng = rand::rng();

    (0..num_transitions)
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

/// Benchmark an algorithm by running training steps
fn benchmark_algorithm(
    name: &str,
    mut train_fn: impl FnMut() -> Option<f32>,
    num_iterations: usize,
) -> (f32, f64) {
    let start = Instant::now();
    let mut total_loss = 0.0;
    let mut valid_steps = 0;

    for _ in 0..num_iterations {
        if let Some(loss) = train_fn() {
            total_loss += loss;
            valid_steps += 1;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let avg_loss = if valid_steps > 0 {
        total_loss / valid_steps as f32
    } else {
        f32::NAN
    };

    println!("  {} - Avg loss: {:.4}, Time: {:.2}s", name, avg_loss, elapsed);
    (avg_loss, elapsed)
}

fn main() {
    println!("========================================");
    println!("  RL Cache Algorithm Comparison");
    println!("========================================\n");

    let state_dim = 32;
    let action_dim = 10;
    let num_transitions = 10000;
    let training_iterations = 100;
    let batch_size = 256;

    // Create synthetic workload (same for all algorithms)
    println!("Creating synthetic workload...");
    let transitions = create_synthetic_transitions(num_transitions, state_dim, action_dim);
    println!("  {} transitions created\n", transitions.len());

    // Device for computation
    let device = &<TestBackend as burn::tensor::backend::Backend>::Device::default();

    // Results table
    let mut results = Vec::new();

    // ============ DQN (Cold-RL style) ============
    {
        println!("Running DQN (Cold-RL style)...");
        let config = DQNTrainerConfig::default()
            .with_batch_size(batch_size)
            .with_buffer_capacity(10000)
            .with_loss_sync_freq(500);
        
        let q_network = SimpleQNetwork::<TestBackend>::new(state_dim, action_dim, device);
        let mut trainer = DQNTrainer::<TestBackend, SimpleQNetwork<TestBackend>>::new(
            q_network, state_dim, config, device.clone()
        ).expect("Failed to create DQN trainer");

        // Fill buffer - convert transitions to GPU tensors
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
            trainer.buffer.push(&state_tensor, t.action, t.reward, &next_state_tensor, t.done);
        }

        let (avg_loss, time) = benchmark_algorithm("DQN", || {
            trainer.train_step()
        }, training_iterations);

        results.push(("DQN (Cold-RL style)", avg_loss, time));
        println!("  ✓ Complete\n");
    }

    // ============ PPO ============
    {
        println!("Running PPO...");
        let config = PpoTrainerConfig::default()
            .with_batch_size(batch_size)
            .with_buffer_capacity(10000);
        
        let mut trainer = PpoTrainer::<TestBackend>::new(
            state_dim, action_dim, config, device.clone()
        ).expect("Failed to create PPO trainer");

        // Fill buffer (PPO uses CpuRingBuffer)
        for t in &transitions {
            trainer.buffer.push(t.clone());
        }

        let (avg_loss, time) = benchmark_algorithm("PPO", || {
            trainer.train_step()
        }, training_iterations);

        results.push(("PPO", avg_loss, time));
        println!("  ✓ Complete\n");
    }

    // ============ Metis (Combined DQN + Bandit) ============
    {
        println!("Running Metis (Combined DQN + Bandit)...");
        let config = MetisTrainerConfig::default()
            .with_batch_size(batch_size)
            .with_buffer_capacity(10000);
        
        let model_config = CombinedModelConfig::new(
            state_dim,
            vec![64], // bandit hidden
            32,       // feature dim
            vec![64], // dqn hidden
            action_dim,
        );
        let model = CombinedModel::<TestBackend>::new(model_config, device);
        
        let mut trainer = MetisTrainer::<TestBackend>::new(
            model, state_dim, config, device.clone()
        ).expect("Failed to create Metis trainer");

        // Fill buffer - convert transitions to GPU tensors
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
            trainer.buffer.push(&state_tensor, t.action, t.reward, &next_state_tensor, t.done);
        }

        let (avg_loss, time) = benchmark_algorithm("Metis", || {
            trainer.train_step()
        }, training_iterations);

        results.push(("Metis (DQN + Bandit)", avg_loss, time));
        println!("  ✓ Complete\n");
    }

    // ============ Results Table ============
    println!("\n========================================");
    println!("  Results Summary");
    println!("========================================");
    println!("{:<25} | {:>12} | {:>10}", "Algorithm", "Avg Loss", "Time (s)");
    println!("{}", "-".repeat(55));

    for (name, loss, time) in &results {
        if loss.is_nan() {
            println!("{:<25} | {:>12} | {:>10.2}", name, "N/A", time);
        } else {
            println!("{:<25} | {:>12.4} | {:>10.2}", name, loss, time);
        }
    }

    println!("\nNote: Lower loss is better. Time shows training speed.");
    println!("DQN = Value-based learning");
    println!("PPO = Policy gradient (on-policy)");
    println!("Metis = Combined DQN + Contextual Bandit");
}
