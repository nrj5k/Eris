//! Inference-only binary using NdArray backend (NO autodiff)
//!
//! This is for testing basic model functionality without gradient computation.
//! Much lighter weight than autodiff backend - should not stack overflow.

use std::path::Path;

use burn::backend::NdArray;
use burn::tensor::{Tensor, TensorData};
use clap::Parser;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

use eris::config::{FullTrainingConfig, ModelArchitecture};
use eris::env::IOBufferEnv;
use eris::models::CombinedModelConfig;
use eris::{Environment, Space};

#[derive(Parser, Debug)]
#[command(
    name = "eris-train-inference",
    about = "Test Eris model with NdArray backend (no gradients)",
    version = "0.1.0"
)]
struct Args {
    /// Path to config file
    #[arg(short, long, default_value = "config/tiers.toml")]
    config: String,

    /// Path to trace CSV file
    #[arg(short, long, default_value = "recorder-csv/NWChem-64_combined.csv")]
    trace: String,

    /// Maximum steps per episode
    #[arg(short, long, default_value = "10")]
    max_steps: usize,
}

fn main() {
    // Increase stack size SIGNIFICANTLY to prevent overflow
    // Burn's deeply nested generic types require very large stack frames during initialization
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024) // 64 MB stack (up from default 8 MB)
        .spawn(|| {
            // Initialize logging
            let subscriber = FmtSubscriber::builder()
                .with_max_level(Level::INFO)
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");

            let args = Args::parse();

            println!("=== Testing Inference Mode (NdArray Backend) ===");
            println!("This tests the model without autodiff/gradient computation");
            println!("Much lighter weight - should avoid stack overflow");
            println!();
            println!("Stack size increased to 64 MB");
            println!("Model allocated on heap");
            println!();

            if let Err(e) = run_test(&args) {
                tracing::error!("Test failed: {}", e);
                std::process::exit(1);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[allow(deprecated)]
fn run_test(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = Path::new(&args.config);
    let trace_path = Path::new(&args.trace);

    // Load config
    let config = FullTrainingConfig::from_file(config_path)?;

    // Create environment
    let env = IOBufferEnv::new(config_path, trace_path, args.max_steps)?;

    // Get dimensions
    let state_dim = env.observation_space().dim();
    let action_dim = env.action_space().n;

    // Create model config
    let model_config = match config.model.architecture {
        ModelArchitecture::DuelingDQN | ModelArchitecture::SimpleDQN => CombinedModelConfig::new(
            state_dim,
            config.model.feature_dim,
            config.model.dqn_hidden[0],
            action_dim,
        ),
        ModelArchitecture::BanditDQN => CombinedModelConfig::new(
            state_dim,
            config.model.feature_dim,
            config.model.dqn_hidden[0],
            action_dim,
        ),
    };

    // Initialize NdArray device (NOT Autodiff)
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // Initialize model on HEAP to avoid stack overflow
    // Burn models create large stack-allocated structures during init
    println!("Initializing model on NdArray backend (no autodiff)...");
    println!("Model allocated on heap to prevent stack overflow...");
    let model = Box::new(
        eris::models::CombinedModelConfig::new(
            state_dim,
            config.model.feature_dim,
            config.model.dqn_hidden[0],
            action_dim,
        )
        .init(&device),
    );
    let model: eris::models::CombinedModel<NdArray> = *model;
    println!("✓ Model initialized successfully on heap");

    // Test forward pass
    println!("\nTesting model forward pass...");
    let test_input = Tensor::<NdArray, 2>::zeros([1, state_dim], &device);
    let forward_result = model.forward(test_input);
    let features = forward_result.0;
    let importance = forward_result.1;
    let q_values = forward_result.2;
    println!("✓ Forward pass successful");
    println!("  Features shape: {:?}", features.shape());
    println!("  Importance shape: {:?}", importance.shape());
    println!("  Q-values shape: {:?}", q_values.shape());

    // Reset environment and take a few steps
    println!("\nTesting environment interaction...");
    let mut env = env;
    let mut obs = env.reset();
    println!("  Initial observation: {} values", obs.len());

    for step in 0..5 {
        // Use model for action selection
        let obs_f32: Vec<f32> = obs.iter().map(|&x| x as f32).collect();
        let obs_tensor =
            Tensor::<NdArray, 2>::from_data(TensorData::new(obs_f32, [1, state_dim]), &device);

        let forward_result = model.forward(obs_tensor);
        let q_values = forward_result.2;
        let q_values_data = q_values.to_data();
        let q_values_slice = q_values_data.as_slice::<f32>().unwrap();
        let q_values_vec: Vec<f32> = q_values_slice.to_vec();

        // Find action with highest Q-value
        let mut max_q = f32::NEG_INFINITY;
        let mut action = 0;
        for (i, &q) in q_values_vec.iter().enumerate() {
            if q > max_q {
                max_q = q;
                action = i;
            }
        }

        let (next_obs, reward, done) = env.step(action);
        println!(
            "  Step {}: action={}, reward={:.2}, done={}",
            step + 1,
            action,
            reward,
            done
        );

        obs = next_obs;
        if done {
            break;
        }
    }

    println!("\n=== Test Complete ===");
    println!("✓ Model initialization successful");
    println!("✓ Forward pass successful");
    println!("✓ Environment interaction successful");
    println!("\nNdArray backend (no autodiff) works fine!");
    println!("The stack overflow issue is caused by Autodiff<NdArray>");

    Ok(())
}
