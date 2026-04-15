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

use eris::config::FullTrainingConfig;
use eris::env::IOBufferEnv;
use eris::{Environment, Space};

#[derive(Clone, Debug, clap::ValueEnum)]
enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Trace file format
#[derive(Clone, Debug, clap::ValueEnum)]
enum TraceFormat {
    Autodetect,
    Recorder,
    Dftracer,
}

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

    /// Trace format: recorder (CSV), dftracer (pfw.gz), or autodetect
    #[arg(long, value_enum, default_value = "autodetect")]
    trace_format: TraceFormat,

    /// Maximum steps per episode
    #[arg(short, long, default_value = "10")]
    max_steps: usize,

    /// Log level for tracing output
    #[arg(long, value_enum, default_value = "info")]
    log_level: LogLevel,
}

fn main() {
    // Increase stack size SIGNIFICANTLY to prevent overflow
    // Burn's deeply nested generic types require very large stack frames during initialization
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024) // 64 MB stack (up from default 8 MB)
        .spawn(|| {
            let args = Args::parse();

            // Initialize logging with user-specified level
            let level = match args.log_level {
                LogLevel::Trace => Level::TRACE,
                LogLevel::Debug => Level::DEBUG,
                LogLevel::Info => Level::INFO,
                LogLevel::Warn => Level::WARN,
                LogLevel::Error => Level::ERROR,
            };
            let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
            tracing::subscriber::set_global_default(subscriber)
                .expect("Failed to set tracing subscriber");

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

fn to_trace_format(format: &TraceFormat) -> eris::TraceFormat {
    match format {
        TraceFormat::Autodetect => eris::TraceFormat::Autodetect,
        TraceFormat::Recorder => eris::TraceFormat::Recorder,
        TraceFormat::Dftracer => eris::TraceFormat::Dftracer,
    }
}

#[allow(deprecated)]
fn run_test(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let config_path = Path::new(&args.config);
    let trace_path = Path::new(&args.trace);

    // Load config
    let config = FullTrainingConfig::from_file(config_path)?;

    // Create environment
    let env = IOBufferEnv::new(
        config_path,
        trace_path,
        to_trace_format(&args.trace_format),
        args.max_steps,
        None,
        None,
    )?;

    // Get dimensions
    let state_dim = env.observation_space().dim();
    let action_dim = env.action_space().n;

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

    // Pre-allocate buffer to avoid allocation in loop
    let mut obs_buf = vec![0.0f32; state_dim];

    for step in 0..5 {
        // Use model for action selection
        // Reuse buffer: copy and convert in-place
        obs_buf
            .iter_mut()
            .zip(obs.iter())
            .for_each(|(dst, &src)| *dst = src as f32);

        // Create tensor from pre-allocated buffer (clone avoids growth/reallocation)
        let obs_tensor = Tensor::<NdArray, 2>::from_data(
            TensorData::new(obs_buf.clone(), [1, state_dim]),
            &device,
        );

        let forward_result = model.forward(obs_tensor);
        let q_values = forward_result.2;

        // Use tensor argmax (matches pattern in combined.rs:161 and train_model.rs:678)
        let action_tensor = q_values.argmax(1); // Returns [1, 1] tensor with index of max
        let action: usize = action_tensor
            .into_data()
            .convert::<i32>()
            .to_vec::<i32>()
            .expect("argmax conversion")[0] as usize;

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
