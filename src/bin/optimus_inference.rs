//! Run Optimus iTransformer inference for cache prefetching
//!
//! Usage:
//!   cargo run --bin optimus_inference --features optimus -- --checkpoint model.mpk --steps 1000

#[cfg(feature = "optimus")]
mod optimus_impl {
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::{backend::Backend, Shape, Tensor};
    use clap::Parser;
    use std::path::Path;

    use burnme_rly::models::optimus::{OptimusConfig, OptimusPolicy, BridgeDevice, parse_device_str};
    use burnme_rly::traits::GpuTrainable;

    type B = Autodiff<NdArray>;

    #[derive(Parser, Debug)]
    #[command(name = "optimus_inference")]
    #[command(about = "Run Optimus iTransformer inference")]
    struct Args {
        /// Checkpoint path
        #[arg(short, long)]
        checkpoint: String,

        /// Configuration file
        #[arg(long, default_value = "config/optimus.toml")]
        config: String,

        /// Number of inference steps
        #[arg(long, default_value = "1000")]
        steps: usize,

        /// Batch size for inference
        #[arg(long, default_value = "1")]
        batch_size: usize,

        /// Trace file for evaluation
        #[arg(long)]
        trace: Option<String>,

        /// Output predictions to file
        #[arg(long)]
        output: Option<String>,

        /// Device for computation (cpu, cuda, cuda:0, cuda:1)
        #[arg(long, default_value = "auto")]
        device: String,
    }

    /// Generate synthetic cache access history
    fn generate_synthetic_history(num_variates: usize, lookback_len: usize) -> Vec<Vec<f32>> {
        (0..num_variates)
            .map(|_| (0..lookback_len).map(|_| rand::random::<f32>()).collect())
            .collect()
    }

    /// Format history into [batch, lookback_len, num_variates] tensor shape (iTransformer format)
    fn history_to_tensor<B: Backend>(history: &[Vec<f32>], device: &B::Device) -> Tensor<B, 3> {
        let num_variates = history.len();
        let lookback_len = history[0].len();

        // iTransformer expects [batch, lookback_len, num_variates]
        // Transpose from [num_variates, lookback_len] to [lookback_len, num_variates]
        let mut flattened: Vec<f32> = Vec::with_capacity(num_variates * lookback_len);
        for t in 0..lookback_len {
            for v in 0..num_variates {
                flattened.push(history[v][t]);
            }
        }

        // Create 1D tensor and reshape to 3D: [batch=1, lookback_len, num_variates]
        let tensor_1d = Tensor::<B, 1>::from_floats(flattened.as_slice(), device);
        tensor_1d.reshape(Shape::new([1, lookback_len, num_variates]))
    }

    pub fn main() {
        let args = Args::parse();

        println!("=== Optimus iTransformer Inference ===");
        println!("Checkpoint: {}", args.checkpoint);
        println!("Steps: {}", args.steps);

        // Create config
        let config = OptimusConfig::new()
            .with_num_variates(128)
            .with_lookback_len(96)
            .with_pred_len(48);

        println!("\nConfig:");
        println!("  num_variates: {}", config.num_variates);
        println!("  lookback_len: {}", config.lookback_len);
        println!("  pred_len: {}", config.pred_len);

        // Parse device string
        let bridge_device = match args.device.as_str() {
            "auto" => BridgeDevice::auto(),
            _ => parse_device_str(&args.device)
                .unwrap_or_else(|| {
                    eprintln!("[WARN] Unknown device '{}', using auto", args.device);
                    BridgeDevice::auto()
                }),
        };

        println!("Using device: {:?}", bridge_device);

        // Create device
        let device = <B as Backend>::Device::default();

        // Create policy
        let action_dim = 10;
        let mut policy = OptimusPolicy::<B>::new(config.clone(), device.clone(), bridge_device, action_dim);

        // Load checkpoint if exists
        let checkpoint_path = Path::new(&args.checkpoint);
        if checkpoint_path.exists() {
            println!("\n[LOAD] Loading checkpoint from {:?}...", checkpoint_path);
            match policy.load_checkpoint(&args.checkpoint) {
                Ok(_) => println!("[LOAD] Checkpoint loaded successfully"),
                Err(e) => {
                    eprintln!("[WARN] Failed to load checkpoint: {}", e);
                    println!("[INFO] Using randomly initialized model");
                }
            }
        } else {
            println!("\n[WARN] Checkpoint not found: {}", args.checkpoint);
            println!("[INFO] Using randomly initialized model");
        }

        println!("\n[INFERENCE] Running {} steps...", args.steps);

        let mut total_predictions = 0usize;
        let mut successful_predictions = 0usize;

        for step in 0..args.steps {
            // Generate synthetic history
            let history = generate_synthetic_history(config.num_variates, config.lookback_len);

            // Convert to tensor
            let history_tensor = history_to_tensor::<B>(&history, &device);

            // Run prediction
            match policy.predict(&history_tensor) {
                Some(predictions) => {
                    successful_predictions += 1;

                    // predictions: [1, pred_len, num_variates]
                    if step % 100 == 0 {
                        let shape = predictions.shape();
                        println!(
                            "[Step {}] Predictions shape: [{}, {}, {}]",
                            step,
                            shape.dims::<3>()[0],
                            shape.dims::<3>()[1],
                            shape.dims::<3>()[2]
                        );
                    }
                }
                None => {
                    if step % 100 == 0 {
                        eprintln!("[Step {}] Prediction failed", step);
                    }
                }
            }

            total_predictions += 1;
        }

        println!("\n{}", "=".repeat(50));
        println!("[COMPLETE] Inference finished");
        println!("  Total steps: {}", total_predictions);
        println!(
            "  Successful: {} ({:.1}%)",
            successful_predictions,
            100.0 * successful_predictions as f64 / total_predictions as f64
        );
        println!("{}", "=".repeat(50));

        // Save final checkpoint
        let final_checkpoint = format!("{}", args.checkpoint);
        if let Err(e) = policy.save_checkpoint(&final_checkpoint) {
            eprintln!("[WARN] Failed to save final checkpoint: {}", e);
        } else {
            println!("\nCheckpoint saved: {}", final_checkpoint);
        }
    }
}

#[cfg(not(feature = "optimus"))]
fn main() {
    eprintln!("Error: Optimus feature not enabled.");
    eprintln!("Please build with: cargo run --bin optimus_inference --features optimus");
    std::process::exit(1);
}

#[cfg(feature = "optimus")]
fn main() {
    optimus_impl::main();
}
