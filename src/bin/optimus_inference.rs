//! Run Optimus iTransformer inference for cache prefetching
//!
//! Usage:
//!   cargo run --bin optimus_inference --features optimus -- --checkpoint model.mpk --steps 1000

#[cfg(feature = "optimus")]
mod optimus_impl {
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::backend::Backend;
    use clap::Parser;
    use std::path::Path;

    use burnme_rly::models::optimus::{
        device_name, format_inference_summary, generate_synthetic_history, history_to_tensor,
        OptimusConfig, OptimusPolicy,
    };
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
        // Note: No --device flag needed! Device is auto-detected from Burn backend.
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

        // Use Display trait (DRY!)
        println!("\nConfig:\n{}", config);

        // Create device - device selection is automatic based on Burn backend
        let device = <B as Backend>::Device::default();

        // Log device info
        println!("\nDevice: {}", device_name::<B>(&device));

        // Create policy - no bridge_device needed, auto-detected from Burn device
        let action_dim = 10;
        let mut policy = OptimusPolicy::<B>::new(config.clone(), device, action_dim);

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
            // Generate synthetic history using library utility (DRY!)
            let history = generate_synthetic_history(config.num_variates, config.lookback_len);

            // Convert to tensor using library utility (DRY!)
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

        // Use library utility for summary (DRY!)
        println!(
            "\n{}",
            format_inference_summary(total_predictions, successful_predictions)
        );

        // Save final checkpoint
        let final_checkpoint = args.checkpoint.to_string();
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
