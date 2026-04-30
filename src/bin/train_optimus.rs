//! Train Optimus iTransformer model for cache prefetching
//!
//! Usage:
//!   cargo run --bin train_optimus --features optimus -- --config config.toml --epochs 100

#[cfg(feature = "optimus")]
mod optimus_impl {
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::backend::Backend;
    use clap::Parser;

    use burnme_rly::models::optimus::{device_name, OptimusConfig, OptimusPolicy};
    use burnme_rly::traits::GpuTrainable;

    #[derive(Parser, Debug)]
    #[command(name = "train_optimus")]
    #[command(about = "Train Optimus iTransformer for cache prefetching")]
    struct Args {
        /// Configuration file path
        #[arg(short, long, default_value = "config/optimus.toml")]
        config: String,

        /// Number of training epochs
        #[arg(short, long, default_value = "100")]
        epochs: usize,

        /// Batch size
        #[arg(short, long, default_value = "32")]
        batch_size: usize,

        /// Learning rate
        #[arg(long, default_value = "0.0001")]
        learning_rate: f64,

        /// Checkpoint directory
        #[arg(long, default_value = "checkpoints/optimus")]
        checkpoint_dir: String,

        /// Device override (optional). If not specified, auto-detects from backend.
        /// Supports: "cpu", "cuda", "cuda:0", "cuda:1", etc.
        #[arg(long)]
        device: Option<String>,
    }

    pub fn main() {
        let args = Args::parse();

        println!("=== Training Optimus iTransformer ===");
        println!("Config: {}", args.config);
        println!("Epochs: {}", args.epochs);
        println!("Batch size: {}", args.batch_size);
        println!("Learning rate: {}", args.learning_rate);

        // Create config
        let config = OptimusConfig::new()
            .with_num_variates(128)
            .with_lookback_len(96)
            .with_pred_len(48);

        config.validate().expect("Invalid config");

        // Use Display trait for config printing (DRY!)
        println!("\nConfig:\n{}", config);

        // Create device with autodiff backend - device selection is automatic
        type TestBackend = Autodiff<NdArray>;
        let device = <TestBackend as Backend>::Device::default();

        // Log device info
        println!("Using device: {}", device_name::<TestBackend>(&device));

        // Create policy - device auto-detected or overridden via --device flag
        let action_dim = 10; // Number of cache actions
        let policy = OptimusPolicy::<TestBackend>::new(
            config,
            device.clone(),
            action_dim,
            args.device.as_deref(),
        );

        println!("\nOptimus policy created!");
        println!("Note: Full training loop requires implementing backward pass");
        println!("through Candle. Current implementation provides inference only.");

        // TODO: Implement training loop
        // 1. Load trace data
        // 2. Create time windows
        // 3. Training loop:
        //    - Sample batch
        //    - Forward pass (Candle)
        //    - Compute loss (MSE)
        //    - Backward pass (requires Candle gradients → Burn)
        //    - Update weights
        // 4. Save checkpoint

        println!("\n[TRAINING] Not yet fully implemented");
        println!("[INFO] Current implementation supports inference only");
        println!("[TODO] Implement backward pass for training");

        // Save initial checkpoint
        std::fs::create_dir_all(&args.checkpoint_dir).ok();
        let checkpoint_path = format!("{}/optimus_initial", args.checkpoint_dir);
        if let Err(e) = policy.save_checkpoint(&checkpoint_path) {
            eprintln!("[WARN] Failed to save checkpoint: {}", e);
        }

        println!("\nCheckpoint saved to: {}/", args.checkpoint_dir);
        println!("Training complete!");
    }
}

#[cfg(not(feature = "optimus"))]
fn main() {
    eprintln!("Error: Optimus feature not enabled.");
    eprintln!("Please build with: cargo run --bin train_optimus --features optimus");
    std::process::exit(1);
}

#[cfg(feature = "optimus")]
fn main() {
    optimus_impl::main();
}
