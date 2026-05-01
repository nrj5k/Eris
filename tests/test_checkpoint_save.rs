//! Test checkpoint saving for both policy and target models

use burn::backend::{Autodiff, NdArray};
use eris::model::ErisDefaults;
use eris::training::{CombinedAgent, TrainingConfig};
use tempfile::TempDir;

#[test]
fn test_checkpoint_saves_both_models() {
    // Initialize logging for debugging
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    // Create temp directory for checkpoints
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let checkpoint_path = temp_dir.path().join("model");

    println!("Temp directory: {:?}", temp_dir.path());
    println!("Checkpoint path: {:?}", checkpoint_path);

    // Setup
    type Backend = Autodiff<NdArray>;
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let training_config = TrainingConfig::default();

    // Use ErisDefaults for model config
    let model_config = ErisDefaults::storage_tier_model(10, 3);

    // Create agent
    let agent = CombinedAgent::<Backend>::new(training_config, model_config, device);

    // Save checkpoint with detailed error handling
    match agent.save_checkpoint(&checkpoint_path, 0, 0.0) {
        Ok(_) => println!("✓ Checkpoint saved successfully"),
        Err(e) => {
            println!("✗ Checkpoint save failed: {}", e);

            // List all files in temp dir to see what WAS created
            println!("\nFiles actually created:");
            for entry in std::fs::read_dir(temp_dir.path()).expect("Failed to read dir") {
                let entry = entry.expect("Failed to read entry");
                println!("  {:?}", entry.path());
            }
            panic!("Failed to save checkpoint: {}", e);
        }
    }

    // Verify both .mpk files exist
    let policy_mp = temp_dir.path().join("model-0.mpk");
    let target_mp = temp_dir.path().join("model_target-0.mpk"); // Changed from model.target to model_target
    let policy_json = temp_dir.path().join("model-0.json");
    let target_json = temp_dir.path().join("model_target-0.json"); // Changed from model.target to model_target

    println!("\nExpected files:");
    println!("  Policy MPK: {:?}", policy_mp);
    println!("  Target MPK: {:?}", target_mp);
    println!("  Policy JSON: {:?}", policy_json);
    println!("  Target JSON: {:?}", target_json);

    // List all files in temp dir
    println!("\nActual files in temp dir:");
    for entry in std::fs::read_dir(temp_dir.path()).expect("Failed to read dir") {
        let entry = entry.expect("Failed to read entry");
        println!("  {:?}", entry.path());
    }

    assert!(
        policy_mp.exists(),
        "Policy checkpoint not found: {:?}",
        policy_mp
    );
    assert!(
        target_mp.exists(),
        "Target checkpoint not found: {:?}",
        target_mp
    );
    assert!(
        policy_json.exists(),
        "Policy metadata not found: {:?}",
        policy_json
    );
    assert!(
        target_json.exists(),
        "Target metadata not found: {:?}",
        target_json
    );
}
