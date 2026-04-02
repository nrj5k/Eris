use eris::trace::{BlobData, TraceReader};
use std::path::Path;

#[test]
fn test_csv_parsing() {
    let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");
    if !trace_path.exists() {
        eprintln!("Skipping test: trace file not found");
        return;
    }

    let reader = TraceReader::from_csv(trace_path).unwrap();
    assert!(reader.len() > 0);
}

#[test]
fn test_blob_data_recency() {
    let mut blob = BlobData {
        offset_id: "test".into(),
        offset_score: 32.0,
        offset_access_frequency: 64,
        access_offset: None,
        access_size: 143_360.0,
        offset_size: 143_360.0,
        is_sequence: false,
        first_seen: false,
        overwrite_amount: 0.0,
        recency: "7.999999999980245e-06".into(),
        io_op: "read".into(),
    };

    assert_eq!(blob.recency_ms(), Some(7.999999999980245e-06));

    blob.recency = "inf".into();
    assert_eq!(blob.recency_ms(), None);
}

#[test]
fn test_blob_is_read() {
    let mut blob = BlobData {
        offset_id: "test".into(),
        offset_score: 32.0,
        offset_access_frequency: 64,
        access_offset: None,
        access_size: 143_360.0,
        offset_size: 143_360.0,
        is_sequence: false,
        first_seen: false,
        overwrite_amount: 0.0,
        recency: "inf".into(),
        io_op: "read".into(),
    };

    assert!(blob.is_read());

    blob.io_op = "write".into();
    assert!(!blob.is_read());
}

#[test]
fn test_default_config() {
    let config = eris::config::Config::default_tiers();
    assert_eq!(config.tier.len(), 5);
    assert_eq!(config.tier[0].name, "Memory");
    assert_eq!(config.tier[4].name, "Tapes");
}

#[test]
fn test_config_from_file() {
    use std::path::Path;

    let config_path = Path::new("config/tiers.toml");
    if !config_path.exists() {
        eprintln!("Skipping test: config file not found");
        return;
    }

    let config = eris::config::Config::from_file(config_path).unwrap();
    assert_eq!(config.tier.len(), 5);
    assert_eq!(config.tier[0].name, "Memory");
}
