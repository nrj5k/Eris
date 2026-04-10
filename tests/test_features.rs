use eris::features::{
    encode_state, hotness_score, AccessRecord, AccessTracker, BlobFeatures, HotnessConfig,
};
use eris::trace::{BlobData, IoOp};
use eris::TierConfig; // Old TierConfig from config_old

fn create_test_blob(offset_id: &str) -> BlobData {
    BlobData {
        offset_id: offset_id.into(),
        offset_score: 100.0,
        offset_access_frequency: 10,
        access_offset: Some(0.0),
        access_size: 1024.0,
        offset_size: 1024.0,
        is_sequence: true,
        first_seen: false,
        overwrite_amount: 0.5,
        recency: "100.0".into(),
        io_op: "read".into(),
    }
}

#[test]
fn test_io_op_from_str() {
    assert_eq!(IoOp::from("read"), IoOp::Read);
    assert_eq!(IoOp::from("Read"), IoOp::Read);
    assert_eq!(IoOp::from("READ"), IoOp::Read);
    assert_eq!(IoOp::from("write"), IoOp::Write);
    assert_eq!(IoOp::from("Write"), IoOp::Write);
    assert_eq!(IoOp::from("WRITE"), IoOp::Write);
    assert_eq!(IoOp::from("unknown"), IoOp::Read); // Default
}

#[test]
fn test_io_op_display() {
    assert_eq!(format!("{}", IoOp::Read), "read");
    assert_eq!(format!("{}", IoOp::Write), "write");
}

#[test]
fn test_blob_io_op_enum() {
    let mut blob = create_test_blob("test");
    assert_eq!(blob.io_op_enum(), IoOp::Read);

    blob.io_op = "write".into();
    assert_eq!(blob.io_op_enum(), IoOp::Write);
}

#[test]
fn test_access_tracker_new() {
    let tracker = AccessTracker::new(1000);
    assert!(tracker.is_empty());
    assert_eq!(tracker.len(), 0);
    assert_eq!(tracker.sequence_len(), 0);
}

#[test]
fn test_access_tracker_recency() {
    let mut tracker = AccessTracker::new(1000);

    tracker.record(AccessRecord {
        blob_id: "blob1".into(),
        timestamp_ms: 1000,
        access_type: IoOp::Read,
        size: 1024.0,
    });

    let recency = tracker.get_recency("blob1", 5000);
    approx::assert_relative_eq!(recency, 4000.0, epsilon = 1e-5);
}

#[test]
fn test_access_tracker_frequency() {
    let mut tracker = AccessTracker::new(1000);

    tracker.record(AccessRecord {
        blob_id: "blob1".into(),
        timestamp_ms: 1000,
        access_type: IoOp::Read,
        size: 1024.0,
    });
    tracker.record(AccessRecord {
        blob_id: "blob1".into(),
        timestamp_ms: 2000,
        access_type: IoOp::Write,
        size: 1024.0,
    });

    assert_eq!(tracker.get_frequency("blob1"), 2);
    assert_eq!(tracker.get_frequency("blob2"), 0); // Never accessed
}

#[test]
fn test_access_tracker_reuse_distance() {
    let mut tracker = AccessTracker::new(1000);

    tracker.record(AccessRecord {
        blob_id: "blob1".into(),
        timestamp_ms: 1000,
        access_type: IoOp::Read,
        size: 1024.0,
    });
    tracker.record(AccessRecord {
        blob_id: "blob2".into(),
        timestamp_ms: 2000,
        access_type: IoOp::Read,
        size: 1024.0,
    });
    tracker.record(AccessRecord {
        blob_id: "blob1".into(),
        timestamp_ms: 3000,
        access_type: IoOp::Read,
        size: 1024.0,
    });

    // Sequence: ["blob1", "blob2", "blob1"]
    // Reuse distance is the position from the end where last access occurred
    // blob1 was accessed at positions 0 and 2, most recent is at position 0 from back
    // blob2 was accessed at position 1, which is position 1 from back
    assert_eq!(tracker.get_reuse_distance("blob1"), Some(0));
    assert_eq!(tracker.get_reuse_distance("blob2"), Some(1));
    assert_eq!(tracker.get_reuse_distance("never_seen"), None);
}

#[test]
fn test_access_tracker_sliding_window() {
    let mut tracker = AccessTracker::new(3);

    for i in 0..5 {
        tracker.record(AccessRecord {
            blob_id: format!("blob{}", i),
            timestamp_ms: i as u64 * 1000,
            access_type: IoOp::Read,
            size: 1024.0,
        });
    }

    // Should only keep last 3 in sliding window
    assert_eq!(tracker.len(), 3);

    // But sequence should have all 5
    assert_eq!(tracker.sequence_len(), 5);

    // Access counts should still be accurate
    assert_eq!(tracker.get_frequency("blob0"), 1);
    assert_eq!(tracker.get_frequency("blob4"), 1);
}

#[test]
fn test_blob_features_extraction() {
    let mut tracker = AccessTracker::new(1000);
    tracker.record(AccessRecord {
        blob_id: "test".into(),
        timestamp_ms: 1000,
        access_type: IoOp::Read,
        size: 1024.0,
    });

    let blob = create_test_blob("test");
    let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);

    assert!(features.recency >= 0.0 && features.recency <= 1.0);
    assert!(features.frequency >= 0.0 && features.frequency <= 1.0);
    approx::assert_relative_eq!(features.is_sequential, 1.0, epsilon = 1e-5);
    approx::assert_relative_eq!(features.last_access_type, 0.0, epsilon = 1e-5); // read
    approx::assert_relative_eq!(features.overwrite_amount, 0.5, epsilon = 1e-5);

    let vec = features.to_vec();
    assert_eq!(vec.len(), 32, "Features must be padded to warp size 32");
}

#[test]
fn test_blob_features_write_type() {
    let mut tracker = AccessTracker::new(1000);
    let mut blob = create_test_blob("test");
    blob.io_op = "write".into();

    let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
    approx::assert_relative_eq!(features.last_access_type, 1.0, epsilon = 1e-5);
}

#[test]
fn test_blob_features_non_sequential() {
    let mut tracker = AccessTracker::new(1000);
    let mut blob = create_test_blob("test");
    blob.is_sequence = false;

    let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
    approx::assert_relative_eq!(features.is_sequential, 0.0, epsilon = 1e-5);
}

#[test]
fn test_blob_features_recency_normalization() {
    let mut tracker = AccessTracker::new(1000);
    tracker.record(AccessRecord {
        blob_id: "test".into(),
        timestamp_ms: 1000,
        access_type: IoOp::Read,
        size: 1024.0,
    });

    let blob = create_test_blob("test");

    // Test with current time = 1000 + 3,600,000 (exactly 1 hour)
    let features = BlobFeatures::extract(&blob, &tracker, 1000 + 3_600_000, 10000.0, 100);
    approx::assert_relative_eq!(features.recency, 1.0, epsilon = 1e-5);

    // Test with current time = 1000 + 1,800,000 (30 minutes)
    let features = BlobFeatures::extract(&blob, &tracker, 1000 + 1_800_000, 10000.0, 100);
    approx::assert_relative_eq!(features.recency, 0.5, epsilon = 1e-5);
}

#[test]
fn test_blob_features_frequency_normalization() {
    let mut tracker = AccessTracker::new(1000);

    // Record 50 accesses
    for i in 0..50 {
        tracker.record(AccessRecord {
            blob_id: "test".into(),
            timestamp_ms: i as u64 * 100,
            access_type: IoOp::Read,
            size: 1024.0,
        });
    }

    let blob = create_test_blob("test");
    let features = BlobFeatures::extract(&blob, &tracker, 50000, 10000.0, 100);

    // frequency = 50 / 100 = 0.5
    approx::assert_relative_eq!(features.frequency, 0.5, epsilon = 1e-5);
}

#[test]
fn test_blob_features_access_intervals() {
    let mut tracker = AccessTracker::new(1000);

    // Record accesses at 1000, 2000, 4000 ms
    tracker.record(AccessRecord {
        blob_id: "test".into(),
        timestamp_ms: 1000,
        access_type: IoOp::Read,
        size: 1024.0,
    });
    tracker.record(AccessRecord {
        blob_id: "test".into(),
        timestamp_ms: 2000,
        access_type: IoOp::Read,
        size: 1024.0,
    });
    tracker.record(AccessRecord {
        blob_id: "test".into(),
        timestamp_ms: 4000,
        access_type: IoOp::Read,
        size: 1024.0,
    });

    let blob = create_test_blob("test");
    let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);

    // Intervals: 1000, 2000
    // Mean: 1500, normalized: 1500 / 3,600,000
    let expected_mean_normalized = 1500.0 / 3_600_000.0;
    approx::assert_relative_eq!(
        features.mean_interval as f64,
        expected_mean_normalized,
        epsilon = 1e-5
    );
    approx::assert_relative_eq!(
        features.mean_interval,
        expected_mean_normalized as f32,
        epsilon = 1e-5
    );

    // Std: sqrt(((1000-1500)^2 + (2000-1500)^2) / 1) = sqrt(500000) ≈ 707.107
    // Normalized: 707.107 / 3,600,000
    let expected_std = (500_000.0_f64).sqrt();
    let expected_std_normalized = expected_std / 3_600_000.0;
    approx::assert_relative_eq!(
        features.std_interval as f64,
        expected_std_normalized,
        epsilon = 1e-3
    );
}

#[test]
fn test_blob_features_reuse_distance_normalization() {
    let mut tracker = AccessTracker::new(1000);

    // Create 100 accesses to different blobs
    for i in 0..100 {
        tracker.record(AccessRecord {
            blob_id: format!("blob{}", i),
            timestamp_ms: i as u64 * 100,
            access_type: IoOp::Read,
            size: 1024.0,
        });
    }

    // Access blob0 again at position 100
    tracker.record(AccessRecord {
        blob_id: "blob0".into(),
        timestamp_ms: 10000,
        access_type: IoOp::Read,
        size: 1024.0,
    });

    let blob = create_test_blob("blob0");
    let features = BlobFeatures::extract(&blob, &tracker, 20000, 10000.0, 100);

    // Reuse distance = 0 (just accessed), so feature should be near 0
    approx::assert_relative_eq!(features.reuse_distance, 0.0, epsilon = 1e-5);
}

#[test]
fn test_blob_features_size_normalization() {
    let mut tracker = AccessTracker::new(1000);
    let mut blob = create_test_blob("test");
    blob.offset_size = 5000.0; // Half of max_size

    let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
    approx::assert_relative_eq!(features.size, 0.5, epsilon = 1e-5);

    // Test size > max_size
    blob.offset_size = 20000.0;
    let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
    approx::assert_relative_eq!(features.size, 1.0, epsilon = 1e-5); // Clamped to 1.0
}

#[test]
fn test_blob_features_overwrite_clamping() {
    let mut tracker = AccessTracker::new(1000);
    let mut blob = create_test_blob("test");

    // Test value > 1
    blob.overwrite_amount = 1.5;
    let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
    approx::assert_relative_eq!(features.overwrite_amount, 1.0, epsilon = 1e-5);

    // Test value < 0
    blob.overwrite_amount = -0.5;
    let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
    approx::assert_relative_eq!(features.overwrite_amount, 0.0, epsilon = 1e-5);
}

#[test]
fn test_blob_features_next_access_pred() {
    let mut tracker = AccessTracker::new(1000);
    tracker.record(AccessRecord {
        blob_id: "test".into(),
        timestamp_ms: 1000,
        access_type: IoOp::Read,
        size: 1024.0,
    });

    let blob = create_test_blob("test");
    let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);

    // next_access_pred = 1 - recency
    approx::assert_relative_eq!(
        features.next_access_pred,
        1.0 - features.recency,
        epsilon = 1e-5
    );
}

#[test]
fn test_state_encoding_dimensions() {
    let tier_sizes = vec![400.0, 1000.0, 2000.0, 10000.0, 50000.0];
    let features = BlobFeatures {
        recency: 0.1,
        frequency: 0.5,
        mean_interval: 100.0,
        std_interval: 50.0,
        is_sequential: 1.0,
        reuse_distance: 0.2,
        last_access_type: 0.0,
        size: 0.3,
        next_access_pred: 0.9,
        overwrite_amount: 0.4,
    };
    let tier_configs = create_test_tier_configs();

    let state = encode_state(&tier_sizes, &features, &tier_configs);

    assert_eq!(state.len(), 32, "State must be padded to warp size 32"); // 5 tier sizes + 10 features + 17 padding
}

#[test]
fn test_state_encoding_tier_normalization() {
    let tier_sizes = vec![400.0, 1000.0, 2000.0, 10000.0, 50000.0];
    let features = BlobFeatures {
        recency: 0.1,
        frequency: 0.5,
        mean_interval: 100.0,
        std_interval: 50.0,
        is_sequential: 1.0,
        reuse_distance: 0.2,
        last_access_type: 0.0,
        size: 0.3,
        next_access_pred: 0.9,
        overwrite_amount: 0.4,
    };
    let tier_configs = create_test_tier_configs();

    let state = encode_state(&tier_sizes, &features, &tier_configs);

    // Check tier normalization
    approx::assert_relative_eq!(state[0], 0.5, epsilon = 1e-5); // 400/800
    approx::assert_relative_eq!(state[1], 0.5, epsilon = 1e-5); // 1000/2000
    approx::assert_relative_eq!(state[2], 0.5, epsilon = 1e-5); // 2000/4000
    approx::assert_relative_eq!(state[3], 0.5, epsilon = 1e-5); // 10000/20000

    // Check that state[4] is approximately 0.05 (50000/999999)
    let expected = 50000.0 / 999999.0;
    approx::assert_relative_eq!(state[4] as f64, expected, epsilon = 1e-5);
}

#[test]
fn test_state_encoding_features_copy() {
    let tier_sizes = vec![400.0, 1000.0, 2000.0, 10000.0, 50000.0];
    let features = BlobFeatures {
        recency: 0.1,
        frequency: 0.5,
        mean_interval: 100.0,
        std_interval: 50.0,
        is_sequential: 1.0,
        reuse_distance: 0.2,
        last_access_type: 0.0,
        size: 0.3,
        next_access_pred: 0.9,
        overwrite_amount: 0.4,
    };
    let tier_configs = create_test_tier_configs();

    let state = encode_state(&tier_sizes, &features, &tier_configs);

    // Check features are copied correctly (starting at index 5)
    approx::assert_relative_eq!(state[5], features.recency, epsilon = 1e-5);
    approx::assert_relative_eq!(state[6], features.frequency, epsilon = 1e-5);
    approx::assert_relative_eq!(state[7], features.mean_interval, epsilon = 1e-5);
    approx::assert_relative_eq!(state[8], features.std_interval, epsilon = 1e-5);
    approx::assert_relative_eq!(state[9], features.is_sequential, epsilon = 1e-5);
    approx::assert_relative_eq!(state[10], features.reuse_distance, epsilon = 1e-5);
    approx::assert_relative_eq!(state[11], features.last_access_type, epsilon = 1e-5);
    approx::assert_relative_eq!(state[12], features.size, epsilon = 1e-5);
    approx::assert_relative_eq!(state[13], features.next_access_pred, epsilon = 1e-5);
    approx::assert_relative_eq!(state[14], features.overwrite_amount, epsilon = 1e-5);
}

#[test]
fn test_hotness_score_calculation() {
    let config = HotnessConfig::default();

    // Test with all default weights
    let score = hotness_score(100.0, true, 0.5, 0.2, &config);

    // 100.0 * 0.4 + 1.0 * 0.2 + 0.5 * 0.3 + 0.2 * (-0.1)
    // = 40.0 + 0.2 + 0.15 - 0.02 = 40.33
    approx::assert_relative_eq!(score, 40.33, epsilon = 1e-5);
}

#[test]
fn test_hotness_score_non_sequential() {
    let config = HotnessConfig::default();

    let score = hotness_score(100.0, false, 0.5, 0.2, &config);

    // 100.0 * 0.4 + 0.0 * 0.2 + 0.5 * 0.3 + 0.2 * (-0.1)
    // = 40.0 + 0.0 + 0.15 - 0.02 = 40.13
    approx::assert_relative_eq!(score, 40.13, epsilon = 1e-5);
}

#[test]
fn test_hotness_score_zero() {
    let config = HotnessConfig::default();

    let score = hotness_score(0.0, false, 0.0, 0.0, &config);
    approx::assert_relative_eq!(score, 0.0, epsilon = 1e-5);
}

#[test]
fn test_access_tracker_get_blob_history() {
    let mut tracker = AccessTracker::new(1000);

    tracker.record(AccessRecord {
        blob_id: "blob1".into(),
        timestamp_ms: 1000,
        access_type: IoOp::Read,
        size: 1024.0,
    });
    tracker.record(AccessRecord {
        blob_id: "blob2".into(),
        timestamp_ms: 2000,
        access_type: IoOp::Read,
        size: 1024.0,
    });
    tracker.record(AccessRecord {
        blob_id: "blob1".into(),
        timestamp_ms: 3000,
        access_type: IoOp::Write,
        size: 2048.0,
    });

    let history = tracker.get_blob_history("blob1", 1);
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].timestamp_ms, 3000);

    let history_all = tracker.get_blob_history("blob1", 10);
    assert_eq!(history_all.len(), 2);
}

#[test]
fn test_access_tracker_clear() {
    let mut tracker = AccessTracker::new(1000);

    tracker.record(AccessRecord {
        blob_id: "blob1".into(),
        timestamp_ms: 1000,
        access_type: IoOp::Read,
        size: 1024.0,
    });

    assert!(!tracker.is_empty());

    tracker.clear();

    assert!(tracker.is_empty());
    assert_eq!(tracker.get_frequency("blob1"), 0);
    assert!(tracker.get_recency("blob1", 5000).is_infinite());
}

fn create_test_tier_configs() -> Vec<TierConfig> {
    vec![
        TierConfig {
            name: "Memory".into(),
            tier_id: 0,
            capacity: 800.0,
            access_latency: 0.01,
            description: String::new(),
        },
        TierConfig {
            name: "NVMe".into(),
            tier_id: 1,
            capacity: 2000.0,
            access_latency: 1.0,
            description: String::new(),
        },
        TierConfig {
            name: "SSD".into(),
            tier_id: 2,
            capacity: 4000.0,
            access_latency: 10.0,
            description: String::new(),
        },
        TierConfig {
            name: "HDD".into(),
            tier_id: 3,
            capacity: 20000.0,
            access_latency: 10000.0,
            description: String::new(),
        },
        TierConfig {
            name: "Tapes".into(),
            tier_id: 4,
            capacity: 999999.0,
            access_latency: 1000000.0,
            description: String::new(),
        },
    ]
}
