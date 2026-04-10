//! Tests for warp-aligned state dimensions (Phase 01: Shape Alignment)
//!
//! These tests verify that state vectors are padded to GPU warp size (32)
//! for optimal memory coalescing and 2-10x speedup on matrix operations.

use eris::config_old::TierConfig;
use eris::features::{aligned_state_dim, encode_state, pad_to_warp_size, BlobFeatures};

#[test]
fn test_state_dimension_is_warp_aligned() {
    let features = BlobFeatures {
        recency: 0.1,
        frequency: 0.5,
        mean_interval: 0.2,
        std_interval: 0.1,
        is_sequential: 1.0,
        reuse_distance: 0.3,
        last_access_type: 0.0,
        size: 0.4,
        next_access_pred: 0.9,
        overwrite_amount: 0.5,
    };
    let vec = features.to_vec();
    assert_eq!(vec.len(), 32, "State must be padded to warp size 32");
}

#[test]
fn test_encode_state_produces_32_dimensions() {
    let tier_sizes = vec![400.0, 1000.0, 2000.0, 10000.0, 50000.0];
    let features = BlobFeatures {
        recency: 0.1,
        frequency: 0.5,
        mean_interval: 0.2,
        std_interval: 0.1,
        is_sequential: 1.0,
        reuse_distance: 0.3,
        last_access_type: 0.0,
        size: 0.4,
        next_access_pred: 0.9,
        overwrite_amount: 0.5,
    };
    let tier_configs = vec![
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
    ];

    let state = encode_state(&tier_sizes, &features, &tier_configs);
    assert_eq!(
        state.len(),
        32,
        "encode_state must produce 32-dimensional output"
    );
}

#[test]
fn test_pad_to_warp_size_function() {
    // Test with less than warp size
    let small = vec![1.0, 2.0, 3.0];
    let padded = pad_to_warp_size(&small);
    assert_eq!(padded.len(), 32);
    assert_eq!(padded[0], 1.0);
    assert_eq!(padded[1], 2.0);
    assert_eq!(padded[2], 3.0);
    for i in 3..32 {
        assert_eq!(padded[i], 0.0);
    }

    // Test with exactly warp size (should not change)
    let exact = vec![1.0; 32];
    let padded_exact = pad_to_warp_size(&exact);
    assert_eq!(padded_exact.len(), 32);
    for val in &padded_exact {
        assert_eq!(*val, 1.0);
    }

    // Test with more than warp size (should not truncate)
    let large = vec![1.0; 50];
    let padded_large = pad_to_warp_size(&large);
    assert_eq!(padded_large.len(), 50);
}

#[test]
fn test_aligned_state_dim_constant() {
    assert_eq!(aligned_state_dim(), 32, "aligned_state_dim must return 32");
}

#[test]
fn test_feature_vector_padding_structure() {
    let features = BlobFeatures {
        recency: 0.1,
        frequency: 0.5,
        mean_interval: 0.2,
        std_interval: 0.1,
        is_sequential: 1.0,
        reuse_distance: 0.3,
        last_access_type: 0.0,
        size: 0.4,
        next_access_pred: 0.9,
        overwrite_amount: 0.5,
    };
    let vec = features.to_vec();

    // First 10 elements should be the actual features
    assert_eq!(vec[0], 0.1); // recency
    assert_eq!(vec[1], 0.5); // frequency
    assert_eq!(vec[2], 0.2); // mean_interval
    assert_eq!(vec[3], 0.1); // std_interval
    assert_eq!(vec[4], 1.0); // is_sequential
    assert_eq!(vec[5], 0.3); // reuse_distance
    assert_eq!(vec[6], 0.0); // last_access_type
    assert_eq!(vec[7], 0.4); // size
    assert_eq!(vec[8], 0.9); // next_access_pred
    assert_eq!(vec[9], 0.5); // overwrite_amount

    // Remaining 22 elements should be zeros (padding)
    for i in 10..32 {
        assert_eq!(vec[i], 0.0, "Padding at index {} should be 0.0", i);
    }
}

#[test]
fn test_state_encoding_padding_structure() {
    let tier_sizes = vec![400.0, 1000.0, 2000.0, 10000.0, 50000.0];
    let features = BlobFeatures {
        recency: 0.1,
        frequency: 0.5,
        mean_interval: 0.2,
        std_interval: 0.1,
        is_sequential: 1.0,
        reuse_distance: 0.3,
        last_access_type: 0.0,
        size: 0.4,
        next_access_pred: 0.9,
        overwrite_amount: 0.5,
    };
    let tier_configs = vec![
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
    ];

    let state = encode_state(&tier_sizes, &features, &tier_configs);

    // First 5 elements: tier utilizations
    approx::assert_relative_eq!(state[0], 0.5, epsilon = 1e-5); // 400/800
    approx::assert_relative_eq!(state[1], 0.5, epsilon = 1e-5); // 1000/2000
    approx::assert_relative_eq!(state[2], 0.5, epsilon = 1e-5); // 2000/4000
    approx::assert_relative_eq!(state[3], 0.5, epsilon = 1e-5); // 10000/20000
    approx::assert_relative_eq!(state[4], 0.050005, epsilon = 1e-5); // 50000/999999 ≈ 0.05

    // Next 10 elements: features
    assert_eq!(state[5], 0.1); // recency
    assert_eq!(state[6], 0.5); // frequency
    assert_eq!(state[7], 0.2); // mean_interval
    assert_eq!(state[8], 0.1); // std_interval
    assert_eq!(state[9], 1.0); // is_sequential
    assert_eq!(state[10], 0.3); // reuse_distance
    assert_eq!(state[11], 0.0); // last_access_type
    assert_eq!(state[12], 0.4); // size
    assert_eq!(state[13], 0.9); // next_access_pred
    assert_eq!(state[14], 0.5); // overwrite_amount

    // Remaining 17 elements: padding (zeros)
    for i in 15..32 {
        assert_eq!(state[i], 0.0, "Padding at index {} should be 0.0", i);
    }
}
