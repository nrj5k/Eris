use crate::config_old::TierConfig;
use crate::features::tracker::AccessTracker;
use crate::trace::BlobData;

/// GPU warp size for optimal memory coalescing and utilization
pub const WARP_SIZE: usize = 32;

/// Pad feature vector to warp-aligned size for optimal GPU performance
///
/// Current: 15 dimensions (5 tiers + 10 features)
/// Target: 32 dimensions (padded with zeros)
///
/// GPU warp size is 32, so dimensions should be multiples of 32
/// for optimal memory coalescing and utilization.
pub fn pad_to_warp_size(features: &[f32]) -> Vec<f32> {
    let mut padded = features.to_vec();
    if padded.len() < WARP_SIZE {
        padded.resize(WARP_SIZE, 0.0);
    }
    padded
}

/// Get the warp-aligned state dimension
pub const fn aligned_state_dim() -> usize {
    WARP_SIZE
}

/// 10-dimensional feature vector for a blob
#[derive(Debug, Clone)]
pub struct BlobFeatures {
    /// Time since last access (normalized to [0, 1])
    pub recency: f32,
    /// Access count (normalized to [0, 1])
    pub frequency: f32,
    /// Mean time between accesses (milliseconds)
    pub mean_interval: f32,
    /// Standard deviation of access intervals (milliseconds)
    pub std_interval: f32,
    /// 1.0 if sequential pattern, else 0.0
    pub is_sequential: f32,
    /// Position since last access (normalized to [0, 1])
    pub reuse_distance: f32,
    /// 0.0 for read, 1.0 for write
    pub last_access_type: f32,
    /// Blob size (normalized to [0, 1])
    pub size: f32,
    /// Predicted next access time (normalized to [0, 1])
    pub next_access_pred: f32,
    /// Write frequency ratio [0, 1]
    pub overwrite_amount: f32,
}

impl BlobFeatures {
    /// Extract 10-dimensional features from blob data and access history
    ///
    /// # Arguments
    /// * `blob` - Blob data from trace
    /// * `tracker` - Access history tracker
    /// * `current_time_ms` - Current timestamp in milliseconds
    /// * `max_size` - Maximum blob size for normalization
    /// * `max_frequency` - Maximum access frequency for normalization
    ///
    /// # Returns
    /// Feature vector for the blob
    pub fn extract(
        blob: &BlobData,
        tracker: &AccessTracker,
        current_time_ms: u64,
        max_size: f64,
        max_frequency: u32,
    ) -> Self {
        // 1. Recency (normalized to [0, 1])
        let recency_raw = tracker.get_recency(&blob.offset_id, current_time_ms);
        let recency = if recency_raw.is_finite() {
            // Normalize: assume max recency of 1 hour (3600000 ms)
            (recency_raw / 3600000.0).min(1.0)
        } else {
            1.0 // Never accessed -> max recency
        };

        // 2. Frequency (normalized to [0, 1])
        let frequency_raw = tracker.get_frequency(&blob.offset_id);
        let frequency = if max_frequency > 0 {
            ((frequency_raw as f32) / (max_frequency as f32)).min(1.0)
        } else {
            0.0
        };

        // 3 & 4. Mean and Std of access intervals (normalized to [0, 1])
        let times = tracker.get_access_times(&blob.offset_id);
        let (mean_interval, std_interval) = if times.len() > 1 {
            let intervals: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
            let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let variance = if intervals.len() > 1 {
                intervals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (intervals.len() - 1) as f64
            } else {
                0.0
            };
            // Normalize to [0, 1] - assume max interval of 1 hour (3600000 ms)
            let mean_normalized = (mean / 3600000.0).min(1.0) as f32;
            let std_normalized = (variance.sqrt() / 3600000.0).min(1.0) as f32;
            (mean_normalized, std_normalized)
        } else {
            (0.0, 0.0) // No interval history
        };

        // 5. Is sequential (blob's is_sequence field)
        let is_sequential = if blob.is_sequence { 1.0 } else { 0.0 };

        // 6. Reuse distance (normalized to [0, 1])
        let reuse_distance = match tracker.get_reuse_distance(&blob.offset_id) {
            Some(dist) => (dist as f32 / 10000.0).min(1.0), // Normalize to 10K
            None => 0.0,                                    // First access
        };

        // 7. Last access type (from blob's io_op)
        let last_access_type = if blob.is_read() { 0.0 } else { 1.0 };

        // 8. Size (normalized to [0, 1])
        let size = if max_size > 0.0 {
            (blob.offset_size as f32 / max_size as f32).min(1.0)
        } else {
            0.0
        };

        // 9. Next access prediction (simple heuristic: inverse of recency)
        let next_access_pred = 1.0 - recency;

        // 10. Overwrite amount (from blob)
        let overwrite_amount = blob.overwrite_amount.clamp(0.0, 1.0);

        Self {
            recency,
            frequency,
            mean_interval,
            std_interval,
            is_sequential,
            reuse_distance,
            last_access_type,
            size,
            next_access_pred,
            overwrite_amount,
        }
    }

    /// Convert features to vector for model input
    ///
    /// Returns a warp-aligned vector of 32 dimensions:
    /// - First 10 elements: actual features
    /// - Remaining 22 elements: zero padding for GPU optimization
    pub fn to_vec(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(WARP_SIZE);

        // Original 10 features
        features.push(self.recency);
        features.push(self.frequency);
        features.push(self.mean_interval);
        features.push(self.std_interval);
        features.push(self.is_sequential);
        features.push(self.reuse_distance);
        features.push(self.last_access_type);
        features.push(self.size);
        features.push(self.next_access_pred);
        features.push(self.overwrite_amount);

        // Pad with zeros to reach warp size (32)
        while features.len() < WARP_SIZE {
            features.push(0.0);
        }

        features
    }
}

/// Encode state: [tier_sizes(5), features(10)] = 15-dimensional vector, padded to 32 for GPU optimization
///
/// # Arguments
/// * `tier_sizes` - Current sizes of each tier (from BufferEnv::tier_sizes())
/// * `features` - Blob feature vector
/// * `tier_configs` - Tier configuration for capacity normalization
///
/// # Returns
/// 32-dimensional state vector for RL agent (padded to warp size for GPU optimization)
pub fn encode_state(
    tier_sizes: &[f64],
    features: &BlobFeatures,
    tier_configs: &[TierConfig],
) -> Vec<f32> {
    let mut state = Vec::with_capacity(WARP_SIZE);

    // Add tier utilizations (up to 5, pad rest)
    // Use tier_sizes for current size, tier_configs for capacity
    for (i, tier) in tier_configs.iter().take(5).enumerate() {
        let current_size = tier_sizes.get(i).copied().unwrap_or(0.0);
        let utilization = (current_size / tier.capacity as f64).min(1.0); // Clamp to [0, 1]
        state.push(utilization as f32);
    }
    // Pad remaining tier slots
    while state.len() < 5 {
        state.push(0.0);
    }

    // Add features (already padded to 32 by to_vec())
    let feature_vec = features.to_vec(); // Now returns 32 elements
    state.extend(&feature_vec[0..10]); // Only take first 10 actual features

    // Ensure total is 32 (warp size)
    state.resize(WARP_SIZE, 0.0);

    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::tracker::AccessRecord;
    use crate::trace::IoOp;

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
    fn test_feature_extraction() {
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
        approx::assert_relative_eq!(features.last_access_type, 0.0, epsilon = 1e-5);
        approx::assert_relative_eq!(features.overwrite_amount, 0.5, epsilon = 1e-5);

        let vec = features.to_vec();
        assert_eq!(vec.len(), 32, "Features must be padded to warp size 32");
    }

    #[test]
    fn test_feature_extraction_write() {
        let mut tracker = AccessTracker::new(1000);
        let mut blob = create_test_blob("test");
        blob.io_op = "write".into();

        let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
        approx::assert_relative_eq!(features.last_access_type, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_feature_extraction_non_sequential() {
        let mut tracker = AccessTracker::new(1000);
        let mut blob = create_test_blob("test");
        blob.is_sequence = false;

        let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
        approx::assert_relative_eq!(features.is_sequential, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_recency_normalization() {
        let mut tracker = AccessTracker::new(1000);
        tracker.record(AccessRecord {
            blob_id: "test".into(),
            timestamp_ms: 1000,
            access_type: IoOp::Read,
            size: 1024.0,
        });

        let blob = create_test_blob("test");

        // Test with current time = 1000 + 3_600_000 (exactly 1 hour)
        let features = BlobFeatures::extract(&blob, &tracker, 1000 + 3_600_000, 10000.0, 100);
        approx::assert_relative_eq!(features.recency, 1.0, epsilon = 1e-5);

        // Test with current time = 1000 + 1_800_000 (30 minutes)
        let features = BlobFeatures::extract(&blob, &tracker, 1000 + 1_800_000, 10000.0, 100);
        approx::assert_relative_eq!(features.recency, 0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_frequency_normalization() {
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
    fn test_access_intervals() {
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
        // Mean: 1500 ms = 1500 / 3600000 (normalized) = 0.00041666...
        let expected_mean = 1500.0_f64 / 3_600_000.0;
        approx::assert_relative_eq!(features.mean_interval, expected_mean as f32, epsilon = 1e-5);

        // Std: sqrt(((1000-1500)^2 + (2000-1500)^2) / 1)
        //    = sqrt((250000 + 250000) / 1)
        //    = sqrt(500000)
        //    ≈ 707.107 ms
        // Normalized: 707.107 / 3600000 ≈ 0.0001964
        let expected_std = ((500000.0_f64).sqrt() / 3_600_000.0) as f32;
        approx::assert_relative_eq!(features.std_interval, expected_std, epsilon = 1e-3);
    }

    #[test]
    fn test_state_encoding() {
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

        assert_eq!(state.len(), 32, "State must be padded to warp size 32"); // 5 tier sizes + 10 features + 17 padding

        // Check tier normalization
        approx::assert_relative_eq!(state[0], 0.5, epsilon = 1e-5); // 400/800
        approx::assert_relative_eq!(state[1], 0.5, epsilon = 1e-5); // 1000/2000
        approx::assert_relative_eq!(state[2], 0.5, epsilon = 1e-5); // 2000/4000
        approx::assert_relative_eq!(state[3], 0.5, epsilon = 1e-5); // 10000/20000

        // Check features are copied correctly (indices 5-14 are the 10 features)
        approx::assert_relative_eq!(state[5], 0.1, epsilon = 1e-5); // recency
        approx::assert_relative_eq!(state[6], 0.5, epsilon = 1e-5); // frequency

        // Check padding (indices 15-31 should be zeros)
        for i in 15..32 {
            approx::assert_relative_eq!(state[i], 0.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_size_normalization() {
        let mut tracker = AccessTracker::new(1000);
        let mut blob = create_test_blob("test");
        blob.offset_size = 5000.0; // Half of max_size

        let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
        approx::assert_relative_eq!(features.size, 0.5, epsilon = 1e-5);
    }

    #[test]
    fn test_overwrite_amount_clamping() {
        let mut tracker = AccessTracker::new(1000);
        let mut blob = create_test_blob("test");

        // Test values outside [0, 1]
        blob.overwrite_amount = 1.5;
        let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
        approx::assert_relative_eq!(features.overwrite_amount, 1.0, epsilon = 1e-5);

        blob.overwrite_amount = -0.5;
        let features = BlobFeatures::extract(&blob, &tracker, 5000, 10000.0, 100);
        approx::assert_relative_eq!(features.overwrite_amount, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_frequency_clamping() {
        // Test that frequency is clamped to [0, 1]
        let mut tracker = AccessTracker::new(1000);

        // Record 150 accesses (more than max_frequency = 100)
        for i in 0..150 {
            tracker.record(AccessRecord {
                blob_id: "test".into(),
                timestamp_ms: i as u64 * 100,
                access_type: IoOp::Read,
                size: 1024.0,
            });
        }

        let blob = create_test_blob("test");
        let features = BlobFeatures::extract(&blob, &tracker, 50000, 10000.0, 100);

        // Should be clamped to 1.0, not 1.5
        approx::assert_relative_eq!(features.frequency, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mean_std_interval_normalization() {
        // Test that mean_interval and std_interval are properly normalized
        let mut tracker = AccessTracker::new(1000);

        // Record accesses at very large intervals (exceeding normalization range)
        tracker.record(AccessRecord {
            blob_id: "test".into(),
            timestamp_ms: 0,
            access_type: IoOp::Read,
            size: 1024.0,
        });
        tracker.record(AccessRecord {
            blob_id: "test".into(),
            timestamp_ms: 5_000_000, // 5 million ms = 83.3 minutes
            access_type: IoOp::Read,
            size: 1024.0,
        });
        tracker.record(AccessRecord {
            blob_id: "test".into(),
            timestamp_ms: 10_000_000, // 10 million ms = 166.7 minutes
            access_type: IoOp::Read,
            size: 1024.0,
        });

        let blob = create_test_blob("test");
        let features = BlobFeatures::extract(&blob, &tracker, 10_000_000, 10000.0, 100);

        // Intervals: 5_000_000 ms each
        // Normalized by 3_600_000 (1 hour = 3600 seconds * 1000)
        // After normalization and clamping to [0, 1], both should be 1.0
        assert!(features.mean_interval >= 0.0 && features.mean_interval <= 1.0);
        assert!(features.std_interval >= 0.0 && features.std_interval <= 1.0);
    }

    #[test]
    fn test_tier_size_normalization_with_clamping() {
        // Test that tier sizes are clamped to [0, 1]
        let tier_sizes = vec![1600.0, 1000.0, 2000.0, 10000.0, 50000.0]; // First size > capacity
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
        let tier_configs = vec![
            TierConfig {
                name: "Memory".into(),
                tier_id: 0,
                capacity: 800.0, // Less than tier_sizes[0]
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

        // First tier size should be clamped to 1.0 (1600/800 = 2.0, clamped to 1.0)
        assert_eq!(state.len(), 32, "State must be padded to warp size 32");
        approx::assert_relative_eq!(state[0], 1.0, epsilon = 1e-5); // Clamped
        approx::assert_relative_eq!(state[1], 0.5, epsilon = 1e-5);
    }
}
