use crate::trace::IoOp;
use std::collections::{HashMap, VecDeque};

/// Access record for tracking blob access history
#[derive(Debug, Clone)]
pub struct AccessRecord {
    /// Unique blob identifier
    pub blob_id: String,
    /// Timestamp in milliseconds since epoch
    pub timestamp_ms: u64,
    /// I/O operation type
    pub access_type: IoOp,
    /// Size of the access in bytes
    pub size: f64,
}

/// Sliding window access history tracker
#[derive(Debug, Clone)]
pub struct AccessTracker {
    /// In-memory sliding window of access records
    memory_window: VecDeque<AccessRecord>,
    /// Maximum size of the sliding window
    max_memory_size: usize,
    /// Global access sequence for reuse distance calculation
    access_sequence: Vec<String>,
    /// Index for O(1) lookup of last access time per blob
    last_access_time: HashMap<String, u64>,
    /// Access count per blob
    access_counts: HashMap<String, u32>,
}

impl AccessTracker {
    /// Create a new tracker with specified window size
    pub fn new(window_size: usize) -> Self {
        Self {
            memory_window: VecDeque::with_capacity(window_size),
            max_memory_size: window_size,
            access_sequence: Vec::new(),
            last_access_time: HashMap::new(),
            access_counts: HashMap::new(),
        }
    }

    /// Record a new access
    pub fn record(&mut self, access: AccessRecord) {
        let blob_id = access.blob_id.clone();
        let timestamp = access.timestamp_ms;

        // Update last access time
        self.last_access_time.insert(blob_id.clone(), timestamp);

        // Update access count
        *self.access_counts.entry(blob_id.clone()).or_insert(0) += 1;

        // Add to sequence
        self.access_sequence.push(blob_id);

        // Add to sliding window
        if self.memory_window.len() >= self.max_memory_size {
            self.memory_window.pop_front();
        }
        self.memory_window.push_back(access);
    }

    /// Get last N accesses for a blob
    pub fn get_blob_history(&self, blob_id: &str, n: usize) -> Vec<&AccessRecord> {
        self.memory_window
            .iter()
            .filter(|r| r.blob_id == blob_id)
            .rev()
            .take(n)
            .collect()
    }

    /// Get time since last access (milliseconds)
    pub fn get_recency(&self, blob_id: &str, current_time_ms: u64) -> f32 {
        match self.last_access_time.get(blob_id) {
            Some(&last_time) => (current_time_ms - last_time) as f32,
            None => f32::INFINITY, // Never accessed
        }
    }

    /// Get access frequency (count)
    pub fn get_frequency(&self, blob_id: &str) -> u32 {
        *self.access_counts.get(blob_id).unwrap_or(&0)
    }

    /// Calculate reuse distance (position since last access)
    ///
    /// The reuse distance represents how recently a blob was accessed relative to the
    /// overall access pattern. This is useful for cache eviction policies and hotness scoring.
    ///
    /// # Returns
    /// * `None` if blob has never been accessed
    /// * `Some(distance)` where distance is the number of positions from the end of
    ///   the access sequence to the most recent occurrence of this blob
    ///
    /// # Examples
    ///
    /// If the access sequence is ["A", "B", "C", "A"] and we query:
    /// - For "A": Returns Some(0) (last occurrence is at position 0 from the end, just accessed)
    /// - For "B": Returns Some(2) (last occurrence is at position 2 from the end)
    /// - For "C": Returns Some(1) (last occurrence is at position 1 from the end)
    /// - For "D": Returns None (never accessed)
    ///
    /// # Note
    /// Lower reuse distance means more recently accessed, which typically indicates
    /// higher temporal locality and potential "hotness" for caching decisions.
    pub fn get_reuse_distance(&self, blob_id: &str) -> Option<usize> {
        // Find position of last access in the sequence (from the back)
        let last_pos = self
            .access_sequence
            .iter()
            .rev()
            .position(|id| id == blob_id)?;

        Some(last_pos)
    }

    /// Get all access timestamps for a blob (for interval calculation)
    pub fn get_access_times(&self, blob_id: &str) -> Vec<u64> {
        self.memory_window
            .iter()
            .filter(|r| r.blob_id == blob_id)
            .map(|r| r.timestamp_ms)
            .collect()
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.memory_window.clear();
        self.access_sequence.clear();
        self.last_access_time.clear();
        self.access_counts.clear();
    }

    /// Get number of records in the sliding window
    pub fn len(&self) -> usize {
        self.memory_window.len()
    }

    /// Check if the tracker is empty
    pub fn is_empty(&self) -> bool {
        self.memory_window.is_empty()
    }

    /// Get total number of accesses in the sequence
    pub fn sequence_len(&self) -> usize {
        self.access_sequence.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tracker() {
        let tracker = AccessTracker::new(1000);
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_record_access() {
        let mut tracker = AccessTracker::new(1000);

        tracker.record(AccessRecord {
            blob_id: "blob1".into(),
            timestamp_ms: 1000,
            access_type: IoOp::Read,
            size: 1024.0,
        });

        assert_eq!(tracker.len(), 1);
        assert_eq!(tracker.get_frequency("blob1"), 1);
    }

    #[test]
    fn test_multiple_accesses() {
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
            size: 2048.0,
        });
        tracker.record(AccessRecord {
            blob_id: "blob2".into(),
            timestamp_ms: 3000,
            access_type: IoOp::Read,
            size: 512.0,
        });

        assert_eq!(tracker.len(), 3);
        assert_eq!(tracker.get_frequency("blob1"), 2);
        assert_eq!(tracker.get_frequency("blob2"), 1);
    }

    #[test]
    fn test_sliding_window() {
        let mut tracker = AccessTracker::new(3);

        for i in 0..5 {
            tracker.record(AccessRecord {
                blob_id: format!("blob{}", i),
                timestamp_ms: i as u64 * 1000,
                access_type: IoOp::Read,
                size: 1024.0,
            });
        }

        // Should only keep last 3
        assert_eq!(tracker.len(), 3);
        assert_eq!(tracker.get_frequency("blob0"), 1); // Count is preserved
        assert_eq!(tracker.get_frequency("blob4"), 1);
    }

    #[test]
    fn test_recency() {
        let mut tracker = AccessTracker::new(1000);

        tracker.record(AccessRecord {
            blob_id: "blob1".into(),
            timestamp_ms: 1000,
            access_type: IoOp::Read,
            size: 1024.0,
        });

        let recency = tracker.get_recency("blob1", 5000);
        approx::assert_relative_eq!(recency, 4000.0, epsilon = 1e-5);

        let recency_never = tracker.get_recency("never_seen", 5000);
        assert!(recency_never.is_infinite());
    }

    #[test]
    fn test_reuse_distance() {
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
        // Reuse distance is position from the end where last access occurred
        // blob1 was accessed at positions 0 and 2, most recent is at position 0 from back
        // blob2 was accessed at position 1, which is position 1 from back
        assert_eq!(tracker.get_reuse_distance("blob1"), Some(0));
        assert_eq!(tracker.get_reuse_distance("blob2"), Some(1));
        assert_eq!(tracker.get_reuse_distance("never_seen"), None);
    }

    #[test]
    fn test_access_times() {
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
            size: 1024.0,
        });

        let times = tracker.get_access_times("blob1");
        assert_eq!(times.len(), 2);
        assert_eq!(times[0], 1000);
        assert_eq!(times[1], 3000);
    }

    #[test]
    fn test_blob_history() {
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
            size: 1024.0,
        });

        let history = tracker.get_blob_history("blob1", 1);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].timestamp_ms, 3000);

        let history_all = tracker.get_blob_history("blob1", 10);
        assert_eq!(history_all.len(), 2);
    }

    #[test]
    fn test_clear() {
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
}
