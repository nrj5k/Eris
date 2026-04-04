use crate::config_old::TierConfig;
use crate::error::{EnvError, Result};
use std::cell::Cell;
use std::collections::HashMap;

/// Represents a single storage tier in the hierarchy.
///
/// Each tier has a fixed capacity and stores blobs by ID with their sizes.
/// Tracks current usage and total access count for metrics.
#[derive(Debug, Clone)]
pub struct Tier {
    /// Configuration for this tier
    pub config: TierConfig,
    /// Storage mapping blob IDs to their sizes
    storage: HashMap<String, f64>,
    /// Current total size stored in this tier
    current_size: f64,
    /// Total number of read operations (for metrics)
    access_count: Cell<u64>,
}

impl Tier {
    /// Create a new tier with the given configuration
    pub fn new(config: TierConfig) -> Self {
        Self {
            config,
            storage: HashMap::new(),
            current_size: 0.0,
            access_count: Cell::new(0),
        }
    }

    /// Write a blob to this tier
    ///
    /// # Errors
    /// Returns `EnvError::CapacityExceeded` if the blob would exceed tier capacity
    pub fn write(&mut self, blob_id: &str, size: f64) -> Result<()> {
        // Remove old size if blob already exists
        if let Some(old_size) = self.storage.get(blob_id) {
            self.current_size -= *old_size;
        }

        if self.current_size + size > self.config.capacity {
            return Err(EnvError::CapacityExceeded {
                tier_id: self.config.tier_id,
                requested: size,
                available: self.available_capacity(),
            });
        }

        self.storage.insert(blob_id.to_string(), size);
        self.current_size += size;
        Ok(())
    }

    /// Read a blob from this tier (returns size if found)
    ///
    /// Increments access count for metrics tracking
    pub fn read(&self, blob_id: &str) -> Option<f64> {
        self.access_count.set(self.access_count.get() + 1);
        self.storage.get(blob_id).copied()
    }

    /// Remove a blob from this tier
    ///
    /// Returns `true` if the blob was present and removed
    pub fn remove(&mut self, blob_id: &str) -> bool {
        if let Some(size) = self.storage.remove(blob_id) {
            self.current_size -= size;
            true
        } else {
            false
        }
    }

    /// Clear all blobs and reset tier state
    pub fn clear(&mut self) {
        self.storage.clear();
        self.current_size = 0.0;
        self.access_count.set(0);
    }

    /// Get available capacity in this tier
    pub fn available_capacity(&self) -> f64 {
        self.config.capacity - self.current_size
    }

    /// Check if a blob exists in this tier
    pub fn contains(&self, blob_id: &str) -> bool {
        self.storage.contains_key(blob_id)
    }

    /// Get utilization ratio [0.0, 1.0] for this tier
    pub fn utilization(&self) -> f32 {
        if self.config.capacity > 0.0 {
            (self.current_size / self.config.capacity) as f32
        } else {
            0.0
        }
    }

    /// Get total access count for this tier
    pub fn access_count(&self) -> u64 {
        self.access_count.get()
    }

    /// Get current stored size
    pub fn current_size(&self) -> f64 {
        self.current_size
    }

    /// Get number of blobs in this tier
    pub fn blob_count(&self) -> usize {
        self.storage.len()
    }

    /// Get all blob IDs stored in this tier
    pub fn storage_keys(&self) -> Vec<String> {
        self.storage.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> TierConfig {
        TierConfig {
            name: "Memory".into(),
            tier_id: 0,
            capacity: 1000.0,
            access_latency: 0.01,
            description: "Test tier".into(),
        }
    }

    #[test]
    fn test_tier_new() {
        let tier = Tier::new(test_config());
        assert_eq!(tier.current_size(), 0.0);
        assert_eq!(tier.access_count(), 0);
        assert_eq!(tier.blob_count(), 0);
    }

    #[test]
    fn test_tier_write_within_capacity() {
        let mut tier = Tier::new(test_config());
        assert!(tier.write("blob1", 500.0).is_ok());
        assert!(tier.contains("blob1"));
        assert_eq!(tier.current_size(), 500.0);
        assert_eq!(tier.blob_count(), 1);
    }

    #[test]
    fn test_tier_write_exceeds_capacity() {
        let mut tier = Tier::new(test_config());
        assert!(tier.write("blob1", 500.0).is_ok());
        assert!(tier.write("blob2", 600.0).is_err());
        assert_eq!(tier.current_size(), 500.0);
        assert_eq!(tier.blob_count(), 1);
    }

    #[test]
    fn test_tier_read() {
        let mut tier = Tier::new(test_config());
        tier.write("blob1", 500.0).unwrap();

        let size = tier.read("blob1");
        assert_eq!(size, Some(500.0));
        assert_eq!(tier.access_count(), 1);

        tier.read("blob1");
        assert_eq!(tier.access_count(), 2);
    }

    #[test]
    fn test_tier_remove() {
        let mut tier = Tier::new(test_config());
        tier.write("blob1", 500.0).unwrap();

        assert!(tier.remove("blob1"));
        assert!(!tier.contains("blob1"));
        assert_eq!(tier.current_size(), 0.0);
        assert_eq!(tier.blob_count(), 0);

        assert!(!tier.remove("nonexistent"));
    }

    #[test]
    fn test_tier_clear() {
        let mut tier = Tier::new(test_config());
        tier.write("blob1", 500.0).unwrap();
        tier.write("blob2", 300.0).unwrap();
        tier.read("blob1");

        tier.clear();
        assert_eq!(tier.current_size(), 0.0);
        assert_eq!(tier.access_count(), 0);
        assert_eq!(tier.blob_count(), 0);
    }

    #[test]
    fn test_utilization() {
        let mut tier = Tier::new(test_config());
        assert_eq!(tier.utilization(), 0.0);

        tier.write("blob1", 500.0).unwrap();
        approx::assert_relative_eq!(tier.utilization(), 0.5, epsilon = 1e-5);

        tier.write("blob2", 250.0).unwrap();
        approx::assert_relative_eq!(tier.utilization(), 0.75, epsilon = 1e-5);
    }

    #[test]
    fn test_duplicate_write_updates_size() {
        // Test that writing the same blob twice updates size correctly
        let mut tier = Tier::new(test_config());

        // Write blob1 with size 500
        tier.write("blob1", 500.0).unwrap();
        assert_eq!(tier.current_size(), 500.0);
        assert_eq!(tier.blob_count(), 1);

        // Write blob1 again with size 700
        tier.write("blob1", 700.0).unwrap();
        assert_eq!(tier.current_size(), 700.0); // Should be 700, not 1200
        assert_eq!(tier.blob_count(), 1);

        // Verify blob size was updated
        let size = tier.read("blob1");
        assert_eq!(size, Some(700.0));
    }

    #[test]
    fn test_duplicate_write_within_capacity() {
        // Test that duplicate write doesn't exceed capacity incorrectly
        let mut tier = Tier::new(test_config());

        // Fill tier to 80% capacity
        tier.write("blob1", 800.0).unwrap();
        assert_eq!(tier.current_size(), 800.0);

        // Overwrite blob1 with new size (should succeed even though 800+300 > capacity)
        // because we remove old size first
        assert!(tier.write("blob1", 300.0).is_ok());
        assert_eq!(tier.current_size(), 300.0);
    }
}
