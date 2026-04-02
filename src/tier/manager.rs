use crate::config::TierConfig;
use crate::tier::selector::TierSelector;
use crate::tier::Tier;

/// Multi-tier storage environment coordinator
///
/// Manages multiple storage tiers and provides aggregate state information.
pub struct BufferEnv {
    tiers: Vec<Tier>,
}

impl BufferEnv {
    /// Create a new buffer environment with the given tier configurations
    pub fn new(configs: Vec<TierConfig>) -> Self {
        let tiers = configs.into_iter().map(Tier::new).collect();
        Self { tiers }
    }

    /// Get current size of each tier
    pub fn tier_sizes(&self) -> Vec<f64> {
        self.tiers.iter().map(|t| t.current_size()).collect()
    }

    /// Get utilization state for all tiers [0.0, 1.0]
    pub fn get_state(&self) -> Vec<f32> {
        self.tiers.iter().map(|t| t.utilization()).collect()
    }

    /// Reset all tiers to empty state
    pub fn reset(&mut self) {
        for tier in &mut self.tiers {
            tier.clear();
        }
    }

    /// Get number of tiers
    pub fn num_tiers(&self) -> usize {
        self.tiers.len()
    }

    /// Create a tier selector for capacity-weighted selection
    pub fn selector(&self) -> TierSelector {
        TierSelector::new(self.tiers.clone())
    }

    /// Get access counts for all tiers
    pub fn tier_accesses(&self) -> Vec<u64> {
        self.tiers.iter().map(|t| t.access_count()).collect()
    }

    /// Get mutable access to a tier by index
    pub fn get_tier(&mut self, idx: usize) -> Option<&mut Tier> {
        self.tiers.get_mut(idx)
    }

    /// Get immutable access to a tier by index
    pub fn get_tier_ref(&self, idx: usize) -> Option<&Tier> {
        self.tiers.get(idx)
    }

    /// Get total storage capacity across all tiers
    pub fn total_capacity(&self) -> f64 {
        self.tiers.iter().map(|t| t.config.capacity).sum()
    }

    /// Get total used storage across all tiers
    pub fn total_used(&self) -> f64 {
        self.tiers.iter().map(|t| t.current_size()).sum()
    }

    /// Find which tier contains a blob (returns tier index)
    pub fn find_blob(&self, blob_id: &str) -> Option<usize> {
        self.tiers.iter().position(|t| t.contains(blob_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_configs() -> Vec<TierConfig> {
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
        ]
    }

    #[test]
    fn test_buffer_env_new() {
        let env = BufferEnv::new(test_configs());
        assert_eq!(env.num_tiers(), 2);
        assert_eq!(env.tier_sizes(), vec![0.0, 0.0]);
        assert_eq!(env.tier_accesses(), vec![0, 0]);
    }

    #[test]
    fn test_buffer_env_state() {
        let env = BufferEnv::new(test_configs());
        let state = env.get_state();
        assert_eq!(state, vec![0.0, 0.0]);
    }

    #[test]
    fn test_buffer_env_write() {
        let mut env = BufferEnv::new(test_configs());

        let tier = env.get_tier(0).unwrap();
        tier.write("blob1", 500.0).unwrap();

        assert_eq!(env.tier_sizes(), vec![500.0, 0.0]);

        let tier1 = env.get_tier(1).unwrap();
        tier1.write("blob2", 1000.0).unwrap();

        assert_eq!(env.tier_sizes(), vec![500.0, 1000.0]);
    }

    #[test]
    fn test_buffer_env_reset() {
        let mut env = BufferEnv::new(test_configs());

        env.get_tier(0).unwrap().write("blob1", 500.0).unwrap();
        env.get_tier(1).unwrap().write("blob2", 1000.0).unwrap();

        assert_eq!(env.tier_sizes(), vec![500.0, 1000.0]);

        env.reset();
        assert_eq!(env.tier_sizes(), vec![0.0, 0.0]);
        assert_eq!(env.tier_accesses(), vec![0, 0]);
    }

    #[test]
    fn test_buffer_env_find_blob() {
        let mut env = BufferEnv::new(test_configs());

        assert_eq!(env.find_blob("blob1"), None);

        env.get_tier(0).unwrap().write("blob1", 500.0).unwrap();
        assert_eq!(env.find_blob("blob1"), Some(0));

        env.get_tier(1).unwrap().write("blob2", 1000.0).unwrap();
        assert_eq!(env.find_blob("blob2"), Some(1));
        assert_eq!(env.find_blob("nonexistent"), None);
    }

    #[test]
    fn test_buffer_env_total_capacity() {
        let mut env = BufferEnv::new(test_configs());
        assert_eq!(env.total_capacity(), 800.0 + 2000.0);

        env.get_tier(0).unwrap().write("blob1", 500.0).unwrap();
        assert_eq!(env.total_capacity(), 800.0 + 2000.0); // Capacity doesn't change
    }

    #[test]
    fn test_buffer_env_total_used() {
        let mut env = BufferEnv::new(test_configs());
        assert_eq!(env.total_used(), 0.0);

        env.get_tier(0).unwrap().write("blob1", 500.0).unwrap();
        env.get_tier(1).unwrap().write("blob2", 1000.0).unwrap();
        assert_eq!(env.total_used(), 1500.0);
    }

    #[test]
    fn test_buffer_env_selector() {
        let env = BufferEnv::new(test_configs());
        let selector = env.selector();

        assert_eq!(selector.num_tiers(), 2);
        assert!(selector.get(0).is_some());
    }

    #[test]
    fn test_buffer_env_access_count() {
        let mut env = BufferEnv::new(test_configs());

        env.get_tier(0).unwrap().write("blob1", 500.0).unwrap();
        env.get_tier_ref(0).unwrap().read("blob1");
        env.get_tier_ref(0).unwrap().read("blob1");

        assert_eq!(env.tier_accesses(), vec![2, 0]);
    }
}
