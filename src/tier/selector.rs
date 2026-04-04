use crate::tier::Tier;

/// Capacity-weighted tier selector
///
/// Selects tiers based on available capacity using proportional distribution.
/// Importance score [0.0, 1.0] is mapped to a tier using cumulative capacity ratios.
pub struct TierSelector {
    tiers: Vec<Tier>,
}

impl TierSelector {
    /// Create a new selector with the given tiers
    pub fn new(tiers: Vec<Tier>) -> Self {
        Self { tiers }
    }

    /// Select tier based on importance score [0.0, 1.0]
    ///
    /// Maps continuous score to discrete tier using capacity-weighted distribution.
    /// Higher importance scores map to slower/higher-capacity tiers.
    /// Tiers with zero available capacity are skipped.
    ///
    /// # Returns
    /// Tier index in the range [0, num_tiers)
    pub fn select_tier(&self, importance: f32) -> usize {
        let available: Vec<f64> = self.tiers.iter().map(|t| t.available_capacity()).collect();

        let total: f64 = available.iter().sum();

        if total <= 0.0 {
            return self.tiers.len().saturating_sub(1);
        }

        let mut cumsum = 0.0;
        for (i, &cap) in available.iter().enumerate() {
            // Skip full tiers (zero available capacity)
            if cap <= 0.0 {
                continue;
            }
            cumsum += cap / total;
            if cumsum >= importance as f64 {
                return i;
            }
        }

        // If all tiers except the last are full, return last tier
        self.tiers.len().saturating_sub(1)
    }

    /// Get the number of tiers
    pub fn num_tiers(&self) -> usize {
        self.tiers.len()
    }

    /// Get tier by index
    pub fn get(&self, idx: usize) -> Option<&Tier> {
        self.tiers.get(idx)
    }

    /// Get mutable tier by index
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut Tier> {
        self.tiers.get_mut(idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_old::TierConfig;

    fn test_tier(id: u32, capacity: f64, name: &str) -> Tier {
        Tier::new(TierConfig {
            name: name.into(),
            tier_id: id,
            capacity,
            access_latency: 0.01,
            description: String::new(),
        })
    }

    #[test]
    fn test_selector_empty_tiers() {
        let selector = TierSelector::new(vec![]);
        assert_eq!(selector.num_tiers(), 0);
        assert_eq!(selector.select_tier(0.5), 0);
    }

    #[test]
    fn test_selector_single_tier() {
        let tier = test_tier(0, 100.0, "T0");
        let selector = TierSelector::new(vec![tier]);

        assert_eq!(selector.num_tiers(), 1);
        assert_eq!(selector.select_tier(0.0), 0);
        assert_eq!(selector.select_tier(0.5), 0);
        assert_eq!(selector.select_tier(1.0), 0);
    }

    #[test]
    fn test_selector_equal_capacity() {
        let tiers = vec![test_tier(0, 100.0, "T0"), test_tier(1, 100.0, "T1")];
        let selector = TierSelector::new(tiers);

        // Both tiers have equal capacity (50/50 split)
        // importance 0.0 → cumsum 0.5 at tier 0 → select tier 0
        assert_eq!(selector.select_tier(0.0), 0);
        assert_eq!(selector.select_tier(0.4), 0);

        // importance 0.5 → cumsum reaches 0.5 at tier 0
        // importance 0.6 → cumsum reaches 1.0 at tier 1
        assert_eq!(selector.select_tier(0.5), 0);
        assert_eq!(selector.select_tier(0.6), 1);
    }

    #[test]
    fn test_selector_different_capacities() {
        let tiers = vec![
            test_tier(0, 100.0, "T0"), // 33% of total
            test_tier(1, 200.0, "T1"), // 67% of total
        ];
        let selector = TierSelector::new(tiers);

        // importance 0.0 → tier 0 (cumsum 0.33 >= 0.0)
        assert_eq!(selector.select_tier(0.0), 0);

        // importance 0.33 → tier 0 (cumsum 0.33 >= 0.33)
        assert_eq!(selector.select_tier(0.33), 0);

        // importance 0.34 → tier 1 (cumsum 0.33 < 0.34, need next tier)
        assert_eq!(selector.select_tier(0.34), 1);

        // importance 1.0 → tier 1 (last tier)
        assert_eq!(selector.select_tier(1.0), 1);
    }

    #[test]
    fn test_selector_full_tier() {
        let mut tier0 = test_tier(0, 100.0, "T0");
        let tier1 = test_tier(1, 100.0, "T1");

        // Fill tier 0
        tier0.write("blob", 100.0).unwrap();

        let selector = TierSelector::new(vec![tier0, tier1]);

        // tier 0 is full, only tier 1 has capacity
        // Any importance should select tier 1
        assert_eq!(selector.select_tier(0.0), 1);
        assert_eq!(selector.select_tier(0.5), 1);
        assert_eq!(selector.select_tier(1.0), 1);
    }

    #[test]
    fn test_selector_all_full() {
        let mut tier0 = test_tier(0, 100.0, "T0");
        let mut tier1 = test_tier(1, 100.0, "T1");

        tier0.write("blob1", 100.0).unwrap();
        tier1.write("blob2", 100.0).unwrap();

        let selector = TierSelector::new(vec![tier0, tier1]);

        // All tiers full, return last tier
        assert_eq!(selector.select_tier(0.0), 1);
        assert_eq!(selector.select_tier(0.5), 1);
    }
}
