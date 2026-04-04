use eris::config::TierConfig;
use eris::features::{HotnessConfig, hotness_score};
use eris::tier::{BufferEnv, Tier, TierSelector};

#[test]
fn test_tier_write_within_capacity() {
    let config = TierConfig {
        name: "Memory".into(),
        tier_id: 0,
        capacity: 1000.0,
        access_latency: 0.01,
        description: String::new(),
    };

    let mut tier = Tier::new(config);

    assert!(tier.write("blob1", 500.0).is_ok());
    assert!(tier.contains("blob1"));
    assert_eq!(tier.current_size(), 500.0);
}

#[test]
fn test_tier_write_exceeds_capacity() {
    let config = TierConfig {
        name: "Memory".into(),
        tier_id: 0,
        capacity: 1000.0,
        access_latency: 0.01,
        description: String::new(),
    };

    let mut tier = Tier::new(config);

    assert!(tier.write("blob1", 500.0).is_ok());
    assert!(tier.write("blob2", 600.0).is_err()); // Exceeds capacity
}

#[test]
fn test_tier_remove() {
    let config = TierConfig {
        name: "Memory".into(),
        tier_id: 0,
        capacity: 1000.0,
        access_latency: 0.01,
        description: String::new(),
    };

    let mut tier = Tier::new(config);
    tier.write("blob1", 500.0).unwrap();

    assert!(tier.remove("blob1"));
    assert!(!tier.contains("blob1"));
    assert_eq!(tier.current_size(), 0.0);
}

#[test]
fn test_tier_selector() {
    let tiers = vec![
        Tier::new(TierConfig {
            name: "T0".into(),
            tier_id: 0,
            capacity: 100.0,
            access_latency: 0.01,
            description: String::new(),
        }),
        Tier::new(TierConfig {
            name: "T1".into(),
            tier_id: 1,
            capacity: 100.0,
            access_latency: 1.0,
            description: String::new(),
        }),
    ];

    let selector = TierSelector::new(tiers);

    // Both tiers empty, equal capacity -> importance 0.0 → tier 0
    assert_eq!(selector.select_tier(0.0), 0);

    // importance 0.5 → tier 0 (cumsum reaches 0.5 at tier 0)
    assert_eq!(selector.select_tier(0.5), 0);

    // importance 0.8 → tier 1
    assert_eq!(selector.select_tier(0.8), 1);
}

#[test]
fn test_hotness_score() {
    let config = HotnessConfig::default();

    let score = hotness_score(100.0, true, 0.5, 0.2, &config);

    // 100.0 * 0.4 + 1.0 * 0.2 + 0.5 * 0.3 + 0.2 * (-0.1)
    // = 40.0 + 0.2 + 0.15 - 0.02
    // = 40.33
    approx::assert_relative_eq!(score, 40.33, epsilon = 1e-5);
}

#[test]
fn test_buffer_env() {
    let configs = vec![
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
    ];

    let env = BufferEnv::new(configs);

    assert_eq!(env.tier_sizes(), vec![0.0, 0.0]);

    let mut tier0 = env.selector().get(0).unwrap().clone();
    tier0.write("blob1", 500.0).unwrap();

    assert_eq!(tier0.current_size(), 500.0);
}

#[test]
fn test_tier_access_count() {
    let config = TierConfig {
        name: "Memory".into(),
        tier_id: 0,
        capacity: 1000.0,
        access_latency: 0.01,
        description: String::new(),
    };

    let mut tier = Tier::new(config);
    tier.write("blob1", 500.0).unwrap();
    tier.write("blob2", 300.0).unwrap();

    // Read operations should increment access count
    tier.read("blob1");
    tier.read("blob1");
    tier.read("blob2");

    assert_eq!(tier.access_count(), 3);
}

#[test]
fn test_tier_utilization() {
    let config = TierConfig {
        name: "Memory".into(),
        tier_id: 0,
        capacity: 1000.0,
        access_latency: 0.01,
        description: String::new(),
    };

    let mut tier = Tier::new(config);
    assert_eq!(tier.utilization(), 0.0);

    tier.write("blob1", 500.0).unwrap();
    approx::assert_relative_eq!(tier.utilization(), 0.5, epsilon = 1e-5);

    tier.write("blob2", 300.0).unwrap();
    approx::assert_relative_eq!(tier.utilization(), 0.8, epsilon = 1e-5);
}

#[test]
fn test_buffer_env_multiple_tiers() {
    let configs = vec![
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
    ];

    let mut env = BufferEnv::new(configs);

    // Write to tier 0
    env.get_tier(0).unwrap().write("blob1", 500.0).unwrap();
    assert_eq!(env.tier_sizes(), vec![500.0, 0.0]);

    // Write to tier 1
    env.get_tier(1).unwrap().write("blob2", 1000.0).unwrap();
    assert_eq!(env.tier_sizes(), vec![500.0, 1000.0]);

    // Test find_blob
    assert_eq!(env.find_blob("blob1"), Some(0));
    assert_eq!(env.find_blob("blob2"), Some(1));
    assert_eq!(env.find_blob("nonexistent"), None);
}

#[test]
fn test_tier_selector_with_different_capacities() {
    let tiers = vec![
        Tier::new(TierConfig {
            name: "Memory".into(),
            tier_id: 0,
            capacity: 300.0,
            access_latency: 0.01,
            description: String::new(),
        }),
        Tier::new(TierConfig {
            name: "NVMe".into(),
            tier_id: 1,
            capacity: 700.0,
            access_latency: 1.0,
            description: String::new(),
        }),
    ];

    let selector = TierSelector::new(tiers);

    // Total capacity = 1000, tier 0 has 30%, tier 1 has 70%
    // importance 0.0 → tier 0 (cumsum 0.3 >= 0.0)
    assert_eq!(selector.select_tier(0.0), 0);

    // importance 0.29 → tier 0 (cumsum 0.3 >= 0.29)
    assert_eq!(selector.select_tier(0.29), 0);

    // importance 0.31 → tier 1 (cumsum 0.3 < 0.31)
    assert_eq!(selector.select_tier(0.31), 1);

    // importance 1.0 → tier 1
    assert_eq!(selector.select_tier(1.0), 1);
}

#[test]
fn test_hotness_score_comparison() {
    let config = HotnessConfig::default();

    // Sequential access should be hotter than non-sequential
    let seq_score = hotness_score(100.0, true, 0.5, 0.2, &config);
    let non_seq_score = hotness_score(100.0, false, 0.5, 0.2, &config);
    assert!(seq_score > non_seq_score);

    // Higher overwrite amount should be hotter
    let high_overwrite = hotness_score(100.0, false, 1.0, 0.2, &config);
    let low_overwrite = hotness_score(100.0, false, 0.5, 0.2, &config);
    assert!(high_overwrite > low_overwrite);

    // More recent (lower recency value) should be hotter since weight is negative
    let recent = hotness_score(100.0, false, 0.5, 0.1, &config);
    let old = hotness_score(100.0, false, 0.5, 1.0, &config);
    assert!(recent > old);
}
