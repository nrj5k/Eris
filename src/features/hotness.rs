/// Configuration for hotness scoring weights
#[derive(Debug, Clone)]
pub struct HotnessConfig {
    /// Weight for offset pattern score
    pub offset_score_weight: f32,
    /// Weight for sequential access indicator
    pub is_sequence_weight: f32,
    /// Weight for overwrite frequency
    pub overwrite_amount_weight: f32,
    /// Weight for recency (negative = colder = less recent)
    pub recency_weight: f32,
}

impl Default for HotnessConfig {
    fn default() -> Self {
        Self {
            offset_score_weight: 0.4,
            is_sequence_weight: 0.2,
            overwrite_amount_weight: 0.3,
            recency_weight: -0.1,
        }
    }
}

/// Compute hotness score for eviction decisions
///
/// Higher score = hotter data = keep in faster tier
/// Lower score = colder data = candidate for demotion
///
/// # Arguments
/// * `offset_score` - Pattern score from offset analysis [0.0, inf)
/// * `is_sequence` - Whether this is a sequential access pattern
/// * `overwrite_amount` - Number/frequency of overwrites [0.0, inf)
/// * `recency` - Recency factor (typically normalized age) [0.0, inf)
/// * `config` - Weight configuration
///
/// # Returns
/// Hotness score for the blob
pub fn hotness_score(
    offset_score: f32,
    is_sequence: bool,
    overwrite_amount: f32,
    recency: f32,
    config: &HotnessConfig,
) -> f32 {
    let sequence_value = if is_sequence { 1.0 } else { 0.0 };

    offset_score * config.offset_score_weight
        + sequence_value * config.is_sequence_weight
        + overwrite_amount * config.overwrite_amount_weight
        + recency * config.recency_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = HotnessConfig::default();
        approx::assert_relative_eq!(config.offset_score_weight, 0.4, epsilon = 1e-5);
        approx::assert_relative_eq!(config.is_sequence_weight, 0.2, epsilon = 1e-5);
        approx::assert_relative_eq!(config.overwrite_amount_weight, 0.3, epsilon = 1e-5);
        approx::assert_relative_eq!(config.recency_weight, -0.1, epsilon = 1e-5);
    }

    #[test]
    fn test_hotness_score_sequential() {
        let config = HotnessConfig::default();

        let score = hotness_score(100.0, true, 0.5, 0.2, &config);

        // 100.0 * 0.4 + 1.0 * 0.2 + 0.5 * 0.3 + 0.2 * (-0.1)
        // = 40.0 + 0.2 + 0.15 - 0.02
        // = 40.33
        approx::assert_relative_eq!(score, 40.33, epsilon = 1e-5);
    }

    #[test]
    fn test_hotness_score_non_sequential() {
        let config = HotnessConfig::default();

        let score = hotness_score(100.0, false, 0.5, 0.2, &config);

        // 100.0 * 0.4 + 0.0 * 0.2 + 0.5 * 0.3 + 0.2 * (-0.1)
        // = 40.0 + 0.0 + 0.15 - 0.02
        // = 40.13
        approx::assert_relative_eq!(score, 40.13, epsilon = 1e-5);
    }

    #[test]
    fn test_hotness_score_zero() {
        let config = HotnessConfig::default();

        let score = hotness_score(0.0, false, 0.0, 0.0, &config);
        approx::assert_relative_eq!(score, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hotness_score_high_weights() {
        let config = HotnessConfig {
            offset_score_weight: 1.0,
            is_sequence_weight: 0.0,
            overwrite_amount_weight: 0.0,
            recency_weight: 0.0,
        };

        let score = hotness_score(50.0, false, 0.0, 0.0, &config);
        approx::assert_relative_eq!(score, 50.0, epsilon = 1e-5);
    }

    #[test]
    fn test_hotness_score_negative_recency() {
        let config = HotnessConfig::default();

        // Older data (higher recency value) should have lower score due to negative weight
        let score_recent = hotness_score(100.0, true, 0.5, 0.1, &config);
        let score_old = hotness_score(100.0, true, 0.5, 1.0, &config);

        assert!(score_recent > score_old);
    }

    #[test]
    fn test_hotness_score_custom_weights() {
        let config = HotnessConfig {
            offset_score_weight: 0.5,
            is_sequence_weight: 0.1,
            overwrite_amount_weight: 0.2,
            recency_weight: 0.2,
        };

        let score = hotness_score(10.0, true, 5.0, 0.5, &config);

        // 10.0 * 0.5 + 1.0 * 0.1 + 5.0 * 0.2 + 0.5 * 0.2
        // = 5.0 + 0.1 + 1.0 + 0.1
        // = 6.2
        approx::assert_relative_eq!(score, 6.2, epsilon = 1e-5);
    }
}
