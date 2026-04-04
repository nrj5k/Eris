use indicatif::{ProgressBar, ProgressStyle};

use crate::config::TierConfig;

/// Callback trait for training monitoring.
///
/// Implement this trait to receive callbacks during training.
/// All methods have default empty implementations, so you can
/// implement only the callbacks you need.
///
/// # Example
///
/// ```
/// use eris::training::TrainingMonitor;
///
/// struct MyMonitor;
///
/// impl TrainingMonitor for MyMonitor {
///     fn on_episode_end(&mut self, episode: usize, reward: f32, epsilon: f32, tier_states: &[f32]) {
///         println!("Episode {}: reward={:.1}, ε={:.3}", episode, reward, epsilon);
///     }
/// }
/// ```
pub trait TrainingMonitor: Send + Sync {
    /// Called at the start of each episode.
    ///
    /// # Arguments
    ///
    /// * `episode` - Current episode number (0-indexed)
    /// * `total_episodes` - Total number of episodes to run
    fn on_episode_start(&mut self, _episode: usize, _total_episodes: usize) {}

    /// Called at the end of each episode.
    ///
    /// # Arguments
    ///
    /// * `episode` - Episode number that just completed (0-indexed)
    /// * `reward` - Total reward for this episode
    /// * `epsilon` - Current exploration rate
    /// * `tier_states` - Utilization states for each tier
    fn on_episode_end(
        &mut self,
        _episode: usize,
        _reward: f32,
        _epsilon: f32,
        _tier_states: &[f32],
    ) {
    }

    /// Called after each training step.
    ///
    /// # Arguments
    ///
    /// * `step` - Training step number
    /// * `loss` - Loss value for this step
    fn on_step(&mut self, _step: usize, _loss: f32) {}
}

/// Console progress monitor using indicatif progress bar.
///
/// Displays a progress bar with episode count, elapsed time,
/// reward, and exploration rate.
///
/// # Example
///
/// ```
/// use eris::training::{ConsoleMonitor, TrainingMonitor};
///
/// let mut monitor = ConsoleMonitor::new(100);
/// monitor.on_episode_end(0, 50.0, 0.95, &[0.5, 0.3]);
/// ```
pub struct ConsoleMonitor {
    progress: ProgressBar,
    total_episodes: usize,
    tier_configs: Option<Vec<TierConfig>>,
}

impl ConsoleMonitor {
    /// Create a new console monitor for the given number of episodes.
    ///
    /// # Arguments
    ///
    /// * `total_episodes` - Total number of episodes to display
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::ConsoleMonitor;
    ///
    /// let monitor = ConsoleMonitor::new(100);
    /// ```
    pub fn new(total_episodes: usize) -> Self {
        let progress = ProgressBar::new(total_episodes as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} episodes ({eta})\n  {msg}")
                .unwrap()
        );
        Self {
            progress,
            total_episodes,
            tier_configs: None,
        }
    }

    /// Set tier configurations for detailed display.
    ///
    /// # Arguments
    ///
    /// * `configs` - Tier configurations
    ///
    /// # Example
    ///
    /// ```
    /// use eris::training::ConsoleMonitor;
    /// use eris::config::TierConfig;
    ///
    /// let monitor = ConsoleMonitor::new(100)
    ///     .with_tier_configs(vec![
    ///         TierConfig::default(),
    ///     ]);
    /// ```
    pub fn with_tier_configs(mut self, configs: Vec<TierConfig>) -> Self {
        self.tier_configs = Some(configs);
        self
    }
}

impl TrainingMonitor for ConsoleMonitor {
    fn on_episode_start(&mut self, episode: usize, _total_episodes: usize) {
        if episode == 0 {
            self.progress
                .set_message("Starting training...".to_string());
        }
    }

    fn on_episode_end(&mut self, episode: usize, reward: f32, epsilon: f32, tier_states: &[f32]) {
        let tier_display = if !tier_states.is_empty() {
            format!(
                " | Tiers: {:.1}%",
                tier_states.iter().sum::<f32>() * 100.0 / tier_states.len() as f32
            )
        } else {
            String::new()
        };

        self.progress.set_message(format!(
            "Reward: {:.1} | ε: {:.3}{}",
            reward, epsilon, tier_display
        ));
        self.progress.inc(1);

        // Print detailed tier bars every 10 episodes (excluding episode 0)
        // Only if we have data and configs
        if !tier_states.is_empty()
            && self.tier_configs.is_some()
            && episode > 0
            && episode % 10 == 0
        {
            // Verify we have actual tier data (not all zeros)
            let has_data = tier_states.iter().any(|&x| x > 0.0);
            if has_data {
                self.progress.println("");
                self.progress.println(&format_tiers(
                    self.tier_configs.as_ref().unwrap(),
                    tier_states,
                ));
            }
        }
    }
}

impl Drop for ConsoleMonitor {
    fn drop(&mut self) {
        self.progress.finish();
    }
}

/// Format tier utilization as a visual bar chart.
///
/// Creates a string representation of tier states with:
/// - Tier name
/// - Visual bar chart (filled/unfilled blocks)
/// - Percentage utilization
///
/// # Arguments
///
/// * `tier_configs` - Configuration for each tier
/// * `states` - Utilization state for each tier (0.0 to 1.0)
///
/// # Returns
///
/// Formatted string with one line per tier
///
/// # Example
///
/// ```
/// use eris::tier::TierConfig;
/// use eris::training::format_tiers;
///
/// let tiers = vec![
///     TierConfig { name: "Tier1".into(), ..Default::default() },
///     TierConfig { name: "Tier2".into(), ..Default::default() },
/// ];
/// let states = vec![0.7, 0.3];
///
/// let output = format_tiers(&tiers, &states);
/// assert!(output.contains("Tier1"));
/// assert!(output.contains("70.0%"));
/// ```
pub fn format_tiers(tier_configs: &[TierConfig], states: &[f32]) -> String {
    let mut output = String::new();
    for (tier, &util) in tier_configs.iter().zip(states.iter()) {
        let filled = (util * 20.0) as usize;
        let empty = 20 - filled;
        output.push_str(&format!(
            "  {:8} [{}{}] {:5.1}%\n",
            tier.name,
            "█".repeat(filled),
            "░".repeat(empty),
            util * 100.0
        ));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TierConfig;

    #[test]
    fn test_format_tiers_empty() {
        let tiers = vec![];
        let states = vec![];
        let output = format_tiers(&tiers, &states);
        assert!(output.is_empty());
    }

    #[test]
    fn test_format_tiers_basic() {
        let tiers = vec![
            TierConfig {
                name: "Fast".into(),
                tier_id: 0,
                capacity: 1_000_000.0,
                access_latency: 1.0,
                description: "Fast tier".into(),
            },
            TierConfig {
                name: "Slow".into(),
                tier_id: 1,
                capacity: 10_000_000.0,
                access_latency: 100.0,
                description: "Slow tier".into(),
            },
        ];
        let states = vec![0.5, 1.0];
        let output = format_tiers(&tiers, &states);

        assert!(output.contains("Fast"));
        assert!(output.contains("Slow"));
        assert!(output.contains("50.0%"));
        assert!(output.contains("100.0%"));
    }
}
