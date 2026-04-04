use crate::config_old::TierConfig;
use crate::env::{Environment, Info, StepResult};
use crate::error::{EnvError, Result};
use crate::features::{AccessRecord, AccessTracker, BlobFeatures, HotnessConfig};
use crate::models::decode_action;
use crate::space::{BoxSpace, DiscreteSpace};
use crate::tier::BufferEnv;
use crate::trace::{BlobData, TraceReader};

/// Maximum number of tiers in the hierarchy
pub const MAX_TIERS: usize = 5;

/// I/O Buffer Environment for reinforcement learning.
///
/// This environment simulates a multi-tier storage system where:
/// - Each tier has a capacity and latency
/// - The agent must decide where to place/read blobs
/// - Rewards are based on latency and capacity constraints
#[derive(Debug, Clone)]
pub struct IOBufferEnv {
    /// Configuration for each tier
    tier_configs: Vec<TierConfig>,
    /// Maximum steps per episode
    max_steps: usize,
    /// Multi-tier storage buffer
    buffer: BufferEnv,
    /// Trace reader for blob data
    trace: TraceReader,
    /// Current step count
    current_step: usize,
    /// Access history tracker
    tracker: AccessTracker,
    /// Current timestamp (ms)
    current_time_ms: u64,
    /// Current blob to process
    current_blob: Option<BlobData>,
    /// Hotness scoring configuration
    hotness_config: HotnessConfig,
    /// Random number generator for the environment
    random: rand_pcg::Pcg64,
    /// Maximum blob size for normalization
    max_blob_size: f64,
    /// Maximum access frequency for normalization
    max_frequency: u32,
}

impl IOBufferEnv {
    /// Create a new I/O buffer environment.
    ///
    /// # Arguments
    /// * `config_path` - Path to tier configuration TOML file
    /// * `trace_path` - Path to CSV trace file
    /// * `max_steps` - Maximum steps per episode
    ///
    /// # Errors
    /// Returns error if config or trace files cannot be loaded.
    pub fn new(
        config_path: &std::path::Path,
        trace_path: &std::path::Path,
        max_steps: usize,
    ) -> Result<Self> {
        use crate::config::Config;
        use rand::SeedableRng;

        let config = Config::from_file(config_path)?;
        let trace = TraceReader::from_csv(trace_path)?;

        let tier_configs = config.tier.clone();
        let buffer = BufferEnv::new(config.tier);

        // Estimate max blob size from first few records (for normalization)
        let max_blob_size = 2_000_000.0; // Default max 2MB
        let max_frequency = 100; // Default max frequency

        Ok(Self {
            tier_configs,
            max_steps,
            buffer,
            trace,
            current_step: 0,
            tracker: AccessTracker::new(10_000),
            current_time_ms: 0,
            current_blob: None,
            hotness_config: HotnessConfig::default(),
            random: rand_pcg::Pcg64::seed_from_u64(42),
            max_blob_size,
            max_frequency,
        })
    }

    /// Process the current blob with the given action.
    ///
    /// # Returns
    /// Reward for the action (negative for latency penalties)
    fn process_action(&mut self, action: usize) -> Result<f64> {
        let (tier_idx, op_type) = decode_action(action);

        // Extract blob data before mutating
        let blob = match self.current_blob.as_ref() {
            Some(b) => b.clone(),
            None => return Err(EnvError::TraceExhausted),
        };

        let blob_id = blob.offset_id.clone();
        let size = blob.offset_size;

        // Update access tracker
        self.tracker.record(AccessRecord {
            blob_id: blob_id.clone(),
            timestamp_ms: self.current_time_ms,
            access_type: blob.io_op_enum(),
            size,
        });

        let reward = match op_type {
            0 => self.handle_read(tier_idx, &blob_id, size)?, // Read (action encoding: op_type 0 = read)
            1 => self.handle_write(tier_idx, &blob_id, size)?, // Write (action encoding: op_type 1 = write)
            _ => unreachable!("Invalid op_type: {}", op_type),
        };

        Ok(reward)
    }

    /// Handle write operation to a specific tier.
    ///
    /// Attempts to write blob to the specified tier. If tier is full,
    /// attempts cascading demotion to lower tiers.
    fn handle_write(&mut self, tier_idx: usize, blob_id: &str, size: f64) -> Result<f64> {
        // Validate tier index
        if tier_idx >= self.buffer.num_tiers() {
            return Err(EnvError::InvalidTierIndex {
                provided: tier_idx,
                max: self.buffer.num_tiers() - 1,
            });
        }

        // Attempt to write to specified tier
        if let Some(tier) = self.buffer.get_tier(tier_idx) {
            if tier.write(blob_id, size).is_ok() {
                return Ok(0.0); // Success, no latency penalty for write
            }
        }

        // Tier is full, attempt cascading demotion
        let mut current_tier_idx = tier_idx;
        while current_tier_idx < self.buffer.num_tiers() - 1 {
            current_tier_idx += 1;

            // Find lowest hotness item in current tier for eviction
            if let Some((evict_id, evict_size)) = self.find_eviction_candidate(current_tier_idx)? {
                // Remove from current tier
                if let Some(tier) = self.buffer.get_tier(current_tier_idx) {
                    tier.remove(&evict_id);
                }

                // Write to next tier down (if not last tier)
                if current_tier_idx < self.buffer.num_tiers() - 1 {
                    if let Some(next_tier) = self.buffer.get_tier(current_tier_idx + 1) {
                        let _ = next_tier.write(&evict_id, evict_size);
                    }
                }

                // Retry write to target tier
                if let Some(target_tier) = self.buffer.get_tier(tier_idx) {
                    if target_tier.write(blob_id, size).is_ok() {
                        return Ok(0.0);
                    }
                }
            } else {
                break;
            }
        }

        // All tiers full, write to last resort (Tapes)
        let last_tier = self.buffer.num_tiers() - 1;
        if let Some(tape) = self.buffer.get_tier(last_tier) {
            tape.write(blob_id, size)?;
            // High latency cost for tape storage
            return Ok(0.0);
        }

        // Write failed
        Ok(-10000.0)
    }

    /// Handle read operation from a specific tier.
    ///
    /// Searches all tiers for the blob and returns latency-based penalty.
    fn handle_read(&mut self, _tier_idx: usize, blob_id: &str, _size: f64) -> Result<f64> {
        // Search all tiers for the blob
        for i in 0..self.buffer.num_tiers() {
            if let Some(tier) = self.buffer.get_tier_ref(i) {
                if tier.contains(blob_id) {
                    // Found, return negative latency as penalty
                    let latency = tier.config.access_latency;
                    return Ok(-(latency as f64));
                }
            }
        }

        // Blob not found, return large penalty
        Ok(-10000.0)
    }

    /// Find eviction candidate with lowest hotness score in the tier.
    fn find_eviction_candidate(&self, tier_idx: usize) -> Result<Option<(String, f64)>> {
        let tier = self
            .buffer
            .get_tier_ref(tier_idx)
            .ok_or(EnvError::InvalidTierIndex {
                provided: tier_idx,
                max: self.buffer.num_tiers() - 1,
            })?;

        let mut lowest_score = f32::INFINITY;
        let mut evict_candidate: Option<(String, f64)> = None;

        // Get all blob IDs in this tier (need to clone to avoid borrow issues)
        let blob_ids: Vec<String> = tier.storage_keys();

        for blob_id in &blob_ids {
            // Calculate hotness score
            let recency = self.tracker.get_recency(blob_id, self.current_time_ms);
            let frequency = self.tracker.get_frequency(blob_id);

            // Hotness score: lower recency + higher frequency = higher score
            // We want to evict low-score items
            let score = if recency.is_finite() {
                recency / (1.0 + frequency as f32)
            } else {
                f32::INFINITY // Never accessed, evict immediately
            };

            if score < lowest_score {
                lowest_score = score;
                if let Some(size) = tier.read(blob_id) {
                    evict_candidate = Some((blob_id.clone(), size));
                }
            }
        }

        Ok(evict_candidate)
    }

    /// Get current observation: [5 tier sizes + 10 features]
    fn get_observation(&self) -> Result<Vec<f64>> {
        let blob = self.current_blob.as_ref().ok_or(EnvError::TraceExhausted)?;

        // Get tier sizes (normalized)
        let tier_sizes = self.buffer.tier_sizes();

        // Extract blob features
        let features = BlobFeatures::extract(
            blob,
            &self.tracker,
            self.current_time_ms,
            self.max_blob_size,
            self.max_frequency,
        );

        // Encode state
        let state = crate::features::encode_state(&tier_sizes, &features, &self.tier_configs);

        // Convert f32 to f64
        Ok(state.iter().map(|&f| f as f64).collect())
    }

    /// Calculate total penalty from tier accesses (for metrics).
    #[allow(dead_code)]
    fn calculate_penalty(&self) -> f32 {
        let mut penalty = 0.0;
        for i in 0..self.buffer.num_tiers() {
            if let Some(tier) = self.buffer.get_tier_ref(i) {
                let count = tier.access_count();
                let latency = tier.config.access_latency;
                penalty += count as f32 * latency;
            }
        }
        -penalty // Negative reward
    }

    /// Load next blob from trace.
    fn advance_trace(&mut self) -> Result<()> {
        match self.trace.next() {
            Some(blob) => {
                self.current_blob = Some(blob.clone());
                self.current_time_ms += 1; // Increment time step
                Ok(())
            }
            None => {
                self.current_blob = None;
                Err(EnvError::TraceExhausted)
            }
        }
    }

    /// Compute observation dimension dynamically
    pub fn observation_dim(&self) -> usize {
        // Number of tiers + 10 blob features
        self.tier_configs.len() + 10
    }

    /// Get tier utilization states [0.0, 1.0] for all tiers
    pub fn get_tier_utilization(&self) -> Vec<f32> {
        self.buffer.get_state()
    }
}

/// Simple API for environment interaction
impl IOBufferEnv {
    /// Reset the environment and return initial observation
    pub fn reset(&mut self) -> Vec<f64> {
        // Reset trace
        self.trace.reset();

        // Reset buffer
        self.buffer.reset();

        // Reset tracker
        self.tracker.clear();

        // Reset state
        self.current_step = 0;
        self.current_time_ms = 0;
        self.current_blob = None;

        // Load first blob
        let _ = self.advance_trace();

        // Get initial observation
        self.get_observation()
            .unwrap_or_else(|_| vec![0.0; self.observation_dim()])
    }

    /// Take a step in the environment
    ///
    /// # Returns
    /// (observation, reward, done)
    pub fn step(&mut self, action: usize) -> (Vec<f64>, f64, bool) {
        // Process action
        let reward = match self.process_action(action) {
            Ok(r) => r,
            Err(_) => -10000.0, // Error penalty
        };

        // Advance to next blob
        let done = self.advance_trace().is_err();

        // Get observation
        let observation = match self.get_observation() {
            Ok(obs) => obs,
            Err(_) => vec![0.0; self.observation_dim()],
        };

        self.current_step += 1;

        // Check if max steps reached
        let terminated = done || self.current_step >= self.max_steps;

        (observation, reward, terminated)
    }
}

/// Environment trait implementation for RL frameworks
impl Environment for IOBufferEnv {
    type Observation = Vec<f64>;
    type Action = usize;

    fn reset(&mut self) -> Self::Observation {
        // Delegate to existing reset method
        IOBufferEnv::reset(self)
    }

    fn step(&mut self, action: Self::Action) -> StepResult {
        // Use existing step logic
        let (observation, reward, done) = IOBufferEnv::step(self, action);

        StepResult {
            observation,
            reward,
            done,
            info: Info::new()
                .with_metric("current_step", self.current_step as f64)
                .with_metric("current_time_ms", self.current_time_ms as f64),
        }
    }

    fn observation_space(&self) -> BoxSpace {
        // Observation is normalized tier sizes + normalized blob features
        // All values in [0, 1]
        BoxSpace::uniform(self.observation_dim(), 0.0, 1.0)
    }

    fn action_space(&self) -> DiscreteSpace {
        // Number of tiers × 2 operations (read/write)
        DiscreteSpace::new(self.tier_configs.len() * 2)
    }

    fn seed(&mut self, seed: u64) {
        use rand::SeedableRng;
        self.random = rand_pcg::Pcg64::seed_from_u64(seed);
    }

    fn get_tier_utilization(&self) -> Vec<f32> {
        IOBufferEnv::get_tier_utilization(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn create_test_env() -> Option<IOBufferEnv> {
        let config_path = Path::new("config/tiers.toml");
        let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");

        if !config_path.exists() || !trace_path.exists() {
            return None;
        }

        IOBufferEnv::new(config_path, trace_path, 100).ok()
    }

    #[test]
    fn test_env_creation() {
        if let Some(env) = create_test_env() {
            assert_eq!(env.max_steps, 100);
            assert_eq!(env.buffer.num_tiers(), 5);
        } else {
            eprintln!("Skipping test: config or trace files not found");
        }
    }

    #[test]
    fn test_env_reset() {
        if let Some(mut env) = create_test_env() {
            let obs = env.reset();
            // Observation should have dynamic dimension based on tier count + features
            assert_eq!(obs.len(), env.observation_dim());
        } else {
            eprintln!("Skipping test: config or trace files not found");
        }
    }

    #[test]
    fn test_env_step() {
        if let Some(mut env) = create_test_env() {
            env.reset();

            // Take a step with action 0 (tier 0, write)
            let (obs, _reward, done) = env.step(0);

            assert_eq!(obs.len(), env.observation_dim());
            assert!(!done);
        } else {
            eprintln!("Skipping test: config or trace files not found");
        }
    }

    #[test]
    fn test_env_episode() {
        if let Some(mut env) = create_test_env() {
            env.reset();

            let mut total_reward = 0.0_f64;
            let mut steps = 0;

            for _ in 0..5 {
                use rand::prelude::*;
                use rand::rng;
                let mut rng = rng();
                let action: usize = rng.random_range(0..10); // Random action
                let (obs, reward, done) = env.step(action);

                total_reward += reward;
                steps += 1;

                if done {
                    break;
                }
            }

            assert!(steps <= 5);
            println!(
                "Episode completed: {} steps, total reward: {}",
                steps, total_reward
            );
        } else {
            eprintln!("Skipping test: config or trace files not found");
        }
    }

    #[test]
    fn test_action_encoding() {
        // Test that we can decode actions correctly
        for action_idx in 0..10 {
            let (tier_idx, op_type) = decode_action(action_idx);
            assert!(tier_idx < MAX_TIERS, "Tier index out of bounds");
            assert!(op_type < 2, "Op type out of bounds");

            // Verify round-trip
            let encoded = crate::models::encode_action(tier_idx, op_type);
            assert_eq!(encoded, action_idx);
        }
    }

    #[test]
    fn test_action_encoding_decoding_comprehensive() {
        // Test all possible action combinations
        for tier_idx in 0..MAX_TIERS {
            for op_type in 0..2 {
                let action = crate::models::encode_action(tier_idx, op_type);
                let (decoded_tier, decoded_op) = decode_action(action);

                assert_eq!(
                    tier_idx, decoded_tier,
                    "Tier decoding mismatch for action {}",
                    action
                );
                assert_eq!(
                    op_type, decoded_op,
                    "Op decoding mismatch for action {}",
                    action
                );
            }
        }
    }

    #[test]
    fn test_action_operation_mapping() {
        // Test that op_type 0 = read and op_type 1 = write
        // This is CRITICAL: op_type encoding must match our processing logic
        for tier_idx in 0..MAX_TIERS {
            // op_type 0 = read
            let read_action = crate::models::encode_action(tier_idx, 0);
            let (decoded_tier, decoded_op) = decode_action(read_action);
            assert_eq!(decoded_op, 0, "Read operation should decode to op_type 0");
            assert_eq!(decoded_tier, tier_idx);

            // op_type 1 = write
            let write_action = crate::models::encode_action(tier_idx, 1);
            let (decoded_tier, decoded_op) = decode_action(write_action);
            assert_eq!(decoded_op, 1, "Write operation should decode to op_type 1");
            assert_eq!(decoded_tier, tier_idx);
        }
    }

    #[test]
    fn test_tier_configs() {
        if let Some(env) = create_test_env() {
            assert_eq!(env.tier_configs.len(), MAX_TIERS);

            // Check tier IDs are sequential
            for (i, config) in env.tier_configs.iter().enumerate() {
                assert_eq!(config.tier_id, i as u32);
            }
        } else {
            eprintln!("Skipping test: config or trace files not found");
        }
    }
}
