use crate::config_old::TierConfig;
use crate::env::{Environment, Info, StepResult};
use crate::error::{EnvError, Result};
use crate::features::{AccessRecord, AccessTracker, BlobFeatures, HotnessConfig};
use crate::models::decode_action;
use crate::space::{BoxSpace, DiscreteSpace};
use crate::tier::BufferEnv;
use crate::trace::{BlobData, TraceData, TraceFormat, TraceReader};
use std::sync::Arc;

/// Worst case latency (tape) for normalization
const WORST_LATENCY: f64 = 1_000_000.0; // 1,000,000 ms
const MISS_PENALTY: f64 = -0.5; // Penalty for cache miss

/// Calculate normalized reward based on relative savings.
///
/// Range: 0.0 (tape/worst) to 1.0 (memory/best)
/// Formula: (worst_latency - actual_latency) / worst_latency
fn calculate_latency_reward(latency: f64) -> f64 {
    let savings = (WORST_LATENCY - latency) / WORST_LATENCY;
    savings.max(0.0).min(1.0) // Clamp to [0, 1]
}

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
    /// Trace reader for blob data (shares loaded data via Arc<TraceData>, but has independent position tracking)
    trace: TraceReader,
    /// Current step count
    current_step: usize,
    /// Access history tracker
    tracker: AccessTracker,
    /// Current timestamp (ms)
    current_time_ms: u64,
    /// Current blob to process
    current_blob: Option<BlobData>,
    /// Hotness scoring configuration (reserved for future use)
    #[allow(dead_code)]
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
    /// * `trace_path` - Path to trace file (CSV or other formats)
    /// * `format` - Format of the trace file (e.g., CSV, Parquet, Autodetect)
    /// * `max_steps` - Maximum steps per episode
    /// * `max_blob_size` - Optional max blob size for normalization (default: 2_000_000.0)
    /// * `max_frequency` - Optional max access frequency for normalization (default: 100)
    ///
    /// # Errors
    /// Returns error if config or trace files cannot be loaded.
    pub fn new(
        config_path: &std::path::Path,
        trace_path: &std::path::Path,
        format: TraceFormat,
        max_steps: usize,
        max_blob_size: Option<f64>,
        max_frequency: Option<u32>,
    ) -> Result<Self> {
        use crate::config::Config;
        use rand::SeedableRng;

        let config = Config::from_file(config_path)?;
        let trace = TraceReader::from_path(trace_path, format)?;

        // Report loading statistics
        println!(
            "✓ Environment initialized with {} trace records",
            trace.total_records()
        );
        if trace.skipped_records() > 0 {
            println!("  ({} malformed rows skipped)", trace.skipped_records());
        }

        let tier_configs = config.tier.clone();
        let buffer = BufferEnv::new(config.tier);

        // Use provided values or defaults for normalization
        let max_blob_size = max_blob_size.unwrap_or(2_000_000.0);
        let max_frequency = max_frequency.unwrap_or(100);

        Ok(Self {
            tier_configs,
            max_steps,
            buffer,
            trace,
            current_step: 0,
            tracker: AccessTracker::new(1000), // Reduced from 10_000 to avoid stack overflow
            current_time_ms: 0,
            current_blob: None,
            hotness_config: HotnessConfig::default(),
            random: rand_pcg::Pcg64::seed_from_u64(42),
            max_blob_size,
            max_frequency,
        })
    }

    /// Create environment with shared trace data (for VecEnv).
    ///
    /// This constructor accepts shared `Arc<TraceData>` to avoid loading
    /// the CSV file multiple times when creating multiple environments.
    /// Each environment has independent position tracking.
    ///
    /// # Arguments
    /// * `config_path` - Path to tier configuration TOML file
    /// * `trace_data` - Shared trace data (loaded once, shared across envs)
    /// * `max_steps` - Maximum steps per episode
    /// * `max_blob_size` - Optional max blob size for normalization (default: 2_000_000.0)
    /// * `max_frequency` - Optional max access frequency for normalization (default: 100)
    ///
    /// # Errors
    /// Returns error if config file cannot be loaded.
    pub fn with_shared_trace(
        config_path: &std::path::Path,
        trace_data: Arc<TraceData>,
        max_steps: usize,
        max_blob_size: Option<f64>,
        max_frequency: Option<u32>,
    ) -> Result<Self> {
        use crate::config::Config;
        use rand::SeedableRng;

        let config = Config::from_file(config_path)?;
        let tier_configs = config.tier.clone();
        let buffer = BufferEnv::new(config.tier);

        // Create a new TraceReader with shared data but independent position tracking
        let trace = TraceReader::from_shared_data(trace_data);

        // Use provided values or defaults for normalization
        let max_blob_size = max_blob_size.unwrap_or(2_000_000.0);
        let max_frequency = max_frequency.unwrap_or(100);

        Ok(Self {
            tier_configs,
            max_steps,
            buffer,
            trace,
            current_step: 0,
            tracker: AccessTracker::new(1000),
            current_time_ms: 0,
            current_blob: None,
            hotness_config: HotnessConfig::default(),
            random: rand_pcg::Pcg64::seed_from_u64(42),
            max_blob_size,
            max_frequency,
        })
    }

    /// Get max steps per episode
    pub fn get_max_steps(&self) -> usize {
        self.max_steps
    }

    /// Get observation dimension (warp-aligned for GPU optimization)
    pub fn observation_dim(&self) -> usize {
        32 // Warp-aligned dimension (5 tier sizes + 10 features + 17 padding)
    }

    /// Get action dimension
    pub fn action_dim(&self) -> usize {
        self.action_space().n
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
    /// evicts coldest blob from TARGET tier with cascading demotion.
    fn handle_write(&mut self, tier_idx: usize, blob_id: &str, size: f64) -> Result<f64> {
        // Validate tier index
        if tier_idx >= self.buffer.num_tiers() {
            return Err(EnvError::InvalidTierIndex {
                provided: tier_idx,
                max: self.buffer.num_tiers() - 1,
            });
        }

        // Fast path: try direct write
        if let Some(tier) = self.buffer.get_tier(tier_idx) {
            if tier.write(blob_id, size).is_ok() {
                let latency = self.tier_configs[tier_idx].access_latency as f64;
                return Ok(calculate_latency_reward(latency));
            }
        }

        // Slow path: evict from TARGET tier with cascading demotion
        let max_attempts = self.buffer.num_tiers() * 2;

        for _ in 0..max_attempts {
            // Find coldest blob in TARGET tier (not tier_idx+1)
            let candidate = self.find_eviction_candidate(tier_idx)?;

            match candidate {
                Some((evict_id, evict_size)) => {
                    // Remove from target tier
                    let removed = self
                        .buffer
                        .get_tier(tier_idx)
                        .map_or(false, |t| t.remove(&evict_id));

                    if !removed {
                        break;
                    }

                    // Try to demote evicted blob to lower tiers
                    let demoted = self.cascade_demote(&evict_id, evict_size, tier_idx + 1);

                    if !demoted {
                        // Demotion failed - RESTORE to avoid data loss
                        if let Some(tier) = self.buffer.get_tier(tier_idx) {
                            let _ = tier.write(&evict_id, evict_size);
                        }
                        break;
                    }

                    // Retry original write
                    if let Some(tier) = self.buffer.get_tier(tier_idx) {
                        if tier.write(blob_id, size).is_ok() {
                            let latency = self.tier_configs[tier_idx].access_latency as f64;
                            return Ok(calculate_latency_reward(latency));
                        }
                    }
                    // Still full - loop to evict another blob
                }
                None => break, // No evictable blobs
            }
        }

        // Fallback: write to last tier with penalty
        let last = self.buffer.num_tiers() - 1;
        if let Some(tape) = self.buffer.get_tier(last) {
            if tape.write(blob_id, size).is_ok() {
                let latency = self.tier_configs[last].access_latency as f64;
                return Ok(calculate_latency_reward(latency));
            }
        }

        Ok(-10000.0)
    }

    /// Cascade demote a blob starting from start_tier downward.
    ///
    /// Attempts to place the blob in successive tiers until successful.
    /// Returns true if placed, false if all tiers are full.
    fn cascade_demote(&mut self, blob_id: &str, size: f64, start_tier: usize) -> bool {
        let mut tier_idx = start_tier;

        while tier_idx < self.buffer.num_tiers() {
            if let Some(tier) = self.buffer.get_tier(tier_idx) {
                if tier.write(blob_id, size).is_ok() {
                    return true;
                }
            }
            tier_idx += 1;
        }

        false // Could not place anywhere
    }

    /// Handle read operation with smart cache lookup.
    ///
    /// The agent specifies a preferred tier via `_tier_idx`, but the system
    /// searches ALL tiers from fastest to slowest to find the blob. This
    /// mimics real storage systems where the controller knows where data lives.
    ///
    /// Reward is based on the ACTUAL tier where the blob was found, not the
    /// requested tier. This encourages the agent to place hot data in fast tiers
    /// since it will be found there regardless of which tier the agent queries.
    ///
    /// # Returns
    /// Normalized reward [0.0, 1.0] based on tier latency, or MISS_PENALTY if not found.
    fn handle_read(&mut self, _tier_idx: usize, blob_id: &str, _size: f64) -> Result<f64> {
        // Search all tiers for the blob
        for i in 0..self.buffer.num_tiers() {
            if let Some(tier) = self.buffer.get_tier_ref(i) {
                if tier.contains(blob_id) {
                    // Found, return normalized reward
                    let latency = tier.config.access_latency as f64;
                    return Ok(calculate_latency_reward(latency));
                }
            }
        }

        // Blob not found, return miss penalty
        Ok(MISS_PENALTY)
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

            // Hotness score: higher frequency + lower recency = higher score
            // We want to evict low-score items
            let score = if recency.is_finite() {
                (frequency as f32 + 1.0) / (recency + 1.0)
            } else {
                0.0 // Never accessed = coldest = evict immediately
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
            action,
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
        // Phase 01: Warp-aligned to 32 for GPU optimization
        let dim = 32; // tier_configs.len() + 10 padded to warp size
        BoxSpace::uniform(dim, 0.0, 1.0)
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

    /// Maximum number of tiers in the hierarchy (test-only constant)
    const MAX_TIERS: usize = 5;

    fn create_test_env() -> Option<IOBufferEnv> {
        let config_path = Path::new("config/tiers.toml");
        let trace_path = Path::new("recorder-csv/NWChem-64_combined.csv");

        if !config_path.exists() || !trace_path.exists() {
            return None;
        }

        IOBufferEnv::new(
            config_path,
            trace_path,
            TraceFormat::Autodetect,
            100,
            None,
            None,
        )
        .ok()
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
                let (_obs, reward, _done) = env.step(action);

                total_reward += reward;
                steps += 1;

                if _done {
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

    #[test]
    fn test_calculate_latency_reward() {
        // Test reward function produces values in [0, 1] range

        // Worst case (tape) - should be near 0
        let tape_latency = 1_000_000.0;
        let tape_reward = calculate_latency_reward(tape_latency);
        assert!((0.0..=1.0).contains(&tape_reward));
        assert!(
            tape_reward < 0.01,
            "Tape reward should be near 0: {}",
            tape_reward
        );

        // Best case (memory) - should be near 1
        let memory_latency = 0.01;
        let memory_reward = calculate_latency_reward(memory_latency);
        assert!((0.0..=1.0).contains(&memory_reward));
        assert!(
            memory_reward > 0.99,
            "Memory reward should be near 1: {}",
            memory_reward
        );

        // Middle tier (SSD) - should be in between
        let ssd_latency = 100.0;
        let ssd_reward = calculate_latency_reward(ssd_latency);
        assert!((0.0..=1.0).contains(&ssd_reward));
        assert!(ssd_reward > tape_reward && ssd_reward < memory_reward);

        // Verify MISS_PENALTY is used for cache misses
        assert!((-1.0..=0.0).contains(&MISS_PENALTY));
    }

    #[test]
    fn test_reward_range() {
        // Test that rewards are properly bounded
        // Very high latency should clamp to 0
        let high_latency = 2_000_000.0;
        let high_reward = calculate_latency_reward(high_latency);
        assert_eq!(high_reward, 0.0, "Very high latency should clamp to 0");

        // Zero latency should give max reward
        let zero_latency = 0.0;
        let zero_reward = calculate_latency_reward(zero_latency);
        assert_eq!(zero_reward, 1.0, "Zero latency should give max reward 1.0");

        // Negative latency doesn't make sense, but should clamp to 1
        let negative_latency = -10.0;
        let negative_reward = calculate_latency_reward(negative_latency);
        assert_eq!(negative_reward, 1.0, "Negative latency should clamp to 1.0");
    }

    // ============================================================
    // Hotness Score Formula Tests
    // Formula: (frequency + 1.0) / (recency + 1.0)
    // ============================================================

    #[test]
    fn test_hotness_score_hot_data() {
        // High frequency (10), low recency (1.0) = HOT = high score
        // Score = (10 + 1) / (1.0 + 1) = 11 / 2 = 5.5
        let frequency = 10;
        let recency = 1.0f32;
        let score = (frequency as f32 + 1.0) / (recency + 1.0);

        assert!(
            score > 1.0,
            "Hot data should have score > 1.0, got {}",
            score
        );
        assert!(
            score > 5.0,
            "High freq + low recency should score very high"
        );
    }

    #[test]
    fn test_hotness_score_cold_data() {
        // Low frequency (1), high recency (1000.0) = COLD = low score
        // Score = (1 + 1) / (1000.0 + 1) = 2 / 1001 ≈ 0.002
        let frequency = 1;
        let recency = 1000.0f32;
        let score = (frequency as f32 + 1.0) / (recency + 1.0);

        assert!(
            score < 1.0,
            "Cold data should have score < 1.0, got {}",
            score
        );
        assert!(
            score < 0.01,
            "Low freq + high recency should score very low"
        );
    }

    #[test]
    fn test_hotness_score_never_accessed() {
        // Never accessed (recency = INFINITY) = COLDEST = score 0.0
        let score = 0.0f32; // Never accessed returns 0.0

        assert_eq!(score, 0.0, "Never accessed should have score 0.0");
    }

    #[test]
    fn test_hotness_score_just_accessed() {
        // Just accessed (recency ≈ 0) = HOTTEST = very high score
        let frequency = 5;
        let recency = 0.001f32; // Just accessed
        let score = (frequency as f32 + 1.0) / (recency + 1.0);

        assert!(
            score > 5.0,
            "Just-accessed hot data should have very high score"
        );
        assert!(
            score < 6.0,
            "Score should be approximately (5+1)/(0.001+1) ≈ 5.99"
        );
    }

    #[test]
    fn test_hotness_score_comparison() {
        // Verify hot data scores higher than cold data
        let hot_score = (10.0 + 1.0) / (1.0 + 1.0); // freq=10, recency=1
        let cold_score = (1.0 + 1.0) / (1000.0 + 1.0); // freq=1, recency=1000

        assert!(
            hot_score > cold_score,
            "Hot data ({}) should score higher than cold data ({})",
            hot_score,
            cold_score
        );
        assert!(
            hot_score > 100.0 * cold_score,
            "Hot should be MUCH higher than cold"
        );
    }
}
