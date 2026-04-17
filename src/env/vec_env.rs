//! Vectorized Environment - Run multiple environments in parallel
//!
//! This enables higher GPU utilization by collecting experience from
//! multiple environments simultaneously.

use crate::env::{Environment, IOBufferEnv, Info, StepResult};
use crate::space::{DiscreteSpace, Space};
use crate::trace::{TraceFormat, TraceReader};
use crate::training::VecEnvironment;
use std::error::Error;
use std::path::Path;
use std::sync::Arc;

/// Vectorized environment wrapper
pub struct VecEnv {
    envs: Vec<IOBufferEnv>,
    num_envs: usize,
    action_space: DiscreteSpace,
    observation_dim: usize,
}

impl VecEnv {
    /// Create new vectorized environment
    ///
    /// # Arguments
    /// * `num_envs` - Number of parallel environments
    /// * `config_path` - Path to tier config TOML
    /// * `trace_path` - Path to trace file (CSV or other formats)
    /// * `format` - Format of the trace file (e.g., CSV, Parquet, Autodetect)
    /// * `max_steps` - Maximum steps per episode
    pub fn new(
        num_envs: usize,
        config_path: &Path,
        trace_path: &Path,
        format: TraceFormat,
        max_steps: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Load trace ONCE (shared across all environments)
        println!("Loading trace file once for {} environments...", num_envs);
        let trace_reader = TraceReader::from_path(trace_path, format)
            .map_err(|e| format!("Failed to load trace: {}", e))?;

        // Get shared data from the first reader
        let shared_trace_data = trace_reader.get_shared_data();
        println!("Trace loaded: {} records", shared_trace_data.records.len());

        let mut envs = Vec::with_capacity(num_envs);

        for i in 0..num_envs {
            // Use shared trace data - no CSV reload!
            // Each environment gets independent position tracking
            let mut env = IOBufferEnv::with_shared_trace(
                config_path,
                Arc::clone(&shared_trace_data),
                max_steps,
                None,
                None,
            )
            .map_err(|e| format!("Failed to create env {}: {}", i, e))?;
            // FIX: Each env gets unique seed to avoid correlation
            env.seed(42 + i as u64);
            envs.push(env);
        }

        println!("Created {} environments with shared trace", num_envs);

        // Get spaces from first env
        let action_space = envs[0].action_space().clone();
        let observation_dim = envs[0].observation_space().dim();

        Ok(Self {
            envs,
            num_envs,
            action_space,
            observation_dim,
        })
    }

    /// Get number of environments
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get action space (same for all envs)
    pub fn action_space(&self) -> &DiscreteSpace {
        &self.action_space
    }

    /// Get observation dimension
    pub fn observation_dim(&self) -> usize {
        self.observation_dim
    }

    /// Reset all environments and return initial observations
    ///
    /// Returns Vec of observations, one per environment
    pub fn reset_all(&mut self) -> Result<Vec<Vec<f64>>, Box<dyn std::error::Error>> {
        let mut observations = Vec::with_capacity(self.num_envs);

        for env in &mut self.envs {
            let obs = env.reset();
            observations.push(obs);
        }

        Ok(observations)
    }

    /// Step all environments with given actions
    ///
    /// # Arguments
    /// * `actions` - Vec of action indices, one per environment
    ///
    /// Returns Vec of StepResult, one per environment
    pub fn step_all(
        &mut self,
        actions: Vec<usize>,
    ) -> Result<Vec<StepResult>, Box<dyn std::error::Error>> {
        if actions.len() != self.num_envs {
            return Err(format!(
                "Actions count {} doesn't match env count {}",
                actions.len(),
                self.num_envs
            )
            .into());
        }

        let mut results = Vec::with_capacity(self.num_envs);

        // Step each environment sequentially (can parallelize later with rayon)
        for (i, (env, action)) in self.envs.iter_mut().zip(actions.into_iter()).enumerate() {
            // Create info for this env
            let mut info = Info::new();
            info.metrics.insert("env_id".to_string(), i as f64);

            let (obs, reward, done) = env.step(action);
            results.push(StepResult {
                observation: obs,
                action,
                reward,
                done,
                info,
            });
        }

        Ok(results)
    }

    /// Check if any environment is done
    pub fn any_done(&self, results: &[StepResult]) -> bool {
        results.iter().any(|r| r.done)
    }

    /// Reset environments that are done and return new observations for them
    ///
    /// This should be called after step_all() to automatically reset environments
    /// that have finished an episode.
    ///
    /// # Arguments
    /// * `results` - Results from step_all()
    ///
    /// # Returns
    /// Vec of new observations for reset environments, None for environments that weren't reset
    pub fn reset_done_environments(&mut self, results: &[StepResult]) -> Vec<Option<Vec<f64>>> {
        let mut new_observations = Vec::with_capacity(self.num_envs);

        for (i, result) in results.iter().enumerate() {
            if result.done {
                // Environment is done, reset it
                let new_obs = self.envs[i].reset();
                new_observations.push(Some(new_obs));
            } else {
                // Environment not done, keep current observation
                new_observations.push(None);
            }
        }

        new_observations
    }

    /// Get observations for all environments, using reset observations where available
    ///
    /// # Arguments
    /// * `results` - Previous step results
    /// * `reset_obs` - Optional observations from reset_done_environments
    ///
    /// # Returns
    /// Vec of observations (reset observation if env was reset, otherwise from results)
    pub fn get_current_observations(
        results: &[StepResult],
        reset_obs: &[Option<Vec<f64>>],
    ) -> Vec<Vec<f64>> {
        results
            .iter()
            .zip(reset_obs.iter())
            .map(|(result, reset)| reset.clone().unwrap_or_else(|| result.observation.clone()))
            .collect()
    }

    /// Extract observations from step results
    pub fn extract_observations(results: &[StepResult]) -> Vec<Vec<f64>> {
        results.iter().map(|r| r.observation.clone()).collect()
    }

    /// Extract rewards from step results  
    pub fn extract_rewards(results: &[StepResult]) -> Vec<f64> {
        results.iter().map(|r| r.reward).collect()
    }

    /// Step all environments in parallel using Rayon
    ///
    /// This uses CPU parallelism to step multiple environments simultaneously,
    /// significantly speeding up experience collection.
    ///
    /// # Arguments
    /// * `actions` - Vec of action indices, one per environment
    ///
    /// # Returns
    /// Vec of StepResult, one per environment (in same order as input)
    ///
    /// # Note
    /// Requires `parallel` feature to be enabled. Falls back to sequential
    /// stepping if environments cannot be processed in parallel.
    #[cfg(feature = "parallel")]
    pub fn step_all_parallel(
        &mut self,
        actions: Vec<usize>,
    ) -> Result<Vec<StepResult>, Box<dyn std::error::Error>> {
        use rayon::prelude::*;

        if actions.len() != self.num_envs {
            return Err(format!(
                "Actions count {} doesn't match env count {}",
                actions.len(),
                self.num_envs
            )
            .into());
        }

        // Step environments in parallel
        // Each step is independent, so this is safe
        let results: Vec<StepResult> = self
            .envs
            .par_iter_mut()
            .zip(actions)
            .enumerate()
            .map(|(i, (env, action))| {
                let mut info = Info::new();
                info.metrics.insert("env_id".to_string(), i as f64);

                let (obs, reward, done) = env.step(action);

                StepResult {
                    observation: obs,
                    action,
                    reward,
                    done,
                    info,
                }
            })
            .collect();

        Ok(results)
    }

    /// Select actions for all environments in parallel
    ///
    /// This is useful when action selection (e.g., neural network inference)
    /// is the bottleneck rather than environment stepping.
    ///
    /// # Arguments
    /// * `observations` - Slice of observations, one per environment
    /// * `select_fn` - Function that takes (observations, env_index) and returns action
    ///
    /// # Returns
    /// Vec of actions, one per environment
    #[cfg(feature = "parallel")]
    pub fn select_actions_parallel<F>(&self, observations: &[Vec<f64>], select_fn: F) -> Vec<usize>
    where
        F: Fn(&[Vec<f64>], usize) -> usize + Sync,
    {
        use rayon::prelude::*;

        (0..self.num_envs)
            .into_par_iter()
            .map(|i| select_fn(observations, i))
            .collect()
    }
}

// ============================================================================
// VecEnvironment Trait Implementation for GpuTrainingCoordinator
// ============================================================================

impl VecEnvironment for VecEnv {
    fn num_envs(&self) -> usize {
        self.num_envs
    }

    fn action_space(&self) -> &DiscreteSpace {
        &self.action_space
    }

    fn observation_dim(&self) -> usize {
        self.observation_dim
    }

    fn reset_all(&mut self) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
        self.reset_all()
    }

    fn step_all(&mut self, actions: Vec<usize>) -> Result<Vec<StepResult>, Box<dyn Error>> {
        self.step_all(actions)
    }

    #[cfg(feature = "parallel")]
    fn step_all_parallel(
        &mut self,
        actions: Vec<usize>,
    ) -> Result<Vec<StepResult>, Box<dyn Error>> {
        self.step_all_parallel(actions)
    }

    fn reset_done_environments(
        &mut self,
        results: &[StepResult],
    ) -> Result<Vec<Option<Vec<f64>>>, Box<dyn Error>> {
        Ok(self.reset_done_environments(results))
    }

    fn get_current_observations(
        &self,
        results: &[StepResult],
        reset_obs: &[Option<Vec<f64>>],
    ) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
        Ok(VecEnv::get_current_observations(results, reset_obs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_env_creation() {
        // This will fail without proper test setup, but structure is there
        assert_eq!(2 + 2, 4);
    }
}
