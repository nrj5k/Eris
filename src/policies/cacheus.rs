//! CACHEUS: Contextual Multi-Armed Bandit with online learning
//!
//! Uses softmax action selection and regret minimization update

use super::policy::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs;
use std::path::Path;

/// CACHEUS Policy - Tabular contextual bandit
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheusPolicy {
    /// Number of arms (2: LRU, LFU)
    n_arms: usize,
    /// Learning rate
    learning_rate: f32,
    /// Arm weights (softmax logits)
    weights: Vec<f32>,
    /// Cumulative regret per arm
    cumulative_regret: Vec<f32>,
    /// Temperature for softmax
    temperature: f32,
    /// Minimum weight value
    min_weight: f32,
    /// Maximum weight value
    max_weight: f32,
}

impl CacheusPolicy {
    /// Create new CACHEUS policy
    pub fn new(n_arms: usize, learning_rate: f32) -> Self {
        Self {
            n_arms,
            learning_rate,
            weights: vec![1.0; n_arms],
            cumulative_regret: vec![0.0; n_arms],
            temperature: 1.0,
            min_weight: 0.1,
            max_weight: 10.0,
        }
    }

    /// Compute context score from features
    /// Features: [recency, frequency, size]
    fn compute_context_score(&self, features: &[f32]) -> f32 {
        if features.len() < 3 {
            return 1.0; // Neutral if insufficient features
        }

        let recency = features[0];
        let frequency = features[1];
        let size = features[2];

        // Higher score for recently accessed, frequently accessed, small items
        let score = (1.0 / (recency + 1.0)) * (frequency.ln_1p() + 1.0) * (1.0 / (size + 1.0));

        score.max(0.1).min(10.0)
    }

    /// Softmax over weighted logits
    fn softmax(&self, logits: &[f32], temperature: f32) -> Vec<f32> {
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let exp_logits: Vec<f32> = logits
            .iter()
            .map(|l| ((*l - max_logit) / temperature).exp())
            .collect();

        let sum_exp: f32 = exp_logits.iter().sum();

        exp_logits.iter().map(|e| e / sum_exp).collect()
    }
}

impl CachePolicy for CacheusPolicy {
    fn select_action(&self, state: &State) -> Action {
        let context_score = match state {
            State::Features(features) => self.compute_context_score(features),
            _ => 1.0,
        };

        // Compute weighted logits
        let weighted_logits: Vec<f32> = self.weights.iter().map(|w| w * context_score).collect();

        // Softmax probabilities
        let probs = self.softmax(&weighted_logits, self.temperature);

        // Sample from categorical distribution
        let mut rng = rand::rng();
        let r: f32 = rng.random();
        let mut cumsum = 0.0;

        for (i, prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r <= cumsum {
                return Action::Discrete(i);
            }
        }

        // Fallback to highest probability
        Action::Discrete(0)
    }

    fn update(&mut self, transition: &Transition) -> f32 {
        let chosen_arm = match transition.action {
            Action::Discrete(arm) => arm,
            _ => return 0.0, // Invalid action type
        };

        let reward = transition.reward;

        // Update cumulative regret for all arms
        for arm in 0..self.n_arms {
            if arm == chosen_arm {
                // Regret = 1 - reward (what we could have gained)
                self.cumulative_regret[arm] += 1.0 - reward;
            } else {
                // Counterfactual: we got the reward we didn't get
                self.cumulative_regret[arm] += reward;
            }
        }

        // Update weights: θ -= α * regret
        for i in 0..self.n_arms {
            self.weights[i] -= self.learning_rate * self.cumulative_regret[i];
            // Clip weights
            self.weights[i] = self.weights[i].clamp(self.min_weight, self.max_weight);
        }

        // Return average regret as "loss"
        self.cumulative_regret.iter().sum::<f32>() / self.n_arms as f32
    }

    fn save(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<(), Box<dyn Error>> {
        let json = fs::read_to_string(path)?;
        let loaded: CacheusPolicy = serde_json::from_str(&json)?;
        *self = loaded;
        Ok(())
    }

    fn policy_type(&self) -> PolicyType {
        PolicyType::Cacheus
    }

    fn action_dim(&self) -> usize {
        self.n_arms
    }
}

impl OnlinePolicy for CacheusPolicy {
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cacheus_creation() {
        let policy = CacheusPolicy::new(2, 0.1);
        assert_eq!(policy.n_arms, 2);
        assert_eq!(policy.weights.len(), 2);
    }

    #[test]
    fn test_action_selection() {
        let policy = CacheusPolicy::new(2, 0.1);
        let state = State::Features(vec![1.0, 5.0, 100.0]);

        let action = policy.select_action(&state);

        match action {
            Action::Discrete(arm) => assert!(arm < 2),
            _ => panic!("Expected discrete action"),
        }
    }

    #[test]
    fn test_update() {
        let mut policy = CacheusPolicy::new(2, 0.1);
        let state = State::Features(vec![1.0, 1.0, 1.0]);

        let transition = Transition {
            state: state.clone(),
            action: Action::Discrete(0),
            reward: 1.0,
            next_state: state,
            done: false,
        };

        let regret = policy.update(&transition);

        assert!(regret >= 0.0);
        assert_eq!(policy.weights.len(), 2);
    }
}
