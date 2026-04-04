use crate::env::Environment;
use crate::tier::TierSelector;
use crate::training::monitor::TrainingMonitor;
use crate::training::{CombinedAgent, Transition};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};

/// Result of training an agent.
///
/// Contains metrics collected during training episodes.
///
/// # Fields
///
/// * `episode_rewards` - Rewards for each completed episode
/// * `losses` - Training losses from each optimization step
/// * `final_epsilon` - Final exploration rate after training
///
/// # Example
///
/// ```
/// use eris::training::TrainingResult;
///
/// let result = TrainingResult {
///     episode_rewards: vec![10.0, 20.0, 15.0],
///     losses: vec![0.5, 0.3, 0.2],
///     final_epsilon: 0.1,
/// };
///
/// println!("Average reward: {}",
///     result.episode_rewards.iter().sum::<f32>() / result.episode_rewards.len() as f32);
/// println!("Losses: {:?}", result.losses);
/// println!("Final exploration rate: {}", result.final_epsilon);
/// ```
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Rewards for each episode
    pub episode_rewards: Vec<f32>,
    /// Losses from training steps
    pub losses: Vec<f32>,
    /// Final exploration rate after training
    pub final_epsilon: f32,
}

/// Train an agent on an environment implementing Environment trait.
///
/// This is the main training function for DQN agents. It executes the complete
/// training loop, running episodes and training the model using experience replay.
///
/// # Arguments
///
/// * `env` - Environment implementing the [`Environment`] trait
/// * `agent` - Combined agent with model and replay buffer
/// * `num_episodes` - Number of episodes to train
/// * `tier_selector` - Tier selector for action selection
/// * `monitor` - Optional training monitor for callbacks
///
/// # Returns
///
/// Training result with episode rewards, losses, and final epsilon
///
/// # Training Process
///
/// The function implements the following流程:
///
/// 1. **Episode Loop**: Run specified number of episodes
/// 2. **Step Loop**: Within each episode, take actions until done
/// 3. **Action Selection**: Use epsilon-greedy policy
/// 4. **Experience Storage**: Store transitions in replay buffer
/// 5. **Model Training**: Sample batches and train when buffer is full
/// 6. **Epsilon Decay**: Gradually reduce exploration rate
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use eris::training::{train_agent, CombinedAgent, TrainingConfig, ConsoleMonitor};
/// use eris::env::Environment;
/// use eris::tier::TierSelector;
///
/// // Create environment with dynamic dimensions
/// let mut env = eris::training::MockEnv::new_with_dims(100, 50, 20);
/// let state_dim = env.observation_space().dim();
/// let action_dim = env.action_space().n;
///
/// // Create model configuration
/// # let model_config = eris::models::CombinedModelConfig::new(state_dim, 20, 128, action_dim);
/// # let device = burn::backend::ndarray::NdArrayDevice::Cpu;
/// # let training_config = TrainingConfig::default();
///
/// // Create agent
/// # let mut agent = CombinedAgent::new(training_config, model_config, device);
/// # let tier_selector = TierSelector::new(vec![]);
///
/// // Train agent with monitor
/// let mut monitor = ConsoleMonitor::new(10);
/// let result = train_agent(&mut env, &mut agent, 10, &tier_selector, Some(&mut monitor));
/// println!("Trained for {} episodes", result.episode_rewards.len());
/// # Ok(())
/// # }
/// ```
pub fn train_agent<
    B: AutodiffBackend,
    E: Environment<Observation = Vec<f64>, Action = usize>,
    M: TrainingMonitor,
>(
    env: &mut E,
    agent: &mut CombinedAgent<B>,
    num_episodes: usize,
    tier_selector: &TierSelector,
    monitor: Option<&mut M>,
) -> TrainingResult {
    let mut episode_rewards = Vec::with_capacity(num_episodes);
    let mut losses = Vec::new();
    let mut monitor = monitor;

    for episode in 0..num_episodes {
        if let Some(m) = monitor.as_mut() {
            m.on_episode_start(episode, num_episodes);
        }

        let mut total_reward = 0.0;
        let mut done = false;
        let mut state = env.reset();

        while !done {
            // Convert state f64 -> f32
            let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();

            // Convert to tensor [1, state_dim]
            let state_data = TensorData::new(state_f32.clone(), [1, state_f32.len()]);
            let state_tensor = Tensor::from_data(state_data.convert::<f32>(), &agent.device);

            // Select action using epsilon-greedy policy
            let action = agent
                .model
                .select_action(state_tensor, tier_selector, agent.epsilon);

            // Step environment
            let result = env.step(action);
            total_reward += result.reward;

            // Convert next_state f64 -> f32
            let next_state_f32: Vec<f32> = result.observation.iter().map(|&x| x as f32).collect();

            // Store transition in replay buffer
            agent.buffer.push(Transition {
                state: state_f32,
                action,
                reward: result.reward as f32,
                next_state: next_state_f32,
                done: result.done,
            });

            // Train if buffer has enough samples
            if agent.buffer.len() >= agent.config.batch_size {
                if let Some(batch) = agent.buffer.sample_batch(agent.config.batch_size) {
                    let loss = agent.train_step(batch);
                    losses.push(loss);

                    // Monitor callback for training step
                    if let Some(m) = monitor.as_mut() {
                        m.on_step(losses.len() - 1, loss);
                    }
                }
            }

            state = result.observation;
            done = result.done;
        }

        episode_rewards.push(total_reward as f32);

        // Monitor callback for episode completion
        if let Some(m) = monitor.as_mut() {
            // Get tier states from environment buffer
            let tier_states: Vec<f32> = env.get_tier_utilization();
            m.on_episode_end(episode, total_reward as f32, agent.epsilon, &tier_states);
        }

        // Epsilon decay after each episode
        agent.epsilon = (agent.epsilon * 0.995).max(0.01);
    }

    TrainingResult {
        episode_rewards,
        losses,
        final_epsilon: agent.epsilon,
    }
}
