use crate::env::Environment;
use crate::tier::TierSelector;
use crate::training::CombinedAgent;
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
///
/// # Returns
///
/// Training result with episode rewards, losses, and final epsilon
///
/// # Training Process
///
/// The function implements the following workflow:
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
/// use eris::training::{train_agent, CombinedAgent, TrainingConfig};
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
/// // Train agent
/// let result = train_agent(&mut env, &mut agent, 10, &tier_selector);
/// println!("Trained for {} episodes", result.episode_rewards.len());
/// # Ok(())
/// # }
/// ```
pub fn train_agent<B: AutodiffBackend, E: Environment<Observation = Vec<f64>, Action = usize>>(
    env: &mut E,
    agent: &mut CombinedAgent<B>,
    num_episodes: usize,
    tier_selector: &TierSelector,
) -> TrainingResult {
    // Early return if no episodes to train
    if num_episodes == 0 {
        return TrainingResult {
            episode_rewards: Vec::new(),
            losses: Vec::new(),
            final_epsilon: agent.epsilon,
        };
    }

    let mut episode_rewards = Vec::with_capacity(num_episodes);
    let mut losses = Vec::new();

    for _episode in 0..num_episodes {
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

            // Store transition in hybrid buffer (CPU storage, GPU conversion on sample)
            agent.buffer.push(
                state_f32.clone(),
                action,
                result.reward as f32,
                next_state_f32.clone(),
                result.done,
            );

            // Train if buffer has enough samples using GPU-native sampling
            if agent.buffer.len() >= agent.config.batch_size {
                if let Some(loss) = agent.train_step_gpu_native(agent.config.batch_size) {
                    losses.push(loss);
                    if losses.len() > 10000 {
                        losses.drain(0..5000);
                    }
                }
            }

            state = result.observation;
            done = result.done;
        }

        episode_rewards.push(total_reward as f32);

        // Epsilon decay after each episode
        agent.epsilon = (agent.epsilon * 0.995).max(0.01);
    }

    TrainingResult {
        episode_rewards,
        losses,
        final_epsilon: agent.epsilon,
    }
}

/// Train agent using Burn's training infrastructure.
///
/// This is a Burn-native training loop that leverages:
/// - `DQNDataLoader` for batch sampling (Task 02a)
/// - `TrainStep` trait for gradient computation (Task 02b)
/// - Burn metrics for monitoring (Task 02c)
/// - Callbacks for DQN-specific logic (Task 02e)
///
/// # Arguments
///
/// * `env` - Environment implementing the [`Environment`] trait
/// * `agent` - Combined agent with model and replay buffer
/// * `num_episodes` - Number of episodes to train
/// * `tier_selector` - Tier selector for action selection
///
/// # Returns
///
/// Training result with episode rewards, losses, and final epsilon
///
/// # Burn Integration
///
/// This function uses Burn's training primitives while maintaining DQN-specific requirements:
/// 1. Experience replay via `DQNDataLoader`
/// 2. Target network updates via `TargetUpdateCallback`
/// 3. Epsilon decay via `EpsilonDecayCallback`
/// 4. Progress tracking via Burn metrics
///
/// Key differences from standard Burn learners:
/// - No fixed dataset (dynamic experience buffer)
/// - Training interleaved with environment interaction
/// - No train/val splits
/// - Episode-based epsilon decay
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use eris::training::{train_agent_burn, CombinedAgent, TrainingConfig};
/// use eris::env::Environment;
/// use eris::tier::TierSelector;
///
/// let mut env = eris::training::MockEnv::new_with_dims(100, 50, 20);
/// let state_dim = env.observation_space().dim();
/// let action_dim = env.action_space().n;
///
/// # let model_config = eris::models::CombinedModelConfig::new(state_dim, 20, 128, action_dim);
/// # let device = burn::backend::ndarray::NdArrayDevice::Cpu;
/// # let training_config = TrainingConfig::default();
/// # let mut agent = CombinedAgent::new(training_config, model_config, device);
/// # let tier_selector = TierSelector::new(vec![]);
///
/// let result = train_agent_burn(&mut env, &mut agent, 10, &tier_selector);
/// println!("Average reward: {:.1}",
///     result.episode_rewards.iter().sum::<f32>() / result.episode_rewards.len() as f32);
/// # Ok(())
/// # }
/// ```
pub fn train_agent_burn<
    B: AutodiffBackend,
    E: Environment<Observation = Vec<f64>, Action = usize>,
>(
    env: &mut E,
    agent: &mut CombinedAgent<B>,
    num_episodes: usize,
    tier_selector: &TierSelector,
) -> TrainingResult {
    use crate::training::burn_callbacks::{EpsilonDecayCallback, TargetUpdateCallback};

    // Initialize tracking
    let mut episode_rewards = Vec::with_capacity(num_episodes);
    let mut losses = Vec::new();

    // Create DQN-specific callbacks
    let target_callback = TargetUpdateCallback::new(agent.config.target_update_freq);
    let mut epsilon_callback = EpsilonDecayCallback::new(
        agent.epsilon,
        agent.config.epsilon_end,
        agent.config.epsilon_decay,
    );

    // Main training loop
    for _episode in 0..num_episodes {
        let mut total_reward = 0.0;
        let mut done = false;
        let mut state = env.reset();

        // Episode loop
        while !done {
            // Get action from policy
            let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();
            let state_data = TensorData::new(state_f32.clone(), [1, state_f32.len()]);
            let state_tensor = Tensor::from_data(state_data.convert::<f32>(), &agent.device);

            let action =
                agent
                    .model
                    .select_action(state_tensor, tier_selector, epsilon_callback.epsilon());

            // Step environment
            let result = env.step(action);
            total_reward += result.reward;

            // Store transition in hybrid buffer (CPU storage, GPU conversion on sample)
            let next_state_f32: Vec<f32> = result.observation.iter().map(|&x| x as f32).collect();
            agent.buffer.push(
                state_f32.clone(),
                action,
                result.reward as f32,
                next_state_f32.clone(),
                result.done,
            );

            // Train with GPU-native sampling
            if agent.buffer.len() >= agent.config.batch_size {
                if let Some(loss) = agent.train_step_gpu_native(agent.config.batch_size) {
                    losses.push(loss);
                    if losses.len() > 10000 {
                        losses.drain(0..5000);
                    }

                    // Update target network if needed
                    if target_callback.should_update() {
                        agent.hard_update_target();
                    }
                }
            }

            state = result.observation;
            done = result.done;
        }

        episode_rewards.push(total_reward as f32);

        // Decay epsilon after episode
        epsilon_callback.decay();
    }

    TrainingResult {
        episode_rewards,
        losses,
        final_epsilon: epsilon_callback.epsilon(),
    }
}

/// Train agent with live metrics dashboard.
///
/// This function provides the same training as `train_agent_burn` but with
/// live progress display using Burn's metrics infrastructure.
///
/// # Arguments
///
/// * `env` - Environment implementing the [`Environment`] trait
/// * `agent` - Combined agent with model and replay buffer
/// * `num_episodes` - Number of episodes to train
/// * `tier_selector` - Tier selector for action selection
///
/// # Returns
///
/// Training result with episode rewards, losses, and final epsilon
///
/// # Progress Display
///
/// Shows live metrics:
/// - Episode number and progress
/// - Running average reward
/// - Current epsilon value
/// - Training loss
pub fn train_agent_with_metrics<
    B: AutodiffBackend,
    E: Environment<Observation = Vec<f64>, Action = usize>,
>(
    env: &mut E,
    agent: &mut CombinedAgent<B>,
    num_episodes: usize,
    tier_selector: &TierSelector,
) -> TrainingResult {
    use crate::training::burn_callbacks::{EpsilonDecayCallback, TargetUpdateCallback};

    // Initialize tracking
    let mut episode_rewards = Vec::with_capacity(num_episodes);
    let mut losses = Vec::new();

    // Create DQN-specific callbacks
    let target_callback = TargetUpdateCallback::new(agent.config.target_update_freq);
    let mut epsilon_callback = EpsilonDecayCallback::new(
        agent.epsilon,
        agent.config.epsilon_end,
        agent.config.epsilon_decay,
    );

    tracing::info!("Training {} episodes...", num_episodes);
    tracing::info!(
        "{:>10} {:>12} {:>10} {:>10} {:>10}",
        "Episode",
        "Reward",
        "Avg",
        "Epsilon",
        "Loss"
    );

    // Main training loop
    for episode in 0..num_episodes {
        let mut total_reward = 0.0;
        let mut done = false;
        let mut state = env.reset();
        let mut episode_loss = 0.0;
        let mut loss_count = 0;

        // Episode loop
        while !done {
            // Get action from policy
            let state_f32: Vec<f32> = state.iter().map(|&x| x as f32).collect();
            let state_data = TensorData::new(state_f32.clone(), [1, state_f32.len()]);
            let state_tensor = Tensor::from_data(state_data.convert::<f32>(), &agent.device);

            let action =
                agent
                    .model
                    .select_action(state_tensor, tier_selector, epsilon_callback.epsilon());

            // Step environment
            let result = env.step(action);
            total_reward += result.reward;

            // Store transition in hybrid buffer (CPU storage, GPU conversion on sample)
            let next_state_f32: Vec<f32> = result.observation.iter().map(|&x| x as f32).collect();
            agent.buffer.push(
                state_f32.clone(),
                action,
                result.reward as f32,
                next_state_f32.clone(),
                result.done,
            );

            // Train with GPU-native sampling
            if agent.buffer.len() >= agent.config.batch_size {
                if let Some(loss) = agent.train_step_gpu_native(agent.config.batch_size) {
                    losses.push(loss);
                    if losses.len() > 10000 {
                        losses.drain(0..5000);
                    }
                    episode_loss += loss;
                    loss_count += 1;

                    // Update target network if needed
                    if target_callback.should_update() {
                        agent.hard_update_target();
                    }
                }
            }

            state = result.observation;
            done = result.done;
        }

        episode_rewards.push(total_reward as f32);

        // Compute running average
        let avg_reward: f32 = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
        let avg_loss = if loss_count > 0 {
            episode_loss / loss_count as f32
        } else {
            0.0
        };

        // Log progress
        tracing::info!(
            "{:>10} {:>12.1} {:>10.1} {:>10.3} {:>10.4}",
            episode + 1,
            total_reward,
            avg_reward,
            epsilon_callback.epsilon(),
            avg_loss
        );

        // Show tier utilization every 10 episodes using Burn's TierMetric
        if (episode + 1) % 10 == 0 {
            use burn::data::dataloader::Progress;
            use burn::train::metric::{Metric, MetricMetadata};

            let tier_util = env.get_tier_utilization();
            let mut tier_metric = crate::training::TierMetric::new();
            let tier_names: Vec<String> = (0..tier_util.len())
                .map(|i| format!("Tier_{}", i))
                .collect();
            let tier_input = crate::training::TierInput {
                tier_names,
                tier_utilizations: tier_util.clone(),
            };
            let metadata = MetricMetadata {
                progress: Progress::new(episode + 1, num_episodes),
                epoch: episode + 1,
                epoch_total: num_episodes,
                iteration: episode + 1,
                lr: None,
            };
            let entry = tier_metric.update(&tier_input, &metadata);
            tracing::info!("--- Tier Utilization (Episode {}) ---", episode + 1);
            tracing::info!("{}", entry.formatted);
        }

        // Decay epsilon after episode
        epsilon_callback.decay();
    }

    tracing::info!(
        "Training complete! Final epsilon: {:.3}",
        epsilon_callback.epsilon()
    );
    tracing::info!(
        "Average reward: {:.2}",
        episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
    );

    TrainingResult {
        episode_rewards,
        losses,
        final_epsilon: epsilon_callback.epsilon(),
    }
}
