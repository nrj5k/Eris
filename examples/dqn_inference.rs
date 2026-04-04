//! DQN Policy Inference Example

use burn::backend::NdArray;
use burn::tensor::Tensor;
use burn_rl::Policy;

use eris::config::DQNConfig;
use eris::rl::{DQNPolicy, Observation};
use eris::models::QNetwork;

fn main() {
    println!("🎯 DQN Policy Inference Example\n");

    type Backend = NdArray<f32>;
    let device = Default::default();

    // Create DQN model
    println!("📊 Creating DQN model (obs=4, actions=2)...\n");
    let dqn_config = DQNConfig::builder()
        .input_dim(4)
        .action_dim(2)
        .hidden_layers(vec![128, 128])
        .dueling(true)
        .build()
        .expect("Failed to build DQN config");

    let q_network: QNetwork<Backend> = dqn_config.init(&device);
    let mut policy = DQNPolicy::new(q_network, 2, 0.1);

    // Example 1: Single observation
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 1: Single Observation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create observation tensor [1, 4]
    let obs_data: [f32; 4] = [0.1, -0.2, 0.05, 0.3];
    let obs: Tensor<Backend, 2> = Tensor::from_floats([obs_data], &device);
    println!("Observation: {:?}", obs_data);

    // Forward pass
    let observation = Observation { tensor: obs };
    let dist = policy.forward(observation);
    let q_values: Vec<f32> = dist.logits.into_data().to_vec().unwrap();

    println!("\n📊 Q-values:");
    println!("  Q(left):  {:.4}", q_values[0]);
    println!("  Q(right): {:.4}", q_values[1]);
    println!("  Best:     {}", if q_values[0] > q_values[1] { "Left" } else { "Right" });

    // Select action
    let obs2: Tensor<Backend, 2> = Tensor::from_floats([obs_data], &device);
    let (action, contexts) = policy.action(Observation { tensor: obs2 }, true);
    let action_data: Vec<f32> = action.indices.into_data().to_vec().unwrap();
    println!("🎮 Action: {} ({})\n", action_data[0] as usize, 
        if action_data[0] < 0.5 { "Left" } else { "Right" });

    // Example 2: Batch inference
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 2: Batch Inference");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Batch observations
    let batch: [[f32; 4]; 3] = [
        [0.1, -0.2, 0.05, 0.3],
        [0.5, 0.1, -0.1, 0.2],
        [-0.3, 0.4, 0.2, -0.1],
    ];
    
    let batch_obs: Tensor<Backend, 2> = Tensor::from_floats(batch, &device);
    println!("Batch of {} states created", batch.len());

    // Batch forward pass
    let batch_dist = policy.forward(Observation { tensor: batch_obs });
    let batch_q: Vec<f32> = batch_dist.logits.into_data().to_vec().unwrap();

    println!("\n📊 Batch Q-values:");
    for i in 0..3 {
        println!("  State {}: Q(left)={:.4}, Q(right)={:.4}", 
            i, batch_q[i*2], batch_q[i*2 + 1]);
    }

    // Batch actions
    let batch_obs2: Tensor<Backend, 2> = Tensor::from_floats(batch, &device);
    let (batch_actions, batch_contexts) = policy.action(Observation { tensor: batch_obs2 }, false);
    let actions: Vec<f32> = batch_actions.indices.into_data().to_vec().unwrap();
    
    println!("\n🎮 Batch actions:");
    for i in 0..3 {
        println!("  State {}: Action {} ({})", i, actions[i] as usize,
            if actions[i] < 0.5 { "Left" } else { "Right" });
    }
    println!("  Contexts: {}", batch_contexts.len());

    // Example 3: Epsilon-greedy
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 3: Epsilon-Greedy (ε=0.5)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    policy.set_epsilon(0.5);
    let test: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    
    println!("Running 10 action selections:");
    for i in 0..10 {
        let test_obs: Tensor<Backend, 2> = Tensor::from_floats([test], &device);
        let (action, _) = policy.action(Observation { tensor: test_obs }, false);
        let idx: Vec<f32> = action.indices.into_data().to_vec().unwrap();
        println!("  {}. Action {} ({})", i+1, idx[0] as usize,
            if idx[0] < 0.5 { "Left" } else { "Right" });
    }

    // Example 4: Policy state
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 4: Policy State Management");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let state = policy.state();
    println!("✓ Retrieved policy state");

    let new_net: QNetwork<Backend> = dqn_config.init(&device);
    let new_policy = DQNPolicy::new(new_net, 2, 0.05);
    policy.update(new_policy.state());
    
    println!("✓ Updated policy");
    println!("  Current ε: {}", policy.epsilon());

    println!("\n✨ Example complete!");
}
