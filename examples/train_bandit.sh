#!/bin/bash
# Train Bandit policy with different exploration strategies
#
# This script demonstrates training the standalone contextual bandit
# policy, which is optimized for fast online learning.
#
# Bandit policy advantages:
# - No replay buffer (lower memory)
# - Fast online updates
# - Direct importance score computation
# - Ideal for real-time tier selection

set -e # Exit on error

echo "========================================="
echo "Bandit Policy Training"
echo "========================================="
echo ""
echo "Bandit Architecture:"
echo "  - Contextual bandit network"
echo "  - Importance score [0, 1] for tier selection"
echo "  - Online learning without replay buffer"
echo ""

# Common parameters
MODEL="bandit"
EPISODES=100
MAX_STEPS=1000
STATE_DIM=15
NUM_TIERS=5
LEARNING_RATE=0.01 # Higher than DQN for online learning

echo "========================================="
echo "Strategy 1: Thompson Sampling (Recommended)"
echo "========================================="
echo ""
echo "Bayesian posterior sampling"
echo "Best for: Bandit problems, real-time adaptation"
echo ""
cargo run --release --bin train_model -- \
	--model ${MODEL} \
	--episodes ${EPISODES} \
	--max-steps ${MAX_STEPS} \
	--state-dim ${STATE_DIM} \
	--num-tiers ${NUM_TIERS} \
	--exploration thompson-sampling \
	--thompson-mean 0.0 \
	--thompson-std 1.0 \
	--learning-rate ${LEARNING_RATE}

echo ""
echo "========================================="
echo "Strategy 2: Upper Confidence Bound (UCB)"
echo "========================================="
echo ""
echo "Theoretically optimal regret bounds"
echo "Best for: Stationary environments, guaranteed exploration"
echo ""
cargo run --release --bin train_model -- \
	--model ${MODEL} \
	--episodes ${EPISODES} \
	--max-steps ${MAX_STEPS} \
	--state-dim ${STATE_DIM} \
	--num-tiers ${NUM_TIERS} \
	--exploration ucb \
	--ucb-c 1.5 \
	--learning-rate ${LEARNING_RATE}

echo ""
echo "========================================="
echo "Strategy 3: Epsilon-Greedy (Baseline)"
echo "========================================="
echo ""
echo "Simple exploration strategy"
echo "Best for: Baseline comparison"
echo ""
cargo run --release --bin train_model -- \
	--model ${MODEL} \
	--episodes ${EPISODES} \
	--max-steps ${MAX_STEPS} \
	--state-dim ${STATE_DIM} \
	--num-tiers ${NUM_TIERS} \
	--exploration epsilon-greedy \
	--epsilon-start 1.0 \
	--epsilon-end 0.01 \
	--epsilon-decay 0.995 \
	--learning-rate ${LEARNING_RATE}

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo ""
echo "Results saved in: checkpoints/bandit/"
echo ""
echo "Key differences from DQN:"
echo "  - Higher learning rate (0.01 vs 0.0001)"
echo "  - Faster convergence (online updates)"
echo "  - Lower memory (no replay buffer)"
echo "  - Direct importance-to-tier mapping"
echo ""
echo "Bandit Policy Tips:"
echo "  1. Thompson Sampling/UCB > Epsilon-Greedy for bandits"
echo "  2. Use higher learning rate (0.01 - 0.001)"
echo "  3. Monitor importance scores for tier preference"
echo "  4. Good for real-time adaptation to workload changes"
