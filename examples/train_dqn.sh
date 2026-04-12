#!/bin/bash
# Train DQN with different exploration strategies
#
# This script demonstrates training the standalone DQN policy
# with three different exploration strategies:
# 1. Epsilon-Greedy (standard for DQN)
# 2. Thompson Sampling (for uncertainty handling)
# 3. UCB (for theoretically optimal exploration)

set -e # Exit on error

echo "========================================="
echo "DQN Training - Exploration Strategies"
echo "========================================="
echo ""

# Common parameters
MODEL="dqn"
EPISODES=100
MAX_STEPS=1000
STATE_DIM=15
ACTION_DIM=10

echo "Strategy 1: Epsilon-Greedy Exploration"
echo "----------------------------------------"
echo "Classic exploration with probability ε of random action"
echo "Recommended for: Simple baseline, most problems"
echo ""
cargo run --release --bin train_model -- \
	--model ${MODEL} \
	--episodes ${EPISODES} \
	--max-steps ${MAX_STEPS} \
	--state-dim ${STATE_DIM} \
	--action-dim ${ACTION_DIM} \
	--exploration epsilon-greedy \
	--epsilon-start 1.0 \
	--epsilon-end 0.01 \
	--epsilon-decay 0.995 \
	--learning-rate 0.0001 \
	--gamma 0.99 \
	--batch-size 512

echo ""
echo "Strategy 2: Thompson Sampling"
echo "----------------------------------------"
echo "Bayesian posterior sampling for exploration"
echo "Recommended for: Non-stationary environments, uncertainty handling"
echo ""
cargo run --release --bin train_model -- \
	--model ${MODEL} \
	--episodes ${EPISODES} \
	--max-steps ${MAX_STEPS} \
	--state-dim ${STATE_DIM} \
	--action-dim ${ACTION_DIM} \
	--exploration thompson-sampling \
	--thompson-mean 0.0 \
	--thompson-std 1.0 \
	--learning-rate 0.0001 \
	--gamma 0.99 \
	--batch-size 512

echo ""
echo "Strategy 3: Upper Confidence Bound (UCB)"
echo "----------------------------------------"
echo "UCB1 formula: Q(a) + c * sqrt(ln(N) / n(a))"
echo "Recommended for: Theoretically optimal regret bounds"
echo ""
cargo run --release --bin train_model -- \
	--model ${MODEL} \
	--episodes ${EPISODES} \
	--max-steps ${MAX_STEPS} \
	--state-dim ${STATE_DIM} \
	--action-dim ${ACTION_DIM} \
	--exploration ucb \
	--ucb-c 2.0 \
	--learning-rate 0.0001 \
	--gamma 0.99 \
	--batch-size 512

echo ""
echo "========================================="
echo "Training completed!"
echo "========================================="
echo ""
echo "Results saved in: checkpoints/dqn/"
echo ""
echo "Comparison tips:"
echo "  - Epsilon-Greedy: Easy to tune, works for most problems"
echo "  - Thompson Sampling: Adapts to uncertainty, better for bandits"
echo "  - UCB: Guaranteed regret bounds, good for stationary environments"
echo ""
echo "Next steps:"
echo "  1. Compare learning curves across strategies"
echo "  2. Analyze exploration behavior (action distributions)"
echo "  3. Test on different environment configurations"
