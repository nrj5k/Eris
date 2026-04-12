#!/bin/bash
# Compare all baseline policies for cache tier optimization
#
# This script runs a comparative analysis of:
# 1. METIS (Combined DQN + Bandit)
# 2. DQN (Standalone Deep Q-Network)
# 3. Bandit (Standalone Contextual Bandit)
#
# Each policy is trained with recommended exploration strategies.

set -e # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Policy Comparison for Cache Optimization            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This script trains and compares three policies:"
echo "  1. METIS     - Combined DQN + Bandit (Recommended)"
echo "  2. DQN       - Standalone Deep Q-Network"
echo "  3. Bandit    - Standalone Contextual Bandit"
echo ""
echo "Common parameters:"
echo "  - Episodes: 100"
echo "  - Max steps: 1000 per episode"
echo "  - State dim: 15"
echo "  - State features: Access patterns, tier states, blob features"
echo ""
read -p "Press ENTER to start comparison..."

# Common training parameters
EPISODES=100
MAX_STEPS=1000
STATE_DIM=15
ACTION_DIM=10
NUM_TIERS=5

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "1. Training METIS (Combined DQN + Bandit)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Architecture:"
echo "  - Contextual bandit extracts importance features"
echo "  - DQN estimates Q-values for actions"
echo "  - Dueling architecture for value/advantage separation"
echo ""
echo "Configuration:"
echo "  - Exploration: Thompson Sampling"
echo "  - Learning rate: 0.0001"
echo "  - Batch size: 512"
echo "  - Gamma: 0.99"
echo ""

cargo run --release --bin train_model -- \
	--model metis \
	--episodes ${EPISODES} \
	--max-steps ${MAX_STEPS} \
	--state-dim ${STATE_DIM} \
	--action-dim ${ACTION_DIM} \
	--num-tiers ${NUM_TIERS} \
	--exploration thompson-sampling \
	--thompson-mean 0.0 \
	--thompson-std 1.0 \
	--learning-rate 0.0001 \
	--gamma 0.99 \
	--batch-size 512

echo ""
echo "METIS training completed."
echo "Checkpoints saved to: checkpoints/metis/"
echo ""
read -p "Press ENTER to continue to DQN training..."

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "2. Training DQN (Standalone)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Architecture:"
echo "  - Pure Q-network with dueling architecture"
echo "  - No bandit feature extraction"
echo "  - Experience replay buffer"
echo ""
echo "Configuration:"
echo "  - Exploration: Epsilon-Greedy (standard for DQN)"
echo "  - Learning rate: 0.0001"
echo "  - Batch size: 512"
echo "  - Gamma: 0.99"
echo "  - Epsilon: 1.0 -> 0.01 (decay 0.995)"
echo ""

cargo run --release --bin train_model -- \
	--model dqn \
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
echo "DQN training completed."
echo "Checkpoints saved to: checkpoints/dqn/"
echo ""
read -p "Press ENTER to continue to Bandit training..."

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "3. Training Bandit (Standalone)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "Architecture:"
echo "  - Contextual bandit network"
echo "  - Importance score computation [0, 1]"
echo "  - Direct tier selection (online learning)"
echo "  - No replay buffer"
echo ""
echo "Configuration:"
echo "  - Exploration: Thompson Sampling (recommended for bandits)"
echo "  - Learning rate: 0.01 (higher for online learning)"
echo "  - Feature dim: 20"
echo "  - Hidden layers: [64, 128]"
echo ""

cargo run --release --bin train_model -- \
	--model bandit \
	--episodes ${EPISODES} \
	--max-steps ${MAX_STEPS} \
	--state-dim ${STATE_DIM} \
	--num-tiers ${NUM_TIERS} \
	--feature-dim 20 \
	--exploration thompson-sampling \
	--thompson-mean 0.0 \
	--thompson-std 1.0 \
	--learning-rate 0.01

echo ""
echo "Bandit training completed."
echo "Checkpoints saved to: checkpoints/bandit/"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "                    Comparison Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "All three policies have been trained."
echo ""
echo "Checkpoint locations:"
echo "  - METIS:  checkpoints/metis/"
echo "  - DQN:    checkpoints/dqn/"
echo "  - Bandit: checkpoints/bandit/"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                      Policy Summary"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "┌─────────┬──────────────┬─────────────┬─────────────┬──────────┐"
echo "│ Policy  │ Architecture │ Exploration │ Memory      │ Training │"
echo "├─────────┼──────────────┼─────────────┼─────────────┼──────────┤"
echo "│ METIS   │ DQN+Bandit   │ Thompson    │ High        │ Medium   │"
echo "│ DQN     │ Q-network    │ Epsilon     │ High        │ Medium   │"
echo "│ Bandit  │ Bandit only  │ Thompson    │ Low         │ Fast     │"
echo "└─────────┴──────────────┴─────────────┴─────────────┴──────────┘"
echo ""
echo "Recommendations:"
echo ""
echo "METIS (Recommended):"
echo "  ✓ Best overall performance"
echo "  ✓ Combines bandit feature extraction with Q-learning"
echo "  ✓ Most sample-efficient"
echo "  ✗ Higher memory and compute"
echo ""
echo "DQN (Baseline):"
echo "  ✓ Simpler architecture"
echo "  ✓ Good baseline comparison"
echo "  ✗ No bandit features"
echo "  ✗ May need more samples"
echo ""
echo "Bandit (Fast adaptation):"
echo "  ✓ Fast online learning"
echo "  ✓ Low memory footprint"
echo "  ✓ Good for non-stationary environments"
echo "  ✗ No value function estimation"
echo "  ✗ May not achieve optimal policy"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    Next Steps"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1. Analyze training logs:"
echo "   grep 'Episode reward' checkpoints/*/train.log"
echo ""
echo "2. Plot learning curves:"
echo "   python scripts/plot_learning.py checkpoints/metis checkpoints/dqn checkpoints/bandit"
echo ""
echo "3. Run evaluation on test set:"
echo "   cargo run --release --bin eval -- --model metis --checkpoint checkpoints/metis/best.bin"
echo "   cargo run --release --bin eval -- --model dqn --checkpoint checkpoints/dqn/best.bin"
echo "   cargo run --release --bin eval -- --model bandit --checkpoint checkpoints/bandit/best.bin"
echo ""
echo "4. Compare exploration strategies:"
echo "   # For DQN: Try Thompson Sampling instead of Epsilon-Greedy"
echo "   ./examples/train_dqn.sh"
echo ""
echo "   # For Bandit: Try UCB instead of Thompson Sampling"
echo "   ./examples/train_bandit.sh"
echo ""
echo "Done! Happy training!"
