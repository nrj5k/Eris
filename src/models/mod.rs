mod bandit;
mod combined;
mod dqn;

pub use bandit::{ContextualBandit, ContextualBanditConfig};
pub use combined::{CombinedModel, CombinedModelConfig, decode_action, encode_action};
pub use dqn::{QNetwork, QNetworkConfig};

// Note: Use the new configuration API from src/config/ for new projects
// The old configs (ContextualBanditConfig, QNetworkConfig, CombinedModelConfig)
// remain for backwards compatibility. Use eris::config::{BanditConfig, DQNConfig, CombinedBanditDQNConfig}
// for new code with better validation and builder patterns.
