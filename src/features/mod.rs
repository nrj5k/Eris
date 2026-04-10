mod extractor;
mod hotness;
mod tracker;

pub use extractor::{aligned_state_dim, encode_state, pad_to_warp_size, BlobFeatures};
pub use hotness::{hotness_score, HotnessConfig};
pub use tracker::{AccessRecord, AccessTracker};

/// GPU warp size for optimal memory coalescing
pub const WARP_SIZE: usize = 32;
