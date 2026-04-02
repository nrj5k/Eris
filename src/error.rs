use thiserror::Error;

/// Comprehensive error types for the environment
#[derive(Error, Debug)]
pub enum EnvError {
    #[error("invalid tier index {provided} (max {max})")]
    InvalidTierIndex { provided: usize, max: usize },

    #[error("tier {tier_id} capacity exceeded: requested {requested} bytes, available {available} bytes")]
    CapacityExceeded {
        tier_id: u32,
        requested: f64,
        available: f64,
    },

    #[error("blob not found: {blob_id}")]
    BlobNotFound { blob_id: String },

    #[error("invalid action: {action} (valid range: 0-{max})")]
    InvalidAction { action: usize, max: usize },

    #[error("configuration error: {message}")]
    ConfigError { message: String },

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CSV parsing error: {0}")]
    CsvError(String),

    #[error("trace exhausted")]
    TraceExhausted,
}

/// Result type alias for convenience
pub type Result<T> = std::result::Result<T, EnvError>;
