use serde::{Deserialize, Deserializer};

/// I/O operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoOp {
    Read,
    Write,
}

impl From<&str> for IoOp {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "write" => IoOp::Write,
            _ => IoOp::Read, // Default to Read
        }
    }
}

impl std::fmt::Display for IoOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoOp::Read => write!(f, "read"),
            IoOp::Write => write!(f, "write"),
        }
    }
}

/// Custom deserializer for boolean fields from CSV that use "True"/"False"
fn deserialize_csv_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let s: String = Deserialize::deserialize(deserializer)?;
    match s.as_str() {
        "True" => Ok(true),
        "False" => Ok(false),
        "true" => Ok(true),
        "false" => Ok(false),
        "1" => Ok(true),
        "0" => Ok(false),
        _ => Err(serde::de::Error::custom(format!(
            "Invalid boolean value: {}",
            s
        ))),
    }
}

/// Represents a blob access record from the trace
#[derive(Debug, Clone, Deserialize)]
pub struct BlobData {
    /// Unique blob identifier
    pub offset_id: String,
    /// Blob score (used for prioritization)
    pub offset_score: f32,
    /// Number of times this blob has been accessed
    pub offset_access_frequency: u32,
    /// Access offset within blob (optional)
    pub access_offset: Option<f64>,
    /// Size of this access in bytes
    pub access_size: f64,
    /// Total size of the blob in bytes
    pub offset_size: f64,
    /// Whether this is a sequential access pattern
    #[serde(deserialize_with = "deserialize_csv_bool")]
    pub is_sequence: bool,
    /// Whether this is the first time seeing this blob
    #[serde(deserialize_with = "deserialize_csv_bool")]
    pub first_seen: bool,
    /// Amount of data overwritten (for writes)
    pub overwrite_amount: f32,
    /// Time since last access (ms or "inf")
    pub recency: String,
    /// I/O operation type ("read" or "write")
    pub io_op: String,
}

impl BlobData {
    /// Parse recency value into milliseconds, returning None for "inf"
    pub fn recency_ms(&self) -> Option<f64> {
        match self.recency.as_str() {
            "inf" => None,
            s => s.parse::<f64>().ok(),
        }
    }

    /// Check if this is a read operation
    pub fn is_read(&self) -> bool {
        self.io_op == "read"
    }

    /// Get the I/O operation as an enum
    pub fn io_op_enum(&self) -> IoOp {
        IoOp::from(self.io_op.as_str())
    }
}
