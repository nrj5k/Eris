use crate::error::{EnvError, Result};
use crate::trace::BlobData;
use std::path::Path;

/// Reader for CSV trace files
#[derive(Debug, Clone)]
pub struct TraceReader {
    /// All records loaded from the trace
    records: Vec<BlobData>,
    /// Current position in the records
    current_idx: usize,
}

impl TraceReader {
    /// Load trace data from a CSV file
    pub fn from_csv(path: &Path) -> Result<Self> {
        let mut reader = csv::Reader::from_path(path)
            .map_err(|e| EnvError::CsvError(format!("Failed to open CSV: {}", e)))?;

        let records: Vec<BlobData> = reader
            .deserialize()
            .filter_map(|result| result.ok())
            .collect();

        if records.is_empty() {
            return Err(EnvError::CsvError("No records found in trace file".into()));
        }

        Ok(Self {
            records,
            current_idx: 0,
        })
    }

    /// Get the next blob record without consuming it
    pub fn next(&mut self) -> Option<&BlobData> {
        if self.current_idx < self.records.len() {
            let record = &self.records[self.current_idx];
            self.current_idx += 1;
            Some(record)
        } else {
            None
        }
    }

    /// Reset reader to the beginning of the trace
    pub fn reset(&mut self) {
        self.current_idx = 0;
    }

    /// Get total number of records
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if trace is empty
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

impl Iterator for TraceReader {
    type Item = BlobData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx < self.records.len() {
            let record = self.records[self.current_idx].clone();
            self.current_idx += 1;
            Some(record)
        } else {
            None
        }
    }
}
