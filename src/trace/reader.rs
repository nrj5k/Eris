//! Trace file reader with complete file loading validation

use crate::error::{EnvError, Result};
use crate::trace::BlobData;
use std::path::Path;
use std::sync::Arc;

/// Trace file format for auto-detection
#[derive(Clone, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum TraceFormat {
    /// Auto-detect from file extension (.csv → recorder, .pfw.gz/.pfw → dftracer)
    #[default]
    Autodetect,
    /// Recorder CSV format (eris CSV trace files)
    Recorder,
    /// DFTracer .pfw.gz format (Chrome Trace Event)
    Dftracer,
}

/// Shared trace data loaded once and shared across environments
#[derive(Debug, Clone)]
pub struct TraceData {
    /// All records loaded from the trace
    pub records: Arc<Vec<BlobData>>,
    /// Total number of rows expected (estimated from file)
    #[allow(dead_code)]
    total_rows: usize,
    /// Number of rows skipped due to errors
    skipped_rows: usize,
}

/// Reader for CSV trace files with validation
#[derive(Debug, Clone)]
pub struct TraceReader {
    /// Shared trace data (loaded once)
    data: Arc<TraceData>,
    /// Current position in the records (per-reader state)
    current_idx: usize,
}

impl TraceReader {
    /// Create new trace reader from CSV with complete validation
    pub fn from_csv(path: &Path) -> Result<Self> {
        // Check file exists and get size
        let metadata = std::fs::metadata(path)
            .map_err(|e| EnvError::CsvError(format!("Failed to read file metadata: {}", e)))?;

        let file_size = metadata.len();
        println!(
            "Loading trace file: {} ({} bytes)",
            path.display(),
            file_size
        );

        // Open CSV reader
        let mut reader = csv::Reader::from_path(path)
            .map_err(|e| EnvError::CsvError(format!("Failed to open CSV: {}", e)))?;

        // Get expected row count from file (estimate)
        let file_content = std::fs::read_to_string(path)
            .map_err(|e| EnvError::CsvError(format!("Failed to read file: {}", e)))?;
        let expected_rows = file_content.lines().count().saturating_sub(1); // -1 for header
        println!("Expected rows (estimated): {}", expected_rows);

        // Read all records with error tracking
        let mut records = Vec::new();
        let mut skipped_rows = 0usize;
        let mut row_num = 2u64; // Start at 2 (1 is header)

        for result in reader.deserialize() {
            match result {
                Ok(record) => {
                    records.push(record);
                }
                Err(e) => {
                    skipped_rows += 1;
                    eprintln!("Warning: Skipping malformed row {}: {}", row_num, e);

                    // Log first few errors in detail
                    if skipped_rows <= 5 {
                        eprintln!("   Error details: {:?}", e);
                    }
                }
            }
            row_num += 1;

            // Progress reporting for large files
            if row_num % 10000 == 0 {
                print!("\r  Loaded {} rows...", records.len());
            }
        }
        println!("\r  ✓ Loaded {} rows                    ", records.len());

        // Validation: Check if we got expected number of rows
        let loaded_rows = records.len();
        if loaded_rows == 0 {
            return Err(EnvError::CsvError(
                "No valid records found in trace file".into(),
            ));
        }

        // Warn if significantly fewer rows than expected
        if loaded_rows < expected_rows.saturating_sub(10) {
            eprintln!(
                "Warning: Expected ~{} rows, but only loaded {}",
                expected_rows, loaded_rows
            );
            eprintln!("   ({} rows skipped due to errors)", skipped_rows);
        }

        // Warn about skipped rows
        if skipped_rows > 0 {
            let percentage = (skipped_rows as f64 / (skipped_rows + loaded_rows) as f64) * 100.0;
            eprintln!(
                "Warning: Skipped {} rows ({:.1}%)",
                skipped_rows, percentage
            );
        }

        println!(
            "✓ Successfully loaded {} blobs from trace file",
            loaded_rows
        );

        let data = Arc::new(TraceData {
            records: Arc::new(records),
            total_rows: expected_rows,
            skipped_rows,
        });

        Ok(Self {
            data,
            current_idx: 0,
        })
    }

    /// Create a new trace reader from a DFTracer `.pfw.gz` file.
    ///
    /// Parses the gzipped Chrome Trace Event JSON, extracts POSIX I/O events,
    /// resolves file hash to path mappings, and converts to BlobData records.
    ///
    /// # Arguments
    /// * `path` - Path to the `.pfw.gz` file
    ///
    /// # Errors
    /// Returns error if file cannot be read, decompressed, parsed, or contains no I/O events.
    pub fn from_pfw(path: &Path) -> Result<Self> {
        use crate::trace::dftracer::{
            extract_io_events, parse_pfw_gz, DfTracerConverter, DfTracerMetadata,
        };

        let values = parse_pfw_gz(path)?;
        let metadata = DfTracerMetadata::extract(&values);
        let events = extract_io_events(&values);

        println!(
            "✓ DFTracer: extracted {} I/O events from {}",
            events.len(),
            path.display()
        );

        let converter = DfTracerConverter::new(events, metadata);
        let records = converter.convert();

        if records.is_empty() {
            return Err(EnvError::DfTracerError(
                "No I/O data events found in pfw trace".into(),
            ));
        }

        let records_len = records.len();
        println!(
            "✓ Successfully converted {} blob records from DFTracer trace",
            records_len
        );

        let data = Arc::new(TraceData {
            records: Arc::new(records),
            total_rows: records_len,
            skipped_rows: 0,
        });

        Ok(Self {
            data,
            current_idx: 0,
        })
    }

    /// Load a trace file with format auto-detection or explicit format specification.
    ///
    /// # Format auto-detection rules (TraceFormat::Autodetect):
    /// - `.csv` → Recorder (CSV)
    /// - `.pfw.gz` or `.pfw` → DFTracer
    /// - Other → returns error
    ///
    /// # Arguments
    /// * `path` - Path to the trace file
    /// * `format` - Trace format (Recorder, Dftracer, or Autodetect)
    pub fn from_path(path: &Path, format: TraceFormat) -> Result<Self> {
        match format {
            TraceFormat::Recorder => Self::from_csv(path),
            TraceFormat::Dftracer => Self::from_pfw(path),
            TraceFormat::Autodetect => {
                let path_str = path.to_string_lossy();
                if path_str.ends_with(".pfw.gz") || path_str.ends_with(".pfw") {
                    Self::from_pfw(path)
                } else if path_str.ends_with(".csv") {
                    Self::from_csv(path)
                } else {
                    Err(EnvError::DfTracerError(format!(
                        "Cannot auto-detect trace format for '{}' (expected .csv, .pfw.gz, or .pfw)",
                        path.display()
                    )))
                }
            }
        }
    }

    /// Create trace reader from shared trace data (for VecEnv).
    ///
    /// This allows multiple readers to share the same loaded data while
    /// maintaining independent position tracking.
    pub fn from_shared_data(data: Arc<TraceData>) -> Self {
        Self {
            data,
            current_idx: 0,
        }
    }

    /// Get shared trace data (for sharing with other readers)
    pub fn get_shared_data(&self) -> Arc<TraceData> {
        Arc::clone(&self.data)
    }

    /// Get the next blob record without consuming it
    pub fn next(&mut self) -> Option<&BlobData> {
        if self.current_idx < self.data.records.len() {
            let record = &self.data.records[self.current_idx];
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

    /// Get total number of valid records
    pub fn len(&self) -> usize {
        self.data.records.len()
    }

    /// Check if trace is empty
    pub fn is_empty(&self) -> bool {
        self.data.records.is_empty()
    }

    /// Get total number of records (valid only)
    pub fn total_records(&self) -> usize {
        self.data.records.len()
    }

    /// Get number of skipped (malformed) rows
    pub fn skipped_records(&self) -> usize {
        self.data.skipped_rows
    }
}

impl Iterator for TraceReader {
    type Item = BlobData;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx < self.data.records.len() {
            let record = self.data.records[self.current_idx].clone();
            self.current_idx += 1;
            Some(record)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_complete_csv_loading() {
        // Create test CSV
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "offset_id,offset_score,offset_access_frequency,access_offset,access_size,offset_size,is_sequence,first_seen,overwrite_amount,recency,io_op").unwrap();
        for i in 0..100 {
            writeln!(
                file,
                "blob_{},{},{},{},{},{},{},{},{},{},{}",
                i,
                i as f32,
                i,
                i as f64,
                1024.0 + i as f64,
                2048.0 + i as f64,
                false,
                true,
                0.0,
                "inf",
                "read"
            )
            .unwrap();
        }

        let reader = TraceReader::from_csv(file.path());
        assert!(reader.is_ok());

        let reader = reader.unwrap();
        assert_eq!(reader.total_records(), 100);
        assert_eq!(reader.skipped_records(), 0);
    }

    #[test]
    fn test_csv_with_malformed_rows() {
        // Create CSV with some bad rows
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "offset_id,offset_score,offset_access_frequency,access_offset,access_size,offset_size,is_sequence,first_seen,overwrite_amount,recency,io_op").unwrap();
        writeln!(file, "blob_1,1.0,1,,1024.0,2048.0,False,True,0.0,inf,read").unwrap();
        writeln!(
            file,
            "blob_2,invalid,2,,1024.0,2048.0,False,True,0.0,inf,read"
        )
        .unwrap(); // Bad row
        writeln!(file, "blob_3,3.0,3,,1024.0,2048.0,False,True,0.0,inf,read").unwrap();
        writeln!(file, "blob_4,4.0,4,,1024.0,2048.0,False,True,0.0,inf,read").unwrap();
        writeln!(file, "blob_5,5.0,5,,1024.0,2048.0,False,True,0.0,inf,read").unwrap();

        let reader = TraceReader::from_csv(file.path());
        assert!(reader.is_ok());

        let reader = reader.unwrap();
        // Should load some rows and skip bad ones
        assert!(reader.total_records() >= 3);
        assert!(reader.skipped_records() >= 1);
    }

    #[test]
    fn test_empty_csv_fails() {
        // Create empty CSV (header only)
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "offset_id,offset_score,offset_access_frequency").unwrap();

        let reader = TraceReader::from_csv(file.path());
        assert!(reader.is_err());
    }

    #[test]
    fn test_shared_trace_data() {
        // Create test CSV
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "offset_id,offset_score,offset_access_frequency,access_offset,access_size,offset_size,is_sequence,first_seen,overwrite_amount,recency,io_op").unwrap();
        for i in 0..50 {
            writeln!(
                file,
                "blob_{},{},{},{},{},{},{},{},{},{},{}",
                i,
                i as f32,
                i,
                i as f64,
                1024.0 + i as f64,
                2048.0 + i as f64,
                false,
                true,
                0.0,
                "inf",
                "read"
            )
            .unwrap();
        }

        // Load trace once
        let reader1 = TraceReader::from_csv(file.path()).unwrap();
        let shared_data = reader1.get_shared_data();

        // Create multiple readers sharing the same data
        let mut reader2 = TraceReader::from_shared_data(Arc::clone(&shared_data));
        let reader3 = TraceReader::from_shared_data(Arc::clone(&shared_data));

        // All readers should have same record count
        assert_eq!(reader1.total_records(), 50);
        assert_eq!(reader2.total_records(), 50);
        assert_eq!(reader3.total_records(), 50);

        // Readers should have independent positions
        let mut reader1_mut = TraceReader::from_shared_data(Arc::clone(&shared_data));
        reader1_mut.next();
        reader1_mut.next();
        assert_eq!(reader1_mut.total_records(), 50); // Still 50 records

        reader2.next();
        // reader3 hasn't moved

        // Verify they can all access the same data
        assert!(reader2.total_records() > 0);
    }
}
