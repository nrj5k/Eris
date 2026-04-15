//! DFTracer `.pfw.gz` trace file parser.
//!
//! The DFTracer pfw format is a gzipped JSON array where each line is a JSON object.
//! Events follow the Chrome Trace Event format with metadata events (ph="M") and
//! complete events (ph="X"). Metadata events include HH (hostname), SH (cmdline),
//! and FH (file hash to path mapping). POSIX I/O operations are recorded as
//! complete events with cat="POSIX" and include operations like open, close,
//! pread, pwrite, mkdir, opendir, __lxstat, and __fxstat.

use std::collections::HashMap;
use std::path::Path;

use crate::error::{EnvError, Result};

/// POSIX I/O event types from DFTracer traces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DfTracerEventType {
    Pread,
    Pwrite, // future-proofing
    Open,
    Close,
    Mkdir,
    Opendir,
    Lxstat,
    Fxstat,
}

/// A single DFTracer trace event (POSIX I/O operation)
#[derive(Debug, Clone)]
pub struct DfTracerEvent {
    pub event_type: DfTracerEventType,
    pub timestamp_us: u64, // microseconds since epoch
    pub duration_us: u64,  // duration in microseconds
    pub pid: u64,
    pub tid: u64,
    pub fd: Option<u64>,
    pub fhash: Option<u64>,  // file hash (resolved via FH metadata)
    pub offset: Option<u64>, // read/write offset (pread/pwrite)
    pub count: Option<u64>,  // bytes requested (pread/pwrite)
    pub ret: Option<i64>,    // return value
    pub flags: Option<u64>,  // open flags
    pub mode: Option<u64>,   // mkdir mode
}

/// Metadata extracted from dftracer metadata events (HH, SH, FH)
#[derive(Debug, Clone, Default)]
pub struct DfTracerMetadata {
    pub hostname: Option<String>,
    pub cmdline: Option<String>,
    pub fhash_to_path: HashMap<u64, String>, // fhash → filepath
}

/// Parse a DFTracer `.pfw.gz` file and return all JSON values.
///
/// Decompresses the gzip file and parses the Chrome Trace Event format.
/// The file format is a NDJSON-in-array: line 1 is `[`, each subsequent line
/// is a JSON object (optionally with a trailing comma), and the last line is `]`.
/// Falls back to standard JSON array parsing for well-formed array files.
pub fn parse_pfw_gz(path: &Path) -> Result<Vec<serde_json::Value>> {
    use flate2::read::GzDecoder;
    use std::io::Read;

    let file = std::fs::File::open(path)
        .map_err(|e| EnvError::DfTracerError(format!("Failed to open pfw.gz: {}", e)))?;
    let mut decoder = GzDecoder::new(file);
    let mut content = String::new();
    decoder
        .read_to_string(&mut content)
        .map_err(|e| EnvError::DfTracerError(format!("Failed to decompress: {}", e)))?;

    let trimmed = content.trim();
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        if let Ok(values) = serde_json::from_str::<Vec<serde_json::Value>>(trimmed) {
            return Ok(values);
        }
    }

    let mut values = Vec::new();
    for line in content.lines() {
        let line = line.trim().trim_end_matches(',');
        if line.is_empty() || line == "[" || line == "]" {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            values.push(value);
        }
    }

    Ok(values)
}

impl DfTracerMetadata {
    /// Extract metadata from DFTracer JSON values.
    ///
    /// Iterates all JSON values, finds metadata events (ph="M"), and extracts:
    /// - HH events: hostname from args.name
    /// - SH events: cmdline from args.name
    /// - FH events: fhash (args.value) → filepath (args.name) mapping
    pub fn extract(values: &[serde_json::Value]) -> Self {
        let mut metadata = DfTracerMetadata::default();

        for value in values {
            // Check if this is a metadata event
            let ph = value.get("ph").and_then(|v| v.as_str());
            if ph != Some("M") {
                continue;
            }

            let name = value.get("name").and_then(|v| v.as_str());
            let args = value.get("args");

            match name {
                Some("HH") => {
                    metadata.hostname = args
                        .and_then(|a| a.get("name"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                }
                Some("SH") => {
                    metadata.cmdline = args
                        .and_then(|a| a.get("name"))
                        .and_then(|v| v.as_str())
                        .map(String::from);
                }
                Some("FH") => {
                    let fhash = args.and_then(|a| a.get("value")).and_then(|v| v.as_u64());
                    let filepath = args
                        .and_then(|a| a.get("name"))
                        .and_then(|v| v.as_str())
                        .map(String::from);

                    if let Some(fhash) = fhash {
                        if let Some(filepath) = filepath {
                            metadata.fhash_to_path.insert(fhash, filepath);
                        }
                    }
                }
                _ => {}
            }
        }

        metadata
    }
}

/// Extract POSIX I/O events from DFTracer JSON values.
///
/// Filters to events where ph="X", cat="POSIX", and name is a known POSIX operation.
/// Maps each event to DfTracerEvent with defensive field extraction.
pub fn extract_io_events(values: &[serde_json::Value]) -> Vec<DfTracerEvent> {
    let mut events = Vec::new();

    for value in values {
        // Check if this is a complete event in POSIX category
        let ph = value.get("ph").and_then(|v| v.as_str());
        let cat = value.get("cat").and_then(|v| v.as_str());
        let name = value.get("name").and_then(|v| v.as_str());

        if ph != Some("X") || cat != Some("POSIX") {
            continue;
        }

        let event_type = match name {
            Some("pread") => DfTracerEventType::Pread,
            Some("pwrite") => DfTracerEventType::Pwrite,
            Some("open") => DfTracerEventType::Open,
            Some("close") => DfTracerEventType::Close,
            Some("mkdir") => DfTracerEventType::Mkdir,
            Some("opendir") => DfTracerEventType::Opendir,
            Some("__lxstat") => DfTracerEventType::Lxstat,
            Some("__fxstat") => DfTracerEventType::Fxstat,
            _ => continue, // Skip unknown event names
        };

        let args = value.get("args");

        let event = DfTracerEvent {
            event_type,
            timestamp_us: value.get("ts").and_then(|v| v.as_u64()).unwrap_or(0),
            duration_us: value.get("dur").and_then(|v| v.as_u64()).unwrap_or(0),
            pid: value.get("pid").and_then(|v| v.as_u64()).unwrap_or(0),
            tid: value.get("tid").and_then(|v| v.as_u64()).unwrap_or(0),
            fd: args.and_then(|a| a.get("fd")).and_then(|v| v.as_u64()),
            fhash: args.and_then(|a| a.get("fhash")).and_then(|v| v.as_u64()),
            offset: args.and_then(|a| a.get("offset")).and_then(|v| v.as_u64()),
            count: args.and_then(|a| a.get("count")).and_then(|v| v.as_u64()),
            ret: args.and_then(|a| a.get("ret")).and_then(|v| v.as_i64()),
            flags: args.and_then(|a| a.get("flags")).and_then(|v| v.as_u64()),
            mode: args.and_then(|a| a.get("mode")).and_then(|v| v.as_u64()),
        };

        events.push(event);
    }

    events
}

impl DfTracerEvent {
    /// Resolve the file path for this event using metadata.
    ///
    /// Returns the filepath string if the fhash is present and found in metadata,
    /// or None if either the fhash is missing or not in the metadata map.
    pub fn resolved_filepath<'a>(&self, metadata: &'a DfTracerMetadata) -> Option<&'a str> {
        self.fhash
            .and_then(|h| metadata.fhash_to_path.get(&h))
            .map(|s| s.as_str())
    }
}

/// Per-file access tracking state for deriving BlobData fields
#[derive(Debug, Default)]
struct FileAccessState {
    /// Number of times this (fhash, offset) pair has been accessed
    access_count: u32,
    /// Timestamp of last access (microseconds)
    last_access_ts: Option<u64>,
}

/// Per-fd tracking state for sequential access detection
#[derive(Debug, Default)]
struct FdAccessState {
    /// Last offset read on this fd
    last_offset: Option<u64>,
    /// Last count (bytes) read on this fd
    last_count: Option<u64>,
}

/// Converts DFTracer I/O events into BlobData records for the RL environment.
///
/// The converter:
/// 1. Sorts events by timestamp
/// 2. Filters to data-bearing events only (Pread/Pwrite)
/// 3. Derives BlobData fields from access patterns
#[derive(Debug)]
pub struct DfTracerConverter {
    events: Vec<DfTracerEvent>,
    metadata: DfTracerMetadata,
}

impl DfTracerConverter {
    /// Create a new converter from extracted events and metadata.
    pub fn new(events: Vec<DfTracerEvent>, metadata: DfTracerMetadata) -> Self {
        Self { events, metadata }
    }

    /// Convert all DFTracer events into BlobData records.
    ///
    /// Only Pread/Pwrite events produce BlobData records.
    /// Open, close, stat, mkdir events are used for context but do not produce records.
    /// Events are sorted by timestamp before processing.
    pub fn convert(mut self) -> Vec<super::BlobData> {
        use super::BlobData;

        // Sort events by timestamp
        self.events.sort_by_key(|e| e.timestamp_us);

        let mut records = Vec::new();
        let mut blob_access: HashMap<(u64, u64), FileAccessState> = HashMap::new();
        let mut fd_state: HashMap<u64, FdAccessState> = HashMap::new();

        for event in &self.events {
            // Only Pread/Pwrite produce BlobData records
            if event.event_type != DfTracerEventType::Pread
                && event.event_type != DfTracerEventType::Pwrite
            {
                continue;
            }

            let fhash = match event.fhash {
                Some(h) => h,
                None => continue, // Skip events without file hash
            };

            let offset = event.offset.unwrap_or(0);
            let count = event.count.unwrap_or(0);
            let fd = event.fd.unwrap_or(0);

            // Derive offset_id: "{filepath}:{offset}" or "{fhash}:{offset}"
            let offset_id = match event.resolved_filepath(&self.metadata) {
                Some(path) => format!("{}:{}", path, offset),
                None => format!("{}:{}", fhash, offset),
            };

            // Derive access_frequency and first_seen from per-blob tracking
            let blob_key = (fhash, offset);
            let access_state = blob_access.entry(blob_key).or_default();
            let access_frequency = access_state.access_count;
            let first_seen = access_state.access_count == 0;
            access_state.access_count += 1;

            // Derive recency: ms since last access to this blob
            let recency = match access_state.last_access_ts {
                Some(last_ts) => {
                    let diff_us = event.timestamp_us.saturating_sub(last_ts);
                    let diff_ms = diff_us as f64 / 1000.0;
                    format!("{}", diff_ms)
                }
                None => "inf".to_string(),
            };
            access_state.last_access_ts = Some(event.timestamp_us);

            // Derive is_sequence: sequential reads on same fd
            let is_sequence = {
                let fd_state = fd_state.entry(fd).or_default();
                let seq = match (fd_state.last_offset, fd_state.last_count) {
                    (Some(last_off), Some(last_cnt)) => offset == last_off + last_cnt,
                    _ => false,
                };
                fd_state.last_offset = Some(offset);
                fd_state.last_count = Some(count);
                seq
            };

            // Derive offset_score: simple heuristic
            // Higher frequency + recent access → higher score
            let recency_factor = match &recency[..] {
                "inf" => 0.0,
                s => s.parse::<f64>().unwrap_or(0.0),
            };
            let recency_bonus = 1.0 / (1.0 + recency_factor / 1000.0); // Normalize to seconds
            let offset_score = (access_frequency as f32) + recency_bonus as f32;

            // Derive io_op
            let io_op = match event.event_type {
                DfTracerEventType::Pread => "read".to_string(),
                DfTracerEventType::Pwrite => "write".to_string(),
                _ => unreachable!(),
            };

            // Build BlobData
            let blob = BlobData {
                offset_id,
                offset_score,
                offset_access_frequency: access_frequency,
                access_offset: Some(offset as f64),
                access_size: count as f64,
                offset_size: count as f64, // Same as access_size for single reads
                is_sequence,
                first_seen,
                overwrite_amount: 0.0, // Reads don't overwrite
                recency,
                io_op,
            };

            records.push(blob);
        }

        records
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_pfw_gz_parsing() {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Build synthetic Chrome Trace Event JSON array
        let json = r#"[
{"id":1,"name":"HH","cat":"dftracer","pid":100,"tid":100,"ph":"M","args":{"hhash":1,"name":"testhost","value":1}},
{"id":2,"name":"FH","cat":"dftracer","pid":100,"tid":100,"ph":"M","args":{"hhash":1,"name":"/data/test.hdf5","value":999}},
{"id":3,"name":"open","cat":"POSIX","pid":100,"tid":100,"ts":1000,"dur":100,"ph":"X","args":{"hhash":1,"p_idx":1,"level":1,"ret":5,"fhash":999,"flags":0}},
{"id":4,"name":"pread","cat":"POSIX","pid":100,"tid":100,"ts":2000,"dur":10,"ph":"X","args":{"hhash":1,"p_idx":3,"fd":5,"fhash":999,"ret":1024,"count":1024,"offset":0,"level":2}},
{"id":5,"name":"pread","cat":"POSIX","pid":100,"tid":100,"ts":3000,"dur":5,"ph":"X","args":{"hhash":1,"p_idx":4,"fd":5,"fhash":999,"ret":1024,"count":1024,"offset":1024,"level":2}},
{"id":6,"name":"close","cat":"POSIX","pid":100,"tid":100,"ts":4000,"dur":50,"ph":"X","args":{"hhash":1,"p_idx":5,"level":2,"ret":0,"fhash":999,"fd":5}}
]"#;

        // Write to gzipped temp file
        let mut temp = NamedTempFile::new().unwrap();
        {
            let mut encoder = GzEncoder::new(&mut temp, Compression::fast());
            encoder.write_all(json.as_bytes()).unwrap();
        }

        let values = parse_pfw_gz(temp.path()).unwrap();
        assert_eq!(values.len(), 6); // 6 events total
    }

    #[test]
    fn test_metadata_extraction() {
        use serde_json::json;

        let values = vec![
            json!({"id":1,"name":"HH","cat":"dftracer","pid":100,"tid":100,"ph":"M","args":{"hhash":1,"name":"testhost","value":1}}),
            json!({"id":2,"name":"FH","cat":"dftracer","pid":100,"tid":100,"ph":"M","args":{"hhash":1,"name":"/data/test.hdf5","value":999}}),
            json!({"id":3,"name":"SH","cat":"dftracer","pid":100,"tid":100,"ph":"M","args":{"hhash":1,"name":"python test.py","value":42}}),
        ];

        let metadata = DfTracerMetadata::extract(&values);
        assert_eq!(metadata.hostname.as_deref(), Some("testhost"));
        assert_eq!(metadata.cmdline.as_deref(), Some("python test.py"));
        assert_eq!(
            metadata.fhash_to_path.get(&999),
            Some(&"/data/test.hdf5".to_string())
        );
    }

    #[test]
    fn test_io_event_extraction() {
        use serde_json::json;

        let values = vec![
            json!({"name":"pread","cat":"POSIX","pid":100,"tid":100,"ts":2000,"dur":10,"ph":"X","args":{"hhash":1,"fd":5,"fhash":999,"ret":1024,"count":1024,"offset":0,"level":2}}),
            json!({"name":"open","cat":"POSIX","pid":100,"tid":100,"ts":1000,"dur":100,"ph":"X","args":{"hhash":1,"level":1,"ret":5,"fhash":999,"flags":0}}),
            json!({"name":"DLIOBenchmark.__init__","cat":"dlio_benchmark","pid":100,"tid":100,"ts":500,"dur":10,"ph":"X","args":{"hhash":1,"level":1}}), // non-POSIX, should be skipped
        ];

        let events = extract_io_events(&values);
        assert_eq!(events.len(), 2); // pread + open only
        assert_eq!(events[0].event_type, DfTracerEventType::Pread);
        assert_eq!(events[1].event_type, DfTracerEventType::Open);
        assert_eq!(events[0].fhash, Some(999));
    }

    #[test]
    fn test_converter_derived_fields() {
        let metadata = DfTracerMetadata {
            hostname: Some("testhost".into()),
            cmdline: None,
            fhash_to_path: HashMap::from([(999u64, "/data/test.hdf5".to_string())]),
        };

        let events = vec![
            DfTracerEvent {
                event_type: DfTracerEventType::Pread,
                timestamp_us: 2000,
                duration_us: 10,
                pid: 100,
                tid: 100,
                fd: Some(5),
                fhash: Some(999),
                offset: Some(0),
                count: Some(1024),
                ret: Some(1024),
                flags: None,
                mode: None,
            },
            DfTracerEvent {
                event_type: DfTracerEventType::Pread,
                timestamp_us: 3000,
                duration_us: 5,
                pid: 100,
                tid: 100,
                fd: Some(5),
                fhash: Some(999),
                offset: Some(1024),
                count: Some(1024),
                ret: Some(1024),
                flags: None,
                mode: None,
            },
            DfTracerEvent {
                event_type: DfTracerEventType::Pread,
                timestamp_us: 50000,
                duration_us: 5,
                pid: 100,
                tid: 100,
                fd: Some(5),
                fhash: Some(999),
                offset: Some(0),
                count: Some(512),
                ret: Some(512),
                flags: None,
                mode: None,
            },
            // close does NOT produce a BlobData record
            DfTracerEvent {
                event_type: DfTracerEventType::Close,
                timestamp_us: 60000,
                duration_us: 50,
                pid: 100,
                tid: 100,
                fd: Some(5),
                fhash: Some(999),
                offset: None,
                count: None,
                ret: Some(0),
                flags: None,
                mode: None,
            },
        ];

        let converter = DfTracerConverter::new(events, metadata);
        let records = converter.convert();

        assert_eq!(records.len(), 3); // 3 preads, close filtered out

        // First blob: first_seen=true, recency="inf", is_sequence=false (no predecessor)
        assert!(records[0].first_seen);
        assert_eq!(records[0].recency, "inf");
        assert!(!records[0].is_sequence);
        assert_eq!(records[0].access_size, 1024.0);
        assert_eq!(records[0].io_op, "read");
        assert!(records[0].offset_id.contains("/data/test.hdf5"));

        // Second blob: first_seen=true (new offset 1024), is_sequence=true (sequential offset)
        // recency="inf" because this is first access to (fhash=999, offset=1024) pair
        assert!(records[1].first_seen); // First access to offset 1024
        assert!(records[1].is_sequence); // offset 1024 follows 0+1024
        assert_eq!(records[1].offset_access_frequency, 0); // First access to this (fhash, offset) pair
        assert_eq!(records[1].recency, "inf"); // No previous access to (999, 1024)

        // Third blob: re-reads offset 0, so first_seen=false (not first time for this offset)
        assert!(!records[2].first_seen);
        assert_eq!(records[2].offset_access_frequency, 1); // seen once before (from record 0)
        assert!(!records[2].is_sequence); // offset 0 != 1024+1024
        assert_eq!(records[2].recency, "48"); // 50000-2000 = 48000us = 48ms
    }

    #[test]
    fn test_resolved_filepath_missing() {
        let metadata = DfTracerMetadata::default();
        let event = DfTracerEvent {
            event_type: DfTracerEventType::Pread,
            timestamp_us: 0,
            duration_us: 0,
            pid: 0,
            tid: 0,
            fd: None,
            fhash: Some(999),
            offset: None,
            count: None,
            ret: None,
            flags: None,
            mode: None,
        };
        // fhash 999 not in metadata
        assert!(event.resolved_filepath(&metadata).is_none());

        // fhash None
        let event_no_fhash = DfTracerEvent {
            event_type: DfTracerEventType::Pread,
            timestamp_us: 0,
            duration_us: 0,
            pid: 0,
            tid: 0,
            fd: None,
            fhash: None,
            offset: None,
            count: None,
            ret: None,
            flags: None,
            mode: None,
        };
        assert!(event_no_fhash.resolved_filepath(&metadata).is_none());
    }

    #[test]
    fn test_from_pfw_real_file() {
        use crate::trace::TraceReader;
        use std::path::Path;

        let pfw_path = Path::new("dftracer-pfw/unet3d.pfw.gz");
        if !pfw_path.exists() {
            eprintln!("Skipping: unet3d.pfw.gz not found");
            return;
        }

        let reader = TraceReader::from_pfw(pfw_path);
        assert!(reader.is_ok(), "from_pfw failed: {:?}", reader.err());

        let mut reader = reader.unwrap();
        assert!(
            reader.len() > 100,
            "Expected at least 100 records, got {}",
            reader.len()
        );

        let first = reader.next();
        assert!(first.is_some());
        let first = first.unwrap();
        assert!(!first.offset_id.is_empty(), "offset_id should not be empty");
        assert!(first.access_size > 0.0, "access_size should be positive");
        assert_eq!(first.io_op, "read");

        // Record count should be close to 945 (number of pread events)
        println!(
            "✓ Loaded {} BlobData records from unet3d.pfw.gz",
            reader.len()
        );
    }
}
