//! Timing and diagnostic utilities

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tracing;

/// A diagnostic helper that prints once per program execution
///
/// Use this to avoid spamming diagnostic messages on every call.
pub struct OneTimeDiag {
    printed: AtomicBool,
}

impl OneTimeDiag {
    /// Create a new one-time diagnostic (const, can be used in static)
    pub const fn new() -> Self {
        Self {
            printed: AtomicBool::new(false),
        }
    }

    /// Check if this is the first call and should print
    ///
    /// Returns true only on the first call, false thereafter.
    pub fn should_print(&self) -> bool {
        !self.printed.swap(true, Ordering::Relaxed)
    }

    /// Run a closure only on the first call
    ///
    /// Usage: `DIAG.run_once(|| println!("first time!"));`
    pub fn run_once<F>(&self, f: F)
    where
        F: FnOnce(),
    {
        if self.should_print() {
            f();
        }
    }
}

impl Default for OneTimeDiag {
    fn default() -> Self {
        Self::new()
    }
}

/// Log step timing with appropriate level based on threshold
///
/// Logs at debug level normally, warn level if above threshold.
pub fn log_step_timing(step_count: usize, label: &str, elapsed: Duration, threshold_ms: u64) {
    let elapsed_ms = elapsed.as_millis() as u64;

    if elapsed_ms > threshold_ms {
        tracing::warn!(
            "{} took {}ms (step {}) - SLOW",
            label,
            elapsed_ms,
            step_count
        );
    } else {
        tracing::trace!(
            "[STAGE:TIME] {} took {}ms (step {})",
            label,
            elapsed_ms,
            step_count
        );
    }
}

/// Log first-call diagnostics
///
/// Combines OneTimeDiag with logging.
pub fn log_first_call(context: &str) {
    tracing::debug!("[STAGE:DIAG] {} (first call only)", context);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_time_diag() {
        static DIAG: OneTimeDiag = OneTimeDiag::new();

        // First call should return true
        assert!(DIAG.should_print());

        // Subsequent calls should return false
        assert!(!DIAG.should_print());
        assert!(!DIAG.should_print());
    }

    #[test]
    fn test_run_once() {
        static DIAG: OneTimeDiag = OneTimeDiag::new();
        let mut counter = 0;

        DIAG.run_once(|| counter += 1);
        DIAG.run_once(|| counter += 1);
        DIAG.run_once(|| counter += 1);

        // Should only run once
        assert_eq!(counter, 1);
    }
}
