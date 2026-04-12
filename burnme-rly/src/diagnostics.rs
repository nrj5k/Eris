use std::time::Instant;

/// Simple timing struct (KISS)
#[derive(Debug, Default)]
pub struct SimpleTimer {
    start: Option<Instant>,
}

impl SimpleTimer {
    pub fn new() -> Self {
        Self { start: None }
    }
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }
    pub fn elapsed_ms(&self) -> f64 {
        self.start
            .map(|s| s.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0)
    }
}

/// Log timing if slow (KISS - only log when needed)
#[macro_export]
macro_rules! time_if_slow {
    ($name:expr, $threshold_ms:expr, $block:block) => {{
        let start = std::time::Instant::now();
        let result = $block;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        if elapsed > $threshold_ms {
            log::warn!("[SLOW] {}: {:.2}ms", $name, elapsed);
        }
        result
    }};
}

/// Print GPU backend info (simple)
pub fn log_backend_info<B: burn::tensor::backend::Backend>(_: &B::Device) {
    log::info!("Using Burn backend for training");
}
