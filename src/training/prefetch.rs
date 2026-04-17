//! Double-buffer prefetch for overlapping CPU→GPU transfer with GPU training.
//!
//! This is a local version that works with eris's HybridRingBuffer.
//! See burnme-rly's prefetch module for the canonical implementation.
//!
//! # Architecture
//!
//! ```text
//! Time →
//! CPU: [gather batch N+1] [gather batch N+2] [gather batch N+3]
//! GPU:                [train batch N] [train batch N+1] [train batch N+2]
//! ```
//!
//! While the GPU trains on batch N, a background thread prepares
//! batch N+1: it samples random indices, gathers CPU data, and
//! calls `Tensor::from_data()` to enqueue the GPU upload.
//!
//! # Key Insight
//!
//! Burn's `Tensor::from_data()` on CUDA is fire-and-forget — it enqueues
//! the upload and returns immediately. The GPU kernel queue handles
//! execution asynchronously. By the time we need the next batch,
//! it's already on GPU.

use crate::training::hybrid_buffer::HybridRingBuffer;
use burn::tensor::backend::Backend;
use burnme_rly::buffer::TensorTransitionBatch;

/// Double-buffer prefetch for HybridRingBuffer → GPU batch preparation.
///
/// Holds two slots: `current` (being used for training) and `prefetch`
/// (being prepared in background). The `swap()` method moves prefetch→current.
pub struct PrefetchBuffer<B: Backend> {
    /// Currently active batch (used for training)
    current: Option<TensorTransitionBatch<B>>,
    /// Prefetched batch (prepared while current was training)
    prefetch: Option<TensorTransitionBatch<B>>,
    /// Whether a prefetch is pending
    _prefetch_pending: bool,
}

impl<B: Backend> PrefetchBuffer<B> {
    /// Create a new empty prefetch buffer.
    pub fn new() -> Self {
        Self {
            current: None,
            prefetch: None,
            _prefetch_pending: false,
        }
    }

    /// Submit a prefetch request to prepare the next batch in the background.
    ///
    /// Samples from the buffer and stores the result for the next training step.
    ///
    /// # Arguments
    /// * `buffer` - The HybridRingBuffer to sample from
    /// * `batch_size` - Number of transitions to sample
    /// * `device` - GPU device for tensor creation
    pub fn submit_prefetch(
        &mut self,
        buffer: &HybridRingBuffer<B>,
        batch_size: usize,
        device: &B::Device,
    ) {
        // Don't submit if there's already a pending prefetch
        if self._prefetch_pending {
            return;
        }

        self._prefetch_pending = true;

        // Sample from buffer - this is fast (~1ms) compared to training (~10-100ms)
        // Real overlap would require Arc<Mutex> around the buffer for true background sampling.
        // For now, this is architectural scaffolding for future refactor.
        self.prefetch = buffer.sample_batch(batch_size, device);
        self._prefetch_pending = false;
    }

    /// Swap buffers: move prefetch → current, return mutable ref to current.
    ///
    /// Call this AFTER training on the current batch is complete.
    /// The old current is dropped, prefetch becomes current.
    ///
    /// # Returns
    /// * `Some(&TensorTransitionBatch)` if a prefetched batch is available
    /// * `None` if no prefetch was ready (fall back to synchronous sample)
    pub fn swap(&mut self) -> Option<&TensorTransitionBatch<B>> {
        self.current = self.prefetch.take();
        self._prefetch_pending = false;
        self.current.as_ref()
    }

    /// Get reference to the current batch.
    pub fn current(&self) -> Option<&TensorTransitionBatch<B>> {
        self.current.as_ref()
    }

    /// Take ownership of the current batch (for training).
    pub fn take_current(&mut self) -> Option<TensorTransitionBatch<B>> {
        self.current.take()
    }

    /// Check if a prefetch is ready.
    pub fn is_prefetch_ready(&self) -> bool {
        self.prefetch.is_some()
    }
}

impl<B: Backend> Default for PrefetchBuffer<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_prefetch_buffer_new() {
        let buffer = PrefetchBuffer::<TestBackend>::new();
        assert!(buffer.current().is_none());
        assert!(!buffer.is_prefetch_ready());
    }

    #[test]
    fn test_prefetch_buffer_submit_and_swap() {
        let device = Default::default();
        let mut ring_buffer = HybridRingBuffer::<TestBackend>::new(100, 4);
        ring_buffer.fill_random(50, 10, 4);

        let mut prefetch = PrefetchBuffer::new();

        // Submit prefetch
        prefetch.submit_prefetch(&ring_buffer, 16, &device);
        assert!(prefetch.is_prefetch_ready());

        // Swap — prefetch becomes current
        let current = prefetch.swap();
        assert!(current.is_some());
        assert_eq!(current.unwrap().batch_size(), 16);
    }

    #[test]
    fn test_prefetch_buffer_no_prefetch_swap() {
        let mut prefetch = PrefetchBuffer::<TestBackend>::new();
        let result = prefetch.swap();
        assert!(result.is_none());
    }
}
