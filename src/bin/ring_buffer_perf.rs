use rand::prelude::*;
use rand::rng;
use std::time::Instant;

fn main() {
    println!("\nRing Buffer Performance Comparison");
    println!("===================================\n");

    let capacities = vec![10_000, 50_000, 100_000];
    let batch_sizes = vec![32, 128, 512, 2048]; // Include optimized default

    for &capacity in &capacities {
        for &batch_size in &batch_sizes {
            println!("Capacity: {}, Batch size: {}", capacity, batch_size);
            println!("----------------------------------------");

            // Generate dummy data
            let buffer: Vec<f32> = (0..capacity).map(|i| i as f32).collect();

            // Benchmark OLD approach (VecDeque-style: collect then sample)
            let start = Instant::now();
            for _ in 0..100 {
                let _ = sample_old(&buffer, batch_size);
            }
            let old_time = start.elapsed();
            let old_avg = old_time.as_nanos() / 100;

            // Benchmark NEW approach (RingBuffer: direct index sampling)
            let start = Instant::now();
            for _ in 0..100 {
                let _ = sample_new(&buffer, capacity, batch_size);
            }
            let new_time = start.elapsed();
            let new_avg = new_time.as_nanos() / 100;

            let speedup = old_time.as_secs_f64() / new_time.as_secs_f64();

            println!("OLD (VecDeque): {:>8} ns/sample", old_avg);
            println!("NEW (RingBuffer): {:>8} ns/sample", new_avg);
            println!("Speedup: {:.2}x", speedup);
            println!();
        }
    }
}

/// OLD APPROACH: VecDeque-style O(n) sampling
fn sample_old(buffer: &[f32], batch_size: usize) -> Vec<f32> {
    let mut rng = rng();

    // COLLECT all items (O(n) memory + time)
    // This is what VecDeque does in the old implementation
    let items: Vec<&f32> = buffer.iter().collect();

    // SAMPLE using the rand slice method
    // .sample() returns IndexedSamples, we need to handle nested references
    let sampled: Vec<f32> = items
        .sample(&mut rng, batch_size)
        .into_iter()
        .map(|&&f| f) // Dereference twice (&&f32 -> f32)
        .collect();

    sampled
}

/// NEW APPROACH: Ring buffer O(batch_size) sampling
fn sample_new(buffer: &[f32], size: usize, batch_size: usize) -> Vec<f32> {
    let mut rng = rng();

    // GENERATE random indices (O(batch_size))
    let indices: Vec<usize> = (0..batch_size).map(|_| rng.random_range(0..size)).collect();

    // ACCESS directly (O(batch_size))
    indices.iter().map(|&i| buffer[i]).collect()
}
