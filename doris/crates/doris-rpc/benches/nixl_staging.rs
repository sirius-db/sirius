//! Criterion benchmarks for nixl exchange staging buffer.
//!
//! Requires GPU + nixl — benchmarks are skipped gracefully when unavailable.
//! Run with: cargo bench --bench nixl_staging -p doris-rpc

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use doris_rpc::cuda_driver::{cuda_alloc, cuda_free};
use doris_rpc::gpu_staging_buffer::GpuStagingBuffer;
use doris_rpc::nixl_exchange::NixlExchange;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Try to create a NixlExchange with staging buffer. Returns None if no GPU/nixl.
fn try_setup(name: &str, staging_mb: usize) -> Option<Arc<NixlExchange>> {
    NixlExchange::try_new_with_staging(name, Some(staging_mb * 1024 * 1024)).map(Arc::new)
}

/// Allocate real GPU memory for use as source buffers.
fn alloc_gpu_buffers(sizes: &[usize]) -> Option<Vec<(usize, usize)>> {
    let mut bufs = Vec::with_capacity(sizes.len());
    for &sz in sizes {
        match cuda_alloc(sz) {
            Ok(addr) => bufs.push((addr, sz)),
            Err(_) => {
                // Clean up already allocated
                for &(a, _) in &bufs {
                    let _ = cuda_free(a);
                }
                return None;
            }
        }
    }
    Some(bufs)
}

fn free_gpu_buffers(bufs: &[(usize, usize)]) {
    for &(addr, _) in bufs {
        let _ = cuda_free(addr);
    }
}

// ---------------------------------------------------------------------------
// Staging buffer setup (allocation + nixl registration)
// ---------------------------------------------------------------------------

fn bench_staging_setup(c: &mut Criterion) {
    // Check if nixl is available at all.
    let Some(probe) = NixlExchange::try_new("bench-probe") else {
        eprintln!("nixl not available, skipping staging setup benchmarks");
        return;
    };

    let mut group = c.benchmark_group("staging_setup");
    group.sample_size(10); // GPU alloc is slow, use fewer samples

    for &size_mb in &[1, 16, 64, 256] {
        let size_bytes = size_mb * 1024 * 1024;
        group.throughput(Throughput::Bytes(size_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("new", format!("{}MB", size_mb)),
            &size_bytes,
            |b, &size| {
                b.iter(|| {
                    let staging = GpuStagingBuffer::new(size, 0, probe.agent(), probe.backend())
                        .expect("staging alloc failed");
                    drop(staging); // free on each iteration
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Staging try_allocate (bump allocation, no D2D copy)
// ---------------------------------------------------------------------------

fn bench_staging_allocate(c: &mut Criterion) {
    let Some(exchange) = try_setup("bench-alloc", 512) else {
        eprintln!("nixl not available, skipping staging allocate benchmarks");
        return;
    };
    let staging = exchange.staging().expect("staging buffer should exist");

    let mut group = c.benchmark_group("staging_allocate");

    // Vary number of sub-allocations per batch
    for &num_bufs in &[1, 5, 10, 20, 50] {
        let buf_size = 1024 * 1024; // 1 MB each
        let sizes: Vec<(u64, u64)> = (0..num_bufs).map(|_| (buf_size as u64, 0u64)).collect();
        let total_bytes = num_bufs as u64 * buf_size as u64;

        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(
            BenchmarkId::new("bump_alloc", format!("{}x1MB", num_bufs)),
            &sizes,
            |b, sizes| {
                b.iter(|| {
                    let leases = staging.try_allocate(sizes).expect("allocate failed");
                    drop(leases); // releases bumps, resets allocator
                });
            },
        );
    }

    // Vary sub-allocation sizes
    for &buf_kb in &[4, 64, 1024, 8192] {
        let buf_size = buf_kb * 1024;
        let sizes: Vec<(u64, u64)> = (0..10).map(|_| (buf_size as u64, 0u64)).collect();
        let total_bytes = 10 * buf_size as u64;

        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(
            BenchmarkId::new("bump_alloc_sized", format!("10x{}KB", buf_kb)),
            &sizes,
            |b, sizes| {
                b.iter(|| {
                    let leases = staging.try_allocate(sizes).expect("allocate failed");
                    drop(leases);
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Staging try_stage (bump allocation + D2D copy)
// ---------------------------------------------------------------------------

fn bench_staging_stage(c: &mut Criterion) {
    let Some(exchange) = try_setup("bench-stage", 512) else {
        eprintln!("nixl not available, skipping staging stage benchmarks");
        return;
    };
    let staging = exchange.staging().expect("staging buffer should exist");

    // Allocate source GPU buffers to copy from.
    let src_sizes = vec![1024 * 1024; 10]; // 10 x 1MB
    let Some(src_bufs) = alloc_gpu_buffers(&src_sizes) else {
        eprintln!("GPU alloc failed, skipping staging stage benchmarks");
        return;
    };

    let mut group = c.benchmark_group("staging_stage");

    // Vary number of buffers staged per call
    for &count in &[1, 5, 10] {
        let stage_bufs: Vec<(usize, usize, u64)> = src_bufs[..count]
            .iter()
            .map(|&(addr, len)| (addr, len, 0u64))
            .collect();
        let total_bytes = count as u64 * 1024 * 1024;

        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(
            BenchmarkId::new("d2d_copy", format!("{}x1MB", count)),
            &stage_bufs,
            |b, bufs| {
                b.iter(|| {
                    let leases = staging.try_stage(bufs).expect("stage failed");
                    drop(leases);
                });
            },
        );
    }

    group.finish();
    free_gpu_buffers(&src_bufs);
}

// ---------------------------------------------------------------------------
// Staging try_stage vs per-buffer allocate_and_register
// ---------------------------------------------------------------------------

fn bench_staging_vs_per_buffer(c: &mut Criterion) {
    let Some(exchange) = try_setup("bench-vs-perf", 512) else {
        eprintln!("nixl not available, skipping staging vs per-buffer benchmarks");
        return;
    };
    let staging = exchange.staging().expect("staging buffer should exist");

    // Source GPU buffers
    let src_sizes = vec![1024 * 1024; 10]; // 10 x 1MB
    let Some(src_bufs) = alloc_gpu_buffers(&src_sizes) else {
        eprintln!("GPU alloc failed, skipping staging vs per-buffer benchmarks");
        return;
    };

    let mut group = c.benchmark_group("staging_vs_per_buffer");
    group.sample_size(10); // per-buffer alloc+register is slow

    for &count in &[1, 5, 10] {
        let total_bytes = count as u64 * 1024 * 1024;
        group.throughput(Throughput::Bytes(total_bytes));

        // Staging buffer path: bump alloc + D2D copy (no nixl registration per transfer)
        let stage_bufs: Vec<(usize, usize, u64)> = src_bufs[..count]
            .iter()
            .map(|&(addr, len)| (addr, len, 0u64))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("staging", format!("{}x1MB", count)),
            &stage_bufs,
            |b, bufs| {
                b.iter(|| {
                    let leases = staging.try_stage(bufs).expect("stage failed");
                    drop(leases);
                });
            },
        );

        // Per-buffer path: cuMemAlloc + nixl register per buffer (old path)
        let alloc_sizes: Vec<(u64, u64)> = (0..count).map(|_| (1024 * 1024u64, 0u64)).collect();

        group.bench_with_input(
            BenchmarkId::new("per_buffer_alloc_register", format!("{}x1MB", count)),
            &alloc_sizes,
            |b, sizes| {
                b.iter(|| {
                    let bufs = exchange
                        .allocate_and_register_gpu_buffers(sizes)
                        .expect("alloc+register failed");
                    drop(bufs); // deregister + cuMemFree on each iteration
                });
            },
        );
    }

    group.finish();
    free_gpu_buffers(&src_bufs);
}

// ---------------------------------------------------------------------------
// Metadata retrieval: cached vs fresh
// ---------------------------------------------------------------------------

fn bench_metadata(c: &mut Criterion) {
    // With staging: metadata is cached at startup
    let Some(exchange_staged) = try_setup("bench-md-staged", 16) else {
        eprintln!("nixl not available, skipping metadata benchmarks");
        return;
    };

    // Without staging: metadata is fetched fresh each time
    let Some(exchange_fresh) = NixlExchange::try_new("bench-md-fresh").map(Arc::new) else {
        return;
    };

    let mut group = c.benchmark_group("metadata");

    group.bench_function("cached", |b| {
        b.iter(|| {
            let md = exchange_staged.get_metadata().expect("get_metadata failed");
            assert!(!md.is_empty());
        });
    });

    group.bench_function("fresh", |b| {
        b.iter(|| {
            let md = exchange_fresh.get_metadata().expect("get_metadata failed");
            assert!(!md.is_empty());
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Lease lifecycle: alloc → use → drop
// ---------------------------------------------------------------------------

fn bench_lease_lifecycle(c: &mut Criterion) {
    let Some(exchange) = try_setup("bench-lease", 512) else {
        eprintln!("nixl not available, skipping lease lifecycle benchmarks");
        return;
    };
    let staging = exchange.staging().expect("staging buffer should exist");

    let mut group = c.benchmark_group("lease_lifecycle");

    // Measure the overhead of lease creation + drop (RAII Arc + mutex release)
    for &count in &[1, 10, 50, 100] {
        let sizes: Vec<(u64, u64)> = (0..count).map(|_| (4096u64, 0u64)).collect();
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::new("alloc_drop", count), &sizes, |b, sizes| {
            b.iter(|| {
                let leases = staging.try_allocate(sizes).expect("allocate failed");
                // Simulate reading lease addresses (prevents optimization)
                let _sum: usize = leases.iter().map(|l| l.addr()).sum();
                drop(leases);
            });
        });
    }

    // Measure interleaved lease patterns (allocate first batch, then second, drop first)
    group.bench_function("interleaved_10", |b| {
        let sizes_a: Vec<(u64, u64)> = (0..5).map(|_| (4096u64, 0u64)).collect();
        let sizes_b: Vec<(u64, u64)> = (0..5).map(|_| (4096u64, 0u64)).collect();

        b.iter(|| {
            let leases_a = staging.try_allocate(&sizes_a).expect("alloc A failed");
            let leases_b = staging.try_allocate(&sizes_b).expect("alloc B failed");
            let _sum: usize = leases_a
                .iter()
                .chain(leases_b.iter())
                .map(|l| l.addr())
                .sum();
            drop(leases_a); // drop first batch (leases remain from B)
            drop(leases_b); // drop second → allocator resets
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// GPU buffer registration throughput (register + deregister cycle)
// ---------------------------------------------------------------------------

fn bench_register_gpu_buffers(c: &mut Criterion) {
    let Some(exchange) = NixlExchange::try_new("bench-register").map(Arc::new) else {
        eprintln!("nixl not available, skipping register benchmarks");
        return;
    };

    // Allocate source GPU buffers.
    let src_sizes = vec![1024 * 1024; 10]; // 10 x 1MB
    let Some(src_bufs) = alloc_gpu_buffers(&src_sizes) else {
        eprintln!("GPU alloc failed, skipping register benchmarks");
        return;
    };

    let mut group = c.benchmark_group("register_gpu_buffers");
    group.sample_size(10); // registration is slow

    for &count in &[1, 5, 10] {
        let bufs: Vec<(usize, usize, u64)> = src_bufs[..count]
            .iter()
            .map(|&(addr, len)| (addr, len, 0u64))
            .collect();
        let total_bytes = count as u64 * 1024 * 1024;

        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(
            BenchmarkId::new("register_deregister", format!("{}x1MB", count)),
            &bufs,
            |b, bufs| {
                b.iter(|| {
                    let registered = exchange
                        .register_gpu_buffers(bufs)
                        .expect("register failed");
                    drop(registered); // deregister on drop
                });
            },
        );
    }

    group.finish();
    free_gpu_buffers(&src_bufs);
}

criterion_group!(
    benches,
    bench_staging_setup,
    bench_staging_allocate,
    bench_staging_stage,
    bench_staging_vs_per_buffer,
    bench_metadata,
    bench_lease_lifecycle,
    bench_register_gpu_buffers,
);
criterion_main!(benches);
