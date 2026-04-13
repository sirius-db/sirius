//! Criterion benchmarks for nixl exchange data path throughput.
//!
//! Measures: PBlock encode/decode roundtrip, StreamVByte codec, bump allocator.

use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use doris_rpc::arrow_to_pblock::arrow_ipc_to_pblock;
use doris_rpc::pblock_decoder::{decode_pblocks, streamvbyte_decode};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_ipc(batch: &RecordBatch) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &batch.schema()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
    buf
}

/// Build an IPC payload with `n` rows of mixed types (INT64, FLOAT64, nullable INT32, STRING).
fn mixed_ipc(n: usize) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("nullable_col", DataType::Int32, true),
        Field::new("name", DataType::Utf8, false),
    ]));

    let ids: Vec<i64> = (0..n as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| i as f64 * 1.1).collect();
    let nullable: Vec<Option<i32>> = (0..n)
        .map(|i| if i % 7 == 0 { None } else { Some(i as i32) })
        .collect();
    let names: Vec<String> = (0..n).map(|i| format!("row_{:06}", i)).collect();

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Float64Array::from(values)),
            Arc::new(Int32Array::from(nullable)),
            Arc::new(StringArray::from(
                names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();

    make_ipc(&batch)
}

/// Build an IPC payload with `n` rows of INT64 only (fixed-width throughput).
fn int64_ipc(n: usize) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
    let values: Vec<i64> = (0..n as i64).collect();
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap();
    make_ipc(&batch)
}

/// Build an IPC payload with `n` rows of strings.
fn string_ipc(n: usize) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
    let strings: Vec<String> = (0..n).map(|i| format!("value_{:08}", i)).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(StringArray::from(
            strings.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    make_ipc(&batch)
}

// ---------------------------------------------------------------------------
// PBlock encode benchmarks
// ---------------------------------------------------------------------------

fn bench_pblock_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("pblock_encode");

    for &n in &[100, 1_000, 10_000, 100_000] {
        let ipc = mixed_ipc(n);
        let bytes = ipc.len() as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::new("mixed", n), &ipc, |b, ipc| {
            b.iter(|| arrow_ipc_to_pblock(ipc).unwrap());
        });
    }

    for &n in &[1_000, 10_000, 100_000] {
        let ipc = int64_ipc(n);
        let bytes = ipc.len() as u64;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::new("int64_only", n), &ipc, |b, ipc| {
            b.iter(|| arrow_ipc_to_pblock(ipc).unwrap());
        });
    }

    for &n in &[1_000, 10_000, 100_000] {
        let ipc = string_ipc(n);
        let bytes = ipc.len() as u64;
        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::new("string_only", n), &ipc, |b, ipc| {
            b.iter(|| arrow_ipc_to_pblock(ipc).unwrap());
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// PBlock decode benchmarks
// ---------------------------------------------------------------------------

fn bench_pblock_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("pblock_decode");

    for &n in &[100, 1_000, 10_000, 100_000] {
        let ipc = mixed_ipc(n);
        let (pblock, _) = arrow_ipc_to_pblock(&ipc).unwrap();
        let pblock_size = pblock.column_values.as_ref().map_or(0, |v| v.len() as u64);

        group.throughput(Throughput::Bytes(pblock_size));
        group.bench_with_input(BenchmarkId::new("mixed", n), &pblock, |b, pb| {
            b.iter(|| decode_pblocks(&[pb.clone()]).unwrap());
        });
    }

    for &n in &[1_000, 10_000, 100_000] {
        let ipc = int64_ipc(n);
        let (pblock, _) = arrow_ipc_to_pblock(&ipc).unwrap();
        let pblock_size = pblock.column_values.as_ref().map_or(0, |v| v.len() as u64);
        group.throughput(Throughput::Bytes(pblock_size));
        group.bench_with_input(BenchmarkId::new("int64_only", n), &pblock, |b, pb| {
            b.iter(|| decode_pblocks(&[pb.clone()]).unwrap());
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// PBlock roundtrip (encode → decode)
// ---------------------------------------------------------------------------

fn bench_pblock_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("pblock_roundtrip");

    for &n in &[1_000, 10_000, 100_000] {
        let ipc = mixed_ipc(n);
        let bytes = ipc.len() as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::new("mixed", n), &ipc, |b, ipc| {
            b.iter(|| {
                let (pblock, _) = arrow_ipc_to_pblock(ipc).unwrap();
                decode_pblocks(&[pblock]).unwrap()
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// StreamVByte codec
// ---------------------------------------------------------------------------

/// StreamVByte encode via the public arrow_to_pblock path (encodes u32 values
/// through the IPC → PBlock pipeline that triggers StreamVByte for data > 256 bytes).
fn bench_streamvbyte(c: &mut Criterion) {
    let mut group = c.benchmark_group("streamvbyte");

    // Decode benchmark: encode once, then benchmark decoding
    for &n in &[256, 1_000, 10_000, 100_000] {
        // Create encoded data by using the encode path indirectly:
        // Build a u32 sequence, encode it, then benchmark decode
        let values: Vec<u32> = (0..n).map(|i| (i * 17 + 42) as u32).collect();
        let encoded = streamvbyte_encode_for_bench(&values);
        let bytes = encoded.len() as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(
            BenchmarkId::new("decode", n),
            &(encoded, n),
            |b, (enc, count)| {
                b.iter(|| streamvbyte_decode(enc, *count).unwrap());
            },
        );
    }

    for &n in &[256, 1_000, 10_000, 100_000] {
        let values: Vec<u32> = (0..n).map(|i| (i * 17 + 42) as u32).collect();
        let bytes = (n * 4) as u64;

        group.throughput(Throughput::Bytes(bytes));
        group.bench_with_input(BenchmarkId::new("encode", n), &values, |b, vals| {
            b.iter(|| streamvbyte_encode_for_bench(vals));
        });
    }

    group.finish();
}

/// Standalone StreamVByte encoder (mirrors arrow_to_pblock::streamvbyte_encode).
fn streamvbyte_encode_for_bench(values: &[u32]) -> Vec<u8> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    let control_len = (n + 3) / 4;
    let mut control = vec![0u8; control_len];
    let mut data = Vec::with_capacity(n * 4);

    for (i, &val) in values.iter().enumerate() {
        let byte_len = match val {
            0..=0xFF => 1,
            0..=0xFFFF => 2,
            0..=0xFFFFFF => 3,
            _ => 4,
        };
        let code = (byte_len - 1) as u8;
        let ctrl_idx = i / 4;
        let shift = (i % 4) * 2;
        control[ctrl_idx] |= code << shift;
        data.extend_from_slice(&val.to_le_bytes()[..byte_len]);
    }

    let mut result = Vec::with_capacity(control_len + data.len());
    result.extend_from_slice(&control);
    result.extend_from_slice(&data);
    result
}

// ---------------------------------------------------------------------------
// Bump allocator (CPU-only, no GPU)
// ---------------------------------------------------------------------------

fn bench_bump_allocator(c: &mut Criterion) {
    use std::sync::Mutex;

    let mut group = c.benchmark_group("bump_allocator");

    // Simulate the BumpAllocator allocation pattern
    for &n in &[10, 100, 1_000] {
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("alloc_release_cycle", n),
            &n,
            |b, &count| {
                b.iter(|| {
                    let alloc = Mutex::new(BumpAllocatorSim::new(512 * 1024 * 1024));
                    let sizes: Vec<usize> = (0..count).map(|i| 1024 * (1 + i % 16)).collect();
                    // Allocate all
                    for &sz in &sizes {
                        alloc.lock().unwrap().alloc(sz);
                    }
                    // Release all
                    for _ in &sizes {
                        alloc.lock().unwrap().release();
                    }
                });
            },
        );
    }

    group.finish();
}

/// CPU-only simulation of the BumpAllocator (no CUDA dependency).
struct BumpAllocatorSim {
    offset: usize,
    capacity: usize,
    active_leases: usize,
}

impl BumpAllocatorSim {
    fn new(capacity: usize) -> Self {
        Self {
            offset: 0,
            capacity,
            active_leases: 0,
        }
    }

    fn alloc(&mut self, size: usize) -> Option<usize> {
        let aligned = (self.offset + 255) & !255;
        let end = aligned + size;
        if end > self.capacity {
            return None;
        }
        self.offset = end;
        self.active_leases += 1;
        Some(aligned)
    }

    fn release(&mut self) {
        self.active_leases = self.active_leases.saturating_sub(1);
        if self.active_leases == 0 {
            self.offset = 0;
        }
    }
}

criterion_group!(
    benches,
    bench_pblock_encode,
    bench_pblock_decode,
    bench_pblock_roundtrip,
    bench_streamvbyte,
    bench_bump_allocator,
);
criterion_main!(benches);
