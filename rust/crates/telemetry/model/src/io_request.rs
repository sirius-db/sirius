//! Scan-split I/O telemetry: one IoRequest per fresh-read split
//! materialization (the read + decode step of `GPU_SCAN`, i.e.
//! `gpu_ingestible::materialize_metadata_to_table`). The Issued→Completed span
//! is the split's full materialize wall time — a sub-span of the owning task's
//! `GPU_SCAN` Computing state — and `read_time_ns` is the storage-read portion
//! measured at the datasource, so read and decode are separable.

use quent_model::{fsm, state};
use uuid::Uuid;

state! {
    // task_uuid/pipeline_uuid: the task/pipeline whose GPU_SCAN Computing span
    // contains this materialization (task_uuid nil outside task execution,
    // e.g. table pinning).
    // file_count: files (parquet row-group slices, or 1 duckdb .db file) read.
    // estimated_compressed_bytes: on-disk bytes the split expects to read, from
    // scan metadata; 0 when the format tracks no compressed estimate.
    // estimated_decoded_bytes: the split's decoded-output estimate.
    Issued {
        attributes: {
            task_uuid: Uuid,
            pipeline_uuid: Uuid,
            file_count: u64,
            estimated_compressed_bytes: u64,
            estimated_decoded_bytes: u64,
        },
    }
}

state! {
    // bytes_read/read_time_ns/read_calls: measured at the split's
    // sirius_datasource(s) — cache hits count at cache-copy speed, and
    // read_time_ns is a sum of per-call spans (async reads may overlap, so it
    // can exceed the critical-path read time). All 0 when the split read
    // through a plain (non-sirius) datasource.
    // rows: rows in the materialized table (post reader-side filter pushdown).
    Completed {
        attributes: {
            bytes_read: u64,
            read_time_ns: u64,
            read_calls: u64,
            rows: u64,
        },
    }
}

fsm! {
    IoRequest {
        states: {
            issued: Issued,
            completed: Completed,
        },
        entry: issued,
        exit_from: { completed },
        transitions: {
            issued => completed,
        },
    }
}
