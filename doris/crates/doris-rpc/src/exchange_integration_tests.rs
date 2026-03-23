//! Integration tests for the exchange table flow.
//!
//! Tests the full roundtrip without a running cluster:
//!   Arrow RecordBatch → IPC → PBlock encode → ExchangeBuffer → PBlock decode
//!   → Arrow IPC → verify data integrity
//!
//! Also tests hash-partitioned exchange: split by hash → encode per-destination
//! → decode per-destination → verify partitioning correctness + data integrity.

use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use doris_thrift::types::TPrimitiveType;

use crate::arrow_to_pblock::arrow_ipc_to_pblock;
use crate::exchange_buffer::{ExchangeBuffer, ExchangeKey};
use crate::hash_partitioner::{compute_dest_assignments, split_by_destination};
use crate::pblock_decoder::{decode_pblocks, decoded_columns_to_arrow_ipc};

fn make_ipc(batch: &RecordBatch) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &batch.schema()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
    buf
}

fn read_ipc(ipc_bytes: &[u8]) -> Vec<RecordBatch> {
    let reader = StreamReader::try_new(std::io::Cursor::new(ipc_bytes), None).unwrap();
    reader.into_iter().collect::<Result<Vec<_>, _>>().unwrap()
}

/// Build a multi-type RecordBatch simulating a GROUP BY result.
///
/// Schema: (group_key INT32, count_val INT64, sum_val DOUBLE, label VARCHAR)
fn make_exchange_batch(n: usize) -> RecordBatch {
    let keys: Vec<i32> = (0..n).map(|i| (i % 10) as i32).collect();
    let counts: Vec<i64> = (0..n).map(|i| (i * 2 + 1) as i64).collect();
    let sums: Vec<f64> = (0..n).map(|i| i as f64 * 1.5).collect();
    let labels: Vec<String> = (0..n).map(|i| format!("group_{}", i % 10)).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("group_key", DataType::Int32, false),
        Field::new("count_val", DataType::Int64, false),
        Field::new("sum_val", DataType::Float64, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Int64Array::from(counts)),
            Arc::new(Float64Array::from(sums)),
            Arc::new(StringArray::from(labels.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        ],
    )
    .unwrap()
}

/// Build a batch with nullable columns.
fn make_nullable_batch(n: usize) -> RecordBatch {
    let keys: Vec<Option<i32>> = (0..n).map(|i| if i % 7 == 0 { None } else { Some(i as i32) }).collect();
    let vals: Vec<Option<&str>> = (0..n).map(|i| if i % 5 == 0 { None } else { Some("hello") }).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(StringArray::from(vals)),
        ],
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Full exchange roundtrip: Arrow → PBlock → decode → Arrow IPC
// ---------------------------------------------------------------------------

#[test]
fn test_exchange_roundtrip_multi_type() {
    let batch = make_exchange_batch(200);
    let ipc = make_ipc(&batch);

    // Encode to PBlock (simulates exchange sender).
    let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();
    assert_eq!(num_rows, 200);
    assert_eq!(pblock.column_metas.len(), 4);

    // Decode PBlock (simulates exchange receiver).
    let decoded = decode_pblocks(&[pblock]).unwrap();
    assert_eq!(decoded.num_rows, 200);
    assert_eq!(decoded.columns.len(), 4);

    // Convert back to Arrow IPC.
    let keep_indices: Vec<usize> = (0..4).collect();
    let result_ipc = decoded_columns_to_arrow_ipc(&decoded, &keep_indices).unwrap();
    let result_batches = read_ipc(&result_ipc);
    assert_eq!(result_batches.len(), 1);
    let result = &result_batches[0];
    assert_eq!(result.num_rows(), 200);
    assert_eq!(result.num_columns(), 4);

    // Verify data integrity: INT32 keys.
    let keys = result.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    for i in 0..200 {
        assert_eq!(keys.value(i), (i % 10) as i32, "key mismatch at row {i}");
    }

    // Verify: INT64 counts.
    let counts = result.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
    for i in 0..200 {
        assert_eq!(counts.value(i), (i * 2 + 1) as i64, "count mismatch at row {i}");
    }

    // Verify: DOUBLE sums.
    let sums = result.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    for i in 0..200 {
        assert!((sums.value(i) - i as f64 * 1.5).abs() < 1e-10, "sum mismatch at row {i}");
    }

    // Verify: VARCHAR labels.
    let labels = result.column(3).as_any().downcast_ref::<StringArray>().unwrap();
    for i in 0..200 {
        assert_eq!(labels.value(i), format!("group_{}", i % 10), "label mismatch at row {i}");
    }
}

#[test]
fn test_exchange_roundtrip_nullable() {
    let batch = make_nullable_batch(150);
    let ipc = make_ipc(&batch);

    let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();
    assert_eq!(num_rows, 150);

    let decoded = decode_pblocks(&[pblock]).unwrap();
    let keep_indices: Vec<usize> = (0..2).collect();
    let result_ipc = decoded_columns_to_arrow_ipc(&decoded, &keep_indices).unwrap();
    let result_batches = read_ipc(&result_ipc);
    let result = &result_batches[0];
    assert_eq!(result.num_rows(), 150);

    let ids = result.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    for i in 0..150 {
        if i % 7 == 0 {
            assert!(ids.is_null(i), "expected null at row {i}");
        } else {
            assert_eq!(ids.value(i), i as i32, "id mismatch at row {i}");
        }
    }

    let names = result.column(1).as_any().downcast_ref::<StringArray>().unwrap();
    for i in 0..150 {
        if i % 5 == 0 {
            assert!(names.is_null(i), "expected null at row {i}");
        } else {
            assert_eq!(names.value(i), "hello", "name mismatch at row {i}");
        }
    }
}

#[test]
fn test_exchange_roundtrip_multiple_blocks() {
    // Simulates multiple transmit_block calls (multi-sender or chunked).
    let batch1 = make_exchange_batch(50);
    let batch2 = make_exchange_batch(75);

    let (pblock1, _) = arrow_ipc_to_pblock(&make_ipc(&batch1)).unwrap();
    let (pblock2, _) = arrow_ipc_to_pblock(&make_ipc(&batch2)).unwrap();

    // Decode concatenated blocks (what the exchange receiver does).
    let decoded = decode_pblocks(&[pblock1, pblock2]).unwrap();
    assert_eq!(decoded.num_rows, 125);

    let keep_indices: Vec<usize> = (0..4).collect();
    let result_ipc = decoded_columns_to_arrow_ipc(&decoded, &keep_indices).unwrap();
    let result_batches = read_ipc(&result_ipc);
    assert_eq!(result_batches[0].num_rows(), 125);
}

// ---------------------------------------------------------------------------
// Exchange buffer integration
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_exchange_buffer_full_flow() {
    let buf = ExchangeBuffer::new();
    let key = ExchangeKey {
        query_id: (42, 99),
        node_id: 5,
    };
    let notify = buf.register(key.clone(), 2);

    // Sender 1: encode and transmit.
    let batch1 = make_exchange_batch(30);
    let (pblock1, _) = arrow_ipc_to_pblock(&make_ipc(&batch1)).unwrap();
    buf.add_block(&key, 0, Some(pblock1), false);
    buf.add_block(&key, 0, None, true); // EOS

    // Sender 2: encode and transmit.
    let batch2 = make_exchange_batch(20);
    let (pblock2, _) = arrow_ipc_to_pblock(&make_ipc(&batch2)).unwrap();
    buf.add_block(&key, 1, Some(pblock2), false);
    let all_done = buf.add_block(&key, 1, None, true); // EOS
    assert!(all_done);

    // Notify should resolve immediately.
    tokio::time::timeout(std::time::Duration::from_millis(100), notify.notified())
        .await
        .expect("notify should resolve");

    // Take and decode.
    let blocks = buf.take(&key);
    assert_eq!(blocks.len(), 2);

    let decoded = decode_pblocks(&blocks).unwrap();
    assert_eq!(decoded.num_rows, 50);

    let keep_indices: Vec<usize> = (0..4).collect();
    let result_ipc = decoded_columns_to_arrow_ipc(&decoded, &keep_indices).unwrap();
    let result_batches = read_ipc(&result_ipc);
    assert_eq!(result_batches[0].num_rows(), 50);
}

// ---------------------------------------------------------------------------
// Hash-partitioned exchange roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_hash_partition_roundtrip() {
    let batch = make_exchange_batch(200);
    let num_dests = 3;

    // Hash-partition by group_key (column 0).
    let assignments = compute_dest_assignments(
        &batch,
        &[0],                       // partition by first column
        num_dests,
        false,                      // CRC32 (not CRC32C)
        &[TPrimitiveType::INT],     // Doris type
    );
    assert_eq!(assignments.len(), 200);

    // Split into per-destination batches.
    let parts = split_by_destination(&batch, &assignments, num_dests).unwrap();
    assert_eq!(parts.len(), num_dests);

    // Each destination should get some rows (200 rows / 3 dests).
    let total_rows: usize = parts.iter().filter_map(|p| p.as_ref()).map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 200, "no rows should be lost in partitioning");

    // Encode each partition to PBlock, decode back, and verify.
    let mut all_keys_after = Vec::new();
    for (dest_idx, part) in parts.iter().enumerate() {
        let part = match part {
            Some(b) => b,
            None => continue,
        };

        // Encode → PBlock.
        let ipc = make_ipc(part);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();
        assert_eq!(num_rows, part.num_rows() as u32);

        // Decode PBlock → Arrow IPC.
        let decoded = decode_pblocks(&[pblock]).unwrap();
        assert_eq!(decoded.num_rows, part.num_rows() as u32);

        let keep_indices: Vec<usize> = (0..4).collect();
        let result_ipc = decoded_columns_to_arrow_ipc(&decoded, &keep_indices).unwrap();
        let result_batches = read_ipc(&result_ipc);
        let result = &result_batches[0];
        assert_eq!(result.num_rows(), part.num_rows());

        // Verify all rows in this partition hash to this destination.
        let keys = result.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let re_assignments = compute_dest_assignments(
            result,
            &[0],
            num_dests,
            false,
            &[TPrimitiveType::INT],
        );
        for (row, &dest) in re_assignments.iter().enumerate() {
            assert_eq!(
                dest, dest_idx as u32,
                "row {row} (key={}) should hash to dest {dest_idx}, got {dest}",
                keys.value(row)
            );
        }

        for i in 0..result.num_rows() {
            all_keys_after.push(keys.value(i));
        }
    }

    // Verify no data loss: all original keys should be present.
    all_keys_after.sort();
    let mut original_keys: Vec<i32> = (0..200).map(|i| (i % 10) as i32).collect();
    original_keys.sort();
    assert_eq!(all_keys_after, original_keys, "partition roundtrip lost or duplicated rows");
}

#[test]
fn test_hash_partition_multi_column() {
    // Hash on two columns: (group_key, count_val).
    let batch = make_exchange_batch(100);
    let num_dests = 4;

    let assignments = compute_dest_assignments(
        &batch,
        &[0, 1],                                       // partition by columns 0 and 1
        num_dests,
        true,                                          // CRC32C
        &[TPrimitiveType::INT, TPrimitiveType::BIGINT],
    );
    assert_eq!(assignments.len(), 100);
    assert!(assignments.iter().all(|&d| d < num_dests as u32));

    let parts = split_by_destination(&batch, &assignments, num_dests).unwrap();
    let total_rows: usize = parts.iter().filter_map(|p| p.as_ref()).map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 100);
}

#[test]
fn test_hash_partition_nullable_key() {
    let batch = make_nullable_batch(100);
    let num_dests = 3;

    // Hash on nullable id column — NULLs should all go to the same destination.
    let assignments = compute_dest_assignments(
        &batch,
        &[0],
        num_dests,
        false,
        &[TPrimitiveType::INT],
    );
    assert_eq!(assignments.len(), 100);

    // All NULL rows (every 7th) should hash to the same destination.
    let null_dests: Vec<u32> = (0..100)
        .filter(|i| i % 7 == 0)
        .map(|i| assignments[i])
        .collect();
    let first_null_dest = null_dests[0];
    assert!(
        null_dests.iter().all(|&d| d == first_null_dest),
        "all NULL rows should hash to the same destination"
    );

    // Roundtrip through PBlock.
    let parts = split_by_destination(&batch, &assignments, num_dests).unwrap();
    for part in parts.iter().flatten() {
        let ipc = make_ipc(part);
        let (pblock, _) = arrow_ipc_to_pblock(&ipc).unwrap();
        let decoded = decode_pblocks(&[pblock]).unwrap();
        assert_eq!(decoded.num_rows, part.num_rows() as u32);
    }
}

// ---------------------------------------------------------------------------
// Cancel during exchange
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_exchange_cancel_prevents_execution() {
    let buf = ExchangeBuffer::new();
    let key = ExchangeKey {
        query_id: (100, 200),
        node_id: 1,
    };
    let notify = buf.register(key.clone(), 1);

    // Cancel before any data arrives.
    buf.cancel_query(100, 200);
    assert!(buf.is_cancelled(100, 200));

    // Notify should fire (cancel calls notify_one).
    tokio::time::timeout(std::time::Duration::from_millis(100), notify.notified())
        .await
        .expect("notify should fire on cancel");

    // take() returns empty — and is_cancelled is true.
    let blocks = buf.take(&key);
    assert!(blocks.is_empty());
    assert!(buf.is_cancelled(100, 200));

    // Clean up.
    buf.clear_cancelled(100, 200);
    assert!(!buf.is_cancelled(100, 200));
}

#[tokio::test]
async fn test_exchange_cancel_after_partial_data() {
    let buf = ExchangeBuffer::new();
    let key = ExchangeKey {
        query_id: (300, 400),
        node_id: 2,
    };
    let notify = buf.register(key.clone(), 2); // expect 2 senders

    // Sender 1 sends data + EOS.
    let batch = make_exchange_batch(10);
    let (pblock, _) = arrow_ipc_to_pblock(&make_ipc(&batch)).unwrap();
    buf.add_block(&key, 0, Some(pblock), false);
    buf.add_block(&key, 0, None, true);

    // Cancel before sender 2 sends anything.
    buf.cancel_query(300, 400);

    // Notify should fire.
    tokio::time::timeout(std::time::Duration::from_millis(100), notify.notified())
        .await
        .expect("notify should fire on cancel");

    // Entry removed by cancel, take() returns empty.
    assert!(buf.take(&key).is_empty());
    assert!(buf.is_cancelled(300, 400));
}
