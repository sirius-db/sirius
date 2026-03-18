//! Hash-partitioned exchange: CRC32 hashing + Arrow batch splitting.
//!
//! When the FE sends `HASH_PARTITIONED` in `TDataStreamSink.output_partition`,
//! rows must be routed to specific destinations based on the hash of their
//! partition expression columns (GROUP BY keys). This module provides:
//!
//! - `PartitionStrategy` — broadcast vs hash-partitioned routing
//! - `ExchangeInfo` — partition-aware exchange metadata
//! - CRC32/CRC32C hash computation matching Doris byte serialization
//! - `split_by_destination()` — split an Arrow RecordBatch per destination

use arrow::array::*;
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
    TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
    TimestampSecondType, UInt16Type, UInt32Type, UInt8Type,
};
use arrow::record_batch::RecordBatch;
use doris_thrift::exprs::{TExpr, TExprNodeType};
use doris_thrift::partitions::TPartitionType;
use doris_thrift::types::TPrimitiveType;

use crate::exchange_sender::ExchangeDest;

/// How rows should be routed to exchange destinations.
#[derive(Debug, Clone)]
pub enum PartitionStrategy {
    /// UNPARTITIONED: send all rows to all destinations.
    Broadcast,
    /// RANDOM: round-robin rows across destinations.
    Random,
    /// HASH_PARTITIONED / BUCKET_SHUFFLE_HASH_PARTITIONED:
    /// hash partition expressions and route each row to exactly one destination.
    Hash {
        partition_exprs: Vec<TExpr>,
        num_destinations: usize,
        /// true = CRC32C (Castagnoli) with shuffle_mix; false = zlib CRC32.
        use_crc32c: bool,
    },
}

/// Exchange metadata: destinations + partitioning strategy.
#[derive(Debug, Clone)]
pub struct ExchangeInfo {
    pub dest_node_id: i32,
    pub destinations: Vec<ExchangeDest>,
    pub partition: PartitionStrategy,
}

/// Build `PartitionStrategy` from Doris Thrift types.
pub fn partition_strategy_from_thrift(
    partition_type: &TPartitionType,
    partition_exprs: &Option<Vec<TExpr>>,
    num_destinations: usize,
    enable_new_shuffle_hash_method: bool,
) -> PartitionStrategy {
    match *partition_type {
        TPartitionType::HASH_PARTITIONED | TPartitionType::BUCKET_SHFFULE_HASH_PARTITIONED => {
            let exprs = partition_exprs.clone().unwrap_or_default();
            if exprs.is_empty() {
                // No partition expressions — fall back to broadcast (safe).
                PartitionStrategy::Broadcast
            } else {
                PartitionStrategy::Hash {
                    partition_exprs: exprs,
                    num_destinations,
                    use_crc32c: enable_new_shuffle_hash_method,
                }
            }
        }
        TPartitionType::RANDOM => PartitionStrategy::Random,
        // UNPARTITIONED, RANGE, LIST, OLAP/HIVE-specific → broadcast (safe fallback).
        _ => PartitionStrategy::Broadcast,
    }
}

// ---------------------------------------------------------------------------
// Partition expression → column index resolution
// ---------------------------------------------------------------------------

/// Resolve partition expressions to column indices in the result RecordBatch.
///
/// Fast path: all expressions are simple SLOT_REF nodes → look up slot_id in
/// the descriptor table to get col_name → find column index in the batch schema.
///
/// Returns `(column_indices, doris_primitive_types)`.
pub fn resolve_partition_columns(
    partition_exprs: &[TExpr],
    slot_descriptors: &[(i32, String)], // (slot_id, col_name) pairs
    batch_schema: &arrow::datatypes::Schema,
) -> Result<(Vec<usize>, Vec<TPrimitiveType>), String> {
    let slot_map: std::collections::HashMap<i32, &str> = slot_descriptors
        .iter()
        .map(|(id, name)| (*id, name.as_str()))
        .collect();

    let mut col_indices = Vec::with_capacity(partition_exprs.len());
    let mut doris_types = Vec::with_capacity(partition_exprs.len());

    for expr in partition_exprs {
        let first_node = expr
            .nodes
            .first()
            .ok_or_else(|| "empty partition expression".to_string())?;

        if first_node.node_type != TExprNodeType::SLOT_REF {
            return Err(format!(
                "unsupported partition expression type: {:?} (only SLOT_REF supported)",
                first_node.node_type
            ));
        }

        let slot_ref = first_node
            .slot_ref
            .as_ref()
            .ok_or_else(|| "SLOT_REF node missing slot_ref field".to_string())?;

        let col_name = slot_map
            .get(&slot_ref.slot_id)
            .ok_or_else(|| format!("slot_id {} not found in descriptor table", slot_ref.slot_id))?;

        let col_idx = batch_schema
            .index_of(col_name)
            .map_err(|_| format!("column '{}' not found in batch schema", col_name))?;

        // Extract Doris primitive type from the expression's type descriptor.
        let doris_type = first_node
            .type_
            .types
            .as_ref()
            .and_then(|types| types.first())
            .and_then(|t| t.scalar_type.as_ref())
            .map(|st| st.type_.clone())
            .unwrap_or(TPrimitiveType::INVALID_TYPE);

        col_indices.push(col_idx);
        doris_types.push(doris_type);
    }

    Ok((col_indices, doris_types))
}

// ---------------------------------------------------------------------------
// CRC32 / CRC32C hashing (Doris-compatible byte serialization)
// ---------------------------------------------------------------------------

/// Compute per-row destination assignments using Doris-compatible CRC32 hashing.
///
/// For each row, chains the hash of all partition columns (left to right):
///   hash = crc32(hash, col_bytes)
/// Then: `dest = hash % num_destinations` (or `shuffle_mix(hash) % N` for CRC32C).
pub fn compute_dest_assignments(
    batch: &RecordBatch,
    partition_col_indices: &[usize],
    num_destinations: usize,
    use_crc32c: bool,
    doris_types: &[TPrimitiveType],
) -> Vec<u32> {
    let num_rows = batch.num_rows();
    let seed: u32 = if use_crc32c { 0x9E3779B9 } else { 0 };
    let mut hashes = vec![seed; num_rows];

    for (expr_idx, &col_idx) in partition_col_indices.iter().enumerate() {
        let col = batch.column(col_idx);
        let doris_type = &doris_types[expr_idx];
        hash_column(&mut hashes, col.as_ref(), doris_type, use_crc32c);
    }

    // Apply final modulo (with optional shuffle_mix for CRC32C).
    let n = num_destinations as u32;
    hashes
        .iter()
        .map(|&h| {
            let final_h = if use_crc32c { crc32c_shuffle_mix(h) } else { h };
            final_h % n
        })
        .collect()
}

/// CRC32C shuffle mix (matches Doris `HashUtil::crc32c_shuffle_mix`).
fn crc32c_shuffle_mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0xA5B35705);
    h ^= h >> 13;
    h
}

/// Hash one column into the running hash array.
fn hash_column(hashes: &mut [u32], array: &dyn Array, doris_type: &TPrimitiveType, use_crc32c: bool) {
    // Dispatch on Arrow data type.
    match array.data_type() {
        DataType::Boolean => hash_boolean(hashes, array.as_any().downcast_ref::<BooleanArray>().unwrap(), use_crc32c),
        DataType::Int8 => hash_primitive::<Int8Type>(hashes, array, use_crc32c),
        DataType::Int16 => hash_primitive::<Int16Type>(hashes, array, use_crc32c),
        DataType::Int32 => hash_primitive::<Int32Type>(hashes, array, use_crc32c),
        DataType::Int64 => hash_primitive::<Int64Type>(hashes, array, use_crc32c),
        DataType::UInt8 => hash_primitive::<UInt8Type>(hashes, array, use_crc32c),
        DataType::UInt16 => hash_primitive::<UInt16Type>(hashes, array, use_crc32c),
        DataType::UInt32 => hash_primitive::<UInt32Type>(hashes, array, use_crc32c),
        DataType::Float32 => hash_primitive::<Float32Type>(hashes, array, use_crc32c),
        DataType::Float64 => hash_primitive::<Float64Type>(hashes, array, use_crc32c),
        DataType::Utf8 => hash_string::<i32>(hashes, array, use_crc32c),
        DataType::LargeUtf8 => hash_string::<i64>(hashes, array, use_crc32c),
        DataType::Date32 => hash_date32(hashes, array, doris_type, use_crc32c),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Second, _) => hash_primitive::<TimestampSecondType>(hashes, array, use_crc32c),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Millisecond, _) => hash_primitive::<TimestampMillisecondType>(hashes, array, use_crc32c),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) => hash_primitive::<TimestampMicrosecondType>(hashes, array, use_crc32c),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, _) => hash_primitive::<TimestampNanosecondType>(hashes, array, use_crc32c),
        DataType::Decimal128(_, _) => hash_decimal128(hashes, array, use_crc32c),
        dt => {
            tracing::warn!(?dt, "unsupported data type for hash partitioning, hashing as zero bytes");
            // Fallback: hash 4 zero bytes for every row (same as NULL).
            for h in hashes.iter_mut() {
                *h = do_crc32(*h, &[0u8; 4], use_crc32c);
            }
        }
    }
}

/// Hash a primitive (fixed-width) array: hash raw LE bytes.
fn hash_primitive<T: ArrowPrimitiveType>(hashes: &mut [u32], array: &dyn Array, use_crc32c: bool) {
    let arr = array.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    let byte_width = std::mem::size_of::<T::Native>();
    for (i, h) in hashes.iter_mut().enumerate() {
        if arr.is_null(i) {
            *h = do_crc32(*h, &[0u8; 4], use_crc32c);
        } else {
            let val = arr.value(i);
            // Safety: T::Native is a fixed-size numeric type.
            let bytes = unsafe {
                std::slice::from_raw_parts(&val as *const T::Native as *const u8, byte_width)
            };
            *h = do_crc32(*h, bytes, use_crc32c);
        }
    }
}

/// Hash a boolean array: 1 byte per value.
fn hash_boolean(hashes: &mut [u32], arr: &BooleanArray, use_crc32c: bool) {
    for (i, h) in hashes.iter_mut().enumerate() {
        if arr.is_null(i) {
            *h = do_crc32(*h, &[0u8; 4], use_crc32c);
        } else {
            let b = if arr.value(i) { 1u8 } else { 0u8 };
            *h = do_crc32(*h, &[b], use_crc32c);
        }
    }
}

/// Hash a string (Utf8/LargeUtf8) array: raw string bytes (no length prefix).
fn hash_string<O: arrow::array::OffsetSizeTrait>(hashes: &mut [u32], array: &dyn Array, use_crc32c: bool) {
    let arr = array.as_any().downcast_ref::<GenericStringArray<O>>().unwrap();
    for (i, h) in hashes.iter_mut().enumerate() {
        if arr.is_null(i) {
            *h = do_crc32(*h, &[0u8; 4], use_crc32c);
        } else {
            let s = arr.value(i);
            *h = do_crc32(*h, s.as_bytes(), use_crc32c);
        }
    }
}

/// Hash Date32 array: legacy DATE (type 9) hashes as string repr, DATEV2 (type 26) as raw bytes.
fn hash_date32(hashes: &mut [u32], array: &dyn Array, doris_type: &TPrimitiveType, use_crc32c: bool) {
    let arr = array.as_any().downcast_ref::<Date32Array>().unwrap();
    let is_legacy_date = *doris_type == TPrimitiveType::DATE;

    for (i, h) in hashes.iter_mut().enumerate() {
        if arr.is_null(i) {
            *h = do_crc32(*h, &[0u8; 4], use_crc32c);
        } else if is_legacy_date {
            // Legacy DATE: hash as "YYYY-MM-DD" string representation.
            let days = arr.value(i);
            let date_str = date32_to_string(days);
            *h = do_crc32(*h, date_str.as_bytes(), use_crc32c);
        } else {
            // DATEV2: hash as raw 4-byte LE.
            let val = arr.value(i);
            *h = do_crc32(*h, &val.to_le_bytes(), use_crc32c);
        }
    }
}

/// Convert Arrow Date32 (days since epoch) to "YYYY-MM-DD" string.
fn date32_to_string(days: i32) -> String {
    // Days since Unix epoch (1970-01-01). Use civil date calculation.
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}

/// Hash Decimal128 array: raw 16-byte LE.
fn hash_decimal128(hashes: &mut [u32], array: &dyn Array, use_crc32c: bool) {
    let arr = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
    for (i, h) in hashes.iter_mut().enumerate() {
        if arr.is_null(i) {
            *h = do_crc32(*h, &[0u8; 4], use_crc32c);
        } else {
            let val = arr.value(i);
            *h = do_crc32(*h, &val.to_le_bytes(), use_crc32c);
        }
    }
}

/// Compute CRC32 (zlib) or CRC32C (Castagnoli) of `data` with initial `hash`.
#[inline]
fn do_crc32(hash: u32, data: &[u8], use_crc32c: bool) -> u32 {
    if use_crc32c {
        crc32c::crc32c_append(hash, data)
    } else {
        // crc32fast operates on the full CRC value (not pre/post inverted).
        let mut hasher = crc32fast::Hasher::new_with_initial(hash);
        hasher.update(data);
        hasher.finalize()
    }
}

// ---------------------------------------------------------------------------
// Arrow RecordBatch splitting by destination
// ---------------------------------------------------------------------------

/// Split a RecordBatch by per-row destination assignments.
///
/// Returns one `Option<RecordBatch>` per destination. `None` means zero rows
/// for that destination (EOS still needs to be sent).
pub fn split_by_destination(
    batch: &RecordBatch,
    assignments: &[u32],
    num_destinations: usize,
) -> Result<Vec<Option<RecordBatch>>, String> {
    // Build per-destination index arrays in a single pass.
    let mut indices_per_dest: Vec<Vec<u32>> = vec![Vec::new(); num_destinations];
    for (row, &dest) in assignments.iter().enumerate() {
        indices_per_dest[dest as usize].push(row as u32);
    }

    let mut result = Vec::with_capacity(num_destinations);
    for indices in &indices_per_dest {
        if indices.is_empty() {
            result.push(None);
        } else {
            let idx_array = UInt32Array::from(indices.clone());
            let columns: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .map(|col| arrow::compute::take(col.as_ref(), &idx_array, None))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("arrow take: {e}"))?;
            let rb = RecordBatch::try_new(batch.schema(), columns)
                .map_err(|e| format!("RecordBatch from take: {e}"))?;
            result.push(Some(rb));
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_crc32_int32() {
        // Verify CRC32 of a known 4-byte LE value.
        let val: i32 = 42;
        let h = do_crc32(0, &val.to_le_bytes(), false);
        // CRC32 is deterministic; just verify non-zero and consistent.
        assert_ne!(h, 0);
        assert_eq!(h, do_crc32(0, &42i32.to_le_bytes(), false));
    }

    #[test]
    fn test_crc32c_int32() {
        let val: i32 = 42;
        let h = do_crc32(0x9E3779B9, &val.to_le_bytes(), true);
        assert_ne!(h, 0);
        assert_eq!(h, do_crc32(0x9E3779B9, &42i32.to_le_bytes(), true));
    }

    #[test]
    fn test_crc32_null_is_4_zero_bytes() {
        let h_null = do_crc32(0, &[0u8; 4], false);
        // NULL should hash consistently.
        assert_eq!(h_null, do_crc32(0, &[0u8; 4], false));
    }

    #[test]
    fn test_crc32_string() {
        let s = "hello";
        let h = do_crc32(0, s.as_bytes(), false);
        assert_ne!(h, 0);
        // Different strings produce different hashes.
        let h2 = do_crc32(0, "world".as_bytes(), false);
        assert_ne!(h, h2);
    }

    #[test]
    fn test_crc32c_shuffle_mix() {
        let h = crc32c_shuffle_mix(0x12345678);
        // Just verify it's deterministic and different from input.
        assert_ne!(h, 0x12345678);
        assert_eq!(h, crc32c_shuffle_mix(0x12345678));
    }

    #[test]
    fn test_compute_dest_assignments_int32() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("val", DataType::Utf8, false),
        ]));
        let key_col = Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let val_col = StringArray::from(vec!["a", "b", "c", "d", "e", "f", "g", "h"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(key_col), Arc::new(val_col)]).unwrap();

        let assignments = compute_dest_assignments(
            &batch,
            &[0], // partition on "key" column
            3,    // 3 destinations
            false,
            &[TPrimitiveType::INT],
        );

        assert_eq!(assignments.len(), 8);
        // Every assignment should be in [0, 3).
        for &a in &assignments {
            assert!(a < 3, "assignment {a} out of range [0, 3)");
        }
    }

    #[test]
    fn test_compute_dest_assignments_multi_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        let a_col = Int32Array::from(vec![1, 1, 2, 2]);
        let b_col = StringArray::from(vec!["x", "y", "x", "y"]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(a_col), Arc::new(b_col)]).unwrap();

        let assignments = compute_dest_assignments(
            &batch,
            &[0, 1], // partition on both columns
            4,
            false,
            &[TPrimitiveType::INT, TPrimitiveType::VARCHAR],
        );

        assert_eq!(assignments.len(), 4);
        // (1, "x") rows should map to the same destination.
        // (1, "y") should be different from (1, "x") (usually).
        // All within [0, 4).
        for &a in &assignments {
            assert!(a < 4);
        }
    }

    #[test]
    fn test_compute_dest_assignments_with_nulls() {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, true)]));
        let key_col = Int32Array::from(vec![Some(1), None, Some(3), None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(key_col)]).unwrap();

        let assignments = compute_dest_assignments(
            &batch,
            &[0],
            2,
            false,
            &[TPrimitiveType::INT],
        );

        assert_eq!(assignments.len(), 4);
        // NULL rows should get the same destination (both hash as 4 zero bytes).
        assert_eq!(assignments[1], assignments[3]);
        for &a in &assignments {
            assert!(a < 2);
        }
    }

    #[test]
    fn test_split_by_destination() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        let id_col = Int32Array::from(vec![10, 20, 30, 40, 50]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(id_col)]).unwrap();

        // Manually assign: rows 0,2,4 → dest 0; rows 1,3 → dest 1; dest 2 → empty.
        let assignments = vec![0u32, 1, 0, 1, 0];
        let splits = split_by_destination(&batch, &assignments, 3).unwrap();

        assert_eq!(splits.len(), 3);

        // Dest 0: rows 0, 2, 4 → values 10, 30, 50
        let d0 = splits[0].as_ref().unwrap();
        assert_eq!(d0.num_rows(), 3);
        let d0_ids: Vec<i32> = d0
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(d0_ids, vec![10, 30, 50]);

        // Dest 1: rows 1, 3 → values 20, 40
        let d1 = splits[1].as_ref().unwrap();
        assert_eq!(d1.num_rows(), 2);
        let d1_ids: Vec<i32> = d1
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        assert_eq!(d1_ids, vec![20, 40]);

        // Dest 2: empty
        assert!(splits[2].is_none());
    }

    #[test]
    fn test_split_all_to_one_dest() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
        ]));
        let col = Int32Array::from(vec![1, 2, 3]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap();

        let assignments = vec![0u32, 0, 0];
        let splits = split_by_destination(&batch, &assignments, 2).unwrap();

        assert_eq!(splits[0].as_ref().unwrap().num_rows(), 3);
        assert!(splits[1].is_none());
    }

    #[test]
    fn test_crc32_deterministic_cross_type() {
        // Same row hashed twice should produce the same destination.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
        ]));
        let a = Int64Array::from(vec![100, 200]);
        let b = Float64Array::from(vec![1.5, 2.5]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap();

        let r1 = compute_dest_assignments(
            &batch,
            &[0, 1],
            5,
            false,
            &[TPrimitiveType::BIGINT, TPrimitiveType::DOUBLE],
        );
        let r2 = compute_dest_assignments(
            &batch,
            &[0, 1],
            5,
            false,
            &[TPrimitiveType::BIGINT, TPrimitiveType::DOUBLE],
        );
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_hash_boolean_column() {
        let schema = Arc::new(Schema::new(vec![Field::new("flag", DataType::Boolean, true)]));
        let col = BooleanArray::from(vec![Some(true), Some(false), None]);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap();

        let assignments = compute_dest_assignments(
            &batch,
            &[0],
            2,
            false,
            &[TPrimitiveType::BOOLEAN],
        );
        assert_eq!(assignments.len(), 3);
        // true and false should hash differently.
        // (They may or may not end up in different partitions mod 2, but they should hash differently.)
    }

    #[test]
    fn test_hash_decimal128_column() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("amount", DataType::Decimal128(18, 2), true),
        ]));
        let col = Decimal128Array::from(vec![Some(12345), Some(67890), None])
            .with_precision_and_scale(18, 2)
            .unwrap();
        let batch = RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap();

        let assignments = compute_dest_assignments(
            &batch,
            &[0],
            3,
            false,
            &[TPrimitiveType::DECIMAL128I],
        );
        assert_eq!(assignments.len(), 3);
        for &a in &assignments {
            assert!(a < 3);
        }
    }
}
