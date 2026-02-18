//! Arrow IPC → PBlock encoder: converts Arrow record batches to Doris PBlock
//! column binary format for inter-BE exchange via `transmit_block`.
//!
//! This is the inverse of `pblock_decoder.rs`. The encoding follows the Doris
//! column_values wire format:
//!   - Per column: header (const_flag + row_num + real_saved_num) + data
//!   - Fixed-width: raw bytes (no StreamVByte, below SERIALIZED_MEM_SIZE_LIMIT)
//!   - STRING: raw offsets + value_len + raw chars (below limit)
//!   - NULLABLE: null_map + inner column data
//!
//! We use the uncompressed path (no StreamVByte/LZ4) since Doris accepts raw data
//! for small columns and all our exchange data goes through snappy/lz4 at the PBlock level.

use std::io::Cursor;

use arrow::ipc::reader::StreamReader;
use doris_proto::doris::{p_column_meta, PBlock, PColumnMeta};

/// Encode Arrow IPC bytes into a single PBlock for transmit_block.
///
/// Returns `(PBlock, num_rows)`. The PBlock is uncompressed; the caller may
/// optionally compress `column_values` before sending.
pub fn arrow_ipc_to_pblock(ipc_bytes: &[u8]) -> Result<(PBlock, u32), String> {
    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None)
        .map_err(|e| format!("parse IPC: {e}"))?;
    let schema = reader.schema();

    let batches: Vec<_> = reader
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("read IPC batches: {e}"))?;

    if batches.is_empty() {
        return Ok((
            PBlock {
                column_metas: vec![],
                column_values: None,
                compressed: Some(false),
                uncompressed_size: None,
                compression_type: None,
                be_exec_version: Some(3),
            },
            0,
        ));
    }

    // Compute total rows across all batches.
    let num_rows: u32 = batches.iter().map(|b| b.num_rows() as u32).sum();
    let num_cols = schema.fields().len();

    let mut column_metas = Vec::with_capacity(num_cols);
    let mut column_values = Vec::new();

    for col_idx in 0..num_cols {
        let field = schema.field(col_idx);
        let col_name = field.name().clone();
        let nullable = field.is_nullable();
        let (type_id, precision, scale) = arrow_type_to_doris_type_id(field.data_type());

        let decimal_param = if precision > 0 {
            Some(p_column_meta::Decimal {
                precision: Some(precision as u32),
                scale: Some(scale as u32),
            })
        } else {
            None
        };
        column_metas.push(PColumnMeta {
            name: Some(col_name),
            r#type: Some(type_id),
            is_nullable: Some(nullable),
            decimal_param,
            children: vec![],
            result_is_nullable: None,
            function_name: None,
            be_exec_version: None,
            column_path: None,
            variant_max_subcolumns_count: None,
        });

        // Collect column data across all batches.
        if nullable {
            encode_nullable_column(
                &batches,
                col_idx,
                num_rows,
                type_id,
                &mut column_values,
            )?;
        } else {
            encode_column(&batches, col_idx, num_rows, type_id, &mut column_values)?;
        }
    }

    let uncompressed_size = column_values.len() as i64;

    Ok((
        PBlock {
            column_metas,
            column_values: Some(column_values),
            compressed: Some(false),
            uncompressed_size: Some(uncompressed_size),
            compression_type: None,
            be_exec_version: Some(3),
        },
        num_rows,
    ))
}

/// Write the per-column header: const_flag(1) + row_num(8) + real_saved_num(8).
fn write_col_header(buf: &mut Vec<u8>, row_num: u64) {
    buf.push(0); // const_flag = false
    buf.extend_from_slice(&row_num.to_le_bytes());
    buf.extend_from_slice(&row_num.to_le_bytes()); // real_saved_num = row_num
}

/// Encode a non-nullable column across all batches.
fn encode_column(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
    num_rows: u32,
    type_id: i32,
    buf: &mut Vec<u8>,
) -> Result<(), String> {
    write_col_header(buf, num_rows as u64);

    if is_string_type_id(type_id) {
        // STRING encoding: offsets + chars
        encode_string_data(batches, col_idx, num_rows, buf)?;
    } else {
        // Fixed-width encoding
        let width = type_byte_width(type_id);
        if width == 0 {
            return Err(format!("unsupported type_id {} for encoding", type_id));
        }

        for batch in batches {
            let col = batch.column(col_idx);
            append_fixed_width_data(col, type_id, width, buf)?;
        }
    }

    Ok(())
}

/// Encode a nullable column: nullable header + null_map + inner column data.
fn encode_nullable_column(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
    num_rows: u32,
    type_id: i32,
    buf: &mut Vec<u8>,
) -> Result<(), String> {
    // Nullable outer header
    write_col_header(buf, num_rows as u64);

    // null_map: one byte per row (0 = not null, 1 = null)
    let mut null_map = Vec::with_capacity(num_rows as usize);
    for batch in batches {
        let col = batch.column(col_idx);
        let nulls = col.nulls();
        for row in 0..col.len() {
            let is_null = nulls.map_or(false, |n| n.is_null(row));
            null_map.push(if is_null { 1u8 } else { 0u8 });
        }
    }
    // Raw null_map (no StreamVByte since we keep data small or handle at PBlock level)
    buf.extend_from_slice(&null_map);

    // Inner column data (with its own header)
    encode_column(batches, col_idx, num_rows, type_id, buf)?;

    Ok(())
}

/// Encode string data: offsets array + value_len + char data.
fn encode_string_data(
    batches: &[arrow::record_batch::RecordBatch],
    col_idx: usize,
    num_rows: u32,
    buf: &mut Vec<u8>,
) -> Result<(), String> {
    use arrow::array::*;

    // Collect all strings, building Doris-style cumulative offsets (with \0 terminators).
    let mut all_chars = Vec::new();
    let mut offsets = Vec::with_capacity(num_rows as usize);
    let mut cumulative: u32 = 0;

    for batch in batches {
        let col = batch.column(col_idx);
        let str_array = col
            .as_any()
            .downcast_ref::<StringArray>()
            .or_else(|| None)
            .ok_or_else(|| {
                format!(
                    "expected StringArray for column {}, got {:?}",
                    col_idx,
                    col.data_type()
                )
            });

        if let Ok(arr) = str_array {
            for i in 0..arr.len() {
                if arr.is_null(i) {
                    // NULL string: write \0 terminator
                    all_chars.push(0u8);
                    cumulative += 1;
                } else {
                    let s = arr.value(i);
                    all_chars.extend_from_slice(s.as_bytes());
                    all_chars.push(0u8); // \0 terminator
                    cumulative += s.len() as u32 + 1;
                }
                offsets.push(cumulative);
            }
        } else {
            // Try LargeStringArray
            let large_arr = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    format!(
                        "expected StringArray or LargeStringArray for column {}, got {:?}",
                        col_idx,
                        col.data_type()
                    )
                })?;
            for i in 0..large_arr.len() {
                if large_arr.is_null(i) {
                    all_chars.push(0u8);
                    cumulative += 1;
                } else {
                    let s = large_arr.value(i);
                    all_chars.extend_from_slice(s.as_bytes());
                    all_chars.push(0u8);
                    cumulative += s.len() as u32 + 1;
                }
                offsets.push(cumulative);
            }
        }
    }

    // Write offsets as raw UInt32 LE array
    let offsets_bytes: Vec<u8> = offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
    buf.extend_from_slice(&offsets_bytes);

    // Write value_len (8 bytes) + char data
    let value_len = all_chars.len() as u64;
    buf.extend_from_slice(&value_len.to_le_bytes());
    buf.extend_from_slice(&all_chars);

    Ok(())
}

/// Append fixed-width column data from an Arrow array to the buffer.
fn append_fixed_width_data(
    col: &dyn arrow::array::Array,
    _type_id: i32,
    _width: usize,
    buf: &mut Vec<u8>,
) -> Result<(), String> {
    use arrow::array::*;
    use arrow::datatypes::{
        DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
        UInt32Type, UInt64Type, UInt8Type,
    };

    // Use the raw buffer data when possible for efficiency.
    match col.data_type() {
        DataType::Boolean => {
            let arr = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            for i in 0..arr.len() {
                buf.push(if arr.value(i) { 1 } else { 0 });
            }
        }
        DataType::Int8 => append_primitive_buffer::<Int8Type>(col, buf),
        DataType::UInt8 => append_primitive_buffer::<UInt8Type>(col, buf),
        DataType::Int16 => append_primitive_buffer::<Int16Type>(col, buf),
        DataType::UInt16 => append_primitive_buffer::<UInt16Type>(col, buf),
        DataType::Int32 => append_primitive_buffer::<Int32Type>(col, buf),
        DataType::UInt32 => append_primitive_buffer::<UInt32Type>(col, buf),
        DataType::Int64 => append_primitive_buffer::<Int64Type>(col, buf),
        DataType::UInt64 => append_primitive_buffer::<UInt64Type>(col, buf),
        DataType::Float32 => append_primitive_buffer::<Float32Type>(col, buf),
        DataType::Float64 => append_primitive_buffer::<Float64Type>(col, buf),
        DataType::Date32 => {
            // Doris DATEv2 is uint32
            let arr = col.as_any().downcast_ref::<Date32Array>().unwrap();
            for i in 0..arr.len() {
                buf.extend_from_slice(&(arr.value(i) as u32).to_le_bytes());
            }
        }
        DataType::Date64 => {
            let arr = col.as_any().downcast_ref::<Date64Array>().unwrap();
            for i in 0..arr.len() {
                buf.extend_from_slice(&arr.value(i).to_le_bytes());
            }
        }
        DataType::Timestamp(_, _) => {
            // Doris DATETIMEV2 is uint64
            let arr = col
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .or_else(|| None);
            if let Some(arr) = arr {
                for i in 0..arr.len() {
                    buf.extend_from_slice(&(arr.value(i) as u64).to_le_bytes());
                }
            } else {
                // Try other timestamp types — fall back to raw i64
                let arr = col
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .unwrap();
                for i in 0..arr.len() {
                    buf.extend_from_slice(&(arr.value(i) as u64).to_le_bytes());
                }
            }
        }
        DataType::Decimal128(_, _) => {
            let arr = col.as_any().downcast_ref::<Decimal128Array>().unwrap();
            let (p, _) = match col.data_type() {
                DataType::Decimal128(p, _) => (*p, 0),
                _ => (38, 0),
            };
            if p <= 9 {
                // DECIMAL32: i32
                for i in 0..arr.len() {
                    buf.extend_from_slice(&(arr.value(i) as i32).to_le_bytes());
                }
            } else if p <= 18 {
                // DECIMAL64: i64
                for i in 0..arr.len() {
                    buf.extend_from_slice(&(arr.value(i) as i64).to_le_bytes());
                }
            } else {
                // DECIMAL128I: i128
                for i in 0..arr.len() {
                    buf.extend_from_slice(&arr.value(i).to_le_bytes());
                }
            }
        }
        _ => {
            return Err(format!(
                "unsupported Arrow type for PBlock encoding: {:?}",
                col.data_type()
            ));
        }
    }
    Ok(())
}

/// Append raw bytes from a primitive Arrow array.
fn append_primitive_buffer<T: arrow::array::ArrowPrimitiveType>(
    col: &dyn arrow::array::Array,
    buf: &mut Vec<u8>,
) {
    use arrow::array::PrimitiveArray;
    let arr = col.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    // Access the underlying buffer directly for zero-copy.
    let values = arr.values();
    let raw = values.inner().as_slice();
    buf.extend_from_slice(raw);
}

/// Map Arrow DataType to Doris PGenericType TypeId, precision, scale.
fn arrow_type_to_doris_type_id(dt: &arrow::datatypes::DataType) -> (i32, i32, i32) {
    use arrow::datatypes::DataType;
    use doris_proto::doris::p_generic_type::TypeId;

    match dt {
        DataType::Boolean => (TypeId::Boolean as i32, 0, 0),
        DataType::Int8 => (TypeId::Int8 as i32, 0, 0),
        DataType::UInt8 => (TypeId::Uint8 as i32, 0, 0),
        DataType::Int16 => (TypeId::Int16 as i32, 0, 0),
        DataType::UInt16 => (TypeId::Uint16 as i32, 0, 0),
        DataType::Int32 => (TypeId::Int32 as i32, 0, 0),
        DataType::UInt32 => (TypeId::Uint32 as i32, 0, 0),
        DataType::Int64 => (TypeId::Int64 as i32, 0, 0),
        DataType::UInt64 => (TypeId::Uint64 as i32, 0, 0),
        DataType::Float32 => (TypeId::Float as i32, 0, 0),
        DataType::Float64 => (TypeId::Double as i32, 0, 0),
        DataType::Date32 => (TypeId::Datev2 as i32, 0, 0),
        DataType::Date64 => (TypeId::Date as i32, 0, 0),
        DataType::Timestamp(_, _) => (TypeId::Datetimev2 as i32, 0, 0),
        DataType::Utf8 | DataType::LargeUtf8 => (TypeId::String as i32, 0, 0),
        DataType::Decimal128(p, s) => {
            let p = *p as i32;
            let s = *s as i32;
            if p <= 9 {
                (TypeId::Decimal32 as i32, p, s)
            } else if p <= 18 {
                (TypeId::Decimal64 as i32, p, s)
            } else {
                (TypeId::Decimal128i as i32, p, s)
            }
        }
        // Fallback to STRING
        _ => (TypeId::String as i32, 0, 0),
    }
}

fn is_string_type_id(type_id: i32) -> bool {
    use doris_proto::doris::p_generic_type::TypeId;
    type_id == TypeId::String as i32
        || type_id == TypeId::Bytes as i32
        || type_id == TypeId::Jsonb as i32
        || type_id == TypeId::Variant as i32
}

fn type_byte_width(type_id: i32) -> usize {
    use doris_proto::doris::p_generic_type::TypeId;
    match type_id {
        x if x == TypeId::Boolean as i32 => 1,
        x if x == TypeId::Int8 as i32 || x == TypeId::Uint8 as i32 => 1,
        x if x == TypeId::Int16 as i32 || x == TypeId::Uint16 as i32 => 2,
        x if x == TypeId::Int32 as i32 || x == TypeId::Uint32 as i32 => 4,
        x if x == TypeId::Int64 as i32 || x == TypeId::Uint64 as i32 => 8,
        x if x == TypeId::Int128 as i32 || x == TypeId::Uint128 as i32 => 16,
        x if x == TypeId::Float as i32 => 4,
        x if x == TypeId::Double as i32 => 8,
        x if x == TypeId::Datev2 as i32 => 4,
        x if x == TypeId::Datetimev2 as i32 => 8,
        x if x == TypeId::Date as i32 => 8,
        x if x == TypeId::Datetime as i32 => 8,
        x if x == TypeId::Decimal32 as i32 => 4,
        x if x == TypeId::Decimal64 as i32 => 8,
        x if x == TypeId::Decimal128 as i32 => 16,
        x if x == TypeId::Decimal128i as i32 => 16,
        x if x == TypeId::Decimal256 as i32 => 32,
        x if x == TypeId::Ipv4 as i32 => 4,
        x if x == TypeId::Ipv6 as i32 => 16,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn make_ipc(batch: &RecordBatch) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut writer = StreamWriter::try_new(&mut buf, &batch.schema()).unwrap();
        writer.write(batch).unwrap();
        writer.finish().unwrap();
        buf
    }

    #[test]
    fn test_encode_int_columns() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();

        let ipc = make_ipc(&batch);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();

        assert_eq!(num_rows, 3);
        assert_eq!(pblock.column_metas.len(), 2);
        assert!(pblock.column_values.is_some());
    }

    #[test]
    fn test_encode_string_column() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["hello", "world"]))],
        )
        .unwrap();

        let ipc = make_ipc(&batch);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();

        assert_eq!(num_rows, 2);
        assert_eq!(pblock.column_metas.len(), 1);
    }

    #[test]
    fn test_encode_nullable_column() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![
                Some(1),
                None,
                Some(3),
            ]))],
        )
        .unwrap();

        let ipc = make_ipc(&batch);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();

        assert_eq!(num_rows, 3);
        assert!(pblock.column_metas[0].is_nullable.unwrap());
    }

    #[test]
    fn test_empty_batch() {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![] as Vec<i32>))]).unwrap();

        let ipc = make_ipc(&batch);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();

        assert_eq!(num_rows, 0);
    }
}
