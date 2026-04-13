//! Arrow IPC → PBlock encoder: converts Arrow record batches to Doris PBlock
//! column binary format for inter-BE exchange via `transmit_block`.
//!
//! This is the inverse of `pblock_decoder.rs`. The encoding follows the Doris
//! column_values wire format:
//!   - Per column: header (const_flag + row_num + real_saved_num) + data
//!   - Fixed-width: raw bytes if mem_size <= 256, else StreamVByte encoded
//!   - STRING: offsets (raw/StreamVByte) + value_len + chars (raw/LZ4)
//!   - NULLABLE: null_map (raw/StreamVByte) + inner column data

use std::io::Cursor;

use arrow::ipc::reader::StreamReader;
use doris_proto::doris::{p_column_meta, PBlock, PColumnMeta};

/// Doris SERIALIZED_MEM_SIZE_LIMIT: above this, per-column data uses StreamVByte/LZ4.
const SERIALIZED_MEM_SIZE_LIMIT: usize = 256;

/// StreamVByte encoder: inverse of `pblock_decoder::streamvbyte_decode()`.
///
/// Encodes N u32 values into: `control_bytes[(N+3)/4] + data_bytes[variable]`.
/// Each control byte has 4 x 2-bit entries encoding the byte length minus 1.
fn streamvbyte_encode(values: &[u32]) -> Vec<u8> {
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

/// Write raw bytes (when `mem_size <= 256`) or StreamVByte-encoded bytes (otherwise).
///
/// Mirror of `pblock_decoder::read_raw_or_fail()`.
fn write_raw_or_encoded(buf: &mut Vec<u8>, raw_bytes: &[u8]) {
    if raw_bytes.len() <= SERIALIZED_MEM_SIZE_LIMIT {
        buf.extend_from_slice(raw_bytes);
    } else {
        // Pad to multiple of 4 bytes, interpret as u32 LE, StreamVByte encode
        let num_u32 = (raw_bytes.len() + 3) / 4;
        let mut padded = raw_bytes.to_vec();
        padded.resize(num_u32 * 4, 0);
        let values: Vec<u32> = padded
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let encoded = streamvbyte_encode(&values);
        buf.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        buf.extend_from_slice(&encoded);
    }
}

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

    // Log Arrow schema types for PBlock encoding diagnostics.
    for (i, field) in schema.fields().iter().enumerate() {
        tracing::debug!(
            col_idx = i,
            name = %field.name(),
            arrow_type = ?field.data_type(),
            nullable = field.is_nullable(),
            "arrow_to_pblock: encoding column"
        );
    }

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
            encode_nullable_column(&batches, col_idx, num_rows, type_id, &mut column_values)?;
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
        // Fixed-width encoding: collect into temp buffer, then write raw or StreamVByte
        let width = type_byte_width(type_id);
        if width == 0 {
            return Err(format!("unsupported type_id {} for encoding", type_id));
        }

        let mut tmp = Vec::with_capacity(num_rows as usize * width);
        for batch in batches {
            let col = batch.column(col_idx);
            append_fixed_width_data(col, type_id, width, &mut tmp)?;
        }
        write_raw_or_encoded(buf, &tmp);
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
    // null_map: raw if <= 256 bytes, StreamVByte encoded otherwise
    write_raw_or_encoded(buf, &null_map);

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

    // Write offsets: raw if N*4 <= 256, StreamVByte encoded otherwise
    let offsets_bytes: Vec<u8> = offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
    write_raw_or_encoded(buf, &offsets_bytes);

    // Write value_len (8 bytes) + chars (raw if <= 256, LZ4 compressed otherwise)
    let value_len = all_chars.len() as u64;
    buf.extend_from_slice(&value_len.to_le_bytes());
    if all_chars.len() <= SERIALIZED_MEM_SIZE_LIMIT {
        buf.extend_from_slice(&all_chars);
    } else {
        let compressed = lz4_flex::compress(&all_chars);
        buf.extend_from_slice(&(compressed.len() as u64).to_le_bytes());
        buf.extend_from_slice(&compressed);
    }

    Ok(())
}

/// Convert days since 1970-01-01 (Arrow Date32) to Doris DATEv2 packed uint32.
/// DATEv2 = (year << 9) | (month << 5) | day
fn days_to_datev2(days: i32) -> u32 {
    // Civil date from days since epoch (algorithm from Howard Hinnant).
    let z = days + 719468; // shift epoch to 0000-03-01
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i32 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    ((y as u32) << 9) | (m << 5) | d
}

/// Convert microseconds since 1970-01-01 (Arrow Timestamp) to Doris DATETIMEV2 packed uint64.
/// DATETIMEV2 = (year<<46 | month<<42 | day<<37 | hour<<32 | minute<<26 | second<<20 | microseconds)
fn micros_to_datetimev2(micros: i64) -> u64 {
    let total_secs = micros.div_euclid(1_000_000);
    let us = micros.rem_euclid(1_000_000) as u64;
    let days = total_secs.div_euclid(86400) as i32;
    let day_secs = total_secs.rem_euclid(86400) as u64;
    let hour = day_secs / 3600;
    let minute = (day_secs % 3600) / 60;
    let second = day_secs % 60;

    let datev2 = days_to_datev2(days);
    let year = (datev2 >> 9) as u64;
    let month = ((datev2 >> 5) & 0xF) as u64;
    let day = (datev2 & 0x1F) as u64;

    (year << 46) | (month << 42) | (day << 37) | (hour << 32) | (minute << 26) | (second << 20) | us
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
            // Arrow Date32 = days since 1970-01-01.
            // Doris DATEv2 = packed uint32: (year<<9 | month<<5 | day).
            let arr = col.as_any().downcast_ref::<Date32Array>().unwrap();
            for i in 0..arr.len() {
                let days = arr.value(i);
                let packed = days_to_datev2(days);
                buf.extend_from_slice(&packed.to_le_bytes());
            }
        }
        DataType::Date64 => {
            // Arrow Date64 = milliseconds since 1970-01-01.
            // Doris DATEv2 = packed uint32: (year<<9 | month<<5 | day).
            let arr = col.as_any().downcast_ref::<Date64Array>().unwrap();
            for i in 0..arr.len() {
                let millis = arr.value(i);
                let days = (millis / 86_400_000) as i32;
                let packed = days_to_datev2(days);
                // Date64 maps to DATEv2 (4 bytes), not Date (8 bytes)
                buf.extend_from_slice(&packed.to_le_bytes());
            }
        }
        DataType::Timestamp(_, _) => {
            // Doris DATETIMEV2 = packed uint64:
            //   (year<<46 | month<<42 | day<<37 | hour<<32 | minute<<26 | second<<20 | microseconds)
            let arr = col
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .or_else(|| None);
            if let Some(arr) = arr {
                for i in 0..arr.len() {
                    let packed = micros_to_datetimev2(arr.value(i));
                    buf.extend_from_slice(&packed.to_le_bytes());
                }
            } else {
                let arr = col
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .unwrap();
                for i in 0..arr.len() {
                    let packed = micros_to_datetimev2(arr.value(i) * 1000);
                    buf.extend_from_slice(&packed.to_le_bytes());
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
        DataType::Date32 | DataType::Date64 => (TypeId::Datev2 as i32, 0, 0),
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
            vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))],
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
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![] as Vec<i32>))],
        )
        .unwrap();

        let ipc = make_ipc(&batch);
        let (_pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();

        assert_eq!(num_rows, 0);
    }

    #[test]
    fn test_streamvbyte_roundtrip() {
        use crate::pblock_decoder::streamvbyte_decode;

        // Mix of 1/2/3/4 byte values
        let values: Vec<u32> = vec![0, 1, 127, 255, 256, 0xFFFF, 0x10000, 0xFFFFFFFF, 42];
        let encoded = streamvbyte_encode(&values);
        let decoded = streamvbyte_decode(&encoded, values.len()).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_streamvbyte_encode_empty() {
        let encoded = streamvbyte_encode(&[]);
        assert!(encoded.is_empty());
    }

    /// Roundtrip: encode 100 INT64 rows (800 bytes > 256) → PBlock → decode → verify.
    #[test]
    fn test_roundtrip_large_fixed_column() {
        use crate::pblock_decoder::decode_pblocks;

        let n = 100;
        let values: Vec<i64> = (0..n).map(|i| i * 1000 + 7).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.clone()))]).unwrap();

        let ipc = make_ipc(&batch);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();
        assert_eq!(num_rows, n as u32);

        // Decode and verify
        let decoded = decode_pblocks(&[pblock]).unwrap();
        assert_eq!(decoded.num_rows, n as u32);
        assert_eq!(decoded.columns.len(), 1);
        let col = &decoded.columns[0];
        assert_eq!(col.data.len(), n as usize * 8);
        for (i, expected) in values.iter().enumerate() {
            let got = i64::from_le_bytes(col.data[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(got, *expected, "mismatch at row {i}");
        }
    }

    /// Roundtrip: 100 nullable INT32 rows (data 400 bytes > 256, triggers StreamVByte).
    #[test]
    fn test_roundtrip_large_nullable_column() {
        use crate::pblock_decoder::decode_pblocks;

        let n = 100;
        let values: Vec<Option<i32>> = (0..n)
            .map(|i| if i % 5 == 0 { None } else { Some(i * 3) })
            .collect();
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values.clone()))]).unwrap();

        let ipc = make_ipc(&batch);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();
        assert_eq!(num_rows, n as u32);

        let decoded = decode_pblocks(&[pblock]).unwrap();
        assert_eq!(decoded.num_rows, n as u32);
        let col = &decoded.columns[0];
        assert!(col.is_nullable);
        let mask = col.null_mask.as_ref().unwrap();
        for (i, expected) in values.iter().enumerate() {
            if expected.is_none() {
                assert_eq!(mask[i], 1, "expected null at row {i}");
            } else {
                assert_eq!(mask[i], 0, "expected non-null at row {i}");
                let got = i32::from_le_bytes(col.data[i * 4..(i + 1) * 4].try_into().unwrap());
                assert_eq!(got, expected.unwrap(), "value mismatch at row {i}");
            }
        }
    }

    /// Roundtrip: 100 strings (offsets > 256 bytes, chars > 256 bytes → LZ4).
    #[test]
    fn test_roundtrip_large_string_column() {
        use crate::pblock_decoder::decode_pblocks;

        let n = 100;
        let strings: Vec<String> = (0..n).map(|i| format!("value_{:04}", i)).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(
                strings.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap();

        let ipc = make_ipc(&batch);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();
        assert_eq!(num_rows, n as u32);

        let decoded = decode_pblocks(&[pblock]).unwrap();
        assert_eq!(decoded.num_rows, n as u32);
        let col = &decoded.columns[0];
        let offsets = col.offsets.as_ref().unwrap();
        assert_eq!(offsets.len(), n as usize + 1);
        for (i, expected) in strings.iter().enumerate() {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            let got = std::str::from_utf8(&col.data[start..end]).unwrap();
            assert_eq!(got, expected, "string mismatch at row {i}");
        }
    }

    /// Roundtrip: small column (< 256 bytes) uses raw path — regression test.
    #[test]
    fn test_roundtrip_small_column_raw() {
        use crate::pblock_decoder::decode_pblocks;

        let values = vec![1i32, 2, 3]; // 12 bytes, well under 256
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(values.clone()))]).unwrap();

        let ipc = make_ipc(&batch);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();
        assert_eq!(num_rows, 3);

        let decoded = decode_pblocks(&[pblock]).unwrap();
        assert_eq!(decoded.num_rows, 3);
        let col = &decoded.columns[0];
        for (i, expected) in values.iter().enumerate() {
            let got = i32::from_le_bytes(col.data[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(got, *expected);
        }
    }

    #[test]
    fn test_days_to_datev2() {
        // 1970-01-01 = day 0
        assert_eq!(days_to_datev2(0), (1970 << 9) | (1 << 5) | 1);
        // 1998-12-01 = day 10561
        let packed = days_to_datev2(10561);
        let y = packed >> 9;
        let m = (packed >> 5) & 0xF;
        let d = packed & 0x1F;
        assert_eq!((y, m, d), (1998, 12, 1));
        // 2024-02-29 (leap year)
        let days_20240229 = 19782; // 2024-02-29
        let packed = days_to_datev2(days_20240229);
        let y = packed >> 9;
        let m = (packed >> 5) & 0xF;
        let d = packed & 0x1F;
        assert_eq!((y, m, d), (2024, 2, 29));
    }

    /// Roundtrip: Date32 columns encode as packed DATEv2 and decode back correctly.
    #[test]
    fn test_roundtrip_date32_column() {
        use crate::pblock_decoder::{column_to_sql_values, decode_pblocks};

        // TPC-H-like dates: 1992-01-02, 1998-12-01
        let days = vec![8036i32, 10561]; // days since epoch for those dates
        let schema = Arc::new(Schema::new(vec![Field::new("d", DataType::Date32, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Date32Array::from(days.clone()))]).unwrap();

        let ipc = make_ipc(&batch);
        let (pblock, num_rows) = arrow_ipc_to_pblock(&ipc).unwrap();
        assert_eq!(num_rows, 2);

        let decoded = decode_pblocks(&[pblock]).unwrap();
        assert_eq!(decoded.num_rows, 2);
        let sql_vals = column_to_sql_values(&decoded.columns[0], 2);
        assert_eq!(sql_vals[0], "'1992-01-02'");
        assert_eq!(sql_vals[1], "'1998-12-01'");
    }
}
