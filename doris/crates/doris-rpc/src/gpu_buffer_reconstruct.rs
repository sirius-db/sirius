//! Reconstruct Arrow RecordBatch from GPU buffers received via nixl transfer.
//!
//! After nixl GPU-direct transfer, the receiver has column data, null bitmaps,
//! and string offsets in GPU memory. This module D2H-copies each buffer,
//! then reconstructs Arrow arrays and serializes them to IPC bytes compatible
//! with `arrow_ipc_to_pblock`.

use std::sync::Arc;

use arrow::array::*;
use arrow::buffer::{Buffer, NullBuffer};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;

use doris_proto::nixl::{PColumnInfo, PGpuBufferDesc};

use crate::cuda_driver::gpu_to_host;

/// cudf type_id enum values (from cudf/types.hpp).
mod cudf_type {
    pub const INT8: i32 = 1;
    pub const INT16: i32 = 2;
    pub const INT32: i32 = 3;
    pub const INT64: i32 = 4;
    pub const UINT8: i32 = 5;
    pub const UINT16: i32 = 6;
    pub const UINT32: i32 = 7;
    pub const UINT64: i32 = 8;
    pub const FLOAT32: i32 = 9;
    pub const FLOAT64: i32 = 10;
    pub const BOOL8: i32 = 11;
    pub const TIMESTAMP_DAYS: i32 = 12;
    pub const TIMESTAMP_SECONDS: i32 = 13;
    pub const TIMESTAMP_MILLISECONDS: i32 = 14;
    pub const TIMESTAMP_MICROSECONDS: i32 = 15;
    pub const TIMESTAMP_NANOSECONDS: i32 = 16;
    pub const STRING: i32 = 23;
    pub const DECIMAL32: i32 = 25;
    pub const DECIMAL64: i32 = 26;
    pub const DECIMAL128: i32 = 27;
}

/// Map cudf type_id to Arrow DataType.
///
/// For DECIMAL types, precision and scale are required (from PColumnInfo).
fn cudf_type_to_arrow(type_id: i32, precision: i32, scale: i32) -> Option<DataType> {
    match type_id {
        cudf_type::INT8 => Some(DataType::Int8),
        cudf_type::INT16 => Some(DataType::Int16),
        cudf_type::INT32 => Some(DataType::Int32),
        cudf_type::INT64 => Some(DataType::Int64),
        cudf_type::UINT8 => Some(DataType::UInt8),
        cudf_type::UINT16 => Some(DataType::UInt16),
        cudf_type::UINT32 => Some(DataType::UInt32),
        cudf_type::UINT64 => Some(DataType::UInt64),
        cudf_type::FLOAT32 => Some(DataType::Float32),
        cudf_type::FLOAT64 => Some(DataType::Float64),
        cudf_type::BOOL8 => Some(DataType::Boolean),
        cudf_type::TIMESTAMP_DAYS => Some(DataType::Date32),
        cudf_type::TIMESTAMP_SECONDS => Some(DataType::Timestamp(
            arrow::datatypes::TimeUnit::Second,
            None,
        )),
        cudf_type::TIMESTAMP_MILLISECONDS => Some(DataType::Timestamp(
            arrow::datatypes::TimeUnit::Millisecond,
            None,
        )),
        cudf_type::TIMESTAMP_MICROSECONDS => Some(DataType::Timestamp(
            arrow::datatypes::TimeUnit::Microsecond,
            None,
        )),
        cudf_type::TIMESTAMP_NANOSECONDS => Some(DataType::Timestamp(
            arrow::datatypes::TimeUnit::Nanosecond,
            None,
        )),
        cudf_type::STRING => Some(DataType::Utf8),
        // cudf DECIMAL32 stores i32 values, DECIMAL64 stores i64, DECIMAL128 stores i128.
        // cudf scale is negated (e.g., scale=-2 means 2 decimal places).
        // Arrow Decimal128 uses positive scale.
        cudf_type::DECIMAL32 | cudf_type::DECIMAL64 | cudf_type::DECIMAL128 => {
            let p = if precision > 0 { precision as u8 } else { 38 };
            let s = (-scale) as i8; // cudf uses negative scale
            Some(DataType::Decimal128(p, s))
        }
        _ => None,
    }
}

/// D2H a GPU buffer. Returns empty Vec if addr is 0 or len is 0.
fn d2h(addr: u64, len: u64) -> Result<Vec<u8>, String> {
    if addr == 0 || len == 0 {
        return Ok(vec![]);
    }
    gpu_to_host(addr as usize, len as usize)
}

/// Convert cudf BOOL8 (1 byte per value) to Arrow bit-packed boolean buffer.
fn bool8_to_arrow_bitmap(data: &[u8], num_rows: usize) -> Buffer {
    let mut bits = vec![0u8; (num_rows + 7) / 8];
    for i in 0..num_rows.min(data.len()) {
        if data[i] != 0 {
            bits[i / 8] |= 1 << (i % 8);
        }
    }
    Buffer::from(bits)
}

/// Build a NullBuffer from raw cudf bitmask bytes.
///
/// cudf uses same convention as Arrow: 1 = valid, 0 = null.
/// The buffer is already in the right format.
fn make_null_buffer(null_mask: &[u8], null_count: i32, num_rows: usize) -> Option<NullBuffer> {
    if null_count == 0 || null_mask.is_empty() {
        return None;
    }
    let buf = Buffer::from(null_mask.to_vec());
    // NullBuffer::try_new returns Err if buffer too small; use new_unchecked.
    Some(NullBuffer::new(arrow::buffer::BooleanBuffer::new(
        buf, 0, num_rows,
    )))
}

/// Build an Arrow array from raw D2H'd host buffers for a single column.
fn build_arrow_array(
    data: Vec<u8>,
    null_mask: Vec<u8>,
    offsets: Vec<u8>,
    null_count: i32,
    num_rows: usize,
    arrow_type: &DataType,
) -> Result<ArrayRef, String> {
    let nulls = make_null_buffer(&null_mask, null_count, num_rows);

    match arrow_type {
        DataType::Boolean => {
            // cudf BOOL8: 1 byte per value → Arrow bit-packed.
            let values_buf = bool8_to_arrow_bitmap(&data, num_rows);
            let bool_buf = arrow::buffer::BooleanBuffer::new(values_buf, 0, num_rows);
            Ok(Arc::new(BooleanArray::new(bool_buf, nulls)))
        }
        DataType::Int8 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(Int8Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::Int16 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(Int16Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::Int32 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(Int32Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::Int64 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(Int64Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::UInt8 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(UInt8Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::UInt16 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(UInt16Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::UInt32 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(UInt32Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::UInt64 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(UInt64Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::Float32 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(Float32Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::Float64 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(Float64Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::Date32 => {
            let buf = Buffer::from(data);
            Ok(Arc::new(Date32Array::new(
                buf.into(),
                nulls,
            )))
        }
        DataType::Timestamp(unit, tz) => {
            let buf = Buffer::from(data);
            let sb: arrow::buffer::ScalarBuffer<i64> = buf.into();
            match unit {
                arrow::datatypes::TimeUnit::Second => Ok(Arc::new(
                    TimestampSecondArray::new(sb, nulls)
                        .with_timezone_opt(tz.clone()),
                )),
                arrow::datatypes::TimeUnit::Millisecond => Ok(Arc::new(
                    TimestampMillisecondArray::new(sb, nulls)
                        .with_timezone_opt(tz.clone()),
                )),
                arrow::datatypes::TimeUnit::Microsecond => Ok(Arc::new(
                    TimestampMicrosecondArray::new(sb, nulls)
                        .with_timezone_opt(tz.clone()),
                )),
                arrow::datatypes::TimeUnit::Nanosecond => Ok(Arc::new(
                    TimestampNanosecondArray::new(sb, nulls)
                        .with_timezone_opt(tz.clone()),
                )),
            }
        }
        DataType::Utf8 => {
            // cudf STRING: chars buffer (data) + INT32 offsets (num_rows+1 entries).
            if offsets.is_empty() {
                return Err("STRING column missing offsets buffer".to_string());
            }
            let offsets_buf = Buffer::from(offsets);
            let values_buf = Buffer::from(data);
            // Safety: offsets are i32, values are raw chars, layout matches Arrow Utf8.
            let offsets_sb: arrow::buffer::ScalarBuffer<i32> = offsets_buf.into();
            let offset_buf = arrow::buffer::OffsetBuffer::new(offsets_sb);
            Ok(Arc::new(GenericStringArray::<i32>::new(
                offset_buf,
                values_buf,
                nulls,
            )))
        }
        DataType::Decimal128(p, s) => {
            // cudf DECIMAL32/64: raw i32/i64 values, widen to i128 for Arrow.
            // cudf DECIMAL128: raw i128 values, use directly.
            let i128_values: Vec<i128> = match data.len() / num_rows.max(1) {
                4 => {
                    // DECIMAL32: i32 values
                    data.chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i128)
                        .collect()
                }
                8 => {
                    // DECIMAL64: i64 values
                    data.chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i128)
                        .collect()
                }
                16 => {
                    // DECIMAL128: i128 values
                    data.chunks_exact(16)
                        .map(|c| i128::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                }
                bytes_per_row => {
                    return Err(format!(
                        "unexpected DECIMAL element size: {} bytes/row",
                        bytes_per_row
                    ));
                }
            };
            Ok(Arc::new(
                Decimal128Array::from(i128_values)
                    .with_precision_and_scale(*p, *s)
                    .map_err(|e| format!("Decimal128 precision/scale: {e}"))?,
            ))
        }
        _ => Err(format!("unsupported Arrow type for GPU reconstruction: {arrow_type:?}")),
    }
}

/// Reconstruct Arrow IPC bytes from GPU buffers received via nixl.
///
/// D2H-copies each GPU buffer, builds Arrow arrays, and serializes to IPC
/// stream format compatible with `arrow_ipc_to_pblock`.
///
/// Returns `Ok(ipc_bytes)` on success, or `Err` if any D2H or reconstruction fails.
/// Caller should fall back to the Arrow IPC bytes sent via gRPC on error.
pub fn reconstruct_ipc_from_gpu_buffers(
    dst_buffers: &[PGpuBufferDesc],
    dst_null_masks: &[PGpuBufferDesc],
    dst_offsets: &[PGpuBufferDesc],
    columns: &[PColumnInfo],
    null_counts: &[i32],
    num_rows: u32,
) -> Result<Vec<u8>, String> {
    let n = dst_buffers.len();
    if n != columns.len() {
        return Err(format!(
            "column count mismatch: {} buffers, {} columns",
            n,
            columns.len()
        ));
    }

    // Check that all types are supported before doing any D2H.
    let arrow_types: Vec<DataType> = columns
        .iter()
        .map(|c| {
            cudf_type_to_arrow(c.type_id, c.precision, c.scale).ok_or_else(|| {
                format!(
                    "unsupported cudf type_id {} for column '{}'",
                    c.type_id, c.name
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // D2H all buffers.
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(n);
    for i in 0..n {
        let data_buf = d2h(dst_buffers[i].addr, dst_buffers[i].len)?;
        let nm_buf = if let Some(nm) = dst_null_masks.get(i) {
            d2h(nm.addr, nm.len)?
        } else {
            vec![]
        };
        let off_buf = if let Some(off) = dst_offsets.get(i) {
            d2h(off.addr, off.len)?
        } else {
            vec![]
        };
        let nc = null_counts.get(i).copied().unwrap_or(0);

        let array = build_arrow_array(
            data_buf,
            nm_buf,
            off_buf,
            nc,
            num_rows as usize,
            &arrow_types[i],
        )?;
        arrays.push(array);
    }

    // Build schema and RecordBatch.
    let fields: Vec<Field> = columns
        .iter()
        .zip(arrow_types.iter())
        .zip(null_counts.iter().chain(std::iter::repeat(&0)))
        .map(|((col, dt), &nc)| Field::new(&col.name, dt.clone(), nc > 0))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)
        .map_err(|e| format!("RecordBatch construction: {e}"))?;

    // Serialize to Arrow IPC stream format.
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)
            .map_err(|e| format!("IPC StreamWriter: {e}"))?;
        writer
            .write(&batch)
            .map_err(|e| format!("IPC write: {e}"))?;
        writer
            .finish()
            .map_err(|e| format!("IPC finish: {e}"))?;
    }

    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cudf_type_mapping() {
        assert_eq!(cudf_type_to_arrow(1, 0, 0), Some(DataType::Int8));
        assert_eq!(cudf_type_to_arrow(4, 0, 0), Some(DataType::Int64));
        assert_eq!(cudf_type_to_arrow(10, 0, 0), Some(DataType::Float64));
        assert_eq!(cudf_type_to_arrow(11, 0, 0), Some(DataType::Boolean));
        assert_eq!(cudf_type_to_arrow(23, 0, 0), Some(DataType::Utf8));
        assert_eq!(cudf_type_to_arrow(12, 0, 0), Some(DataType::Date32));
        // Decimal with precision/scale.
        assert_eq!(
            cudf_type_to_arrow(26, 18, -2),
            Some(DataType::Decimal128(18, 2))
        );
        assert_eq!(cudf_type_to_arrow(99, 0, 0), None);
    }

    #[test]
    fn test_bool8_to_bitmap() {
        let data = vec![1u8, 0, 1, 1, 0, 1, 0, 0, 1];
        let buf = bool8_to_arrow_bitmap(&data, 9);
        // Bit 0=1, 1=0, 2=1, 3=1, 4=0, 5=1, 6=0, 7=0 → 0b00101101 = 0x2D
        // Bit 8=1 → 0b00000001 = 0x01
        assert_eq!(buf.as_slice(), &[0x2D, 0x01]);
    }
}
