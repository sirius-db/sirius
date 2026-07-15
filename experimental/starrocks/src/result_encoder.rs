//! Encodes executed-fragment output into the StarRocks result wire format.
//!
//! StarRocks delivers SELECT results to the FE as a `TResultBatch` whose `rows` are MySQL
//! text-protocol resultset rows: each column value is a length-encoded string, NULL is the single
//! byte `0xFB`. The FE forwards those row bodies straight to the MySQL client.

use arrow_array::temporal_conversions::{as_date, as_datetime};
use arrow_array::types::{Date32Type, TimestampMicrosecondType};
use arrow_array::{
    Array, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    StringViewArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use starrocks_thrift::data::TResultBatch;
use thrift::protocol::{TBinaryOutputProtocol, TSerializable};

/// Encodes Arrow result batches into a StarRocks `TResultBatch` of MySQL text rows.
#[derive(Default)]
pub(crate) struct MysqlResultEncoder {
    rows: Vec<Vec<u8>>,
}

impl MysqlResultEncoder {
    /// Encodes `batches` into a `TResultBatch` tagged with `packet_seq`.
    pub(crate) fn encode(batches: &[RecordBatch], packet_seq: i64) -> Result<TResultBatch, String> {
        let mut encoder = Self::default();
        for batch in batches {
            encoder.add_batch(batch)?;
        }
        Ok(encoder.into_result_batch(packet_seq))
    }

    /// Encodes every row of `batch` as a MySQL text row.
    fn add_batch(&mut self, batch: &RecordBatch) -> Result<(), String> {
        let columns = batch.columns();
        for row in 0..batch.num_rows() {
            let mut encoded = MysqlTextRow::default();
            for column in columns {
                encoded.push_cell(Self::render_cell(column.as_ref(), row)?.as_deref());
            }
            self.rows.push(encoded.into_bytes());
        }
        Ok(())
    }

    /// Consumes the accumulated rows into a `TResultBatch`.
    ///
    /// `is_compressed = false`: the Rust CN never snappy-compresses result rows yet.
    fn into_result_batch(self, packet_seq: i64) -> TResultBatch {
        TResultBatch::new(self.rows, false, packet_seq, None)
    }

    /// Renders one column value as raw text bytes, or `None` for SQL NULL.
    fn render_cell(array: &dyn Array, row: usize) -> Result<Option<Vec<u8>>, String> {
        if array.is_null(row) {
            return Ok(None);
        }
        // Render the value as the MySQL client expects in the text protocol: a decimal/string form.
        macro_rules! render_primitive {
            ($ty:ty) => {{
                let typed = Self::downcast::<$ty>(array)?;
                Ok(Some(typed.value(row).to_string().into_bytes()))
            }};
        }
        match array.data_type() {
            DataType::Utf8 => {
                let typed = Self::downcast::<StringArray>(array)?;
                Ok(Some(typed.value(row).as_bytes().to_vec()))
            }
            DataType::Boolean => {
                let typed = Self::downcast::<BooleanArray>(array)?;
                Ok(Some(if typed.value(row) {
                    b"1".to_vec()
                } else {
                    b"0".to_vec()
                }))
            }
            DataType::Int8 => render_primitive!(Int8Array),
            DataType::Int16 => render_primitive!(Int16Array),
            DataType::Int32 => render_primitive!(Int32Array),
            DataType::Int64 => render_primitive!(Int64Array),
            DataType::Float32 => render_primitive!(Float32Array),
            DataType::Float64 => render_primitive!(Float64Array),
            DataType::LargeUtf8 => {
                let typed = Self::downcast::<LargeStringArray>(array)?;
                Ok(Some(typed.value(row).as_bytes().to_vec()))
            }
            DataType::Utf8View => {
                let typed = Self::downcast::<StringViewArray>(array)?;
                Ok(Some(typed.value(row).as_bytes().to_vec()))
            }
            DataType::Decimal128(_, _) => {
                let typed = Self::downcast::<Decimal128Array>(array)?;
                Ok(Some(typed.value_as_string(row).into_bytes()))
            }
            DataType::Date32 => {
                let typed = Self::downcast::<Date32Array>(array)?;
                let date = as_date::<Date32Type>(i64::from(typed.value(row)))
                    .ok_or_else(|| format!("date32 value {} out of range", typed.value(row)))?;
                Ok(Some(date.format("%Y-%m-%d").to_string().into_bytes()))
            }
            // StarRocks DATETIME is timezone-naive wall-clock; a tz-aware timestamp would need
            // offset handling we don't implement, so let `Some(tz)` fall through to the error arm.
            DataType::Timestamp(TimeUnit::Microsecond, None) => {
                let typed = Self::downcast::<TimestampMicrosecondArray>(array)?;
                let value = typed.value(row);
                let datetime = as_datetime::<TimestampMicrosecondType>(value)
                    .ok_or_else(|| format!("timestamp value {value} out of range"))?;
                // MySQL text format; StarRocks omits the fractional part when it is zero.
                let format = if value % 1_000_000 == 0 {
                    "%Y-%m-%d %H:%M:%S"
                } else {
                    "%Y-%m-%d %H:%M:%S%.6f"
                };
                Ok(Some(datetime.format(format).to_string().into_bytes()))
            }
            other => Err(format!(
                "result encoding for arrow type {other:?} is not implemented yet"
            )),
        }
    }

    /// Downcasts an Arrow array to a concrete type, mapping a mismatch to a descriptive error.
    fn downcast<T: 'static>(array: &dyn Array) -> Result<&T, String> {
        array.as_any().downcast_ref::<T>().ok_or_else(|| {
            format!(
                "arrow array did not downcast to {}",
                std::any::type_name::<T>()
            )
        })
    }
}

/// One MySQL text-protocol resultset row: a sequence of length-encoded column values.
#[derive(Default)]
struct MysqlTextRow {
    buf: Vec<u8>,
}

impl MysqlTextRow {
    /// Appends one column value as a MySQL length-encoded string, or the NULL sentinel `0xFB`.
    fn push_cell(&mut self, value: Option<&[u8]>) {
        match value {
            Some(bytes) => {
                self.push_length(bytes.len());
                self.buf.extend_from_slice(bytes);
            }
            None => self.buf.push(0xFB),
        }
    }

    /// Writes a MySQL `length-encoded integer` prefix. `0xFB` is reserved for NULL, so a one-byte
    /// length never reaches 251.
    fn push_length(&mut self, len: usize) {
        if len < 251 {
            self.buf.push(len as u8);
        } else if len < 1 << 16 {
            self.buf.push(0xFC);
            self.buf.extend_from_slice(&(len as u16).to_le_bytes());
        } else if len < 1 << 24 {
            self.buf.push(0xFD);
            self.buf.extend_from_slice(&(len as u32).to_le_bytes()[..3]);
        } else {
            self.buf.push(0xFE);
            self.buf.extend_from_slice(&(len as u64).to_le_bytes());
        }
    }

    /// Consumes the row into its encoded bytes.
    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Serializes a thrift value with the binary protocol the FE expects for BRPC attachments.
pub(crate) trait ThriftBinary {
    /// Serializes `self` to thrift binary-protocol bytes.
    fn to_binary(&self) -> Result<Vec<u8>, String>;
}

impl<T: TSerializable> ThriftBinary for T {
    fn to_binary(&self) -> Result<Vec<u8>, String> {
        // A plain growable buffer: result batches can be arbitrarily large, so a fixed-capacity
        // channel would truncate (surface as a transport error) on big results.
        let mut buffer = Vec::new();
        let mut protocol = TBinaryOutputProtocol::new(&mut buffer, true);
        self.write_to_out_protocol(&mut protocol)
            .map_err(|err| format!("failed to serialize thrift value: {err}"))?;
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::ArrayRef;
    use arrow_schema::{Field, Schema};

    #[test]
    fn encodes_length_prefixed_text_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let ids: ArrayRef = Arc::new(Int64Array::from(vec![Some(42), None]));
        let names: ArrayRef = Arc::new(StringArray::from(vec![Some("hi"), Some("x")]));
        let batch = RecordBatch::try_new(schema, vec![ids, names]).unwrap();

        let result = MysqlResultEncoder::encode(&[batch], 0).unwrap();

        assert_eq!(result.rows.len(), 2);
        // Row 0: "42" (len 2) then "hi" (len 2).
        assert_eq!(result.rows[0], vec![0x02, b'4', b'2', 0x02, b'h', b'i']);
        // Row 1: NULL int (0xFB) then "x" (len 1).
        assert_eq!(result.rows[1], vec![0xFB, 0x01, b'x']);
    }

    #[test]
    fn encodes_decimal_date_and_timestamp_as_text() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Decimal128(15, 2), true),
            Field::new("day", DataType::Date32, true),
            Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        ]));
        let decimals: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(-123456), None, None])
                .with_precision_and_scale(15, 2)
                .unwrap(),
        );
        // 10471 days = 1998-09-02.
        let days: ArrayRef = Arc::new(Date32Array::from(vec![Some(10471), Some(0), Some(0)]));
        let stamps: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            Some(904_694_400_000_000),
            Some(1_500_000),
            // Sub-millisecond fractional part exercises %.6f leading-zero padding.
            Some(1_500),
        ]));
        let batch = RecordBatch::try_new(schema, vec![decimals, days, stamps]).unwrap();

        let result = MysqlResultEncoder::encode(&[batch], 0).unwrap();

        let cells = |row: &[u8]| -> Vec<String> {
            // All test values are short, so every cell is a one-byte length prefix.
            let mut cells = Vec::new();
            let mut idx = 0;
            while idx < row.len() {
                if row[idx] == 0xFB {
                    cells.push("NULL".to_string());
                    idx += 1;
                } else {
                    let len = row[idx] as usize;
                    cells.push(String::from_utf8(row[idx + 1..idx + 1 + len].to_vec()).unwrap());
                    idx += 1 + len;
                }
            }
            cells
        };
        assert_eq!(
            cells(&result.rows[0]),
            vec!["-1234.56", "1998-09-02", "1998-09-02 00:00:00"]
        );
        assert_eq!(
            cells(&result.rows[1]),
            vec!["NULL", "1970-01-01", "1970-01-01 00:00:01.500000"]
        );
        assert_eq!(
            cells(&result.rows[2]),
            vec!["NULL", "1970-01-01", "1970-01-01 00:00:00.001500"]
        );
    }

    #[test]
    fn long_value_uses_two_byte_length_prefix() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)]));
        let long = "z".repeat(300);
        let col: ArrayRef = Arc::new(StringArray::from(vec![Some(long.as_str())]));
        let batch = RecordBatch::try_new(schema, vec![col]).unwrap();

        let result = MysqlResultEncoder::encode(&[batch], 0).unwrap();

        // 300 = 0x012C, encoded as 0xFC then little-endian u16.
        assert_eq!(&result.rows[0][..3], &[0xFC, 0x2C, 0x01]);
        assert_eq!(result.rows[0].len(), 3 + 300);
    }
}
