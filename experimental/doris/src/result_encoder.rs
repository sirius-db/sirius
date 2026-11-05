//! Encodes executed-fragment output into the Doris result wire format.
//!
//! Doris delivers SELECT results to the FE as a `TResultBatch` (thrift **binary**, inside
//! `PFetchDataResult.row_batch`) whose `rows` are MySQL text-protocol resultset rows: each
//! column value is a length-encoded string, NULL is the single byte `0xFB`. The FE forwards
//! those row bodies straight to the MySQL client.

use arrow_array::temporal_conversions::{as_date, as_datetime};
use arrow_array::types::{
    Date32Type, TimestampMicrosecondType, TimestampMillisecondType, TimestampSecondType,
};
use arrow_array::{
    Array, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int8Array,
    Int16Array, Int32Array, Int64Array, LargeStringArray, RecordBatch, StringArray,
    StringViewArray, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampSecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use doris_thrift::data::TResultBatch;
use thrift::protocol::{TBinaryOutputProtocol, TSerializable};

const MYSQL_NULL_MARKER: u8 = 0xFB;
const LENENC_2_BYTE_PREFIX: u8 = 0xFC;
const LENENC_3_BYTE_PREFIX: u8 = 0xFD;
const LENENC_8_BYTE_PREFIX: u8 = 0xFE;
const MYSQL_INLINE_LENGTH_LIMIT: usize = 251;
const LENENC_2_BYTE_LIMIT: usize = 1 << 16;
const LENENC_3_BYTE_LIMIT: usize = 1 << 24;
const MILLIS_PER_SECOND: i64 = 1_000;
const MICROS_PER_SECOND: i64 = 1_000_000;

/// Encodes Arrow result batches into a Doris `TResultBatch` of MySQL text rows.
#[derive(Default)]
pub(crate) struct MysqlResultEncoder {
    rows: Vec<Vec<u8>>,
}

impl MysqlResultEncoder {
    /// Encodes `batches` into a `TResultBatch` tagged with `packet_seq`.
    #[cfg(test)]
    pub(crate) fn encode(batches: &[RecordBatch], packet_seq: i64) -> Result<TResultBatch, String> {
        let mut encoder = Self::default();
        for batch in batches {
            encoder.add_batch(batch)?;
        }
        Ok(encoder.into_result_batch(packet_seq))
    }

    /// Encodes rows into bounded thrift batches and emits each as soon as it fills. A single
    /// row larger than the limit is refused because the FE cannot receive that packet.
    pub(crate) fn encode_bounded(
        batch: &RecordBatch,
        max_bytes: usize,
        mut emit: impl FnMut(TResultBatch) -> Result<(), String>,
    ) -> Result<(), String> {
        let mut encoder = Self::default();
        let mut bytes = 0usize;
        let renderers = batch
            .columns()
            .iter()
            .map(|column| CellRenderer::new(column.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for row in 0..batch.num_rows() {
            let mut encoded = MysqlTextRow {
                buf: Vec::with_capacity(batch.num_columns() * 16),
            };
            for renderer in &renderers {
                renderer.write(row, &mut encoded)?;
            }
            let row_bytes = encoded.buf.len().saturating_add(4); // thrift binary row-length prefix
            if row_bytes > max_bytes {
                return Err(format!(
                    "encoded result row exceeds the {max_bytes}-byte packet limit"
                ));
            }
            if bytes + row_bytes > max_bytes && !encoder.rows.is_empty() {
                emit(std::mem::take(&mut encoder).into_result_batch(0))?;
                bytes = 0;
            }
            bytes += row_bytes;
            encoder.rows.push(encoded.into_bytes());
        }
        if !encoder.rows.is_empty() {
            emit(encoder.into_result_batch(0))?;
        }
        Ok(())
    }

    /// Encodes every row of `batch` as a MySQL text row.
    #[cfg(test)]
    fn add_batch(&mut self, batch: &RecordBatch) -> Result<(), String> {
        let renderers = batch
            .columns()
            .iter()
            .map(|column| CellRenderer::new(column.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        for row in 0..batch.num_rows() {
            let mut encoded = MysqlTextRow::default();
            for renderer in &renderers {
                renderer.write(row, &mut encoded)?;
            }
            self.rows.push(encoded.into_bytes());
        }
        Ok(())
    }

    /// Consumes the accumulated rows into a `TResultBatch` (`is_compressed = false`: rows are
    /// never snappy-compressed here).
    fn into_result_batch(self, packet_seq: i64) -> TResultBatch {
        TResultBatch::new(self.rows, false, packet_seq, None)
    }
}

/// A column downcast once per Arrow batch, then rendered directly into each MySQL row.
enum CellKind<'a> {
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
    Utf8View(&'a StringViewArray),
    Boolean(&'a BooleanArray),
    Int8(&'a Int8Array),
    Int16(&'a Int16Array),
    Int32(&'a Int32Array),
    Int64(&'a Int64Array),
    Float32(&'a Float32Array),
    Float64(&'a Float64Array),
    Decimal128(&'a Decimal128Array),
    Date32(&'a Date32Array),
    TimestampSecond(&'a TimestampSecondArray),
    TimestampMillisecond(&'a TimestampMillisecondArray),
    TimestampMicrosecond(&'a TimestampMicrosecondArray),
}

struct CellRenderer<'a> {
    array: &'a dyn Array,
    kind: CellKind<'a>,
}

impl<'a> CellRenderer<'a> {
    fn new(array: &'a dyn Array) -> Result<Self, String> {
        macro_rules! downcast {
            ($ty:ty) => {
                array.as_any().downcast_ref::<$ty>().ok_or_else(|| {
                    format!(
                        "arrow array did not downcast to {}",
                        std::any::type_name::<$ty>()
                    )
                })?
            };
        }
        let kind = match array.data_type() {
            DataType::Utf8 => CellKind::Utf8(downcast!(StringArray)),
            DataType::LargeUtf8 => CellKind::LargeUtf8(downcast!(LargeStringArray)),
            DataType::Utf8View => CellKind::Utf8View(downcast!(StringViewArray)),
            DataType::Boolean => CellKind::Boolean(downcast!(BooleanArray)),
            DataType::Int8 => CellKind::Int8(downcast!(Int8Array)),
            DataType::Int16 => CellKind::Int16(downcast!(Int16Array)),
            DataType::Int32 => CellKind::Int32(downcast!(Int32Array)),
            DataType::Int64 => CellKind::Int64(downcast!(Int64Array)),
            DataType::Float32 => CellKind::Float32(downcast!(Float32Array)),
            DataType::Float64 => CellKind::Float64(downcast!(Float64Array)),
            DataType::Decimal128(_, _) => CellKind::Decimal128(downcast!(Decimal128Array)),
            DataType::Date32 => CellKind::Date32(downcast!(Date32Array)),
            DataType::Timestamp(TimeUnit::Second, None) => {
                CellKind::TimestampSecond(downcast!(TimestampSecondArray))
            }
            DataType::Timestamp(TimeUnit::Millisecond, None) => {
                CellKind::TimestampMillisecond(downcast!(TimestampMillisecondArray))
            }
            DataType::Timestamp(TimeUnit::Microsecond, None) => {
                CellKind::TimestampMicrosecond(downcast!(TimestampMicrosecondArray))
            }
            other => {
                return Err(format!(
                    "result encoding for arrow type {other:?} is not implemented yet"
                ));
            }
        };
        Ok(Self { array, kind })
    }

    fn write(&self, row: usize, output: &mut MysqlTextRow) -> Result<(), String> {
        if self.array.is_null(row) {
            output.push_cell(None);
            return Ok(());
        }
        macro_rules! integer {
            ($typed:expr) => {{
                let mut buffer = itoa::Buffer::new();
                output.push_cell(Some(buffer.format($typed.value(row)).as_bytes()));
            }};
        }
        macro_rules! floating {
            ($typed:expr) => {{
                let mut buffer = ryu::Buffer::new();
                output.push_cell(Some(buffer.format($typed.value(row)).as_bytes()));
            }};
        }
        macro_rules! timestamp {
            ($typed:expr, $arrow_type:ty, $units_per_second:expr, $fraction_format:expr) => {{
                let value = $typed.value(row);
                let datetime = as_datetime::<$arrow_type>(value)
                    .ok_or_else(|| format!("timestamp value {value} out of range"))?;
                let format = if value % $units_per_second == 0 {
                    "%Y-%m-%d %H:%M:%S"
                } else {
                    $fraction_format
                };
                output.push_cell(Some(datetime.format(format).to_string().as_bytes()));
            }};
        }
        match &self.kind {
            CellKind::Utf8(v) => output.push_cell(Some(v.value(row).as_bytes())),
            CellKind::LargeUtf8(v) => output.push_cell(Some(v.value(row).as_bytes())),
            CellKind::Utf8View(v) => output.push_cell(Some(v.value(row).as_bytes())),
            CellKind::Boolean(v) => output.push_cell(Some(if v.value(row) { b"1" } else { b"0" })),
            CellKind::Int8(v) => integer!(v),
            CellKind::Int16(v) => integer!(v),
            CellKind::Int32(v) => integer!(v),
            CellKind::Int64(v) => integer!(v),
            CellKind::Float32(v) => floating!(v),
            CellKind::Float64(v) => floating!(v),
            CellKind::Decimal128(v) => output.push_cell(Some(v.value_as_string(row).as_bytes())),
            CellKind::Date32(v) => {
                let date = as_date::<Date32Type>(i64::from(v.value(row)))
                    .ok_or_else(|| format!("date32 value {} out of range", v.value(row)))?;
                output.push_cell(Some(date.format("%Y-%m-%d").to_string().as_bytes()));
            }
            // Doris DATETIME is timezone-naive; a timestamp with a timezone is rejected above.
            CellKind::TimestampSecond(v) => {
                let value = v.value(row);
                let datetime = as_datetime::<TimestampSecondType>(value)
                    .ok_or_else(|| format!("timestamp value {value} out of range"))?;
                output.push_cell(Some(
                    datetime.format("%Y-%m-%d %H:%M:%S").to_string().as_bytes(),
                ));
            }
            CellKind::TimestampMillisecond(v) => timestamp!(
                v,
                TimestampMillisecondType,
                MILLIS_PER_SECOND,
                "%Y-%m-%d %H:%M:%S%.3f"
            ),
            CellKind::TimestampMicrosecond(v) => timestamp!(
                v,
                TimestampMicrosecondType,
                MICROS_PER_SECOND,
                "%Y-%m-%d %H:%M:%S%.6f"
            ),
        }
        Ok(())
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
            None => self.buf.push(MYSQL_NULL_MARKER),
        }
    }

    /// Writes a MySQL `length-encoded integer` prefix. `0xFB` is reserved for NULL, so a one-byte
    /// length never reaches 251.
    fn push_length(&mut self, len: usize) {
        if len < MYSQL_INLINE_LENGTH_LIMIT {
            self.buf.push(len as u8);
        } else if len < LENENC_2_BYTE_LIMIT {
            self.buf.push(LENENC_2_BYTE_PREFIX);
            self.buf.extend_from_slice(&(len as u16).to_le_bytes());
        } else if len < LENENC_3_BYTE_LIMIT {
            self.buf.push(LENENC_3_BYTE_PREFIX);
            self.buf.extend_from_slice(&(len as u32).to_le_bytes()[..3]);
        } else {
            self.buf.push(LENENC_8_BYTE_PREFIX);
            self.buf.extend_from_slice(&(len as u64).to_le_bytes());
        }
    }

    /// Consumes the row into its encoded bytes.
    fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

/// Serializes a thrift value with the binary protocol the FE expects for `row_batch`.
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
    fn length_prefix_boundaries_do_not_collide_with_null() {
        let mut row = MysqlTextRow::default();
        row.push_length(250);
        row.push_length(251);
        row.push_length(1 << 16);
        row.push_length(1 << 24);
        assert_eq!(&row.buf[..4], &[250, LENENC_2_BYTE_PREFIX, 251, 0]);
        assert_eq!(row.buf[4], LENENC_3_BYTE_PREFIX);
        assert_eq!(&row.buf[5..8], &[0, 0, 1]);
        assert_eq!(row.buf[8], LENENC_8_BYTE_PREFIX);
        assert_eq!(&row.buf[9..17], &(1_u64 << 24).to_le_bytes());
    }

    #[test]
    fn encodes_second_and_millisecond_timestamps() {
        let second = TimestampSecondArray::from(vec![Some(1_500_000_000)]);
        let millisecond = TimestampMillisecondArray::from(vec![Some(1_500_000_000_123)]);
        let mut second_row = MysqlTextRow::default();
        CellRenderer::new(&second)
            .unwrap()
            .write(0, &mut second_row)
            .unwrap();
        let mut millisecond_row = MysqlTextRow::default();
        CellRenderer::new(&millisecond)
            .unwrap()
            .write(0, &mut millisecond_row)
            .unwrap();
        assert_eq!(&second_row.buf[1..], b"2017-07-14 02:40:00");
        assert_eq!(&millisecond_row.buf[1..], b"2017-07-14 02:40:00.123");
    }

    #[test]
    fn bounded_encoding_emits_more_than_one_packet() {
        let batch = RecordBatch::try_from_iter(vec![(
            "text",
            Arc::new(StringArray::from(vec!["abcdef", "ghijkl"])) as ArrayRef,
        )])
        .unwrap();
        let mut packets = Vec::new();
        MysqlResultEncoder::encode_bounded(&batch, 15, |packet| {
            packets.push(packet);
            Ok(())
        })
        .unwrap();
        assert_eq!(packets.len(), 2);
        assert_eq!(packets[0].rows, vec![b"\x06abcdef".to_vec()]);
        assert_eq!(packets[1].rows, vec![b"\x06ghijkl".to_vec()]);
        assert!(MysqlResultEncoder::encode_bounded(&batch, 5, |_| Ok(())).is_err());
    }

    #[test]
    fn encodes_remaining_primitive_types_and_refuses_binary() {
        use arrow_array::BinaryArray;
        let batch = RecordBatch::try_from_iter(vec![
            ("bool", Arc::new(BooleanArray::from(vec![true])) as ArrayRef),
            ("i8", Arc::new(Int8Array::from(vec![-8])) as ArrayRef),
            ("i16", Arc::new(Int16Array::from(vec![16])) as ArrayRef),
            ("i32", Arc::new(Int32Array::from(vec![32])) as ArrayRef),
            ("f32", Arc::new(Float32Array::from(vec![1.5])) as ArrayRef),
            ("f64", Arc::new(Float64Array::from(vec![2.5])) as ArrayRef),
            (
                "view",
                Arc::new(StringViewArray::from(vec!["view"])) as ArrayRef,
            ),
        ])
        .unwrap();
        let row = &MysqlResultEncoder::encode(&[batch], 0).unwrap().rows[0];
        assert_eq!(row, b"\x011\x02-8\x0216\x0232\x031.5\x032.5\x04view");

        let unsupported = BinaryArray::from(vec![b"bytes".as_slice()]);
        let error = match CellRenderer::new(&unsupported) {
            Ok(_) => panic!("binary array should not be encoded"),
            Err(error) => error,
        };
        assert!(error.contains("not implemented"));
    }

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

    #[test]
    fn result_batch_round_trips_through_thrift_binary() {
        use thrift::protocol::TBinaryInputProtocol;
        use thrift::transport::TBufferChannel;

        let batch = TResultBatch::new(vec![vec![0x01, b'a']], false, 3, None);
        let bytes = batch.to_binary().unwrap();
        let mut channel = TBufferChannel::with_capacity(bytes.len(), 0);
        channel.set_readable_bytes(&bytes);
        let mut protocol = TBinaryInputProtocol::new(channel, true);
        let decoded = TResultBatch::read_from_in_protocol(&mut protocol).unwrap();
        assert_eq!(decoded, batch);
    }
}
