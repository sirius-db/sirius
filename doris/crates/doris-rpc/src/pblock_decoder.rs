//! PBlock decoder: decode Doris PBlock binary format into per-column data.
//!
//! PBlock.column_values format (after decompression):
//! 1. `uint32 num_rows` (first 4 bytes LE)
//! 2. Per column (in column_metas order):
//!    - If nullable: N bytes null map (0=not-null, 1=null)
//!    - Fixed-width: N * sizeof(type) bytes LE
//!    - STRING: `uint64 data_len` + `data[data_len]` + `uint64 offsets[N+1]`

use doris_proto::doris::p_generic_type::TypeId;
use doris_proto::doris::segment_v2::CompressionTypePb;
use doris_proto::doris::{PBlock, PColumnMeta};

/// Decoded column data from a PBlock.
pub struct DecodedColumn {
    /// PGenericType::TypeId value.
    pub type_id: i32,
    /// Column name.
    pub col_name: String,
    /// Whether this column is nullable.
    pub is_nullable: bool,
    /// Raw column data bytes.
    pub data: Vec<u8>,
    /// Null mask (one byte per row: 0=not-null, 1=null). Only if nullable.
    pub null_mask: Option<Vec<u8>>,
    /// String offsets (N+1 entries for N rows). Only for STRING/BYTES types.
    pub offsets: Option<Vec<u64>>,
    /// Decimal precision (if applicable).
    pub precision: u32,
    /// Decimal scale (if applicable).
    pub scale: u32,
}

/// Result of decoding one or more PBlocks.
pub struct DecodedColumns {
    pub columns: Vec<DecodedColumn>,
    pub num_rows: u32,
}

/// Return the byte width of a fixed-width PGenericType, or 0 for variable-width.
fn type_byte_width(type_id: i32) -> usize {
    match type_id {
        x if x == TypeId::Boolean as i32 => 1,
        x if x == TypeId::Int8 as i32 || x == TypeId::Uint8 as i32 => 1,
        x if x == TypeId::Int16 as i32 || x == TypeId::Uint16 as i32 => 2,
        x if x == TypeId::Int32 as i32 || x == TypeId::Uint32 as i32 => 4,
        x if x == TypeId::Int64 as i32 || x == TypeId::Uint64 as i32 => 8,
        x if x == TypeId::Int128 as i32 || x == TypeId::Uint128 as i32 => 16,
        x if x == TypeId::Float as i32 => 4,
        x if x == TypeId::Double as i32 => 8,
        x if x == TypeId::Datev2 as i32 => 4,       // uint32
        x if x == TypeId::Datetimev2 as i32 => 8,    // uint64
        x if x == TypeId::Date as i32 => 8,           // int64 (VecDateTimeValue)
        x if x == TypeId::Datetime as i32 => 8,       // int64
        x if x == TypeId::Decimal32 as i32 => 4,
        x if x == TypeId::Decimal64 as i32 => 8,
        x if x == TypeId::Decimal128 as i32 => 16,
        x if x == TypeId::Decimal128i as i32 => 16,
        x if x == TypeId::Decimal256 as i32 => 32,
        x if x == TypeId::Ipv4 as i32 => 4,
        x if x == TypeId::Ipv6 as i32 => 16,
        // Variable-width types
        x if x == TypeId::String as i32 => 0,
        x if x == TypeId::Bytes as i32 => 0,
        x if x == TypeId::Jsonb as i32 => 0,
        x if x == TypeId::Variant as i32 => 0,
        _ => 0,
    }
}

fn is_string_type(type_id: i32) -> bool {
    type_id == TypeId::String as i32
        || type_id == TypeId::Bytes as i32
        || type_id == TypeId::Jsonb as i32
        || type_id == TypeId::Variant as i32
}

/// Decompress PBlock column_values based on compression_type.
fn decompress(
    data: &[u8],
    compression_type: i32,
    uncompressed_size: usize,
) -> Result<Vec<u8>, String> {
    match compression_type {
        x if x == CompressionTypePb::NoCompression as i32 => Ok(data.to_vec()),
        x if x == CompressionTypePb::Snappy as i32 => {
            let mut out = vec![0u8; uncompressed_size];
            snap::raw::Decoder::new()
                .decompress(data, &mut out)
                .map_err(|e| format!("snappy decompress: {e}"))?;
            Ok(out)
        }
        x if x == CompressionTypePb::Lz4 as i32 || x == CompressionTypePb::Lz4f as i32 => {
            lz4_flex::decompress(data, uncompressed_size)
                .map_err(|e| format!("lz4 decompress: {e}"))
        }
        x if x == CompressionTypePb::Zstd as i32 => {
            let mut out = Vec::with_capacity(uncompressed_size);
            let cursor = std::io::Cursor::new(data);
            let mut decoder = zstd::Decoder::new(cursor)
                .map_err(|e| format!("zstd decoder init: {e}"))?;
            std::io::Read::read_to_end(&mut decoder, &mut out)
                .map_err(|e| format!("zstd decompress: {e}"))?;
            Ok(out)
        }
        other => Err(format!("unsupported compression type: {other}")),
    }
}

/// Decode a single PBlock into column data.
fn decode_single_block(block: &PBlock) -> Result<(Vec<DecodedColumn>, u32), String> {
    let raw_bytes = block
        .column_values
        .as_ref()
        .ok_or("PBlock missing column_values")?;

    // Decompress if needed.
    let data = if block.compressed.unwrap_or(false) {
        let uncompressed_size = block.uncompressed_size.unwrap_or(0) as usize;
        let compression = block.compression_type.unwrap_or(CompressionTypePb::Snappy as i32);
        decompress(raw_bytes, compression, uncompressed_size)?
    } else {
        raw_bytes.clone()
    };

    if data.len() < 4 {
        return Err("PBlock column_values too short for num_rows".to_string());
    }

    // First 4 bytes: num_rows (LE uint32).
    let num_rows = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let mut offset = 4usize;

    let mut columns = Vec::with_capacity(block.column_metas.len());

    for meta in &block.column_metas {
        let type_id = meta.r#type.unwrap_or(TypeId::Unknown as i32);
        let is_nullable = meta.is_nullable.unwrap_or(false);
        let col_name = meta.name.clone().unwrap_or_default();
        let precision = meta
            .decimal_param
            .as_ref()
            .and_then(|d| d.precision)
            .unwrap_or(0);
        let scale = meta
            .decimal_param
            .as_ref()
            .and_then(|d| d.scale)
            .unwrap_or(0);

        // Read null mask if nullable.
        let null_mask = if is_nullable {
            let mask_size = num_rows as usize;
            if offset + mask_size > data.len() {
                return Err(format!(
                    "PBlock truncated reading null mask for column '{}': need {} bytes at offset {}, have {}",
                    col_name, mask_size, offset, data.len()
                ));
            }
            let mask = data[offset..offset + mask_size].to_vec();
            offset += mask_size;
            Some(mask)
        } else {
            None
        };

        // Read column data.
        let (col_data, col_offsets) = if is_string_type(type_id) {
            // STRING format: uint64 data_len + data[data_len] + uint64 offsets[N+1]
            if offset + 8 > data.len() {
                return Err(format!(
                    "PBlock truncated reading string data_len for column '{col_name}'"
                ));
            }
            let data_len =
                u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
            offset += 8;

            if offset + data_len > data.len() {
                return Err(format!(
                    "PBlock truncated reading string data for column '{col_name}': need {data_len} at offset {offset}, have {}",
                    data.len()
                ));
            }
            let string_data = data[offset..offset + data_len].to_vec();
            offset += data_len;

            // Read offsets array: (N+1) uint64 values.
            let offsets_count = (num_rows + 1) as usize;
            let offsets_bytes = offsets_count * 8;
            if offset + offsets_bytes > data.len() {
                return Err(format!(
                    "PBlock truncated reading string offsets for column '{col_name}'"
                ));
            }
            let mut offsets = Vec::with_capacity(offsets_count);
            for i in 0..offsets_count {
                let o = u64::from_le_bytes(
                    data[offset + i * 8..offset + i * 8 + 8].try_into().unwrap(),
                );
                offsets.push(o);
            }
            offset += offsets_bytes;

            (string_data, Some(offsets))
        } else {
            // Fixed-width column.
            let width = type_byte_width(type_id);
            if width == 0 {
                return Err(format!(
                    "unknown fixed-width size for type_id={type_id} column '{col_name}'"
                ));
            }
            let col_size = num_rows as usize * width;
            if offset + col_size > data.len() {
                return Err(format!(
                    "PBlock truncated reading column '{col_name}': need {col_size} at offset {offset}, have {}",
                    data.len()
                ));
            }
            let col_data = data[offset..offset + col_size].to_vec();
            offset += col_size;
            (col_data, None)
        };

        columns.push(DecodedColumn {
            type_id,
            col_name,
            is_nullable,
            data: col_data,
            null_mask,
            offsets: col_offsets,
            precision,
            scale,
        });
    }

    Ok((columns, num_rows))
}

/// Map a PGenericType::TypeId to a DuckDB SQL type name.
pub fn type_id_to_sql(type_id: i32, precision: u32, scale: u32) -> String {
    match type_id {
        x if x == TypeId::Boolean as i32 => "BOOLEAN".to_string(),
        x if x == TypeId::Int8 as i32 => "TINYINT".to_string(),
        x if x == TypeId::Int16 as i32 => "SMALLINT".to_string(),
        x if x == TypeId::Int32 as i32 => "INTEGER".to_string(),
        x if x == TypeId::Int64 as i32 => "BIGINT".to_string(),
        x if x == TypeId::Int128 as i32 => "HUGEINT".to_string(),
        x if x == TypeId::Uint8 as i32 => "UTINYINT".to_string(),
        x if x == TypeId::Uint16 as i32 => "USMALLINT".to_string(),
        x if x == TypeId::Uint32 as i32 => "UINTEGER".to_string(),
        x if x == TypeId::Uint64 as i32 => "UBIGINT".to_string(),
        x if x == TypeId::Float as i32 => "FLOAT".to_string(),
        x if x == TypeId::Double as i32 => "DOUBLE".to_string(),
        x if x == TypeId::String as i32 => "VARCHAR".to_string(),
        x if x == TypeId::Bytes as i32 => "BLOB".to_string(),
        x if x == TypeId::Datev2 as i32 || x == TypeId::Date as i32 => "DATE".to_string(),
        x if x == TypeId::Datetimev2 as i32 || x == TypeId::Datetime as i32 => "TIMESTAMP".to_string(),
        x if x == TypeId::Decimal32 as i32
            || x == TypeId::Decimal64 as i32
            || x == TypeId::Decimal128 as i32
            || x == TypeId::Decimal128i as i32
            || x == TypeId::Decimal256 as i32 =>
        {
            format!("DECIMAL({}, {})", precision, scale)
        }
        x if x == TypeId::Jsonb as i32 => "VARCHAR".to_string(),
        _ => "VARCHAR".to_string(),
    }
}

/// Convert a decoded column's data to SQL literal strings (one per row).
///
/// Returns a Vec of SQL literal strings suitable for VALUES insertion.
pub fn column_to_sql_values(col: &DecodedColumn, num_rows: u32) -> Vec<String> {
    let mut values = Vec::with_capacity(num_rows as usize);

    for row in 0..num_rows as usize {
        // Check null.
        if let Some(mask) = &col.null_mask {
            if row < mask.len() && mask[row] != 0 {
                values.push("NULL".to_string());
                continue;
            }
        }

        let val = if is_string_type(col.type_id) {
            // String: use offsets to extract the substring.
            if let Some(offsets) = &col.offsets {
                let start = offsets[row] as usize;
                let end = offsets[row + 1] as usize;
                let s = String::from_utf8_lossy(&col.data[start..end]);
                // Escape single quotes for SQL.
                format!("'{}'", s.replace('\'', "''"))
            } else {
                "''".to_string()
            }
        } else if col.type_id == TypeId::Boolean as i32 {
            let v = col.data[row];
            if v != 0 { "TRUE".to_string() } else { "FALSE".to_string() }
        } else {
            // Fixed-width numeric: read LE bytes and format.
            let width = type_byte_width(col.type_id);
            let offset = row * width;
            if offset + width > col.data.len() {
                "NULL".to_string()
            } else {
                let bytes = &col.data[offset..offset + width];
                format_numeric_value(col.type_id, bytes, col.precision, col.scale)
            }
        };
        values.push(val);
    }

    values
}

fn format_numeric_value(type_id: i32, bytes: &[u8], _precision: u32, scale: u32) -> String {
    match type_id {
        x if x == TypeId::Int8 as i32 || x == TypeId::Uint8 as i32 => {
            format!("{}", bytes[0] as i8)
        }
        x if x == TypeId::Int16 as i32 || x == TypeId::Uint16 as i32 => {
            let v = i16::from_le_bytes(bytes[..2].try_into().unwrap());
            format!("{}", v)
        }
        x if x == TypeId::Int32 as i32 || x == TypeId::Uint32 as i32 => {
            let v = i32::from_le_bytes(bytes[..4].try_into().unwrap());
            format!("{}", v)
        }
        x if x == TypeId::Int64 as i32 || x == TypeId::Uint64 as i32 => {
            let v = i64::from_le_bytes(bytes[..8].try_into().unwrap());
            format!("{}", v)
        }
        x if x == TypeId::Float as i32 => {
            let v = f32::from_le_bytes(bytes[..4].try_into().unwrap());
            format!("{}", v)
        }
        x if x == TypeId::Double as i32 => {
            let v = f64::from_le_bytes(bytes[..8].try_into().unwrap());
            format!("{}", v)
        }
        x if x == TypeId::Datev2 as i32 => {
            // DATEV2: uint32 packed date (year<<9 | month<<5 | day)
            let v = u32::from_le_bytes(bytes[..4].try_into().unwrap());
            let year = v >> 9;
            let month = (v >> 5) & 0xF;
            let day = v & 0x1F;
            format!("'{:04}-{:02}-{:02}'", year, month, day)
        }
        x if x == TypeId::Datetimev2 as i32 => {
            // DATETIMEV2: uint64 packed datetime
            let v = u64::from_le_bytes(bytes[..8].try_into().unwrap());
            let microseconds = v & ((1 << 20) - 1);
            let second = (v >> 20) & 0x3F;
            let minute = (v >> 26) & 0x3F;
            let hour = (v >> 32) & 0x1F;
            let day = (v >> 37) & 0x1F;
            let month = (v >> 42) & 0xF;
            let year = v >> 46;
            format!(
                "'{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:06}'",
                year, month, day, hour, minute, second, microseconds
            )
        }
        x if x == TypeId::Decimal32 as i32 => {
            let v = i32::from_le_bytes(bytes[..4].try_into().unwrap());
            format_decimal(v as i128, scale)
        }
        x if x == TypeId::Decimal64 as i32 => {
            let v = i64::from_le_bytes(bytes[..8].try_into().unwrap());
            format_decimal(v as i128, scale)
        }
        x if x == TypeId::Decimal128 as i32 || x == TypeId::Decimal128i as i32 => {
            let v = i128::from_le_bytes(bytes[..16].try_into().unwrap());
            format_decimal(v, scale)
        }
        _ => {
            // Fallback: format as integer.
            if bytes.len() >= 8 {
                let v = i64::from_le_bytes(bytes[..8].try_into().unwrap());
                format!("{}", v)
            } else if bytes.len() >= 4 {
                let v = i32::from_le_bytes(bytes[..4].try_into().unwrap());
                format!("{}", v)
            } else {
                "0".to_string()
            }
        }
    }
}

fn format_decimal(value: i128, scale: u32) -> String {
    if scale == 0 {
        return format!("{}", value);
    }
    let divisor = 10i128.pow(scale);
    let integer_part = value / divisor;
    let frac_part = (value % divisor).unsigned_abs();
    format!("{}.{:0>width$}", integer_part, frac_part, width = scale as usize)
}

/// Extract column metadata (name + SQL type) from a PBlock's column_metas.
pub fn extract_column_info(metas: &[PColumnMeta]) -> Vec<(String, String)> {
    metas
        .iter()
        .map(|m| {
            let name = m.name.clone().unwrap_or_default();
            let type_id = m.r#type.unwrap_or(TypeId::Unknown as i32);
            let precision = m.decimal_param.as_ref().and_then(|d| d.precision).unwrap_or(0);
            let scale = m.decimal_param.as_ref().and_then(|d| d.scale).unwrap_or(0);
            let sql_type = type_id_to_sql(type_id, precision, scale);
            (name, sql_type)
        })
        .collect()
}

/// Decode multiple PBlocks into a single DecodedColumns, concatenating row data.
pub fn decode_pblocks(blocks: &[PBlock]) -> Result<DecodedColumns, String> {
    if blocks.is_empty() {
        return Ok(DecodedColumns {
            columns: vec![],
            num_rows: 0,
        });
    }

    // Decode first block to establish column structure.
    let (mut merged_columns, mut total_rows) = decode_single_block(&blocks[0])?;

    // Decode and append subsequent blocks.
    for block in &blocks[1..] {
        let (block_columns, block_rows) = decode_single_block(block)?;
        if block_columns.len() != merged_columns.len() {
            return Err(format!(
                "PBlock column count mismatch: expected {}, got {}",
                merged_columns.len(),
                block_columns.len()
            ));
        }

        for (merged, new) in merged_columns.iter_mut().zip(block_columns.into_iter()) {
            merged.data.extend_from_slice(&new.data);
            if let (Some(mask), Some(new_mask)) = (&mut merged.null_mask, &new.null_mask) {
                mask.extend_from_slice(new_mask);
            }
            if let (Some(offsets), Some(new_offsets)) = (&mut merged.offsets, &new.offsets) {
                // Offset the new offsets by the current data length.
                let base = *offsets.last().unwrap_or(&0);
                // Skip the first offset of the new block (it's 0, redundant with our last).
                for &o in &new_offsets[1..] {
                    offsets.push(base + o);
                }
            }
        }
        total_rows += block_rows;
    }

    Ok(DecodedColumns {
        columns: merged_columns,
        num_rows: total_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use doris_proto::doris::p_column_meta::Decimal;

    /// Build a synthetic PBlock with the given column data.
    fn make_pblock(
        column_metas: Vec<PColumnMeta>,
        num_rows: u32,
        column_bytes: Vec<u8>,
    ) -> PBlock {
        // Prepend num_rows as LE u32.
        let mut data = num_rows.to_le_bytes().to_vec();
        data.extend_from_slice(&column_bytes);

        PBlock {
            column_metas,
            column_values: Some(data),
            compressed: Some(false),
            uncompressed_size: None,
            compression_type: None,
            be_exec_version: None,
        }
    }

    fn int32_meta(name: &str) -> PColumnMeta {
        PColumnMeta {
            name: Some(name.to_string()),
            r#type: Some(TypeId::Int32 as i32),
            is_nullable: Some(false),
            decimal_param: None,
            children: vec![],
            result_is_nullable: None,
            function_name: None,
            be_exec_version: None,
            column_path: None,
            variant_max_subcolumns_count: None,
        }
    }

    fn nullable_int64_meta(name: &str) -> PColumnMeta {
        PColumnMeta {
            name: Some(name.to_string()),
            r#type: Some(TypeId::Int64 as i32),
            is_nullable: Some(true),
            decimal_param: None,
            children: vec![],
            result_is_nullable: None,
            function_name: None,
            be_exec_version: None,
            column_path: None,
            variant_max_subcolumns_count: None,
        }
    }

    fn string_meta(name: &str) -> PColumnMeta {
        PColumnMeta {
            name: Some(name.to_string()),
            r#type: Some(TypeId::String as i32),
            is_nullable: Some(false),
            decimal_param: None,
            children: vec![],
            result_is_nullable: None,
            function_name: None,
            be_exec_version: None,
            column_path: None,
            variant_max_subcolumns_count: None,
        }
    }

    #[test]
    fn test_decode_single_int32_column() {
        // 3 rows of INT32: [10, 20, 30]
        let mut col_bytes = Vec::new();
        col_bytes.extend_from_slice(&10i32.to_le_bytes());
        col_bytes.extend_from_slice(&20i32.to_le_bytes());
        col_bytes.extend_from_slice(&30i32.to_le_bytes());

        let block = make_pblock(vec![int32_meta("id")], 3, col_bytes);
        let decoded = decode_pblocks(&[block]).unwrap();

        assert_eq!(decoded.num_rows, 3);
        assert_eq!(decoded.columns.len(), 1);
        assert_eq!(decoded.columns[0].col_name, "id");
        assert_eq!(decoded.columns[0].type_id, TypeId::Int32 as i32);
        assert!(decoded.columns[0].null_mask.is_none());

        // Verify raw data bytes.
        let data = &decoded.columns[0].data;
        assert_eq!(data.len(), 12); // 3 * 4 bytes
        assert_eq!(i32::from_le_bytes(data[0..4].try_into().unwrap()), 10);
        assert_eq!(i32::from_le_bytes(data[4..8].try_into().unwrap()), 20);
        assert_eq!(i32::from_le_bytes(data[8..12].try_into().unwrap()), 30);
    }

    #[test]
    fn test_decode_nullable_int64_column() {
        // 3 rows of nullable INT64: [100, NULL, 300]
        let mut col_bytes = Vec::new();
        // Null mask: [0, 1, 0] (row 1 is null)
        col_bytes.extend_from_slice(&[0u8, 1, 0]);
        // Data: 3 * 8 bytes (null rows still have placeholder data)
        col_bytes.extend_from_slice(&100i64.to_le_bytes());
        col_bytes.extend_from_slice(&0i64.to_le_bytes()); // placeholder for NULL
        col_bytes.extend_from_slice(&300i64.to_le_bytes());

        let block = make_pblock(vec![nullable_int64_meta("value")], 3, col_bytes);
        let decoded = decode_pblocks(&[block]).unwrap();

        assert_eq!(decoded.num_rows, 3);
        let col = &decoded.columns[0];
        assert_eq!(col.col_name, "value");
        assert!(col.is_nullable);

        let mask = col.null_mask.as_ref().unwrap();
        assert_eq!(mask, &[0, 1, 0]);

        assert_eq!(i64::from_le_bytes(col.data[0..8].try_into().unwrap()), 100);
        assert_eq!(i64::from_le_bytes(col.data[16..24].try_into().unwrap()), 300);
    }

    #[test]
    fn test_decode_string_column() {
        // 2 rows of STRING: ["hello", "world"]
        let string_data = b"helloworld";
        let offsets: Vec<u64> = vec![0, 5, 10]; // 3 offsets for 2 rows

        let mut col_bytes = Vec::new();
        // data_len as uint64
        col_bytes.extend_from_slice(&(string_data.len() as u64).to_le_bytes());
        // string data
        col_bytes.extend_from_slice(string_data);
        // offsets: 3 x uint64
        for &o in &offsets {
            col_bytes.extend_from_slice(&o.to_le_bytes());
        }

        let block = make_pblock(vec![string_meta("name")], 2, col_bytes);
        let decoded = decode_pblocks(&[block]).unwrap();

        assert_eq!(decoded.num_rows, 2);
        let col = &decoded.columns[0];
        assert_eq!(col.col_name, "name");
        assert!(col.offsets.is_some());

        let offsets = col.offsets.as_ref().unwrap();
        assert_eq!(offsets, &[0, 5, 10]);
        assert_eq!(&col.data[0..5], b"hello");
        assert_eq!(&col.data[5..10], b"world");
    }

    #[test]
    fn test_decode_multi_column_block() {
        // 2 rows: INT32 "id" + STRING "name"
        let mut col_bytes = Vec::new();

        // Column 0: INT32 [42, 99]
        col_bytes.extend_from_slice(&42i32.to_le_bytes());
        col_bytes.extend_from_slice(&99i32.to_le_bytes());

        // Column 1: STRING ["foo", "bar"]
        let string_data = b"foobar";
        let offsets: Vec<u64> = vec![0, 3, 6];
        col_bytes.extend_from_slice(&(string_data.len() as u64).to_le_bytes());
        col_bytes.extend_from_slice(string_data);
        for &o in &offsets {
            col_bytes.extend_from_slice(&o.to_le_bytes());
        }

        let block = make_pblock(
            vec![int32_meta("id"), string_meta("name")],
            2,
            col_bytes,
        );
        let decoded = decode_pblocks(&[block]).unwrap();

        assert_eq!(decoded.num_rows, 2);
        assert_eq!(decoded.columns.len(), 2);
        assert_eq!(decoded.columns[0].col_name, "id");
        assert_eq!(decoded.columns[1].col_name, "name");

        // Verify int column.
        assert_eq!(
            i32::from_le_bytes(decoded.columns[0].data[0..4].try_into().unwrap()),
            42
        );

        // Verify string column.
        let offsets = decoded.columns[1].offsets.as_ref().unwrap();
        let data = &decoded.columns[1].data;
        let s0 = std::str::from_utf8(&data[offsets[0] as usize..offsets[1] as usize]).unwrap();
        let s1 = std::str::from_utf8(&data[offsets[1] as usize..offsets[2] as usize]).unwrap();
        assert_eq!(s0, "foo");
        assert_eq!(s1, "bar");
    }

    #[test]
    fn test_decode_multiple_blocks_concatenation() {
        // Block 1: INT32 [1, 2]
        let mut bytes1 = Vec::new();
        bytes1.extend_from_slice(&1i32.to_le_bytes());
        bytes1.extend_from_slice(&2i32.to_le_bytes());
        let block1 = make_pblock(vec![int32_meta("x")], 2, bytes1);

        // Block 2: INT32 [3, 4, 5]
        let mut bytes2 = Vec::new();
        bytes2.extend_from_slice(&3i32.to_le_bytes());
        bytes2.extend_from_slice(&4i32.to_le_bytes());
        bytes2.extend_from_slice(&5i32.to_le_bytes());
        let block2 = make_pblock(vec![int32_meta("x")], 3, bytes2);

        let decoded = decode_pblocks(&[block1, block2]).unwrap();
        assert_eq!(decoded.num_rows, 5);
        assert_eq!(decoded.columns[0].data.len(), 20); // 5 * 4 bytes

        // Verify all values.
        for (i, expected) in [1i32, 2, 3, 4, 5].iter().enumerate() {
            let offset = i * 4;
            let val = i32::from_le_bytes(
                decoded.columns[0].data[offset..offset + 4].try_into().unwrap(),
            );
            assert_eq!(val, *expected, "row {} mismatch", i);
        }
    }

    #[test]
    fn test_column_to_sql_values_int32() {
        let col = DecodedColumn {
            type_id: TypeId::Int32 as i32,
            col_name: "x".to_string(),
            is_nullable: false,
            data: {
                let mut d = Vec::new();
                d.extend_from_slice(&42i32.to_le_bytes());
                d.extend_from_slice(&(-7i32).to_le_bytes());
                d
            },
            null_mask: None,
            offsets: None,
            precision: 0,
            scale: 0,
        };

        let vals = column_to_sql_values(&col, 2);
        assert_eq!(vals, vec!["42", "-7"]);
    }

    #[test]
    fn test_column_to_sql_values_nullable() {
        let col = DecodedColumn {
            type_id: TypeId::Int64 as i32,
            col_name: "v".to_string(),
            is_nullable: true,
            data: {
                let mut d = Vec::new();
                d.extend_from_slice(&100i64.to_le_bytes());
                d.extend_from_slice(&0i64.to_le_bytes());
                d
            },
            null_mask: Some(vec![0, 1]),
            offsets: None,
            precision: 0,
            scale: 0,
        };

        let vals = column_to_sql_values(&col, 2);
        assert_eq!(vals, vec!["100", "NULL"]);
    }

    #[test]
    fn test_column_to_sql_values_string() {
        let col = DecodedColumn {
            type_id: TypeId::String as i32,
            col_name: "s".to_string(),
            is_nullable: false,
            data: b"helloworld".to_vec(),
            null_mask: None,
            offsets: Some(vec![0, 5, 10]),
            precision: 0,
            scale: 0,
        };

        let vals = column_to_sql_values(&col, 2);
        assert_eq!(vals, vec!["'hello'", "'world'"]);
    }

    #[test]
    fn test_column_to_sql_values_string_with_quotes() {
        let col = DecodedColumn {
            type_id: TypeId::String as i32,
            col_name: "s".to_string(),
            is_nullable: false,
            data: b"it's".to_vec(),
            null_mask: None,
            offsets: Some(vec![0, 4]),
            precision: 0,
            scale: 0,
        };

        let vals = column_to_sql_values(&col, 1);
        assert_eq!(vals, vec!["'it''s'"]);
    }

    #[test]
    fn test_type_id_to_sql() {
        assert_eq!(type_id_to_sql(TypeId::Int32 as i32, 0, 0), "INTEGER");
        assert_eq!(type_id_to_sql(TypeId::String as i32, 0, 0), "VARCHAR");
        assert_eq!(type_id_to_sql(TypeId::Decimal64 as i32, 18, 6), "DECIMAL(18, 6)");
        assert_eq!(type_id_to_sql(TypeId::Boolean as i32, 0, 0), "BOOLEAN");
    }

    #[test]
    fn test_extract_column_info() {
        let metas = vec![
            int32_meta("id"),
            PColumnMeta {
                name: Some("price".to_string()),
                r#type: Some(TypeId::Decimal64 as i32),
                is_nullable: Some(true),
                decimal_param: Some(Decimal {
                    precision: Some(10),
                    scale: Some(2),
                }),
                children: vec![],
                result_is_nullable: None,
                function_name: None,
                be_exec_version: None,
                column_path: None,
                variant_max_subcolumns_count: None,
            },
        ];

        let info = extract_column_info(&metas);
        assert_eq!(info.len(), 2);
        assert_eq!(info[0], ("id".to_string(), "INTEGER".to_string()));
        assert_eq!(info[1], ("price".to_string(), "DECIMAL(10, 2)".to_string()));
    }

    #[test]
    fn test_decode_empty_blocks() {
        let decoded = decode_pblocks(&[]).unwrap();
        assert_eq!(decoded.num_rows, 0);
        assert!(decoded.columns.is_empty());
    }

    #[test]
    fn test_snappy_compressed_block() {
        // Build uncompressed payload: num_rows(2) + 2 INT32 values.
        let mut uncompressed = Vec::new();
        uncompressed.extend_from_slice(&2u32.to_le_bytes());
        uncompressed.extend_from_slice(&42i32.to_le_bytes());
        uncompressed.extend_from_slice(&99i32.to_le_bytes());

        // Compress with snappy.
        let mut encoder = snap::raw::Encoder::new();
        let compressed = encoder.compress_vec(&uncompressed).unwrap();

        let block = PBlock {
            column_metas: vec![int32_meta("val")],
            column_values: Some(compressed),
            compressed: Some(true),
            uncompressed_size: Some(uncompressed.len() as i64),
            compression_type: Some(CompressionTypePb::Snappy as i32),
            be_exec_version: None,
        };

        let decoded = decode_pblocks(&[block]).unwrap();
        assert_eq!(decoded.num_rows, 2);
        assert_eq!(
            i32::from_le_bytes(decoded.columns[0].data[0..4].try_into().unwrap()),
            42
        );
        assert_eq!(
            i32::from_le_bytes(decoded.columns[0].data[4..8].try_into().unwrap()),
            99
        );
    }

    #[test]
    fn test_format_decimal() {
        assert_eq!(format_decimal(12345, 2), "123.45");
        assert_eq!(format_decimal(-12345, 2), "-123.45");
        assert_eq!(format_decimal(100, 0), "100");
        assert_eq!(format_decimal(5, 3), "0.005");
    }
}
