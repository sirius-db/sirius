//! PBlock decoder: decode Doris PBlock binary format into per-column data.
//!
//! Doris column_values format (after block-level decompression):
//!   No block-level header — columns are serialized consecutively.
//!
//! Per column (type->serialize):
//!   Header: const_flag(1 byte bool) + row_num(8 bytes size_t) + real_saved_num(8 bytes size_t)
//!     If const: real_saved_num = 1, data is one element (expanded on decode)
//!     Else:     real_saved_num = row_num
//!
//!   Fixed-width (DataTypeNumberBase):
//!     If mem_size <= 256: raw data (real_saved_num * sizeof(T))
//!     Else: encode_size(8 bytes) + StreamVByte encoded data
//!
//!   STRING (DataTypeString):
//!     Offsets: If N*4 <= 256: raw UInt32[N]; else encode_size(8) + StreamVByte
//!     value_len(8 bytes size_t) + chars:
//!       If value_len <= 256: raw chars
//!       Else: encode_size(8) + LZ4 compressed chars
//!     Doris ColumnString offsets are UInt32 cumulative end positions (including \0 after each string)
//!
//!   NULLABLE (DataTypeNullable):
//!     Header (own const_flag + row_num + real_saved_num)
//!     null_map: If N <= 256: raw bytes; else encode_size(8) + StreamVByte
//!     Then: nested type serialize (with its OWN header + data)

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
    /// Raw column data bytes (fixed-width: N*width; string: concatenated without \0).
    pub data: Vec<u8>,
    /// Null mask (one byte per row: 0=not-null, 1=null). Only if nullable.
    pub null_mask: Option<Vec<u8>>,
    /// String offsets (N+1 entries for N rows: [0, end_0, end_1, ...]). Only for STRING types.
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

// Per-column header: const_flag(1) + row_num(8) + real_saved_num(8)
const COL_HEADER_SIZE: usize = 1 + 8 + 8;

// Doris SERIALIZED_MEM_SIZE_LIMIT: above this, per-column data is compressed
const SERIALIZED_MEM_SIZE_LIMIT: usize = 256;

struct ColHeader {
    _is_const: bool,
    row_num: u64,
    real_saved_num: u64,
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

fn is_fixed_length_object(type_id: i32) -> bool {
    type_id == TypeId::Fixedlengthobject as i32
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

/// Read the per-column header: const_flag(1) + row_num(8) + real_saved_num(8).
fn read_col_header(data: &[u8], offset: &mut usize) -> Result<ColHeader, String> {
    if *offset + COL_HEADER_SIZE > data.len() {
        return Err(format!(
            "truncated column header at offset {}, need {} bytes, have {}",
            *offset,
            COL_HEADER_SIZE,
            data.len() - *offset
        ));
    }
    let is_const = data[*offset] != 0;
    *offset += 1;
    let row_num = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    let real_saved_num = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap());
    *offset += 8;
    Ok(ColHeader {
        _is_const: is_const,
        row_num,
        real_saved_num,
    })
}

/// Read raw bytes (for data ≤ SERIALIZED_MEM_SIZE_LIMIT).
fn read_raw_or_fail(
    data: &[u8],
    offset: &mut usize,
    mem_size: usize,
    context: &str,
) -> Result<Vec<u8>, String> {
    if mem_size <= SERIALIZED_MEM_SIZE_LIMIT {
        if *offset + mem_size > data.len() {
            return Err(format!(
                "truncated {context} at offset {}: need {mem_size}, have {}",
                *offset,
                data.len() - *offset
            ));
        }
        let result = data[*offset..*offset + mem_size].to_vec();
        *offset += mem_size;
        Ok(result)
    } else {
        // StreamVByte encoded: read encode_size(8 bytes) + encoded data
        if *offset + 8 > data.len() {
            return Err(format!(
                "truncated {context} encode_size at offset {}",
                *offset
            ));
        }
        let encode_size =
            u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap()) as usize;
        *offset += 8;
        if *offset + encode_size > data.len() {
            return Err(format!(
                "truncated {context} encoded data at offset {}: need {encode_size}, have {}",
                *offset,
                data.len() - *offset
            ));
        }
        // StreamVByte decoding: treat as array of uint32 values
        let num_u32 = (mem_size + 3) / 4; // upper_int32
        let decoded = streamvbyte_decode(&data[*offset..*offset + encode_size], num_u32)?;
        *offset += encode_size;
        // Convert Vec<u32> back to raw bytes (LE)
        let mut result = Vec::with_capacity(mem_size);
        for val in &decoded {
            result.extend_from_slice(&val.to_le_bytes());
        }
        result.truncate(mem_size);
        Ok(result)
    }
}

/// Read string chars: value_len(8) + raw or LZ4 compressed chars.
fn read_string_chars(data: &[u8], offset: &mut usize) -> Result<Vec<u8>, String> {
    if *offset + 8 > data.len() {
        return Err(format!(
            "truncated string value_len at offset {}",
            *offset
        ));
    }
    let value_len = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap()) as usize;
    *offset += 8;

    if value_len <= SERIALIZED_MEM_SIZE_LIMIT {
        if *offset + value_len > data.len() {
            return Err(format!(
                "truncated string chars at offset {}: need {value_len}, have {}",
                *offset,
                data.len() - *offset
            ));
        }
        let chars = data[*offset..*offset + value_len].to_vec();
        *offset += value_len;
        Ok(chars)
    } else {
        // LZ4 compressed: encode_size(8) + compressed data
        if *offset + 8 > data.len() {
            return Err(format!(
                "truncated string LZ4 encode_size at offset {}",
                *offset
            ));
        }
        let encode_size =
            u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap()) as usize;
        *offset += 8;
        if *offset + encode_size > data.len() {
            return Err(format!(
                "truncated string LZ4 data at offset {}: need {encode_size}, have {}",
                *offset,
                data.len() - *offset
            ));
        }
        let chars = lz4_flex::decompress(&data[*offset..*offset + encode_size], value_len)
            .map_err(|e| format!("string LZ4 decompress: {e}"))?;
        *offset += encode_size;
        Ok(chars)
    }
}

/// Decode a fixed-width column's data (after header).
fn decode_fixed_data(
    data: &[u8],
    offset: &mut usize,
    real_saved_num: u64,
    row_num: u64,
    width: usize,
    col_name: &str,
) -> Result<Vec<u8>, String> {
    let n = real_saved_num as usize;
    let mem_size = n * width;
    let raw = read_raw_or_fail(data, offset, mem_size, &format!("fixed column '{col_name}'"))?;

    // Expand const column: replicate single value to row_num
    if real_saved_num == 1 && row_num > 1 {
        let mut expanded = Vec::with_capacity(row_num as usize * width);
        for _ in 0..row_num {
            expanded.extend_from_slice(&raw[..width]);
        }
        Ok(expanded)
    } else {
        Ok(raw)
    }
}

/// Decode a string column's data (after header).
/// Returns (clean_data, clean_offsets) where offsets are N+1 start positions.
fn decode_string_data(
    data: &[u8],
    offset: &mut usize,
    real_saved_num: u64,
    row_num: u64,
    col_name: &str,
) -> Result<(Vec<u8>, Vec<u64>), String> {
    let n = real_saved_num as usize;

    // Read offsets: N * sizeof(UInt32) = N * 4 bytes
    let offsets_mem_size = n * 4;
    let raw_offsets =
        read_raw_or_fail(data, offset, offsets_mem_size, &format!("string offsets '{col_name}'"))?;

    // Parse UInt32 offsets
    let mut doris_offsets = Vec::with_capacity(n);
    for i in 0..n {
        let o = u32::from_le_bytes(raw_offsets[i * 4..i * 4 + 4].try_into().unwrap());
        doris_offsets.push(o as usize);
    }

    // Read chars: value_len(8) + raw/LZ4 chars
    let chars = read_string_chars(data, offset)?;

    // Convert Doris offsets to clean format:
    // Doris offsets[i] = end of string i in chars (AFTER \0 terminator)
    // Our format: N+1 offsets [0, len0, len0+len1, ...] into clean_data (no \0)
    let actual_n = if real_saved_num == 1 && row_num > 1 {
        // Const: expand below
        real_saved_num as usize
    } else {
        n
    };

    let mut clean_data = Vec::new();
    let mut clean_offsets = vec![0u64];
    for i in 0..actual_n {
        let start = if i == 0 { 0 } else { doris_offsets[i - 1] };
        let end_with_null = doris_offsets[i];
        // String bytes are between start and end_with_null-1 (skip the \0)
        let end = if end_with_null > start { end_with_null - 1 } else { start };
        if end > start && end <= chars.len() {
            clean_data.extend_from_slice(&chars[start..end]);
        }
        clean_offsets.push(clean_data.len() as u64);
    }

    // Expand const column: replicate single string row_num times
    if real_saved_num == 1 && row_num > 1 {
        let single_data = clean_data.clone();
        // Already have [0, single_data.len()] for one row
        for _ in 1..row_num {
            clean_data.extend_from_slice(&single_data);
            clean_offsets.push(clean_data.len() as u64);
        }
    }

    Ok((clean_data, clean_offsets))
}

/// Decode a FIXEDLENGTHOBJECT column's data (after header).
/// Format: item_size(8 bytes) + raw data (real_saved_num * item_size) or StreamVByte encoded.
fn decode_fixed_length_object(
    data: &[u8],
    offset: &mut usize,
    real_saved_num: u64,
    row_num: u64,
    col_name: &str,
) -> Result<Vec<u8>, String> {
    // Read item_size (8 bytes size_t)
    if *offset + 8 > data.len() {
        return Err(format!(
            "truncated FIXEDLENGTHOBJECT item_size for column '{col_name}'"
        ));
    }
    let item_size = u64::from_le_bytes(data[*offset..*offset + 8].try_into().unwrap()) as usize;
    *offset += 8;

    let n = real_saved_num as usize;
    let mem_size = n * item_size;
    let raw = read_raw_or_fail(
        data,
        offset,
        mem_size,
        &format!("FIXEDLENGTHOBJECT column '{col_name}'"),
    )?;

    // Expand const column
    if real_saved_num == 1 && row_num > 1 {
        let mut expanded = Vec::with_capacity(row_num as usize * item_size);
        for _ in 0..row_num {
            expanded.extend_from_slice(&raw[..item_size]);
        }
        Ok(expanded)
    } else {
        Ok(raw)
    }
}

/// Decode one non-nullable column from the data buffer.
fn decode_typed_column(
    data: &[u8],
    offset: &mut usize,
    type_id: i32,
    meta: &PColumnMeta,
) -> Result<(DecodedColumn, u64), String> {
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

    let header = read_col_header(data, offset)?;

    if is_string_type(type_id) {
        let (col_data, col_offsets) = decode_string_data(
            data,
            offset,
            header.real_saved_num,
            header.row_num,
            &col_name,
        )?;
        Ok((
            DecodedColumn {
                type_id,
                col_name,
                is_nullable: false,
                data: col_data,
                null_mask: None,
                offsets: Some(col_offsets),
                precision,
                scale,
            },
            header.row_num,
        ))
    } else if is_fixed_length_object(type_id) {
        // FIXEDLENGTHOBJECT: header → item_size(8 bytes) → raw data (N * item_size)
        let col_data = decode_fixed_length_object(
            data,
            offset,
            header.real_saved_num,
            header.row_num,
            &col_name,
        )?;
        Ok((
            DecodedColumn {
                type_id,
                col_name,
                is_nullable: false,
                data: col_data,
                null_mask: None,
                offsets: None,
                precision,
                scale,
            },
            header.row_num,
        ))
    } else {
        let width = type_byte_width(type_id);
        if width == 0 {
            return Err(format!(
                "unknown fixed-width size for type_id={type_id} column '{col_name}'"
            ));
        }
        let col_data = decode_fixed_data(
            data,
            offset,
            header.real_saved_num,
            header.row_num,
            width,
            &col_name,
        )?;
        Ok((
            DecodedColumn {
                type_id,
                col_name,
                is_nullable: false,
                data: col_data,
                null_mask: None,
                offsets: None,
                precision,
                scale,
            },
            header.row_num,
        ))
    }
}

/// Decode one nullable column from the data buffer.
/// Format: [nullable header] [null_map] [nested column with own header]
fn decode_nullable_column(
    data: &[u8],
    offset: &mut usize,
    type_id: i32,
    meta: &PColumnMeta,
) -> Result<(DecodedColumn, u64), String> {
    let col_name = meta.name.clone().unwrap_or_default();

    // Read nullable outer header
    let null_header = read_col_header(data, offset)?;

    // Read null map: real_saved_num * sizeof(bool) = real_saved_num bytes
    let null_mem_size = null_header.real_saved_num as usize;
    let raw_null_map =
        read_raw_or_fail(data, offset, null_mem_size, &format!("null map '{col_name}'"))?;

    // Expand const null map if needed
    let null_map = if null_header.real_saved_num == 1 && null_header.row_num > 1 {
        vec![raw_null_map[0]; null_header.row_num as usize]
    } else {
        raw_null_map
    };

    // Read nested column (has its own header)
    let (mut inner_col, _inner_row_num) = decode_typed_column(data, offset, type_id, meta)?;

    // Apply nullable info
    inner_col.is_nullable = true;
    inner_col.null_mask = Some(null_map);

    Ok((inner_col, null_header.row_num))
}

/// Decode a single PBlock into column data.
fn decode_single_block(block: &PBlock) -> Result<(Vec<DecodedColumn>, u32), String> {
    let raw_bytes = block
        .column_values
        .as_ref()
        .ok_or("PBlock missing column_values")?;

    // Decompress block-level compression if needed.
    let data = if block.compressed.unwrap_or(false) {
        let uncompressed_size = block.uncompressed_size.unwrap_or(0) as usize;
        let compression = block.compression_type.unwrap_or(CompressionTypePb::Snappy as i32);
        decompress(raw_bytes, compression, uncompressed_size)?
    } else {
        raw_bytes.clone()
    };

    let mut offset = 0usize;
    let mut block_num_rows = 0u32;
    let mut columns = Vec::with_capacity(block.column_metas.len());

    for (col_idx, meta) in block.column_metas.iter().enumerate() {
        let type_id = meta.r#type.unwrap_or(TypeId::Unknown as i32);
        let is_nullable = meta.is_nullable.unwrap_or(false);

        let (col, row_num) = if is_nullable {
            decode_nullable_column(&data, &mut offset, type_id, meta)?
        } else {
            decode_typed_column(&data, &mut offset, type_id, meta)?
        };

        if col_idx == 0 {
            block_num_rows = row_num as u32;
        }

        columns.push(col);
    }

    Ok((columns, block_num_rows))
}

/// Minimal StreamVByte decoder.
///
/// StreamVByte encodes N uint32 values. The format is:
///   control_bytes[(N+3)/4] + data_bytes[variable]
/// Each control byte has 4 x 2-bit entries, each encoding the byte length (1-4) of a value.
fn streamvbyte_decode(encoded: &[u8], count: usize) -> Result<Vec<u32>, String> {
    if count == 0 {
        return Ok(vec![]);
    }

    let control_len = (count + 3) / 4;
    if encoded.len() < control_len {
        return Err(format!(
            "StreamVByte: need at least {control_len} control bytes, have {}",
            encoded.len()
        ));
    }

    let control = &encoded[..control_len];
    let mut data_offset = control_len;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let ctrl_byte = control[i / 4];
        let shift = (i % 4) * 2;
        let code = (ctrl_byte >> shift) & 0x03;
        let byte_len = (code + 1) as usize; // 0→1, 1→2, 2→3, 3→4

        if data_offset + byte_len > encoded.len() {
            return Err(format!(
                "StreamVByte: data truncated at element {i}, need {byte_len} bytes at offset {data_offset}"
            ));
        }

        let mut buf = [0u8; 4];
        buf[..byte_len].copy_from_slice(&encoded[data_offset..data_offset + byte_len]);
        result.push(u32::from_le_bytes(buf));
        data_offset += byte_len;
    }

    Ok(result)
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
        x if x == TypeId::Fixedlengthobject as i32 => "BLOB".to_string(),
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

        let val = if is_fixed_length_object(col.type_id) {
            // FIXEDLENGTHOBJECT: format as hex blob literal
            // We don't know item_size here, so just dump all data for this row
            // This is a best-effort representation; DuckDB may not handle agg states
            "NULL".to_string()
        } else if is_string_type(col.type_id) {
            // String: use offsets to extract the substring.
            if let Some(offsets) = &col.offsets {
                let start = offsets[row] as usize;
                let end = offsets[row + 1] as usize;
                let bytes = &col.data[start..end];
                // Check if the data is valid UTF-8 without \0 bytes.
                // DuckDB C FFI uses null-terminated strings, so \0 in the
                // middle would truncate the value. Use hex blob for binary data.
                if bytes.contains(&0) || std::str::from_utf8(bytes).is_err() {
                    // Binary data: encode as DuckDB hex blob literal.
                    let hex: String = bytes.iter().map(|b| format!("{:02X}", b)).collect();
                    format!("'\\x{}'::BLOB", hex)
                } else {
                    let s = std::str::from_utf8(bytes).unwrap();
                    // Escape single quotes for SQL.
                    format!("'{}'", s.replace('\'', "''"))
                }
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

    /// Build a synthetic PBlock with the correct Doris serialization format.
    /// Each column's data (with per-column header) is provided in column_bytes.
    fn make_pblock(column_metas: Vec<PColumnMeta>, column_bytes: Vec<u8>) -> PBlock {
        PBlock {
            column_metas,
            column_values: Some(column_bytes),
            compressed: Some(false),
            uncompressed_size: None,
            compression_type: None,
            be_exec_version: None,
        }
    }

    /// Write a per-column header: const_flag(1) + row_num(8) + real_saved_num(8).
    fn write_col_header(buf: &mut Vec<u8>, is_const: bool, row_num: u64, real_saved_num: u64) {
        buf.push(if is_const { 1 } else { 0 });
        buf.extend_from_slice(&row_num.to_le_bytes());
        buf.extend_from_slice(&real_saved_num.to_le_bytes());
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

    /// Serialize a non-nullable INT32 column in Doris format.
    fn serialize_int32_col(buf: &mut Vec<u8>, values: &[i32]) {
        let n = values.len() as u64;
        write_col_header(buf, false, n, n);
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// Serialize a non-nullable STRING column in Doris format.
    /// Doris format: header + offsets(UInt32, cumulative with \0) + value_len(8) + chars(with \0)
    fn serialize_string_col(buf: &mut Vec<u8>, strings: &[&str]) {
        let n = strings.len() as u64;
        write_col_header(buf, false, n, n);

        // Build Doris-style chars (with \0 after each string) and offsets
        let mut chars = Vec::new();
        let mut offsets = Vec::new();
        for s in strings {
            chars.extend_from_slice(s.as_bytes());
            chars.push(0); // null terminator
            offsets.push(chars.len() as u32);
        }

        // Write offsets: N * UInt32
        for o in &offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }

        // Write value_len (size_t = 8 bytes) + chars
        buf.extend_from_slice(&(chars.len() as u64).to_le_bytes());
        buf.extend_from_slice(&chars);
    }

    /// Serialize a nullable column wrapper in Doris format.
    /// Format: [nullable header] [null_map] [inner column data with own header]
    fn serialize_nullable_wrapper(
        buf: &mut Vec<u8>,
        null_map: &[u8],
        inner_data: &[u8], // already serialized inner column (with its own header)
    ) {
        let n = null_map.len() as u64;
        write_col_header(buf, false, n, n);
        buf.extend_from_slice(null_map);
        buf.extend_from_slice(inner_data);
    }

    #[test]
    fn test_decode_single_int32_column() {
        // 3 rows of INT32: [10, 20, 30]
        let mut col_bytes = Vec::new();
        serialize_int32_col(&mut col_bytes, &[10, 20, 30]);

        let block = make_pblock(vec![int32_meta("id")], col_bytes);
        let decoded = decode_pblocks(&[block]).unwrap();

        assert_eq!(decoded.num_rows, 3);
        assert_eq!(decoded.columns.len(), 1);
        assert_eq!(decoded.columns[0].col_name, "id");
        assert_eq!(decoded.columns[0].type_id, TypeId::Int32 as i32);
        assert!(decoded.columns[0].null_mask.is_none());

        let data = &decoded.columns[0].data;
        assert_eq!(data.len(), 12);
        assert_eq!(i32::from_le_bytes(data[0..4].try_into().unwrap()), 10);
        assert_eq!(i32::from_le_bytes(data[4..8].try_into().unwrap()), 20);
        assert_eq!(i32::from_le_bytes(data[8..12].try_into().unwrap()), 30);
    }

    #[test]
    fn test_decode_nullable_int64_column() {
        // 3 rows of nullable INT64: [100, NULL, 300]
        let null_map = [0u8, 1, 0];

        // Inner INT64 column (with its own header)
        let mut inner = Vec::new();
        write_col_header(&mut inner, false, 3, 3);
        inner.extend_from_slice(&100i64.to_le_bytes());
        inner.extend_from_slice(&0i64.to_le_bytes()); // placeholder for NULL
        inner.extend_from_slice(&300i64.to_le_bytes());

        let mut col_bytes = Vec::new();
        serialize_nullable_wrapper(&mut col_bytes, &null_map, &inner);

        let block = make_pblock(vec![nullable_int64_meta("value")], col_bytes);
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
        let mut col_bytes = Vec::new();
        serialize_string_col(&mut col_bytes, &["hello", "world"]);

        let block = make_pblock(vec![string_meta("name")], col_bytes);
        let decoded = decode_pblocks(&[block]).unwrap();

        assert_eq!(decoded.num_rows, 2);
        let col = &decoded.columns[0];
        assert_eq!(col.col_name, "name");
        assert!(col.offsets.is_some());

        let offsets = col.offsets.as_ref().unwrap();
        assert_eq!(offsets, &[0, 5, 10]); // N+1 clean offsets
        assert_eq!(&col.data[0..5], b"hello");
        assert_eq!(&col.data[5..10], b"world");
    }

    #[test]
    fn test_decode_multi_column_block() {
        // 2 rows: INT32 "id" + STRING "name"
        let mut col_bytes = Vec::new();

        // Column 0: INT32 [42, 99]
        serialize_int32_col(&mut col_bytes, &[42, 99]);

        // Column 1: STRING ["foo", "bar"]
        serialize_string_col(&mut col_bytes, &["foo", "bar"]);

        let block = make_pblock(
            vec![int32_meta("id"), string_meta("name")],
            col_bytes,
        );
        let decoded = decode_pblocks(&[block]).unwrap();

        assert_eq!(decoded.num_rows, 2);
        assert_eq!(decoded.columns.len(), 2);
        assert_eq!(decoded.columns[0].col_name, "id");
        assert_eq!(decoded.columns[1].col_name, "name");

        assert_eq!(
            i32::from_le_bytes(decoded.columns[0].data[0..4].try_into().unwrap()),
            42
        );

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
        serialize_int32_col(&mut bytes1, &[1, 2]);
        let block1 = make_pblock(vec![int32_meta("x")], bytes1);

        // Block 2: INT32 [3, 4, 5]
        let mut bytes2 = Vec::new();
        serialize_int32_col(&mut bytes2, &[3, 4, 5]);
        let block2 = make_pblock(vec![int32_meta("x")], bytes2);

        let decoded = decode_pblocks(&[block1, block2]).unwrap();
        assert_eq!(decoded.num_rows, 5);
        assert_eq!(decoded.columns[0].data.len(), 20); // 5 * 4 bytes

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
        // Build uncompressed payload: INT32 column with 2 values
        let mut uncompressed = Vec::new();
        serialize_int32_col(&mut uncompressed, &[42, 99]);

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

    #[test]
    fn test_streamvbyte_decode_small() {
        // Encode 4 small values: [1, 2, 3, 4] — all fit in 1 byte
        // Control byte: each entry is 0b00 (1 byte), so control = 0b00_00_00_00 = 0x00
        let encoded = [0x00u8, 1, 2, 3, 4];
        let result = streamvbyte_decode(&encoded, 4).unwrap();
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_streamvbyte_decode_mixed() {
        // 2 values: [256, 1]
        // 256 = 0x00000100, needs 2 bytes → code 0b01
        // 1 = 0x01, needs 1 byte → code 0b00
        // Control byte: val0=01, val1=00 → 0b00_00_00_01 = 0x01
        let encoded = [0x01u8, 0x00, 0x01, 0x01];
        let result = streamvbyte_decode(&encoded, 2).unwrap();
        assert_eq!(result, vec![256, 1]);
    }

    #[test]
    fn test_decode_nullable_string() {
        // 2 rows of nullable STRING: ["abc", NULL]
        let null_map = [0u8, 1];

        // Inner STRING column (with its own header)
        let mut inner = Vec::new();
        serialize_string_col(&mut inner, &["abc", ""]);

        let mut col_bytes = Vec::new();
        serialize_nullable_wrapper(&mut col_bytes, &null_map, &inner);

        let meta = PColumnMeta {
            name: Some("s".to_string()),
            r#type: Some(TypeId::String as i32),
            is_nullable: Some(true),
            decimal_param: None,
            children: vec![],
            result_is_nullable: None,
            function_name: None,
            be_exec_version: None,
            column_path: None,
            variant_max_subcolumns_count: None,
        };

        let block = make_pblock(vec![meta], col_bytes);
        let decoded = decode_pblocks(&[block]).unwrap();

        assert_eq!(decoded.num_rows, 2);
        let col = &decoded.columns[0];
        assert!(col.is_nullable);
        let mask = col.null_mask.as_ref().unwrap();
        assert_eq!(mask, &[0, 1]);

        let vals = column_to_sql_values(col, 2);
        assert_eq!(vals[0], "'abc'");
        assert_eq!(vals[1], "NULL");
    }

    #[test]
    fn test_col_header_parsing() {
        let mut buf = Vec::new();
        write_col_header(&mut buf, false, 42, 42);
        let mut offset = 0;
        let header = read_col_header(&buf, &mut offset).unwrap();
        assert_eq!(header.row_num, 42);
        assert_eq!(header.real_saved_num, 42);
        assert_eq!(offset, COL_HEADER_SIZE);
    }

    #[test]
    fn test_column_to_sql_values_binary_string() {
        // String with embedded \0 bytes should produce a BLOB literal.
        let col = DecodedColumn {
            type_id: TypeId::String as i32,
            col_name: "rowid".to_string(),
            is_nullable: false,
            data: {
                let mut d = Vec::new();
                d.extend_from_slice(b"abc\x00def");
                d.extend_from_slice(b"clean");
                d
            },
            null_mask: None,
            offsets: Some(vec![0, 7, 12]),
            precision: 0,
            scale: 0,
        };

        let vals = column_to_sql_values(&col, 2);
        // First row has \0: should be a blob literal
        assert!(vals[0].starts_with("'\\x"), "expected blob literal, got: {}", vals[0]);
        assert!(vals[0].ends_with("'::BLOB"));
        // Second row is clean UTF-8: should be a normal string literal
        assert_eq!(vals[1], "'clean'");
    }
}
