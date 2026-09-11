//! Parquet schema inference for the FE's `get_file_schema` RPC (FILES() table function).

use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::basic::{ConvertedType, LogicalType, Type as PhysicalType};
use parquet::schema::types::Type;
use starrocks_thrift::types::{TPrimitiveType, TTypeNodeType};

use crate::proto::starrocks::{PScalarType, PSlotDescriptor, PTypeDesc, PTypeNode};

/// Length reported for inferred string/binary columns (StarRocks max VARCHAR length).
const INFERRED_VARCHAR_LEN: i32 = 1_048_576;

/// Reads a parquet file's footer and maps each top-level column to a StarRocks slot descriptor.
pub(crate) async fn parquet_file_schema(path: &str) -> Result<Vec<PSlotDescriptor>, String> {
    let local = local_file_path(path)?;
    let file = tokio::fs::File::open(&local)
        .await
        .map_err(|err| format!("failed to open {local}: {err}"))?;
    let builder = ParquetRecordBatchStreamBuilder::new(file)
        .await
        .map_err(|err| format!("failed to read parquet metadata from {local}: {err}"))?;
    // Iterate the top-level fields (not physical leaves) so column names and nesting are preserved.
    let slots = builder
        .metadata()
        .file_metadata()
        .schema_descr()
        .root_schema()
        .get_fields()
        .iter()
        .enumerate()
        .map(|(idx, field)| {
            Ok(slot_descriptor(
                idx as i32,
                field.name(),
                scalar_type(field)?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    // StarRocks resolves column names case-insensitively, so names colliding only by case
    // would be ambiguous; the native scanner rejects them too.
    let mut seen = std::collections::HashSet::new();
    for slot in &slots {
        if !seen.insert(slot.col_name.to_ascii_lowercase()) {
            return Err(format!(
                "column name '{}' collides case-insensitively with another column",
                slot.col_name
            ));
        }
    }
    Ok(slots)
}

/// Infers one schema for a multi-file FILES() call.
///
/// The schema comes from the first file; every other file must agree on column names
/// (ASCII case-insensitive, matching StarRocks name resolution) and types. Positional
/// slot fields (id/slot_idx/column_pos) are per-file bookkeeping and are not compared.
///
/// Deliberately stricter than native StarRocks, which samples `schema_sample_file_count`
/// files and promotes conflicting types: the scan reads every file with the inferred
/// schema, so whole-set agreement is the correct fail-closed contract here.
pub(crate) async fn parquet_files_schema(paths: &[String]) -> Result<Vec<PSlotDescriptor>, String> {
    let (first, rest) = paths
        .split_first()
        .ok_or_else(|| "no file paths to infer a schema from".to_string())?;
    let slots = parquet_file_schema(first).await?;
    for path in rest {
        let other = parquet_file_schema(path).await?;
        if other.len() != slots.len() {
            return Err(format!(
                "FILES() schema mismatch: '{path}' has {} columns but '{first}' has {}",
                other.len(),
                slots.len()
            ));
        }
        for (idx, (expected, found)) in slots.iter().zip(&other).enumerate() {
            if !expected.col_name.eq_ignore_ascii_case(&found.col_name) {
                return Err(format!(
                    "FILES() schema mismatch: column {idx} is named '{}' in '{first}' but '{}' in '{path}'",
                    expected.col_name, found.col_name
                ));
            }
            if expected.slot_type != found.slot_type {
                return Err(format!(
                    "FILES() schema mismatch: column '{}' has a different type in '{path}' than in '{first}'",
                    expected.col_name
                ));
            }
        }
    }
    Ok(slots)
}

/// Resolves a StarRocks file path to a local filesystem path, rejecting remote URIs.
///
/// Accepts `file:/abs`, `file:///abs`, and `file://localhost/abs`; remote schemes
/// (s3://, hdfs://, ...) and non-local file authorities need broker handling, not yet wired.
fn local_file_path(path: &str) -> Result<String, String> {
    if let Some(rest) = path.strip_prefix("file://") {
        // `file:///abs` has an empty authority; `file://host/abs` has one before the path.
        if rest.starts_with('/') {
            return Ok(rest.to_string());
        }
        let authority = rest.split('/').next().unwrap_or(rest);
        if authority == "localhost" {
            return Ok(rest[authority.len()..].to_string());
        }
        return Err(format!("file URI authority '{authority}' is not local"));
    }
    if let Some(rest) = path.strip_prefix("file:") {
        return Ok(rest.to_string());
    }
    if let Some((scheme, _)) = path.split_once("://") {
        return Err(format!(
            "remote path scheme '{scheme}://' is not supported yet; only local file paths are implemented"
        ));
    }
    Ok(path.to_string())
}

/// Maps one top-level parquet field to a StarRocks scalar type, preferring the logical type.
fn scalar_type(field: &Type) -> Result<PScalarType, String> {
    if !field.is_primitive() {
        return Err(format!(
            "column '{}' has a nested parquet type; only flat columns are supported",
            field.name()
        ));
    }
    if let Some(logical) = field.get_basic_info().logical_type_ref() {
        return Ok(match logical {
            LogicalType::String | LogicalType::Enum | LogicalType::Uuid => varchar(),
            LogicalType::Bson => varbinary(),
            LogicalType::Json => primitive(TPrimitiveType::JSON),
            LogicalType::Decimal(d) => decimal(d.precision, d.scale),
            LogicalType::Date => primitive(TPrimitiveType::DATE),
            LogicalType::Time(_) => primitive(TPrimitiveType::TIME),
            LogicalType::Timestamp(_) => primitive(TPrimitiveType::DATETIME),
            // Integers (incl. unsigned/sub-width) follow the native mapping: by physical width.
            _ => physical_scalar_type(field),
        });
    }
    Ok(physical_scalar_type(field))
}

/// Maps a field without a logical-type annotation via its converted/physical type.
fn physical_scalar_type(field: &Type) -> PScalarType {
    match field.get_basic_info().converted_type() {
        ConvertedType::UTF8 | ConvertedType::ENUM => return varchar(),
        ConvertedType::BSON => return varbinary(),
        ConvertedType::JSON => return primitive(TPrimitiveType::JSON),
        ConvertedType::DECIMAL => return decimal(field.get_precision(), field.get_scale()),
        ConvertedType::DATE => return primitive(TPrimitiveType::DATE),
        ConvertedType::TIMESTAMP_MILLIS | ConvertedType::TIMESTAMP_MICROS => {
            return primitive(TPrimitiveType::DATETIME);
        }
        ConvertedType::TIME_MILLIS | ConvertedType::TIME_MICROS => {
            return primitive(TPrimitiveType::TIME);
        }
        _ => {}
    }
    // Integer annotations (INT_8/UINT_32/...) are not special-cased: like the native reader,
    // integers map purely by physical width (INT32 -> INT, INT64 -> BIGINT).
    match field.get_physical_type() {
        PhysicalType::BOOLEAN => primitive(TPrimitiveType::BOOLEAN),
        PhysicalType::INT32 => primitive(TPrimitiveType::INT),
        PhysicalType::INT64 => primitive(TPrimitiveType::BIGINT),
        PhysicalType::INT96 => primitive(TPrimitiveType::DATETIME),
        PhysicalType::FLOAT => primitive(TPrimitiveType::FLOAT),
        PhysicalType::DOUBLE => primitive(TPrimitiveType::DOUBLE),
        // Raw variable byte arrays are binary; the reader accepts them as VARBINARY.
        PhysicalType::BYTE_ARRAY => varbinary(),
        // The native parquet reader only accepts fixed-length byte arrays as varchar/char/decimal.
        PhysicalType::FIXED_LEN_BYTE_ARRAY => varchar(),
    }
}

/// Picks the DecimalV3 width StarRocks uses for the given precision.
fn decimal(precision: i32, scale: i32) -> PScalarType {
    let primitive = if precision <= 9 {
        TPrimitiveType::DECIMAL32
    } else if precision <= 18 {
        TPrimitiveType::DECIMAL64
    } else if precision <= 38 {
        TPrimitiveType::DECIMAL128
    } else {
        // The reader has no DECIMAL256 converter; native promotion falls back to varchar.
        return varchar();
    };
    PScalarType {
        r#type: primitive.0,
        len: None,
        precision: Some(precision),
        scale: Some(scale),
    }
}

/// Scalar type carrying only a primitive tag (no length/precision/scale).
fn primitive(primitive: TPrimitiveType) -> PScalarType {
    PScalarType {
        r#type: primitive.0,
        len: None,
        precision: None,
        scale: None,
    }
}

/// Inferred VARCHAR scalar type.
fn varchar() -> PScalarType {
    sized(TPrimitiveType::VARCHAR)
}

/// Inferred VARBINARY scalar type.
fn varbinary() -> PScalarType {
    sized(TPrimitiveType::VARBINARY)
}

/// Length-carrying scalar type (VARCHAR/VARBINARY) at the inferred max length.
fn sized(primitive: TPrimitiveType) -> PScalarType {
    PScalarType {
        r#type: primitive.0,
        len: Some(INFERRED_VARCHAR_LEN),
        precision: None,
        scale: None,
    }
}

/// Wraps a scalar type in the slot descriptor the FE expects (only col_name and slot_type are read).
fn slot_descriptor(idx: i32, name: &str, scalar: PScalarType) -> PSlotDescriptor {
    PSlotDescriptor {
        id: idx,
        parent: 0,
        slot_type: PTypeDesc {
            types: vec![PTypeNode {
                r#type: TTypeNodeType::SCALAR.0,
                scalar_type: Some(scalar),
                struct_fields: Vec::new(),
            }],
        },
        column_pos: idx,
        byte_offset: 0,
        null_indicator_byte: 0,
        null_indicator_bit: 0,
        col_name: name.to_string(),
        slot_idx: idx,
        is_materialized: true,
        global_dict_words: Vec::new(),
        global_dict_version: None,
    }
}

/// Parquet fixture writing shared with the compute-node service tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;

    /// Writes a schema-only parquet file (no row groups) to a unique temp path.
    pub(crate) fn write_parquet(tag: &str, message: &str) -> std::path::PathBuf {
        let schema = Arc::new(parse_message_type(message).unwrap());
        let path = std::env::temp_dir().join(format!("sr_cn_{}_{tag}.parquet", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        let props = Arc::new(WriterProperties::builder().build());
        SerializedFileWriter::new(file, schema, props)
            .unwrap()
            .close()
            .unwrap();
        path
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::test_support::write_parquet;
    use super::*;

    fn scalar(slot: &PSlotDescriptor) -> &PScalarType {
        slot.slot_type.types[0].scalar_type.as_ref().unwrap()
    }

    #[tokio::test]
    async fn maps_parquet_columns_to_starrocks_scalars() {
        let path = write_parquet(
            "types",
            "message lineitem {
                optional int64 l_orderkey;
                optional int32 l_linenumber;
                optional double l_quantity;
                optional int32 l_shipdate (DATE);
                optional binary l_comment (UTF8);
                optional binary l_raw;
                optional int32 l_small (UINT_8);
                optional int32 l_amount (DECIMAL(9,2));
            }",
        );
        let slots = parquet_file_schema(path.to_str().unwrap()).await.unwrap();
        std::fs::remove_file(&path).ok();

        let by_name: HashMap<&str, &PSlotDescriptor> =
            slots.iter().map(|s| (s.col_name.as_str(), s)).collect();
        assert_eq!(
            scalar(by_name["l_orderkey"]).r#type,
            TPrimitiveType::BIGINT.0
        );
        assert_eq!(
            scalar(by_name["l_linenumber"]).r#type,
            TPrimitiveType::INT.0
        );
        assert_eq!(
            scalar(by_name["l_quantity"]).r#type,
            TPrimitiveType::DOUBLE.0
        );
        assert_eq!(scalar(by_name["l_shipdate"]).r#type, TPrimitiveType::DATE.0);
        assert_eq!(
            scalar(by_name["l_comment"]).r#type,
            TPrimitiveType::VARCHAR.0
        );
        // Raw byte arrays stay binary; INT32-backed integers (incl. UINT_8) map to INT like native.
        assert_eq!(scalar(by_name["l_raw"]).r#type, TPrimitiveType::VARBINARY.0);
        assert_eq!(scalar(by_name["l_small"]).r#type, TPrimitiveType::INT.0);
        let amount = scalar(by_name["l_amount"]);
        assert_eq!(amount.r#type, TPrimitiveType::DECIMAL32.0);
        assert_eq!(amount.precision, Some(9));
        assert_eq!(amount.scale, Some(2));
    }

    #[tokio::test]
    async fn strips_both_file_scheme_forms() {
        let path = write_parquet("scheme", "message s { optional int64 a; }");
        let bare = path.to_str().unwrap();
        // Bare, file:/abs, file:///abs, and file://localhost/abs all resolve to the same file.
        for candidate in [
            bare.to_string(),
            format!("file:{bare}"),
            format!("file://{bare}"),
            format!("file://localhost{bare}"),
        ] {
            let slots = parquet_file_schema(&candidate).await.unwrap();
            assert_eq!(slots.len(), 1, "{candidate}");
            assert_eq!(slots[0].col_name, "a");
        }
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn rejects_non_local_file_authority() {
        let result = parquet_file_schema("file://remotehost/tmp/data.parquet").await;
        assert!(
            result.as_ref().is_err_and(|err| err.contains("not local")),
            "{result:?}"
        );
    }

    #[test]
    fn decimal_falls_back_to_varchar_past_decimal128() {
        assert_eq!(decimal(9, 0).r#type, TPrimitiveType::DECIMAL32.0);
        assert_eq!(decimal(18, 0).r#type, TPrimitiveType::DECIMAL64.0);
        assert_eq!(decimal(38, 2).r#type, TPrimitiveType::DECIMAL128.0);
        // Precision past DECIMAL128 has no reader converter, so it degrades to varchar.
        assert_eq!(decimal(40, 2).r#type, TPrimitiveType::VARCHAR.0);
    }

    #[tokio::test]
    async fn maps_json_to_json_not_varchar() {
        let path = write_parquet("json", "message j { optional binary payload (JSON); }");
        let slots = parquet_file_schema(path.to_str().unwrap()).await.unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(scalar(&slots[0]).r#type, TPrimitiveType::JSON.0);
    }

    #[tokio::test]
    async fn rejects_remote_paths() {
        let result = parquet_file_schema("s3://bucket/lineitem.parquet").await;
        assert!(
            result.as_ref().is_err_and(|err| err.contains("s3://")),
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn rejects_case_insensitive_duplicate_columns() {
        let path = write_parquet(
            "dupcase",
            "message d { optional int64 Col; optional int64 col; }",
        );
        let result = parquet_file_schema(path.to_str().unwrap()).await;
        std::fs::remove_file(&path).ok();
        assert!(
            result
                .as_ref()
                .is_err_and(|err| err.contains("case-insensitively")),
            "{result:?}"
        );
    }

    /// Runs `parquet_files_schema` over already-written fixture files and removes them.
    async fn files_schema(paths: &[&std::path::PathBuf]) -> Result<Vec<PSlotDescriptor>, String> {
        let strings: Vec<String> = paths
            .iter()
            .map(|p| p.to_str().unwrap().to_string())
            .collect();
        let result = parquet_files_schema(&strings).await;
        for path in paths {
            std::fs::remove_file(path).ok();
        }
        result
    }

    #[tokio::test]
    async fn multi_file_agreeing_schemas_infer_the_single_file_slots() {
        let message = "message m { optional int64 a; optional double b; }";
        let first = write_parquet("agree_a", message);
        let second = write_parquet("agree_b", message);
        let single = parquet_file_schema(first.to_str().unwrap()).await.unwrap();
        let multi = files_schema(&[&first, &second]).await.unwrap();
        assert_eq!(multi, single);
    }

    #[tokio::test]
    async fn multi_file_rejects_column_count_mismatch() {
        let first = write_parquet(
            "arity_a",
            "message m { optional int64 a; optional double b; }",
        );
        let second = write_parquet("arity_b", "message m { optional int64 a; }");
        let result = files_schema(&[&first, &second]).await;
        let err = result.unwrap_err();
        assert!(
            err.contains("has 1 columns but") && err.contains("has 2"),
            "{err}"
        );
        assert!(
            err.contains(first.to_str().unwrap()) && err.contains(second.to_str().unwrap()),
            "{err}"
        );
    }

    #[tokio::test]
    async fn multi_file_rejects_column_name_mismatch() {
        let first = write_parquet(
            "name_a",
            "message m { optional int64 a; optional double l_tax; }",
        );
        let second = write_parquet(
            "name_b",
            "message m { optional int64 a; optional double l_taxx; }",
        );
        let err = files_schema(&[&first, &second]).await.unwrap_err();
        assert!(
            err.contains("column 1 is named 'l_tax'") && err.contains("but 'l_taxx'"),
            "{err}"
        );
    }

    #[tokio::test]
    async fn multi_file_rejects_column_type_mismatch() {
        let first = write_parquet("type_a", "message m { optional int64 l_quantity; }");
        let second = write_parquet("type_b", "message m { optional double l_quantity; }");
        let err = files_schema(&[&first, &second]).await.unwrap_err();
        assert!(
            err.contains("column 'l_quantity' has a different type"),
            "{err}"
        );
        assert!(err.contains(second.to_str().unwrap()), "{err}");
    }

    #[tokio::test]
    async fn multi_file_accepts_case_differing_column_names() {
        let first = write_parquet("case_a", "message m { optional int64 L_ORDERKEY; }");
        let second = write_parquet("case_b", "message m { optional int64 l_orderkey; }");
        let slots = files_schema(&[&first, &second]).await.unwrap();
        // The first file wins the spelling, matching StarRocks case-insensitive resolution.
        assert_eq!(slots[0].col_name, "L_ORDERKEY");
    }

    #[tokio::test]
    async fn multi_file_missing_file_error_names_its_path() {
        let first = write_parquet("missing_a", "message m { optional int64 a; }");
        let missing = "/nonexistent/sr_cn_missing_b.parquet".to_string();
        let result =
            parquet_files_schema(&[first.to_str().unwrap().to_string(), missing.clone()]).await;
        std::fs::remove_file(&first).ok();
        assert!(
            result.as_ref().is_err_and(|err| err.contains(&missing)),
            "{result:?}"
        );
    }

    #[tokio::test]
    async fn rejects_nested_columns() {
        let path = write_parquet(
            "nested",
            "message nested {
                optional group address {
                    optional binary city (UTF8);
                }
            }",
        );
        let result = parquet_file_schema(path.to_str().unwrap()).await;
        std::fs::remove_file(&path).ok();
        assert!(
            result.as_ref().is_err_and(|err| err.contains("nested")),
            "{result:?}"
        );
    }
}
