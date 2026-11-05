//! Parquet schema inference for the FE's analysis-time `fetch_table_schema` RPC, plus the
//! `glob` listing behind the `local()` table-valued function.
//!
//! The FE resolves `local(...)`/`s3(...)`/`hdfs(...)` columns by asking one alive backend
//! for the schema of the first file (`ExternalFileTableValuedFunction.getTableColumns`), and
//! for `local()` first asks a backend to expand the path glob. Without both RPCs no TVF
//! query is ever planned. The type mapping mirrors the BE's parquet reader
//! (`FieldDescriptor::convert_to_doris_type`) so views and plans come out the same as
//! against a real BE.

use doris_proto::{PScalarType, PTypeDesc, PTypeNode, p_glob_response::PFileInfo};
use doris_thrift::types::{TPrimitiveType, TTypeNodeType};
use parquet::arrow::async_reader::ParquetRecordBatchStreamBuilder;
use parquet::basic::{ConvertedType, LogicalType, TimeUnit, Type as PhysicalType};
use parquet::schema::types::Type;

/// One inferred column: the name as stored in the file and its Doris type.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InferredColumn {
    pub(crate) name: String,
    pub(crate) type_desc: PTypeDesc,
}

/// Reads a parquet file's footer and maps each top-level column to a Doris type.
pub(crate) async fn parquet_file_schema(path: &str) -> Result<Vec<InferredColumn>, String> {
    let local = local_file_path(path)?;
    let file = tokio::fs::File::open(&local)
        .await
        .map_err(|err| format!("failed to open {local}: {err}"))?;
    let builder = ParquetRecordBatchStreamBuilder::new(file)
        .await
        .map_err(|err| format!("failed to read parquet metadata from {local}: {err}"))?;
    // Iterate the top-level fields (not physical leaves) so column names and nesting are preserved.
    let columns = builder
        .metadata()
        .file_metadata()
        .schema_descr()
        .root_schema()
        .get_fields()
        .iter()
        .map(|field| {
            Ok(InferredColumn {
                name: field.name().to_string(),
                type_desc: scalar_type_desc(scalar_type(field)?),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    // The FE lower-cases column names and rejects case-only collisions
    // (`fillColumns`: "Repeated lowercase column names"); fail here with the same rule so the
    // error names the column instead of surfacing as an FE-side NotSupportedException.
    let mut seen = std::collections::HashSet::new();
    for column in &columns {
        if !seen.insert(column.name.to_ascii_lowercase()) {
            return Err(format!(
                "column name '{}' collides case-insensitively with another column",
                column.name
            ));
        }
    }
    Ok(columns)
}

/// Expands a `local()` path glob into `(file, size)` entries, files only, sorted by path.
///
/// The FE (`LocalTableValuedFunction`, `shared_storage=true`) asks one backend to list the
/// files, then distributes them across backends — so every backend must see the same
/// directory layout (ADR-011: parquet is replicated per host).
pub(crate) fn glob_files(pattern: &str) -> Result<Vec<PFileInfo>, String> {
    let pattern = local_file_path(pattern)?;
    let paths = glob::glob(&pattern).map_err(|err| format!("invalid glob '{pattern}': {err}"))?;
    let mut files = Vec::new();
    for entry in paths {
        let path = entry.map_err(|err| format!("failed to read glob entry: {err}"))?;
        let metadata = std::fs::metadata(&path)
            .map_err(|err| format!("failed to stat {}: {err}", path.display()))?;
        if !metadata.is_file() {
            continue;
        }
        files.push(PFileInfo {
            file: Some(path.to_string_lossy().into_owned()),
            size: Some(metadata.len() as i64),
        });
    }
    files.sort_by(|a, b| a.file.cmp(&b.file));
    Ok(files)
}

/// Resolves a Doris file path to a local filesystem path, rejecting remote URIs.
///
/// Accepts bare paths, `file:/abs`, `file:///abs`, and `file://localhost/abs`; remote schemes
/// (s3://, hdfs://, ...) are out of scope for this backend (ADR-011: `local()` only).
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
            "remote path scheme '{scheme}://' is not supported; only local() file paths are implemented"
        ));
    }
    Ok(path.to_string())
}

/// Maps one top-level parquet field to a Doris scalar type the way the BE's parquet reader
/// does: logical type first, then converted type, then physical type.
fn scalar_type(field: &Type) -> Result<PScalarType, String> {
    if !field.is_primitive() {
        return Err(format!(
            "column '{}' has a nested parquet type (ARRAY/MAP/STRUCT); only flat columns are supported",
            field.name()
        ));
    }
    if let Some(logical) = field.get_basic_info().logical_type_ref() {
        match logical {
            LogicalType::String | LogicalType::Enum | LogicalType::Json | LogicalType::Uuid => {
                return Ok(primitive(TPrimitiveType::STRING));
            }
            LogicalType::Decimal(d) => return Ok(decimal(d.precision, d.scale)),
            LogicalType::Date => return Ok(primitive(TPrimitiveType::DATEV2)),
            LogicalType::Integer(i) => return Ok(integer(i.bit_width, i.is_signed)),
            LogicalType::Time(_) => return Ok(primitive(TPrimitiveType::TIMEV2)),
            LogicalType::Timestamp(t) => return Ok(datetime_v2(time_unit_scale(t.unit))),
            LogicalType::Float16 => return Ok(primitive(TPrimitiveType::FLOAT)),
            // Other logical types (BSON, unknown) fall through to the physical mapping, as the
            // BE swallows its "Not supported parquet logicalType" exception and does the same.
            _ => {}
        }
    }
    match field.get_basic_info().converted_type() {
        ConvertedType::UTF8 | ConvertedType::ENUM | ConvertedType::JSON => {
            return Ok(primitive(TPrimitiveType::STRING));
        }
        ConvertedType::DECIMAL => return Ok(decimal(field.get_precision(), field.get_scale())),
        ConvertedType::DATE => return Ok(primitive(TPrimitiveType::DATEV2)),
        ConvertedType::TIME_MILLIS | ConvertedType::TIME_MICROS => {
            return Ok(primitive(TPrimitiveType::TIMEV2));
        }
        ConvertedType::TIMESTAMP_MILLIS => return Ok(datetime_v2(3)),
        ConvertedType::TIMESTAMP_MICROS => return Ok(datetime_v2(6)),
        ConvertedType::INT_8 => return Ok(primitive(TPrimitiveType::TINYINT)),
        ConvertedType::UINT_8 | ConvertedType::INT_16 => {
            return Ok(primitive(TPrimitiveType::SMALLINT));
        }
        ConvertedType::UINT_16 | ConvertedType::INT_32 => {
            return Ok(primitive(TPrimitiveType::INT));
        }
        ConvertedType::UINT_32 | ConvertedType::INT_64 => {
            return Ok(primitive(TPrimitiveType::BIGINT));
        }
        ConvertedType::UINT_64 => return Ok(primitive(TPrimitiveType::LARGEINT)),
        _ => {}
    }
    Ok(match field.get_physical_type() {
        PhysicalType::BOOLEAN => primitive(TPrimitiveType::BOOLEAN),
        PhysicalType::INT32 => primitive(TPrimitiveType::INT),
        PhysicalType::INT64 => primitive(TPrimitiveType::BIGINT),
        // "in most cases, it's a nano timestamp"
        PhysicalType::INT96 => datetime_v2(6),
        PhysicalType::FLOAT => primitive(TPrimitiveType::FLOAT),
        PhysicalType::DOUBLE => primitive(TPrimitiveType::DOUBLE),
        // Unannotated byte arrays are strings unless `enable_mapping_varbinary` (we never set it).
        PhysicalType::BYTE_ARRAY | PhysicalType::FIXED_LEN_BYTE_ARRAY => {
            primitive(TPrimitiveType::STRING)
        }
    })
}

/// Signed parquet integers map by width; unsigned ones widen (the BE marks these
/// `is_type_compatibility`), with UINT64 landing on LARGEINT.
fn integer(bit_width: i8, is_signed: bool) -> PScalarType {
    let primitive_type = match (is_signed, bit_width) {
        (true, ..=8) => TPrimitiveType::TINYINT,
        (true, ..=16) => TPrimitiveType::SMALLINT,
        (true, ..=32) => TPrimitiveType::INT,
        (true, _) => TPrimitiveType::BIGINT,
        (false, ..=8) => TPrimitiveType::SMALLINT,
        (false, ..=16) => TPrimitiveType::INT,
        (false, ..=32) => TPrimitiveType::BIGINT,
        (false, _) => TPrimitiveType::LARGEINT,
    };
    primitive(primitive_type)
}

/// Fractional-second digits the BE assigns a timestamp unit (MILLIS → 3, otherwise 6).
fn time_unit_scale(unit: TimeUnit) -> i32 {
    match unit {
        TimeUnit::MILLIS => 3,
        TimeUnit::MICROS | TimeUnit::NANOS => 6,
    }
}

/// Decimal columns are reported as DECIMAL128I with their precision and scale; the FE
/// normalizes to DECIMALV3(precision, scale) regardless of the reported width.
fn decimal(precision: i32, scale: i32) -> PScalarType {
    PScalarType {
        r#type: TPrimitiveType::DECIMAL128I.0,
        len: None,
        precision: Some(precision),
        scale: Some(scale),
    }
}

/// DATETIMEV2 with the given fractional-second scale.
fn datetime_v2(scale: i32) -> PScalarType {
    PScalarType {
        r#type: TPrimitiveType::DATETIMEV2.0,
        len: None,
        precision: None,
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

/// Wraps a scalar type in the single-node `PTypeDesc` the FE decodes (`getColumnType`).
fn scalar_type_desc(scalar: PScalarType) -> PTypeDesc {
    PTypeDesc {
        types: vec![PTypeNode {
            r#type: TTypeNodeType::SCALAR.0,
            scalar_type: Some(scalar),
            ..Default::default()
        }],
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use parquet::file::properties::WriterProperties;
    use parquet::file::writer::SerializedFileWriter;
    use parquet::schema::parser::parse_message_type;

    use super::*;

    /// Writes a schema-only parquet file (no row groups) to a unique temp path.
    fn write_parquet(tag: &str, message: &str) -> std::path::PathBuf {
        let schema = Arc::new(parse_message_type(message).unwrap());
        let path =
            std::env::temp_dir().join(format!("sirius_be_{}_{tag}.parquet", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        let props = Arc::new(WriterProperties::builder().build());
        SerializedFileWriter::new(file, schema, props)
            .unwrap()
            .close()
            .unwrap();
        path
    }

    fn scalar(column: &InferredColumn) -> &PScalarType {
        column.type_desc.types[0].scalar_type.as_ref().unwrap()
    }

    #[tokio::test]
    async fn maps_parquet_columns_like_the_be_reader() {
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
                optional int64 l_ts (TIMESTAMP_MICROS);
                optional boolean l_flag;
                optional float l_f;
            }",
        );
        let columns = parquet_file_schema(path.to_str().unwrap()).await.unwrap();
        std::fs::remove_file(&path).ok();

        let by_name: HashMap<&str, &InferredColumn> =
            columns.iter().map(|c| (c.name.as_str(), c)).collect();
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
        assert_eq!(
            scalar(by_name["l_shipdate"]).r#type,
            TPrimitiveType::DATEV2.0
        );
        assert_eq!(
            scalar(by_name["l_comment"]).r#type,
            TPrimitiveType::STRING.0
        );
        // Unannotated byte arrays are STRING (the BE default without enable_mapping_varbinary).
        assert_eq!(scalar(by_name["l_raw"]).r#type, TPrimitiveType::STRING.0);
        // UINT_8 widens to SMALLINT.
        assert_eq!(
            scalar(by_name["l_small"]).r#type,
            TPrimitiveType::SMALLINT.0
        );
        let amount = scalar(by_name["l_amount"]);
        assert_eq!(amount.r#type, TPrimitiveType::DECIMAL128I.0);
        assert_eq!(amount.precision, Some(9));
        assert_eq!(amount.scale, Some(2));
        let ts = scalar(by_name["l_ts"]);
        assert_eq!(ts.r#type, TPrimitiveType::DATETIMEV2.0);
        assert_eq!(ts.scale, Some(6));
        assert_eq!(scalar(by_name["l_flag"]).r#type, TPrimitiveType::BOOLEAN.0);
        assert_eq!(scalar(by_name["l_f"]).r#type, TPrimitiveType::FLOAT.0);
        // Column order follows the file.
        assert_eq!(columns[0].name, "l_orderkey");
        assert_eq!(columns.len(), 11);
    }

    #[test]
    fn logical_integers_map_by_width_and_sign() {
        assert_eq!(integer(8, true).r#type, TPrimitiveType::TINYINT.0);
        assert_eq!(integer(16, true).r#type, TPrimitiveType::SMALLINT.0);
        assert_eq!(integer(32, true).r#type, TPrimitiveType::INT.0);
        assert_eq!(integer(64, true).r#type, TPrimitiveType::BIGINT.0);
        assert_eq!(integer(8, false).r#type, TPrimitiveType::SMALLINT.0);
        assert_eq!(integer(32, false).r#type, TPrimitiveType::BIGINT.0);
        assert_eq!(integer(64, false).r#type, TPrimitiveType::LARGEINT.0);
    }

    #[tokio::test]
    async fn strips_both_file_scheme_forms() {
        let path = write_parquet("scheme", "message s { optional int64 a; }");
        let bare = path.to_str().unwrap();
        for candidate in [
            bare.to_string(),
            format!("file:{bare}"),
            format!("file://{bare}"),
            format!("file://localhost{bare}"),
        ] {
            let columns = parquet_file_schema(&candidate).await.unwrap();
            assert_eq!(columns.len(), 1, "{candidate}");
            assert_eq!(columns[0].name, "a");
        }
        std::fs::remove_file(&path).ok();
    }

    #[tokio::test]
    async fn rejects_remote_paths_and_non_local_authorities() {
        let result = parquet_file_schema("s3://bucket/lineitem.parquet").await;
        assert!(
            result.as_ref().is_err_and(|err| err.contains("s3://")),
            "{result:?}"
        );
        let result = parquet_file_schema("file://remotehost/tmp/data.parquet").await;
        assert!(
            result.as_ref().is_err_and(|err| err.contains("not local")),
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

    #[test]
    fn glob_lists_files_only_sorted_with_sizes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("b.parquet"), b"bb").unwrap();
        std::fs::write(dir.path().join("a.parquet"), b"a").unwrap();
        std::fs::write(dir.path().join("notes.txt"), b"x").unwrap();
        std::fs::create_dir(dir.path().join("sub.parquet")).unwrap();

        let files = glob_files(&format!("{}/*.parquet", dir.path().display())).unwrap();

        let names: Vec<_> = files
            .iter()
            .map(|f| {
                std::path::Path::new(f.file.as_deref().unwrap())
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, vec!["a.parquet", "b.parquet"]);
        assert_eq!(files[0].size, Some(1));
        assert_eq!(files[1].size, Some(2));
    }

    #[test]
    fn glob_rejects_remote_schemes() {
        let result = glob_files("s3://bucket/*.parquet");
        assert!(result.is_err_and(|err| err.contains("s3://")));
    }
}
