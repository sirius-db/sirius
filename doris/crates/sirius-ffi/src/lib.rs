//! Sirius query execution engine.
//!
//! Provides a unified API for executing SQL queries and returning Arrow IPC bytes.
//!
//! # Feature gating
//!
//! With `duckdb-bundled`: uses a bundled DuckDB instance (CPU-only, no external deps).
//! Without any feature (default): stub implementations return `EngineError::NotCompiled`.

/// High-level query execution engine.
///
/// Wraps a DuckDB connection (when compiled with `duckdb-bundled`) and provides
/// a safe API for executing SQL queries that return Arrow IPC stream bytes.
pub struct SiriusEngine {
    #[cfg(feature = "duckdb-bundled")]
    conn: duckdb::Connection,

    #[cfg(not(feature = "duckdb-bundled"))]
    _phantom: (),
}

/// Serialize Arrow record batches into an IPC stream.
#[cfg(feature = "duckdb-bundled")]
fn batches_to_ipc(batches: Vec<arrow::record_batch::RecordBatch>) -> Result<Vec<u8>, EngineError> {
    use arrow::ipc::writer::StreamWriter;

    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let schema = batches[0].schema();
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &schema)
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        for batch in &batches {
            writer
                .write(batch)
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        }
        writer
            .finish()
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
    }
    Ok(buf)
}

impl SiriusEngine {
    /// Create a new engine.
    ///
    /// Loads the DuckDB Substrait extension from the local build.
    pub fn new() -> Result<Self, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            // Allow unsigned extensions (needed for locally-built extensions).
            let config = duckdb::Config::default()
                .with("allow_unsigned_extensions", "true")
                .map_err(|e| EngineError::InitFailed(e.to_string()))?;
            let conn = duckdb::Connection::open_in_memory_with_flags(config)
                .map_err(|e| EngineError::InitFailed(e.to_string()))?;

            // Load locally-built extensions from the Sirius build output.
            let sirius_root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
            let substrait_ext = format!(
                "{}/build/release/extension/substrait/substrait.duckdb_extension",
                sirius_root
            );
            let sirius_ext = format!(
                "{}/build/release/extension/sirius/sirius.duckdb_extension",
                sirius_root
            );
            conn.execute_batch(&format!("LOAD '{}'", substrait_ext))
                .map_err(|e| EngineError::InitFailed(format!("load substrait extension: {e}")))?;
            conn.execute_batch(&format!("LOAD '{}'", sirius_ext))
                .map_err(|e| EngineError::InitFailed(format!("load sirius extension: {e}")))?;

            Ok(Self { conn })
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            Err(EngineError::NotCompiled)
        }
    }

    /// Execute a SQL query directly via DuckDB (CPU fallback).
    pub fn execute_sql(&self, sql: &str) -> Result<Vec<u8>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            use arrow::record_batch::RecordBatch;

            let mut stmt = self
                .conn
                .prepare(sql)
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let batches: Vec<RecordBatch> = stmt
                .query_arrow([])
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?
                .collect();
            batches_to_ipc(batches)
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = sql;
            Err(EngineError::NotCompiled)
        }
    }

    /// Execute a SQL query via Sirius GPU (`gpu_processing` table function).
    pub fn execute_gpu(&self, sql: &str) -> Result<Vec<u8>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            use arrow::record_batch::RecordBatch;

            let mut stmt = self
                .conn
                .prepare("SELECT * FROM gpu_processing(?)")
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let batches: Vec<RecordBatch> = stmt
                .query_arrow(duckdb::params![sql])
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?
                .collect();
            batches_to_ipc(batches)
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = sql;
            Err(EngineError::NotCompiled)
        }
    }

    /// Execute a Substrait plan via Sirius GPU (`gpu_processing_substrait`).
    pub fn execute_substrait(&self, plan_bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            use arrow::record_batch::RecordBatch;

            let mut stmt = self
                .conn
                .prepare("SELECT * FROM gpu_processing_substrait(?::blob)")
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let batches: Vec<RecordBatch> = stmt
                .query_arrow(duckdb::params![plan_bytes])
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?
                .collect();
            batches_to_ipc(batches)
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = plan_bytes;
            Err(EngineError::NotCompiled)
        }
    }

    /// Execute a Substrait plan via DuckDB CPU (`from_substrait`).
    pub fn from_substrait(&self, plan_bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            use arrow::record_batch::RecordBatch;

            let mut stmt = self
                .conn
                .prepare("SELECT * FROM from_substrait(?::blob)")
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let batches: Vec<RecordBatch> = stmt
                .query_arrow(duckdb::params![plan_bytes])
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?
                .collect();
            batches_to_ipc(batches)
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = plan_bytes;
            Err(EngineError::NotCompiled)
        }
    }

    /// Get the Arrow schema of a file as IPC stream bytes (schema only, no data).
    ///
    /// Uses `LIMIT 0` + `query_arrow().get_schema()` to extract the schema from
    /// the prepared statement metadata, without reading any data rows.
    pub fn get_file_schema_ipc(&self, file_path: &str, format: &str) -> Result<Vec<u8>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            use arrow::ipc::writer::StreamWriter;

            let reader_fn = match format {
                "parquet" => "read_parquet",
                "csv" => "read_csv_auto",
                "json" => "read_json_auto",
                "orc" => "read_parquet",
                other => return Err(EngineError::ExecFailed(format!("unsupported file format: {other}"))),
            };
            let sql = format!("SELECT * FROM {}('{}') LIMIT 0", reader_fn, file_path);
            let mut stmt = self
                .conn
                .prepare(&sql)
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let arrow_result = stmt
                .query_arrow([])
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let schema = arrow_result.get_schema();

            // Write schema-only IPC stream (header + EOS, no data batches).
            let mut buf = Vec::new();
            {
                let mut writer = StreamWriter::try_new(&mut buf, &schema)
                    .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
                writer
                    .finish()
                    .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            }
            Ok(buf)
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = (file_path, format);
            Err(EngineError::NotCompiled)
        }
    }

    /// Register a file as a DuckDB table (e.g. Parquet, CSV, JSON).
    ///
    /// When `columns` is non-empty, only those columns are loaded (in order),
    /// ensuring DuckDB's table schema matches the Substrait ReadRel field references.
    /// This must be called before executing a plan that references the table.
    pub fn register_file_table(&self, table_name: &str, file_path: &str, format: &str, columns: &[String]) -> Result<(), EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            let reader_fn = match format {
                "parquet" => "read_parquet",
                "csv" => "read_csv_auto",
                "json" => "read_json_auto",
                "orc" => "read_parquet", // DuckDB doesn't have read_orc, parquet reader handles it
                other => return Err(EngineError::ExecFailed(format!("unsupported file format: {other}"))),
            };
            let select = if columns.is_empty() {
                "*".to_string()
            } else {
                columns.iter().map(|c| format!("\"{}\"", c)).collect::<Vec<_>>().join(", ")
            };
            let sql = format!(
                "CREATE OR REPLACE TABLE \"{}\" AS SELECT {} FROM {}('{}')",
                table_name, select, reader_fn, file_path
            );
            self.conn
                .execute_batch(&sql)
                .map_err(|e| EngineError::ExecFailed(format!("register_file_table: {e}")))?;
            Ok(())
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = (table_name, file_path, format, columns);
            Err(EngineError::NotCompiled)
        }
    }

    /// Get column names of a registered DuckDB table, in ordinal order.
    pub fn get_table_columns(&self, table_name: &str) -> Result<Vec<String>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            let sql = format!("DESCRIBE \"{}\"", table_name);
            let mut stmt = self
                .conn
                .prepare(&sql)
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let mut rows = stmt
                .query([])
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let mut columns = Vec::new();
            while let Some(row) = rows.next().map_err(|e| EngineError::ExecFailed(e.to_string()))? {
                let name: String = row.get(0).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
                columns.push(name);
            }
            Ok(columns)
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = table_name;
            Err(EngineError::NotCompiled)
        }
    }

    /// Initialize GPU buffer manager. Must be called before `execute_gpu`.
    pub fn init_gpu_buffers(&self, cache_size: &str, processing_size: &str) -> Result<(), EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            let sql = format!(
                "SELECT * FROM gpu_buffer_init('{}', '{}')",
                cache_size, processing_size
            );
            self.conn
                .execute_batch(&sql)
                .map_err(|e| EngineError::InitFailed(format!("gpu_buffer_init: {e}")))?;
            Ok(())
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = (cache_size, processing_size);
            Err(EngineError::NotCompiled)
        }
    }
}

/// Errors from the Sirius engine.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("engine not compiled (enable the `duckdb-bundled` feature)")]
    NotCompiled,

    #[error("engine initialization failed: {0}")]
    InitFailed(String),

    #[error("query execution failed: {0}")]
    ExecFailed(String),
}
