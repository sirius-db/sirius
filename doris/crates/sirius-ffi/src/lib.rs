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

            // Load the locally-built substrait extension.
            let ext_path = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../thirdparty/duckdb-substrait-extension/build/release/extension/substrait/substrait.duckdb_extension"
            );
            let load_sql = format!("LOAD '{}'", ext_path);
            conn.execute_batch(&load_sql)
                .map_err(|e| EngineError::InitFailed(format!("load substrait extension: {e}")))?;

            Ok(Self { conn })
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            Err(EngineError::NotCompiled)
        }
    }

    /// Execute a SQL query and return Arrow IPC stream bytes.
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

    /// Execute a Substrait plan (protobuf bytes) and return Arrow IPC stream bytes.
    pub fn execute_substrait(&self, plan_bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
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
