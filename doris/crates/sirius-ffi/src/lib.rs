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

impl SiriusEngine {
    /// Create a new engine.
    pub fn new() -> Result<Self, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            let conn = duckdb::Connection::open_in_memory()
                .map_err(|e| EngineError::InitFailed(e.to_string()))?;
            Ok(Self { conn })
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            Err(EngineError::NotCompiled)
        }
    }

    /// Execute a SQL query and return Arrow IPC stream bytes.
    ///
    /// The returned bytes contain a complete Arrow IPC stream (schema message
    /// followed by zero or more record-batch messages and an EOS marker).
    pub fn execute_sql(&self, sql: &str) -> Result<Vec<u8>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            use arrow::ipc::writer::StreamWriter;
            use arrow::record_batch::RecordBatch;

            let mut stmt = self
                .conn
                .prepare(sql)
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let batches: Vec<RecordBatch> = stmt
                .query_arrow([])
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?
                .collect();

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

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = sql;
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
