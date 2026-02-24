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

    #[cfg(feature = "duckdb-bundled")]
    has_substrait: bool,

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

            // Try to load locally-built extensions from the Sirius build output.
            // These are optional: SQL path works without them; Substrait/GPU
            // paths will fail at call time if the extensions are missing.
            //
            // Extension lookup order:
            //   1. SIRIUS_EXTENSION_DIR env var (for Docker/deployment)
            //   2. Compile-time paths relative to CARGO_MANIFEST_DIR
            let sirius_root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../..");
            let (substrait_ext, sirius_ext) = if let Ok(dir) = std::env::var("SIRIUS_EXTENSION_DIR") {
                (
                    format!("{}/substrait.duckdb_extension", dir),
                    format!("{}/sirius.duckdb_extension", dir),
                )
            } else {
                (
                    format!(
                        "{}/doris/thirdparty/duckdb-substrait-extension/build/release/extension/substrait/substrait.duckdb_extension",
                        sirius_root
                    ),
                    format!(
                        "{}/build/release/extension/sirius/sirius.duckdb_extension",
                        sirius_root
                    ),
                )
            };
            let mut has_substrait = false;
            if let Err(e) = conn.execute_batch(&format!("LOAD '{}'", substrait_ext)) {
                eprintln!("warning: substrait extension not loaded: {e}");
            } else {
                has_substrait = true;
            }
            if let Err(e) = conn.execute_batch(&format!("LOAD '{}'", sirius_ext)) {
                eprintln!("warning: sirius extension not loaded: {e}");
            }

            // Register an "if" macro so DuckDB can handle Substrait IfThen expressions.
            // The DuckDB Substrait extension maps IfThen to a scalar function named "if",
            // which doesn't exist as a built-in. Multi-clause CASE generates nested if() calls.
            conn.execute_batch(
                "CREATE MACRO \"if\"(cond, then_val, else_val) AS CASE WHEN cond THEN then_val ELSE else_val END"
            ).ok(); // Non-fatal if it fails

            Ok(Self { conn, has_substrait })
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            Err(EngineError::NotCompiled)
        }
    }

    /// Enable Sirius `enable_fallback_check` — throws for unsupported GPU ops.
    pub fn set_no_cpu_fallback(&self) -> Result<(), EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            self.conn
                .execute_batch("SET enable_fallback_check = true")
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            Ok(())
        }
        #[cfg(not(feature = "duckdb-bundled"))]
        {
            Ok(())
        }
    }

    /// Whether the Substrait extension is loaded (needed for `from_substrait`).
    pub fn has_substrait(&self) -> bool {
        #[cfg(feature = "duckdb-bundled")]
        { self.has_substrait }
        #[cfg(not(feature = "duckdb-bundled"))]
        { false }
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
                .prepare("SELECT * FROM gpu_execution(?)")
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

    /// Execute a Substrait plan via Sirius GPU (`gpu_processing_substrait`),
    /// optionally with a SQL ORDER BY / LIMIT suffix.
    pub fn execute_substrait(&self, plan_bytes: &[u8], order_sql: Option<&str>) -> Result<Vec<u8>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            use arrow::record_batch::RecordBatch;

            let sql = if let Some(suffix) = order_sql {
                format!("SELECT * FROM gpu_processing_substrait(?::blob) {}", suffix)
            } else {
                "SELECT * FROM gpu_processing_substrait(?::blob)".to_string()
            };
            let mut stmt = self
                .conn
                .prepare(&sql)
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let batches: Vec<RecordBatch> = stmt
                .query_arrow(duckdb::params![plan_bytes])
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?
                .collect();
            batches_to_ipc(batches)
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = (plan_bytes, order_sql);
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

    /// Execute a Substrait plan via DuckDB CPU with an ORDER BY / LIMIT SQL suffix.
    ///
    /// DuckDB's `from_substrait()` doesn't reliably preserve sort order, so this
    /// wraps the call: `SELECT * FROM from_substrait(?::blob) ORDER BY 1 DESC LIMIT 10`.
    pub fn from_substrait_sorted(&self, plan_bytes: &[u8], order_sql: &str) -> Result<Vec<u8>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            use arrow::record_batch::RecordBatch;

            let sql = format!("SELECT * FROM from_substrait(?::blob) {}", order_sql);
            let mut stmt = self
                .conn
                .prepare(&sql)
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let batches: Vec<RecordBatch> = stmt
                .query_arrow(duckdb::params![plan_bytes])
                .map_err(|e| EngineError::ExecFailed(e.to_string()))?
                .collect();
            batches_to_ipc(batches)
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = (plan_bytes, order_sql);
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

    /// Register exchange data as a DuckDB table.
    ///
    /// Creates a table from decoded PBlock column data so that the Substrait plan
    /// can reference it via a NamedTable ReadRel. On CPU builds, this creates a
    /// real DuckDB table. With GPU, this would go through the C++ bridge to
    /// `GPUBufferManager::tables`.
    pub fn register_exchange_table(
        &self,
        table_name: &str,
        column_names: &[String],
        column_types_sql: &[String],
        num_rows: u32,
        column_data_csv: &[Vec<String>],
    ) -> Result<(), EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            if column_names.is_empty() || num_rows == 0 {
                // Create an empty table.
                let cols: Vec<String> = column_names
                    .iter()
                    .zip(column_types_sql.iter())
                    .map(|(n, t)| format!("\"{}\" {}", n, t))
                    .collect();
                let sql = format!(
                    "CREATE OR REPLACE TABLE \"{}\" ({})",
                    table_name,
                    cols.join(", ")
                );
                self.conn
                    .execute_batch(&sql)
                    .map_err(|e| EngineError::ExecFailed(format!("register_exchange_table: {e}")))?;
                return Ok(());
            }

            // Build a VALUES-based INSERT to load the data.
            let cols: Vec<String> = column_names
                .iter()
                .zip(column_types_sql.iter())
                .map(|(n, t)| format!("\"{}\" {}", n, t))
                .collect();
            let create_sql = format!(
                "CREATE OR REPLACE TABLE \"{}\" ({})",
                table_name,
                cols.join(", ")
            );
            self.conn
                .execute_batch(&create_sql)
                .map_err(|e| EngineError::ExecFailed(format!("register_exchange_table create: {e}")))?;

            // Insert in batches to avoid overly large SQL statements.
            let batch_size = 1000usize;
            let num_cols = column_names.len();
            for batch_start in (0..num_rows as usize).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_rows as usize);
                let mut rows = Vec::new();
                for row in batch_start..batch_end {
                    let mut vals = Vec::new();
                    for col in 0..num_cols {
                        vals.push(column_data_csv[col][row].clone());
                    }
                    rows.push(format!("({})", vals.join(", ")));
                }
                let insert_sql = format!(
                    "INSERT INTO \"{}\" VALUES {}",
                    table_name,
                    rows.join(", ")
                );
                self.conn
                    .execute_batch(&insert_sql)
                    .map_err(|e| EngineError::ExecFailed(format!("register_exchange_table insert: {e}")))?;
            }

            Ok(())
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = (table_name, column_names, column_types_sql, num_rows, column_data_csv);
            Err(EngineError::NotCompiled)
        }
    }

    /// Register GPU memory directly as a DuckDB table without CPU copy.
    ///
    /// Used by the nixl exchange path to make GPU-resident data available to
    /// DuckDB's Substrait executor. The GPU pointers and schema describe Arrow
    /// columnar buffers already in VRAM.
    ///
    /// On CPU builds (duckdb-bundled without GPU), this falls back to creating
    /// an empty table with the given schema — the actual data must be provided
    /// separately via `register_exchange_table`.
    pub fn register_gpu_exchange_table(
        &self,
        table_name: &str,
        column_names: &[String],
        column_types_sql: &[String],
        num_rows: u32,
        gpu_ptrs: &[(usize, usize)], // (addr, len) pairs per column
    ) -> Result<(), EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            // GPU table registration via the Sirius DuckDB extension's
            // gpu_register_exchange_table function.
            // If the extension isn't loaded, fall back to creating an empty table.
            let cols: Vec<String> = column_names
                .iter()
                .zip(column_types_sql.iter())
                .map(|(n, t)| format!("\"{}\" {}", n, t))
                .collect();
            let create_sql = format!(
                "CREATE OR REPLACE TABLE \"{}\" ({})",
                table_name,
                cols.join(", ")
            );
            self.conn
                .execute_batch(&create_sql)
                .map_err(|e| EngineError::ExecFailed(format!("register_gpu_exchange_table create: {e}")))?;

            // Try to use gpu_register_table to point DuckDB at GPU memory directly.
            // Format: gpu_register_table('table_name', [ptr1, ptr2, ...], [len1, len2, ...], num_rows)
            let ptrs_str: Vec<String> = gpu_ptrs.iter().map(|(addr, _)| format!("{}", addr)).collect();
            let lens_str: Vec<String> = gpu_ptrs.iter().map(|(_, len)| format!("{}", len)).collect();
            let sql = format!(
                "SELECT * FROM gpu_register_table('{}', [{}], [{}], {})",
                table_name,
                ptrs_str.join(", "),
                lens_str.join(", "),
                num_rows
            );
            match self.conn.execute_batch(&sql) {
                Ok(()) => {
                    tracing::info!(table = table_name, "registered GPU exchange table via gpu_register_table");
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        table = table_name,
                        "gpu_register_table not available, table created as empty schema"
                    );
                    // Table already created with schema above — caller can populate via
                    // register_exchange_table CPU path as fallback.
                }
            }

            Ok(())
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = (table_name, column_names, column_types_sql, num_rows, gpu_ptrs);
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

    /// Allocate GPU buffers for receiving nixl transfer data.
    ///
    /// Asks the Sirius GPU buffer manager to allocate `num_buffers` buffers of
    /// the given sizes. Returns (addr, len, device_id) for each allocated buffer.
    /// Used by the nixl metadata exchange receiver to prepare destination buffers.
    pub fn allocate_gpu_buffers(
        &self,
        sizes: &[(usize, u64)], // (len, device_id) per buffer
    ) -> Result<Vec<(usize, usize, u64)>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            // Call the Sirius extension's gpu_allocate_buffers function.
            // It takes a list of sizes and returns allocated GPU addresses.
            let sizes_str: Vec<String> = sizes.iter().map(|(len, _)| format!("{}", len)).collect();
            let device_id = sizes.first().map(|(_, d)| *d).unwrap_or(0);
            let sql = format!(
                "SELECT buffer_id, addr, len FROM gpu_allocate_buffers([{}], {})",
                sizes_str.join(", "),
                device_id
            );

            let mut stmt = self
                .conn
                .prepare(&sql)
                .map_err(|e| EngineError::ExecFailed(format!("gpu_allocate_buffers: {e}")))?;

            let mut rows = stmt
                .query([])
                .map_err(|e| EngineError::ExecFailed(format!("query allocated buffers: {e}")))?;

            let mut result = Vec::new();
            while let Some(row) = rows.next().map_err(|e| EngineError::ExecFailed(e.to_string()))? {
                let addr: i64 = row.get(1).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
                let len: i64 = row.get(2).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
                result.push((addr as usize, len as usize, device_id));
            }

            Ok(result)
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            let _ = sizes;
            Err(EngineError::NotCompiled)
        }
    }

    /// Get GPU buffer pointers from the last execution (for nixl GPU-direct exchange).
    ///
    /// Returns `Ok(Some(...))` if the last query was GPU-accelerated and buffers are
    /// still resident in GPU memory. Returns `Ok(None)` if executed on CPU or buffers
    /// have been freed. Returns `Err` on query failure.
    ///
    /// The returned structure contains:
    /// - `buffer_addrs`: Vec of (addr, len, device_id) for each Arrow buffer
    /// - `column_info`: Vec of (column_name, type_id) pairs
    /// - `num_rows`: Total row count
    /// - `schema_ipc`: Arrow IPC schema bytes (for receiver reconstruction)
    pub fn get_last_gpu_result_buffers(&self) -> Result<Option<GpuResultInfo>, EngineError> {
        #[cfg(feature = "duckdb-bundled")]
        {
            // Query the Sirius extension for GPU buffer information.
            // This uses a special function: `sirius_get_last_gpu_buffers()` that
            // returns a table with columns: buffer_id, addr, len, device_id, column_name, type_id, num_rows.
            //
            // If the last query was CPU-only or buffers have been freed, returns empty set.

            let sql = "SELECT buffer_id, addr, len, device_id, column_name, type_id, num_rows FROM sirius_get_last_gpu_buffers()";
            let mut stmt = self
                .conn
                .prepare(sql)
                .map_err(|e| EngineError::ExecFailed(format!("sirius_get_last_gpu_buffers: {e}")))?;

            let mut rows = stmt
                .query([])
                .map_err(|e| EngineError::ExecFailed(format!("query gpu buffers: {e}")))?;

            let mut buffer_addrs = Vec::new();
            let mut column_info = Vec::new();
            let mut num_rows_opt = None;

            while let Some(row) = rows.next().map_err(|e| EngineError::ExecFailed(e.to_string()))? {
                let addr: i64 = row.get(1).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
                let len: i64 = row.get(2).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
                let device_id: i64 = row.get(3).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
                let column_name: String = row.get(4).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
                let type_id: i32 = row.get(5).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
                let num_rows: i64 = row.get(6).map_err(|e| EngineError::ExecFailed(e.to_string()))?;

                buffer_addrs.push((addr as usize, len as usize, device_id as u64));
                column_info.push((column_name, type_id));
                num_rows_opt = Some(num_rows as u32);
            }

            if buffer_addrs.is_empty() {
                return Ok(None);
            }

            // Get schema IPC bytes (query the schema without data).
            let schema_ipc = Vec::new(); // TODO: extract schema from last query

            Ok(Some(GpuResultInfo {
                buffer_addrs,
                column_info,
                num_rows: num_rows_opt.unwrap_or(0),
                schema_ipc,
            }))
        }

        #[cfg(not(feature = "duckdb-bundled"))]
        {
            Err(EngineError::NotCompiled)
        }
    }
}

/// GPU result buffer information for nixl transfers.
#[derive(Debug, Clone)]
pub struct GpuResultInfo {
    /// GPU buffer addresses: (addr, len, device_id).
    pub buffer_addrs: Vec<(usize, usize, u64)>,
    /// Column metadata: (name, type_id).
    pub column_info: Vec<(String, i32)>,
    /// Number of rows.
    pub num_rows: u32,
    /// Arrow IPC schema bytes.
    pub schema_ipc: Vec<u8>,
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
