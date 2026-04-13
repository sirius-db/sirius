//! Sirius query execution engine.
//!
//! Provides a unified API for executing SQL queries and returning Arrow IPC bytes.
//! Uses a bundled DuckDB instance with Sirius GPU extensions.

use std::ffi::{c_char, c_void, CStr};
use std::sync::OnceLock;

/// High-level query execution engine.
///
/// Wraps a DuckDB connection and provides a safe API for executing SQL queries
/// that return Arrow IPC stream bytes.
pub struct SiriusEngine {
    conn: duckdb::Connection,
    has_substrait: bool,
    exchange_api: &'static ExchangeApi,
}

type BeginExchangeCaptureFn = unsafe extern "C" fn(u64, *const i32, usize) -> u64;
type TakeExchangeArtifactFn = unsafe extern "C" fn() -> *mut c_void;
type DestroyExchangeArtifactFn = unsafe extern "C" fn(*mut c_void);
type ArtifactStagingBaseFn = unsafe extern "C" fn(*const c_void) -> u64;
type ArtifactCountFn = unsafe extern "C" fn(*const c_void) -> u64;
type ArtifactGetPartitionFn = unsafe extern "C" fn(*const c_void, usize, *mut RawPackedPartitionView) -> i32;
type ArtifactGetBroadcastFn = unsafe extern "C" fn(*const c_void, usize, *mut RawPackedBroadcastEntryView) -> i32;

#[derive(Clone, Copy)]
struct ExchangeApi {
    last_error: unsafe extern "C" fn() -> *const c_char,
    finalize_exchange_tables_direct: unsafe extern "C" fn() -> i32,
    begin_exchange_capture: BeginExchangeCaptureFn,
    take_exchange_artifact: TakeExchangeArtifactFn,
    exchange_artifact_destroy: DestroyExchangeArtifactFn,
    exchange_artifact_staging_base: ArtifactStagingBaseFn,
    exchange_artifact_partition_count: ArtifactCountFn,
    exchange_artifact_get_partition: ArtifactGetPartitionFn,
    exchange_artifact_broadcast_count: ArtifactCountFn,
    exchange_artifact_get_broadcast_entry: ArtifactGetBroadcastFn,
}

#[repr(C)]
#[derive(Default)]
struct RawPackedPartitionView {
    partition_id: u64,
    staging_offset: u64,
    packed_size: u64,
    metadata_ptr: *const u8,
    metadata_len: u64,
    num_rows: i32,
    overflow_gpu_addr: u64,
}

#[repr(C)]
#[derive(Default)]
struct RawPackedBroadcastEntryView {
    entry_id: u64,
    staging_offset: u64,
    packed_size: u64,
    metadata_ptr: *const u8,
    metadata_len: u64,
    num_rows: i32,
    overflow_gpu_addr: u64,
}

static EXCHANGE_API: OnceLock<ExchangeApi> = OnceLock::new();
static EXCHANGE_API_PATH: OnceLock<String> = OnceLock::new();

fn load_exchange_api(sirius_ext_path: &str) -> Result<&'static ExchangeApi, EngineError> {
    if let Some(api) = EXCHANGE_API.get() {
        if let Some(path) = EXCHANGE_API_PATH.get() {
            if path != sirius_ext_path {
                return Err(EngineError::InitFailed(format!(
                    "sirius exchange API already loaded from {path}, cannot switch to {sirius_ext_path}"
                )));
            }
        }
        return Ok(api);
    }

    let lib = unsafe { libloading::Library::new(sirius_ext_path) }
        .map_err(|e| EngineError::InitFailed(format!("load sirius exchange API: {e}")))?;
    let lib = Box::leak(Box::new(lib));
    let api = ExchangeApi {
        last_error: unsafe { load_symbol(lib, b"sirius_exchange_last_error\0")? },
        finalize_exchange_tables_direct: unsafe {
            load_symbol(lib, b"sirius_finalize_exchange_tables_direct\0")?
        },
        begin_exchange_capture: unsafe { load_symbol(lib, b"sirius_begin_exchange_capture\0")? },
        take_exchange_artifact: unsafe { load_symbol(lib, b"sirius_take_exchange_artifact\0")? },
        exchange_artifact_destroy: unsafe {
            load_symbol(lib, b"sirius_exchange_artifact_destroy\0")?
        },
        exchange_artifact_staging_base: unsafe {
            load_symbol(lib, b"sirius_exchange_artifact_staging_base\0")?
        },
        exchange_artifact_partition_count: unsafe {
            load_symbol(lib, b"sirius_exchange_artifact_partition_count\0")?
        },
        exchange_artifact_get_partition: unsafe {
            load_symbol(lib, b"sirius_exchange_artifact_get_partition\0")?
        },
        exchange_artifact_broadcast_count: unsafe {
            load_symbol(lib, b"sirius_exchange_artifact_broadcast_count\0")?
        },
        exchange_artifact_get_broadcast_entry: unsafe {
            load_symbol(lib, b"sirius_exchange_artifact_get_broadcast_entry\0")?
        },
    };

    let _ = EXCHANGE_API_PATH.set(sirius_ext_path.to_string());
    EXCHANGE_API
        .set(api)
        .map_err(|_| EngineError::InitFailed("exchange API already initialized".into()))?;
    Ok(EXCHANGE_API.get().expect("exchange API just initialized"))
}

unsafe fn load_symbol<T: Copy>(
    lib: &'static libloading::Library,
    name: &[u8],
) -> Result<T, EngineError> {
    let symbol: libloading::Symbol<T> = lib
        .get(name)
        .map_err(|e| EngineError::InitFailed(format!("load symbol {}: {e}", String::from_utf8_lossy(name))))?;
    Ok(*symbol)
}

fn exchange_last_error(api: &'static ExchangeApi) -> String {
    let ptr = unsafe { (api.last_error)() };
    if ptr.is_null() {
        return "unknown sirius exchange error".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .trim()
        .to_string()
}

fn copy_raw_bytes(ptr: *const u8, len: usize) -> Vec<u8> {
    if ptr.is_null() || len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }
}

/// Serialize Arrow record batches into an IPC stream.
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

/// Serialize an Arrow schema into a schema-only IPC stream.
fn schema_to_ipc(schema: &arrow::datatypes::SchemaRef) -> Result<Vec<u8>, EngineError> {
    use arrow::ipc::writer::StreamWriter;

    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema)
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
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
        // Allow unsigned extensions (needed for locally-built extensions).
        let config = duckdb::Config::default()
            .with("allow_unsigned_extensions", "true")
            .map_err(|e| EngineError::InitFailed(e.to_string()))?;
        let conn = duckdb::Connection::open_in_memory_with_flags(config)
            .map_err(|e| EngineError::InitFailed(e.to_string()))?;

        // Load required DuckDB extensions (substrait + sirius GPU engine).
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
                    "{}/substrait/build/release/extension/substrait/substrait.duckdb_extension",
                    sirius_root
                ),
                format!(
                    "{}/build/release/extension/sirius/sirius.duckdb_extension",
                    sirius_root
                ),
            )
        };
        // Load extensions — tolerate "already loaded" when the extension is
        // preloaded by DuckDB's autoload or a previous connection in the same process.
        // Do NOT tolerate "already loaded in another process" — that means a lock file
        // conflict and the extension did not actually load.
        if let Err(e) = conn.execute_batch(&format!("LOAD '{}'", substrait_ext)) {
            let msg = e.to_string();
            if msg.contains("already loaded") && !msg.contains("another process") {
                eprintln!("substrait extension already loaded, continuing");
            } else {
                return Err(EngineError::InitFailed(format!("substrait extension: {e}")));
            }
        }
        let has_substrait = true;
        if let Err(e) = conn.execute_batch(&format!("LOAD '{}'", sirius_ext)) {
            let msg = e.to_string();
            if msg.contains("already loaded") && !msg.contains("another process") {
                eprintln!("sirius extension already loaded, continuing");
            } else {
                return Err(EngineError::InitFailed(format!("sirius extension: {e}")));
            }
        }

        // Register an "if" macro so DuckDB can handle Substrait IfThen expressions.
        // The DuckDB Substrait extension maps IfThen to a scalar function named "if",
        // which doesn't exist as a built-in. Multi-clause CASE generates nested if() calls.
        conn.execute_batch(
            "CREATE MACRO \"if\"(cond, then_val, else_val) AS CASE WHEN cond THEN then_val ELSE else_val END"
        ).ok(); // Non-fatal if it fails

        let exchange_api = load_exchange_api(&sirius_ext)?;

        Ok(Self {
            conn,
            has_substrait,
            exchange_api,
        })
    }

    /// Enable Sirius `enable_fallback_check` — throws for unsupported GPU ops.
    pub fn set_no_cpu_fallback(&self) -> Result<(), EngineError> {
        self.conn
            .execute_batch("SET enable_fallback_check = true")
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        Ok(())
    }

    /// Whether the Substrait extension is loaded (needed for `from_substrait`).
    pub fn has_substrait(&self) -> bool {
        self.has_substrait
    }

    /// Execute a SQL query directly via DuckDB (CPU fallback).
    pub fn execute_sql(&self, sql: &str) -> Result<Vec<u8>, EngineError> {
        use arrow::record_batch::RecordBatch;

        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let arrow_result = stmt
            .query_arrow([])
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let schema = arrow_result.get_schema();
        let batches: Vec<RecordBatch> = arrow_result.collect();
        if batches.is_empty() {
            return schema_to_ipc(&schema);
        }
        batches_to_ipc(batches)
    }

    /// Execute a SQL query via Sirius GPU (`gpu_execution` table function).
    pub fn execute_gpu(&self, sql: &str) -> Result<Vec<u8>, EngineError> {
        use arrow::record_batch::RecordBatch;

        let mut stmt = self
            .conn
            .prepare("SELECT * FROM gpu_execution(?)")
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let arrow_result = stmt
            .query_arrow(duckdb::params![sql])
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let schema = arrow_result.get_schema();
        let batches: Vec<RecordBatch> = arrow_result.collect();
        if batches.is_empty() {
            return schema_to_ipc(&schema);
        }
        batches_to_ipc(batches)
    }

    /// Execute a Substrait plan via Sirius GPU (`gpu_execution_substrait`),
    /// optionally with a SQL ORDER BY / LIMIT suffix.
    pub fn execute_substrait(&self, plan_bytes: &[u8], order_sql: Option<&str>) -> Result<Vec<u8>, EngineError> {
        use arrow::record_batch::RecordBatch;

        let sql = if let Some(suffix) = order_sql {
            format!("SELECT * FROM gpu_execution_substrait(?::blob) {}", suffix)
        } else {
            "SELECT * FROM gpu_execution_substrait(?::blob)".to_string()
        };
        let t0 = std::time::Instant::now();
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        eprintln!("[sirius-ffi] execute_substrait: prepare took {}ms", t0.elapsed().as_millis());
        let t1 = std::time::Instant::now();
        let arrow_result = stmt
            .query_arrow(duckdb::params![plan_bytes])
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let schema = arrow_result.get_schema();
        let batches: Vec<RecordBatch> = arrow_result.collect();
        eprintln!("[sirius-ffi] execute_substrait: query_arrow took {}ms, {} batches", t1.elapsed().as_millis(), batches.len());
        if batches.is_empty() {
            return schema_to_ipc(&schema);
        }
        let t2 = std::time::Instant::now();
        let result = batches_to_ipc(batches);
        eprintln!("[sirius-ffi] execute_substrait: batches_to_ipc took {}ms", t2.elapsed().as_millis());
        result
    }

    /// Execute a Substrait plan via DuckDB CPU (`from_substrait`).
    pub fn from_substrait(&self, plan_bytes: &[u8]) -> Result<Vec<u8>, EngineError> {
        use arrow::record_batch::RecordBatch;

        let mut stmt = self
            .conn
            .prepare("SELECT * FROM from_substrait(?::blob)")
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let arrow_result = stmt
            .query_arrow(duckdb::params![plan_bytes])
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let schema = arrow_result.get_schema();
        let batches: Vec<RecordBatch> = arrow_result.collect();
        tracing::info!(
            batch_count = batches.len(),
            row_count = batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            "from_substrait completed"
        );
        if batches.is_empty() {
            return schema_to_ipc(&schema);
        }
        batches_to_ipc(batches)
    }

    /// Execute a Substrait plan via DuckDB CPU with an ORDER BY / LIMIT SQL suffix.
    ///
    /// DuckDB's `from_substrait()` doesn't reliably preserve sort order, so this
    /// wraps the call: `SELECT * FROM from_substrait(?::blob) ORDER BY 1 DESC LIMIT 10`.
    pub fn from_substrait_sorted(&self, plan_bytes: &[u8], order_sql: &str) -> Result<Vec<u8>, EngineError> {
        use arrow::record_batch::RecordBatch;

        let sql = format!("SELECT * FROM from_substrait(?::blob) {}", order_sql);
        let mut stmt = self
            .conn
            .prepare(&sql)
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let arrow_result = stmt
            .query_arrow(duckdb::params![plan_bytes])
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let schema = arrow_result.get_schema();
        let batches: Vec<RecordBatch> = arrow_result.collect();
        tracing::info!(
            batch_count = batches.len(),
            row_count = batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            order_sql,
            "from_substrait_sorted completed"
        );
        if batches.is_empty() {
            return schema_to_ipc(&schema);
        }
        batches_to_ipc(batches)
    }

    /// Apply ORDER BY / LIMIT / OFFSET SQL to an already materialized Arrow IPC result.
    ///
    /// This is used as a fallback for CPU paths where DuckDB's Substrait SortRel
    /// or `SELECT * FROM from_substrait(...) ORDER BY ...` produce incorrect results.
    pub fn sort_ipc_result(&self, ipc_bytes: &[u8], sql_suffix: &str) -> Result<Vec<u8>, EngineError> {
        const TEMP_TABLE: &str = "__SIRIUS_POSTSORT";

        let _ = self.conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{}\"", TEMP_TABLE));
        self.register_exchange_table_from_ipc(TEMP_TABLE, ipc_bytes)?;
        let sql = format!("SELECT * FROM \"{}\" {}", TEMP_TABLE, sql_suffix);
        let result = self.execute_sql(&sql);
        let _ = self.conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{}\"", TEMP_TABLE));
        result
    }

    /// Get the Arrow schema of a file as IPC stream bytes (schema only, no data).
    ///
    /// Uses `LIMIT 0` + `query_arrow().get_schema()` to extract the schema from
    /// the prepared statement metadata, without reading any data rows.
    pub fn get_file_schema_ipc(&self, file_path: &str, format: &str) -> Result<Vec<u8>, EngineError> {
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

    /// Register a file as a DuckDB table (e.g. Parquet, CSV, JSON).
    ///
    /// When `columns` is non-empty, only those columns are loaded (in order),
    /// ensuring DuckDB's table schema matches the Substrait ReadRel field references.
    /// This must be called before executing a plan that references the table.
    pub fn register_file_table(&self, table_name: &str, file_path: &str, format: &str, columns: &[String]) -> Result<(), EngineError> {
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
        // Drop any existing VIEW with this name first — DuckDB can't replace a VIEW with a TABLE.
        let _ = self.conn.execute_batch(&format!("DROP VIEW IF EXISTS \"{}\"", table_name));
        let sql = format!(
            "CREATE OR REPLACE TABLE \"{}\" AS SELECT {} FROM {}('{}')",
            table_name, select, reader_fn, file_path
        );
        self.conn
            .execute_batch(&sql)
            .map_err(|e| EngineError::ExecFailed(format!("register_file_table: {e}")))?;
        Ok(())
    }

    /// Register a DuckDB VIEW over multiple parquet files.
    ///
    /// Creates `CREATE OR REPLACE VIEW "name" AS SELECT * FROM parquet_scan([file1, ...])`.
    /// DuckDB resolves views to parquet_scan at plan time, which Sirius routes to
    /// sirius_physical_parquet_scan for GPU-native parquet reading.
    pub fn register_parquet_view(&self, table_name: &str, file_paths: &[String]) -> Result<(), EngineError> {
        let file_list = file_paths
            .iter()
            .map(|p| format!("'{}'", p))
            .collect::<Vec<_>>()
            .join(", ");
        // Drop any existing TABLE with this name first — DuckDB can't replace a TABLE with a VIEW.
        let _ = self.conn.execute_batch(&format!("DROP TABLE IF EXISTS \"{}\"", table_name));
        let sql = format!(
            "CREATE OR REPLACE VIEW \"{}\" AS SELECT * FROM parquet_scan([{}])",
            table_name, file_list
        );
        self.conn
            .execute_batch(&sql)
            .map_err(|e| EngineError::ExecFailed(format!("register_parquet_view: {e}")))?;
        Ok(())
    }

    /// Get column names of a parquet file without materializing it as a DuckDB table.
    ///
    /// Runs `DESCRIBE SELECT * FROM read_parquet('...')` to extract column names.
    /// Used for LocalFiles scan path where we don't need to CREATE TABLE.
    pub fn get_parquet_columns(&self, file_path: &str) -> Result<Vec<String>, EngineError> {
        let sql = format!("DESCRIBE SELECT * FROM read_parquet('{}')", file_path);
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

    /// Get column names of a registered DuckDB table, in ordinal order.
    pub fn get_table_columns(&self, table_name: &str) -> Result<Vec<String>, EngineError> {
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

    /// Register exchange data as a DuckDB table.
    ///
    /// Creates a table from decoded PBlock column data so that the Substrait plan
    /// can reference it via a NamedTable ReadRel.
    pub fn register_exchange_table(
        &self,
        table_name: &str,
        column_names: &[String],
        column_types_sql: &[String],
        num_rows: u32,
        column_data_csv: &[Vec<String>],
    ) -> Result<(), EngineError> {
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

    /// Register an exchange table from Arrow IPC bytes via DuckDB's Arrow Appender.
    ///
    /// This is orders of magnitude faster than SQL VALUES for large datasets
    /// (6M rows: <1s vs minutes). Deserializes Arrow IPC → RecordBatch, then
    /// uses DuckDB's `Appender::append_record_batch()` for zero-copy insertion.
    pub fn register_exchange_table_from_ipc(
        &self,
        table_name: &str,
        ipc_bytes: &[u8],
    ) -> Result<(), EngineError> {
        use arrow::ipc::reader::StreamReader;
        use std::io::Cursor;

        // Deserialize Arrow IPC stream → RecordBatches.
        let cursor = Cursor::new(ipc_bytes);
        let reader = StreamReader::try_new(cursor, None)
            .map_err(|e| EngineError::ExecFailed(format!("Arrow IPC read: {e}")))?;

        let schema = reader.schema();

        // Create the table with the right schema.
        // Sanitize column names: expression text (e.g. `sum("if"((...))`) from
        // Doris PBlock metadata breaks SQL parsing. Use positional aliases.
        let cols: Vec<String> = schema.fields().iter().enumerate().map(|(i, f)| {
            let dt = arrow_type_to_duckdb_sql(f.data_type());
            let name = f.name();
            let safe_name = if name.chars().all(|c| c.is_alphanumeric() || c == '_') && !name.is_empty() {
                name.to_string()
            } else {
                format!("col_{}", i)
            };
            format!("\"{}\" {}", safe_name, dt)
        }).collect();
        let create_sql = format!(
            "CREATE OR REPLACE TABLE \"{}\" ({})",
            table_name, cols.join(", ")
        );
        self.conn
            .execute_batch(&create_sql)
            .map_err(|e| EngineError::ExecFailed(format!("create exchange table: {e}")))?;

        // Use DuckDB's Arrow Appender for bulk insertion.
        let mut appender = self.conn
            .appender(table_name)
            .map_err(|e| EngineError::ExecFailed(format!("create appender: {e}")))?;

        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| EngineError::ExecFailed(format!("read Arrow batch: {e}")))?;
            appender
                .append_record_batch(batch)
                .map_err(|e| EngineError::ExecFailed(format!("append batch: {e}")))?;
        }

        // Make the appender boundary explicit before later from_substrait calls
        // open a fresh connection and read the exchange table.
        appender
            .flush()
            .map_err(|e| EngineError::ExecFailed(format!("flush exchange table appender: {e}")))?;
        drop(appender);

        Ok(())
    }

    /// Register GPU memory directly as a DuckDB table without CPU copy.
    ///
    /// Used by the nixl exchange path to make GPU-resident data available to
    /// DuckDB's Substrait executor.
    pub fn register_gpu_exchange_table(
        &self,
        table_name: &str,
        column_names: &[String],
        column_types_sql: &[String],
        num_rows: u32,
        gpu_ptrs: &[(usize, usize)], // (addr, len) pairs per column
    ) -> Result<(), EngineError> {
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
            }
        }

        Ok(())
    }

    /// Initialize GPU buffer manager. Must be called before `execute_gpu`.
    pub fn init_gpu_buffers(&self, cache_size: &str, processing_size: &str) -> Result<(), EngineError> {
        let sql = format!(
            "SELECT * FROM gpu_buffer_init('{}', '{}')",
            cache_size, processing_size
        );
        self.conn
            .execute_batch(&sql)
            .map_err(|e| EngineError::InitFailed(format!("gpu_buffer_init: {e}")))?;
        Ok(())
    }

    /// Allocate GPU buffers for receiving nixl transfer data.
    ///
    /// Asks the Sirius GPU buffer manager to allocate `num_buffers` buffers of
    /// the given sizes. Returns (addr, len, device_id) for each allocated buffer.
    pub fn allocate_gpu_buffers(
        &self,
        sizes: &[(usize, u64)], // (len, device_id) per buffer
    ) -> Result<Vec<(usize, usize, u64)>, EngineError> {
        // Call the Sirius extension's gpu_allocate_buffers function.
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

    /// Get the RMM processing pool base address and size.
    ///
    /// Returns `(base_addr, size, device_id)` for the GPU processing pool.
    /// Used by nixl to register the pool region for GPU-direct transfers.
    pub fn get_pool_info(&self) -> Result<Option<(usize, usize, u32)>, EngineError> {
        let mut stmt = self.conn
            .prepare("SELECT pool_base, pool_size, device_id FROM sirius_get_pool_info()")
            .map_err(|e| EngineError::ExecFailed(format!("sirius_get_pool_info: {e}")))?;
        let batches: Vec<_> = stmt.query_arrow([])
            .map_err(|e| EngineError::ExecFailed(format!("sirius_get_pool_info: {e}")))?
            .collect();
        if batches.is_empty() || batches[0].num_rows() == 0 {
            return Ok(None);
        }
        let batch = &batches[0];
        use arrow::array::{Int32Array, Int64Array};
        let base = batch.column(0).as_any().downcast_ref::<Int64Array>()
            .ok_or_else(|| EngineError::ExecFailed("pool_base not i64".into()))?.value(0) as usize;
        let size = batch.column(1).as_any().downcast_ref::<Int64Array>()
            .ok_or_else(|| EngineError::ExecFailed("pool_size not i64".into()))?.value(0) as usize;
        let device = batch.column(2).as_any().downcast_ref::<Int32Array>()
            .ok_or_else(|| EngineError::ExecFailed("device_id not i32".into()))?.value(0) as u32;
        Ok(Some((base, size, device)))
    }

    /// Finalize any pending exchange tables before staging-backed buffers are reused.
    pub fn finalize_exchange_tables_direct(&self) -> Result<(), EngineError> {
        let ok = unsafe { (self.exchange_api.finalize_exchange_tables_direct)() };
        if ok == 0 {
            return Err(EngineError::ExecFailed(exchange_last_error(self.exchange_api)));
        }
        Ok(())
    }

    /// Begin capturing the next GPU execution into an owned exchange artifact.
    ///
    /// `partition_spec` configures the result collector for hash-partitioned
    /// exchange. When `None`, the collector uses the broadcast packing path.
    pub fn begin_exchange_capture(
        &self,
        partition_spec: Option<(usize, &[i32])>,
    ) -> Result<u64, EngineError> {
        let (num_partitions, cols) = partition_spec
            .map(|(num, cols)| (num as u64, cols))
            .unwrap_or((0, &[][..]));
        let session_id = unsafe {
            (self.exchange_api.begin_exchange_capture)(num_partitions, cols.as_ptr(), cols.len())
        };
        if session_id == 0 {
            return Err(EngineError::ExecFailed(exchange_last_error(self.exchange_api)));
        }
        Ok(session_id)
    }

    /// Move the active exchange capture into a Rust-owned artifact.
    pub fn take_exchange_artifact(&self) -> Result<Option<ExchangeArtifact>, EngineError> {
        let handle = unsafe { (self.exchange_api.take_exchange_artifact)() };
        if handle.is_null() {
            let last_error = exchange_last_error(self.exchange_api);
            if !last_error.is_empty() {
                return Err(EngineError::ExecFailed(last_error));
            }
            return Ok(None);
        }

        let staging_base =
            unsafe { (self.exchange_api.exchange_artifact_staging_base)(handle as *const c_void) }
                as usize;

        let partition_count =
            unsafe { (self.exchange_api.exchange_artifact_partition_count)(handle as *const c_void) }
                as usize;
        let mut packed_partitions = Vec::with_capacity(partition_count);
        for index in 0..partition_count {
            let mut raw = RawPackedPartitionView::default();
            let ok = unsafe {
                (self.exchange_api.exchange_artifact_get_partition)(
                    handle as *const c_void,
                    index,
                    &mut raw,
                )
            };
            if ok == 0 {
                unsafe { (self.exchange_api.exchange_artifact_destroy)(handle) };
                return Err(EngineError::ExecFailed(exchange_last_error(self.exchange_api)));
            }
            packed_partitions.push(PackedPartition {
                partition_id: raw.partition_id as usize,
                staging_offset: raw.staging_offset as usize,
                packed_size: raw.packed_size as usize,
                metadata: copy_raw_bytes(raw.metadata_ptr, raw.metadata_len as usize),
                num_rows: raw.num_rows as u32,
                overflow_gpu_addr: raw.overflow_gpu_addr as usize,
            });
        }

        let broadcast_count =
            unsafe { (self.exchange_api.exchange_artifact_broadcast_count)(handle as *const c_void) }
                as usize;
        let mut packed_broadcast = Vec::with_capacity(broadcast_count);
        for index in 0..broadcast_count {
            let mut raw = RawPackedBroadcastEntryView::default();
            let ok = unsafe {
                (self.exchange_api.exchange_artifact_get_broadcast_entry)(
                    handle as *const c_void,
                    index,
                    &mut raw,
                )
            };
            if ok == 0 {
                unsafe { (self.exchange_api.exchange_artifact_destroy)(handle) };
                return Err(EngineError::ExecFailed(exchange_last_error(self.exchange_api)));
            }
            packed_broadcast.push(PackedBroadcastEntry {
                entry_id: raw.entry_id as usize,
                staging_offset: raw.staging_offset as usize,
                packed_size: raw.packed_size as usize,
                metadata: copy_raw_bytes(raw.metadata_ptr, raw.metadata_len as usize),
                num_rows: raw.num_rows as u32,
                overflow_gpu_addr: raw.overflow_gpu_addr as usize,
            });
        }

        Ok(Some(ExchangeArtifact {
            handle,
            destroy: self.exchange_api.exchange_artifact_destroy,
            staging_base,
            packed_partitions,
            packed_broadcast,
        }))
    }

    /// Get GPU buffer pointers from the last execution (for nixl GPU-direct exchange).
    ///
    /// Returns `Ok(Some(...))` if the last query was GPU-accelerated and buffers are
    /// still resident in GPU memory. Returns `Ok(None)` if executed on CPU or buffers
    /// have been freed. Returns `Err` on query failure.
    pub fn get_last_gpu_result_buffers(&self) -> Result<Option<GpuResultInfo>, EngineError> {
        let sql = "SELECT buffer_id, addr, len, device_id, column_name, type_id, num_rows, null_mask_addr, null_mask_len, offsets_addr, offsets_len, null_count, scale FROM sirius_get_last_gpu_buffers()";
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| EngineError::ExecFailed(format!("sirius_get_last_gpu_buffers: {e}")))?;

        let mut rows = stmt
            .query([])
            .map_err(|e| EngineError::ExecFailed(format!("query gpu buffers: {e}")))?;

        let mut buffer_addrs = Vec::new();
        let mut column_info = Vec::new();
        let mut column_buffers = Vec::new();
        let mut num_rows_opt = None;

        while let Some(row) = rows.next().map_err(|e| EngineError::ExecFailed(e.to_string()))? {
            let addr: i64 = row.get(1).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let len: i64 = row.get(2).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let device_id: i64 = row.get(3).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let column_name: String = row.get(4).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let type_id: i32 = row.get(5).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let num_rows: i64 = row.get(6).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let null_mask_addr: i64 = row.get(7).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let null_mask_len: i64 = row.get(8).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let offsets_addr: i64 = row.get(9).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let offsets_len: i64 = row.get(10).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let null_count: i32 = row.get(11).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
            let scale: i32 = row.get(12).map_err(|e| EngineError::ExecFailed(e.to_string()))?;

            buffer_addrs.push((addr as usize, len as usize, device_id as u64));
            column_info.push((column_name, type_id));
            column_buffers.push(GpuColumnBuffers {
                null_mask_addr: null_mask_addr as usize,
                null_mask_len: null_mask_len as usize,
                offsets_addr: offsets_addr as usize,
                offsets_len: offsets_len as usize,
                null_count,
                scale,
            });
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
            column_buffers,
            num_rows: num_rows_opt.unwrap_or(0),
            schema_ipc,
        }))
    }

    /// Finalize a single exchange table's pending views.
    /// Must be called while the packed buffer data is still valid (leases alive).
    pub fn finalize_exchange_table(&self, table_name: &str) -> Result<(), EngineError> {
        let sql = format!("SELECT * FROM sirius_finalize_exchange_table('{}')", table_name);
        let mut stmt = self.conn
            .prepare(&sql)
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let _: Vec<_> = stmt.query_arrow([])
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?
            .collect();
        Ok(())
    }

    /// Tell the C++ result collector where the nixl staging buffer is.
    ///
    /// After this, GPU results will be packed directly into the staging buffer
    /// via cudf::chunked_pack — zero per-query nixl registrations.
    pub fn set_staging_buffer(&self, addr: usize, size: usize) -> Result<(), EngineError> {
        let sql = format!("SELECT * FROM sirius_set_staging_buffer({}, {})", addr as i64, size as i64);
        let mut stmt = self.conn
            .prepare(&sql)
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let _: Vec<_> = stmt.query_arrow([])
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?
            .collect();
        Ok(())
    }

    /// Register a packed GPU buffer as a DuckDB table via cudf::unpack + gpu_register_table.
    ///
    /// Unpacks the cudf metadata to get a table_view pointing into the GPU buffer,
    /// then registers the per-column GPU pointers with DuckDB's GPU execution engine.
    /// Zero CPU copies — the entire path is GPU→GPU.
    /// Unpack a cudf::pack'd GPU buffer and register it as a DuckDB table.
    ///
    /// Two steps:
    /// 1. Call C++ sirius_register_packed_table to cudf::unpack and store
    ///    per-column GPU pointers in LastGPUBuffers
    /// 2. Use register_gpu_exchange_table to create the DuckDB table and
    ///    call gpu_register_table with those pointers
    pub fn register_packed_table(
        &self,
        table_name: &str,
        gpu_addr: usize,
        gpu_size: usize,
        cudf_metadata: &[u8],
    ) -> Result<(), EngineError> {
        // Step 1: C++ cudf::unpack → registers in GPUBufferManager::tables (zero-copy).
        // Returns the CREATE TABLE SQL for us to execute (can't run inside the bind).
        let sql = "SELECT * FROM sirius_register_packed_table(?, ?, ?, ?)";
        let mut stmt = self.conn
            .prepare(sql)
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let mut rows = stmt.query(duckdb::params![
            table_name,
            gpu_addr as i64,
            gpu_size as i64,
            cudf_metadata,
        ])
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let row = rows.next()
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?
            .ok_or_else(|| EngineError::ExecFailed("register_packed_table returned no rows".into()))?;
        let create_table_sql: String = row.get(2)
            .map_err(|e| EngineError::ExecFailed(format!("get create_table_sql: {e}")))?;
        drop(rows);
        drop(stmt);

        // Step 2: Create schema-only DuckDB table on THIS connection.
        self.conn.execute_batch(&create_table_sql)
            .map_err(|e| EngineError::ExecFailed(format!("CREATE TABLE for GPU exchange: {e}")))?;

        Ok(())
    }

    /// Allocate GPU memory via the C++ CUDA runtime (cudaMalloc).
    ///
    /// Used for the nixl staging buffer, which must be allocated in the same
    /// CUDA runtime as RMM (DuckDB loads extensions with RTLD_LOCAL, creating
    /// separate CUDA runtimes for C++ and Rust).
    pub fn cuda_alloc(&self, size: usize) -> Result<usize, EngineError> {
        let sql = format!("SELECT * FROM sirius_cuda_alloc({})", size as i64);
        let mut stmt = self.conn
            .prepare(&sql)
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let mut rows = stmt.query([])
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        let row = rows.next()
            .map_err(|e| EngineError::ExecFailed(e.to_string()))?
            .ok_or_else(|| EngineError::ExecFailed("sirius_cuda_alloc: no result".into()))?;
        let addr: i64 = row.get(0).map_err(|e| EngineError::ExecFailed(e.to_string()))?;
        Ok(addr as usize)
    }
}

/// Owned exchange result artifact returned by the modern GPU exchange path.
///
/// Keeps the underlying C++ exchange session alive until the artifact is
/// dropped, while exposing stable Rust-owned descriptors for packed partitions
/// and broadcast entries.
pub struct ExchangeArtifact {
    handle: *mut c_void,
    destroy: DestroyExchangeArtifactFn,
    staging_base: usize,
    packed_partitions: Vec<PackedPartition>,
    packed_broadcast: Vec<PackedBroadcastEntry>,
}

impl ExchangeArtifact {
    pub fn staging_base(&self) -> usize {
        self.staging_base
    }

    pub fn packed_partitions(&self) -> &[PackedPartition] {
        &self.packed_partitions
    }

    pub fn packed_broadcast(&self) -> &[PackedBroadcastEntry] {
        &self.packed_broadcast
    }

    pub fn has_exchange_data(&self) -> bool {
        !self.packed_partitions.is_empty() || !self.packed_broadcast.is_empty()
    }
}

impl std::fmt::Debug for ExchangeArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExchangeArtifact")
            .field("handle", &self.handle)
            .field("staging_base", &format_args!("0x{:x}", self.staging_base))
            .field("packed_partitions", &self.packed_partitions.len())
            .field("packed_broadcast", &self.packed_broadcast.len())
            .finish()
    }
}

unsafe impl Send for ExchangeArtifact {}
unsafe impl Sync for ExchangeArtifact {}

impl Drop for ExchangeArtifact {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { (self.destroy)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

/// A per-partition packed GPU buffer from GPU hash_partition + chunked_pack.
#[derive(Debug, Clone)]
pub struct PackedPartition {
    pub partition_id: usize,
    pub staging_offset: usize,
    pub packed_size: usize,
    pub metadata: Vec<u8>,
    pub num_rows: u32,
    /// When non-zero, this partition overflowed the staging buffer and lives at
    /// this GPU address in a separate rmm::device_buffer.
    pub overflow_gpu_addr: usize,
}

/// A per-batch packed GPU buffer for the broadcast exchange path.
#[derive(Debug, Clone)]
pub struct PackedBroadcastEntry {
    pub entry_id: usize,
    pub staging_offset: usize,
    pub packed_size: usize,
    pub metadata: Vec<u8>,
    pub num_rows: u32,
    /// When non-zero, this entry overflowed the staging buffer.
    pub overflow_gpu_addr: usize,
}

/// Per-column extended GPU buffer info (null mask, string offsets).
#[derive(Debug, Clone, Default)]
pub struct GpuColumnBuffers {
    /// Null bitmap GPU address (0 if all-valid).
    pub null_mask_addr: usize,
    /// Null bitmap size in bytes.
    pub null_mask_len: usize,
    /// String offsets GPU address (0 if not a string column).
    pub offsets_addr: usize,
    /// String offsets buffer size in bytes.
    pub offsets_len: usize,
    /// Number of null values in this column.
    pub null_count: i32,
    /// Decimal scale (from cudf data_type::scale()), 0 for non-decimal types.
    pub scale: i32,
}

/// GPU result buffer information for nixl transfers.
#[derive(Debug, Clone)]
pub struct GpuResultInfo {
    /// GPU buffer addresses: (addr, len, device_id).
    pub buffer_addrs: Vec<(usize, usize, u64)>,
    /// Column metadata: (name, type_id).
    pub column_info: Vec<(String, i32)>,
    /// Extended per-column buffer info (null masks, string offsets).
    pub column_buffers: Vec<GpuColumnBuffers>,
    /// Number of rows.
    pub num_rows: u32,
    /// Arrow IPC schema bytes.
    pub schema_ipc: Vec<u8>,
}

/// Map Arrow DataType to DuckDB SQL type string.
fn arrow_type_to_duckdb_sql(dt: &arrow::datatypes::DataType) -> String {
    use arrow::datatypes::DataType;
    match dt {
        DataType::Boolean => "BOOLEAN".to_string(),
        DataType::Int8 => "TINYINT".to_string(),
        DataType::Int16 => "SMALLINT".to_string(),
        DataType::Int32 => "INTEGER".to_string(),
        DataType::Int64 => "BIGINT".to_string(),
        DataType::UInt8 => "UTINYINT".to_string(),
        DataType::UInt16 => "USMALLINT".to_string(),
        DataType::UInt32 => "UINTEGER".to_string(),
        DataType::UInt64 => "UBIGINT".to_string(),
        DataType::Float16 | DataType::Float32 => "FLOAT".to_string(),
        DataType::Float64 => "DOUBLE".to_string(),
        DataType::Utf8 | DataType::LargeUtf8 => "VARCHAR".to_string(),
        DataType::Binary | DataType::LargeBinary => "BLOB".to_string(),
        DataType::Decimal128(p, s) => format!("DECIMAL({}, {})", p, s),
        DataType::Date32 => "DATE".to_string(),
        DataType::Date64 => "DATE".to_string(),
        DataType::Timestamp(_, _) => "TIMESTAMP".to_string(),
        DataType::Time32(_) | DataType::Time64(_) => "TIME".to_string(),
        DataType::Duration(_) => "INTERVAL".to_string(),
        DataType::Interval(_) => "INTERVAL".to_string(),
        DataType::List(f) | DataType::LargeList(f) => {
            format!("{}[]", arrow_type_to_duckdb_sql(f.data_type()))
        }
        DataType::Struct(fields) => {
            let cols: Vec<String> = fields
                .iter()
                .map(|f| format!("\"{}\" {}", f.name(), arrow_type_to_duckdb_sql(f.data_type())))
                .collect();
            format!("STRUCT({})", cols.join(", "))
        }
        other => {
            tracing::warn!(arrow_type = ?other, "arrow_type_to_duckdb_sql: unmapped type, falling back to VARCHAR");
            "VARCHAR".to_string()
        }
    }
}

/// Errors from the Sirius engine.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("engine initialization failed: {0}")]
    InitFailed(String),

    #[error("query execution failed: {0}")]
    ExecFailed(String),
}

#[cfg(test)]
mod tests {
    use super::SiriusEngine;

    #[test]
    fn exchange_artifact_empty_roundtrip() {
        let Ok(engine) = SiriusEngine::new() else {
            return;
        };

        engine.finalize_exchange_tables_direct().unwrap();
        let session_id = engine.begin_exchange_capture(None).unwrap();
        assert!(session_id > 0);

        let artifact = engine
            .take_exchange_artifact()
            .unwrap()
            .expect("capture should yield an artifact");
        assert!(!artifact.has_exchange_data());
        assert!(artifact.packed_partitions().is_empty());
        assert!(artifact.packed_broadcast().is_empty());
    }

    #[test]
    #[ignore]
    fn exchange_artifact_broadcast_gpu_roundtrip() {
        let engine = SiriusEngine::new().expect("engine");
        engine
            .conn
            .execute_batch("LOAD parquet")
            .expect("load parquet");
        engine
            .init_gpu_buffers("1GB", "1GB")
            .expect("init_gpu_buffers");
        engine
            .conn
            .execute_batch("SET enable_fallback_check = true")
            .expect("enable_fallback_check");

        let staging_size = 16 * 1024 * 1024;
        let staging_addr = engine.cuda_alloc(staging_size).expect("cuda_alloc staging");
        engine
            .set_staging_buffer(staging_addr, staging_size)
            .expect("set_staging_buffer");

        engine
            .finalize_exchange_tables_direct()
            .expect("finalize_exchange_tables_direct");
        let session_id = engine.begin_exchange_capture(None).expect("begin_exchange_capture");
        assert!(session_id > 0);

        let ipc = engine
            .execute_gpu(
                "SELECT n_nationkey, n_name \
                 FROM read_parquet('/data/tpch/sf1/p16/snappy/nation/*.parquet')",
            )
            .expect("execute_gpu nation");
        assert!(!ipc.is_empty());

        let artifact = engine
            .take_exchange_artifact()
            .expect("take_exchange_artifact")
            .expect("exchange artifact");
        assert!(artifact.has_exchange_data());
        assert_eq!(artifact.staging_base(), staging_addr);
        assert!(artifact.packed_partitions().is_empty());
        assert!(!artifact.packed_broadcast().is_empty());
        assert!(
            artifact
                .packed_broadcast()
                .iter()
                .any(|entry| entry.packed_size > 0 && !entry.metadata.is_empty())
        );
    }
}
