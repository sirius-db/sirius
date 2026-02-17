//! Concurrent result storage for query fragment results.
//!
//! Stores Arrow record batches keyed by fragment instance ID (hi/lo pair).
//! Shared between the gRPC service (stores results, serves schemas) and
//! the Arrow Flight service (streams record batches).

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::Array;
use arrow::datatypes::SchemaRef;
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::{IpcWriteOptions, StreamWriter};
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use tokio::sync::Notify;
use tracing::{debug, warn};

/// Unique identifier for a fragment instance result, matching Doris PUniqueId.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct FinstId {
    pub hi: i64,
    pub lo: i64,
}

impl std::fmt::Display for FinstId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}:{:016x}", self.hi as u64, self.lo as u64)
    }
}

impl FinstId {
    /// Encode as 16 bytes (hi LE + lo LE) for use as an Arrow Flight ticket.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&self.hi.to_le_bytes());
        buf.extend_from_slice(&self.lo.to_le_bytes());
        buf
    }

    /// Decode from 16 bytes (hi LE + lo LE).
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 16 {
            return None;
        }
        let hi = i64::from_le_bytes(bytes[..8].try_into().ok()?);
        let lo = i64::from_le_bytes(bytes[8..16].try_into().ok()?);
        Some(Self { hi, lo })
    }
}

/// A stored query result: schema + record batches.
pub struct ResultEntry {
    pub schema: SchemaRef,
    pub batches: Vec<RecordBatch>,
}

impl ResultEntry {
    /// Serialize just the Arrow schema as IPC bytes (for `fetch_arrow_flight_schema`).
    pub fn schema_ipc_bytes(&self) -> Result<Vec<u8>, arrow::error::ArrowError> {
        let mut buf = Vec::new();
        {
            let mut writer =
                StreamWriter::try_new_with_options(&mut buf, &self.schema, IpcWriteOptions::default())?;
            writer.finish()?;
        }
        Ok(buf)
    }

    /// Total row count across all batches.
    pub fn num_rows(&self) -> usize {
        self.batches.iter().map(|b| b.num_rows()).sum()
    }

    /// Convert all batches to MySQL text protocol rows.
    ///
    /// Each returned `Vec<u8>` is one row: a concatenation of length-encoded
    /// column value strings (or 0xFB for NULL).
    pub fn to_mysql_rows(&self) -> Vec<Vec<u8>> {
        let mut rows = Vec::with_capacity(self.num_rows());
        for batch in &self.batches {
            for row_idx in 0..batch.num_rows() {
                let mut row_buf = Vec::new();
                for col_idx in 0..batch.num_columns() {
                    let col = batch.column(col_idx);
                    if col.is_null(row_idx) {
                        row_buf.push(0xFB); // MySQL NULL marker
                    } else {
                        let s = arrow::util::display::array_value_to_string(col, row_idx)
                            .unwrap_or_default();
                        mysql_encode_string(&s, &mut row_buf);
                    }
                }
                rows.push(row_buf);
            }
        }
        rows
    }
}

/// Encode a string as a MySQL length-encoded string, appending to `buf`.
fn mysql_encode_string(s: &str, buf: &mut Vec<u8>) {
    let len = s.len();
    if len < 251 {
        buf.push(len as u8);
    } else if len < 65536 {
        buf.push(0xFC);
        buf.extend_from_slice(&(len as u16).to_le_bytes());
    } else if len < 16_777_216 {
        buf.push(0xFD);
        buf.push(len as u8);
        buf.push((len >> 8) as u8);
        buf.push((len >> 16) as u8);
    } else {
        buf.push(0xFE);
        buf.extend_from_slice(&(len as u64).to_le_bytes());
    }
    buf.extend_from_slice(s.as_bytes());
}

/// Concurrent storage for fragment results.
///
/// Thread-safe via DashMap. Shared between the gRPC handler
/// (writes results, serves schemas) and the Arrow Flight service
/// (reads and streams record batches).
#[derive(Clone)]
pub struct ResultStore {
    results: Arc<DashMap<FinstId, Arc<ResultEntry>>>,
    /// Notifies waiters when any result is stored.
    notify: Arc<Notify>,
}

impl ResultStore {
    pub fn new() -> Self {
        Self {
            results: Arc::new(DashMap::new()),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Store Arrow IPC stream bytes as a result for the given fragment instance.
    ///
    /// Parses the IPC stream to extract schema + record batches.
    pub fn store_ipc_result(&self, id: FinstId, ipc_bytes: &[u8]) -> Result<(), String> {
        let cursor = Cursor::new(ipc_bytes);
        let reader = StreamReader::try_new(cursor, None)
            .map_err(|e| format!("failed to parse Arrow IPC stream: {e}"))?;

        let schema = reader.schema();
        let mut batches = Vec::new();
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| format!("failed to read Arrow batch: {e}"))?;
            batches.push(batch);
        }

        debug!(%id, batches = batches.len(), "stored result");
        self.results
            .insert(id, Arc::new(ResultEntry { schema, batches }));
        self.notify.notify_waiters();
        Ok(())
    }

    /// Store an already-parsed result directly.
    pub fn store_result(&self, id: FinstId, schema: SchemaRef, batches: Vec<RecordBatch>) {
        debug!(%id, batch_count = batches.len(), "stored result");
        self.results
            .insert(id, Arc::new(ResultEntry { schema, batches }));
        self.notify.notify_waiters();
    }

    /// Get a result entry by fragment instance ID.
    pub fn get(&self, id: &FinstId) -> Option<Arc<ResultEntry>> {
        self.results.get(id).map(|entry| entry.value().clone())
    }

    /// Wait for a result to become available, with timeout.
    pub async fn wait_for(&self, id: &FinstId, timeout: std::time::Duration) -> Option<Arc<ResultEntry>> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if let Some(entry) = self.get(id) {
                return Some(entry);
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            tokio::select! {
                _ = self.notify.notified() => {}
                _ = tokio::time::sleep(remaining) => { return self.get(id); }
            }
        }
    }

    /// Remove a result (after it has been consumed or cancelled).
    pub fn remove(&self, id: &FinstId) -> bool {
        let removed = self.results.remove(id).is_some();
        if removed {
            debug!(%id, "removed result");
        } else {
            warn!(%id, "remove: result not found");
        }
        removed
    }

    /// Remove all results for a given query (matching hi part).
    pub fn remove_query(&self, query_hi: i64, query_lo: i64) {
        self.results
            .retain(|id, _| id.hi != query_hi || id.lo != query_lo);
    }

    /// Number of stored results.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}
