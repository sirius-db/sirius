//! Safe, idiomatic Rust bindings for [Sirius](https://github.com/sirius-db/sirius),
//! the GPU-native SQL engine.
//!
//! This crate wraps the low-level [`sirius-sys`] cxx bindings in safe Rust types
//! — the entry point for driving Sirius from Rust.
//!
//! Today it binds just enough to prove the toolchain links against the real
//! Sirius library: constructing a [`SiriusContext`] from defaults or a YAML
//! config file. More of the API surface is added in later PRs.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::path::Path;

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use cxx::{Exception, UniquePtr, let_cxx_string};

/// An initialized Sirius engine context.
///
/// Constructing one brings up the engine (GPU resources included); dropping it
/// tears the engine down. The `cxx::UniquePtr` owns the C++ object, so lifetime
/// is pure RAII — there is no uninitialized or manually-freed state.
///
/// Bring-up is fallible: GPU initialization or parsing a config file can fail,
/// surfaced here as a [`cxx::Exception`].
///
/// The engine keeps process-global GPU state, so it currently supports a single
/// live context per process; constructing or holding more than one concurrently
/// is not yet supported (enforcement is a follow-up).
pub struct SiriusContext {
    // RAII handle owning the C++ engine context for its lifetime.
    //
    // Behind a `RefCell` because every call into C++ needs `Pin<&mut Context>` while a
    // [`Fragment`] only borrows the context immutably: several fragments of one query are alive
    // at once (senders parked, waiting for their receiver), and a `&mut self` factory would allow
    // exactly one. The borrow is taken and released within each call, and the context is neither
    // `Send` nor `Sync`, so there is no cross-thread aliasing to worry about.
    inner: RefCell<UniquePtr<sirius_sys::Context>>,
}

/// Fully materialized output of one Substrait execution.
pub struct SubstraitResult {
    /// Arrow schema reported by the result stream, also available for empty results.
    pub schema: SchemaRef,
    /// Eagerly collected output batches.
    pub batches: Vec<RecordBatch>,
}

impl SiriusContext {
    /// Bring up a new, initialized Sirius engine context configured from
    /// built-in defaults.
    pub fn new() -> Result<Self, Exception> {
        Ok(Self {
            inner: RefCell::new(sirius_sys::make_context()?),
        })
    }

    /// Bring up a new, initialized Sirius engine context configured from the
    /// YAML config file at `path`.
    pub fn from_config_file(path: &Path) -> Result<Self, Exception> {
        // cxx passes the path to the C++ `const std::string&` parameter; build
        // one from the (lossy) UTF-8 form of the platform path.
        let_cxx_string!(config_path = path.to_string_lossy().as_ref());
        Ok(Self {
            inner: RefCell::new(sirius_sys::make_context_from_config(&config_path)?),
        })
    }

    /// Start a new [`Fragment`] on this context.
    ///
    /// The returned fragment borrows the context, so the compiler enforces the one lifetime rule
    /// the C++ side cannot: a fragment must not outlive the engine it runs on.
    pub fn fragment(&self) -> Result<Fragment<'_>, Exception> {
        let mut context = self.inner.borrow_mut();
        Ok(Fragment {
            inner: sirius_sys::make_fragment(context.pin_mut())?,
            _context: PhantomData,
        })
    }

    /// Execute a serialized Substrait plan on the GPU and return the result rows.
    ///
    /// `plan` is the protobuf-encoded `substrait::Plan`; its reads must be
    /// resolvable by the embedded DuckDB used for lowering (e.g. `local_files`
    /// parquet reads), and every operator must be one the GPU engine supports —
    /// there is no CPU fallback here, so an unsupported plan returns an error.
    ///
    /// The result is collected eagerly into owned [`RecordBatch`]es. DuckDB's
    /// Arrow stream converts each batch using this context's client state, so the
    /// stream is drained while the context is alive; the returned batches own
    /// their buffers and are independent of the context's lifetime.
    pub fn execute_substrait(&self, plan: &[u8]) -> Result<Vec<RecordBatch>, SiriusError> {
        Ok(self.execute_substrait_result(plan)?.batches)
    }

    /// Execute a serialized Substrait plan and retain its Arrow schema for empty results.
    pub fn execute_substrait_result(&self, plan: &[u8]) -> Result<SubstraitResult, SiriusError> {
        // The engine writes a self-owning Arrow C Data Interface stream into
        // `stream`; the FFI takes its address as an integer (a `uintptr_t`).
        let mut stream = FFI_ArrowArrayStream::empty();
        let out_stream_addr = std::ptr::addr_of_mut!(stream) as usize;
        let_cxx_string!(plan = plan);
        // SAFETY: `out_stream_addr` is the address of `stream`, a live, writable
        // `FFI_ArrowArrayStream` owned by this stack frame for the call's duration.
        unsafe {
            self.inner
                .borrow_mut()
                .pin_mut()
                .execute_substrait(&plan, out_stream_addr)
                .map_err(SiriusError::Engine)?;
        }
        // Drain fully while `self` is alive (conversion dereferences the context).
        collect_arrow_stream(stream)
    }

    /// Lease `len` bytes of the exchange staging arena, returning the lease's byte offset from
    /// [`staging_base`](Self::staging_base).
    ///
    /// The arena exists only when `SIRIUS_EXCHANGE_STAGING_BYTES` was set at context bring-up;
    /// without one, every staging call is an error rather than a silent slow path. Exhaustion is
    /// an error naming the requested/free/capacity byte counts.
    pub fn staging_lease(&self, len: u64) -> Result<u64, Exception> {
        self.inner.borrow_mut().pin_mut().staging_lease(len)
    }

    /// Return the staging lease at `offset`. Leases are short-lived by design
    /// (copy-out-on-arrival); when the last outstanding lease is released the arena's bump head
    /// resets.
    pub fn staging_release(&self, offset: u64) -> Result<(), Exception> {
        self.inner.borrow_mut().pin_mut().staging_release(offset)
    }

    /// Device base address of the staging arena, for transport memory registration.
    pub fn staging_base(&self) -> Result<usize, Exception> {
        self.inner.borrow().staging_base()
    }

    /// Capacity of the staging arena in bytes.
    pub fn staging_capacity(&self) -> Result<u64, Exception> {
        self.inner.borrow().staging_capacity()
    }

    /// Thread-safe handle to the exchange staging arena, or `None` when no arena is configured
    /// (`SIRIUS_EXCHANGE_STAGING_BYTES` unset at bring-up).
    ///
    /// Unlike the `staging_*` methods above — which go through this `!Sync` context and
    /// therefore its owning thread — the handle is `Send + Sync` and serves leases from any
    /// thread, concurrently with the context thread's own staging traffic. It shares ownership
    /// of the ONE C++ allocator ([`Fragment::export_packed`] leases from the same arena), so
    /// the two sides can never double-book a region, and the handle stays valid even if this
    /// context is dropped first.
    pub fn staging_arena(&self) -> Option<StagingArena> {
        let handle = self.inner.borrow().staging_arena_handle();
        (!handle.is_null()).then(|| StagingArena { inner: handle })
    }
}

/// Drains a filled Arrow C Data Interface stream into owned batches, retaining the schema.
fn collect_arrow_stream(stream: FFI_ArrowArrayStream) -> Result<SubstraitResult, SiriusError> {
    let reader = ArrowArrayStreamReader::try_new(stream).map_err(SiriusError::Arrow)?;
    let schema = reader.schema();
    let batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(SiriusError::Arrow)?;
    Ok(SubstraitResult { schema, batches })
}

/// The name of the view a plan must read to consume input stream `stream_id`.
///
/// A front end emits a read of this name where it would otherwise emit a file scan; the engine
/// creates the view when the fragment is built. Both sides call this, so the convention has one
/// definition.
pub fn stream_view_name(stream_id: u64) -> String {
    sirius_sys::stream_view_name(stream_id)
        .to_string_lossy()
        .into_owned()
}

/// One plan fragment of a multi-fragment query.
///
/// A fragment declaring one or more **output streams** is rooted in a streaming sink: its results
/// stay on the GPU as native batches that outlive its own query, ready for a downstream fragment
/// to take with [`Fragment::relay_from`]. A fragment declaring **none** is a result fragment and
/// produces Arrow via [`Fragment::into_arrow`].
///
/// Calls are ordered: declare, [`build`](Fragment::build), relay from every sender,
/// [`run`](Fragment::run), then drain. `build` opens a query lifecycle on the shared engine that
/// `run` closes, so one fragment at a time may sit between the two — dropping a built-but-unrun
/// fragment closes the lifecycle for you.
pub struct Fragment<'ctx> {
    inner: UniquePtr<sirius_sys::Fragment>,
    /// Ties the fragment's lifetime to the context that made it.
    _context: PhantomData<&'ctx SiriusContext>,
}

impl Fragment<'_> {
    /// Declare one column of input stream `stream_id`, in plan order. `ty` is a DuckDB type name
    /// (`BIGINT`, `DECIMAL(15,2)`, `DATE`, …).
    pub fn declare_input_column(
        &mut self,
        stream_id: u64,
        name: &str,
        ty: &str,
    ) -> Result<(), Exception> {
        let_cxx_string!(name = name);
        let_cxx_string!(ty = ty);
        self.inner
            .pin_mut()
            .declare_input_column(stream_id, &name, &ty)
    }

    /// Declare a sender that must close input stream `stream_id` before it ends. With none
    /// declared the stream expects the single sender `0`.
    pub fn declare_input_sender(
        &mut self,
        stream_id: u64,
        sender_id: u32,
    ) -> Result<(), Exception> {
        self.inner
            .pin_mut()
            .declare_input_sender(stream_id, sender_id)
    }

    /// Declare the row count of input stream `stream_id`, summed over all its senders — exact
    /// when the caller already holds the stream's batches (parked locally or staged remotely),
    /// an estimate otherwise.
    ///
    /// Optional but load-bearing for plan quality: a stream source binds with no rows behind
    /// it, so without this DuckDB's optimizer sees cardinality 1 on every stream and picks hash
    /// join build sides blind. Undeclared streams keep that legacy behavior. Last call wins.
    pub fn declare_input_cardinality(
        &mut self,
        stream_id: u64,
        rows: u64,
    ) -> Result<(), Exception> {
        self.inner
            .pin_mut()
            .declare_input_cardinality(stream_id, rows)
    }

    /// Declare an output stream. A fragment with no output stream is a result fragment.
    pub fn declare_output(&mut self, stream_id: u64) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output(stream_id)
    }

    /// Every declared output stream receives the full fragment output (a broadcast sink);
    /// output 0 keeps the original batches, the rest carry independent deep copies.
    pub fn declare_output_broadcast(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output_broadcast()
    }

    /// Declares one hash-partition key (an output column index): rows hash-route by the
    /// declared keys, output stream i taking partition i. Call once per key, in the exchange's
    /// shared partition-expression order.
    pub fn declare_output_hash_key(&mut self, column_index: u32) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output_hash_key(column_index)
    }

    /// Plan `substrait_plan` against the declared streams and open the fragment's query lifecycle.
    pub fn build(&mut self, substrait_plan: &[u8]) -> Result<(), Exception> {
        let_cxx_string!(plan = substrait_plan);
        self.inner.pin_mut().build(&plan)
    }

    /// Move every batch parked on `source`'s output stream into this fragment's input stream —
    /// as native GPU batch handles, with no Arrow and no file in between — then close `sender_id`.
    ///
    /// Returns the number of batches moved. `source` must have finished [`run`](Fragment::run).
    pub fn relay_from(
        &mut self,
        source: &mut Fragment<'_>,
        source_stream_id: u64,
        input_stream_id: u64,
        sender_id: u32,
    ) -> Result<usize, Exception> {
        self.inner.pin_mut().relay_from(
            source.inner.pin_mut(),
            source_stream_id,
            input_stream_id,
            sender_id,
        )
    }

    /// Execute the fragment and close its query lifecycle. Blocks until its pipelines finish.
    pub fn run(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().run()
    }

    /// Collect a result fragment's rows over the Arrow C Data Interface.
    pub fn into_arrow(&mut self) -> Result<SubstraitResult, SiriusError> {
        let mut stream = FFI_ArrowArrayStream::empty();
        let out_stream_addr = std::ptr::addr_of_mut!(stream) as usize;
        // SAFETY: `out_stream_addr` is the address of `stream`, a live, writable
        // `FFI_ArrowArrayStream` owned by this stack frame for the call's duration.
        unsafe {
            self.inner
                .pin_mut()
                .result_to_arrow(out_stream_addr)
                .map_err(SiriusError::Engine)?;
        }
        collect_arrow_stream(stream)
    }

    /// Batches currently parked on output stream `stream_id`.
    pub fn output_batch_count(&self, stream_id: u64) -> Result<usize, Exception> {
        self.inner.output_batch_count(stream_id)
    }

    /// Total rows parked on output stream `stream_id`, without draining it — what a local
    /// relay's receiver feeds [`declare_input_cardinality`](Fragment::declare_input_cardinality)
    /// before its own [`build`](Fragment::build). Errs on an unknown stream or a spilled
    /// (non-GPU) parked batch; skip the declaration then rather than failing the fragment.
    pub fn output_row_count(&self, stream_id: u64) -> Result<u64, Exception> {
        self.inner.output_row_count(stream_id)
    }

    /// DuckDB type names of this built fragment's output (sink) columns — the types every batch
    /// leaving the fragment actually carries, exactly what the receiving hop's schema guard
    /// compares against the receiver's declared input columns. Errs before
    /// [`build`](Fragment::build) and on a result fragment (which has no streaming sink).
    pub fn output_types(&self) -> Result<Vec<String>, Exception> {
        Ok(self
            .inner
            .output_types()?
            .iter()
            .map(|ty| ty.to_string_lossy().into_owned())
            .collect())
    }

    /// Pack the next batch parked on output stream `stream_id` into a fresh staging-arena lease,
    /// as cudf packed bytes.
    ///
    /// `Ok(None)` means nothing is parked right now — for a fragment that finished
    /// [`run`](Fragment::run), the stream is drained. The packed device bytes are complete when
    /// this returns, so a transport may transmit from
    /// `[staging_base() + offset, + len)` immediately; the lease stays live until the caller
    /// hands it back with [`SiriusContext::staging_release`] after the transmit completes.
    ///
    /// A zero-row batch comes back metadata-only: `offset == 0` with `len == 0` means NO lease
    /// exists for it, and the caller must not release anything.
    pub fn export_packed(&mut self, stream_id: u64) -> Result<Option<PackedBatch>, Exception> {
        let mut offset = 0u64;
        let mut len = 0u64;
        let mut rows = 0u64;
        let metadata = self
            .inner
            .pin_mut()
            .export_packed(stream_id, &mut offset, &mut len, &mut rows)?;
        if metadata.is_null() {
            return Ok(None);
        }
        Ok(Some(PackedBatch {
            metadata: metadata.as_slice().to_vec(),
            offset,
            len,
            rows,
        }))
    }

    /// Push a packed batch sitting in the staging arena into input stream `stream_id`: the
    /// receive-side mirror of [`export_packed`](Fragment::export_packed).
    ///
    /// The table is deep-copied out of the lease into ordinary pool memory before this returns
    /// (copy-out-on-arrival), so the lease is immediately reusable — and releasable. Legal
    /// between [`build`](Fragment::build) and [`run`](Fragment::run), like
    /// [`relay_from`](Fragment::relay_from); pushing after the stream ended is an error, never a
    /// silent drop.
    pub fn push_packed(&mut self, stream_id: u64, batch: &PackedBatch) -> Result<(), Exception> {
        // SAFETY: the metadata pointer/length name `batch.metadata`'s buffer, which this borrow
        // keeps alive and readable for the duration of the call.
        unsafe {
            self.inner.pin_mut().push_packed(
                stream_id,
                batch.metadata.as_ptr() as usize,
                batch.metadata.len(),
                batch.offset,
                batch.len,
            )
        }
    }

    /// Record that `sender_id` finished producing into input stream `stream_id` — the EOS mirror
    /// of [`push_packed`](Fragment::push_packed) for remote senders
    /// ([`relay_from`](Fragment::relay_from) closes its own sender). The stream ends once every
    /// expected sender has closed.
    pub fn close_input(&mut self, stream_id: u64, sender_id: u32) -> Result<(), Exception> {
        self.inner.pin_mut().close_input(stream_id, sender_id)
    }
}

/// One batch exported into the exchange staging arena as cudf packed bytes.
///
/// `metadata` is the host-side cudf pack metadata (it travels with the payload on the wire);
/// `offset`/`len` locate the packed device payload inside the staging arena of the context that
/// exported it. The exporter's lease at `offset` stays outstanding until
/// [`SiriusContext::staging_release`] is called with it — except for a metadata-only zero-row
/// batch (`offset == 0`, `len == 0`), which holds no lease and must not be released.
pub struct PackedBatch {
    /// cudf pack metadata bytes (host memory).
    pub metadata: Vec<u8>,
    /// Byte offset of the packed payload from the arena base.
    pub offset: u64,
    /// Length of the packed payload in bytes.
    pub len: u64,
    /// Exact row count of the packed table, filled by
    /// [`export_packed`](Fragment::export_packed) so a transport can carry it to the receiver's
    /// [`declare_input_cardinality`](Fragment::declare_input_cardinality). Ignored by
    /// [`push_packed`](Fragment::push_packed).
    pub rows: u64,
}

/// Thread-safe handle to a context's exchange staging arena, from
/// [`SiriusContext::staging_arena`].
///
/// This is what lets a transport/RPC thread serve `lease`/`release` while the context's owning
/// thread is busy running a fragment: the two contend on nothing but the arena's own mutex, so
/// an engine stall can never starve a peer's staging lease.
pub struct StagingArena {
    inner: UniquePtr<sirius_sys::StagingArena>,
}

// SAFETY: the C++ `StagingArena` is a `shared_ptr` to the one `exchange_staging_arena`. Every
// method (`lease`, `release`, and the immutable `base`/`capacity` reads) serializes on the
// arena's internal `std::mutex` and makes NO CUDA calls — the region is a single `cudaMalloc`
// made at arena construction — so there is no thread-affine state behind any operation. The
// `shared_ptr` keeps the device region alive independently of the `SiriusContext`, so the
// handle cannot dangle if the context is torn down first.
unsafe impl Send for StagingArena {}
unsafe impl Sync for StagingArena {}

impl StagingArena {
    /// Lease `len` bytes, returning the lease's byte offset from [`base`](Self::base). Errors
    /// on exhaustion or a zero-length request — the arena never blocks.
    pub fn lease(&self, len: u64) -> Result<u64, Exception> {
        self.inner.lease(len)
    }

    /// Return the lease at `offset`; the block goes back to the arena's free list and
    /// coalesces with its free neighbours, so the space is reusable regardless of release
    /// order.
    pub fn release(&self, offset: u64) -> Result<(), Exception> {
        self.inner.release(offset)
    }

    /// Device base address of the arena, for transport memory registration.
    pub fn base(&self) -> usize {
        self.inner.base()
    }

    /// Capacity of the arena in bytes.
    pub fn capacity(&self) -> u64 {
        self.inner.capacity()
    }

    /// Leases currently held. Nonzero once a query has quiesced means a leaked lease — this is
    /// the only way the CN can observe one.
    pub fn outstanding(&self) -> Result<usize, Exception> {
        self.inner.outstanding()
    }
}

impl std::fmt::Debug for StagingArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingArena")
            .field("base", &self.base())
            .field("capacity", &self.capacity())
            .finish()
    }
}

/// Error returned by [`SiriusContext::execute_substrait`].
#[derive(Debug)]
pub enum SiriusError {
    /// The engine failed to translate or execute the plan (a C++ exception).
    Engine(Exception),
    /// The Arrow result stream could not be consumed.
    Arrow(arrow_schema::ArrowError),
}

impl std::fmt::Display for SiriusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Engine(err) => write!(f, "sirius engine error: {err}"),
            Self::Arrow(err) => write!(f, "arrow result error: {err}"),
        }
    }
}

impl std::error::Error for SiriusError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Engine(err) => Some(err),
            Self::Arrow(err) => Some(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use arrow_array::{ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use prost::Message;

    use super::SiriusContext;

    /// The engine keeps process-global GPU state, so at most one context may be
    /// live at a time; context-constructing tests hold this for their duration.
    static GPU_CONTEXT_LOCK: Mutex<()> = Mutex::new(());

    /// Proof-of-life: bring up a real Sirius engine context and drop it. This
    /// links the real Sirius library and exercises the full cxx round-trip +
    /// `initialize()`/teardown. Requires a GPU (construction does GPU bring-up).
    #[test]
    fn constructs_and_drops() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let _ctx = SiriusContext::new().expect("bring up default Sirius context");
    }

    /// Encodes a `local_files` parquet read plan with `names` as the root output names and one
    /// item per `(path, start, length)` — `(0, 0)` meaning the whole file. The shape DuckDB's
    /// Substrait reader resolves to `parquet_scan(<path>)`; a non-zero range is what a compute
    /// node emits for one byte-range split of a distributed scan.
    fn local_files_plan_ranged(items: &[(&str, u64, u64)], names: Vec<String>) -> Vec<u8> {
        use substrait::proto::read_rel::local_files::FileOrFiles;
        use substrait::proto::read_rel::local_files::file_or_files::{
            FileFormat, ParquetReadOptions, PathType,
        };
        use substrait::proto::read_rel::{LocalFiles, ReadType};
        use substrait::proto::{Plan, PlanRel, ReadRel, Rel, RelRoot, plan_rel, rel};

        let read = Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                read_type: Some(ReadType::LocalFiles(LocalFiles {
                    items: items
                        .iter()
                        .map(|(path, start, length)| FileOrFiles {
                            path_type: Some(PathType::UriFile(path.to_string())),
                            file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                            start: *start,
                            length: *length,
                            ..Default::default()
                        })
                        .collect(),
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(read),
                    names,
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    /// Encodes a single-file whole-file `local_files` parquet read plan.
    fn local_files_plan(path: &str, names: Vec<String>) -> Vec<u8> {
        local_files_plan_ranged(&[(path, 0, 0)], names)
    }

    /// Writes the tiny `(id BIGINT, name VARCHAR)` parquet fixture at `path`:
    /// rows (1, "a"), (2, "b"), (3, "c").
    fn write_users_parquet(path: &Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let ids: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let names: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, names]).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Writes a `(id BIGINT, name VARCHAR)` parquet with `rows` rows split into row groups of
    /// `rows_per_group`, so a byte range of the file holds a real subset of its row groups.
    fn write_multi_row_group_parquet(path: &Path, rows: i64, rows_per_group: usize) {
        use parquet::file::properties::WriterProperties;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let ids: ArrayRef = Arc::new(Int64Array::from((0..rows).collect::<Vec<_>>()));
        let names: ArrayRef = Arc::new(StringArray::from(
            (0..rows).map(|i| format!("n{i}")).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, names]).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_size(rows_per_group)
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Writes a `(price DECIMAL(15,2), name VARCHAR)` parquet with `rows` distinct prices
    /// (row i carries the scaled value i, i.e. i/100) split into row groups of
    /// `rows_per_group`. Precision 15 keeps the parquet physical type INT64 — the TPC-H
    /// money shape, and exactly what q10's shuffle key looks like on the wire.
    fn write_multi_row_group_decimal_parquet(path: &Path, rows: i64, rows_per_group: usize) {
        use parquet::file::properties::WriterProperties;
        let schema = Arc::new(Schema::new(vec![
            Field::new("price", DataType::Decimal128(15, 2), false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let prices: ArrayRef = Arc::new(
            Decimal128Array::from_iter_values((0..rows).map(i128::from))
                .with_precision_and_scale(15, 2)
                .unwrap(),
        );
        let names: ArrayRef = Arc::new(StringArray::from(
            (0..rows).map(|i| format!("n{i}")).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(schema.clone(), vec![prices, names]).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_size(rows_per_group)
            .build();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// End-to-end: execute a `local_files` parquet plan on the GPU and read the
    /// result rows back over the Arrow C Data Interface. Requires a GPU.
    #[test]
    fn executes_local_files_plan_on_gpu() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);

        let plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );

        // Execute twice on one context to verify standalone query state is reset,
        // then drop it before inspecting the returned owned results.
        let results = {
            let ctx = SiriusContext::new().expect("bring up sirius context");
            vec![
                ctx.execute_substrait_result(&plan)
                    .expect("execute first substrait plan"),
                ctx.execute_substrait_result(&plan)
                    .expect("execute second substrait plan"),
            ]
        };

        for result in results {
            assert_eq!(result.schema.fields().len(), 2);
            assert_eq!(result.schema.field(0).name(), "id");
            assert_eq!(result.schema.field(1).name(), "name");
            let total_rows: usize = result.batches.iter().map(RecordBatch::num_rows).sum();
            assert_eq!(total_rows, 3, "expected 3 rows from the parquet fixture");
        }
    }

    /// The first multi-output fragment end to end: a broadcast sender's two output streams
    /// each deliver the FULL result to their own receiver — the build side of a broadcast
    /// join once the compute node fans destinations out. Requires a GPU.
    #[test]
    fn broadcast_fragment_feeds_every_destination() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let sender_plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );

        let ctx = SiriusContext::new().expect("bring up sirius context");
        let mut sender = ctx.fragment().unwrap();
        sender.declare_output(0).unwrap();
        sender.declare_output(1).unwrap();
        sender.declare_output_broadcast().unwrap();
        sender.build(&sender_plan).unwrap();
        sender.run().unwrap();

        let expected = vec![
            (1, "a".to_string()),
            (2, "b".to_string()),
            (3, "c".to_string()),
        ];
        for output_stream in [0u64, 1u64] {
            let mut receiver = ctx.fragment().unwrap();
            receiver.declare_input_column(0, "id", "BIGINT").unwrap();
            receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
            receiver.build(&stream_read_plan(0)).unwrap();
            let moved = receiver
                .relay_from(&mut sender, output_stream, 0, 0)
                .unwrap();
            assert!(moved > 0, "stream {output_stream} must carry the broadcast");
            receiver.run().unwrap();
            let result = receiver.into_arrow().unwrap();
            assert_eq!(rows(&result), expected, "stream {output_stream}");
        }
    }

    /// Hash-partitioned fan-out: two output streams partition the rows by key -- disjoint,
    /// union == whole -- and two INDEPENDENTLY built senders over the same keys (different
    /// row-group boundaries) assign every key to the SAME stream. That determinism is the
    /// cross-sender hash-parity contract a distributed shuffle join rests on. Requires a GPU.
    #[test]
    fn hash_partitioned_fragment_routes_keys_deterministically() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("keys_a.parquet");
        let path_b = dir.path().join("keys_b.parquet");
        // Same 20k rows, different row-group boundaries: the assignment must not depend on
        // how batches slice the data.
        write_multi_row_group_parquet(&path_a, 20000, 5000);
        write_multi_row_group_parquet(&path_b, 20000, 7000);

        let ctx = SiriusContext::new().expect("bring up sirius context");
        let mut assignments: Vec<std::collections::HashMap<i64, u64>> = Vec::new();
        for path in [&path_a, &path_b] {
            let mut sender = ctx.fragment().unwrap();
            sender.declare_output(0).unwrap();
            sender.declare_output(1).unwrap();
            sender.declare_output_hash_key(0).unwrap();
            sender
                .build(&local_files_plan(
                    path.to_str().unwrap(),
                    vec!["id".to_string(), "name".to_string()],
                ))
                .unwrap();
            sender.run().unwrap();

            let mut map = std::collections::HashMap::new();
            let mut total = 0usize;
            for stream in [0u64, 1u64] {
                let mut receiver = ctx.fragment().unwrap();
                receiver.declare_input_column(0, "id", "BIGINT").unwrap();
                receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
                receiver.build(&stream_read_plan(0)).unwrap();
                receiver.relay_from(&mut sender, stream, 0, 0).unwrap();
                receiver.run().unwrap();
                let result = receiver.into_arrow().unwrap();
                let partition = rows(&result);
                assert!(!partition.is_empty(), "stream {stream} owns no keys");
                total += partition.len();
                for (id, _) in partition {
                    assert!(map.insert(id, stream).is_none(), "key {id} in two streams");
                }
            }
            assert_eq!(total, 20000, "partitions must union to the whole input");
            assignments.push(map);
        }
        assert_eq!(
            assignments[0], assignments[1],
            "independently built senders must route every key identically"
        );
    }

    /// The same determinism contract for a DECIMAL(15,2) key — TPC-H q10's shuffle-key shape.
    /// The engine hashes decimal keys through a FLOAT64 cast; what a shuffle needs from that
    /// cast is determinism (equal values -> equal buckets on every sender), not injectivity,
    /// so the assertions are the same as for INT64 keys: two independently built senders over
    /// the same keys (different row-group boundaries) agree on every key's stream, and the
    /// partitions union to the whole input. Requires a GPU.
    #[test]
    fn hash_partitioned_fragment_routes_decimal_keys_deterministically() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("decimal_keys_a.parquet");
        let path_b = dir.path().join("decimal_keys_b.parquet");
        write_multi_row_group_decimal_parquet(&path_a, 20000, 5000);
        write_multi_row_group_decimal_parquet(&path_b, 20000, 7000);

        let ctx = SiriusContext::new().expect("bring up sirius context");
        let mut assignments: Vec<std::collections::HashMap<i128, u64>> = Vec::new();
        for path in [&path_a, &path_b] {
            let mut sender = ctx.fragment().unwrap();
            sender.declare_output(0).unwrap();
            sender.declare_output(1).unwrap();
            sender.declare_output_hash_key(0).unwrap();
            sender
                .build(&local_files_plan(
                    path.to_str().unwrap(),
                    vec!["price".to_string(), "name".to_string()],
                ))
                .unwrap();
            sender.run().unwrap();

            let mut map = std::collections::HashMap::new();
            let mut total = 0usize;
            for stream in [0u64, 1u64] {
                let mut receiver = ctx.fragment().unwrap();
                receiver
                    .declare_input_column(0, "price", "DECIMAL(15,2)")
                    .unwrap();
                receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
                receiver.build(&decimal_stream_read_plan(0)).unwrap();
                receiver.relay_from(&mut sender, stream, 0, 0).unwrap();
                receiver.run().unwrap();
                let result = receiver.into_arrow().unwrap();
                let partition = decimal_rows(&result);
                assert!(!partition.is_empty(), "stream {stream} owns no keys");
                total += partition.len();
                for (price, _) in partition {
                    assert!(
                        map.insert(price, stream).is_none(),
                        "key {price} in two streams"
                    );
                }
            }
            assert_eq!(total, 20000, "partitions must union to the whole input");
            assignments.push(map);
        }
        assert_eq!(
            assignments[0], assignments[1],
            "independently built senders must route every decimal key identically"
        );
    }

    /// Byte-range splits of one parquet file must read every row exactly once: each split
    /// yields a disjoint subset, their union is the whole file, and one plan carrying both
    /// splits as separate LocalFiles items equals the whole-file plan. Requires a GPU.
    #[test]
    fn byte_range_splits_read_every_row_exactly_once() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("many.parquet");
        // 30k rows in ~6 row groups: big enough that both halves hold data pages.
        write_multi_row_group_parquet(&path, 30000, 5000);
        let file_size = std::fs::metadata(&path).unwrap().len();
        let half = file_size / 2;
        let p = path.to_str().unwrap();
        let names = || vec!["id".to_string(), "name".to_string()];

        let ctx = SiriusContext::new().expect("bring up sirius context");
        let whole = ctx
            .execute_substrait_result(&local_files_plan(p, names()))
            .expect("whole-file plan");
        let left = ctx
            .execute_substrait_result(&local_files_plan_ranged(&[(p, 0, half)], names()))
            .expect("left split plan");
        let right = ctx
            .execute_substrait_result(&local_files_plan_ranged(
                &[(p, half, file_size - half)],
                names(),
            ))
            .expect("right split plan");

        let whole_rows = rows(&whole);
        let left_rows = rows(&left);
        let right_rows = rows(&right);
        assert_eq!(whole_rows.len(), 30000);
        assert!(!left_rows.is_empty(), "left split must own row groups");
        assert!(!right_rows.is_empty(), "right split must own row groups");
        let mut union = left_rows.clone();
        union.extend(right_rows.iter().cloned());
        union.sort();
        assert_eq!(
            union, whole_rows,
            "splits must partition the file: no duplication, no loss"
        );

        // Both splits in ONE plan (two LocalFiles items for the same path) also equal the
        // whole file — the multi-range-per-instance shape a CN emits when the FE co-locates
        // two splits of one file.
        let both = ctx
            .execute_substrait_result(&local_files_plan_ranged(
                &[(p, 0, half), (p, half, file_size - half)],
                names(),
            ))
            .expect("two-splits-one-plan");
        assert_eq!(rows(&both), whole_rows);

        // An empty split (range inside one row group) is a valid empty result, not an error.
        let empty = ctx
            .execute_substrait_result(&local_files_plan_ranged(&[(p, 10, 5)], names()))
            .expect("empty split plan");
        assert_eq!(rows(&empty).len(), 0);
    }

    /// A missing config file is rejected before any GPU work (`load_from_file`
    /// throws first), so this exercises the fallible config path and the cxx
    /// exception round-trip without needing a GPU.
    #[test]
    fn missing_config_file_is_an_error() {
        let result = SiriusContext::from_config_file(Path::new(
            "/nonexistent/sirius-config-does-not-exist.yaml",
        ));
        assert!(result.is_err(), "missing config file should fail bring-up");
    }

    /// A plan whose only read is the engine's stream view for input stream `stream_id`,
    /// projecting the users fixture's `(id BIGINT, name VARCHAR)` schema — the shape a front
    /// end emits where it would otherwise emit a file scan.
    fn stream_read_plan(stream_id: u64) -> Vec<u8> {
        use substrait::proto::read_rel::{NamedTable, ReadType};
        use substrait::proto::{
            NamedStruct, Plan, PlanRel, ReadRel, Rel, RelRoot, Type, plan_rel, rel, r#type,
        };

        let names = vec!["id".to_string(), "name".to_string()];
        let types = vec![
            Type {
                kind: Some(r#type::Kind::I64(r#type::I64 {
                    type_variation_reference: 0,
                    nullability: r#type::Nullability::Nullable as i32,
                })),
            },
            Type {
                kind: Some(r#type::Kind::String(r#type::String {
                    type_variation_reference: 0,
                    nullability: r#type::Nullability::Nullable as i32,
                })),
            },
        ];
        let read = Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                base_schema: Some(NamedStruct {
                    names: names.clone(),
                    r#struct: Some(r#type::Struct {
                        types,
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Required as i32,
                    }),
                }),
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: vec![super::stream_view_name(stream_id)],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(read),
                    names,
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    /// `stream_read_plan` for the decimal fixture's `(price DECIMAL(15,2), name VARCHAR)`
    /// schema.
    fn decimal_stream_read_plan(stream_id: u64) -> Vec<u8> {
        use substrait::proto::read_rel::{NamedTable, ReadType};
        use substrait::proto::{
            NamedStruct, Plan, PlanRel, ReadRel, Rel, RelRoot, Type, plan_rel, rel, r#type,
        };

        let names = vec!["price".to_string(), "name".to_string()];
        let types = vec![
            Type {
                kind: Some(r#type::Kind::Decimal(r#type::Decimal {
                    precision: 15,
                    scale: 2,
                    type_variation_reference: 0,
                    nullability: r#type::Nullability::Nullable as i32,
                })),
            },
            Type {
                kind: Some(r#type::Kind::String(r#type::String {
                    type_variation_reference: 0,
                    nullability: r#type::Nullability::Nullable as i32,
                })),
            },
        ];
        let read = Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                base_schema: Some(NamedStruct {
                    names: names.clone(),
                    r#struct: Some(r#type::Struct {
                        types,
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Required as i32,
                    }),
                }),
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: vec![super::stream_view_name(stream_id)],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(read),
                    names,
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    /// Flattens a result into sorted `(scaled price, name)` rows. DECIMAL(15,2) arrives as
    /// Decimal128, so the price is its scaled integer (1.50 -> 150).
    fn decimal_rows(result: &super::SubstraitResult) -> Vec<(i128, String)> {
        let mut rows = Vec::new();
        for batch in &result.batches {
            let prices = batch
                .column(0)
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .expect("price column");
            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("name column");
            for i in 0..batch.num_rows() {
                rows.push((prices.value(i), names.value(i).to_string()));
            }
        }
        rows.sort();
        rows
    }

    /// Flattens a result into sorted `(id, name)` rows, so comparisons are by value and
    /// independent of batch boundaries.
    fn rows(result: &super::SubstraitResult) -> Vec<(i64, String)> {
        let mut rows = Vec::new();
        for batch in &result.batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("id column");
            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("name column");
            for i in 0..batch.num_rows() {
                rows.push((ids.value(i), names.value(i).to_string()));
            }
        }
        rows.sort();
        rows
    }

    /// The decisive equivalence for the packed FFI pair: a fragment hop carried by the staging
    /// arena (`export_packed` → transmit-from-lease → `push_packed`) must deliver exactly the
    /// values the proven in-process `relay_from` hop delivers for the identical plan pair.
    /// Also pins the surrounding contracts: drained-stream export is `None`, push after EOS is
    /// a loud error, and leases release (and the bump head resets) before the receiver runs —
    /// which only works if push really copied the data out of the lease. Requires a GPU.
    #[test]
    fn packed_hop_matches_relay_hop() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        // The arena is constructed at context bring-up, only when this is set.
        // SAFETY: the GPU lock is held, so no other thread touches the environment here.
        unsafe { std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "64MiB") };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let sender_plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );
        let receiver_plan = stream_read_plan(0);

        let ctx = SiriusContext::new().expect("bring up sirius context");
        assert_eq!(ctx.staging_capacity().unwrap(), 64 << 20);
        assert_ne!(ctx.staging_base().unwrap(), 0);

        // Declares `(id, name)` on input stream 0 and builds the stream-view read. The declared
        // cardinality is what a CN feeds from parked/staged row counts; on this single-stream
        // plan it must be harmless, and it proves the declare -> bind -> optimize path end to
        // end on a real build.
        let make_receiver = || {
            let mut receiver = ctx.fragment().unwrap();
            receiver.declare_input_column(0, "id", "BIGINT").unwrap();
            receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
            receiver.declare_input_cardinality(0, 3).unwrap();
            receiver.build(&receiver_plan).unwrap();
            receiver
        };

        // Reference: the proven in-process native relay.
        let relay_result = {
            let mut sender = ctx.fragment().unwrap();
            sender.declare_output(0).unwrap();
            sender.build(&sender_plan).unwrap();
            sender.run().unwrap();

            // The receiver-side cardinality source for a LOCAL hop: exact parked rows, counted
            // without draining the stream.
            assert_eq!(sender.output_row_count(0).unwrap(), 3);
            assert!(sender.output_batch_count(0).unwrap() > 0);

            let mut receiver = make_receiver();
            let moved = receiver.relay_from(&mut sender, 0, 0, 0).unwrap();
            assert!(moved > 0, "the relay hop must carry batches");
            receiver.run().unwrap();
            receiver.into_arrow().unwrap()
        };

        // The same hop through the staging arena as packed bytes.
        let packed_result = {
            let mut sender = ctx.fragment().unwrap();
            sender.declare_output(0).unwrap();
            sender.build(&sender_plan).unwrap();
            sender.run().unwrap();

            let mut staged = Vec::new();
            while let Some(batch) = sender.export_packed(0).unwrap() {
                assert!(batch.len > 0, "a packed batch carries payload bytes");
                assert!(!batch.metadata.is_empty(), "pack metadata is never empty");
                assert!(batch.rows > 0, "a non-empty packed batch reports its rows");
                staged.push(batch);
            }
            assert!(!staged.is_empty(), "the sender parked batches to export");
            // The receiver-side cardinality source for a REMOTE hop: per-batch exact row counts
            // that ride the transmit frames and sum to the stream's total.
            assert_eq!(staged.iter().map(|batch| batch.rows).sum::<u64>(), 3);
            // A drained stream is `None`, not an error.
            assert!(sender.export_packed(0).unwrap().is_none());

            let mut receiver = make_receiver();
            for batch in &staged {
                receiver.push_packed(0, batch).unwrap();
            }
            receiver.close_input(0, 0).unwrap();

            // A push after EOS must refuse loudly, never vanish.
            let refused = receiver.push_packed(0, &staged[0]);
            assert!(refused.is_err(), "push_packed after close_input must error");
            assert!(refused.unwrap_err().what().contains("already ended"));

            // Copy-out-on-arrival: the data left the leases at push time, so they can all go
            // back before the receiver runs...
            for batch in &staged {
                ctx.staging_release(batch.offset).unwrap();
            }
            // ...and with nothing outstanding the free list coalesced back to one whole-arena
            // block, so the next lease lands at the base.
            let probe = ctx.staging_lease(1024).unwrap();
            assert_eq!(probe, 0);
            ctx.staging_release(probe).unwrap();

            receiver.run().unwrap();
            receiver.into_arrow().unwrap()
        };

        assert_eq!(rows(&relay_result), rows(&packed_result));
        assert_eq!(
            rows(&packed_result),
            vec![
                (1, "a".to_string()),
                (2, "b".to_string()),
                (3, "c".to_string()),
            ]
        );

        // Keep the arena out of the other tests' context bring-ups.
        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
    }

    /// The zero-row export contract: a zero-row batch leaves `export_packed` as a metadata-only
    /// frame (`offset == 0`, `len == 0`) holding NO staging lease, and the frame round-trips
    /// through `push_packed` into an empty result. Pins the q15 leak: `export_packed` used to
    /// lease `total + slack` even for `total == 0`, while the transport contract says `len == 0`
    /// means nothing to release — every empty cross-CN batch orphaned a >= 8 MiB lease that
    /// pinned the arena's bump head for the process lifetime, exhausting it after ~20 passing
    /// queries. The zero-row batch comes from an empty byte-range split (a range inside one row
    /// group scans zero row groups and emits exactly one empty batch). Requires a GPU.
    #[test]
    fn zero_row_export_is_metadata_only_and_holds_no_lease() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        // The arena is constructed at context bring-up, only when this is set.
        // SAFETY: the GPU lock is held, so no other thread touches the environment here.
        unsafe { std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "64MiB") };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let names = vec!["id".to_string(), "name".to_string()];
        // A byte range strictly inside the file's single row group owns no row groups: the
        // zero-row sender instance of a distributed scan (q15's leaking shape).
        let empty_split_plan =
            local_files_plan_ranged(&[(path.to_str().unwrap(), 10, 5)], names.clone());

        let ctx = SiriusContext::new().expect("bring up sirius context");
        let capacity = ctx.staging_capacity().unwrap();

        let mut sender = ctx.fragment().unwrap();
        sender.declare_output(0).unwrap();
        sender.build(&empty_split_plan).unwrap();
        sender.run().unwrap();

        let mut staged = Vec::new();
        while let Some(batch) = sender.export_packed(0).unwrap() {
            assert!(!batch.metadata.is_empty(), "pack metadata is never empty");
            staged.push(batch);
        }
        assert!(
            staged.iter().any(|batch| batch.len == 0),
            "the empty split must park a zero-row batch to export"
        );
        for batch in &staged {
            if batch.len == 0 {
                assert_eq!(batch.offset, 0, "a metadata-only frame names no lease");
                assert_eq!(batch.rows, 0, "a metadata-only frame carries zero rows");
            } else {
                ctx.staging_release(batch.offset).unwrap();
            }
        }

        // The reclaim guarantee: with only `len > 0` leases released (a metadata-only frame has
        // nothing to release), zero leases are outstanding and the bump head is back at the
        // base, so the ENTIRE arena is grantable as one lease. Under the leak this throws
        // exhaustion — the orphaned slack lease still pins the head.
        let probe = ctx.staging_lease(capacity).expect(
            "the whole arena must be grantable again after a zero-row export cycle \
             (an outstanding lease here is the q15 leak)",
        );
        assert_eq!(probe, 0);
        ctx.staging_release(probe).unwrap();

        // The frame is a legitimate wire citizen: pushing it delivers an empty stream, not an
        // error, and the receiver terminates with zero rows.
        let mut receiver = ctx.fragment().unwrap();
        receiver.declare_input_column(0, "id", "BIGINT").unwrap();
        receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
        receiver.build(&stream_read_plan(0)).unwrap();
        for batch in &staged {
            receiver.push_packed(0, batch).unwrap();
        }
        receiver.close_input(0, 0).unwrap();
        receiver.run().unwrap();
        let result = receiver.into_arrow().unwrap();
        assert_eq!(rows(&result), Vec::new(), "an empty split carries no rows");

        // Keep the arena out of the other tests' context bring-ups.
        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
    }

    /// Writes the row a two-phase merge fragment reads off the exchange for
    /// `sum(q), avg(q), count(*) GROUP BY rf, ls`: the grouping keys, then one column per
    /// partial-state column the partial fragment ships — avg's state being the sum/count pair.
    /// Two rows per group stand for two partial senders.
    fn write_merge_states_parquet(path: &Path) {
        write_merge_states(path, States::Normal);
    }

    /// Which partial states arrive at the merge fragment.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum States {
        /// Two senders per group, every group with values.
        Normal,
        /// No rows at all: the partition a hash fan-out left empty on one compute node.
        Empty,
        /// A group whose avg argument was NULL in every row: a zero count with a NULL sum,
        /// which is the case the finalizing division guards against.
        ZeroCount,
    }

    /// Writes the exchange row a merge fragment reads, in one of the [`States`] shapes.
    fn write_merge_states(path: &Path, states: States) {
        use arrow_array::Float64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("rf", DataType::Utf8, true),
            Field::new("ls", DataType::Utf8, true),
            Field::new("q_sum", DataType::Float64, true),
            Field::new("a_sum", DataType::Float64, true),
            Field::new("a_cnt", DataType::Int64, true),
            Field::new("c_cnt", DataType::Int64, true),
        ]));
        let (rf, ls, q_sum, a_sum, a_cnt, c_cnt): (
            Vec<&str>,
            Vec<&str>,
            Vec<Option<f64>>,
            Vec<Option<f64>>,
            Vec<i64>,
            Vec<i64>,
        ) = match states {
            States::Normal => (
                vec!["A", "A", "N", "N", "R", "R"],
                vec!["F", "F", "O", "O", "F", "F"],
                vec![
                    Some(10.0),
                    Some(20.0),
                    Some(3.0),
                    Some(7.0),
                    Some(5.0),
                    Some(5.0),
                ],
                vec![
                    Some(10.0),
                    Some(20.0),
                    Some(3.0),
                    Some(7.0),
                    Some(5.0),
                    Some(5.0),
                ],
                vec![2, 4, 1, 1, 2, 3],
                vec![2, 4, 1, 1, 2, 3],
            ),
            // The rows exist but never reach the merge fragment: an empty partition is an
            // input stream that ends without a batch, not a zero-row file.
            States::Empty => (
                vec!["A"],
                vec!["F"],
                vec![Some(1.0)],
                vec![Some(1.0)],
                vec![1],
                vec![1],
            ),
            // Group ("A","F") saw only NULL values: no sum, nothing counted.
            States::ZeroCount => (
                vec!["A", "A", "N", "N"],
                vec!["F", "F", "O", "O"],
                vec![None, None, Some(3.0), Some(7.0)],
                vec![None, None, Some(3.0), Some(7.0)],
                vec![0, 0, 1, 1],
                vec![2, 4, 1, 1],
            ),
        };
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(rf)),
            Arc::new(StringArray::from(ls)),
            Arc::new(Float64Array::from(q_sum)),
            Arc::new(Float64Array::from(a_sum)),
            Arc::new(Int64Array::from(a_cnt)),
            Arc::new(Int64Array::from(c_cnt)),
        ];
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// The exchange row a two-phase merge fragment reads, as `(name, DuckDB type)` — the
    /// declaration the CN derives from the translator's `stream_inputs`.
    const MERGE_STATE_COLUMNS: [(&str, &str); 6] = [
        ("rf", "VARCHAR"),
        ("ls", "VARCHAR"),
        ("q_sum", "DOUBLE"),
        ("a_sum", "DOUBLE"),
        ("a_cnt", "BIGINT"),
        ("c_cnt", "BIGINT"),
    ];

    /// Where a merge shape reads its partial states from.
    #[derive(Clone, Copy, Debug)]
    enum Source<'a> {
        /// A parquet file standing in for the exchange row.
        Parquet(&'a str),
        /// The engine's view of input stream 0 — what a real merge fragment reads.
        Stream,
    }

    /// Builds the merge fragment's plan shape over the partial-state fixture:
    /// `Aggregate(sum × 4) -> Project(finalize) [-> Project(sort tuple) -> Sort]`, which is what
    /// the StarRocks translator emits for a grouped two-phase avg (with an ORDER BY above it).
    fn merge_shape_plan(source: Source<'_>, with_sort: bool) -> Vec<u8> {
        use substrait::proto::expression::field_reference::{ReferenceType as RefType, RootType};
        use substrait::proto::expression::reference_segment::ReferenceType as SegmentType;
        use substrait::proto::expression::{
            FieldReference, IfThen, Literal, ReferenceSegment, RexType, ScalarFunction,
            field_reference, if_then, literal, reference_segment,
        };
        use substrait::proto::extensions::{
            SimpleExtensionDeclaration, SimpleExtensionUrn, simple_extension_declaration,
        };
        use substrait::proto::function_argument::ArgType;
        use substrait::proto::read_rel::local_files::FileOrFiles;
        use substrait::proto::read_rel::local_files::file_or_files::{
            FileFormat, ParquetReadOptions, PathType,
        };
        use substrait::proto::read_rel::{LocalFiles, ReadType};
        use substrait::proto::rel_common::{Emit, EmitKind};
        use substrait::proto::{
            AggregateFunction, AggregateRel, Expression, FunctionArgument, Plan, PlanRel,
            ProjectRel, ReadRel, Rel, RelCommon, RelRoot, SortField, SortRel, Type,
            aggregate_function, aggregate_rel, plan_rel, rel, sort_field, r#type,
        };

        // Anchors: 1 = arithmetic sum, 2 = arithmetic divide, 3 = comparison equal.
        let extension_urns = vec![
            SimpleExtensionUrn {
                extension_urn_anchor: 1,
                urn: "extension:io.substrait:functions_arithmetic".to_string(),
            },
            SimpleExtensionUrn {
                extension_urn_anchor: 2,
                urn: "extension:io.substrait:functions_comparison".to_string(),
            },
        ];
        let declare = |urn_anchor: u32, anchor: u32, name: &str| SimpleExtensionDeclaration {
            mapping_type: Some(
                simple_extension_declaration::MappingType::ExtensionFunction(
                    simple_extension_declaration::ExtensionFunction {
                        extension_urn_reference: urn_anchor,
                        function_anchor: anchor,
                        name: name.to_string(),
                    },
                ),
            ),
        };
        let extensions = vec![
            declare(1, 1, "sum"),
            declare(1, 2, "divide"),
            declare(2, 3, "equal"),
        ];

        let fp64 = || Type {
            kind: Some(r#type::Kind::Fp64(r#type::Fp64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        };
        let i64_ty = || Type {
            kind: Some(r#type::Kind::I64(r#type::I64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        };
        let bool_ty = || Type {
            kind: Some(r#type::Kind::Bool(r#type::Boolean {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        };
        let field = |index: i32| Expression {
            rex_type: Some(RexType::Selection(Box::new(FieldReference {
                reference_type: Some(RefType::DirectReference(ReferenceSegment {
                    reference_type: Some(SegmentType::StructField(Box::new(
                        reference_segment::StructField {
                            field: index,
                            child: None,
                        },
                    ))),
                })),
                root_type: Some(RootType::RootReference(field_reference::RootReference {})),
            }))),
        };
        let call = |anchor: u32, args: Vec<Expression>, ty: Type| Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: anchor,
                arguments: args
                    .into_iter()
                    .map(|expr| FunctionArgument {
                        arg_type: Some(ArgType::Value(expr)),
                    })
                    .collect(),
                output_type: Some(ty),
                ..Default::default()
            })),
        };
        let cast_fp64 = |input: Expression| Expression {
            rex_type: Some(RexType::Cast(Box::new(
                substrait::proto::expression::Cast {
                    r#type: Some(fp64()),
                    input: Some(Box::new(input)),
                    failure_behavior:
                        substrait::proto::expression::cast::FailureBehavior::ThrowException as i32,
                },
            ))),
        };

        let read = match source {
            Source::Parquet(path) => Rel {
                rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                    read_type: Some(ReadType::LocalFiles(LocalFiles {
                        items: vec![FileOrFiles {
                            path_type: Some(PathType::UriFile(path.to_string())),
                            file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                            ..Default::default()
                        }],
                        ..Default::default()
                    })),
                    ..Default::default()
                }))),
            },
            Source::Stream => {
                let varchar = || Type {
                    kind: Some(r#type::Kind::Varchar(r#type::VarChar {
                        length: 65535,
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Nullable as i32,
                    })),
                };
                Rel {
                    rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                        base_schema: Some(substrait::proto::NamedStruct {
                            names: MERGE_STATE_COLUMNS
                                .iter()
                                .map(|(name, _)| name.to_string())
                                .collect(),
                            r#struct: Some(r#type::Struct {
                                types: vec![
                                    varchar(),
                                    varchar(),
                                    fp64(),
                                    fp64(),
                                    i64_ty(),
                                    i64_ty(),
                                ],
                                type_variation_reference: 0,
                                nullability: r#type::Nullability::Required as i32,
                            }),
                        }),
                        read_type: Some(ReadType::NamedTable(
                            substrait::proto::read_rel::NamedTable {
                                names: vec![super::stream_view_name(0)],
                                ..Default::default()
                            },
                        )),
                        ..Default::default()
                    }))),
                }
            }
        };

        // One sum per state column: the summed sum, avg's summed sum and summed count, and the
        // merged count.
        let measure = |field_index: i32, ty: Type| aggregate_rel::Measure {
            measure: Some(AggregateFunction {
                function_reference: 1,
                arguments: vec![FunctionArgument {
                    arg_type: Some(ArgType::Value(field(field_index))),
                }],
                output_type: Some(ty),
                invocation: aggregate_function::AggregationInvocation::All as i32,
                phase: substrait::proto::AggregationPhase::IntermediateToResult as i32,
                ..Default::default()
            }),
            filter: None,
        };
        #[allow(deprecated)]
        let grouping = aggregate_rel::Grouping {
            grouping_expressions: Vec::new(),
            expression_references: vec![0, 1],
        };
        let aggregate = Rel {
            rel_type: Some(rel::RelType::Aggregate(Box::new(AggregateRel {
                input: Some(Box::new(read)),
                groupings: vec![grouping],
                grouping_expressions: vec![field(0), field(1)],
                measures: vec![
                    measure(2, fp64()),
                    measure(3, fp64()),
                    measure(4, i64_ty()),
                    measure(5, i64_ty()),
                ],
                ..Default::default()
            }))),
        };

        // Aggregate output: [rf, ls, sum(q), sum(a_sum), sum(a_cnt), sum(c_cnt)]. The average
        // is the state divided, with SQL's empty-input NULL guarding the zero count.
        let average = Expression {
            rex_type: Some(RexType::IfThen(Box::new(IfThen {
                ifs: vec![if_then::IfClause {
                    r#if: Some(call(
                        3,
                        vec![
                            field(4),
                            Expression {
                                rex_type: Some(RexType::Literal(Literal {
                                    literal_type: Some(literal::LiteralType::I64(0)),
                                    ..Default::default()
                                })),
                            },
                        ],
                        bool_ty(),
                    )),
                    then: Some(Expression {
                        rex_type: Some(RexType::Literal(Literal {
                            literal_type: Some(literal::LiteralType::Null(fp64())),
                            ..Default::default()
                        })),
                    }),
                }],
                r#else: Some(Box::new(call(
                    2,
                    vec![field(3), cast_fp64(field(4))],
                    fp64(),
                ))),
            }))),
        };
        let project = |input: Rel, expressions: Vec<Expression>, input_width: i32| Rel {
            rel_type: Some(rel::RelType::Project(Box::new(ProjectRel {
                common: Some(RelCommon {
                    emit_kind: Some(EmitKind::Emit(Emit {
                        output_mapping: (input_width..input_width + expressions.len() as i32)
                            .collect(),
                    })),
                    ..Default::default()
                }),
                input: Some(Box::new(input)),
                expressions,
                ..Default::default()
            }))),
        };
        // The finalizing projection: keys, the merged sum, the divided average, the count.
        let finalized = project(
            aggregate,
            vec![field(0), field(1), field(2), average, field(5)],
            6,
        );

        let root_input = if with_sort {
            // A StarRocks SORT node materializes its sort tuple first, then orders it.
            let sort_tuple = project(
                finalized,
                vec![field(0), field(1), field(2), field(3), field(4)],
                5,
            );
            let ascending = |index: i32| SortField {
                expr: Some(field(index)),
                sort_kind: Some(sort_field::SortKind::Direction(
                    sort_field::SortDirection::AscNullsLast as i32,
                )),
            };
            Rel {
                rel_type: Some(rel::RelType::Sort(Box::new(SortRel {
                    input: Some(Box::new(sort_tuple)),
                    sorts: vec![ascending(0), ascending(1)],
                    ..Default::default()
                }))),
            }
        } else {
            finalized
        };

        Plan {
            extension_urns,
            extensions,
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(root_input),
                    names: ["rf", "ls", "q", "a", "c"]
                        .into_iter()
                        .map(str::to_string)
                        .collect(),
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    /// Reads a merged-count column as integers, whichever width the engine returned it in:
    /// `sum(BIGINT)` binds to DuckDB HUGEINT, which arrives as a 38-digit decimal.
    fn count_values(column: &ArrayRef) -> Vec<i64> {
        use arrow_array::Decimal128Array;

        if let Some(values) = column.as_any().downcast_ref::<Int64Array>() {
            return (0..values.len()).map(|i| values.value(i)).collect();
        }
        let values = column
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap_or_else(|| panic!("count column has type {:?}", column.data_type()));
        (0..values.len())
            .map(|i| i64::try_from(values.value(i)).expect("count fits in i64"))
            .collect()
    }

    /// Runs one merge shape on the GPU under a watchdog, returning its `(rf, ls, q, a, c)` rows.
    ///
    /// A plan the engine never finishes is the one outcome that must not reach a human as
    /// silence, so the watchdog turns it into a failure with a name attached. The engine call
    /// cannot be cancelled once it is stuck, so the process is torn down with it.
    fn run_merge_shape(with_sort: bool) -> Vec<MergeRow> {
        under_watchdog(format!("sort={with_sort}"), move || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("states.parquet");
            write_merge_states_parquet(&path);
            let plan = merge_shape_plan(Source::Parquet(path.to_str().unwrap()), with_sort);
            let ctx = SiriusContext::new().expect("bring up sirius context");
            ctx.execute_substrait_result(&plan)
                .map(|result| merge_rows(&result))
                .map_err(|err| err.to_string())
        })
    }

    /// Runs the merge shape the way a compute node runs it: a sender fragment parks the partial
    /// states as native GPU batches, and the merge fragment reads them through the engine's
    /// stream view — with its own output stream, since a merge fragment feeding an ORDER BY
    /// result fragment is a sender too.
    fn run_merge_fragment(with_sort: bool, to_stream: bool) -> Vec<MergeRow> {
        run_merge_fragment_over(with_sort, to_stream, States::Normal)
    }

    /// [`run_merge_fragment`] over a chosen partial-state shape.
    fn run_merge_fragment_over(with_sort: bool, to_stream: bool, states: States) -> Vec<MergeRow> {
        let label = format!("merge fragment sort={with_sort}/stream={to_stream}/{states:?}");
        under_watchdog(label, move || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("states.parquet");
            write_merge_states(&path, states);
            let sender_plan = local_files_plan(
                path.to_str().unwrap(),
                MERGE_STATE_COLUMNS
                    .iter()
                    .map(|(name, _)| name.to_string())
                    .collect(),
            );
            let merge_plan = merge_shape_plan(Source::Stream, with_sort);

            let ctx = SiriusContext::new().expect("bring up sirius context");
            let mut partial = ctx.fragment().map_err(|err| err.to_string())?;
            partial.declare_output(0).map_err(|err| err.to_string())?;
            partial.build(&sender_plan).map_err(|err| err.to_string())?;
            partial.run().map_err(|err| err.to_string())?;

            let mut merge = ctx.fragment().map_err(|err| err.to_string())?;
            for (name, ty) in MERGE_STATE_COLUMNS {
                merge
                    .declare_input_column(0, name, ty)
                    .map_err(|err| err.to_string())?;
            }
            if to_stream {
                merge.declare_output(0).map_err(|err| err.to_string())?;
            }
            merge.build(&merge_plan).map_err(|err| err.to_string())?;
            if states == States::Empty {
                // The sender had nothing for this partition: the stream just ends.
                merge.close_input(0, 0).map_err(|err| err.to_string())?;
            } else {
                let moved = merge
                    .relay_from(&mut partial, 0, 0, 0)
                    .map_err(|err| err.to_string())?;
                assert!(moved > 0, "the partial fragment parked no batches");
            }
            merge.run().map_err(|err| err.to_string())?;

            if !to_stream {
                return merge
                    .into_arrow()
                    .map(|result| merge_rows(&result))
                    .map_err(|err| err.to_string());
            }
            // Drain the merge fragment's own output stream through a result fragment, which is
            // what the ORDER BY result fragment does on the cluster.
            let mut result_fragment = ctx.fragment().map_err(|err| err.to_string())?;
            for (name, ty) in merge_output_columns() {
                result_fragment
                    .declare_input_column(0, name, ty)
                    .map_err(|err| err.to_string())?;
            }
            result_fragment
                .build(&merge_output_read_plan())
                .map_err(|err| err.to_string())?;
            result_fragment
                .relay_from(&mut merge, 0, 0, 0)
                .map_err(|err| err.to_string())?;
            result_fragment.run().map_err(|err| err.to_string())?;
            result_fragment
                .into_arrow()
                .map(|result| merge_rows(&result))
                .map_err(|err| err.to_string())
        })
    }

    /// The columns a merge fragment emits: the keys, the merged sum, the finalized average, and
    /// the merged count (`sum(BIGINT)`, which the engine produces as HUGEINT).
    fn merge_output_columns() -> [(&'static str, &'static str); 5] {
        [
            ("rf", "VARCHAR"),
            ("ls", "VARCHAR"),
            ("q", "DOUBLE"),
            ("a", "DOUBLE"),
            ("c", "HUGEINT"),
        ]
    }

    /// A plain read of the merge fragment's output stream, for draining it into Arrow.
    fn merge_output_read_plan() -> Vec<u8> {
        use substrait::proto::read_rel::{NamedTable, ReadType};
        use substrait::proto::{
            NamedStruct, Plan, PlanRel, ReadRel, Rel, RelRoot, Type, plan_rel, rel, r#type,
        };

        let nullable = r#type::Nullability::Nullable as i32;
        let types = vec![
            Type {
                kind: Some(r#type::Kind::Varchar(r#type::VarChar {
                    length: 65535,
                    type_variation_reference: 0,
                    nullability: nullable,
                })),
            },
            Type {
                kind: Some(r#type::Kind::Varchar(r#type::VarChar {
                    length: 65535,
                    type_variation_reference: 0,
                    nullability: nullable,
                })),
            },
            Type {
                kind: Some(r#type::Kind::Fp64(r#type::Fp64 {
                    type_variation_reference: 0,
                    nullability: nullable,
                })),
            },
            Type {
                kind: Some(r#type::Kind::Fp64(r#type::Fp64 {
                    type_variation_reference: 0,
                    nullability: nullable,
                })),
            },
            Type {
                kind: Some(r#type::Kind::Decimal(r#type::Decimal {
                    precision: 38,
                    scale: 0,
                    type_variation_reference: 0,
                    nullability: nullable,
                })),
            },
        ];
        let names: Vec<String> = merge_output_columns()
            .iter()
            .map(|(name, _)| name.to_string())
            .collect();
        let read = Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                base_schema: Some(NamedStruct {
                    names: names.clone(),
                    r#struct: Some(r#type::Struct {
                        types,
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Required as i32,
                    }),
                }),
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: vec![super::stream_view_name(0)],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(read),
                    names,
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    /// One finalized merge row: the two keys, the merged sum, the average, the merged count.
    type MergeRow = (String, String, f64, f64, i64);

    /// Flattens a merge result into key-sorted rows, reading a NULL measure as NaN so the
    /// average of no values stays distinguishable from a zero.
    fn merge_rows(result: &super::SubstraitResult) -> Vec<MergeRow> {
        use arrow_array::{Array, Float64Array};

        let mut rows = Vec::new();
        for batch in &result.batches {
            let rf = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let ls = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let q = batch
                .column(2)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let a = batch
                .column(3)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            // A merged count is `sum(BIGINT)`, which DuckDB binds to HUGEINT and hands back as
            // a 38-digit decimal.
            let c = count_values(batch.column(4));
            for i in 0..batch.num_rows() {
                let measure = |values: &Float64Array, index: usize| {
                    if values.is_null(index) {
                        f64::NAN
                    } else {
                        values.value(index)
                    }
                };
                rows.push((
                    rf.value(i).to_string(),
                    ls.value(i).to_string(),
                    measure(q, i),
                    measure(a, i),
                    c[i],
                ));
            }
        }
        // Keys only: the measures are floats, and one NaN would make the ordering partial and
        // panic the comparison instead of failing the assertion.
        rows.sort_by(|left, right| (&left.0, &left.1).cmp(&(&right.0, &right.1)));
        rows
    }

    /// Runs `work` on its own engine thread and fails the test if it does not finish.
    ///
    /// A plan the engine never finishes is the one outcome that must not reach a human as
    /// silence, so the watchdog turns it into a failure with a name attached. The engine call
    /// cannot be cancelled once it is stuck, so the process is torn down with it.
    fn under_watchdog<T: Send + 'static>(
        label: String,
        work: impl FnOnce() -> Result<T, String> + Send + 'static,
    ) -> T {
        let (sender, receiver) = std::sync::mpsc::channel();
        let engine = std::thread::spawn(move || {
            let _ = sender.send(work());
        });

        match receiver.recv_timeout(std::time::Duration::from_secs(120)) {
            Ok(Ok(value)) => {
                // Join before returning: the engine context is dropped on that thread, and a
                // teardown racing the process exit is not what these tests are measuring.
                engine.join().expect("engine thread");
                value
            }
            Ok(Err(message)) => panic!("{label} failed: {message}"),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                panic!("{label}: the engine thread panicked (see the panic above)")
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                // The engine thread is wedged inside a GPU call and cannot be joined or
                // cancelled; report loudly and take the process down rather than hang the suite.
                eprintln!("{label} did not finish within 120s: the engine hung");
                std::process::exit(101);
            }
        }
    }

    /// The rows every merge shape must produce, with the average already divided.
    fn expected_merge_rows() -> Vec<MergeRow> {
        let average = |sum: f64, count: i64| sum / count as f64;
        vec![
            ("A".to_string(), "F".to_string(), 30.0, average(30.0, 6), 6),
            ("N".to_string(), "O".to_string(), 10.0, average(10.0, 2), 2),
            ("R".to_string(), "F".to_string(), 10.0, average(10.0, 5), 5),
        ]
    }

    /// The merge shape the translator emits for a two-phase avg, executed standalone: the
    /// summed state divided back into an average. Requires a GPU.
    #[test]
    fn merge_shape_finalizes_avg() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        assert_eq!(run_merge_shape(false), expected_merge_rows());
    }

    /// The whole merge fragment of an ORDER BY query: the finalizing projection with a sort
    /// above it. Requires a GPU.
    #[test]
    fn merge_shape_finalizes_avg_with_sort() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        assert_eq!(run_merge_shape(true), expected_merge_rows());
    }

    /// An input stream that ends without ever carrying a batch is what a hash fan-out hands the
    /// compute node that owns no keys. Reading it must finish with no rows: a fragment that
    /// waits forever for data that is never coming takes the whole query down with it, and does
    /// it silently. Requires a GPU.
    #[test]
    fn fragment_over_an_empty_input_stream_terminates() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let rows = under_watchdog("empty input stream".to_string(), || {
            let ctx = SiriusContext::new().expect("bring up sirius context");
            let mut receiver = ctx.fragment().map_err(|err| err.to_string())?;
            receiver
                .declare_input_column(0, "id", "BIGINT")
                .map_err(|err| err.to_string())?;
            receiver
                .declare_input_column(0, "name", "VARCHAR")
                .map_err(|err| err.to_string())?;
            receiver
                .build(&stream_read_plan(0))
                .map_err(|err| err.to_string())?;
            // No sender ever parked a batch for this partition; the stream just ends.
            receiver.close_input(0, 0).map_err(|err| err.to_string())?;
            receiver.run().map_err(|err| err.to_string())?;
            receiver
                .into_arrow()
                .map(|result| rows(&result))
                .map_err(|err| err.to_string())
        });
        assert!(rows.is_empty(), "an empty stream yields no rows: {rows:?}");
    }

    /// An inner join of two input streams, the shape the CN builds for a shuffle join:
    /// `Join[Inner, equal($0, $2)]` over `Read stream_0` (probe) and `Read stream_1` (build).
    fn inner_join_streams_plan() -> Vec<u8> {
        use substrait::proto::expression::field_reference::{ReferenceType as RefType, RootType};
        use substrait::proto::expression::reference_segment::ReferenceType as SegmentType;
        use substrait::proto::expression::{
            FieldReference, ReferenceSegment, RexType, ScalarFunction, field_reference,
            reference_segment,
        };
        use substrait::proto::extensions::{
            SimpleExtensionDeclaration, SimpleExtensionUrn, simple_extension_declaration,
        };
        use substrait::proto::function_argument::ArgType;
        use substrait::proto::read_rel::{NamedTable, ReadType};
        use substrait::proto::{
            Expression, FunctionArgument, JoinRel, NamedStruct, Plan, PlanRel, ReadRel, Rel,
            RelRoot, Type, join_rel, plan_rel, rel, r#type,
        };

        let i64_ty = || Type {
            kind: Some(r#type::Kind::I64(r#type::I64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        };
        let varchar = || Type {
            kind: Some(r#type::Kind::String(r#type::String {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        };
        let stream_read = |stream_id: u64, names: [&str; 2]| Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                base_schema: Some(NamedStruct {
                    names: names.iter().map(|name| name.to_string()).collect(),
                    r#struct: Some(r#type::Struct {
                        types: vec![i64_ty(), varchar()],
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Required as i32,
                    }),
                }),
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: vec![super::stream_view_name(stream_id)],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        let field = |index: i32| Expression {
            rex_type: Some(RexType::Selection(Box::new(FieldReference {
                reference_type: Some(RefType::DirectReference(ReferenceSegment {
                    reference_type: Some(SegmentType::StructField(Box::new(
                        reference_segment::StructField {
                            field: index,
                            child: None,
                        },
                    ))),
                })),
                root_type: Some(RootType::RootReference(field_reference::RootReference {})),
            }))),
        };
        // The equality condition over the combined row: probe key $0 = build key $2.
        let condition = Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: 1,
                arguments: [field(0), field(2)]
                    .into_iter()
                    .map(|expr| FunctionArgument {
                        arg_type: Some(ArgType::Value(expr)),
                    })
                    .collect(),
                output_type: Some(Type {
                    kind: Some(r#type::Kind::Bool(r#type::Boolean {
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Nullable as i32,
                    })),
                }),
                ..Default::default()
            })),
        };
        let join = Rel {
            rel_type: Some(rel::RelType::Join(Box::new(JoinRel {
                left: Some(Box::new(stream_read(0, ["id", "name"]))),
                right: Some(Box::new(stream_read(1, ["rid", "rname"]))),
                expression: Some(Box::new(condition)),
                r#type: join_rel::JoinType::Inner as i32,
                ..Default::default()
            }))),
        };
        Plan {
            extension_urns: vec![SimpleExtensionUrn {
                extension_urn_anchor: 1,
                urn: "extension:io.substrait:functions_comparison".to_string(),
            }],
            extensions: vec![SimpleExtensionDeclaration {
                mapping_type: Some(
                    simple_extension_declaration::MappingType::ExtensionFunction(
                        simple_extension_declaration::ExtensionFunction {
                            extension_urn_reference: 1,
                            function_anchor: 1,
                            name: "equal".to_string(),
                        },
                    ),
                ),
            }],
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(join),
                    names: ["id", "name", "rid", "rname"]
                        .into_iter()
                        .map(str::to_string)
                        .collect(),
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    /// A shuffle join's build stream can end with ZERO batches on one instance — every build row
    /// hashed to the other compute node (TPC-H q02's one-row region build does exactly this).
    /// The join must return no rows and, crucially, must TERMINATE: before the never-buildable
    /// fix the BUILD_PROBE state machine answered wait_for_build forever and the engine thread
    /// wedged in run(), silently, holding the query lifecycle. Running a second fragment on the
    /// same context proves the lifecycle closed and the engine is not head-of-line blocked.
    /// Requires a GPU.
    #[test]
    fn inner_join_over_an_empty_build_stream_terminates() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let joined = under_watchdog("empty build stream inner join".to_string(), || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("users.parquet");
            write_users_parquet(&path);
            let probe_plan = local_files_plan(
                path.to_str().unwrap(),
                vec!["id".to_string(), "name".to_string()],
            );

            let ctx = SiriusContext::new().expect("bring up sirius context");
            let run_join = |ctx: &SiriusContext| -> Result<usize, String> {
                let mut probe_sender = ctx.fragment().map_err(|err| err.to_string())?;
                probe_sender
                    .declare_output(0)
                    .map_err(|err| err.to_string())?;
                probe_sender
                    .build(&probe_plan)
                    .map_err(|err| err.to_string())?;
                probe_sender.run().map_err(|err| err.to_string())?;

                let mut join = ctx.fragment().map_err(|err| err.to_string())?;
                for (stream, id, name) in [(0, "id", "name"), (1, "rid", "rname")] {
                    join.declare_input_column(stream, id, "BIGINT")
                        .map_err(|err| err.to_string())?;
                    join.declare_input_column(stream, name, "VARCHAR")
                        .map_err(|err| err.to_string())?;
                }
                join.build(&inner_join_streams_plan())
                    .map_err(|err| err.to_string())?;
                let moved = join
                    .relay_from(&mut probe_sender, 0, 0, 0)
                    .map_err(|err| err.to_string())?;
                assert!(moved > 0, "the probe fragment parked no batches");
                // The build side ends without ever carrying a batch.
                join.close_input(1, 0).map_err(|err| err.to_string())?;
                join.run().map_err(|err| err.to_string())?;
                join.into_arrow()
                    .map(|result| {
                        result
                            .batches
                            .iter()
                            .map(arrow_array::RecordBatch::num_rows)
                            .sum()
                    })
                    .map_err(|err| err.to_string())
            };
            let first = run_join(&ctx)?;
            // The lifecycle must have closed: a second query on the same engine must not block.
            let second = run_join(&ctx)?;
            Ok((first, second))
        });
        assert_eq!(joined, (0, 0), "an empty build side joins to no rows");
    }

    /// A hash fan-out can leave one compute node's partition empty; the merge fragment there
    /// must still finish, with no rows. Requires a GPU.
    #[test]
    fn merge_fragment_over_an_empty_partition() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        assert_eq!(
            run_merge_fragment_over(true, false, States::Empty),
            Vec::new()
        );
    }

    /// A group whose avg argument was NULL in every row arrives with a zero count: the average
    /// of no values is NULL, and dividing by that count must not be a division by zero.
    /// Requires a GPU.
    #[test]
    fn merge_fragment_zero_count_group_is_null() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let rows = run_merge_fragment_over(false, false, States::ZeroCount);
        assert_eq!(rows.len(), 2);
        // ("A","F") counted nothing: SQL's average of no values is NULL, which arrives as a
        // null the reader surfaces as NaN — never a division by zero, and never 0.
        assert_eq!(rows[0].0, "A");
        assert!(
            rows[0].3.is_nan(),
            "a zero-count group must average to NULL, got {}",
            rows[0].3
        );
        assert_eq!(rows[0].4, 6, "its count still merges");
        assert_eq!(rows[1].3, 5.0, "the group with values keeps its average");
    }

    /// The merge fragment as a compute node runs it: partial states arriving as native GPU
    /// batches on an input stream, finalized, and parked on its own output stream. Requires a
    /// GPU.
    #[test]
    fn merge_fragment_over_a_stream_finalizes_avg() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        assert_eq!(run_merge_fragment(false, false), expected_merge_rows());
    }

    /// The same fragment with the ORDER BY sort above it and a downstream result fragment
    /// draining its output stream — the exact shape the two-CN demo runs. Requires a GPU.
    #[test]
    fn merge_fragment_with_sort_feeds_its_output_stream() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        assert_eq!(run_merge_fragment(true, true), expected_merge_rows());
    }

    /// Writes `(name VARCHAR, price DECIMAL(15,2))`, the shape a TPC-H measure column has
    /// before a two-phase aggregation lowers it.
    fn write_prices_parquet(path: &Path) {
        use arrow_array::Decimal128Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("price", DataType::Decimal128(15, 2), true),
        ]));
        let names: ArrayRef = Arc::new(StringArray::from(vec!["a", "a", "b", "b"]));
        let prices: ArrayRef = Arc::new(
            Decimal128Array::from(vec![100, 200, 300, 400])
                .with_precision_and_scale(15, 2)
                .unwrap(),
        );
        let batch = RecordBatch::try_new(schema.clone(), vec![names, prices]).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Builds the partial half of a two-phase avg over a decimal column:
    /// `Aggregate[$0 => sum(cast($1 AS DOUBLE)), count($1)]` — the expansion the translator
    /// emits, summing the lowered value and counting the raw one.
    fn partial_avg_plan(path: &str) -> Vec<u8> {
        use substrait::proto::expression::field_reference::{ReferenceType as RefType, RootType};
        use substrait::proto::expression::reference_segment::ReferenceType as SegmentType;
        use substrait::proto::expression::{
            FieldReference, ReferenceSegment, RexType, field_reference, reference_segment,
        };
        use substrait::proto::extensions::{
            SimpleExtensionDeclaration, SimpleExtensionUrn, simple_extension_declaration,
        };
        use substrait::proto::function_argument::ArgType;
        use substrait::proto::read_rel::local_files::FileOrFiles;
        use substrait::proto::read_rel::local_files::file_or_files::{
            FileFormat, ParquetReadOptions, PathType,
        };
        use substrait::proto::read_rel::{LocalFiles, ReadType};
        use substrait::proto::{
            AggregateFunction, AggregateRel, Expression, FunctionArgument, Plan, PlanRel, ReadRel,
            Rel, RelRoot, Type, aggregate_function, aggregate_rel, plan_rel, rel, r#type,
        };

        let extension_urns = vec![
            SimpleExtensionUrn {
                extension_urn_anchor: 1,
                urn: "extension:io.substrait:functions_arithmetic".to_string(),
            },
            SimpleExtensionUrn {
                extension_urn_anchor: 2,
                urn: "extension:io.substrait:functions_aggregate_generic".to_string(),
            },
        ];
        let declare = |urn_anchor: u32, anchor: u32, name: &str| SimpleExtensionDeclaration {
            mapping_type: Some(
                simple_extension_declaration::MappingType::ExtensionFunction(
                    simple_extension_declaration::ExtensionFunction {
                        extension_urn_reference: urn_anchor,
                        function_anchor: anchor,
                        name: name.to_string(),
                    },
                ),
            ),
        };
        let extensions = vec![declare(1, 1, "sum"), declare(2, 2, "count")];

        let fp64 = || Type {
            kind: Some(r#type::Kind::Fp64(r#type::Fp64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        };
        let i64_ty = || Type {
            kind: Some(r#type::Kind::I64(r#type::I64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        };
        let field = |index: i32| Expression {
            rex_type: Some(RexType::Selection(Box::new(FieldReference {
                reference_type: Some(RefType::DirectReference(ReferenceSegment {
                    reference_type: Some(SegmentType::StructField(Box::new(
                        reference_segment::StructField {
                            field: index,
                            child: None,
                        },
                    ))),
                })),
                root_type: Some(RootType::RootReference(field_reference::RootReference {})),
            }))),
        };
        let cast_fp64 = |input: Expression| Expression {
            rex_type: Some(RexType::Cast(Box::new(
                substrait::proto::expression::Cast {
                    r#type: Some(fp64()),
                    input: Some(Box::new(input)),
                    failure_behavior:
                        substrait::proto::expression::cast::FailureBehavior::ThrowException as i32,
                },
            ))),
        };
        let measure = |anchor: u32, argument: Expression, ty: Type| aggregate_rel::Measure {
            measure: Some(AggregateFunction {
                function_reference: anchor,
                arguments: vec![FunctionArgument {
                    arg_type: Some(ArgType::Value(argument)),
                }],
                output_type: Some(ty),
                invocation: aggregate_function::AggregationInvocation::All as i32,
                phase: substrait::proto::AggregationPhase::InitialToIntermediate as i32,
                ..Default::default()
            }),
            filter: None,
        };

        let read = Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                read_type: Some(ReadType::LocalFiles(LocalFiles {
                    items: vec![FileOrFiles {
                        path_type: Some(PathType::UriFile(path.to_string())),
                        file_format: Some(FileFormat::Parquet(ParquetReadOptions {})),
                        ..Default::default()
                    }],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        #[allow(deprecated)]
        let grouping = aggregate_rel::Grouping {
            grouping_expressions: Vec::new(),
            expression_references: vec![0],
        };
        let aggregate = Rel {
            rel_type: Some(rel::RelType::Aggregate(Box::new(AggregateRel {
                input: Some(Box::new(read)),
                groupings: vec![grouping],
                grouping_expressions: vec![field(0)],
                measures: vec![
                    measure(1, cast_fp64(field(1)), fp64()),
                    measure(2, field(1), i64_ty()),
                ],
                ..Default::default()
            }))),
        };
        Plan {
            extension_urns,
            extensions,
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(aggregate),
                    names: ["name", "a", "a__count"]
                        .into_iter()
                        .map(str::to_string)
                        .collect(),
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    /// Builds the partial half of `sum(price), avg(price) GROUP BY name`: the plain sum and
    /// avg's sum half are the *same* measure over the same argument, which is the shape the
    /// avg expansion creates whenever a query sums and averages one column.
    fn partial_duplicate_sum_plan(path: &str) -> Vec<u8> {
        use prost::Message as _;
        use substrait::proto::{Plan, rel};

        let mut plan = Plan::decode(partial_avg_plan(path).as_slice()).unwrap();
        let root = match plan.relations[0].rel_type.as_mut().unwrap() {
            substrait::proto::plan_rel::RelType::Root(root) => root,
            other => panic!("expected a root relation, got {other:?}"),
        };
        let rel::RelType::Aggregate(aggregate) =
            root.input.as_mut().unwrap().rel_type.as_mut().unwrap()
        else {
            panic!("expected an aggregate relation");
        };
        // sum, sum (avg's half), count — the plain sum duplicating avg's sum exactly.
        aggregate.measures.insert(0, aggregate.measures[0].clone());
        root.names = ["name", "s", "a", "a__count"]
            .into_iter()
            .map(str::to_string)
            .collect();
        plan.encode_to_vec()
    }

    /// Runs the partial half of `sum(price), avg(price)`, returning `(name, sum, sum, count)`.
    fn run_partial_duplicate_sum() -> Vec<(String, f64, f64, i64)> {
        use arrow_array::Float64Array;

        under_watchdog("partial duplicate sum".to_string(), move || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("prices.parquet");
            write_prices_parquet(&path);
            let plan = partial_duplicate_sum_plan(path.to_str().unwrap());
            let ctx = SiriusContext::new().expect("bring up sirius context");
            ctx.execute_substrait_result(&plan)
                .map(|result| {
                    let mut rows = Vec::new();
                    for batch in &result.batches {
                        let name = batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap();
                        let sum = batch
                            .column(1)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap();
                        let avg_sum = batch
                            .column(2)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap();
                        let count = count_values(batch.column(3));
                        for i in 0..batch.num_rows() {
                            rows.push((
                                name.value(i).to_string(),
                                sum.value(i),
                                avg_sum.value(i),
                                count[i],
                            ));
                        }
                    }
                    rows.sort_by(|left, right| left.0.cmp(&right.0));
                    rows
                })
                .map_err(|err| err.to_string())
        })
    }

    /// A query that both sums and averages one column makes the partial fragment emit the same
    /// measure twice. Requires a GPU.
    #[test]
    fn partial_sum_and_avg_of_one_column() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        assert_eq!(
            run_partial_duplicate_sum(),
            vec![
                ("a".to_string(), 3.0, 3.0, 2),
                ("b".to_string(), 7.0, 7.0, 2)
            ]
        );
    }

    /// Runs the partial half of a two-phase avg over a decimal column, returning
    /// `(name, sum, count)` rows.
    fn run_partial_avg() -> Vec<(String, f64, i64)> {
        use arrow_array::Float64Array;

        under_watchdog("partial avg".to_string(), move || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("prices.parquet");
            write_prices_parquet(&path);
            let plan = partial_avg_plan(path.to_str().unwrap());
            let ctx = SiriusContext::new().expect("bring up sirius context");
            ctx.execute_substrait_result(&plan)
                .map(|result| {
                    let mut rows = Vec::new();
                    for batch in &result.batches {
                        let name = batch
                            .column(0)
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap();
                        let sum = batch
                            .column(1)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap();
                        let count = count_values(batch.column(2));
                        for i in 0..batch.num_rows() {
                            rows.push((name.value(i).to_string(), sum.value(i), count[i]));
                        }
                    }
                    rows.sort_by(|left, right| left.0.cmp(&right.0));
                    rows
                })
                .map_err(|err| err.to_string())
        })
    }

    /// The partial half of a two-phase avg counts the values it sums. Counting the raw decimal
    /// column is the shape the expansion emits by default. Requires a GPU.
    #[test]
    fn partial_avg_counts_a_decimal_column() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        assert_eq!(
            run_partial_avg(),
            vec![("a".to_string(), 3.0, 2), ("b".to_string(), 7.0, 2)]
        );
    }

    /// A `name, n BIGINT` parquet whose values are large enough that a 64-bit sum could wrap.
    fn write_big_values_parquet(path: &Path) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("n", DataType::Int64, true),
        ]));
        let names: ArrayRef = Arc::new(StringArray::from(vec!["a", "a", "b", "b"]));
        let values: ArrayRef = Arc::new(Int64Array::from(vec![i64::MAX / 2; 4]));
        let batch = RecordBatch::try_new(schema.clone(), vec![names, values]).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Builds `sum(n) GROUP BY name` over an int64 column — the sum DuckDB models as HUGEINT
    /// but the engine runs in an int64 accumulator (the HUGEINT-to-BIGINT downcast path).
    fn int64_sum_plan(path: &str) -> Vec<u8> {
        use prost::Message as _;
        use substrait::proto::{Plan, rel};

        let mut plan = Plan::decode(partial_avg_plan(path).as_slice()).unwrap();
        let root = match plan.relations[0].rel_type.as_mut().unwrap() {
            substrait::proto::plan_rel::RelType::Root(root) => root,
            other => panic!("expected a root relation, got {other:?}"),
        };
        let rel::RelType::Aggregate(aggregate) =
            root.input.as_mut().unwrap().rel_type.as_mut().unwrap()
        else {
            panic!("expected an aggregate relation");
        };
        // Keep only the count measure and repoint it at the arithmetic sum: sum($1) AS BIGINT.
        aggregate.measures.remove(0);
        aggregate.measures[0]
            .measure
            .as_mut()
            .unwrap()
            .function_reference = 1;
        root.names = ["name", "s"].into_iter().map(str::to_string).collect();
        plan.encode_to_vec()
    }

    /// A 64-bit integer sum whose values could wrap the int64 accumulator fails loudly: cuDF
    /// has no INT128 to widen into, so the engine refuses instead of wrapping silently.
    /// Requires a GPU.
    #[test]
    fn int64_sum_that_could_overflow_fails_loudly() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let message = under_watchdog("overflowing int64 sum".to_string(), move || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("big.parquet");
            write_big_values_parquet(&path);
            let plan = int64_sum_plan(path.to_str().unwrap());
            let ctx = SiriusContext::new().expect("bring up sirius context");
            match ctx.execute_substrait_result(&plan) {
                Ok(_) => Err("a sum that could overflow int64 was allowed to run".to_string()),
                Err(err) => Ok(err.to_string()),
            }
        });
        assert!(message.contains("could overflow int64"), "{message}");
    }

    /// Like [`stream_read_plan`] but declaring `id` as FP64 — for the schema-mismatch
    /// negatives, whose receiver deliberately declares a type the sender does not produce.
    fn stream_read_plan_f64(stream_id: u64) -> Vec<u8> {
        use substrait::proto::read_rel::{NamedTable, ReadType};
        use substrait::proto::{
            NamedStruct, Plan, PlanRel, ReadRel, Rel, RelRoot, Type, plan_rel, rel, r#type,
        };

        let names = vec!["id".to_string(), "name".to_string()];
        let types = vec![
            Type {
                kind: Some(r#type::Kind::Fp64(r#type::Fp64 {
                    type_variation_reference: 0,
                    nullability: r#type::Nullability::Nullable as i32,
                })),
            },
            Type {
                kind: Some(r#type::Kind::String(r#type::String {
                    type_variation_reference: 0,
                    nullability: r#type::Nullability::Nullable as i32,
                })),
            },
        ];
        let read = Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                base_schema: Some(NamedStruct {
                    names: names.clone(),
                    r#struct: Some(r#type::Struct {
                        types,
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Required as i32,
                    }),
                }),
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: vec![super::stream_view_name(stream_id)],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(read),
                    names,
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    /// The declared input schema is what the receiver's plan binds against; a source whose sink
    /// produces different column types must fail at the hop, before any batch moves, instead of
    /// having its columns reinterpreted downstream. Requires a GPU.
    #[test]
    fn relay_from_rejects_a_mismatched_schema() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let sender_plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );

        let ctx = SiriusContext::new().expect("bring up sirius context");
        let mut sender = ctx.fragment().unwrap();
        sender.declare_output(0).unwrap();
        // Before build() there is no bound sink to describe — loudly, not as an empty vec.
        assert!(sender.output_types().is_err());
        sender.build(&sender_plan).unwrap();
        // What the guard below compares against: the sink's actual column types, by name.
        assert_eq!(sender.output_types().unwrap(), vec!["BIGINT", "VARCHAR"]);
        sender.run().unwrap();

        // The sender's sink produces (BIGINT, VARCHAR); the receiver declares id as DOUBLE.
        let mut receiver = ctx.fragment().unwrap();
        receiver.declare_input_column(0, "id", "DOUBLE").unwrap();
        receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
        receiver.build(&stream_read_plan_f64(0)).unwrap();

        let err = receiver.relay_from(&mut sender, 0, 0, 0).unwrap_err();
        let what = err.what().to_string();
        assert!(what.contains("column 0"), "unexpected error: {what}");
        assert!(
            what.contains("DOUBLE") && what.contains("BIGINT"),
            "the error must name the declared and the produced type: {what}"
        );
    }

    /// The packed leg of the same guard: a packed batch whose unpacked cudf types disagree with
    /// the receiver's declared stream schema must be refused by `push_packed`. Requires a GPU.
    #[test]
    fn push_packed_rejects_a_mismatched_schema() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        // SAFETY: the GPU lock is held, so no other thread touches the environment here.
        unsafe { std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "64MiB") };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let sender_plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );

        let ctx = SiriusContext::new().expect("bring up sirius context");
        let mut sender = ctx.fragment().unwrap();
        sender.declare_output(0).unwrap();
        sender.build(&sender_plan).unwrap();
        sender.run().unwrap();

        let batch = sender
            .export_packed(0)
            .unwrap()
            .expect("the sender parked a batch to export");

        let mut receiver = ctx.fragment().unwrap();
        receiver.declare_input_column(0, "id", "DOUBLE").unwrap();
        receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
        receiver.build(&stream_read_plan_f64(0)).unwrap();

        let err = receiver.push_packed(0, &batch).unwrap_err();
        let what = err.what().to_string();
        assert!(what.contains("column 0"), "unexpected error: {what}");
        assert!(
            what.contains("declared DOUBLE"),
            "the error must name the declared type: {what}"
        );

        ctx.staging_release(batch.offset).unwrap();
        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
    }
}
