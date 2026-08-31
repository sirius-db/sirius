//! Safe, idiomatic Rust bindings for [Sirius](https://github.com/sirius-db/sirius),
//! the GPU-native SQL engine.
//!
//! This crate wraps the low-level [`sirius-sys`] cxx bindings in safe Rust types
//! — the entry point for driving Sirius from Rust.
//!
//! Two entry points:
//!
//! * [`SiriusContext`] — an initialized engine, constructed from defaults or a
//!   YAML config file, able to execute a whole Substrait plan in one call.
//! * [`Fragment`] — one plan fragment of a multi-fragment query, for driving a
//!   distributed plan a piece at a time and moving native GPU batches between
//!   the pieces without going through Arrow.

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
///
/// Neither `Send` nor `Sync` — drive one context from one thread. `!Send` comes from the cxx
/// opaque `Context` itself; the `RefCell` adds `!Sync`, and buys
/// [`fragment`](SiriusContext::fragment) a `&self` receiver so several fragments of one query
/// can be alive at once.
pub struct SiriusContext {
    // RAII handle owning the C++ engine context for its lifetime.
    //
    // Behind a `RefCell` because every call into C++ needs `Pin<&mut Context>` while a
    // [`Fragment`] only borrows the context immutably. Several fragments of one query are alive
    // at once — senders parked, waiting on their receiver — and a `&mut self` factory would allow
    // exactly one.
    //
    // The borrow never outlives the call that takes it, so it cannot be observed by user code and
    // the runtime check is unreachable without re-entering from a C++ callback. The context is
    // neither `Send` nor `Sync`, so there is no cross-thread aliasing either.
    //
    // What this costs: a `&self` factory gives up compile-time enforcement of "exactly one
    // fragment sits between its own build() and run()" — two can be mid-lifecycle at once and the
    // engine rejects that at runtime. The whole-plan path is unaffected: it keeps `&mut self`, so
    // running a standalone query while any fragment is alive stays a compile error.
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

    /// Create a [`Fragment`] on this context — one plan fragment of a multi-fragment query.
    ///
    /// Takes `&self`, not `&mut self`: a distributed plan keeps several fragments alive at once,
    /// with senders parked until their receiver relays from them, so an exclusive borrow would
    /// permit exactly one. The returned fragment borrows the context, so it cannot outlive it.
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
    ///
    /// Takes `&mut self` where [`fragment`](SiriusContext::fragment) takes `&self`: the whole-plan
    /// path has no reason to run while a fragment is alive, so the exclusive borrow turns that
    /// into a compile error rather than a runtime one.
    pub fn execute_substrait(&mut self, plan: &[u8]) -> Result<Vec<RecordBatch>, SiriusError> {
        Ok(self.execute_substrait_result(plan)?.batches)
    }

    /// Execute a serialized Substrait plan and retain its Arrow schema for empty results.
    pub fn execute_substrait_result(
        &mut self,
        plan: &[u8],
    ) -> Result<SubstraitResult, SiriusError> {
        // The engine writes a self-owning Arrow C Data Interface stream into
        // `stream`; the FFI takes its address as an integer (a `uintptr_t`).
        let_cxx_string!(plan = plan);
        drain_arrow(|out_stream_addr| {
            // SAFETY: `drain_arrow` passes the address of a live, writable
            // `FFI_ArrowArrayStream` it owns for the closure's duration.
            unsafe {
                self.inner
                    .borrow_mut()
                    .pin_mut()
                    .execute_substrait(&plan, out_stream_addr)
            }
        })
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
    /// (copy-out-on-arrival); a released block goes back to the arena's free list and coalesces
    /// with its free neighbours.
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

/// Name of the DuckDB view a plan reads to consume input stream `stream_id`.
///
/// The single definition of the convention: a front end emitting the read and the engine creating
/// the view resolve the same name, so they cannot drift apart.
pub fn stream_view_name(stream_id: u64) -> String {
    sirius_sys::stream_view_name(stream_id)
        .to_string_lossy()
        .into_owned()
}

/// One plan fragment of a multi-fragment query.
///
/// A fragment either declares output streams — an *intermediate* fragment, whose results park as
/// native GPU batches that outlive its own query — or declares none, making it a *result*
/// fragment that produces Arrow.
///
/// The usage order is fixed: declare inputs and outputs, [`build`](Fragment::build) the plan,
/// [`relay_from`](Fragment::relay_from) every sender, [`run`](Fragment::run), then drain via
/// `relay_from` (from downstream) or [`result_to_arrow`](Fragment::result_to_arrow). Exactly
/// one fragment may sit between its own `build` and `run` — the engine serializes queries.
///
/// Borrows the [`SiriusContext`] that created it, so a fragment cannot outlive its engine.
pub struct Fragment<'ctx> {
    inner: UniquePtr<sirius_sys::Fragment>,
    /// Ties the fragment's lifetime to the context that made it.
    _context: PhantomData<&'ctx SiriusContext>,
}

impl Fragment<'_> {
    /// Declare one column of input stream `stream_id`, in plan order. `ty` is a DuckDB type name
    /// (`BIGINT`, `DECIMAL(15,2)`, `DATE`, …) — a stream has no file to probe, so the schema is
    /// given rather than inferred.
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

    /// Declare an output stream. A fragment with no output stream is a result fragment.
    pub fn declare_output(&mut self, stream_id: u64) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output(stream_id)
    }

    /// Replicate every batch to all declared outputs.
    ///
    /// Returns `Err` if a hash key was already declared (the two modes are mutually exclusive) or
    /// if the fragment is already built. Whether enough destinations exist to route between is not
    /// known until [`build`](Fragment::build), which is where that is rejected.
    pub fn declare_output_broadcast(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output_broadcast()
    }

    /// Hash-partition across the declared outputs on `column_index`. Call once per key column, in
    /// key order. Same exclusivity and ordering rules as
    /// [`declare_output_broadcast`](Fragment::declare_output_broadcast).
    pub fn declare_output_hash_key(&mut self, column_index: u32) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output_hash_key(column_index)
    }

    /// Plan `substrait_plan` against the declared streams and open the fragment's query lifecycle.
    ///
    /// The declaration-time rules that could not be checked earlier are checked here, so both
    /// surface as `Err` from this call rather than from the `declare_*` that caused them:
    ///
    /// * a routing mode on a fragment with fewer than two outputs — every row would reach
    ///   destination 0 regardless, so silently dropping the spec would look like it worked;
    /// * a plan that reads the same declared input stream id more than once (a self-join over one
    ///   stream, say) — the second read would overwrite the first's registration and strand its
    ///   pipeline waiting for a push that never arrives.
    pub fn build(&mut self, substrait_plan: &[u8]) -> Result<(), Exception> {
        let_cxx_string!(plan = substrait_plan);
        self.inner.pin_mut().build(&plan)
    }

    /// Move every batch parked on `source`'s output stream into this fragment's input stream — as
    /// native GPU batch handles, with no Arrow and no file in between — then close `sender_id`.
    /// Returns the number of batches moved.
    ///
    /// Returns `Err` unless `source` has already finished [`run`](Fragment::run): before that, an
    /// empty stream is indistinguishable from a finished one, and the input would be closed after
    /// zero batches, silently truncating the result. Also `Err` if `input_stream_id` was never
    /// declared here, or if `source` is a result fragment and so has nothing to relay.
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
        let metadata = self
            .inner
            .pin_mut()
            .export_packed(stream_id, &mut offset, &mut len)?;
        if metadata.is_null() {
            return Ok(None);
        }
        Ok(Some(PackedBatch {
            metadata: metadata.as_slice().to_vec(),
            offset,
            len,
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

    /// Close `sender_id` on input stream `stream_id`.
    ///
    /// The end-of-stream mirror for senders that are not local fragments — [`relay_from`] closes
    /// its own sender. Idempotent per sender.
    ///
    /// [`relay_from`]: Fragment::relay_from
    pub fn close_input(&mut self, stream_id: u64, sender_id: u32) -> Result<(), Exception> {
        self.inner.pin_mut().close_input(stream_id, sender_id)
    }

    /// Execute the fragment and close its query lifecycle. Blocks until its pipelines finish.
    pub fn run(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().run()
    }

    /// Batches currently parked on output stream `stream_id` — the evidence that a fragment
    /// boundary carried native batches rather than nothing.
    pub fn output_batch_count(&self, stream_id: u64) -> Result<usize, Exception> {
        self.inner.output_batch_count(stream_id)
    }

    /// DuckDB type names of this fragment's output columns, in plan order.
    ///
    /// Available after [`build`](Fragment::build). Returns `Err` for a result fragment, which has
    /// no output streams to describe.
    pub fn output_types(&self) -> Result<Vec<String>, Exception> {
        Ok(self
            .inner
            .output_types()?
            .iter()
            .map(|ty| ty.to_string_lossy().into_owned())
            .collect())
    }

    /// Collect a result fragment's rows over the Arrow C Data Interface.
    ///
    /// Named for the C++ verb it binds rather than `into_*`, which Rust reserves for by-value
    /// conversions — this borrows and could in principle be called more than once.
    ///
    /// Returns `Err` on a fragment that declared output streams (it parks native batches for a
    /// peer instead of producing Arrow) or before [`run`](Fragment::run).
    pub fn result_to_arrow(&mut self) -> Result<SubstraitResult, SiriusError> {
        drain_arrow(|out_stream_addr| {
            // SAFETY: `drain_arrow` passes the address of a live, writable
            // `FFI_ArrowArrayStream` it owns for the closure's duration.
            unsafe { self.inner.pin_mut().result_to_arrow(out_stream_addr) }
        })
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
    /// the only way a caller outside C++ can observe one.
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

/// Run `emit` against a freshly-created Arrow C Data Interface stream, then drain it fully.
///
/// Both Arrow-producing entry points — whole-plan execution and a result fragment — hand C++ the
/// address of a stack-owned `FFI_ArrowArrayStream` and then collect it. Keeping that sequence in
/// one place keeps the two from drifting: it is the only spot where a raw address crosses the FFI
/// boundary, and the draining must happen while the producer is still alive, because the
/// conversion dereferences the engine context.
fn drain_arrow(
    emit: impl FnOnce(usize) -> Result<(), Exception>,
) -> Result<SubstraitResult, SiriusError> {
    let mut stream = FFI_ArrowArrayStream::empty();
    let out_stream_addr = std::ptr::addr_of_mut!(stream) as usize;
    emit(out_stream_addr).map_err(SiriusError::Engine)?;
    let reader = ArrowArrayStreamReader::try_new(stream).map_err(SiriusError::Arrow)?;
    let schema = reader.schema();
    let batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(SiriusError::Arrow)?;
    Ok(SubstraitResult { schema, batches })
}

/// Error returned by the Arrow-producing entry points: [`SiriusContext::execute_substrait`],
/// [`SiriusContext::execute_substrait_result`], and [`Fragment::result_to_arrow`].
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

    use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
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
            .set_max_row_group_row_count(Some(rows_per_group))
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

        // Write a tiny parquet fixture.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let ids: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        let names: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, names]).unwrap();
        {
            let file = std::fs::File::create(&path).unwrap();
            let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );

        // Execute twice on one context to verify standalone query state is reset,
        // then drop it before inspecting the returned owned results.
        let results = {
            let mut ctx = SiriusContext::new().expect("bring up sirius context");
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

    /// The view-name convention is shared with the engine, so a front end can emit a read for a
    /// stream it has declared. Pure string formatting — no GPU, no context.
    #[test]
    fn stream_view_name_matches_the_engine_convention() {
        assert_eq!(super::stream_view_name(0), "sirius_stream_0");
        assert_eq!(super::stream_view_name(42), "sirius_stream_42");
        assert_eq!(
            super::stream_view_name(u64::MAX),
            format!("sirius_stream_{}", u64::MAX)
        );
    }

    /// The two routing modes are mutually exclusive, and that *is* enforced at declaration time —
    /// unlike the "needs at least two destinations" rule, which cannot be known until `build()`
    /// and is asserted there, not here. Requires a GPU only for context bring-up.
    #[test]
    fn routing_modes_are_mutually_exclusive() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let ctx = SiriusContext::new().expect("bring up sirius context");

        let mut broadcast_first = ctx.fragment().expect("create fragment");
        broadcast_first
            .declare_output_broadcast()
            .expect("broadcast alone is accepted at declare time");
        assert!(
            broadcast_first.declare_output_hash_key(0).is_err(),
            "a hash key after broadcast must be rejected"
        );

        let mut hash_first = ctx.fragment().expect("create fragment");
        hash_first
            .declare_output_hash_key(0)
            .expect("a hash key alone is accepted at declare time");
        assert!(
            hash_first.declare_output_broadcast().is_err(),
            "broadcast after a hash key must be rejected"
        );
    }

    /// Several fragments are alive at once in a distributed plan — senders parked until their
    /// receiver relays from them. This is why `fragment()` takes `&self`; with a `&mut self`
    /// factory the second `ctx.fragment()` below would not borrow-check.
    #[test]
    fn context_makes_several_fragments_at_once() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let ctx = SiriusContext::new().expect("bring up sirius context");

        let mut sender = ctx.fragment().expect("create sender fragment");
        let mut receiver = ctx.fragment().expect("create receiver fragment");

        // Both are live here — the point of the test.
        sender.declare_output(7).expect("declare output");
        receiver
            .declare_input_column(7, "id", "BIGINT")
            .expect("declare input column");

        // Neither has been built, so this trips the build-ordering guard. It does NOT reach the
        // "source must have run" guard — that one needs two built fragments and a real plan, so
        // it is covered by the C++ suite rather than pretended at here.
        assert!(
            receiver.relay_from(&mut sender, 7, 7, 0).is_err(),
            "relay_from before build() must be rejected"
        );
    }

    /// Flattens a result's batches into sorted `(id, name)` rows, so two scans can be
    /// compared as sets regardless of how the engine chunked them into batches.
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

        let mut ctx = SiriusContext::new().expect("bring up sirius context");
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

    /// Writes the 3-row `(id BIGINT, name VARCHAR)` users fixture.
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
                .result_to_arrow()
                .map(|result| rows(&result))
                .map_err(|err| err.to_string())
        });
        assert!(rows.is_empty(), "an empty stream yields no rows: {rows:?}");
    }

    /// The decisive equivalence for the packed FFI pair: a fragment hop carried by the staging
    /// arena (`export_packed` → transmit-from-lease → `push_packed`) must deliver exactly the
    /// values the proven in-process `relay_from` hop delivers for the identical plan pair.
    /// Also pins the surrounding contracts: drained-stream export is `None`, push after EOS is
    /// a loud error, and leases release (with the free list coalescing back to one whole-arena
    /// block) before the receiver runs — which only works if push really copied the data out of
    /// the lease. Requires a GPU.
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

        // Declares `(id, name)` on input stream 0 and builds the stream-view read.
        let make_receiver = || {
            let mut receiver = ctx.fragment().unwrap();
            receiver.declare_input_column(0, "id", "BIGINT").unwrap();
            receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
            receiver.build(&receiver_plan).unwrap();
            receiver
        };

        // Reference: the proven in-process native relay.
        let relay_result = {
            let mut sender = ctx.fragment().unwrap();
            sender.declare_output(0).unwrap();
            sender.build(&sender_plan).unwrap();
            sender.run().unwrap();

            let mut receiver = make_receiver();
            let moved = receiver.relay_from(&mut sender, 0, 0, 0).unwrap();
            assert!(moved > 0, "the relay hop must carry batches");
            receiver.run().unwrap();
            receiver.result_to_arrow().unwrap()
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
                staged.push(batch);
            }
            assert!(!staged.is_empty(), "the sender parked batches to export");
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
            receiver.result_to_arrow().unwrap()
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
    /// through `push_packed` into an empty result. Pins the leak the demo hit on q15:
    /// `export_packed` used to lease `total + slack` even for `total == 0`, while the transport
    /// contract says `len == 0` means nothing to release — every empty cross-CN batch orphaned
    /// a >= 8 MiB lease that pinned staging space for the process lifetime. The zero-row batch
    /// comes from an empty byte-range split (a range inside one row group scans zero row groups
    /// and emits exactly one empty batch). Requires a GPU.
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
        // zero-row sender instance of a distributed scan.
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
            } else {
                ctx.staging_release(batch.offset).unwrap();
            }
        }

        // The reclaim guarantee: with only `len > 0` leases released (a metadata-only frame has
        // nothing to release), zero leases are outstanding, so the ENTIRE arena is grantable as
        // one lease. Under the leak this throws exhaustion — the orphaned slack lease still
        // pins its block.
        let probe = ctx.staging_lease(capacity).expect(
            "the whole arena must be grantable again after a zero-row export cycle \
             (an outstanding lease here is the orphaned-slack leak)",
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
        let result = receiver.result_to_arrow().unwrap();
        assert_eq!(rows(&result), Vec::new(), "an empty split carries no rows");

        // Keep the arena out of the other tests' context bring-ups.
        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
    }
}
