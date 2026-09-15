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
//!   distributed plan a piece at a time. Same-process hops use native GPU
//!   batches ([`Fragment::relay_from`]); a process edge hops packed GPU bytes
//!   ([`Fragment::export_packed`] / [`Fragment::push_packed`]).

use std::cell::RefCell;
use std::marker::PhantomData;
use std::path::Path;

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use cxx::{let_cxx_string, Exception, UniquePtr};

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
    /// Takes `&self`, not `&mut self`: a distributed plan keeps several fragments alive at once,
    /// with senders parked until their receiver relays from them, so an exclusive borrow would
    /// permit exactly one. The returned fragment borrows the context, so the compiler enforces
    /// the one lifetime rule the C++ side cannot: a fragment must not outlive the engine it runs
    /// on.
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
    /// without one, every staging call is an error rather than a silent host-copy path.
    /// Exhaustion is an error naming the requested/free/capacity byte counts.
    pub fn staging_lease(&self, len: u64) -> Result<u64, Exception> {
        self.inner.borrow_mut().pin_mut().staging_lease(len)
    }

    /// Return the staging lease at `offset`. The released block goes back to the arena's
    /// address-ordered free list and coalesces with its free neighbours, so the space is
    /// reusable regardless of release order.
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
/// to take with [`Fragment::relay_from`] or [`Fragment::export_packed`]. A fragment declaring
/// **none** is a result fragment and produces Arrow via [`Fragment::result_to_arrow`].
///
/// Calls are ordered: declare, [`build`](Fragment::build), fill every sender,
/// [`run`](Fragment::run), then drain. `build` opens a query lifecycle on the shared engine that
/// `run` closes, so one fragment at a time may sit between the two — dropping a built-but-unrun
/// fragment closes the lifecycle for you.
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

    /// Every declared output stream receives the full fragment output (a broadcast sink);
    /// output 0 keeps the original batches, the rest carry independent deep copies.
    ///
    /// Returns `Err` if a hash key was already declared (the two modes are mutually exclusive) or
    /// if the fragment is already built. Whether enough destinations exist to route between is not
    /// known until [`build`](Fragment::build), which is where that is rejected.
    pub fn declare_output_broadcast(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output_broadcast()
    }

    /// Declares one hash-partition key (an output column index): rows hash-route by the
    /// declared keys, output stream i taking partition i. Call once per key, in the exchange's
    /// shared partition-expression order. Same exclusivity and ordering rules as
    /// [`declare_output_broadcast`](Fragment::declare_output_broadcast).
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
    /// `[staging_base() + offset, + len)` immediately; the exporter's lease stays live until the
    /// caller hands it back with [`SiriusContext::staging_release`] after the transmit completes.
    ///
    /// A zero-row batch comes back metadata-only: `offset == 0` with `len == 0` means NO lease
    /// exists for it, and the caller must not release anything.
    ///
    /// In a same-process loopback the receiver's [`push_packed`](Fragment::push_packed) consumes
    /// that lease (this study path has no inbound ticket store). A remote hop still releases the
    /// exporter's lease after the write, on this process.
    pub fn export_packed(&mut self, stream_id: u64) -> Result<Option<PackedBatch>, Exception> {
        let mut offset = 0u64;
        let mut len = 0u64;
        let mut rows = 0u64;
        let metadata =
            self.inner
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
    /// The table is deep-copied out of the lease into ordinary pool memory before this returns.
    /// When `batch.len != 0`, this call also releases that receiver lease. Legal between
    /// [`build`](Fragment::build) and [`run`](Fragment::run), like
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
    /// ([`relay_from`](Fragment::relay_from) closes its own sender). Idempotent per sender; the
    /// stream ends once every expected sender has closed.
    pub fn close_input(&mut self, stream_id: u64, sender_id: u32) -> Result<(), Exception> {
        self.inner.pin_mut().close_input(stream_id, sender_id)
    }

    /// Execute the fragment and close its query lifecycle. Blocks until its pipelines finish.
    pub fn run(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().run()
    }

    /// Collect a result fragment's rows over the Arrow C Data Interface.
    ///
    /// Named for the C++ verb it binds rather than `into_*`, which Rust reserves for by-value
    /// conversions — this borrows and could in principle be called more than once.
    ///
    /// Returns `Err` on a fragment that declared output streams (it parks native batches for a
    /// peer instead of producing Arrow) or before [`run`](Fragment::run).
    pub fn result_to_arrow(&mut self) -> Result<SubstraitResult, SiriusError> {
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

    /// True when output stream `stream_id` has ended. False means "not done", including
    /// "nothing parked right now".
    pub fn drained(&mut self, stream_id: u64) -> Result<bool, Exception> {
        self.inner.pin_mut().drained(stream_id)
    }

    /// Batches currently parked on output stream `stream_id` — the evidence that a fragment
    /// boundary carried native batches rather than nothing.
    pub fn output_batch_count(&self, stream_id: u64) -> Result<usize, Exception> {
        self.inner.output_batch_count(stream_id)
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
}

/// One batch exported into the exchange staging arena as cudf packed bytes.
///
/// `metadata` is the host-side cudf pack metadata (it travels with the payload on the wire);
/// `offset`/`len` locate the packed device payload inside the staging arena. The exporter's
/// lease at `offset` stays outstanding until [`SiriusContext::staging_release`] — except for a
/// metadata-only zero-row batch (`offset == 0`, `len == 0`), which holds no lease and must not
/// be released. On this study path, [`Fragment::push_packed`] also releases a nonzero receiver
/// lease, so a same-process loopback must not release again after a successful push.
pub struct PackedBatch {
    /// cudf pack metadata bytes (host memory).
    pub metadata: Vec<u8>,
    /// Byte offset of the packed payload from the arena base.
    pub offset: u64,
    /// Length of the packed payload in bytes.
    pub len: u64,
    /// Exact row count of the packed table, filled by
    /// [`export_packed`](Fragment::export_packed). Ignored by
    /// [`push_packed`](Fragment::push_packed).
    pub rows: u64,
}

/// Thread-safe handle to a context's exchange staging arena, from
/// [`SiriusContext::staging_arena`].
///
/// This is what lets a transport thread serve `lease`/`release` while the context's owning
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
    /// the only way a compute node can observe one.
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
    use std::collections::HashMap;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use arrow_array::{ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use prost::Message;
    use substrait::proto::extensions::simple_extension_declaration;
    use substrait::proto::extensions::{SimpleExtensionDeclaration, SimpleExtensionUrn};
    use substrait::proto::read_rel::{NamedTable, ReadType};
    use substrait::proto::{
        aggregate_function, aggregate_rel, expression, function_argument, plan_rel, r#type, rel,
        AggregateFunction, AggregateRel, Expression, FunctionArgument, NamedStruct, Plan, PlanRel,
        ReadRel, Rel, RelRoot, Type,
    };

    use super::{stream_view_name, Fragment, SiriusContext, SubstraitResult};

    /// The engine keeps process-global GPU state, so at most one context may be
    /// live at a time; context-constructing tests hold this for their duration.
    static GPU_CONTEXT_LOCK: Mutex<()> = Mutex::new(());

    /// Arena size is read at context bring-up. Packed-hop tests set it while they
    /// hold [`GPU_CONTEXT_LOCK`] and restore the previous value on drop.
    struct StagingEnv {
        previous: Option<String>,
    }

    impl StagingEnv {
        fn install() -> Self {
            let previous = std::env::var("SIRIUS_EXCHANGE_STAGING_BYTES").ok();
            // SAFETY: the GPU context lock is held, so no other test mutates this env here.
            unsafe { std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "64MiB") };
            Self { previous }
        }
    }

    impl Drop for StagingEnv {
        fn drop(&mut self) {
            // SAFETY: same as [`StagingEnv::install`].
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", value),
                    None => std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES"),
                }
            }
        }
    }

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

    /// Encodes a single-file `local_files` parquet read plan with `names` as the
    /// root output names — the shape DuckDB's Substrait reader resolves to
    /// `parquet_scan(<path>)`.
    fn local_files_plan(path: &str, names: Vec<String>) -> Vec<u8> {
        Plan {
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(local_files_read(path)),
                    names,
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    fn local_files_read(path: &str) -> Rel {
        use substrait::proto::read_rel::local_files::file_or_files::{
            FileFormat, ParquetReadOptions, PathType,
        };
        use substrait::proto::read_rel::local_files::FileOrFiles;
        use substrait::proto::read_rel::LocalFiles;

        Rel {
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
        }
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

    /// Writes a `(id BIGINT, name VARCHAR)` parquet with `rows` consecutive ids.
    fn write_id_parquet(path: &Path, rows: i64) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let ids: ArrayRef = Arc::new(Int64Array::from((0..rows).collect::<Vec<_>>()));
        let names: ArrayRef = Arc::new(StringArray::from(
            (0..rows).map(|i| format!("n{i}")).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(schema.clone(), vec![ids, names]).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    /// Writes a `(region VARCHAR, amount BIGINT)` parquet from the given rows.
    fn write_sales_parquet(path: &Path, rows: &[(&str, i64)]) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Int64, false),
        ]));
        let regions: ArrayRef = Arc::new(StringArray::from(
            rows.iter().map(|(region, _)| *region).collect::<Vec<_>>(),
        ));
        let amounts: ArrayRef = Arc::new(Int64Array::from(
            rows.iter().map(|(_, amount)| *amount).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(schema.clone(), vec![regions, amounts]).unwrap();
        let file = std::fs::File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    fn field_ref(field: i32) -> Expression {
        Expression {
            rex_type: Some(expression::RexType::Selection(Box::new(
                expression::FieldReference {
                    reference_type: Some(
                        expression::field_reference::ReferenceType::DirectReference(
                            expression::ReferenceSegment {
                                reference_type: Some(
                                    expression::reference_segment::ReferenceType::StructField(
                                        Box::new(expression::reference_segment::StructField {
                                            field,
                                            child: None,
                                        }),
                                    ),
                                ),
                            },
                        ),
                    ),
                    root_type: Some(expression::field_reference::RootType::RootReference(
                        expression::field_reference::RootReference {},
                    )),
                },
            ))),
        }
    }

    fn nullable_i64() -> Type {
        Type {
            kind: Some(r#type::Kind::I64(r#type::I64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        }
    }

    fn nullable_string() -> Type {
        Type {
            kind: Some(r#type::Kind::String(r#type::String {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        }
    }

    fn nullable_fp64() -> Type {
        Type {
            kind: Some(r#type::Kind::Fp64(r#type::Fp64 {
                type_variation_reference: 0,
                nullability: r#type::Nullability::Nullable as i32,
            })),
        }
    }

    fn sum_extensions() -> (Vec<SimpleExtensionUrn>, Vec<SimpleExtensionDeclaration>) {
        (
            vec![SimpleExtensionUrn {
                extension_urn_anchor: 1,
                urn: "extension:io.substrait:functions_arithmetic".to_string(),
            }],
            vec![SimpleExtensionDeclaration {
                mapping_type: Some(
                    simple_extension_declaration::MappingType::ExtensionFunction(
                        simple_extension_declaration::ExtensionFunction {
                            extension_urn_reference: 1,
                            function_anchor: 1,
                            name: "sum".to_string(),
                        },
                    ),
                ),
            }],
        )
    }

    fn aggregate_sum_by_key(input: Rel, key_field: i32, sum_field: i32) -> Rel {
        #[allow(deprecated)]
        let grouping = aggregate_rel::Grouping {
            grouping_expressions: Vec::new(),
            expression_references: vec![0],
        };
        Rel {
            rel_type: Some(rel::RelType::Aggregate(Box::new(AggregateRel {
                input: Some(Box::new(input)),
                grouping_expressions: vec![field_ref(key_field)],
                groupings: vec![grouping],
                measures: vec![aggregate_rel::Measure {
                    measure: Some(AggregateFunction {
                        function_reference: 1,
                        arguments: vec![FunctionArgument {
                            arg_type: Some(function_argument::ArgType::Value(field_ref(sum_field))),
                        }],
                        output_type: Some(nullable_i64()),
                        invocation: aggregate_function::AggregationInvocation::All as i32,
                        ..Default::default()
                    }),
                    filter: None,
                }],
                ..Default::default()
            }))),
        }
    }

    fn encode_plan(input: Rel, names: Vec<String>, with_sum: bool) -> Vec<u8> {
        let (extension_urns, extensions) = if with_sum {
            sum_extensions()
        } else {
            (Vec::new(), Vec::new())
        };
        Plan {
            extension_urns,
            extensions,
            relations: vec![PlanRel {
                rel_type: Some(plan_rel::RelType::Root(RelRoot {
                    input: Some(input),
                    names,
                })),
            }],
            ..Default::default()
        }
        .encode_to_vec()
    }

    fn named_table_read(stream_id: u64, names: Vec<String>, types: Vec<Type>) -> Rel {
        Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                base_schema: Some(NamedStruct {
                    names,
                    r#struct: Some(r#type::Struct {
                        types,
                        type_variation_reference: 0,
                        nullability: r#type::Nullability::Required as i32,
                    }),
                }),
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: vec![stream_view_name(stream_id)],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        }
    }

    /// A plan whose only read is the engine's stream view for input stream `stream_id`,
    /// projecting the users fixture's `(id BIGINT, name VARCHAR)` schema.
    fn stream_read_plan(stream_id: u64) -> Vec<u8> {
        let names = vec!["id".to_string(), "name".to_string()];
        encode_plan(
            named_table_read(
                stream_id,
                names.clone(),
                vec![nullable_i64(), nullable_string()],
            ),
            names,
            false,
        )
    }

    /// Like [`stream_read_plan`] but declaring `id` as FP64 — for schema-mismatch negatives.
    fn stream_read_plan_f64(stream_id: u64) -> Vec<u8> {
        let names = vec!["id".to_string(), "name".to_string()];
        encode_plan(
            named_table_read(
                stream_id,
                names.clone(),
                vec![nullable_fp64(), nullable_string()],
            ),
            names,
            false,
        )
    }

    fn sales_stream_read(stream_id: u64) -> Rel {
        named_table_read(
            stream_id,
            vec!["region".to_string(), "amount".to_string()],
            vec![nullable_string(), nullable_i64()],
        )
    }

    fn sales_names() -> Vec<String> {
        vec!["region".to_string(), "amount".to_string()]
    }

    fn local_files_groupby_plan(path: &str) -> Vec<u8> {
        encode_plan(
            aggregate_sum_by_key(local_files_read(path), 0, 1),
            sales_names(),
            true,
        )
    }

    fn stream_groupby_plan(stream_id: u64) -> Vec<u8> {
        encode_plan(
            aggregate_sum_by_key(sales_stream_read(stream_id), 0, 1),
            sales_names(),
            true,
        )
    }

    /// Flattens a result into sorted `(id, name)` rows, so comparisons are by value and
    /// independent of batch boundaries.
    fn rows(result: &SubstraitResult) -> Vec<(i64, String)> {
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

    fn i64_values(array: &ArrayRef) -> Vec<i64> {
        if let Some(ints) = array.as_any().downcast_ref::<Int64Array>() {
            return (0..ints.len()).map(|i| ints.value(i)).collect();
        }
        // DuckDB's Substrait `sum` over BIGINT binds DECIMAL(38, 0); the GPU SQL
        // path in FRAG-6 stays BIGINT. Accept both so the same totals compare.
        if let Some(decs) = array.as_any().downcast_ref::<Decimal128Array>() {
            return (0..decs.len())
                .map(|i| {
                    i64::try_from(decs.value(i))
                        .unwrap_or_else(|_| panic!("decimal {} does not fit i64", decs.value(i)))
                })
                .collect();
        }
        panic!("expected Int64 or Decimal128, got {:?}", array.data_type());
    }

    fn group_rows(result: &SubstraitResult) -> Vec<(String, i64)> {
        let mut rows = Vec::new();
        for batch in &result.batches {
            let regions = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("region column");
            for (i, amount) in i64_values(batch.column(1)).into_iter().enumerate() {
                rows.push((regions.value(i).to_string(), amount));
            }
        }
        rows.sort();
        rows
    }

    fn hop_packed(
        sender: &mut Fragment<'_>,
        receiver: &mut Fragment<'_>,
        source_stream: u64,
        input_stream: u64,
        sender_id: u32,
    ) -> usize {
        let mut moved = 0usize;
        while let Some(batch) = sender.export_packed(source_stream).unwrap() {
            receiver.push_packed(input_stream, &batch).unwrap();
            moved += 1;
        }
        assert!(
            sender.drained(source_stream).unwrap(),
            "export_packed must exhaust a finished sender"
        );
        receiver.close_input(input_stream, sender_id).unwrap();
        moved
    }

    fn declare_sales_input(root: &mut Fragment<'_>, stream_id: u64, senders: &[u32]) {
        root.declare_input_column(stream_id, "region", "VARCHAR")
            .unwrap();
        root.declare_input_column(stream_id, "amount", "BIGINT")
            .unwrap();
        for sender in senders {
            root.declare_input_sender(stream_id, *sender).unwrap();
        }
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

    /// The view-name convention is shared with the engine, so a front end can emit a read for a
    /// stream it has declared. Pure string formatting — no GPU, no context.
    #[test]
    fn stream_view_name_matches_the_engine_convention() {
        assert_eq!(stream_view_name(0), "sirius_stream_0");
        assert_eq!(stream_view_name(42), "sirius_stream_42");
        assert_eq!(
            stream_view_name(u64::MAX),
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

        // Neither has been built, so this trips the build-ordering guard.
        assert!(
            receiver.relay_from(&mut sender, 7, 7, 0).is_err(),
            "relay_from before build() must be rejected"
        );
    }

    /// A parquet scan fragment feeds a stream-read fragment through native `relay_from`.
    /// Requires a GPU.
    #[test]
    fn fragment_relays_parquet_scan() {
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
        assert!(sender.output_types().is_err());
        sender.build(&sender_plan).unwrap();
        assert_eq!(sender.output_types().unwrap(), vec!["BIGINT", "VARCHAR"]);
        sender.run().unwrap();
        assert!(sender.output_batch_count(0).unwrap() > 0);

        let mut receiver = ctx.fragment().unwrap();
        receiver.declare_input_column(0, "id", "BIGINT").unwrap();
        receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
        receiver.build(&stream_read_plan(0)).unwrap();
        let moved = receiver.relay_from(&mut sender, 0, 0, 0).unwrap();
        assert!(moved > 0, "the relay hop must carry batches");
        receiver.run().unwrap();
        assert_eq!(
            rows(&receiver.result_to_arrow().unwrap()),
            vec![
                (1, "a".to_string()),
                (2, "b".to_string()),
                (3, "c".to_string()),
            ]
        );
    }

    /// The packed process-edge hop (`export_packed` → `push_packed` → `close_input`) must deliver
    /// the same rows as native `relay_from` for the identical plan pair. Requires a GPU.
    #[test]
    fn packed_hop_matches_relay_hop() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let _staging = StagingEnv::install();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let sender_plan = local_files_plan(
            path.to_str().unwrap(),
            vec!["id".to_string(), "name".to_string()],
        );
        let receiver_plan = stream_read_plan(0);

        let ctx = SiriusContext::new().expect("bring up sirius context");
        let arena = ctx.staging_arena().expect("arena configured");
        assert_eq!(ctx.staging_capacity().unwrap(), 64 << 20);
        assert_ne!(ctx.staging_base().unwrap(), 0);

        let relay_result = {
            let mut sender = ctx.fragment().unwrap();
            sender.declare_output(0).unwrap();
            sender.build(&sender_plan).unwrap();
            sender.run().unwrap();

            let mut receiver = ctx.fragment().unwrap();
            receiver.declare_input_column(0, "id", "BIGINT").unwrap();
            receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
            receiver.build(&receiver_plan).unwrap();
            let moved = receiver.relay_from(&mut sender, 0, 0, 0).unwrap();
            assert!(moved > 0, "the relay hop must carry batches");
            receiver.run().unwrap();
            receiver.result_to_arrow().unwrap()
        };

        let packed_result = {
            let mut sender = ctx.fragment().unwrap();
            sender.declare_output(0).unwrap();
            sender.build(&sender_plan).unwrap();
            sender.run().unwrap();

            let mut receiver = ctx.fragment().unwrap();
            receiver.declare_input_column(0, "id", "BIGINT").unwrap();
            receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
            receiver.build(&receiver_plan).unwrap();
            let moved = hop_packed(&mut sender, &mut receiver, 0, 0, 0);
            assert!(moved > 0, "the packed hop must carry batches");
            assert!(
                sender.export_packed(0).unwrap().is_none(),
                "a drained stream yields no extra batch"
            );
            assert_eq!(
                arena.outstanding().unwrap(),
                0,
                "push_packed consumes the same-process lease"
            );
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
    }

    /// A broadcast sender's two output streams each deliver the FULL result to their own
    /// receiver. Requires a GPU.
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
            let result = receiver.result_to_arrow().unwrap();
            assert_eq!(rows(&result), expected, "stream {output_stream}");
        }
    }

    /// Hash-partitioned fan-out: two output streams partition the rows by key — disjoint,
    /// union == whole. Requires a GPU.
    #[test]
    fn hash_partitioned_fragment_routes_keys_disjointly() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.parquet");
        write_id_parquet(&path, 64);

        let ctx = SiriusContext::new().expect("bring up sirius context");
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

        let mut map = HashMap::new();
        let mut total = 0usize;
        for stream in [0u64, 1u64] {
            let mut receiver = ctx.fragment().unwrap();
            receiver.declare_input_column(0, "id", "BIGINT").unwrap();
            receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
            receiver.build(&stream_read_plan(0)).unwrap();
            receiver.relay_from(&mut sender, stream, 0, 0).unwrap();
            receiver.run().unwrap();
            let partition = rows(&receiver.result_to_arrow().unwrap());
            assert!(!partition.is_empty(), "stream {stream} owns no keys");
            total += partition.len();
            for (id, _) in partition {
                assert!(map.insert(id, stream).is_none(), "key {id} in two streams");
            }
        }
        assert_eq!(total, 64, "partitions must union to the whole input");
    }

    /// Two-shard GROUP BY: partial SUM on each parquet leaf, hash N=2 on the key, merge SUM
    /// on each destination through the packed hop. Same numbers as the C++ FRAG-6 fixture.
    /// Requires a GPU.
    #[test]
    fn two_shard_group_by_over_packed_hop() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let _staging = StagingEnv::install();
        let dir = tempfile::tempdir().unwrap();
        let sales_0 = dir.path().join("sales_0.parquet");
        let sales_1 = dir.path().join("sales_1.parquet");
        write_sales_parquet(
            &sales_0,
            &[
                ("east", 10),
                ("west", 20),
                ("east", 5),
                ("north", 3),
                ("south", 8),
            ],
        );
        write_sales_parquet(
            &sales_1,
            &[
                ("west", 7),
                ("east", 3),
                ("north", 11),
                ("south", 1),
                ("midwest", 4),
            ],
        );

        let expected = vec![
            ("east".to_string(), 18),
            ("midwest".to_string(), 4),
            ("north".to_string(), 14),
            ("south".to_string(), 9),
            ("west".to_string(), 27),
        ];

        let ctx = SiriusContext::new().expect("bring up sirius context");

        let run_leaf = |path: &Path| {
            let mut leaf = ctx.fragment().unwrap();
            leaf.declare_output(0).unwrap();
            leaf.declare_output(1).unwrap();
            leaf.declare_output_hash_key(0).unwrap();
            leaf.build(&local_files_groupby_plan(path.to_str().unwrap()))
                .unwrap();
            leaf.run().unwrap();
            leaf
        };
        let mut leaf0 = run_leaf(&sales_0);
        let mut leaf1 = run_leaf(&sales_1);
        assert_eq!(leaf0.output_types().unwrap(), vec!["VARCHAR", "BIGINT"]);
        assert_eq!(leaf1.output_types().unwrap(), leaf0.output_types().unwrap());

        let mut got = Vec::new();
        for dest in [0u64, 1u64] {
            let mut root = ctx.fragment().unwrap();
            declare_sales_input(&mut root, 0, &[0, 1]);
            root.build(&stream_groupby_plan(0)).unwrap();
            hop_packed(&mut leaf0, &mut root, dest, 0, 0);
            hop_packed(&mut leaf1, &mut root, dest, 0, 1);
            root.run().unwrap();
            got.extend(group_rows(&root.result_to_arrow().unwrap()));
        }
        got.sort();
        assert_eq!(got, expected);
    }

    /// The declared input schema is what the receiver's plan binds against; a source whose sink
    /// produces different column types must fail at the hop, before any batch moves.
    /// Requires a GPU.
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
        sender.build(&sender_plan).unwrap();
        sender.run().unwrap();

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

    /// `push_packed` refuses a batch whose unpacked types disagree with the declared stream.
    /// Requires a GPU.
    #[test]
    fn push_packed_rejects_a_mismatched_schema() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let _staging = StagingEnv::install();
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
            .expect("sender parked a batch");

        let mut receiver = ctx.fragment().unwrap();
        receiver.declare_input_column(0, "id", "DOUBLE").unwrap();
        receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
        receiver.build(&stream_read_plan_f64(0)).unwrap();

        let err = receiver.push_packed(0, &batch).unwrap_err();
        let what = err.what().to_string();
        assert!(
            what.contains("DOUBLE") && (what.contains("BIGINT") || what.contains("int64")),
            "the error must name the declared and the produced type: {what}"
        );
        if batch.len != 0 {
            ctx.staging_release(batch.offset).unwrap();
        }
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
}
