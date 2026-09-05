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
    /// without one, every staging call is an error rather than a silent slow path. Exhaustion is
    /// an error naming the requested/free/capacity byte counts.
    pub fn staging_lease(&self, len: u64) -> Result<u64, Exception> {
        self.inner.borrow_mut().pin_mut().staging_lease(len)
    }

    /// Return the staging lease at `offset`. Leases are short-lived by design
    /// (copy-out-on-arrival); the released block goes back to the arena's address-ordered free
    /// list and coalesces with its free neighbours, so the space is reusable regardless of
    /// release order.
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

    /// Thread-safe handle to the inbound store, or `None` when no staging arena is configured
    /// (an inbound frame always arrives through the arena).
    ///
    /// A transport/RPC thread stages every arriving frame with [`InboundStore::stage`] and
    /// releases the arena lease at once; the receiver fragment takes the staged batch later
    /// with [`Fragment::push_inbound`]. Callable from any thread, concurrently with the context
    /// thread, like [`staging_arena`](Self::staging_arena).
    pub fn inbound_store(&self) -> Option<InboundStore> {
        let handle = self.inner.borrow().inbound_store_handle();
        (!handle.is_null()).then(|| InboundStore { inner: handle })
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
/// produces Arrow via [`Fragment::result_to_arrow`].
///
/// Calls are ordered: declare, [`build`](Fragment::build), relay from every sender,
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

    /// Batches currently parked on output stream `stream_id` — the evidence that a fragment
    /// boundary carried native batches rather than nothing.
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

    /// Publish independently owned access to one completed output stream. Create this on
    /// the fragment's owner thread after `run`; packing can then bypass that thread entirely.
    /// Exactly one destination may destructively consume a stream, by relay or export.
    pub fn export_provider(&mut self, stream_id: u64) -> Result<ExportProvider, Exception> {
        Ok(ExportProvider {
            inner: self.inner.pin_mut().export_provider(stream_id)?,
        })
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

    /// Move the batch staged under `ticket` (see [`InboundStore::stage`]) into input stream
    /// `stream_id`. No copy: the batch already lives in pool memory. Same schema guard and
    /// lifecycle rules as [`push_packed`](Fragment::push_packed); an unknown or already taken
    /// ticket is an error.
    pub fn push_inbound(&mut self, stream_id: u64, ticket: u64) -> Result<(), Exception> {
        self.inner.pin_mut().push_inbound(stream_id, ticket)
    }

    /// Record that `sender_id` finished producing into input stream `stream_id` — the EOS mirror
    /// of [`push_packed`](Fragment::push_packed) for remote senders
    /// ([`relay_from`](Fragment::relay_from) closes its own sender). Idempotent per sender; the
    /// stream ends once every expected sender has closed.
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

/// Buffer-only packing access to one completed output repository. The C++ provider keeps
/// its repository and packing state alive, serializes its stream, and rejects new work after
/// cancellation/context teardown. Neither `Fragment` nor `SiriusContext` crosses threads.
pub struct ExportProvider {
    inner: UniquePtr<sirius_sys::ExportProvider>,
}

// SAFETY: C++ owns the repository and a context-lifetime fence, with no borrowed Fragment,
// DuckDB connection, or session pointers. Packs serialize their own device stream; context
// destruction waits for active work and clears buffers before releasing the memory manager.
unsafe impl Send for ExportProvider {}
unsafe impl Sync for ExportProvider {}

impl ExportProvider {
    /// Pack one batch on the independent CUDA stream. On return the source read guard has
    /// completed, while the staging lease remains valid until the transport releases it.
    pub fn export_packed(&self) -> Result<Option<PackedBatch>, Exception> {
        let (mut offset, mut len, mut rows) = (0, 0, 0);
        let metadata = self.inner.export_packed(&mut offset, &mut len, &mut rows)?;
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

    /// Stop future claims without invalidating an active pack's buffers.
    pub fn cancel(&self) {
        self.inner.cancel();
    }
}

impl std::fmt::Debug for ExportProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExportProvider").finish_non_exhaustive()
    }
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

/// Thread-safe handle to a context's inbound store, from [`SiriusContext::inbound_store`].
///
/// Exchange frames are copied out of their staging-arena lease into ordinary pool memory the
/// moment they arrive ([`stage`](Self::stage)), so the lease goes back to the arena at once and
/// the arena only ever holds frames in flight. The staged batch waits under its ticket until the
/// receiver fragment takes it with [`Fragment::push_inbound`], or a failed query drops it with
/// [`drop`](Self::drop).
pub struct InboundStore {
    inner: UniquePtr<sirius_sys::InboundStore>,
}

// SAFETY: the C++ `InboundStore` is a `shared_ptr` to the context's store state. `stage` copies on
// the store's own non-blocking CUDA stream and synchronizes it before returning; allocations come
// from the pool's thread-safe resource; the ticket map sits behind the store's own mutex. There
// is no thread-affine state behind any operation, and the `shared_ptr` keeps the store alive
// independently of the `SiriusContext` (which closes it at teardown, so a late call errors).
unsafe impl Send for InboundStore {}
unsafe impl Sync for InboundStore {}

impl InboundStore {
    /// Reserve physical host payload capacity before accepting a receive frame.
    pub fn reserve(&self, length: u64) -> Result<u64, Exception> {
        self.inner.reserve(length)
    }

    /// Cancel a reservation that was never passed to `stage_reserved` (idempotent).
    pub fn cancel_reservation(&self, reservation: u64) -> Result<(), Exception> {
        self.inner.cancel_reservation(reservation)
    }

    /// Stage a frame with guaranteed copy-out capacity. The reservation is consumed even
    /// when the call fails, so callers must not reuse it for another frame.
    pub fn stage_reserved(&self, batch: &PackedBatch, reservation: u64) -> Result<u64, Exception> {
        // SAFETY: batch owns the metadata throughout the synchronous C++ call.
        unsafe {
            self.inner.stage_reserved(
                batch.metadata.as_ptr() as usize,
                batch.metadata.len(),
                batch.offset,
                batch.len,
                reservation,
            )
        }
    }

    /// Copy `batch`'s packed bytes out of the staging arena into pool memory; returns the ticket
    /// the receiver names the batch by. The lease at `batch.offset` is still the caller's to
    /// release, immediately after this returns. A metadata-only zero-row batch stages an empty
    /// table.
    pub fn stage(&self, batch: &PackedBatch) -> Result<u64, Exception> {
        // SAFETY: the metadata pointer/length name `batch.metadata`'s buffer, which this borrow
        // keeps alive and readable for the duration of the call.
        unsafe {
            self.inner.stage(
                batch.metadata.as_ptr() as usize,
                batch.metadata.len(),
                batch.offset,
                batch.len,
            )
        }
    }

    /// Drop the staged batch under `ticket`, freeing its pool memory: the release path for a
    /// frame whose receiver will never run. An unknown ticket is an error (double drop).
    pub fn drop(&self, ticket: u64) -> Result<(), Exception> {
        self.inner.drop_ticket(ticket)
    }

    /// Batches currently staged. Nonzero once every query has quiesced means a leak.
    pub fn outstanding(&self) -> Result<usize, Exception> {
        self.inner.outstanding()
    }

    /// Bytes currently staged.
    pub fn outstanding_bytes(&self) -> Result<u64, Exception> {
        self.inner.outstanding_bytes()
    }
}

impl std::fmt::Debug for InboundStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InboundStore")
            .field("outstanding", &self.outstanding().ok())
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
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use arrow_array::{Array, ArrayRef, Decimal128Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use prost::Message;

    use super::SiriusContext;

    /// The engine keeps process-global GPU state, so at most one context may be
    /// live at a time; context-constructing tests hold this for their duration.
    static GPU_CONTEXT_LOCK: Mutex<()> = Mutex::new(());

    /// Exchange fixtures need only small pools, including on systems with unified memory.
    fn bounded_exchange_test_context(dir: &Path) -> SiriusContext {
        let config = dir.join("sirius.yaml");
        std::fs::write(
            &config,
            "sirius:\n  memory:\n    gpu:\n      usage_limit_bytes: 4 GiB\n      reservation_limit_fraction: 1.0\n    host:\n      capacity_bytes: 1 GiB\n      initial_number_pools: 2\n      pool_size: 64\n      block_size: 1048576\n",
        )
        .unwrap();
        SiriusContext::from_config_file(&config).expect("bounded exchange context")
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
    /// `rows_per_group`, so two files over the same keys can slice them into different batches.
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
        // it is covered by the fan-out tests below rather than pretended at here.
        assert!(
            receiver.relay_from(&mut sender, 7, 7, 0).is_err(),
            "relay_from before build() must be rejected"
        );
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
            let result = receiver.result_to_arrow().unwrap();
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
                let result = receiver.result_to_arrow().unwrap();
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
                let result = receiver.result_to_arrow().unwrap();
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
    /// a loud error, and leases release (and the free list coalesces back to one block) before
    /// the receiver runs — which only works if push really copied the data out of the lease.
    /// Requires a GPU.
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

    /// Copy-out-on-arrival through the inbound store: frames exported into the arena are staged
    /// into pool memory FROM ANOTHER THREAD and their leases released before the receiver
    /// fragment exists; the receiver then takes them by ticket and produces the same rows as the
    /// `push_packed` hop. Also pins: a taken ticket cannot be pushed or dropped twice, and the
    /// store is empty once the receiver holds everything. Requires a GPU.
    #[test]
    fn inbound_store_hop_matches_packed_hop() {
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
        let receiver_plan = stream_read_plan(0);

        let ctx = SiriusContext::new().expect("bring up sirius context");
        let arena = ctx.staging_arena().expect("arena configured");
        let store = ctx
            .inbound_store()
            .expect("inbound store follows the arena");
        assert_eq!(store.outstanding().unwrap(), 0);

        let mut sender = ctx.fragment().unwrap();
        sender.declare_output(0).unwrap();
        sender.build(&sender_plan).unwrap();
        sender.run().unwrap();
        let mut staged = Vec::new();
        while let Some(batch) = sender.export_packed(0).unwrap() {
            staged.push(batch);
        }
        assert!(!staged.is_empty());
        assert!(arena.outstanding().unwrap() >= staged.len());

        // Another thread stages every frame and hands the leases back, the way an RPC thread
        // does while the engine thread is busy elsewhere.
        let rows_total: u64 = staged.iter().map(|batch| batch.rows).sum();
        let tickets = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    staged
                        .iter()
                        .map(|batch| {
                            let ticket = store.stage(batch).unwrap();
                            if batch.len > 0 {
                                arena.release(batch.offset).unwrap();
                            }
                            ticket
                        })
                        .collect::<Vec<u64>>()
                })
                .join()
                .unwrap()
        });
        // Nothing is left in the arena before any receiver exists: the free list coalesced back
        // to one block, so the next lease lands at the base.
        assert_eq!(arena.outstanding().unwrap(), 0);
        let probe = arena.lease(1024).unwrap();
        assert_eq!(probe, 0);
        arena.release(probe).unwrap();
        assert_eq!(store.outstanding().unwrap(), tickets.len());
        assert!(store.outstanding_bytes().unwrap() > 0);

        let mut receiver = ctx.fragment().unwrap();
        receiver.declare_input_column(0, "id", "BIGINT").unwrap();
        receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
        receiver.declare_input_cardinality(0, rows_total).unwrap();
        receiver.build(&receiver_plan).unwrap();
        for &ticket in &tickets {
            receiver.push_inbound(0, ticket).unwrap();
        }
        assert_eq!(store.outstanding().unwrap(), 0);
        // A ticket is taken exactly once, whichever way.
        let again = receiver.push_inbound(0, tickets[0]);
        assert!(again.is_err());
        assert!(again.unwrap_err().what().contains("not staged"));
        assert!(store.drop(tickets[0]).is_err());
        receiver.close_input(0, 0).unwrap();
        receiver.run().unwrap();
        let result = receiver.result_to_arrow().unwrap();
        assert_eq!(
            rows(&result),
            vec![
                (1, "a".to_string()),
                (2, "b".to_string()),
                (3, "c".to_string()),
            ]
        );

        // A staged frame whose receiver never comes is dropped by ticket and frees the store.
        let mut sender = ctx.fragment().unwrap();
        sender.declare_output(0).unwrap();
        sender.build(&sender_plan).unwrap();
        sender.run().unwrap();
        let batch = sender.export_packed(0).unwrap().expect("one batch");
        let ticket = store.stage(&batch).unwrap();
        arena.release(batch.offset).unwrap();
        assert_eq!(store.outstanding().unwrap(), 1);
        store.drop(ticket).unwrap();
        assert_eq!(store.outstanding().unwrap(), 0);

        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
    }

    /// Reserve host evacuation before admitting a frame, release staging before a receiver
    /// exists, then execute directly from the retained HOST representation. Nullable strings,
    /// empty strings and a value crossing the packer's 8 MiB chunk boundary are checked exactly.
    #[test]
    fn reserved_ingress_preserves_null_strings_and_releases_staging_before_receiver() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        unsafe {
            std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "128MiB");
            std::env::set_var("SIRIUS_EXCHANGE_OPTIMIZED", "1");
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nullable-users.parquet");
        let long = "x".repeat((9 << 20) + 13);
        let expected = vec![
            (Some(1i64), Some("alpha".to_owned())),
            (Some(2), None),
            (None, Some(String::new())),
            (Some(4), Some("snowman: ☃".to_owned())),
            (Some(5), Some("embedded\0nul".to_owned())),
            (Some(6), Some(long)),
        ];
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(
                    expected.iter().map(|row| row.0).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(
                    expected
                        .iter()
                        .map(|row| row.1.as_deref())
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        let plan = local_files_plan(path.to_str().unwrap(), vec!["id".into(), "name".into()]);
        let ctx = bounded_exchange_test_context(dir.path());
        let arena = ctx.staging_arena().unwrap();
        let store = ctx.inbound_store().unwrap();
        let mut sender = ctx.fragment().unwrap();
        sender.declare_output(0).unwrap();
        sender.build(&plan).unwrap();
        sender.run().unwrap();
        let provider = sender.export_provider(0).unwrap();
        let tickets = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let mut tickets = Vec::new();
                    while let Some(batch) = provider.export_packed().unwrap() {
                        let abandoned = store.reserve(batch.len).unwrap();
                        store.cancel_reservation(abandoned).unwrap();
                        store.cancel_reservation(abandoned).unwrap();
                        let reservation = store.reserve(batch.len).unwrap();
                        tickets.push(store.stage_reserved(&batch, reservation).unwrap());
                        if batch.len > 0 {
                            arena.release(batch.offset).unwrap();
                        }
                    }
                    tickets
                })
                .join()
                .unwrap()
        });
        assert!(!tickets.is_empty());
        assert_eq!(arena.outstanding().unwrap(), 0);
        assert!(store.outstanding_bytes().unwrap() > (8 << 20));
        drop(provider);
        drop(sender);
        let mut receiver = ctx.fragment().unwrap();
        receiver.declare_input_column(0, "id", "BIGINT").unwrap();
        receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
        receiver
            .declare_input_cardinality(0, expected.len() as u64)
            .unwrap();
        receiver.build(&stream_read_plan(0)).unwrap();
        for ticket in tickets {
            receiver.push_inbound(0, ticket).unwrap();
        }
        assert_eq!(store.outstanding().unwrap(), 0);
        receiver.close_input(0, 0).unwrap();
        receiver.run().unwrap();
        let result = receiver.result_to_arrow().unwrap();
        let mut actual = Vec::new();
        for batch in result.batches {
            let ids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let names = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for row in 0..batch.num_rows() {
                actual.push((
                    (!ids.is_null(row)).then(|| ids.value(row)),
                    (!names.is_null(row)).then(|| names.value(row).to_owned()),
                ));
            }
        }
        actual.sort();
        let mut expected = expected;
        expected.sort();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual.0, expected.0, "id mismatch at row {index}");
            assert!(
                actual.1 == expected.1,
                "nullable string mismatch at row {index}"
            );
        }
        drop(receiver);
        drop(ctx);
        // The store handle can outlive its context, but cannot accept another reservation.
        assert!(store.reserve(1024).is_err());
        unsafe {
            std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES");
            std::env::remove_var("SIRIUS_EXCHANGE_OPTIMIZED");
        }
    }

    /// A provider owns buffers independently of Fragment, but context destruction fences and
    /// closes it. Subsequent calls from another thread fail before accessing freed resources.
    #[test]
    fn export_provider_rejects_work_after_context_teardown() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        unsafe {
            std::env::set_var("SIRIUS_EXCHANGE_STAGING_BYTES", "64MiB");
            std::env::set_var("SIRIUS_EXCHANGE_OPTIMIZED", "1");
        }
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let ctx = bounded_exchange_test_context(dir.path());
        let provider = {
            let mut sender = ctx.fragment().unwrap();
            sender.declare_output(0).unwrap();
            sender
                .build(&local_files_plan(
                    path.to_str().unwrap(),
                    vec!["id".into(), "name".into()],
                ))
                .unwrap();
            sender.run().unwrap();
            sender.export_provider(0).unwrap()
        };
        drop(ctx);
        std::thread::spawn(move || {
            assert!(provider.export_packed().is_err());
            provider.cancel();
        })
        .join()
        .unwrap();
        unsafe {
            std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES");
            std::env::remove_var("SIRIUS_EXCHANGE_OPTIMIZED");
        }
    }

    /// The zero-row export contract: a zero-row batch leaves `export_packed` as a metadata-only
    /// frame (`offset == 0`, `len == 0`) holding NO staging lease, and the frame round-trips
    /// through `push_packed` into an empty result. Pins the q15 leak: `export_packed` used to
    /// lease `total + slack` even for `total == 0`, while the transport contract says `len == 0`
    /// means nothing to release — every empty cross-CN batch orphaned a >= 8 MiB lease that
    /// stayed outstanding for the process lifetime, exhausting the arena after ~20 passing
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
        // nothing to release), zero leases are outstanding and the free list has coalesced back
        // to one whole-arena block, so the ENTIRE arena is grantable as one lease. Under the leak
        // this throws exhaustion — the orphaned slack lease still splits the free space.
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
        let result = receiver.result_to_arrow().unwrap();
        assert_eq!(rows(&result), Vec::new(), "an empty split carries no rows");

        // Keep the arena out of the other tests' context bring-ups.
        // SAFETY: the GPU lock is still held.
        unsafe { std::env::remove_var("SIRIUS_EXCHANGE_STAGING_BYTES") };
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
