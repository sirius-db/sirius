//! Low-level `cxx` bindings to the Sirius C++ API.
//!
//! This crate is intentionally thin: it exposes the C++ types and free functions
//! declared in the `#[cxx::bridge]` module below and nothing else. Safe, idiomatic
//! wrappers live in the [`sirius`](https://docs.rs/sirius) crate.
//!
//! The bridge binds Sirius's **public C++ surface** (`src/include/sirius_ffi.hpp`):
//! an RAII [`Context`] held via [`cxx::UniquePtr`], plus the [`Fragment`] it
//! creates — one plan fragment of a multi-fragment query. Constructing a context
//! brings up an initialized engine; dropping the `UniquePtr` tears it down. The
//! header is lightweight, so the bridge compiles without any of Sirius's internal headers
//! (cudf/rmm/duckdb). It is the seed of the public API `libsirius` will expose;
//! the bindings link whichever Sirius artifact provides these symbols (the DuckDB
//! extension today, a dedicated `libsirius` later — see `build.rs`).
//!
//! The `make_context*` functions are bound as fallible (`Result`): bringing up
//! the engine (or parsing a config file) can throw, and cxx turns a C++ exception
//! into `Err(cxx::Exception)` instead of aborting, so consumers can fail fast.

// The `# Safety` docs on the unsafe bridge fns live on the declarations below;
// cxx's macro expansion hides them from clippy's `missing_safety_doc`, so allow
// it for the generated module.
#[allow(clippy::missing_safety_doc)]
#[cxx::bridge(namespace = "sirius::ffi")]
mod ffi {
    unsafe extern "C++" {
        include!("sirius_ffi.hpp");

        /// RAII handle to an initialized Sirius engine context.
        type Context;

        /// Construct an initialized [`Context`] from built-in defaults, owned by
        /// the returned `UniquePtr`.
        fn make_context() -> Result<UniquePtr<Context>>;

        /// Construct an initialized [`Context`] from the YAML config file at
        /// `config_path`, owned by the returned `UniquePtr`. `config_path` binds
        /// to the C++ `const std::string&` parameter.
        fn make_context_from_config(config_path: &CxxString) -> Result<UniquePtr<Context>>;

        /// Execute a serialized Substrait plan on the GPU, writing the results
        /// into the Arrow C Data Interface stream at `out_stream_addr` — the
        /// address (as `usize`) of a caller-owned `ArrowArrayStream` the caller
        /// releases per the Arrow ABI. `plan` binds to the C++ `const
        /// std::string&` and carries the protobuf-encoded `substrait::Plan`
        /// bytes. Bound as fallible: translation or execution failure surfaces as
        /// `Err(cxx::Exception)`.
        ///
        /// # Safety
        /// `out_stream_addr` must be the address of a valid, writable
        /// `ArrowArrayStream` that outlives this call; C++ writes the result
        /// stream through it. The safe [`sirius`](https://docs.rs/sirius) wrapper
        /// upholds this.
        unsafe fn execute_substrait(
            self: Pin<&mut Context>,
            plan: &CxxString,
            out_stream_addr: usize,
        ) -> Result<()>;

        /// Lease `len` bytes of the exchange staging arena, returning the
        /// lease's byte offset from `staging_base()`. Fallible: no configured
        /// arena (`SIRIUS_EXCHANGE_STAGING_BYTES` unset) and exhaustion both
        /// surface as `Err`.
        fn staging_lease(self: Pin<&mut Context>, len: u64) -> Result<u64>;

        /// Return the staging lease at `offset`; the block goes back to the
        /// arena's free list and coalesces with its free neighbours, so the
        /// space is reusable regardless of release order.
        fn staging_release(self: Pin<&mut Context>, offset: u64) -> Result<()>;

        /// Device base address of the staging arena, for transport memory
        /// registration.
        fn staging_base(self: &Context) -> Result<usize>;

        /// Capacity of the staging arena in bytes.
        fn staging_capacity(self: &Context) -> Result<u64>;

        /// Thread-safe handle to the context's exchange staging arena, sharing
        /// ownership of the ONE allocator with the context (whose
        /// `export_packed` leases from the same arena). Unlike the `staging_*`
        /// methods on [`Context`] — reachable only through the context's owning
        /// thread — every method here may be called from any thread: the C++
        /// side serializes on the arena's internal mutex and makes no CUDA
        /// calls.
        type StagingArena;

        /// The context's staging arena handle, or a null `UniquePtr` when no
        /// arena is configured (`SIRIUS_EXCHANGE_STAGING_BYTES` unset).
        fn staging_arena_handle(self: &Context) -> UniquePtr<StagingArena>;

        /// Lease `len` bytes of the arena, returning the lease's byte offset
        /// from `base()`. Fallible on exhaustion (the arena never blocks).
        fn lease(self: &StagingArena, len: u64) -> Result<u64>;

        /// Return the lease at `offset`; the block goes back to the arena's
        /// free list and coalesces with its free neighbours.
        fn release(self: &StagingArena, offset: u64) -> Result<()>;

        /// Device base address of the arena, for transport memory registration.
        fn base(self: &StagingArena) -> usize;

        /// Capacity of the arena in bytes.
        fn capacity(self: &StagingArena) -> u64;

        /// Leases currently held. Nonzero at quiesce means a leaked lease.
        /// `Result` rather than a bare `usize` because, unlike `base`/`capacity`,
        /// the C++ side takes the arena mutex and is therefore not `noexcept`.
        fn outstanding(self: &StagingArena) -> Result<usize>;

        /// One plan fragment of a multi-fragment query. Either declares output
        /// streams (an intermediate fragment, whose results park as native GPU
        /// batches that outlive its own query) or none (a result fragment, which
        /// produces Arrow).
        ///
        /// Usage order: `declare_*` → `build` → `relay_from` every sender → `run`
        /// → drain via `relay_from` or `result_to_arrow`. Exactly one fragment may
        /// sit between its own `build` and `run`; the engine serializes queries.
        type Fragment;

        /// Create a [`Fragment`] on `context`, which must outlive it.
        fn make_fragment(context: Pin<&mut Context>) -> Result<UniquePtr<Fragment>>;

        /// Name of the view a plan reads to consume input stream `stream_id` — the
        /// single definition of the convention, so a front end emitting the read
        /// and the engine creating the view cannot drift apart.
        fn stream_view_name(stream_id: u64) -> UniquePtr<CxxString>;

        /// Declare one column of an input stream, in plan order. `ty` is a DuckDB
        /// type name; a stream has no file to probe, so the schema is given, never
        /// inferred.
        fn declare_input_column(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            name: &CxxString,
            ty: &CxxString,
        ) -> Result<()>;

        /// Declare a sender that must close this input stream before it ends.
        fn declare_input_sender(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            sender_id: u32,
        ) -> Result<()>;

        /// Declare the row count of an input stream (summed over all its
        /// senders; exact when the caller already holds the stream's batches).
        /// Optional: DuckDB's optimizer uses it for join order / build-side
        /// selection; undeclared streams keep the blind default (cardinality
        /// 1). Last call wins.
        fn declare_input_cardinality(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            rows: u64,
        ) -> Result<()>;

        /// Declare an output stream. A fragment with none is a result fragment.
        fn declare_output(self: Pin<&mut Fragment>, stream_id: u64) -> Result<()>;

        /// Replicate every batch to all declared outputs. Mutually exclusive with
        /// [`declare_output_hash_key`](Fragment::declare_output_hash_key), and must
        /// precede `build`; both are the errors raised here. Whether enough
        /// destinations exist is not known until `build`.
        fn declare_output_broadcast(self: Pin<&mut Fragment>) -> Result<()>;

        /// Hash-partition across the declared outputs on `column_index` (call once
        /// per key column, in key order). Same exclusivity and ordering rules as
        /// broadcast.
        fn declare_output_hash_key(self: Pin<&mut Fragment>, column_index: u32) -> Result<()>;

        /// Plan `substrait_plan` against the declared streams and open the
        /// fragment's query lifecycle. Where the declaration-time rules cannot be
        /// checked, they are checked here:
        ///
        /// * a routing mode declared on a fragment with fewer than two outputs is
        ///   rejected — every row would reach destination 0 regardless, so a
        ///   silently-dropped spec would look like it worked;
        /// * a plan that reads the same declared input stream id more than once is
        ///   rejected — the second read would overwrite the first's registration
        ///   and strand its pipeline waiting for a push that never comes.
        fn build(self: Pin<&mut Fragment>, substrait_plan: &CxxString) -> Result<()>;

        /// Move every batch parked on `source`'s output stream into this fragment's
        /// input stream as native handles — no Arrow, no file, no copy — then close
        /// `sender_id`. Returns the number of batches moved.
        ///
        /// `source` must have finished [`run`](Fragment::run): before that, an empty
        /// stream and a finished one are indistinguishable, and the input would be
        /// closed after zero batches. `input_stream_id` must be declared on this
        /// fragment, and `source` must have output streams (a result fragment is
        /// rejected).
        fn relay_from(
            self: Pin<&mut Fragment>,
            source: Pin<&mut Fragment>,
            source_stream_id: u64,
            input_stream_id: u64,
            sender_id: u32,
        ) -> Result<usize>;

        /// Pack the next batch parked on an output stream into a fresh
        /// staging-arena lease. Returns the cudf pack metadata (a null
        /// `UniquePtr` when nothing is parked right now) and writes the lease
        /// offset, packed payload length, and the batch's exact row count (so
        /// a transport can carry it to the receiver's
        /// `declare_input_cardinality`); the device bytes are complete on
        /// return (the packing stream is synchronized). Releasing the lease —
        /// via `staging_release(offset)`, after the transmit completes — is
        /// the caller's job.
        fn export_packed(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            offset: &mut u64,
            length: &mut u64,
            rows: &mut u64,
        ) -> Result<UniquePtr<CxxVector<u8>>>;

        /// Unpack `length` packed bytes at staging offset `offset` with the
        /// cudf pack metadata at `metadata_addr` (`metadata_len` bytes of host
        /// memory), deep-copy the table out of the lease into pool memory, and
        /// push it into an input stream. Legal between `build()` and `run()`;
        /// the lease is reusable on return. A push after the stream ended is
        /// an `Err`, never a silent drop.
        ///
        /// # Safety
        /// `metadata_addr` must point at `metadata_len` readable bytes of pack
        /// metadata that outlive this call. The safe
        /// [`sirius`](https://docs.rs/sirius) wrapper upholds this.
        unsafe fn push_packed(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            metadata_addr: usize,
            metadata_len: usize,
            offset: u64,
            length: u64,
        ) -> Result<()>;

        /// Import one host-memory Arrow record batch (Arrow C Data Interface) at
        /// `array_addr` / `schema_addr` into input stream `stream_id` as sender
        /// `sender_id`: the host-memory twin of `push_packed`. The buffers are
        /// copied to the GPU before this returns, so the caller may release the
        /// Arrow structs right after. Legal between `build()` and `run()`; it
        /// does not close the sender. A schema mismatch, an undeclared sender
        /// and a push after the stream ended are all `Err`, never a silent drop.
        ///
        /// # Safety
        /// `array_addr` and `schema_addr` must be the addresses of a valid,
        /// readable `ArrowArray` (a struct array, one child per declared column)
        /// and its `ArrowSchema`, both outliving this call. The safe
        /// [`sirius`](https://docs.rs/sirius) wrapper upholds this.
        unsafe fn push_arrow(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            sender_id: u32,
            array_addr: usize,
            schema_addr: usize,
        ) -> Result<()>;

        /// Close `sender_id` on input stream `stream_id`. The end-of-stream mirror
        /// for senders that are not local fragments — `relay_from` closes its own
        /// sender; `push_packed` and `push_arrow` do not. Idempotent per sender;
        /// the stream ends once every expected sender has closed.
        fn close_input(self: Pin<&mut Fragment>, stream_id: u64, sender_id: u32) -> Result<()>;

        /// Execute the fragment and close its query lifecycle. Blocks until its
        /// pipelines finish.
        fn run(self: Pin<&mut Fragment>) -> Result<()>;

        /// Write a result fragment's rows into the caller-owned `ArrowArrayStream`
        /// at `out_stream_addr`.
        ///
        /// # Safety
        /// `out_stream_addr` must be the address of a valid, writable
        /// `ArrowArrayStream` that outlives this call. The safe
        /// [`sirius`](https://docs.rs/sirius) wrapper upholds this.
        unsafe fn result_to_arrow(self: Pin<&mut Fragment>, out_stream_addr: usize) -> Result<()>;

        /// Batches currently parked on an output stream — the evidence that a
        /// fragment boundary carried native batches rather than nothing.
        fn output_batch_count(self: &Fragment, stream_id: u64) -> Result<usize>;

        /// Total rows parked on an output stream, without draining it — what a
        /// local relay's receiver feeds `declare_input_cardinality` before its
        /// own build. Fallible: an unknown stream and a spilled (non-GPU)
        /// parked batch both surface as `Err`; the caller should then skip the
        /// cardinality declaration rather than fail the fragment.
        fn output_row_count(self: &Fragment, stream_id: u64) -> Result<u64>;

        /// DuckDB type names of a built fragment's output (sink) columns — the
        /// types every batch leaving the fragment actually carries, exactly
        /// what `relay_from`'s schema guard compares against a receiver's
        /// declared input columns. Fallible: unbuilt and result fragments both
        /// surface as `Err`.
        fn output_types(self: &Fragment) -> Result<UniquePtr<CxxVector<CxxString>>>;
    }
}

pub use ffi::{
    Context, Fragment, StagingArena, make_context, make_context_from_config, make_fragment,
    stream_view_name,
};
