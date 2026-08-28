//! Low-level `cxx` bindings to the Sirius C++ API.
//!
//! This crate is intentionally thin: it exposes the C++ types and free functions
//! declared in the `#[cxx::bridge]` module below and nothing else. Safe, idiomatic
//! wrappers live in the [`sirius`](https://docs.rs/sirius) crate.
//!
//! The bridge binds Sirius's **public C++ surface** (`src/include/sirius_ffi.hpp`):
//! an RAII [`Context`] held via [`cxx::UniquePtr`]. Constructing it brings up an
//! initialized engine; dropping the `UniquePtr` tears it down. The header is
//! lightweight, so the bridge compiles without any of Sirius's internal headers
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

        /// Return the staging lease at `offset`; the arena's bump head resets
        /// when the last outstanding lease is released.
        fn staging_release(self: Pin<&mut Context>, offset: u64) -> Result<()>;

        /// Device base address of the staging arena, for transport memory
        /// registration.
        fn staging_base(self: &Context) -> Result<usize>;

        /// Capacity of the staging arena in bytes.
        fn staging_capacity(self: &Context) -> Result<u64>;

        /// Pin a table into the engine's scan cache so later plans that scan
        /// the same resolved source are served from memory. Runs the same
        /// `pin_table` table function `CALL pin_table(...)` runs on the DuckDB
        /// extension path, on the context's embedded connection. `cols_joined`
        /// is a `'\n'`-separated column list (empty pins every column);
        /// `format` is `"parquet"`/`"duckdb"` or empty to infer from the path
        /// suffix; `schema_name` applies to format `duckdb` only. Must be
        /// called on the context's owning thread and never between a
        /// fragment's `build` and `run`. Returns a one-line summary; fallible
        /// (bad arguments, unmatched glob, engine failure).
        fn pin_table(
            self: Pin<&mut Context>,
            path: &CxxString,
            tier: &CxxString,
            name: &CxxString,
            cols_joined: &CxxString,
            format: &CxxString,
            schema_name: &CxxString,
        ) -> Result<UniquePtr<CxxString>>;

        /// Remove the pinned entry `name` and release its memory. Same
        /// threading contract as `pin_table`. Returns a one-line summary.
        fn unpin_table(self: Pin<&mut Context>, name: &CxxString) -> Result<UniquePtr<CxxString>>;

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
        /// batches that outlive its own query) or none (a result fragment,
        /// which produces Arrow).
        type Fragment;

        /// Create a [`Fragment`] on `context`, which must outlive it.
        fn make_fragment(context: Pin<&mut Context>) -> Result<UniquePtr<Fragment>>;

        /// Name of the view a plan reads to consume input stream `stream_id` —
        /// the single definition of the convention, so a front end emitting the
        /// read and the engine creating the view cannot drift apart.
        fn stream_view_name(stream_id: u64) -> UniquePtr<CxxString>;

        /// Declare one column of an input stream, in plan order. `ty` is a
        /// DuckDB type name; a stream has no file to probe.
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
        fn declare_output_broadcast(self: Pin<&mut Fragment>) -> Result<()>;
        fn declare_output_hash_key(self: Pin<&mut Fragment>, column_index: u32) -> Result<()>;

        /// Plan `substrait_plan` against the declared streams and open the
        /// fragment's query lifecycle.
        fn build(self: Pin<&mut Fragment>, substrait_plan: &CxxString) -> Result<()>;

        /// Move every batch parked on `source`'s output stream into this
        /// fragment's input stream as native handles — no Arrow, no file, no
        /// copy — then close `sender_id`. Returns the number of batches moved.
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

        /// Record that `sender_id` finished producing into an input stream —
        /// the EOS mirror of `push_packed` for remote senders.
        fn close_input(self: Pin<&mut Fragment>, stream_id: u64, sender_id: u32) -> Result<()>;

        /// Execute the fragment and close its query lifecycle.
        fn run(self: Pin<&mut Fragment>) -> Result<()>;

        /// Write a result fragment's rows into the caller-owned
        /// `ArrowArrayStream` at `out_stream_addr`.
        ///
        /// # Safety
        /// `out_stream_addr` must be the address of a valid, writable
        /// `ArrowArrayStream` that outlives this call.
        unsafe fn result_to_arrow(
            self: Pin<&mut Fragment>,
            out_stream_addr: usize,
        ) -> Result<()>;

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
