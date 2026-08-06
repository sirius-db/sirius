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

        /// Declare an output stream. A fragment with none is a result fragment.
        fn declare_output(self: Pin<&mut Fragment>, stream_id: u64) -> Result<()>;
        fn declare_output_broadcast(self: Pin<&mut Fragment>) -> Result<()>;

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
        /// offset and packed payload length; the device bytes are complete on
        /// return (the packing stream is synchronized). Releasing the lease —
        /// via `staging_release(offset)`, after the transmit completes — is
        /// the caller's job.
        fn export_packed(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            offset: &mut u64,
            length: &mut u64,
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
    }
}

pub use ffi::{Context, Fragment, make_context, make_context_from_config, make_fragment,
              stream_view_name};
