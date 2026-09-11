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

        /// Close `sender_id` on input stream `stream_id`. The end-of-stream mirror
        /// for senders that are not local fragments — `relay_from` closes its own.
        /// Idempotent per sender; the stream ends once every expected sender has
        /// closed.
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

        /// DuckDB type names of a built fragment's output (sink) columns — the
        /// types every batch leaving the fragment actually carries, exactly
        /// what `relay_from`'s schema guard compares against a receiver's
        /// declared input columns. Fallible: unbuilt and result fragments both
        /// surface as `Err`.
        fn output_types(self: &Fragment) -> Result<UniquePtr<CxxVector<CxxString>>>;
    }
}

pub use ffi::{
    Context, Fragment, make_context, make_context_from_config, make_fragment, stream_view_name,
};
