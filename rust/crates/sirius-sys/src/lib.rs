//! Low-level `cxx` bindings to the Sirius C++ API.
//!
//! This crate is intentionally thin: it exposes the C++ types and free functions
//! declared in the `#[cxx::bridge]` module below and nothing else. Safe, idiomatic
//! wrappers live in the [`sirius`](https://docs.rs/sirius) crate.
//!
//! The bridge binds Sirius's **public C++ surface** (`include/sirius/ffi.hpp`):
//! an RAII [`Context`] held via [`cxx::UniquePtr`], plus the [`Fragment`] it
//! creates — one plan fragment of a multi-fragment query. Constructing a context
//! brings up an initialized engine; dropping the `UniquePtr` tears it down. The
//! header is lightweight, so the bridge compiles without any of Sirius's internal
//! headers (cudf/rmm/duckdb). It is the seed of the public API `libsirius` will
//! expose; the bindings link whichever Sirius artifact provides these symbols
//! (the DuckDB extension today, a dedicated `libsirius` later — see `build.rs`).
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
        include!("sirius/ffi.hpp");

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

        /// Receives batches straight into a Context's GPU slab. Callable from any
        /// thread; the contract of every method is documented on the C++ class.
        type DirectExchange;

        /// This context's direct exchange, or null unless it has one GPU memory
        /// space using `allocator: slab`.
        fn direct_exchange(self: &Context) -> UniquePtr<DirectExchange>;

        /// CUDA device of the slab.
        fn device(self: &DirectExchange) -> i32;

        /// Device address of the slab.
        fn region_base(self: &DirectExchange) -> usize;

        /// Length of the slab in bytes.
        fn region_len(self: &DirectExchange) -> u64;

        /// Allocate buffers for a sender's layout and set `token` to them.
        /// Returns `[address, length]` per buffer.
        ///
        /// # Safety
        /// `layout_addr` must point at `layout_len` readable bytes that outlive
        /// this call. The safe wrapper upholds this.
        unsafe fn allocate(
            self: &DirectExchange,
            layout_addr: usize,
            layout_len: usize,
            token: &mut u64,
        ) -> Result<UniquePtr<CxxVector<u64>>>;

        /// Free what `token` holds. Unknown and consumed tokens are ignored.
        fn release(self: &DirectExchange, token: u64) -> Result<()>;

        /// Tokens neither released nor consumed.
        fn outstanding(self: &DirectExchange) -> Result<usize>;

        /// One plan fragment of a multi-fragment query. Either declares output
        /// streams (an intermediate fragment, whose results park as native GPU
        /// batches) or none (a result fragment, which produces Arrow). The
        /// contract of every method is documented on the C++ class.
        type Fragment;

        /// Create a [`Fragment`] on `context`, which must outlive it.
        fn make_fragment(context: Pin<&mut Context>) -> Result<UniquePtr<Fragment>>;

        /// Name of the view a plan reads to consume input stream `stream_id`.
        fn stream_view_name(stream_id: u64) -> UniquePtr<CxxString>;

        /// Declare one column of an input stream, in plan order. `ty` is a
        /// DuckDB type name.
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

        /// Declare the row count of an input stream, summed over its senders.
        fn declare_input_cardinality(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            rows: u64,
        ) -> Result<()>;

        /// Declare an output stream. A fragment with none is a result fragment.
        fn declare_output(self: Pin<&mut Fragment>, stream_id: u64) -> Result<()>;

        /// Replicate the fragment output to every declared output stream.
        fn declare_output_broadcast(self: Pin<&mut Fragment>) -> Result<()>;

        /// Hash-partition across the declared outputs on `column_index`; call
        /// once per key column, in key order.
        fn declare_output_hash_key(self: Pin<&mut Fragment>, column_index: u32) -> Result<()>;

        /// Plan `substrait_plan` against the declared streams.
        fn build(self: Pin<&mut Fragment>, substrait_plan: &CxxString) -> Result<()>;

        /// Move every batch parked on `source`'s output stream into this
        /// fragment's input stream as native handles, then close `sender_id`.
        /// Returns the number of batches moved.
        fn relay_from(
            self: Pin<&mut Fragment>,
            source: Pin<&mut Fragment>,
            source_stream_id: u64,
            input_stream_id: u64,
            sender_id: u32,
        ) -> Result<usize>;

        /// Export the next batch with rows parked on an output stream: sets
        /// `token`, `rows` and `src` (`[address, length]` per buffer) and returns
        /// the layout, or null once the stream is drained.
        fn export_direct(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            token: &mut u64,
            rows: &mut u64,
            src: Pin<&mut CxxVector<u64>>,
        ) -> Result<UniquePtr<CxxVector<u8>>>;

        /// Push the batch received under `token` into an input stream.
        fn push_received(self: Pin<&mut Fragment>, stream_id: u64, token: u64) -> Result<()>;

        /// Close `sender_id` on input stream `stream_id`, for a sender that is
        /// not a local fragment.
        fn close_input(self: Pin<&mut Fragment>, stream_id: u64, sender_id: u32) -> Result<()>;

        /// Execute the fragment. Blocks until its pipelines finish.
        fn run(self: Pin<&mut Fragment>) -> Result<()>;

        /// Write a result fragment's rows into the caller-owned
        /// `ArrowArrayStream` at `out_stream_addr`.
        ///
        /// # Safety
        /// `out_stream_addr` must be the address of a valid, writable
        /// `ArrowArrayStream` that outlives this call. The safe wrapper upholds
        /// this.
        unsafe fn result_to_arrow(self: Pin<&mut Fragment>, out_stream_addr: usize) -> Result<()>;

        /// Batches currently parked on an output stream.
        fn output_batch_count(self: &Fragment, stream_id: u64) -> Result<usize>;

        /// Total rows parked on an output stream, without draining it.
        fn output_row_count(self: &Fragment, stream_id: u64) -> Result<u64>;

        /// DuckDB type names of a built fragment's output columns.
        fn output_types(self: &Fragment) -> Result<UniquePtr<CxxVector<CxxString>>>;
    }
}

pub use ffi::{
    Context, DirectExchange, Fragment, make_context, make_context_from_config, make_fragment,
    stream_view_name,
};
