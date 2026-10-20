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

        /// Create a [`Context`] with a ring exchange: send and receive rings of
        /// `frame_bytes` frames allocated before the engine's GPU pool, and
        /// landing limits on the GPU and host tiers. An empty `config_path`
        /// uses built-in defaults.
        fn make_context_with_ring(
            config_path: &CxxString,
            frame_bytes: u64,
            send_bytes: u64,
            recv_bytes: u64,
            gpu_landing_fraction: f64,
            host_landing_bytes: u64,
            disk_writers: u64,
        ) -> Result<UniquePtr<Context>>;

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

        /// A handle that cancels this context's run in progress from another
        /// thread. It may outlive the context, after which it does nothing.
        fn interrupter(self: &Context) -> UniquePtr<Interrupter>;

        /// Handle from [`Context::interrupter`]; its methods are safe to call
        /// from any thread, concurrently with the context's own calls.
        type Interrupter;

        /// Cancel the `execute_substrait` (or fragment run) in progress on the
        /// handle's context: it stops within one GPU task and fails with
        /// DuckDB's interrupt error. With no run in progress, or once the
        /// context is gone, this does nothing.
        fn interrupt(self: &Interrupter);

        /// Holds batches a transport received until a fragment takes them by
        /// token. Callable from any thread; the contract of every method is
        /// documented on the C++ class.
        type ExchangeInbox;

        /// This context's exchange inbox, or null unless it has one GPU memory
        /// space.
        fn exchange_inbox(self: &Context) -> UniquePtr<ExchangeInbox>;

        fn release(self: &ExchangeInbox, token: u64) -> Result<()>;
        fn outstanding(self: &ExchangeInbox) -> Result<usize>;

        /// The engine's ends of the ring exchange: the send and receive rings and
        /// the copies in and out of them. Callable from any thread; the contract
        /// of every method is documented on the C++ class.
        type RingExchange;

        /// This context's ring exchange, or null unless it was made with one.
        fn ring_exchange(self: &Context) -> UniquePtr<RingExchange>;

        fn device(self: &RingExchange) -> i32;
        fn send_base(self: &RingExchange) -> usize;
        fn send_len(self: &RingExchange) -> u64;
        fn recv_base(self: &RingExchange) -> usize;
        fn recv_len(self: &RingExchange) -> u64;
        fn frame_bytes(self: &RingExchange) -> u64;

        fn lease_send(self: &RingExchange) -> Result<i64>;
        fn lease_recv(self: &RingExchange) -> Result<i64>;
        fn release_send(self: &RingExchange, frame: u32) -> Result<()>;
        fn release_recv(self: &RingExchange, frame: u32) -> Result<()>;

        fn stage(self: &RingExchange, token: u64, k: u32, frame: u32) -> Result<u64>;
        fn release_job(self: &RingExchange, token: u64) -> Result<()>;

        /// # Safety
        /// `header_addr` must point at `header_len` readable bytes that outlive
        /// this call. The safe wrapper upholds this.
        unsafe fn begin(
            self: &RingExchange,
            header_addr: usize,
            header_len: usize,
            tier: &mut u8,
            rows: &mut u64,
            frames: &mut u32,
        ) -> Result<u64>;
        fn copy_out(self: &RingExchange, landing: u64, k: u32, frame: u32, len: u64)
        -> Result<u64>;
        fn seal(self: &RingExchange, landing: u64) -> Result<u64>;
        fn abort(self: &RingExchange, landing: u64) -> Result<()>;
        fn done(self: &RingExchange, ticket: u64) -> Result<bool>;
        fn stats_line(self: &RingExchange) -> Result<UniquePtr<CxxString>>;

        /// Add the non-null rows, minimum and maximum of a signed integer
        /// column of the sealed batch under `token`, leaving it in place.
        fn key_stats(
            self: &ExchangeInbox,
            token: u64,
            column: u32,
            rows: &mut u64,
            min: &mut i64,
            max: &mut i64,
        ) -> Result<()>;

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

        fn output_drain(self: &Fragment, stream_id: u64) -> Result<UniquePtr<OutputDrain>>;

        fn export_ring(
            self: Pin<&mut Fragment>,
            stream_id: u64,
            token: &mut u64,
            rows: &mut u64,
            stream_bytes: &mut u64,
        ) -> Result<UniquePtr<CxxVector<u8>>>;

        /// Add the non-null rows, minimum and maximum of a signed integer
        /// column over every batch parked on an output stream, without
        /// draining it.
        fn output_key_stats(
            self: &Fragment,
            stream_id: u64,
            column: u32,
            rows: &mut u64,
            min: &mut i64,
            max: &mut i64,
        ) -> Result<()>;

        /// Push a copy of one column of every batch parked on `source`'s
        /// output stream into an input stream, leaving the source as it was.
        fn copy_output_column(
            self: Pin<&mut Fragment>,
            source: Pin<&mut Fragment>,
            source_stream_id: u64,
            input_stream_id: u64,
            column: u32,
        ) -> Result<usize>;

        /// Push a copy of one column of the sealed batch under `token` into an
        /// input stream, leaving the token for its receiver.
        fn copy_received_column(
            self: Pin<&mut Fragment>,
            input_stream_id: u64,
            token: u64,
            column: u32,
        ) -> Result<()>;

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

        /// Exports one output stream of a fragment while it runs. Callable from
        /// any thread; the contract is documented on the C++ class.
        type OutputDrain;

        fn export_ring_next(
            self: &OutputDrain,
            timeout_ms: u32,
            ended: &mut bool,
            token: &mut u64,
            rows: &mut u64,
            stream_bytes: &mut u64,
        ) -> Result<UniquePtr<CxxVector<u8>>>;
    }
}

pub use ffi::{
    Context, ExchangeInbox, Fragment, Interrupter, OutputDrain, RingExchange, make_context,
    make_context_from_config, make_context_with_ring, make_fragment, stream_view_name,
};
