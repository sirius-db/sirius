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

    /// Close `sender_id` on input stream `stream_id`.
    ///
    /// The end-of-stream mirror for senders that are not local fragments —
    /// [`relay_from`](Fragment::relay_from) closes its own sender. Idempotent per sender; the
    /// stream ends once every expected sender has closed.
    pub fn close_input(&mut self, stream_id: u64, sender_id: u32) -> Result<(), Exception> {
        self.inner.pin_mut().close_input(stream_id, sender_id)
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
}
