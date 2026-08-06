//! Safe, idiomatic Rust bindings for [Sirius](https://github.com/sirius-db/sirius),
//! the GPU-native SQL engine.
//!
//! This crate wraps the low-level [`sirius-sys`] cxx bindings in safe Rust types
//! — the entry point for driving Sirius from Rust.
//!
//! Today it binds just enough to prove the toolchain links against the real
//! Sirius library: constructing a [`SiriusContext`] from defaults or a YAML
//! config file. More of the API surface is added in later PRs.

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
    /// The returned fragment borrows the context, so the compiler enforces the one lifetime rule
    /// the C++ side cannot: a fragment must not outlive the engine it runs on.
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
    /// (copy-out-on-arrival); when the last outstanding lease is released the arena's bump head
    /// resets.
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
    sirius_sys::stream_view_name(stream_id).to_string_lossy().into_owned()
}

/// One plan fragment of a multi-fragment query.
///
/// A fragment declaring one or more **output streams** is rooted in a streaming sink: its results
/// stay on the GPU as native batches that outlive its own query, ready for a downstream fragment
/// to take with [`Fragment::relay_from`]. A fragment declaring **none** is a result fragment and
/// produces Arrow via [`Fragment::into_arrow`].
///
/// Calls are ordered: declare, [`build`](Fragment::build), relay from every sender,
/// [`run`](Fragment::run), then drain. `build` opens a query lifecycle on the shared engine that
/// `run` closes, so one fragment at a time may sit between the two — dropping a built-but-unrun
/// fragment closes the lifecycle for you.
pub struct Fragment<'ctx> {
    inner: UniquePtr<sirius_sys::Fragment>,
    /// Ties the fragment's lifetime to the context that made it.
    _context: PhantomData<&'ctx SiriusContext>,
}

impl Fragment<'_> {
    /// Declare one column of input stream `stream_id`, in plan order. `ty` is a DuckDB type name
    /// (`BIGINT`, `DECIMAL(15,2)`, `DATE`, …).
    pub fn declare_input_column(
        &mut self,
        stream_id: u64,
        name: &str,
        ty: &str,
    ) -> Result<(), Exception> {
        let_cxx_string!(name = name);
        let_cxx_string!(ty = ty);
        self.inner.pin_mut().declare_input_column(stream_id, &name, &ty)
    }

    /// Declare a sender that must close input stream `stream_id` before it ends. With none
    /// declared the stream expects the single sender `0`.
    pub fn declare_input_sender(
        &mut self,
        stream_id: u64,
        sender_id: u32,
    ) -> Result<(), Exception> {
        self.inner.pin_mut().declare_input_sender(stream_id, sender_id)
    }

    /// Declare an output stream. A fragment with no output stream is a result fragment.
    pub fn declare_output(&mut self, stream_id: u64) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output(stream_id)
    }

    /// Every declared output stream receives the full fragment output (a broadcast sink);
    /// output 0 keeps the original batches, the rest carry independent deep copies.
    pub fn declare_output_broadcast(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output_broadcast()
    }

    /// Plan `substrait_plan` against the declared streams and open the fragment's query lifecycle.
    pub fn build(&mut self, substrait_plan: &[u8]) -> Result<(), Exception> {
        let_cxx_string!(plan = substrait_plan);
        self.inner.pin_mut().build(&plan)
    }

    /// Move every batch parked on `source`'s output stream into this fragment's input stream —
    /// as native GPU batch handles, with no Arrow and no file in between — then close `sender_id`.
    ///
    /// Returns the number of batches moved. `source` must have finished [`run`](Fragment::run).
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
    pub fn into_arrow(&mut self) -> Result<SubstraitResult, SiriusError> {
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

    /// Batches currently parked on output stream `stream_id`.
    pub fn output_batch_count(&self, stream_id: u64) -> Result<usize, Exception> {
        self.inner.output_batch_count(stream_id)
    }

    /// Pack the next batch parked on output stream `stream_id` into a fresh staging-arena lease,
    /// as cudf packed bytes.
    ///
    /// `Ok(None)` means nothing is parked right now — for a fragment that finished
    /// [`run`](Fragment::run), the stream is drained. The packed device bytes are complete when
    /// this returns, so a transport may transmit from
    /// `[staging_base() + offset, + len)` immediately; the lease stays live until the caller
    /// hands it back with [`SiriusContext::staging_release`] after the transmit completes.
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

    /// Record that `sender_id` finished producing into input stream `stream_id` — the EOS mirror
    /// of [`push_packed`](Fragment::push_packed) for remote senders
    /// ([`relay_from`](Fragment::relay_from) closes its own sender). The stream ends once every
    /// expected sender has closed.
    pub fn close_input(&mut self, stream_id: u64, sender_id: u32) -> Result<(), Exception> {
        self.inner.pin_mut().close_input(stream_id, sender_id)
    }
}

/// One batch exported into the exchange staging arena as cudf packed bytes.
///
/// `metadata` is the host-side cudf pack metadata (it travels with the payload on the wire);
/// `offset`/`len` locate the packed device payload inside the staging arena of the context that
/// exported it. The exporter's lease at `offset` stays outstanding until
/// [`SiriusContext::staging_release`] is called with it.
pub struct PackedBatch {
    /// cudf pack metadata bytes (host memory).
    pub metadata: Vec<u8>,
    /// Byte offset of the packed payload from the arena base.
    pub offset: u64,
    /// Length of the packed payload in bytes.
    pub len: u64,
}

/// Error returned by [`SiriusContext::execute_substrait`].
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

    /// Points the embedded DuckDB at the locally-built parquet extension (the
    /// SIRIUS_BUILD_DIR default mirrors sirius-sys's build.rs) so it can bind
    /// `parquet_scan`. Call under the GPU lock so no other context bring-up
    /// reads the environment concurrently.
    fn ensure_parquet_extension_env() {
        if std::env::var_os("SIRIUS_DUCKDB_PARQUET_EXTENSION").is_none() {
            let manifest = env!("CARGO_MANIFEST_DIR");
            let build_dir = std::env::var("SIRIUS_BUILD_DIR")
                .unwrap_or_else(|_| format!("{manifest}/../../../build/release"));
            let parquet = format!("{build_dir}/extension/parquet/parquet.duckdb_extension");
            // SAFETY: the GPU lock is held, so no other thread touches the environment here.
            unsafe { std::env::set_var("SIRIUS_DUCKDB_PARQUET_EXTENSION", parquet) };
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
            .set_max_row_group_size(rows_per_group)
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
        ensure_parquet_extension_env();

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

    /// The first multi-output fragment end to end: a broadcast sender's two output streams
    /// each deliver the FULL result to their own receiver — the build side of a broadcast
    /// join once the compute node fans destinations out. Requires a GPU.
    #[test]
    fn broadcast_fragment_feeds_every_destination() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        ensure_parquet_extension_env();

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
            let moved = receiver.relay_from(&mut sender, output_stream, 0, 0).unwrap();
            assert!(moved > 0, "stream {output_stream} must carry the broadcast");
            receiver.run().unwrap();
            let result = receiver.into_arrow().unwrap();
            assert_eq!(rows(&result), expected, "stream {output_stream}");
        }
    }

    /// Byte-range splits of one parquet file must read every row exactly once: each split
    /// yields a disjoint subset, their union is the whole file, and one plan carrying both
    /// splits as separate LocalFiles items equals the whole-file plan. Requires a GPU.
    #[test]
    fn byte_range_splits_read_every_row_exactly_once() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        ensure_parquet_extension_env();

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
            NamedStruct, Plan, PlanRel, ReadRel, Rel, RelRoot, Type, plan_rel, r#type, rel,
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
    /// a loud error, and leases release (and the bump head resets) before the receiver runs —
    /// which only works if push really copied the data out of the lease. Requires a GPU.
    #[test]
    fn packed_hop_matches_relay_hop() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        ensure_parquet_extension_env();
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
            receiver.into_arrow().unwrap()
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
            // ...and with nothing outstanding the bump head reset to the base.
            let probe = ctx.staging_lease(1024).unwrap();
            assert_eq!(probe, 0);
            ctx.staging_release(probe).unwrap();

            receiver.run().unwrap();
            receiver.into_arrow().unwrap()
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

    /// Like [`stream_read_plan`] but declaring `id` as FP64 — for the schema-mismatch
    /// negatives, whose receiver deliberately declares a type the sender does not produce.
    fn stream_read_plan_f64(stream_id: u64) -> Vec<u8> {
        use substrait::proto::read_rel::{NamedTable, ReadType};
        use substrait::proto::{
            NamedStruct, Plan, PlanRel, ReadRel, Rel, RelRoot, Type, plan_rel, r#type, rel,
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
        ensure_parquet_extension_env();

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
        ensure_parquet_extension_env();
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
