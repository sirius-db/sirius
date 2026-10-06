//! Safe, idiomatic Rust bindings for [Sirius](https://github.com/sirius-db/sirius),
//! the GPU-native SQL engine.
//!
//! This crate wraps the low-level [`sirius-sys`][sirius_sys] cxx bindings in safe Rust types
//! — the entry point for driving Sirius from Rust.
//!
//! Two entry points:
//!
//! * [`SiriusContext`] — an initialized engine, constructed from defaults or a
//!   YAML config file, able to execute a whole Substrait plan in one call.
//! * [`Fragment`] — one plan fragment of a multi-fragment query, for driving a
//!   distributed plan a piece at a time.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::path::Path;

use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use cxx::{CxxVector, Exception, UniquePtr, let_cxx_string};

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
    // Behind a `RefCell` because every call into C++ needs `Pin<&mut Context>`, yet a distributed
    // plan keeps several fragments alive at once (senders parked until their receiver relays from
    // them), each borrowing the context through `&self`. Every call borrows mutably only for its
    // own duration.
    inner: RefCell<UniquePtr<sirius_sys::Context>>,
}

/// Arguments for [`SiriusContext::pin_table`], mirroring the SQL surface of
/// `CALL pin_table(...)` on the DuckDB extension path.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PinTableSpec {
    /// Parquet file or glob; `None` for `format = "duckdb"`, where `name` is
    /// the catalog table to pin.
    pub path: Option<String>,
    /// `"gpu"` or `"host"`.
    pub tier: String,
    /// Pin-registry key; also selects the compression plan file.
    pub name: String,
    /// Columns to pin; `None` (or empty) pins every column.
    pub cols: Option<Vec<String>>,
    /// `"parquet"` or `"duckdb"`; `None` infers from the path suffix (a glob
    /// not ending in `.parquet` needs the explicit format).
    pub format: Option<String>,
    /// Schema containing the table, for `format = "duckdb"` only.
    pub schema: Option<String>,
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

    /// Start a new [`Fragment`] on this context. The fragment borrows the context, so it cannot
    /// outlive the engine it runs on.
    pub fn fragment(&self) -> Result<Fragment<'_>, Exception> {
        Ok(Fragment {
            inner: sirius_sys::make_fragment(self.inner.borrow_mut().pin_mut())?,
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

    /// This context's [`DirectExchange`], or `None` unless it has one GPU memory space configured
    /// with `allocator: slab`.
    pub fn direct_exchange(&self) -> Option<DirectExchange> {
        let inner = self.inner.borrow().direct_exchange();
        (!inner.is_null()).then_some(DirectExchange { inner })
    }

    /// Pin a table into the engine's scan cache so later plans that scan the
    /// same resolved source are served from memory — the FFI mirror of
    /// `CALL pin_table(...)`.
    ///
    /// Blocks for the whole materialization on this context's owning thread;
    /// never call it while a fragment sits between `build` and `run`.
    /// Compression engages per the context's YAML `sirius.compression.*`
    /// config. Returns a one-line summary.
    pub fn pin_table(&self, spec: &PinTableSpec) -> Result<String, Exception> {
        let_cxx_string!(path = spec.path.as_deref().unwrap_or(""));
        let_cxx_string!(tier = spec.tier.as_str());
        let_cxx_string!(name = spec.name.as_str());
        let cols_joined = spec.cols.as_deref().unwrap_or(&[]).join("\n");
        let_cxx_string!(cols = cols_joined.as_str());
        let_cxx_string!(format = spec.format.as_deref().unwrap_or(""));
        let_cxx_string!(schema = spec.schema.as_deref().unwrap_or(""));
        let summary = self
            .inner
            .borrow_mut()
            .pin_mut()
            .pin_table(&path, &tier, &name, &cols, &format, &schema)?;
        Ok(summary.to_string_lossy().into_owned())
    }

    /// Remove the pinned entry `name` and release its memory. Same threading
    /// contract as [`pin_table`](Self::pin_table).
    pub fn unpin_table(&self, name: &str) -> Result<String, Exception> {
        let_cxx_string!(name = name);
        let summary = self.inner.borrow_mut().pin_mut().unpin_table(&name)?;
        Ok(summary.to_string_lossy().into_owned())
    }
}

fn collect_arrow_stream(stream: FFI_ArrowArrayStream) -> Result<SubstraitResult, SiriusError> {
    let reader = ArrowArrayStreamReader::try_new(stream).map_err(SiriusError::Arrow)?;
    let schema = reader.schema();
    let batches = reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(SiriusError::Arrow)?;
    Ok(SubstraitResult { schema, batches })
}

/// The name of the view a plan must read to consume input stream `stream_id`.
pub fn stream_view_name(stream_id: u64) -> String {
    sirius_sys::stream_view_name(stream_id)
        .to_string_lossy()
        .into_owned()
}

/// One plan fragment of a multi-fragment query.
///
/// A fragment declaring one or more output streams parks its results on the GPU as native batches
/// for a downstream fragment to take with [`Fragment::relay_from`]. A fragment declaring none is a
/// result fragment and produces Arrow via [`Fragment::result_to_arrow`].
///
/// Calls are ordered: declare, [`build`](Fragment::build), close every input sender,
/// [`run`](Fragment::run), then drain. Any number of fragments may be built before any runs.
/// The full contract of each method is documented on the C++ `sirius::ffi::Fragment`.
pub struct Fragment<'ctx> {
    inner: UniquePtr<sirius_sys::Fragment>,
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

    /// Declare the row count of input stream `stream_id`, summed over its senders, so the
    /// optimizer can pick a join's build side. Undeclared streams plan as 1 row.
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

    /// Every declared output stream receives the full fragment output.
    pub fn declare_output_broadcast(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output_broadcast()
    }

    /// Declare one hash-partition key column; output stream `i` takes partition `i`. Call once
    /// per key, in partition-expression order.
    pub fn declare_output_hash_key(&mut self, column_index: u32) -> Result<(), Exception> {
        self.inner.pin_mut().declare_output_hash_key(column_index)
    }

    /// Plan `substrait_plan` against the declared streams.
    pub fn build(&mut self, substrait_plan: &[u8]) -> Result<(), Exception> {
        let_cxx_string!(plan = substrait_plan);
        self.inner.pin_mut().build(&plan)
    }

    /// Move every batch parked on `source`'s output stream into this fragment's input stream as
    /// native GPU batches, then close `sender_id`. Returns the number of batches moved. `source`
    /// must have finished [`run`](Fragment::run).
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

    /// Export the next batch with rows parked on output stream `stream_id` for a
    /// [`DirectExchange`] peer, or `None` once the stream is drained. Release `token` on this
    /// context's exchange once the buffers in `src` were written.
    pub fn export_direct(&mut self, stream_id: u64) -> Result<Option<ExportedBatch>, Exception> {
        let (mut token, mut rows) = (0, 0);
        let mut src = CxxVector::new();
        let layout =
            self.inner
                .pin_mut()
                .export_direct(stream_id, &mut token, &mut rows, src.pin_mut())?;
        if layout.is_null() {
            return Ok(None);
        }
        Ok(Some(ExportedBatch {
            token,
            rows,
            layout: layout.as_slice().to_vec(),
            src: address_pairs(&src),
        }))
    }

    /// A handle that exports output stream `stream_id` from any thread, including while
    /// [`run`](Fragment::run) executes, so the output ships as it is produced. Take it after
    /// [`build`](Fragment::build) and before `run`. A fragment dropped without running fails it.
    pub fn output_drain(&self, stream_id: u64) -> Result<OutputDrain, Exception> {
        Ok(OutputDrain {
            inner: self.inner.output_drain(stream_id)?,
        })
    }

    /// Push the batch received under `token` into input stream `stream_id`, consuming the token.
    pub fn push_received(&mut self, stream_id: u64, token: u64) -> Result<(), Exception> {
        self.inner.pin_mut().push_received(stream_id, token)
    }

    /// Close `sender_id` on input stream `stream_id`: the end of stream for a sender that is not
    /// a local fragment ([`relay_from`](Fragment::relay_from) closes its own). Idempotent.
    pub fn close_input(&mut self, stream_id: u64, sender_id: u32) -> Result<(), Exception> {
        self.inner.pin_mut().close_input(stream_id, sender_id)
    }

    /// Execute the fragment. Blocks until its pipelines finish.
    pub fn run(&mut self) -> Result<(), Exception> {
        self.inner.pin_mut().run()
    }

    /// Collect a result fragment's rows over the Arrow C Data Interface. Callable once, after
    /// [`run`](Fragment::run).
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

    /// Batches currently parked on output stream `stream_id`.
    pub fn output_batch_count(&self, stream_id: u64) -> Result<usize, Exception> {
        self.inner.output_batch_count(stream_id)
    }

    /// Total rows parked on output stream `stream_id`, without draining it.
    pub fn output_row_count(&self, stream_id: u64) -> Result<u64, Exception> {
        self.inner.output_row_count(stream_id)
    }

    /// DuckDB type names of this built fragment's output columns, in the form
    /// [`declare_input_column`](Fragment::declare_input_column) accepts.
    pub fn output_types(&self) -> Result<Vec<String>, Exception> {
        Ok(self
            .inner
            .output_types()?
            .iter()
            .map(|ty| ty.to_string_lossy().into_owned())
            .collect())
    }
}

/// One batch exported by [`Fragment::export_direct`].
#[derive(Debug)]
pub struct ExportedBatch {
    /// Releases the batch on the sender's [`DirectExchange`] once `src` was written.
    pub token: u64,
    pub rows: u64,
    /// What the receiver passes to [`DirectExchange::allocate`].
    pub layout: Vec<u8>,
    /// `(address, length)` of each buffer, pairing with the receiver's allocation.
    pub src: Vec<(u64, u64)>,
}

/// What [`OutputDrain::next`] found within its timeout.
#[derive(Debug)]
pub enum DrainNext {
    Batch(ExportedBatch),
    /// Nothing yet; the fragment may still produce more.
    Waiting,
    /// The stream ended and every batch was exported.
    End,
}

/// Exports one output stream of a [`Fragment`] while it runs, from [`Fragment::output_drain`].
/// The full contract is documented on the C++ `sirius::ffi::OutputDrain`.
pub struct OutputDrain {
    inner: UniquePtr<sirius_sys::OutputDrain>,
}

// SAFETY: the C++ handle shares only the output's batch stream, which locks its own state, and
// the direct exchange, which serializes every call on one mutex and sets the CUDA device itself.
unsafe impl Send for OutputDrain {}

impl OutputDrain {
    /// The next batch with rows, waiting up to `timeout` for the fragment to produce one. Errors
    /// once the fragment's run failed.
    pub fn next(&mut self, timeout: std::time::Duration) -> Result<DrainNext, Exception> {
        let timeout_ms = u32::try_from(timeout.as_millis()).unwrap_or(u32::MAX);
        let (mut ended, mut token, mut rows) = (false, 0, 0);
        let mut src = CxxVector::new();
        let layout =
            self.inner
                .export_next(timeout_ms, &mut ended, &mut token, &mut rows, src.pin_mut())?;
        if !layout.is_null() {
            return Ok(DrainNext::Batch(ExportedBatch {
                token,
                rows,
                layout: layout.as_slice().to_vec(),
                src: address_pairs(&src),
            }));
        }
        Ok(if ended {
            DrainNext::End
        } else {
            DrainNext::Waiting
        })
    }
}

impl std::fmt::Debug for OutputDrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("OutputDrain")
    }
}

/// Receives batches straight into a context's GPU slab, from [`SiriusContext::direct_exchange`].
/// The full contract is documented on the C++ `sirius::ffi::DirectExchange`.
pub struct DirectExchange {
    inner: UniquePtr<sirius_sys::DirectExchange>,
}

// SAFETY: the C++ handle shares the context's registry, which serializes every call on one mutex
// and sets the CUDA device itself, so no state behind it is tied to a thread.
unsafe impl Send for DirectExchange {}
unsafe impl Sync for DirectExchange {}

impl DirectExchange {
    /// The CUDA device, base address and length of the slab every buffer lies in.
    pub fn region(&self) -> (i32, usize, u64) {
        (
            self.inner.device(),
            self.inner.region_base(),
            self.inner.region_len(),
        )
    }

    /// Allocate buffers for a sender's `layout` without waiting for memory. Returns the token
    /// and the `(address, length)` of each buffer, pairing with the sender's `src`.
    pub fn allocate(&self, layout: &[u8]) -> Result<(u64, Vec<(u64, u64)>), Exception> {
        let mut token = 0;
        // SAFETY: the address and length name `layout`, borrowed for the whole call.
        let dst = unsafe {
            self.inner
                .allocate(layout.as_ptr() as usize, layout.len(), &mut token)?
        };
        Ok((token, address_pairs(&dst)))
    }

    /// Free what `token` holds. Unknown and consumed tokens are ignored.
    pub fn release(&self, token: u64) -> Result<(), Exception> {
        self.inner.release(token)
    }

    /// Holds a fully received batch so it may spill to host while it waits for its receiver. A
    /// sealed token is still pushed with [`Fragment::push_received`] and freed with `release`.
    pub fn seal(&self, token: u64) -> Result<(), Exception> {
        self.inner.seal(token)
    }

    /// Tokens neither released nor consumed.
    pub fn outstanding(&self) -> Result<usize, Exception> {
        self.inner.outstanding()
    }
}

impl std::fmt::Debug for DirectExchange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DirectExchange")
            .field(&self.region())
            .finish()
    }
}

fn address_pairs(flat: &CxxVector<u64>) -> Vec<(u64, u64)> {
    let (pairs, _) = flat.as_slice().as_chunks::<2>();
    pairs.iter().map(|&[address, len]| (address, len)).collect()
}

/// Error returned by [`SiriusContext::execute_substrait`] and [`Fragment::result_to_arrow`].
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
    use substrait::proto::{Plan, PlanRel, ReadRel, Rel, RelRoot, plan_rel, rel};

    use super::{DrainNext, SiriusContext, SubstraitResult, stream_view_name};

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
        root_plan(read, names)
    }

    /// Encodes a plan that reads input stream `stream_id` as `(id BIGINT, name VARCHAR)`.
    fn stream_read_plan(stream_id: u64) -> Vec<u8> {
        use substrait::proto::read_rel::{NamedTable, ReadType};
        use substrait::proto::r#type::{self, Kind, Nullability};
        use substrait::proto::{NamedStruct, Type};

        let nullability = Nullability::Nullable as i32;
        let types = vec![
            Type {
                kind: Some(Kind::I64(r#type::I64 {
                    type_variation_reference: 0,
                    nullability,
                })),
            },
            Type {
                kind: Some(Kind::String(r#type::String {
                    type_variation_reference: 0,
                    nullability,
                })),
            },
        ];
        let names = vec!["id".to_string(), "name".to_string()];
        let read = Rel {
            rel_type: Some(rel::RelType::Read(Box::new(ReadRel {
                base_schema: Some(NamedStruct {
                    names: names.clone(),
                    r#struct: Some(r#type::Struct {
                        types,
                        type_variation_reference: 0,
                        nullability: Nullability::Required as i32,
                    }),
                }),
                read_type: Some(ReadType::NamedTable(NamedTable {
                    names: vec![stream_view_name(stream_id)],
                    ..Default::default()
                })),
                ..Default::default()
            }))),
        };
        root_plan(read, names)
    }

    fn root_plan(read: Rel, names: Vec<String>) -> Vec<u8> {
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
    /// `rows_per_group`, so byte ranges of one file can own different row groups.
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

    /// Sorted `(id, name)` rows of a result, independent of batch boundaries.
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

    /// A hash-partitioned scan fragment feeds both partitions into one stream-read fragment
    /// through `relay_from`, while a third declared sender closes empty as a remote sender would.
    /// Requires a GPU.
    #[test]
    fn fragment_relays_hash_partitioned_scan() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);

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
        assert_eq!(sender.output_types().unwrap(), vec!["BIGINT", "VARCHAR"]);
        sender.run().unwrap();
        let partition_rows = [0, 1].map(|stream| sender.output_row_count(stream).unwrap());
        assert!(
            partition_rows.iter().all(|&rows| rows > 0),
            "both partitions must own rows: {partition_rows:?}"
        );
        let parked_rows = partition_rows.iter().sum();
        assert_eq!(parked_rows, 3);

        let mut receiver = ctx.fragment().unwrap();
        receiver.declare_input_column(0, "id", "BIGINT").unwrap();
        receiver.declare_input_column(0, "name", "VARCHAR").unwrap();
        for sender_id in 0..3 {
            receiver.declare_input_sender(0, sender_id).unwrap();
        }
        receiver.declare_input_cardinality(0, parked_rows).unwrap();
        receiver.build(&stream_read_plan(0)).unwrap();
        for stream in 0..2 {
            let parked = sender.output_batch_count(stream).unwrap();
            let moved = receiver
                .relay_from(&mut sender, stream, 0, stream as u32)
                .unwrap();
            assert_eq!(moved, parked, "stream {stream}");
        }
        receiver.close_input(0, 2).unwrap();
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

    /// Exports a scan's batches and allocates their receive buffers on the same context's slab,
    /// without writing them: the buffers pair up and every token is accounted for. Requires a
    /// GPU.
    #[test]
    fn direct_exchange_pairs_buffers_and_accounts_tokens() {
        let _guard = GPU_CONTEXT_LOCK
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("users.parquet");
        write_users_parquet(&path);
        let config = dir.path().join("slab.yaml");
        std::fs::write(
            &config,
            "sirius:
  topology:
    num_gpus: 1
  space:
    gpu:
      - device_id: 0
        memory_capacity: 2147483648
        allocator: slab
    host:
      - numa_id: 0
        memory_capacity: 4294967296
",
        )
        .unwrap();

        let ctx = SiriusContext::from_config_file(&config).expect("bring up slab context");
        let exchange = ctx
            .direct_exchange()
            .expect("slab context has a direct exchange");
        let mut sender = ctx.fragment().unwrap();
        sender.declare_output(0).unwrap();
        sender
            .build(&local_files_plan(
                path.to_str().unwrap(),
                vec!["id".to_string(), "name".to_string()],
            ))
            .unwrap();
        sender.run().unwrap();
        let mut rows = 0;
        while let Some(batch) = sender.export_direct(0).unwrap() {
            rows += batch.rows;
            let (token, dst) = exchange.allocate(&batch.layout).unwrap();
            let lengths = |buffers: &[(u64, u64)]| buffers.iter().map(|b| b.1).collect::<Vec<_>>();
            assert_eq!(lengths(&dst), lengths(&batch.src));
            exchange.release(token).unwrap();
            exchange.release(token).unwrap();
            exchange.release(batch.token).unwrap();
        }
        assert_eq!(rows, 3);
        assert_eq!(exchange.outstanding().unwrap(), 0);
        assert!(exchange.allocate(b"not a layout").is_err());

        // A drain taken before the run waits, then delivers every batch and the end of the
        // stream from another thread.
        let mut sender = ctx.fragment().unwrap();
        sender.declare_output(0).unwrap();
        sender
            .build(&local_files_plan(
                path.to_str().unwrap(),
                vec!["id".to_string(), "name".to_string()],
            ))
            .unwrap();
        let mut drain = sender.output_drain(0).unwrap();
        assert!(sender.output_drain(1).is_err());
        assert!(matches!(
            drain.next(std::time::Duration::ZERO).unwrap(),
            DrainNext::Waiting
        ));
        sender.run().unwrap();
        let streamed = std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let mut rows = 0;
                    loop {
                        match drain.next(std::time::Duration::from_millis(10)).unwrap() {
                            DrainNext::Batch(batch) => {
                                rows += batch.rows;
                                exchange.release(batch.token).unwrap();
                            }
                            DrainNext::Waiting => {}
                            DrainNext::End => return rows,
                        }
                    }
                })
                .join()
                .unwrap()
        });
        assert_eq!(streamed, 3);
        assert_eq!(exchange.outstanding().unwrap(), 0);
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
}
