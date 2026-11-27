// SPDX-License-Identifier: Apache-2.0
//
// vortex_gpu_ingest_bench — timed GPU decode of a device-resident Vortex file.
//
// The Vortex counterpart to parquet_ingest_bench and `simpatico benchmark`: the compressed input
// is staged in device memory once, untimed, and only the decode to canonical device arrays is
// timed (vortex-cuda's execute_cuda, then a stream synchronize). Throughput is decoded columnar
// bytes per second, counted the way the GPU tools count them (a cudf column's size), so the three
// tools' GB/s columns share a denominator.
//
// The Vortex file is converted from the Parquet input with vortex-cuda's write strategy
// (CUDA-compatible encodings, --chunk-rows rows per block, as cuDF's NDS-H Vortex writer) and
// cached at --vortex-file so repeated runs decode the identical file. Opening it with vortex-cuda's reader
// places every data segment in device memory; the scan's chunk arrays, still compressed, are held
// across iterations so no host-to-device copy happens in the timed region.
//
// Usage:
//   vortex_gpu_ingest_bench --input FILE.parquet [--vortex-file PATH] [--chunk-rows N]
//                           [--mode per-column|full-table] [--dispatch auto|standalone|fused]
//                           [--warmup N] [--iters N] [--table-out PATH] [--csv-out PATH] [--verify]

use std::collections::HashMap;
use std::io::Cursor;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use bytes::Bytes;
use clap::Parser;
use clap::ValueEnum;
use futures::StreamExt;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::DType;
use vortex::file::Footer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutRef;
use vortex::layout::scan::split_by::SplitBy;
use vortex::session::VortexSession;
use vortex_cuda::CanonicalCudaExt;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaOpenOptionsExt;
use vortex_cuda::CudaSession;
use vortex_cuda::DictionaryExport;
use vortex_cuda::arrow::ArrowDeviceArray;
use vortex_cuda::arrow::release_device_array;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda::layout::cuda_write_strategy;
use vortex_cuda::layout::register_cuda_layout;
use vortex_cuda_driver::result;
use vortex_cuda_driver::sys::CUmemPool_attribute_enum;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Rows per block: the cuDF NDS-H Vortex writer's default rows_per_chunk.
const DEFAULT_CHUNK_ROWS: usize = 16 << 20;

/// Rows per batch of the untimed Parquet reference decode.
const PARQUET_BATCH_ROWS: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Mode {
    PerColumn,
    FullTable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Dispatch {
    /// Fused dynamic dispatch, falling back to per-encoding kernels.
    Auto,
    /// Per-encoding kernels only.
    Standalone,
    /// Fused or partially fused dynamic dispatch only.
    Fused,
}

impl From<Dispatch> for CudaDispatchMode {
    fn from(d: Dispatch) -> Self {
        match d {
            Dispatch::Auto => CudaDispatchMode::Auto,
            Dispatch::Standalone => CudaDispatchMode::StandaloneOnly,
            Dispatch::Fused => CudaDispatchMode::DynDispatchOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Output {
    /// Vortex canonical device arrays (strings as 16-byte views).
    Canonical,
    /// Arrow Device arrays in the layout cudf::from_arrow_device imports: offset-based Utf8
    /// strings, dictionaries decoded (the cuDF NDS-H Vortex reader's export).
    Arrow,
}

#[derive(Parser, Debug)]
#[command(about = "Timed GPU decode of a device-resident Vortex file")]
struct Args {
    #[arg(long)]
    input: PathBuf,
    /// Where the CUDA-compatible Vortex conversion is cached (default: next to --input).
    #[arg(long)]
    vortex_file: Option<PathBuf>,
    /// Rows per block when converting; the scan splits on the file's blocks.
    #[arg(long, default_value_t = DEFAULT_CHUNK_ROWS)]
    chunk_rows: usize,
    #[arg(long, value_enum, default_value = "per-column")]
    mode: Mode,
    #[arg(long, value_enum, default_value = "auto")]
    dispatch: Dispatch,
    #[arg(long, value_enum, default_value = "arrow")]
    output: Output,
    #[arg(long, default_value_t = 3)]
    warmup: usize,
    #[arg(long, default_value_t = 10)]
    iters: usize,
    #[arg(long)]
    table_out: Option<PathBuf>,
    #[arg(long)]
    csv_out: Option<PathBuf>,
    /// Restrict to these columns (repeatable).
    #[arg(long = "select-column")]
    select_columns: Vec<String>,
    /// Before timing, check every selected column's GPU decode against the Parquet decode.
    #[arg(long)]
    verify: bool,
    /// Print each column's compressed size and per-chunk encoding trees, read on the host, then
    /// exit without touching the GPU.
    #[arg(long)]
    describe: bool,
}

// ── Decoded-size accounting (cudf's column size, as the GPU tools report it) ──

fn dtype_name(dt: &DataType) -> String {
    match dt {
        DataType::Int8 => "i8".into(),
        DataType::Int16 => "i16".into(),
        DataType::Int32 => "i32".into(),
        DataType::Int64 => "i64".into(),
        DataType::UInt8 => "u8".into(),
        DataType::UInt16 => "u16".into(),
        DataType::UInt32 => "u32".into(),
        DataType::UInt64 => "u64".into(),
        DataType::Float32 => "f32".into(),
        DataType::Float64 => "f64".into(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => "str".into(),
        DataType::Date32 => "date".into(),
        DataType::Decimal32(..) | DataType::Decimal64(..) | DataType::Decimal128(..) => {
            format!("dec{}", decimal_width(dt) * 8)
        }
        other => format!("{other}"),
    }
}

/// cudf stores a decimal in the narrowest of 32/64/128 bits that holds its precision.
fn decimal_width(dt: &DataType) -> usize {
    let precision = match dt {
        DataType::Decimal32(p, _)
        | DataType::Decimal64(p, _)
        | DataType::Decimal128(p, _)
        | DataType::Decimal256(p, _) => *p,
        _ => unreachable!(),
    };
    match precision {
        0..=9 => 4,
        10..=18 => 8,
        _ => 16,
    }
}

/// A column's size as cudf holds it: fixed-width values, or int32 offsets plus chars for strings.
fn cudf_bytes(arrays: &[ArrowArrayRef]) -> usize {
    let rows: usize = arrays.iter().map(|a| a.len()).sum();
    let Some(first) = arrays.first() else {
        return 0;
    };
    match first.data_type() {
        DataType::Utf8View => {
            let chars: usize = arrays
                .iter()
                .map(|a| a.as_string_view().views().iter().map(|v| *v as u32 as usize).sum::<usize>())
                .sum();
            (rows + 1) * 4 + chars
        }
        DataType::Utf8 => {
            let chars: usize = arrays.iter().map(|a| a.as_string::<i32>().values().len()).sum();
            (rows + 1) * 4 + chars
        }
        dt @ (DataType::Decimal32(..) | DataType::Decimal64(..) | DataType::Decimal128(..)) => {
            rows * decimal_width(dt)
        }
        dt => rows * dt.primitive_width().unwrap_or(0),
    }
}

fn gbps(bytes: usize, ms: f64) -> f64 {
    if ms <= 0.0 {
        return 0.0;
    }
    (bytes as f64 / 1e9) / (ms / 1e3)
}

// ── Timing (the same shape as the GPU tools') ────────────────────────────────

#[derive(Clone, Copy, Default)]
struct TimingStats {
    min: f64,
    median: f64,
    mean: f64,
}

fn compute_stats(ms: &[f64]) -> TimingStats {
    if ms.is_empty() {
        return TimingStats::default();
    }
    let mut sorted = ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    TimingStats {
        min: sorted[0],
        median: if n % 2 == 1 {
            sorted[n / 2]
        } else {
            0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
        },
        mean: sorted.iter().sum::<f64>() / n as f64,
    }
}

// ── Vortex ───────────────────────────────────────────────────────────────────

/// Keep freed device memory in the default pool the decode allocates from, as the cuDF tool's
/// RMM pool does, so steady-state iterations do not map and unmap their outputs.
fn retain_device_pool() -> Result<()> {
    let device = result::device::get(0).map_err(|e| anyhow!("cuda device: {e:?}"))?;
    let pool = unsafe { result::device::get_mem_pool(device) }.map_err(|e| anyhow!("cuda pool: {e:?}"))?;
    let mut threshold = u64::MAX;
    unsafe {
        result::mem_pool::set_attribute(
            pool,
            CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&raw mut threshold).cast(),
        )
    }
    .map_err(|e| anyhow!("cuda pool release threshold: {e:?}"))
}

/// Convert the decoded Parquet batches with vortex-cuda's write strategy (CUDA-compatible
/// encodings, `block_rows` rows per block), as the cuDF NDS-H Vortex writer does.
async fn vortex_write(session: &VortexSession, batches: &[RecordBatch], block_rows: usize) -> Result<Vec<u8>> {
    let schema = batches.first().context("cannot convert an empty table")?.schema();
    let mut chunks = Vec::with_capacity(batches.len());
    for batch in batches {
        chunks.push(session.arrow().from_arrow_record_batch(batch.clone(), &schema)?);
    }
    let dtype: DType = chunks[0].dtype().clone();
    let array = ChunkedArray::try_new(chunks, dtype)?.into_array();
    let mut buf = Vec::new();
    session
        .write_options()
        .with_strategy(cuda_write_strategy(session, block_rows))
        .write(&mut Cursor::new(&mut buf), array.to_array_stream())
        .await?;
    Ok(buf)
}

fn layout_bytes(layout: &LayoutRef, footer: &Footer) -> Result<u64> {
    let segments = footer.segment_map();
    let mut total: u64 = layout
        .segment_ids()
        .iter()
        .map(|id| u64::from(segments[**id as usize].length))
        .sum();
    for child in layout.children()? {
        total += layout_bytes(&child, footer)?;
    }
    Ok(total)
}

/// Segment bytes under each root field's layout, statistics and zone maps included.
fn vortex_compressed_bytes(footer: &Footer) -> Result<HashMap<String, i64>> {
    let root = footer.layout();
    let mut out = HashMap::new();
    for (name, child) in root.child_names().zip(root.children()?) {
        out.insert(name.to_string(), layout_bytes(&child, footer)? as i64);
    }
    Ok(out)
}

/// An exported column, released when dropped.
struct Exported(ArrowDeviceArray);

impl Drop for Exported {
    fn drop(&mut self) {
        release_device_array(&mut self.0);
    }
}

enum Decoded {
    Canonical(#[allow(dead_code)] Vec<ArrayRef>),
    Arrow(#[allow(dead_code)] Vec<Exported>),
}

/// Decode `columns` of every staged chunk on the GPU, returning the results once the stream has
/// drained.
async fn decode(
    staged: &[Vec<ArrayRef>],
    columns: &[usize],
    output: Output,
    ctx: &mut CudaExecutionCtx,
) -> Result<Decoded> {
    let decoded = match output {
        Output::Canonical => {
            let mut out = Vec::with_capacity(staged.len() * columns.len());
            for chunk in staged {
                for &c in columns {
                    out.push(chunk[c].clone().execute_cuda(ctx).await?.into_array());
                }
            }
            Decoded::Canonical(out)
        }
        Output::Arrow => {
            let exporter = Arc::clone(ctx.exporter());
            let mut out = Vec::with_capacity(staged.len() * columns.len());
            for chunk in staged {
                for &c in columns {
                    out.push(Exported(exporter.export_device_array(chunk[c].clone(), ctx).await?));
                }
            }
            Decoded::Arrow(out)
        }
    };
    ctx.synchronize_stream()?;
    Ok(decoded)
}

// ── Reporting (the CSV schema parquet_ingest_bench writes) ───────────────────

struct BenchRow {
    column: String,
    dtype: String,
    rows: usize,
    bytes: usize,
    compressed_bytes: i64,
    decode_ms: TimingStats,
}

impl BenchRow {
    fn ratio(&self) -> f64 {
        if self.compressed_bytes > 0 {
            self.bytes as f64 / self.compressed_bytes as f64
        } else {
            0.0
        }
    }
}

fn total_row(rows: &[BenchRow]) -> BenchRow {
    BenchRow {
        column: "TOTAL".into(),
        dtype: "(all)".into(),
        rows: rows.first().map_or(0, |r| r.rows),
        bytes: rows.iter().map(|r| r.bytes).sum(),
        compressed_bytes: rows.iter().map(|r| r.compressed_bytes).sum(),
        decode_ms: TimingStats {
            min: rows.iter().map(|r| r.decode_ms.min).sum(),
            median: rows.iter().map(|r| r.decode_ms.median).sum(),
            mean: rows.iter().map(|r| r.decode_ms.mean).sum(),
        },
    }
}

// The decode touches no file metadata, so metadata_ms is 0 and decode_only equals decode.
fn write_csv(path: &PathBuf, rows: &[BenchRow]) -> Result<()> {
    let mut out = std::fs::File::create(path)?;
    writeln!(
        out,
        "column,dtype,rows,bytes,compressed_bytes,ratio,decode_ms_min,decode_ms_median,decode_ms_mean,\
         decode_gbps_median,metadata_ms,decode_only_ms_median,decode_only_gbps_median"
    )?;
    for r in rows {
        let gb = gbps(r.bytes, r.decode_ms.median);
        writeln!(
            out,
            "{},{},{},{},{},{},{},{},{},{},0,{},{}",
            r.column,
            r.dtype,
            r.rows,
            r.bytes,
            r.compressed_bytes,
            r.ratio(),
            r.decode_ms.min,
            r.decode_ms.median,
            r.decode_ms.mean,
            gb,
            r.decode_ms.median,
            gb
        )?;
    }
    Ok(())
}

fn write_table(out: &mut dyn Write, rows: &[BenchRow], args: &Args, chunks: usize) -> Result<()> {
    writeln!(
        out,
        "# vortex_gpu_ingest_bench mode={:?} output={:?} dispatch={:?} chunk_rows={} chunks={} warmup={} iters={}",
        args.mode, args.output, args.dispatch, args.chunk_rows, chunks, args.warmup, args.iters
    )?;
    writeln!(out, "# column | dtype | rows | bytes | compressed_bytes | ratio | decode_ms(min/med/mean) | decode_GBps")?;
    for r in rows {
        writeln!(
            out,
            "{} | {} | {} | {} | {} | {:.3}x | {:.3}/{:.3}/{:.3} | {:.2}",
            r.column,
            r.dtype,
            r.rows,
            r.bytes,
            r.compressed_bytes,
            r.ratio(),
            r.decode_ms.min,
            r.decode_ms.median,
            r.decode_ms.mean,
            gbps(r.bytes, r.decode_ms.median)
        )?;
    }
    Ok(())
}

/// An encoding tree with array lengths removed, so trees of different-sized chunks compare equal.
fn encoding_shape(array: &ArrayRef) -> String {
    let tree = array.display_tree_encodings_only().to_string();
    let mut out = String::with_capacity(tree.len());
    let mut rest = tree.as_str();
    while let Some(at) = rest.find(", len=") {
        out.push_str(&rest[..at]);
        rest = &rest[at + ", len=".len()..];
        rest = rest.trim_start_matches(|c: char| c.is_ascii_digit());
    }
    out.push_str(rest);
    out
}

async fn describe(
    session: &VortexSession,
    path: &PathBuf,
    names: &[String],
    column_bytes: &[usize],
) -> Result<()> {
    let file = session.open_options().open_path(path).await?;
    let compressed = vortex_compressed_bytes(file.footer())?;
    let mut ctx = session.create_execution_ctx();
    let mut batches = file.scan()?.with_split_by(SplitBy::Layout).into_array_stream()?;
    let mut shapes: Vec<Vec<(String, usize)>> = vec![Vec::new(); names.len()];
    let mut chunks = 0;
    while let Some(batch) = batches.next().await {
        let record = batch?.execute::<StructArray>(&mut ctx)?;
        for (i, field) in record.iter_unmasked_fields().enumerate() {
            let shape = encoding_shape(field);
            match shapes[i].iter_mut().find(|(s, _)| *s == shape) {
                Some((_, n)) => *n += 1,
                None => shapes[i].push((shape, 1)),
            }
        }
        chunks += 1;
    }
    for (i, name) in names.iter().enumerate() {
        let c = compressed.get(name).copied().unwrap_or(0);
        println!(
            "== {name}: {} bytes, {c} compressed, ratio {:.2}x",
            column_bytes[i],
            column_bytes[i] as f64 / c.max(1) as f64
        );
        for (shape, n) in &shapes[i] {
            println!("-- {n}/{chunks} chunks\n{shape}");
        }
    }
    Ok(())
}

// ── Main ─────────────────────────────────────────────────────────────────────

/// Rows `offset..offset + len` of column `i` across the reference batches.
fn rows_of(batches: &[RecordBatch], i: usize, offset: usize, len: usize) -> Result<ArrowArrayRef> {
    let (mut start, mut pieces) = (0, Vec::new());
    for batch in batches {
        let n = batch.num_rows();
        let (lo, hi) = (offset.max(start), (offset + len).min(start + n));
        if lo < hi {
            pieces.push(batch.column(i).slice(lo - start, hi - lo));
        }
        start += n;
    }
    let refs: Vec<&dyn Array> = pieces.iter().map(|a| a.as_ref()).collect();
    Ok(arrow_select::concat::concat(&refs)?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let _guard = rt.enter();
    let session = VortexSession::default().with_tokio();
    register_cuda_layout(&session);

    // Untimed: decode the Parquet input once for the byte accounting, the Vortex conversion and
    // verification.
    let t0 = Instant::now();
    let parquet_bytes = Bytes::from(std::fs::read(&args.input).with_context(|| format!("reading {:?}", args.input))?);
    let decoded: Vec<RecordBatch> = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes)?
        .with_batch_size(PARQUET_BATCH_ROWS)
        .build()?
        .collect::<std::result::Result<_, _>>()?;
    let schema = decoded.first().context("empty input")?.schema();
    let num_rows: usize = decoded.iter().map(RecordBatch::num_rows).sum();
    eprintln!("decoded {num_rows} rows from {:?} in {:.1}s", args.input, t0.elapsed().as_secs_f64());
    let all_names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let column_bytes: Vec<usize> = (0..all_names.len())
        .map(|i| cudf_bytes(&decoded.iter().map(|b| Arc::clone(b.column(i))).collect::<Vec<_>>()))
        .collect();
    let selected: Vec<usize> = if args.select_columns.is_empty() {
        (0..all_names.len()).collect()
    } else {
        args.select_columns
            .iter()
            .map(|c| all_names.iter().position(|n| n == c).with_context(|| format!("no column '{c}'")))
            .collect::<Result<_>>()?
    };

    let vortex_path = args
        .vortex_file
        .clone()
        .unwrap_or_else(|| args.input.with_extension(format!("gpu{}.vortex", args.chunk_rows)));
    if !vortex_path.exists() {
        let t0 = Instant::now();
        let buf = rt.block_on(vortex_write(&session, &decoded, args.chunk_rows))?;
        std::fs::write(&vortex_path, &buf)?;
        eprintln!("wrote {} bytes to {:?} in {:.1}s", buf.len(), vortex_path, t0.elapsed().as_secs_f64());
    }

    if args.describe {
        return rt.block_on(describe(&session, &vortex_path, &all_names, &column_bytes));
    }

    let mut ctx = CudaSession::create_execution_ctx(&session)?
        .with_dispatch_mode(args.dispatch.into())
        .with_dictionary_export(DictionaryExport::Decode);
    retain_device_pool()?;

    // Untimed: open through the CUDA reader, which reads data segments into device memory, and
    // hold every chunk's still-compressed field arrays.
    let (staged, compressed) = rt.block_on(async {
        let file = session.open_options().with_cuda().open_path(&vortex_path).await?;
        let compressed = vortex_compressed_bytes(file.footer())?;
        let mut batches = file
            .scan()?
            .with_split_by(SplitBy::Layout)
            .into_array_stream()?;
        let mut staged: Vec<Vec<ArrayRef>> = Vec::new();
        while let Some(batch) = batches.next().await {
            let record = batch?.execute::<StructArray>(ctx.execution_ctx())?;
            staged.push(record.iter_unmasked_fields().cloned().collect());
        }
        ctx.synchronize_stream()?;
        anyhow::Ok((staged, compressed))
    })?;
    let host_fields = staged.iter().flatten().filter(|a| a.is_host()).count();
    eprintln!(
        "staged {} chunks x {} columns ({} with host-resident buffers)",
        staged.len(),
        all_names.len(),
        host_fields
    );

    if args.verify {
        let mut host_ctx = session.create_execution_ctx();
        for &i in &selected {
            // Chunk by chunk: a whole string column can overflow Utf8's 32-bit offsets.
            let target = Field::new("", schema.field(i).data_type().clone(), schema.field(i).is_nullable());
            let mut offset = 0;
            for chunk in &staged {
                let host = rt.block_on(async {
                    let decoded = chunk[i].clone().execute_cuda(&mut ctx).await?;
                    ctx.synchronize_stream()?;
                    anyhow::Ok(decoded.into_host(ctx.execution_ctx()).await?.into_array())
                })?;
                let actual = session.arrow().execute_arrow(host, Some(&target), &mut host_ctx)?;
                let expected = rows_of(&decoded, i, offset, actual.len())?;
                if actual.to_data() != expected.to_data() {
                    bail!("GPU decode of '{}' differs from the Parquet decode at rows {offset}..", all_names[i]);
                }
                offset += actual.len();
            }
            if offset != num_rows {
                bail!("GPU decode of '{}' produced {offset} rows, expected {num_rows}", all_names[i]);
            }
            eprintln!("verified {} ({:?})", all_names[i], schema.field(i).data_type());
        }
    }
    drop(decoded);

    let mut row_for = |columns: &[usize], column: String, dtype: String| -> Result<BenchRow> {
        for _ in 0..args.warmup {
            drop(rt.block_on(decode(&staged, columns, args.output, &mut ctx))?);
        }
        let mut samples = Vec::with_capacity(args.iters);
        for _ in 0..args.iters {
            let t0 = Instant::now();
            let out = rt.block_on(decode(&staged, columns, args.output, &mut ctx))?;
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
            drop(out);
        }
        Ok(BenchRow {
            column,
            dtype,
            rows: num_rows,
            bytes: columns.iter().map(|&i| column_bytes[i]).sum(),
            compressed_bytes: columns.iter().map(|&i| compressed.get(&all_names[i]).copied().unwrap_or(0)).sum(),
            decode_ms: compute_stats(&samples),
        })
    };

    let mut rows = Vec::new();
    match args.mode {
        Mode::PerColumn => {
            for &i in &selected {
                let row = row_for(&[i], all_names[i].clone(), dtype_name(schema.field(i).data_type()))?;
                eprintln!(
                    "{:<16} {:>8.2} GB/s ({:.3} ms)",
                    row.column,
                    gbps(row.bytes, row.decode_ms.median),
                    row.decode_ms.median
                );
                rows.push(row);
            }
            let total = total_row(&rows);
            rows.push(total);
        }
        Mode::FullTable => rows.push(row_for(&selected, "TOTAL".into(), "(all)".into())?),
    }

    match &args.table_out {
        Some(path) => write_table(&mut std::fs::File::create(path)?, &rows, &args, staged.len())?,
        None => write_table(&mut std::io::stdout(), &rows, &args, staged.len())?,
    }
    if let Some(path) = &args.csv_out {
        write_csv(path, &rows)?;
    }
    Ok(())
}
