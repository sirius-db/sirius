//! Arrow-over-brpc exchange transport: the `SIRIUS_CN_EXCHANGE_TRANSPORT=arrow` alternative to the
//! nixl tier for a sender whose output goes to a REMOTE receiver.
//!
//! Wire shape, per parked output stream: the sender pops each parked batch as a host Arrow
//! `RecordBatch` (`export_arrow_next`), slices it into chunks of at most [`MAX_CHUNK_BYTES`],
//! serializes every chunk as one Arrow IPC stream into the brpc attachment of the existing
//! `transmit_packed` RPC (`arrow_ipc = true`, `offset == length == 0`, no staging lease, no nixl
//! WRITE), then sends the `eos` frame and drops the parked output. The receiver
//! (`compute_node_service.rs`, `handle_transmit_packed`) decodes each attachment back into
//! `RecordBatch`es and stages them lease-free; the engine feeds them through `push_arrow`.
//!
//! Same-CN exchanges never come here: they stay native relays (or fusions) on the GPU.
//!
//! ORDERING: the receiver fails a query on a `seq` gap per (exchange key, sender ordinal). Every
//! frame of one destination — the counter and the eos — is issued by the one call of
//! [`send_fragment`] that drains that destination, so the invariant holds without a dedicated
//! transport thread; distinct destinations are independent counters.

use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{Field, Schema};
use prost::Message;
use starrocks_thrift::status_code::TStatusCode;
use tracing::{info, warn};

use crate::fragment_executor::FragmentExecutor;
use crate::nixl_transport::RemoteSendSpec;
use crate::proto::starrocks::p_internal_service_brpc::methods;
use crate::proto::starrocks::{PTransmitPackedParams, PTransmitPackedResult, PUniqueId, StatusPb};
use crate::prpc_client::PrpcClient;

/// Upper bound on the Arrow buffer bytes one `transmit_packed` attachment carries. The PRPC
/// decoder refuses a whole message above 256 MiB (`prpc.rs`, `MAX_PRPC_MESSAGE_SIZE`), so a
/// chunk must sit well under it with room for IPC framing and the variable-width slack of an
/// estimate by rows.
pub(crate) const MAX_CHUNK_BYTES: usize = 64 << 20;

/// Slices `batch` into row ranges whose estimated buffer bytes stay at or under `max_bytes`,
/// preserving row order; one chunk when it already fits. The estimate is
/// `RecordBatch::get_array_memory_size` spread evenly over the rows, so a batch of very uneven
/// variable-width rows can still overshoot on one chunk — the bound is a sizing rule for the
/// PRPC cap, not a wire guarantee. Every chunk shares the input's buffers (no copy).
pub(crate) fn chunk_by_rows(batch: &RecordBatch, max_bytes: usize) -> Vec<RecordBatch> {
    let rows = batch.num_rows();
    let bytes = batch.get_array_memory_size();
    if rows <= 1 || bytes <= max_bytes {
        return vec![batch.clone()];
    }
    let chunks = bytes.div_ceil(max_bytes.max(1)).min(rows);
    let rows_per_chunk = rows.div_ceil(chunks);
    (0..rows)
        .step_by(rows_per_chunk)
        .map(|start| batch.slice(start, rows_per_chunk.min(rows - start)))
        .collect()
}

/// Serializes one batch as an Arrow IPC stream (schema message, one record batch, end marker).
pub(crate) fn encode_ipc(batch: &RecordBatch) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(batch.get_array_memory_size() + 4096);
    let mut writer = StreamWriter::try_new(&mut out, batch.schema_ref().as_ref())
        .map_err(|err| format!("failed to start an Arrow IPC stream: {err}"))?;
    writer
        .write(batch)
        .map_err(|err| format!("failed to serialize a record batch as Arrow IPC: {err}"))?;
    writer
        .finish()
        .map_err(|err| format!("failed to finish an Arrow IPC stream: {err}"))?;
    drop(writer);
    Ok(out)
}

/// Decodes one Arrow IPC stream into its record batches (one or more).
pub(crate) fn decode_ipc(bytes: &[u8]) -> Result<Vec<RecordBatch>, String> {
    let reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|err| format!("attachment is not an Arrow IPC stream: {err}"))?;
    reader
        .collect::<Result<Vec<_>, _>>()
        .map_err(|err| format!("Arrow IPC stream did not decode: {err}"))
}

/// The batch with its columns renamed positionally to `names` (the engine exports types only;
/// the sender's plan knows the names). Shares the column buffers.
pub(crate) fn with_names(batch: &RecordBatch, names: &[String]) -> Result<RecordBatch, String> {
    if names.len() != batch.num_columns() {
        return Err(format!(
            "exported Arrow batch carries {} columns but the sender plan names {}",
            batch.num_columns(),
            names.len()
        ));
    }
    let fields: Vec<Field> = batch
        .schema_ref()
        .fields()
        .iter()
        .zip(names)
        .map(|(field, name)| field.as_ref().clone().with_name(name))
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), batch.columns().to_vec())
        .map_err(|err| format!("failed to rename the exported Arrow batch: {err}"))
}

/// Turns a method-level StarRocks status into `Err` naming the method.
pub(crate) fn check_status(what: &str, status: &StatusPb) -> Result<(), String> {
    if status.status_code == TStatusCode::OK.0 {
        return Ok(());
    }
    Err(format!(
        "{what} failed with status {}: {}",
        status.status_code,
        status.error_msgs.join("; ")
    ))
}

/// `transmit_packed` over brpc with an Arrow IPC stream (or nothing, for eos) in the attachment.
fn rpc_transmit(
    client: &mut PrpcClient,
    params: PTransmitPackedParams,
    attachment: Vec<u8>,
) -> Result<(), String> {
    let response = client.call(methods::TRANSMIT_PACKED, params.encode_to_vec(), attachment)?;
    let result = PTransmitPackedResult::decode(response.body.as_slice())
        .map_err(|err| format!("undecodable transmit_packed reply: {err}"))?;
    check_status("transmit_packed", &result.status)
}

/// Sender flow: drain one parked output to a remote receiver as Arrow IPC frames and drop the
/// parked output. Blocks until every chunk and the eos frame have been acknowledged; on a failed
/// send the parked output is still dropped (best-effort), so a dead query does not pin it.
pub(crate) fn send_fragment(
    executor: &dyn FragmentExecutor,
    spec: &RemoteSendSpec,
) -> Result<(), String> {
    let started = Instant::now();
    let mut client = PrpcClient::new(&spec.host, spec.brpc_port);
    let (hi, lo) = spec.slot.fragment_instance_id.as_halves();
    let finst_id = PUniqueId { hi, lo };
    let frame = |eos: bool, seq: i64, rows: Option<u64>| PTransmitPackedParams {
        finst_id: Some(finst_id),
        node_id: Some(spec.slot.node_id),
        sender_id: Some(spec.slot.sender_id),
        eos: Some(eos),
        seq: Some(seq),
        // No staging lease exists for an Arrow frame; the receiver reads the attachment.
        offset: Some(0),
        length: Some(0),
        column_names: spec.names.clone(),
        canary: None,
        rows,
        arrow_ipc: Some(true),
    };
    let mut seq: i64 = 0;
    let mut batches: u64 = 0;
    let mut bytes: u64 = 0;

    let sent = (|| -> Result<(), String> {
        while let Some(batch) = executor.export_arrow_next(spec.slot)? {
            let named = with_names(&batch, &spec.names)?;
            for chunk in chunk_by_rows(&named, MAX_CHUNK_BYTES) {
                let payload = encode_ipc(&chunk)?;
                bytes += payload.len() as u64;
                rpc_transmit(
                    &mut client,
                    frame(false, seq, Some(chunk.num_rows() as u64)),
                    payload,
                )?;
                seq += 1;
                batches += 1;
            }
        }
        rpc_transmit(&mut client, frame(true, seq, None), Vec::new())
    })();
    if sent.is_err() {
        // Best-effort GPU cleanup, as the nixl tier does: without it a failed transmit pins the
        // parked output for the process lifetime. A slot already retired with its query is Ok.
        if let Err(drop_err) = executor.drop_parked(spec.slot) {
            warn!(
                slot = ?spec.slot,
                error = %drop_err,
                "failed to drop the parked output of a failed remote Arrow transmit"
            );
        }
    }
    sent?;
    executor.drop_parked(spec.slot)?;
    info!(
        stream_id = spec.slot.node_id,
        sender_id = spec.slot.sender_id,
        dest = %client.peer(),
        batches,
        bytes,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "transmitted batches via arrow"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow_array::{
        Array, ArrayRef, BooleanArray, Date32Array, Decimal64Array, Decimal128Array, Float64Array,
        Int64Array, StringArray,
    };
    use arrow_schema::DataType;

    use super::*;

    fn fixed_width_batch(rows: usize) -> RecordBatch {
        let ids = Int64Array::from_iter_values(0..rows as i64);
        let xs = Float64Array::from_iter_values((0..rows).map(|i| i as f64 * 0.5));
        RecordBatch::try_from_iter([
            ("id", Arc::new(ids) as ArrayRef),
            ("x", Arc::new(xs) as ArrayRef),
        ])
        .unwrap()
    }

    /// The bytes bound: every chunk of a 16 MiB batch serializes under a 1 MiB budget plus
    /// framing; row conservation: the chunks tile the input in order with nothing lost or
    /// duplicated.
    #[test]
    fn chunks_stay_under_the_byte_bound_and_conserve_rows() {
        let rows = 1 << 20; // 16 MiB of fixed-width buffers (plus array bookkeeping)
        let batch = fixed_width_batch(rows);
        let max_bytes = 1 << 20;
        let chunks = chunk_by_rows(&batch, max_bytes);
        assert_eq!(
            chunks.len(),
            batch.get_array_memory_size().div_ceil(max_bytes),
            "one chunk per max_bytes of the estimate"
        );
        assert!(
            chunks.len() >= 16,
            "16 MiB of buffers need at least 16 chunks"
        );

        let mut next_id = 0i64;
        for chunk in &chunks {
            let encoded = encode_ipc(chunk).unwrap();
            assert!(
                encoded.len() <= max_bytes + 4096,
                "a chunk serialized to {} bytes, over the {max_bytes} bound",
                encoded.len()
            );
            let ids = chunk
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for id in ids.values() {
                assert_eq!(*id, next_id, "rows must tile the input in order");
                next_id += 1;
            }
        }
        assert_eq!(next_id, rows as i64, "every row lands in exactly one chunk");
        assert_eq!(
            chunks.iter().map(RecordBatch::num_rows).sum::<usize>(),
            rows
        );
    }

    /// A batch that already fits, and an empty one, are passed through as one chunk.
    #[test]
    fn a_fitting_or_empty_batch_is_one_chunk() {
        let small = fixed_width_batch(10);
        let chunks = chunk_by_rows(&small, MAX_CHUNK_BYTES);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], small);

        let empty = fixed_width_batch(0);
        let chunks = chunk_by_rows(&empty, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].num_rows(), 0);
        // And an empty batch still round-trips through IPC (a zero-row parked batch is legal).
        assert_eq!(
            decode_ipc(&encode_ipc(&empty).unwrap()).unwrap(),
            vec![empty]
        );
    }

    /// The IPC hop preserves values, nulls and schema for the types the engine exports: the
    /// TPC-H column set (int64, double, utf8, date32, decimal64 as cudf spells DECIMAL(15,2))
    /// plus bool and decimal128.
    #[test]
    fn ipc_round_trip_preserves_values_nulls_and_schema() {
        let batch = RecordBatch::try_from_iter_with_nullable([
            (
                "id",
                Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef,
                true,
            ),
            (
                "x",
                Arc::new(Float64Array::from(vec![Some(0.5), Some(-1.0), None])) as ArrayRef,
                true,
            ),
            (
                "flag",
                Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])) as ArrayRef,
                true,
            ),
            (
                "name",
                Arc::new(StringArray::from(vec![Some("a"), Some(""), None])) as ArrayRef,
                true,
            ),
            (
                "price",
                Arc::new(
                    Decimal64Array::from(vec![Some(1234), None, Some(-5)])
                        .with_precision_and_scale(18, 2)
                        .unwrap(),
                ) as ArrayRef,
                true,
            ),
            (
                "wide",
                Arc::new(
                    Decimal128Array::from(vec![Some(1), Some(2), None])
                        .with_precision_and_scale(38, 4)
                        .unwrap(),
                ) as ArrayRef,
                true,
            ),
            (
                "day",
                Arc::new(Date32Array::from(vec![Some(19000), None, Some(0)])) as ArrayRef,
                true,
            ),
        ])
        .unwrap();

        let decoded = decode_ipc(&encode_ipc(&batch).unwrap()).unwrap();
        assert_eq!(decoded, vec![batch.clone()]);
        assert_eq!(
            decoded[0].schema_ref().field(4).data_type(),
            &DataType::Decimal64(18, 2)
        );

        // A sliced chunk (non-zero offsets on every child) round-trips as the slice alone.
        let slice = batch.slice(1, 2);
        let decoded = decode_ipc(&encode_ipc(&slice).unwrap()).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].num_rows(), 2);
        assert_eq!(decoded[0], slice);
    }

    /// Garbage is refused, never decoded into rows.
    #[test]
    fn a_non_ipc_attachment_is_an_error() {
        let err = decode_ipc(&[0xAB; 16]).unwrap_err();
        assert!(err.contains("Arrow IPC"), "{err}");
    }

    /// The engine exports types only; the sender's plan names ride along positionally.
    #[test]
    fn with_names_renames_positionally_and_refuses_a_count_mismatch() {
        let batch = fixed_width_batch(3);
        let named = with_names(&batch, &["a".to_string(), "b".to_string()]).unwrap();
        assert_eq!(named.schema_ref().field(0).name(), "a");
        assert_eq!(named.schema_ref().field(1).name(), "b");
        assert_eq!(named.column(0), batch.column(0));

        let err = with_names(&batch, &["only".to_string()]).unwrap_err();
        assert!(
            err.contains("2 columns") && err.contains("names 1"),
            "{err}"
        );
    }
}
