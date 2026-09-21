//! Arrow IPC hop between CN processes over `PInternalService.transmit_chunk`.
//!
//! A remote destination is one PRPC `transmit_chunk` to the peer's advertised brpc port.
//! Protobuf names the receiver `(finst_id, node_id, sender_id)` plus a per-sender `sequence`
//! and `eos`; the attachment is an Arrow IPC stream of zero or one batch. Column names live
//! in the IPC schema (a schema-only stream when the frame has no batch). This is not
//! StarRocks `ChunkPB` and not packed GPU bytes.

use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::RecordBatch;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};
use prost::Message;

use crate::proto::starrocks::{
    PTransmitChunkParams, PTransmitChunkResult, p_internal_service_brpc::SERVICE_NAME,
    p_internal_service_brpc::methods,
};
use crate::prpc;
use crate::result_store::FragmentInstanceId;

const TRANSMIT_CHUNK_TIMEOUT: Duration = Duration::from_secs(60);

/// One Arrow hop frame: identity, sequence, optional batch, and eos.
#[derive(Clone, Debug)]
pub(crate) struct ArrowExchangeFrame {
    pub(crate) fragment_instance_id: FragmentInstanceId,
    pub(crate) dest_stream: i32,
    pub(crate) sender_id: i32,
    pub(crate) seq: i64,
    pub(crate) eos: bool,
    pub(crate) names: Vec<String>,
    pub(crate) batch: Option<RecordBatch>,
}

impl ArrowExchangeFrame {
    /// Encodes this frame as `PTransmitChunkParams` plus an Arrow IPC attachment.
    pub(crate) fn encode(&self) -> Result<(PTransmitChunkParams, Vec<u8>), String> {
        Ok((
            PTransmitChunkParams {
                finst_id: Some(self.fragment_instance_id.to_proto()),
                node_id: Some(self.dest_stream),
                sender_id: Some(self.sender_id),
                be_number: Some(0),
                eos: Some(self.eos),
                sequence: Some(self.seq),
                chunks: Vec::new(),
                query_statistics: None,
                use_pass_through: Some(false),
                is_pipeline_level_shuffle: Some(false),
                driver_sequences: Vec::new(),
            },
            encode_ipc(&self.names, self.batch.as_ref())?,
        ))
    }

    /// Decodes a `transmit_chunk` request. Rejects native `ChunkPB` / pass-through / pipeline
    /// shuffle so a stock BE cannot be silently misread as Arrow.
    pub(crate) fn decode(params: &PTransmitChunkParams, attachment: &[u8]) -> Result<Self, String> {
        if params.use_pass_through.unwrap_or(false) {
            return Err("pass-through transmit_chunk is not supported".to_string());
        }
        if params.is_pipeline_level_shuffle.unwrap_or(false) {
            return Err("pipeline-level shuffle transmit_chunk is not supported".to_string());
        }
        if !params.chunks.is_empty() {
            return Err(
                "transmit_chunk ChunkPB payloads are not supported; Arrow IPC must be in the \
                 attachment"
                    .to_string(),
            );
        }
        let finst_id = params
            .finst_id
            .as_ref()
            .ok_or_else(|| "transmit_chunk is missing finst_id".to_string())?;
        let dest_stream = params
            .node_id
            .ok_or_else(|| "transmit_chunk is missing node_id".to_string())?;
        let sender_id = params
            .sender_id
            .ok_or_else(|| "transmit_chunk is missing sender_id".to_string())?;
        let seq = params
            .sequence
            .ok_or_else(|| "transmit_chunk is missing sequence".to_string())?;
        let eos = params
            .eos
            .ok_or_else(|| "transmit_chunk is missing eos".to_string())?;
        let (names, batch) = decode_ipc(attachment)?;
        Ok(Self {
            fragment_instance_id: FragmentInstanceId::from(finst_id),
            dest_stream,
            sender_id,
            seq,
            eos,
            names,
            batch,
        })
    }
}

/// Blocking `transmit_chunk` for park-then-send off the BRPC runtime (a `spawn_blocking` worker).
pub(crate) fn transmit_chunk_blocking(
    peer: SocketAddr,
    frame: &ArrowExchangeFrame,
) -> Result<(), String> {
    let (params, attachment) = frame.encode()?;
    let (body, _) = prpc::call_blocking(
        peer,
        SERVICE_NAME,
        methods::TRANSMIT_CHUNK,
        params.encode_to_vec(),
        attachment,
        TRANSMIT_CHUNK_TIMEOUT,
    )
    .map_err(|err| err.to_string())?;
    let result = PTransmitChunkResult::decode(body.as_slice())
        .map_err(|err| format!("transmit_chunk response: {err}"))?;
    match result.status {
        Some(status) if status.status_code == 0 => Ok(()),
        Some(status) => Err(if status.error_msgs.is_empty() {
            format!(
                "transmit_chunk {peer} failed with status {}",
                status.status_code
            )
        } else {
            status.error_msgs.join("; ")
        }),
        None => Err(format!("transmit_chunk {peer} returned no status")),
    }
}

fn encode_ipc(names: &[String], batch: Option<&RecordBatch>) -> Result<Vec<u8>, String> {
    // Parked engine batches often have empty field names. The translator's output
    // names are the hop schema; stamp them on every frame so data and eos match.
    let (schema, owned_batch) = match batch {
        Some(batch) => {
            let schema = stamp_names(batch.schema(), names)?;
            require_column_names(schema.fields().iter().map(|field| field.name().as_str()))?;
            let renamed = RecordBatch::try_new(schema.clone(), batch.columns().to_vec())
                .map_err(|err| format!("arrow ipc rename: {err}"))?;
            (schema, Some(renamed))
        }
        None => {
            require_column_names(names.iter().map(String::as_str))?;
            (
                Arc::new(Schema::new(
                    names
                        .iter()
                        .map(|name| Field::new(name.as_str(), DataType::Null, true))
                        .collect::<Vec<_>>(),
                )),
                None,
            )
        }
    };
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, schema.as_ref())
            .map_err(|err| format!("arrow ipc write: {err}"))?;
        if let Some(batch) = owned_batch.as_ref() {
            writer
                .write(batch)
                .map_err(|err| format!("arrow ipc write batch: {err}"))?;
        }
        writer
            .finish()
            .map_err(|err| format!("arrow ipc finish: {err}"))?;
    }
    Ok(buf)
}

fn stamp_names(
    schema: arrow_schema::SchemaRef,
    names: &[String],
) -> Result<arrow_schema::SchemaRef, String> {
    if names.is_empty() {
        return Ok(schema);
    }
    if names.len() != schema.fields().len() {
        return Err(format!(
            "output names ({}) do not match batch columns ({})",
            names.len(),
            schema.fields().len()
        ));
    }
    Ok(Arc::new(Schema::new(
        schema
            .fields()
            .iter()
            .zip(names)
            .map(|(field, name)| field.as_ref().clone().with_name(name))
            .collect::<Vec<_>>(),
    )))
}

fn require_column_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<(), String> {
    let names: Vec<&str> = names.into_iter().collect();
    if names.is_empty() || names.iter().any(|name| name.is_empty()) {
        return Err("Arrow exchange frames need a non-empty name for every column".to_string());
    }
    Ok(())
}

fn decode_ipc(bytes: &[u8]) -> Result<(Vec<String>, Option<RecordBatch>), String> {
    if bytes.is_empty() {
        return Err(
            "transmit_chunk attachment is empty; expected an Arrow IPC stream carrying column \
             names"
                .to_string(),
        );
    }
    let mut reader = StreamReader::try_new(Cursor::new(bytes), None)
        .map_err(|err| format!("arrow ipc read: {err}"))?;
    let names = reader
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().to_string())
        .collect::<Vec<_>>();
    let batch = reader
        .next()
        .transpose()
        .map_err(|err| format!("arrow ipc batch: {err}"))?;
    Ok((names, batch))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use starrocks_thrift::internal_service::{InternalServiceVersion, TExecPlanFragmentParams};
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::brpc::BrpcServer;
    use crate::compute_node_service::SiriusComputeNodeService;
    use crate::local_exchange::{LocalExchange, SenderSource};

    fn params() -> TExecPlanFragmentParams {
        TExecPlanFragmentParams {
            protocol_version: InternalServiceVersion::V1,
            fragment: None,
            desc_tbl: None,
            params: None,
            coord: None,
            backend_num: None,
            query_globals: None,
            query_options: None,
            enable_profile: None,
            resource_info: None,
            import_label: None,
            db_name: None,
            load_job_id: None,
            load_error_hub_info: None,
            is_pipeline: None,
            pipeline_dop: None,
            per_scan_node_dop: None,
            workgroup: None,
            enable_resource_group: None,
            func_version: None,
            enable_shared_scan: None,
            is_stream_pipeline: None,
            adaptive_dop_param: None,
            group_execution_scan_dop: None,
            pred_tree_params: None,
            exec_stats_node_ids: None,
            arrow_flight_sql_version: None,
        }
    }

    fn int_batch(value: i64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, true)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![Some(value)]))]).unwrap()
    }

    fn is_permission_denied(err: &anyhow::Error) -> bool {
        err.chain().any(|cause| {
            cause
                .downcast_ref::<std::io::Error>()
                .is_some_and(|err| err.kind() == std::io::ErrorKind::PermissionDenied)
        })
    }

    #[test]
    fn ipc_round_trip_preserves_batch_and_schema_only_eos_names() {
        let (names, batch) =
            decode_ipc(&encode_ipc(&["id".to_string()], Some(&int_batch(7))).unwrap()).unwrap();
        assert_eq!(names, ["id"]);
        assert_eq!(
            batch
                .unwrap()
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            7
        );

        let (eos_names, eos_batch) =
            decode_ipc(&encode_ipc(&["id".to_string(), "amount".to_string()], None).unwrap())
                .unwrap();
        assert_eq!(eos_names, ["id", "amount"]);
        assert!(eos_batch.is_none());
    }

    #[test]
    fn ipc_stamps_translator_names_onto_unnamed_engine_batches() {
        use arrow_array::StringArray;

        let unnamed = Arc::new(Schema::new(vec![
            Field::new("", DataType::Utf8, true),
            Field::new("", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            unnamed,
            vec![
                Arc::new(StringArray::from(vec![Some("east")])),
                Arc::new(Int64Array::from(vec![Some(10)])),
            ],
        )
        .unwrap();
        let names = vec!["col_1".to_string(), "col_3".to_string()];
        let (data_names, data_batch) =
            decode_ipc(&encode_ipc(&names, Some(&batch)).unwrap()).unwrap();
        let (eos_names, eos_batch) = decode_ipc(&encode_ipc(&names, None).unwrap()).unwrap();
        assert_eq!(data_names, names);
        assert_eq!(eos_names, names);
        assert!(eos_batch.is_none());
        assert_eq!(
            data_batch
                .unwrap()
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().to_string())
                .collect::<Vec<_>>(),
            names
        );
    }

    #[test]
    fn transmit_chunk_hop_delivers_arrow_batch_and_eos_into_the_rendezvous() {
        let exchange = Arc::new(LocalExchange::default());
        let service = SiriusComputeNodeService::with_executor_and_exchange(
            Arc::new(crate::fragment_executor::StubExecutor),
            exchange.clone(),
            crate::compute_node_service::ExchangeIdentity::default(),
        );
        let listener = match BrpcServer::bind("127.0.0.1", 0) {
            Ok(listener) => listener,
            Err(err) if is_permission_denied(&err) => return,
            Err(err) => panic!("{err:?}"),
        };
        let peer = listener.local_addr().unwrap();
        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let join = thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .unwrap();
            runtime.block_on(
                BrpcServer::with_service(service)
                    .serve_with_listener_shutdown(listener, server_shutdown.cancelled_owned()),
            )
        });
        let instance = FragmentInstanceId::from_halves(11, 22);

        transmit_chunk_blocking(
            peer,
            &ArrowExchangeFrame {
                fragment_instance_id: instance,
                dest_stream: 7,
                sender_id: 0,
                seq: 0,
                eos: false,
                names: vec!["id".to_string()],
                batch: Some(int_batch(42)),
            },
        )
        .unwrap();
        transmit_chunk_blocking(
            peer,
            &ArrowExchangeFrame {
                fragment_instance_id: instance,
                dest_stream: 7,
                sender_id: 0,
                seq: 1,
                eos: true,
                names: vec!["id".to_string()],
                batch: None,
            },
        )
        .unwrap();

        let ready = exchange
            .register_receiver(instance, vec![(7, 1)], params())
            .unwrap()
            .expect("eos already arrived over transmit_chunk");
        let SenderSource::Remote {
            names,
            sender_id,
            batches,
            closed,
        } = &ready.inputs[0].sources[0]
        else {
            panic!("expected a remote source");
        };
        assert_eq!(names, &["id".to_string()]);
        assert_eq!(*sender_id, 0);
        assert!(*closed);
        assert_eq!(batches.len(), 1);
        assert_eq!(
            batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            42
        );

        shutdown.cancel();
        join.join().unwrap().unwrap();
    }
}
