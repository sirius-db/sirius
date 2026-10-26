//! Decoding of the FE's fragment dispatch payload and the per-query dump used to collect the
//! TPC-H corpus.
//!
//! `PExecPlanFragmentRequest.request` is a thrift-encoded `TPipelineFragmentParamsList`
//! (TCompact when `compact=true`, which `Config.use_compact_thrift_rpc` makes the default;
//! `version` must be `VERSION_3`). One RPC carries every fragment the FE assigned to this
//! backend, **root fragment first** (the FE reverses the list so a backend can bind receivers
//! before senders). Only the first entry carries `desc_tbl`, `file_scan_params`, `coord`,
//! `query_globals` and `resource_info`; the rest are `is_simplified_param=true` and share them
//! (`ThriftPlansBuilder` strips them "to reduce rpc message size"). [`decode_fragment_params_list`]
//! undoes that so every fragment is self-contained for the translator.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use doris_proto::{PExecPlanFragmentRequest, PFragmentRequestVersion};
use doris_thrift::data_sinks::TDataSinkType;
use doris_thrift::palo_internal_service::{TPipelineFragmentParams, TPipelineFragmentParamsList};
use doris_thrift::plan_nodes::TPlanNodeType;
use doris_thrift::types::TUniqueId;
use thrift::protocol::{TBinaryInputProtocol, TCompactInputProtocol, TSerializable};
use thrift::transport::TBufferChannel;

/// Environment variable naming the directory fragment dumps are written to (unset = off).
pub const DUMP_FRAGMENTS_ENV: &str = "SIRIUS_BE_DUMP_FRAGMENTS";

/// Everything one `exec_plan_fragment(_prepare)` RPC asked this backend to run.
#[derive(Clone, Debug, PartialEq)]
pub struct FragmentBatch {
    /// Query id shared by every fragment in the batch.
    pub query_id: TUniqueId,
    /// The fragments in FE order (root first), each with the shared fields merged in.
    pub fragments: Vec<DispatchedFragment>,
}

/// One fragment of a batch, self-contained (shared query-level fields merged in).
#[derive(Clone, Debug, PartialEq)]
pub struct DispatchedFragment {
    /// Position in the FE's list (0 = root fragment).
    pub index: usize,
    /// The fragment parameters with `desc_tbl`/`file_scan_params`/`coord`/`query_globals`/
    /// `resource_info` restored from the batch's first fragment when this one was simplified.
    pub params: TPipelineFragmentParams,
}

impl DispatchedFragment {
    /// FE fragment id (`TPipelineFragmentParams.fragment_id`), if set.
    pub fn fragment_id(&self) -> Option<i32> {
        self.params.fragment_id
    }

    /// Instance ids this backend runs for the fragment (one per `local_params` entry).
    pub fn instance_ids(&self) -> impl Iterator<Item = &TUniqueId> {
        self.params
            .local_params
            .iter()
            .flatten()
            .map(|instance| &instance.fragment_instance_id)
    }

    /// The fragment's output sink type, when the FE set one.
    pub fn sink_type(&self) -> Option<TDataSinkType> {
        self.params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.output_sink.as_ref())
            .map(|sink| sink.type_)
    }

    /// Whether this fragment ends in a `RESULT_SINK` (its rows go to `fetch_data`).
    pub fn is_result_fragment(&self) -> bool {
        self.sink_type() == Some(TDataSinkType::RESULT_SINK)
    }

    /// Plan node types in the FE's flat preorder, root first.
    pub fn node_types(&self) -> Vec<TPlanNodeType> {
        self.params
            .fragment
            .as_ref()
            .and_then(|fragment| fragment.plan.as_ref())
            .map(|plan| plan.nodes.iter().map(|node| node.node_type).collect())
            .unwrap_or_default()
    }

    /// One-line shape summary for logs and the corpus index: sink, node sequence, instances.
    pub fn shape(&self) -> String {
        let nodes = self
            .node_types()
            .iter()
            .map(|node_type| format!("{node_type:?}"))
            .collect::<Vec<_>>()
            .join(" > ");
        format!(
            "fragment_id={} sink={} instances={} nodes=[{nodes}]",
            self.fragment_id()
                .map(|id| id.to_string())
                .unwrap_or_else(|| "?".to_string()),
            self.sink_type()
                .map(|sink| format!("{sink:?}"))
                .unwrap_or_else(|| "none".to_string()),
            self.params.local_params.as_ref().map_or(0, Vec::len),
        )
    }
}

/// Decodes and normalizes one dispatch request into self-contained fragments.
pub fn decode_fragment_params_list(
    request: &PExecPlanFragmentRequest,
) -> Result<FragmentBatch, String> {
    let version = request
        .version
        .unwrap_or(PFragmentRequestVersion::Version2 as i32);
    if version != PFragmentRequestVersion::Version3 as i32 {
        return Err(format!(
            "unsupported PExecPlanFragmentRequest.version {version}; only VERSION_3 \
             (TPipelineFragmentParamsList) is implemented"
        ));
    }
    let bytes = request
        .request
        .as_deref()
        .ok_or_else(|| "PExecPlanFragmentRequest.request is missing".to_string())?;
    let list = if request.compact.unwrap_or(false) {
        deserialize::<TPipelineFragmentParamsList, _>(bytes, |channel| {
            TCompactInputProtocol::new(channel)
        })
    } else {
        deserialize::<TPipelineFragmentParamsList, _>(bytes, |channel| {
            TBinaryInputProtocol::new(channel, true)
        })
    }
    .map_err(|err| format!("failed to deserialize TPipelineFragmentParamsList: {err}"))?;
    normalize(list)
}

/// Merges the batch's shared fields into every simplified fragment.
fn normalize(list: TPipelineFragmentParamsList) -> Result<FragmentBatch, String> {
    let params_list = list
        .params_list
        .ok_or_else(|| "TPipelineFragmentParamsList.params_list is missing".to_string())?;
    let first = params_list
        .first()
        .ok_or_else(|| "TPipelineFragmentParamsList.params_list is empty".to_string())?;
    if first.is_simplified_param == Some(true) {
        return Err(
            "first fragment of TPipelineFragmentParamsList is is_simplified_param=true; \
             its descriptor table is missing"
                .to_string(),
        );
    }
    let query_id = first.query_id.clone();
    // The shared fields live in the first fragment (4.1.x); the list-level copies exist for
    // newer FE versions, so prefer whichever is present.
    let desc_tbl = first.desc_tbl.clone().or(list.desc_tbl);
    let file_scan_params = first.file_scan_params.clone().or(list.file_scan_params);
    let coord = first.coord.clone().or(list.coord);
    let query_globals = first.query_globals.clone().or(list.query_globals);
    let resource_info = first.resource_info.clone().or(list.resource_info);
    let query_options = first.query_options.clone().or(list.query_options);

    let fragments = params_list
        .into_iter()
        .enumerate()
        .map(|(index, mut params)| {
            if params.query_id != query_id {
                return Err(format!(
                    "fragment {index} carries query id {:?} but the batch is for {query_id:?}",
                    params.query_id
                ));
            }
            if params.desc_tbl.is_none() {
                params.desc_tbl = desc_tbl.clone();
            }
            if params.file_scan_params.is_none() {
                params.file_scan_params = file_scan_params.clone();
            }
            if params.coord.is_none() {
                params.coord = coord.clone();
            }
            if params.query_globals.is_none() {
                params.query_globals = query_globals.clone();
            }
            if params.resource_info.is_none() {
                params.resource_info = resource_info.clone();
            }
            if params.query_options.is_none() {
                params.query_options = query_options.clone();
            }
            Ok(DispatchedFragment { index, params })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(FragmentBatch {
        query_id,
        fragments,
    })
}

/// Deserializes a thrift struct from `bytes` with the protocol `make_protocol` wraps around a
/// buffer channel.
fn deserialize<T, P>(
    bytes: &[u8],
    make_protocol: impl FnOnce(TBufferChannel) -> P,
) -> thrift::Result<T>
where
    T: TSerializable,
    P: thrift::protocol::TInputProtocol,
{
    let mut channel = TBufferChannel::with_capacity(bytes.len(), 0);
    let bytes_copied = channel.set_readable_bytes(bytes);
    if bytes_copied != bytes.len() {
        return Err(thrift::Error::Application(thrift::ApplicationError::new(
            thrift::ApplicationErrorKind::Unknown,
            "failed to stage complete thrift payload".to_string(),
        )));
    }
    let mut protocol = make_protocol(channel);
    T::read_from_in_protocol(&mut protocol)
}

/// Renders a `TUniqueId` the way the FE prints it (`hi-lo` in hex).
pub fn print_id(id: &TUniqueId) -> String {
    format!("{:x}-{:x}", id.hi, id.lo)
}

/// Writes a dispatch to `$SIRIUS_BE_DUMP_FRAGMENTS/<query id>/` for offline plan analysis:
/// the raw request bytes (replayable by tests), each fragment in `Debug` form, and a shape
/// summary. Returns the directory, or `None` when dumping is off.
///
/// A query dispatches once per backend, but a backend can receive several batches for one
/// query (a re-dispatch after failure); the per-query sequence number keeps them apart.
pub fn dump_batch(request: &PExecPlanFragmentRequest, batch: &FragmentBatch) -> Option<PathBuf> {
    let dir = std::env::var_os(DUMP_FRAGMENTS_ENV)?;
    let query_dir = Path::new(&dir).join(print_id(&batch.query_id));
    if let Err(err) = std::fs::create_dir_all(&query_dir) {
        tracing::warn!(error = %err, path = %query_dir.display(), "failed to create fragment dump directory");
        return None;
    }
    // Dumps are serialized so two batches of one query cannot pick the same sequence number.
    static DUMP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = DUMP_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let seq = std::fs::read_dir(&query_dir)
        .map(|entries| {
            entries
                .flatten()
                .filter(|entry| {
                    let name = entry.file_name();
                    let name = name.to_string_lossy();
                    name.starts_with("batch-") && name.ends_with("-summary.txt")
                })
                .count()
        })
        .unwrap_or(0);
    let mut summary = String::new();
    let _ = writeln!(summary, "query_id={}", print_id(&batch.query_id));
    let _ = writeln!(summary, "batch_seq={seq}");
    let _ = writeln!(summary, "fragments={}", batch.fragments.len());
    for fragment in &batch.fragments {
        let _ = writeln!(summary, "[{}] {}", fragment.index, fragment.shape());
        let path = query_dir.join(format!(
            "batch-{seq:02}-fragment-{:02}-f{}.txt",
            fragment.index,
            fragment
                .fragment_id()
                .map(|id| id.to_string())
                .unwrap_or_else(|| "x".to_string())
        ));
        write_or_warn(&path, format!("{:#?}", fragment.params).as_bytes());
    }
    if let Some(bytes) = request.request.as_deref() {
        let encoding = if request.compact.unwrap_or(false) {
            "tcompact"
        } else {
            "tbinary"
        };
        write_or_warn(
            &query_dir.join(format!("batch-{seq:02}-request.{encoding}")),
            bytes,
        );
    }
    write_or_warn(
        &query_dir.join(format!("batch-{seq:02}-summary.txt")),
        summary.as_bytes(),
    );
    Some(query_dir)
}

fn write_or_warn(path: &Path, bytes: &[u8]) {
    if let Err(err) = std::fs::write(path, bytes) {
        tracing::warn!(error = %err, path = %path.display(), "failed to write fragment dump");
    }
}

#[cfg(test)]
mod tests {
    use doris_thrift::data_sinks::TDataSink;
    use doris_thrift::descriptors::TDescriptorTable;
    use doris_thrift::palo_internal_service::{
        PaloInternalServiceVersion, TPipelineInstanceParams, TQueryGlobals,
    };
    use doris_thrift::partitions::{TDataPartition, TPartitionType};
    use doris_thrift::plan_nodes::{TPlan, TPlanNode};
    use doris_thrift::planner::TPlanFragment;
    use doris_thrift::types::TNetworkAddress;
    use thrift::protocol::{TCompactOutputProtocol, TOutputProtocol};

    use super::*;

    fn node(node_id: i32, node_type: TPlanNodeType, num_children: i32) -> TPlanNode {
        TPlanNode {
            node_id,
            node_type,
            num_children,
            limit: -1,
            row_tuples: vec![0],
            nullable_tuples: vec![false],
            compact_data: false,
            ..Default::default()
        }
    }

    fn fragment(
        fragment_id: i32,
        sink: TDataSinkType,
        nodes: Vec<TPlanNode>,
        instance: (i64, i64),
    ) -> TPipelineFragmentParams {
        TPipelineFragmentParams {
            protocol_version: PaloInternalServiceVersion::V1,
            query_id: TUniqueId::new(1, 2),
            fragment_id: Some(fragment_id),
            fragment: Some(TPlanFragment {
                plan: Some(TPlan { nodes }),
                output_sink: Some(TDataSink {
                    type_: sink,
                    ..Default::default()
                }),
                partition: TDataPartition {
                    type_: TPartitionType::UNPARTITIONED,
                    ..Default::default()
                },
                ..Default::default()
            }),
            local_params: Some(vec![TPipelineInstanceParams {
                fragment_instance_id: TUniqueId::new(instance.0, instance.1),
                ..Default::default()
            }]),
            ..Default::default()
        }
    }

    /// A root (RESULT_SINK over EXCHANGE) fragment plus a simplified leaf fragment, the way
    /// the FE ships a two-fragment query to one backend.
    fn two_fragment_list() -> TPipelineFragmentParamsList {
        let mut root = fragment(
            0,
            TDataSinkType::RESULT_SINK,
            vec![node(3, TPlanNodeType::EXCHANGE_NODE, 0)],
            (10, 0),
        );
        root.desc_tbl = Some(TDescriptorTable::default());
        root.coord = Some(TNetworkAddress::new("127.0.0.1".to_string(), 9020));
        root.query_globals = Some(TQueryGlobals::default());
        root.is_simplified_param = Some(false);
        let mut leaf = fragment(
            1,
            TDataSinkType::DATA_STREAM_SINK,
            vec![
                node(2, TPlanNodeType::AGGREGATION_NODE, 1),
                node(1, TPlanNodeType::FILE_SCAN_NODE, 0),
            ],
            (10, 1),
        );
        leaf.is_simplified_param = Some(true);
        TPipelineFragmentParamsList {
            params_list: Some(vec![root, leaf]),
            ..Default::default()
        }
    }

    fn compact_request(list: &TPipelineFragmentParamsList) -> PExecPlanFragmentRequest {
        let mut buffer = Vec::new();
        {
            let mut protocol = TCompactOutputProtocol::new(&mut buffer);
            list.write_to_out_protocol(&mut protocol).unwrap();
            protocol.flush().unwrap();
        }
        PExecPlanFragmentRequest {
            request: Some(buffer),
            compact: Some(true),
            version: Some(PFragmentRequestVersion::Version3 as i32),
        }
    }

    #[test]
    fn decodes_tcompact_list_and_restores_shared_fields() {
        let batch = decode_fragment_params_list(&compact_request(&two_fragment_list())).unwrap();

        assert_eq!(batch.query_id, TUniqueId::new(1, 2));
        assert_eq!(batch.fragments.len(), 2);
        let root = &batch.fragments[0];
        assert_eq!(root.fragment_id(), Some(0));
        assert!(root.is_result_fragment());
        assert_eq!(root.node_types(), vec![TPlanNodeType::EXCHANGE_NODE]);
        let leaf = &batch.fragments[1];
        assert_eq!(leaf.fragment_id(), Some(1));
        assert!(!leaf.is_result_fragment());
        assert_eq!(
            leaf.node_types(),
            vec![
                TPlanNodeType::AGGREGATION_NODE,
                TPlanNodeType::FILE_SCAN_NODE
            ]
        );
        // The simplified leaf got the root's shared fields back.
        assert!(leaf.params.desc_tbl.is_some());
        assert_eq!(
            leaf.params.coord,
            Some(TNetworkAddress::new("127.0.0.1".to_string(), 9020))
        );
        assert!(leaf.params.query_globals.is_some());
        assert_eq!(
            leaf.instance_ids().cloned().collect::<Vec<_>>(),
            vec![TUniqueId::new(10, 1)]
        );
        assert_eq!(
            leaf.shape(),
            "fragment_id=1 sink=DATA_STREAM_SINK instances=1 nodes=[AGGREGATION_NODE > FILE_SCAN_NODE]"
        );
    }

    #[test]
    fn rejects_non_version_3_requests() {
        let mut request = compact_request(&two_fragment_list());
        request.version = Some(PFragmentRequestVersion::Version2 as i32);
        let err = decode_fragment_params_list(&request).unwrap_err();
        assert!(err.contains("VERSION_3"), "{err}");
    }

    #[test]
    fn rejects_garbage_payload() {
        let request = PExecPlanFragmentRequest {
            request: Some(b"not thrift".to_vec()),
            compact: Some(true),
            version: Some(PFragmentRequestVersion::Version3 as i32),
        };
        let err = decode_fragment_params_list(&request).unwrap_err();
        assert!(err.contains("failed to deserialize"), "{err}");
    }

    #[test]
    fn rejects_batch_whose_first_fragment_is_simplified() {
        let mut list = two_fragment_list();
        list.params_list.as_mut().unwrap()[0].is_simplified_param = Some(true);
        let err = decode_fragment_params_list(&compact_request(&list)).unwrap_err();
        assert!(err.contains("is_simplified_param"), "{err}");
    }

    #[test]
    fn rejects_mixed_query_ids() {
        let mut list = two_fragment_list();
        list.params_list.as_mut().unwrap()[1].query_id = TUniqueId::new(9, 9);
        let err = decode_fragment_params_list(&compact_request(&list)).unwrap_err();
        assert!(err.contains("query id"), "{err}");
    }

    #[test]
    fn print_id_matches_fe_format() {
        assert_eq!(print_id(&TUniqueId::new(0x1a2b, 0x3c)), "1a2b-3c");
    }
}
