//! PBackendService gRPC handler.
//!
//! This is the main entry point for query execution. The Doris FE sends
//! `exec_plan_fragment` requests containing Thrift-serialized `TPipelineFragmentParams`.

use std::io::Cursor;
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};
use tracing::{info, instrument, warn};

use doris_proto::doris::p_backend_service_server::PBackendService;
use doris_proto::doris::*;
use doris_thrift::data_sinks::TDataSinkType;
use doris_thrift::palo_internal_service::{TPipelineFragmentParams, TPipelineFragmentParamsList};
use doris_thrift::plan_nodes::{TFileFormatType, TPlanNode, TPlanNodeType};

use super::hash_partitioner::{partition_strategy_from_thrift, ExchangeInfo};
use result_formatter::result_store::{FinstId, ResultStore};
use sirius_ffi::SiriusEngine;

use super::exchange_buffer::{ExchangeBuffer, ExchangeKey};
use super::exchange_sender::ExchangeDest;
use super::heartbeat_service::BeState;
use super::pblock_decoder;

fn ok_status() -> PStatus {
    PStatus {
        status_code: 0,
        error_msgs: vec![],
    }
}

fn unimpl() -> Status {
    Status::unimplemented("not supported on Sirius GPU backend")
}

fn err_status(msg: &str) -> PStatus {
    PStatus {
        status_code: 1,
        error_msgs: vec![msg.to_string()],
    }
}

/// Deserialize Thrift TPipelineFragmentParams from raw bytes.
///
/// `version` corresponds to `PFragmentRequestVersion`:
///   1 = single TExecPlanFragmentParams (unsupported)
///   2 = single TPipelineFragmentParams
///   3 = TPipelineFragmentParamsList (shared fields at list level)
fn deserialize_params(data: Vec<u8>, compact: bool, version: i32) -> Result<Vec<TPipelineFragmentParams>, String> {
    use thrift::protocol::{TBinaryInputProtocol, TCompactInputProtocol, TSerializable};
    use thrift::transport::TBufferedReadTransport;

    if version == 3 {
        // VERSION_3: bytes contain TPipelineFragmentParamsList.
        // Shared fields (desc_tbl, query_globals, etc.) are at the list level.
        // The params_list may contain MULTIPLE fragments (scan + exchange, etc.).
        let list = if compact {
            let transport = TBufferedReadTransport::new(Box::new(Cursor::new(data)));
            let mut protocol = TCompactInputProtocol::new(transport);
            TPipelineFragmentParamsList::read_from_in_protocol(&mut protocol)
                .map_err(|e| format!("compact thrift deserialize (v3 list): {e}"))?
        } else {
            let transport = TBufferedReadTransport::new(Box::new(Cursor::new(data)));
            let mut protocol = TBinaryInputProtocol::new(transport, true);
            TPipelineFragmentParamsList::read_from_in_protocol(&mut protocol)
                .map_err(|e| format!("binary thrift deserialize (v3 list): {e}"))?
        };

        let params_list = list
            .params_list
            .filter(|v| !v.is_empty())
            .ok_or_else(|| "VERSION_3: empty params_list".to_string())?;

        // Merge shared fields into each per-fragment params.
        // Fields come from two sources (in priority order):
        //   1. List-level fields (TPipelineFragmentParamsList top-level)
        //   2. First fragment in the list (when is_simplified_param is used)
        // The FE may put shared fields at the list level OR in the first fragment.
        let mut all_params: Vec<TPipelineFragmentParams> = params_list;

        // Extract shared fields: prefer list-level, fall back to first fragment.
        let shared_desc_tbl = list.desc_tbl.clone()
            .or_else(|| all_params.first().and_then(|p| p.desc_tbl.clone()));
        let shared_query_globals = list.query_globals.clone()
            .or_else(|| all_params.first().and_then(|p| p.query_globals.clone()));
        let shared_query_options = list.query_options.clone()
            .or_else(|| all_params.first().and_then(|p| p.query_options.clone()));
        let shared_coord = list.coord.clone()
            .or_else(|| all_params.first().and_then(|p| p.coord.clone()));
        let shared_resource_info = list.resource_info.clone()
            .or_else(|| all_params.first().and_then(|p| p.resource_info.clone()));
        let shared_fragment_num = list.fragment_num_on_host
            .or_else(|| all_params.first().and_then(|p| p.fragment_num_on_host));
        let shared_file_scan_params = list.file_scan_params.clone()
            .or_else(|| all_params.first().and_then(|p| p.file_scan_params.clone()));

        for params in &mut all_params {
            if params.desc_tbl.is_none() {
                params.desc_tbl = shared_desc_tbl.clone();
            }
            if params.query_globals.is_none() {
                params.query_globals = shared_query_globals.clone();
            }
            if params.query_options.is_none() {
                params.query_options = shared_query_options.clone();
            }
            if params.coord.is_none() {
                params.coord = shared_coord.clone();
            }
            if params.resource_info.is_none() {
                params.resource_info = shared_resource_info.clone();
            }
            if params.fragment_num_on_host.is_none() {
                params.fragment_num_on_host = shared_fragment_num;
            }
            if params.file_scan_params.is_none() {
                params.file_scan_params = shared_file_scan_params.clone();
            }
        }

        Ok(all_params)
    } else if version == 2 || version == 0 {
        // VERSION_2 (or default): single TPipelineFragmentParams.
        let params = if compact {
            let transport = TBufferedReadTransport::new(Box::new(Cursor::new(data)));
            let mut protocol = TCompactInputProtocol::new(transport);
            TPipelineFragmentParams::read_from_in_protocol(&mut protocol)
                .map_err(|e| format!("compact thrift deserialize: {e}"))?
        } else {
            let transport = TBufferedReadTransport::new(Box::new(Cursor::new(data)));
            let mut protocol = TBinaryInputProtocol::new(transport, true);
            TPipelineFragmentParams::read_from_in_protocol(&mut protocol)
                .map_err(|e| format!("binary thrift deserialize: {e}"))?
        };
        Ok(vec![params])
    } else {
        Err(format!("unsupported PFragmentRequestVersion: {version}"))
    }
}

/// Serialize MySQL rows as a Thrift binary-encoded TResultBatch.
///
/// The Doris FE deserializes row_batch using TBinaryProtocol (see ResultReceiver.java),
/// NOT TCompactProtocol.
fn serialize_result_batch(rows: &[Vec<u8>], packet_seq: i64) -> Result<Vec<u8>, String> {
    use thrift::protocol::{TBinaryOutputProtocol, TOutputProtocol, TSerializable};

    let batch = doris_thrift::data::TResultBatch::new(
        rows.to_vec(),
        false, // is_compressed
        packet_seq,
        None::<std::collections::BTreeMap<String, String>>,
    );

    let mut buf = Vec::new();
    {
        let mut protocol = TBinaryOutputProtocol::new(Cursor::new(&mut buf), true);
        batch
            .write_to_out_protocol(&mut protocol)
            .map_err(|e| format!("thrift serialize TResultBatch: {e}"))?;
        protocol
            .flush()
            .map_err(|e| format!("thrift flush: {e}"))?;
    }
    Ok(buf)
}

/// Extract FinstId for result storage from fragment params.
///
/// The FE uses the result-sink fragment's `fragment_instance_id` (from
/// `local_params[0]`) to call `fetch_data`, not the query-level `query_id`.
/// Fall back to `query_id` if `local_params` is empty.
/// Find the finst_id for the result-sink fragment (fragment_id=0).
///
/// Before merging, we need to identify which fragment's finst_id the FE will
/// use for fetch_data. This is always the result-sink fragment (fragment_id=0),
/// which has an EXCHANGE_NODE at the root with 0 children. After merge, the
/// merged fragment may have a different local_params[0] so we must capture
/// the correct finst_id before merge.
/// Map an Arrow DataType to a Doris PTypeDesc.
///
/// TPrimitiveType values (from Doris thrift Types.thrift):
///   BOOLEAN=2, TINYINT=3, SMALLINT=4, INT=5, BIGINT=6, FLOAT=7, DOUBLE=8,
///   DATE=10, DATETIME=11, VARCHAR=16, DECIMALV2=12, LARGEINT=15,
///   CHAR=17, DATEV2=28, DATETIMEV2=29, DECIMAL32=47, DECIMAL64=48, DECIMAL128I=49
fn arrow_type_to_doris(dt: &arrow::datatypes::DataType) -> PTypeDesc {
    // TPrimitiveType values from Types.thrift:
    // BOOLEAN=2, TINYINT=3, SMALLINT=4, INT=5, BIGINT=6, FLOAT=7, DOUBLE=8,
    // DATE=9, DATETIME=10, CHAR=13, VARCHAR=15, STRING=23,
    // DATEV2=26, DATETIMEV2=27, DECIMAL32=29, DECIMAL64=30, DECIMAL128I=31
    use arrow::datatypes::DataType;
    let scalar = match dt {
        DataType::Boolean => PScalarType { r#type: 2, len: None, precision: None, scale: None },
        DataType::Int8 => PScalarType { r#type: 3, len: None, precision: None, scale: None },
        DataType::Int16 => PScalarType { r#type: 4, len: None, precision: None, scale: None },
        DataType::Int32 => PScalarType { r#type: 5, len: None, precision: None, scale: None },
        DataType::Int64 => PScalarType { r#type: 6, len: None, precision: None, scale: None },
        DataType::UInt8 => PScalarType { r#type: 3, len: None, precision: None, scale: None },
        DataType::UInt16 => PScalarType { r#type: 4, len: None, precision: None, scale: None },
        DataType::UInt32 => PScalarType { r#type: 5, len: None, precision: None, scale: None },
        DataType::UInt64 => PScalarType { r#type: 6, len: None, precision: None, scale: None },
        DataType::Float16 | DataType::Float32 => PScalarType { r#type: 7, len: None, precision: None, scale: None },
        DataType::Float64 => PScalarType { r#type: 8, len: None, precision: None, scale: None },
        DataType::Date32 | DataType::Date64 => PScalarType { r#type: 26, len: None, precision: None, scale: None }, // DATEV2
        DataType::Timestamp(_, _) => PScalarType { r#type: 27, len: None, precision: None, scale: None }, // DATETIMEV2
        DataType::Utf8 | DataType::LargeUtf8 => PScalarType { r#type: 23, len: Some(65533), precision: None, scale: None }, // STRING
        DataType::Decimal128(p, s) => {
            let p = *p as i32;
            let s = *s as i32;
            if p <= 9 {
                PScalarType { r#type: 29, len: None, precision: Some(p), scale: Some(s) } // DECIMAL32
            } else if p <= 18 {
                PScalarType { r#type: 30, len: None, precision: Some(p), scale: Some(s) } // DECIMAL64
            } else {
                PScalarType { r#type: 31, len: None, precision: Some(p), scale: Some(s) } // DECIMAL128I
            }
        }
        DataType::Binary | DataType::LargeBinary => PScalarType { r#type: 23, len: Some(65533), precision: None, scale: None }, // STRING (binary as string)
        DataType::Time32(_) | DataType::Time64(_) => PScalarType { r#type: 23, len: Some(65533), precision: None, scale: None }, // STRING (no native Doris TIME)
        DataType::Duration(_) | DataType::Interval(_) => PScalarType { r#type: 23, len: Some(65533), precision: None, scale: None }, // STRING
        DataType::List(_) | DataType::LargeList(_) => PScalarType { r#type: 23, len: Some(65533), precision: None, scale: None }, // STRING (array as string)
        DataType::Struct(_) => PScalarType { r#type: 23, len: Some(65533), precision: None, scale: None }, // STRING (struct as string)
        other => {
            warn!(arrow_type = ?other, "arrow_type_to_doris: unmapped type, falling back to STRING");
            PScalarType { r#type: 23, len: Some(65533), precision: None, scale: None }
        }
    };
    PTypeDesc {
        types: vec![PTypeNode {
            r#type: 0, // SCALAR
            scalar_type: Some(scalar),
            struct_fields: vec![],
            contains_null: None,
            contains_nulls: vec![],
            variant_max_subcolumns_count: None,
        }],
    }
}

/// Extract file scan info from fragment params (FILE_SCAN_NODE → all file paths + format).
///
/// Walks the plan nodes looking for FILE_SCAN_NODE, then extracts ALL file paths
/// from `local_params[*].per_node_scan_ranges[node_id][*].scan_range...ranges[*]`
/// and the format from `file_scan_params`. Returns one `FileScanInfo` per scan node,
/// containing all files assigned to this BE.
fn extract_file_scan_info(params: &TPipelineFragmentParams) -> Vec<plan_translator::FileScanInfo> {
    let mut result = Vec::new();

    let fragment = match &params.fragment {
        Some(f) => f,
        None => return result,
    };
    let plan = match &fragment.plan {
        Some(p) => p,
        None => return result,
    };

    for node in &plan.nodes {
        if node.node_type != TPlanNodeType::FILE_SCAN_NODE {
            continue;
        }
        let node_id = node.node_id;

        // Table name from TFileScanNode.table_name or fallback.
        // Append node_id to make each scan's table unique (needed for self-joins where
        // multiple scans reference the same file — DuckDB can't handle duplicate column
        // names in Substrait JoinRel output from same-named tables).
        let table_name = node
            .file_scan_node
            .as_ref()
            .and_then(|fsn| fsn.table_name.clone())
            .map(|name| format!("{}_{}", name, node_id))
            .unwrap_or_else(|| format!("scan_{}", node_id));

        // Format from file_scan_params[node_id].format_type.
        let format = params
            .file_scan_params
            .as_ref()
            .and_then(|m| m.get(&node_id))
            .and_then(|p| p.format_type.as_ref())
            .map(|ft| match *ft {
                TFileFormatType::FORMAT_PARQUET => "parquet",
                TFileFormatType::FORMAT_ORC => "orc",
                TFileFormatType::FORMAT_JSON => "json",
                TFileFormatType::FORMAT_CSV_PLAIN => "csv",
                _ => "parquet", // default fallback
            })
            .unwrap_or("parquet")
            .to_string();

        // Extract ALL file paths from all local_params and all scan ranges.
        // With shared_storage=true, FE distributes multiple partition files per BE.
        let mut files = Vec::new();
        if let Some(local_params) = &params.local_params {
            for inst in local_params {
                if let Some(scan_range_list) = inst.per_node_scan_ranges.get(&node_id) {
                    for scan_range_params in scan_range_list {
                        if let Some(file_scan_range) = scan_range_params
                            .scan_range
                            .ext_scan_range
                            .as_ref()
                            .and_then(|esr| esr.file_scan_range.as_ref())
                        {
                            if let Some(ranges) = &file_scan_range.ranges {
                                for range in ranges {
                                    if let Some(path) = &range.path {
                                        let clean_path = path
                                            .strip_prefix("file://")
                                            .unwrap_or(path)
                                            .to_string();
                                        files.push(plan_translator::FileScanFile {
                                            path: clean_path,
                                            start_offset: range.start_offset.unwrap_or(0),
                                            length: range.size.unwrap_or(-1),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Deduplicate files by path. FE may send multiple scan ranges for the
        // same file (intra-file parallelism). We only need each file once for
        // DuckDB's parquet_scan which reads entire files.
        let mut seen_paths = std::collections::HashSet::new();
        files.retain(|f| seen_paths.insert(f.path.clone()));

        if !files.is_empty() {
            info!(
                table = %table_name,
                format = %format,
                num_files = files.len(),
                first_file = %files[0].path,
                "extracted file scan info"
            );
            result.push(plan_translator::FileScanInfo {
                table_name,
                format,
                files,
            });
        } else {
            let all_keys: Vec<Vec<i32>> = params
                .local_params
                .as_ref()
                .map(|lps| {
                    lps.iter()
                        .map(|inst| inst.per_node_scan_ranges.keys().copied().collect())
                        .collect()
                })
                .unwrap_or_default();
            tracing::debug!(
                node_id,
                has_local_params = params.local_params.is_some(),
                local_params_len = params.local_params.as_ref().map(|l| l.len()),
                all_per_node_keys = ?all_keys,
                "FILE_SCAN_NODE found but no file path extracted"
            );
        }
    }

    result
}

/// Merge multi-fragment plans for single-BE execution.
///
/// In a multi-fragment plan (e.g. ORDER BY + LIMIT), the FE sends:
///   - Fragment A: EXCHANGE_NODE(0 children)  — result collector
///   - Fragment B: SORT_NODE → EXCHANGE_NODE(0 children)  — intermediate processing
///   - Fragment C: FILE_SCAN_NODE  — leaf scan
///
/// For single-BE execution, we merge by:
///   1. Skipping result-collector fragments (EXCHANGE at root, 0 children)
///   2. Finding leaf fragments (scans) and intermediate fragments (with EXCHANGE children)
///   3. Replacing EXCHANGE_NODE(0 children) in intermediate plans with the leaf plan
///
/// This produces a single merged fragment that can be translated and executed atomically.
fn merge_fragment_plans(
    all_params: &[TPipelineFragmentParams],
) -> Vec<TPipelineFragmentParams> {
    // Classify fragments.
    let mut leaf_fragments: Vec<&TPipelineFragmentParams> = Vec::new();
    let mut intermediate_fragments: Vec<TPipelineFragmentParams> = Vec::new();
    let mut exchange_root_fragments: Vec<TPipelineFragmentParams> = Vec::new();

    for params in all_params {
        let plan = match params.fragment.as_ref().and_then(|f| f.plan.as_ref()) {
            Some(p) => p,
            None => continue,
        };
        let root = match plan.nodes.first() {
            Some(r) => r,
            None => continue,
        };

        // Result-collector fragments (EXCHANGE at root, 0 children).
        if root.node_type == TPlanNodeType::EXCHANGE_NODE && root.num_children == 0 {
            exchange_root_fragments.push(params.clone());
            continue;
        }

        // Check if this fragment has any EXCHANGE_NODE(0 children) as a non-root node.
        let has_exchange_child = plan.nodes.iter().skip(1).any(|n| {
            n.node_type == TPlanNodeType::EXCHANGE_NODE && n.num_children == 0
        });

        if has_exchange_child {
            intermediate_fragments.push(params.clone());
        } else {
            leaf_fragments.push(params);
        }
    }

    // Log classification details for each fragment.
    for (i, params) in all_params.iter().enumerate() {
        let plan = params.fragment.as_ref().and_then(|f| f.plan.as_ref());
        let node_types: Vec<_> = plan
            .map(|p| p.nodes.iter().map(|n| (n.node_id, n.node_type, n.num_children)).collect())
            .unwrap_or_default();
        let dest_node_id = params
            .fragment
            .as_ref()
            .and_then(|f| f.output_sink.as_ref())
            .filter(|s| s.type_ == TDataSinkType::DATA_STREAM_SINK)
            .and_then(|s| s.stream_sink.as_ref())
            .map(|ss| ss.dest_node_id);
        let sink_type = params
            .fragment
            .as_ref()
            .and_then(|f| f.output_sink.as_ref())
            .map(|s| format!("{:?}", s.type_));
        info!(
            fragment_idx = i,
            fragment_id = ?params.fragment_id,
            ?node_types,
            ?dest_node_id,
            ?sink_type,
            per_exch = ?params.per_exch_num_senders,
            "merge_fragment_plans: fragment detail"
        );
    }

    info!(
        num_exchange_root = exchange_root_fragments.len(),
        num_intermediate = intermediate_fragments.len(),
        num_leaf = leaf_fragments.len(),
        "merge_fragment_plans: classified fragments"
    );

    // If there are no leaf fragments to merge with, the exchange-root fragments
    // are not simple result collectors — they depend on exchange data from
    // remote BEs. Return them so exec_plan_fragment can handle them via
    // the async exchange path.
    if intermediate_fragments.is_empty() && leaf_fragments.is_empty() {
        return exchange_root_fragments;
    }

    // Build set of exchange node IDs that ANY local fragment can provide data for.
    // Each fragment's output_sink.stream_sink.dest_node_id tells us which EXCHANGE_NODE
    // it feeds. We include leaves AND intermediates because intermediate-to-intermediate
    // connections are also local (e.g. subquery fragment → main join fragment).
    // EXCHANGE nodes not in this set depend on remote BEs.
    let extract_dest_id = |p: &TPipelineFragmentParams| {
        p.fragment
            .as_ref()
            .and_then(|f| f.output_sink.as_ref())
            .filter(|s| s.type_ == TDataSinkType::DATA_STREAM_SINK)
            .and_then(|s| s.stream_sink.as_ref())
            .map(|ss| ss.dest_node_id)
    };
    let leaf_dest_ids: std::collections::HashSet<i32> = leaf_fragments
        .iter()
        .filter_map(|p| extract_dest_id(p))
        .chain(intermediate_fragments.iter().filter_map(|p| extract_dest_id(p)))
        .chain(exchange_root_fragments.iter().filter_map(|p| extract_dest_id(p)))
        .collect();

    // Check if any fragment has EXCHANGE nodes that can't be satisfied by local
    // fragments. This catches the case where a fragment depends on data from a
    // remote BE (its EXCHANGE node ID not in any local fragment's dest_node_id).
    let has_remote_exchanges = |params: &TPipelineFragmentParams| -> bool {
        let plan = match params.fragment.as_ref().and_then(|f| f.plan.as_ref()) {
            Some(p) => p,
            None => return false,
        };
        let exchange_count = plan.nodes.iter().filter(|n| {
            n.node_type == TPlanNodeType::EXCHANGE_NODE && n.num_children == 0
        }).count();

        // When leaves have proper dest_node_ids, check if every EXCHANGE node
        // has a matching leaf. An EXCHANGE without a matching leaf depends on
        // remote data. When leaves lack dest_node_ids (e.g. single leaf, no
        // output sink), fall back to count-based check.
        let has_unmatched = if !leaf_dest_ids.is_empty() {
            plan.nodes.iter().any(|n| {
                n.node_type == TPlanNodeType::EXCHANGE_NODE
                    && n.num_children == 0
                    && !leaf_dest_ids.contains(&n.node_id)
            })
        } else {
            exchange_count > leaf_fragments.len()
        };

        // Also check per-exchange sender counts: if any exchange expects more
        // than 1 sender, it receives from multiple BEs and can't be merged
        // (we can only provide one local leaf's data).
        let any_multi_sender = plan.nodes.iter().any(|n| {
            n.node_type == TPlanNodeType::EXCHANGE_NODE
                && n.num_children == 0
                && get_num_senders(params, n.node_id) > 1
        });

        if has_unmatched || any_multi_sender {
            info!(
                ?leaf_dest_ids,
                exchange_count,
                num_leaves = leaf_fragments.len(),
                per_exch = ?params.per_exch_num_senders,
                "fragment has remote exchanges, skipping merge"
            );
            return true;
        }
        false
    };

    let any_remote = exchange_root_fragments.iter().any(|p| has_remote_exchanges(p))
        || intermediate_fragments.iter().any(|p| has_remote_exchanges(p));

    if any_remote {
        info!("merge_fragment_plans: remote exchanges detected, returning all fragments separately");
        // Return all fragments separately:
        // - Exchange-root/intermediate: async exchange path (wait for senders)
        // - Leaf fragments: execute scan, send results via bRPC to exchange destinations
        let mut result = Vec::new();
        result.extend(exchange_root_fragments);
        result.extend(intermediate_fragments);
        result.extend(leaf_fragments.into_iter().cloned());
        return result;
    }

    // If no intermediate fragments but we have exchange_root + leaf, treat the
    // exchange root as intermediate. The FE's fetch_data uses the exchange root's
    // local_params[0].fragment_instance_id, so we must preserve it as the result key.
    if intermediate_fragments.is_empty() && !exchange_root_fragments.is_empty() {
        intermediate_fragments = exchange_root_fragments;
        exchange_root_fragments = Vec::new();
    }

    // If no intermediate fragments (and no exchange root), just return the leaf fragments.
    if intermediate_fragments.is_empty() {
        return leaf_fragments.into_iter().cloned().collect();
    }

    // Build a map from dest_node_id → leaf fragment, so each EXCHANGE_NODE in the
    // intermediate fragment gets replaced with its specific leaf's plan (not just the first).
    // This is critical for JOIN queries where each side scans a different table.
    let mut leaf_by_dest: std::collections::HashMap<i32, &TPipelineFragmentParams> =
        std::collections::HashMap::new();
    for leaf in &leaf_fragments {
        if let Some(dest_node_id) = leaf
            .fragment
            .as_ref()
            .and_then(|f| f.output_sink.as_ref())
            .filter(|s| s.type_ == TDataSinkType::DATA_STREAM_SINK)
            .and_then(|s| s.stream_sink.as_ref())
            .map(|ss| ss.dest_node_id)
        {
            leaf_by_dest.insert(dest_node_id, leaf);
        }
    }

    // Fallback: if no dest_node_id mapping (single leaf, no output sink), use first leaf for all.
    let single_leaf = leaf_fragments.first().copied();

    // Collect all leaf scan params and local params for merging into intermediates.
    let mut merged_leaf_scan_params = leaf_fragments
        .first()
        .and_then(|p| p.file_scan_params.clone());
    for leaf in leaf_fragments.iter().skip(1) {
        if let Some(scan) = &leaf.file_scan_params {
            let merged = merged_leaf_scan_params.get_or_insert_with(Default::default);
            for (k, v) in scan {
                merged.entry(*k).or_insert_with(|| v.clone());
            }
        }
    }
    let mut merged_leaf_local_params = leaf_fragments
        .first()
        .and_then(|p| p.local_params.clone());
    for leaf in leaf_fragments.iter().skip(1) {
        if let Some(local) = &leaf.local_params {
            let merged = merged_leaf_local_params.get_or_insert_with(Vec::new);
            merged.extend(local.iter().cloned());
        }
    }

    let mut current_scan_params = merged_leaf_scan_params;
    let mut current_local_params = merged_leaf_local_params;

    // Topological cascade merge: process intermediates from innermost (depends on
    // leaves) to outermost (depends on other intermediates), building up merged plan
    // nodes at each level.
    //
    // Each fragment's output_sink.stream_sink.dest_node_id tells us which EXCHANGE_NODE
    // in a parent fragment it feeds. We use this to determine dependencies:
    // - An intermediate whose EXCHANGE_NODE IDs are all in leaf_by_dest is innermost.
    // - After merging an intermediate, its dest_node_id becomes a resolved source
    //   for the next level.
    //
    // resolved_sources maps EXCHANGE_NODE node_id → merged plan nodes for that source.
    let mut resolved_sources: std::collections::HashMap<i32, Vec<TPlanNode>> =
        std::collections::HashMap::new();

    // Seed resolved sources from leaves.
    for leaf in &leaf_fragments {
        if let Some(dest_node_id) = leaf
            .fragment
            .as_ref()
            .and_then(|f| f.output_sink.as_ref())
            .filter(|s| s.type_ == TDataSinkType::DATA_STREAM_SINK)
            .and_then(|s| s.stream_sink.as_ref())
            .map(|ss| ss.dest_node_id)
        {
            if let Some(nodes) = leaf
                .fragment
                .as_ref()
                .and_then(|f| f.plan.as_ref())
                .map(|p| p.nodes.clone())
            {
                resolved_sources.insert(dest_node_id, nodes);
            }
        }
    }

    // Process intermediates in topological order: each pass resolves intermediates
    // whose EXCHANGE_NODE IDs are all in resolved_sources. Continue until all are
    // processed or no more can be resolved (cycle or missing dependency).
    let mut remaining = intermediate_fragments;
    let mut ordered: Vec<TPipelineFragmentParams> = Vec::new();

    loop {
        let before = remaining.len();
        let mut still_remaining = Vec::new();
        for mut params in remaining {
            // Check if all EXCHANGE_NODE IDs in this fragment can be resolved.
            let exchange_ids: Vec<i32> = params
                .fragment
                .as_ref()
                .and_then(|f| f.plan.as_ref())
                .map(|p| {
                    p.nodes
                        .iter()
                        .filter(|n| {
                            n.node_type == TPlanNodeType::EXCHANGE_NODE && n.num_children == 0
                        })
                        .map(|n| n.node_id)
                        .collect()
                })
                .unwrap_or_default();

            let all_resolved = exchange_ids
                .iter()
                .all(|id| resolved_sources.contains_key(id));

            // Allow merge when:
            // - All exchange IDs resolved via resolved_sources, OR
            // - No exchange IDs to resolve, OR
            // - No dest_node_id mapping exists (leaf_dest_ids empty) but we have
            //   a single_leaf fallback (legacy: simple pipelines without output_sink)
            if all_resolved || exchange_ids.is_empty()
                || (leaf_dest_ids.is_empty() && single_leaf.is_some()) {
                // Merge: replace EXCHANGE_NODE(0 children) with resolved source nodes.
                if let Some(plan) = params
                    .fragment
                    .as_mut()
                    .and_then(|f| f.plan.as_mut())
                {
                    let mut merged_nodes = Vec::new();
                    for node in &plan.nodes {
                        if node.node_type == TPlanNodeType::EXCHANGE_NODE
                            && node.num_children == 0
                        {
                            if let Some(source) = resolved_sources.get(&node.node_id) {
                                merged_nodes.extend_from_slice(source);
                            } else if let Some(leaf_nodes) = single_leaf
                                .and_then(|l| l.fragment.as_ref())
                                .and_then(|f| f.plan.as_ref())
                                .map(|p| &p.nodes)
                            {
                                merged_nodes.extend_from_slice(leaf_nodes);
                            } else {
                                merged_nodes.push(node.clone());
                            }
                        } else {
                            merged_nodes.push(node.clone());
                        }
                    }
                    plan.nodes = merged_nodes;
                }

                // Register this fragment's merged output as a resolved source.
                if let Some(dest_node_id) = params
                    .fragment
                    .as_ref()
                    .and_then(|f| f.output_sink.as_ref())
                    .filter(|s| s.type_ == TDataSinkType::DATA_STREAM_SINK)
                    .and_then(|s| s.stream_sink.as_ref())
                    .map(|ss| ss.dest_node_id)
                {
                    if let Some(nodes) = params
                        .fragment
                        .as_ref()
                        .and_then(|f| f.plan.as_ref())
                        .map(|p| p.nodes.clone())
                    {
                        resolved_sources.insert(dest_node_id, nodes);
                    }
                }

                // Merge scan params and local params from inner levels.
                if let Some(inner_scan) = &current_scan_params {
                    let merged_scan = params
                        .file_scan_params
                        .get_or_insert_with(Default::default);
                    for (k, v) in inner_scan {
                        merged_scan.entry(*k).or_insert_with(|| v.clone());
                    }
                }
                current_scan_params = params.file_scan_params.clone();

                if let Some(inner_local) = &current_local_params {
                    if params.local_params.is_none() {
                        params.local_params = Some(inner_local.clone());
                    } else {
                        let merged_local = params.local_params.get_or_insert_with(Vec::new);
                        for lp in inner_local {
                            merged_local.push(lp.clone());
                        }
                    }
                }
                current_local_params = params.local_params.clone();

                ordered.push(params);
            } else {
                still_remaining.push(params);
            }
        }
        remaining = still_remaining;
        if remaining.is_empty() || remaining.len() == before {
            // All resolved, or stuck (no progress) — break.
            break;
        }
    }
    // Append any unresolved intermediates at the end (shouldn't happen in valid plans).
    ordered.extend(remaining);
    intermediate_fragments = ordered;

    // Keep only the outermost intermediate (last in topological order).
    // All inner intermediates have been merged into it via the cascade.
    if intermediate_fragments.len() > 1 {
        let outermost = intermediate_fragments.pop().unwrap();
        intermediate_fragments = vec![outermost];
    }

    // If the cascade produced a single fragment with no exchange_root to consume it,
    // clear the output_sink so the result is stored locally (not sent via exchange).
    if exchange_root_fragments.is_empty() && intermediate_fragments.len() == 1 {
        if let Some(fragment) = intermediate_fragments[0].fragment.as_mut() {
            if fragment.output_sink.as_ref()
                .map(|s| s.type_ == TDataSinkType::DATA_STREAM_SINK)
                .unwrap_or(false)
            {
                fragment.output_sink = None;
            }
        }
    }

    // Second pass: if there are exchange_root fragments remaining, merge them with
    // the now-merged intermediates. The exchange_root is a pure EXCHANGE_NODE(0 children)
    // that collects results. Replace its plan with the merged intermediate's plan.
    // Preserve the exchange_root's local_params (FE uses its fragment_instance_id for fetch_data).
    if !exchange_root_fragments.is_empty() && !intermediate_fragments.is_empty() {
        let outermost = intermediate_fragments.last().unwrap();
        let intermediate_nodes: Vec<_> = outermost
            .fragment
            .as_ref()
            .and_then(|f| f.plan.as_ref())
            .map(|p| p.nodes.clone())
            .unwrap_or_default();

        for params in &mut exchange_root_fragments {
            if let Some(plan) = params
                .fragment
                .as_mut()
                .and_then(|f| f.plan.as_mut())
            {
                // Replace the root EXCHANGE_NODE with the merged intermediate's plan.
                let mut merged_nodes = Vec::new();
                for node in &plan.nodes {
                    if node.node_type == TPlanNodeType::EXCHANGE_NODE && node.num_children == 0 {
                        merged_nodes.extend_from_slice(&intermediate_nodes);
                    } else {
                        merged_nodes.push(node.clone());
                    }
                }
                plan.nodes = merged_nodes;
            }

            // Copy file_scan_params and additional local_params from outermost intermediate.
            if let Some(inter_scan) = &outermost.file_scan_params {
                let merged_scan = params
                    .file_scan_params
                    .get_or_insert_with(Default::default);
                for (k, v) in inter_scan {
                    merged_scan.entry(*k).or_insert_with(|| v.clone());
                }
            }
            // Append intermediate's local_params (scan ranges) but keep
            // exchange_root's local_params[0] first (has the result instance_id).
            if let Some(inter_local) = &outermost.local_params {
                let merged_local = params.local_params.get_or_insert_with(Vec::new);
                for lp in inter_local {
                    merged_local.push(lp.clone());
                }
            }
        }

        return exchange_root_fragments;
    }

    intermediate_fragments
}

/// Execution plan: either Substrait bytes or SQL string.
enum ExecPlan {
    /// Substrait plan eligible for GPU acceleration (has real data tables).
    Substrait {
        bytes: Vec<u8>,
        sort_limit_sql: Option<String>,
    },
    /// Substrait plan that should only run on CPU (e.g. VirtualTable-only plans).
    SubstraitCpuOnly {
        bytes: Vec<u8>,
        sort_limit_sql: Option<String>,
    },
    /// SQL that should only run on CPU (e.g. exchange fragments reading pre-computed data).
    /// Skips gpu_execution() which can cause INTERNAL errors on non-GPU tables.
    SqlCpuOnly(String),
}

/// Execute a plan via Sirius GPU, falling back to DuckDB CPU.
fn execute_plan(engine: &SiriusEngine, plan: ExecPlan, no_cpu_fallback: bool, force_cpu: bool) -> Result<Vec<u8>, String> {
    // When force_cpu is set, downgrade GPU plans to CPU-only equivalents.
    if force_cpu {
        let cpu_plan = match plan {
            ExecPlan::Substrait { bytes, sort_limit_sql } =>
                ExecPlan::SubstraitCpuOnly { bytes, sort_limit_sql },
            other => other, // already CPU-only
        };
        return execute_plan(engine, cpu_plan, no_cpu_fallback, false);
    }
    match plan {
        ExecPlan::Substrait { bytes, sort_limit_sql } => {
            // Strip SortRel/FetchRel from Substrait before GPU execution — the Sirius
            // GPU planner doesn't handle ORDER BY/LIMIT operators. The sort/limit SQL
            // wrapper is applied outside via `execute_substrait`.
            let gpu_bytes = if sort_limit_sql.is_some() {
                strip_sort_limit_from_substrait(&bytes)
            } else {
                None
            };
            let plan_bytes = gpu_bytes.as_deref().unwrap_or(&bytes);
            let t0 = std::time::Instant::now();
            tracing::info!(substrait_bytes = plan_bytes.len(), "gpu_execution_substrait starting");
            match engine.execute_substrait(plan_bytes, sort_limit_sql.as_deref()) {
                Ok(ipc) => {
                    tracing::info!(elapsed_ms = t0.elapsed().as_millis() as u64, ipc_len = ipc.len(), "executed via gpu_execution_substrait");
                    Ok(ipc)
                }
                Err(e) if no_cpu_fallback => {
                    tracing::error!(error = %e, "gpu_execution_substrait failed (no CPU fallback)");
                    Err(format!("GPU execution failed: {e}"))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "gpu_execution_substrait failed, falling back to CPU from_substrait");
                    from_substrait_with_sort(engine, &bytes, sort_limit_sql.as_deref())
                }
            }
        }
        ExecPlan::SubstraitCpuOnly { bytes, sort_limit_sql } => {
            tracing::info!("executing via CPU from_substrait (no data tables, GPU skipped)");
            from_substrait_with_sort(engine, &bytes, sort_limit_sql.as_deref())
        }
        ExecPlan::SqlCpuOnly(sql) => {
            tracing::info!(sql = %sql, "executing via CPU SQL (GPU skipped)");
            engine.execute_sql(&sql).map_err(|e| e.to_string())
        }
    }
}

/// Execute from_substrait with optional sort/limit SQL suffix.
///
/// When sort_limit_sql is provided, strips the SortRel/FetchRel from the Substrait plan
/// first — otherwise from_substrait applies the LIMIT to unsorted data, yielding wrong rows.
/// The SQL wrapper then applies the correct ORDER BY + LIMIT on the full result.
fn from_substrait_with_sort(
    engine: &SiriusEngine,
    bytes: &[u8],
    sort_limit_sql: Option<&str>,
) -> Result<Vec<u8>, String> {
    if let Some(sql) = sort_limit_sql {
        tracing::info!(sort_limit = %sql, "applying sort/limit wrapper to from_substrait");
        let stripped = strip_sort_limit_from_substrait(bytes);
        let plan_bytes = stripped.as_deref().unwrap_or(bytes);
        engine.from_substrait_sorted(plan_bytes, sql).map_err(|e| e.to_string())
    } else {
        engine.from_substrait(bytes).map_err(|e| e.to_string())
    }
}

/// Strip SortRel and FetchRel from the outermost relation in a Substrait plan.
///
/// Returns Some(new_bytes) if stripping succeeded, None if the plan doesn't have
/// a sort/fetch at the root (or deserialization failed).
fn strip_sort_limit_from_substrait(bytes: &[u8]) -> Option<Vec<u8>> {
    use prost::Message;
    use substrait::proto::{plan_rel, Plan};

    let mut plan = Plan::decode(bytes).ok()?;
    let relation = plan.relations.first_mut()?;
    let plan_rel::RelType::Root(root) = relation.rel_type.as_mut()? else {
        return None;
    };

    // Strip SortRel/FetchRel from the top of the Rel tree.
    // The tree may have a ProjectRel wrapper (for column selection/reorder),
    // so we look both at root.input directly and inside a ProjectRel.
    strip_sort_from_rel(root.input.as_mut()?)?;
    Some(plan.encode_to_vec())
}

/// Strip SortRel/FetchRel from a Rel, handling ProjectRel wrappers.
fn strip_sort_from_rel(rel: &mut substrait::proto::Rel) -> Option<()> {
    use substrait::proto::rel;

    match rel.rel_type.as_mut()? {
        rel::RelType::Fetch(fetch) => {
            let input = fetch.input.as_ref()?;
            let inner = match input.rel_type.as_ref()? {
                rel::RelType::Sort(sort) => sort.input.as_ref()?.as_ref().clone(),
                _ => input.as_ref().clone(),
            };
            rel.rel_type = Some(inner.rel_type?);
            Some(())
        }
        rel::RelType::Sort(sort) => {
            let inner = sort.input.as_ref()?.as_ref().clone();
            rel.rel_type = Some(inner.rel_type?);
            Some(())
        }
        rel::RelType::Project(project) => {
            // ProjectRel wraps a SortRel — strip the sort from the ProjectRel's input.
            strip_sort_from_rel(project.input.as_mut()?)
        }
        _ => None,
    }
}

/// Project/reorder Arrow IPC result columns to match FE expectations.
///
/// When `explicit_indices` is Some, use those to reorder/project columns by position.
/// Otherwise, use `output_names` to find columns by name in the DuckDB result.
fn project_ipc_columns(
    ipc_bytes: &[u8],
    output_names: &[String],
    explicit_indices: Option<&[usize]>,
) -> Result<Vec<u8>, String> {
    use arrow::ipc::reader::StreamReader;
    use arrow::ipc::writer::StreamWriter;

    if ipc_bytes.is_empty() {
        return Err("parse IPC for projection: empty IPC bytes (no results)".to_string());
    }
    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None)
        .map_err(|e| format!("parse IPC for projection: {e}"))?;
    let schema = reader.schema();
    let schema_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    // Diagnostic: dump first row values to verify column-name-to-data correspondence.
    if schema.fields().len() > 4 {
        if let Ok(diag_reader) = StreamReader::try_new(Cursor::new(ipc_bytes), None) {
            for batch in diag_reader {
                if let Ok(batch) = batch {
                    if batch.num_rows() > 0 {
                        let mut first_row: Vec<String> = Vec::new();
                        for col_idx in 0..batch.num_columns().min(15) {
                            let col = batch.column(col_idx);
                            let val = arrow::util::display::array_value_to_string(col, 0)
                                .unwrap_or_else(|_| "?".to_string());
                            first_row.push(format!("{}={}", schema_names[col_idx], val));
                        }
                        tracing::info!(first_row = ?first_row, "DIAG: first row from DuckDB IPC");
                        break;
                    }
                }
            }
        }
    }

    // Determine column indices to project.
    // Prefer name-based matching over explicit indices because DuckDB's Substrait
    // execution can produce output where column names and types don't correspond
    // to positional indices (e.g., JoinRel output may scramble types vs names).
    let indices = if let Some(explicit) = explicit_indices {
        // Validate: check that schema names at explicit indices match output_names.
        let names_match = explicit.len() == output_names.len()
            && explicit.iter().zip(output_names.iter()).all(|(&idx, name)| {
                idx < schema_names.len() && (schema_names[idx] == name || name.is_empty())
            });
        if names_match {
            tracing::info!(
                indices = ?explicit,
                duckdb_schema = ?schema_names,
                "using explicit column indices (validated)"
            );
            explicit.to_vec()
        } else {
            // Names don't match at explicit positions — fall back to name-based.
            tracing::warn!(
                explicit_indices = ?explicit,
                output_names = ?output_names,
                duckdb_schema = ?schema_names,
                "explicit indices don't match schema names, falling back to name-based projection"
            );
            let mut name_indices = Vec::new();
            for name in output_names {
                match schema.index_of(name) {
                    Ok(idx) => name_indices.push(idx),
                    Err(_) => {
                        tracing::warn!(
                            column = %name,
                            duckdb_schema = ?schema_names,
                            "projection column not found in result, skipping projection"
                        );
                        return Ok(ipc_bytes.to_vec());
                    }
                }
            }
            name_indices
        }
    } else if schema.fields().len() == output_names.len()
        && schema_names
            .iter()
            .zip(output_names.iter())
            .all(|(a, b)| a == b || b.is_empty())
    {
        // Column count AND order matches — no projection needed.
        tracing::info!(
            cols = schema.fields().len(),
            duckdb_schema = ?schema_names,
            output_names = ?output_names,
            "IPC column count and order match, no projection needed"
        );
        return Ok(ipc_bytes.to_vec());
    } else {
        // Find column indices by name.
        let mut name_indices = Vec::new();
        for name in output_names {
            match schema.index_of(name) {
                Ok(idx) => name_indices.push(idx),
                Err(_) => {
                    tracing::warn!(
                        column = %name,
                        duckdb_schema = ?schema_names,
                        "projection column not found in result, skipping projection"
                    );
                    return Ok(ipc_bytes.to_vec());
                }
            }
        }
        name_indices
    };

    let projected_schema = Arc::new(
        schema
            .project(&indices)
            .map_err(|e| format!("project schema: {e}"))?,
    );
    // Log projected schema types for diagnostics.
    let projected_types: Vec<String> = projected_schema
        .fields()
        .iter()
        .map(|f| format!("{}:{:?}", f.name(), f.data_type()))
        .collect();
    tracing::debug!(projected_types = ?projected_types, "projected schema types");
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &projected_schema)
            .map_err(|e| format!("IPC writer: {e}"))?;
        for batch_result in reader {
            let batch = batch_result.map_err(|e| format!("read IPC batch: {e}"))?;
            let projected = batch
                .project(&indices)
                .map_err(|e| format!("project batch: {e}"))?;
            writer
                .write(&projected)
                .map_err(|e| format!("write projected batch: {e}"))?;
        }
        writer.finish().map_err(|e| format!("finish IPC: {e}"))?;
    }
    tracing::info!(
        from_cols = schema.fields().len(),
        to_cols = indices.len(),
        indices = ?indices,
        duckdb_schema = ?schema_names,
        "projected IPC result columns"
    );
    Ok(buf)
}

/// Extract FE output column names from fragment output_exprs + desc_tbl.
///
/// Returns the column names in the order the FE's SELECT list expects them.
fn extract_fe_output_names(params: &TPipelineFragmentParams) -> Vec<String> {
    use doris_thrift::exprs::TExprNodeType;

    let output_exprs = match params.fragment.as_ref().and_then(|f| f.output_exprs.as_ref()) {
        Some(exprs) => exprs,
        None => return Vec::new(),
    };
    let desc_tbl = match params.desc_tbl.as_ref() {
        Some(d) => d,
        None => return Vec::new(),
    };

    // Build slot_id → col_name map from descriptor table.
    let slot_map: std::collections::HashMap<i32, &str> = desc_tbl
        .slot_descriptors
        .as_ref()
        .map(|slots| {
            slots
                .iter()
                .map(|s| (s.id, s.col_name.as_str()))
                .collect()
        })
        .unwrap_or_default();

    output_exprs
        .iter()
        .enumerate()
        .map(|(i, expr)| {
            expr.nodes
                .first()
                .and_then(|n| {
                    if n.node_type == TExprNodeType::SLOT_REF {
                        n.slot_ref.as_ref()
                    } else {
                        None
                    }
                })
                .and_then(|sr| slot_map.get(&sr.slot_id))
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("expr_{}", i))
        })
        .collect()
}

/// Reorder IPC columns to match FE output order and pad with NULLs for missing columns.
///
/// When Doris uses late materialization (VMaterializeNode), the exchange data has
/// fewer columns than the FE expects, and may be in a different order (e.g., sort key
/// order vs. SELECT list order). This function:
/// 1. Matches IPC columns to FE output positions by name
/// 2. Reorders columns to match FE's expected order
/// 3. Inserts NullArray columns for late-materialized columns not in the exchange
fn reorder_and_pad_ipc(
    ipc_bytes: &[u8],
    fe_output_names: &[String],
) -> Result<Vec<u8>, String> {
    use arrow::ipc::reader::StreamReader;
    use arrow::ipc::writer::StreamWriter;

    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None)
        .map_err(|e| format!("parse IPC for reorder+pad: {e}"))?;
    let schema = reader.schema();

    if schema.fields().len() >= fe_output_names.len() {
        return Ok(ipc_bytes.to_vec());
    }

    // Build name → IPC column index map.
    let ipc_name_to_idx: std::collections::HashMap<&str, usize> = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| (f.name().as_str(), i))
        .collect();

    // For each FE output position, find the source IPC column index (or None for NULL).
    let mapping: Vec<Option<usize>> = fe_output_names
        .iter()
        .map(|name| ipc_name_to_idx.get(name.as_str()).copied())
        .collect();

    let matched = mapping.iter().filter(|m| m.is_some()).count();
    let null_count = mapping.iter().filter(|m| m.is_none()).count();
    info!(
        ipc_cols = schema.fields().len(),
        fe_cols = fe_output_names.len(),
        matched,
        null_count,
        fe_names = ?fe_output_names,
        ipc_names = ?schema.fields().iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
        "reordering and padding IPC for late materialization"
    );

    // Build padded schema in FE output order.
    let mut fields: Vec<arrow::datatypes::FieldRef> = Vec::with_capacity(fe_output_names.len());
    for (i, maybe_idx) in mapping.iter().enumerate() {
        match maybe_idx {
            Some(idx) => fields.push(schema.field(*idx).clone().into()),
            None => fields.push(Arc::new(arrow::datatypes::Field::new(
                &fe_output_names[i],
                arrow::datatypes::DataType::Utf8,
                true,
            ))),
        }
    }
    let padded_schema = Arc::new(arrow::datatypes::Schema::new(fields));

    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, &padded_schema)
            .map_err(|e| format!("IPC writer: {e}"))?;
        for batch_result in reader {
            let batch = batch_result.map_err(|e| format!("read IPC batch: {e}"))?;
            let num_rows = batch.num_rows();
            let mut columns: Vec<Arc<dyn arrow::array::Array>> =
                Vec::with_capacity(fe_output_names.len());
            for maybe_idx in &mapping {
                match maybe_idx {
                    Some(idx) => columns.push(batch.column(*idx).clone()),
                    None => columns.push(arrow::array::new_null_array(
                        &arrow::datatypes::DataType::Utf8,
                        num_rows,
                    )),
                }
            }
            let padded_batch = arrow::record_batch::RecordBatch::try_new(
                padded_schema.clone(),
                columns,
            )
            .map_err(|e| format!("create padded batch: {e}"))?;
            writer
                .write(&padded_batch)
                .map_err(|e| format!("write padded batch: {e}"))?;
        }
        writer.finish().map_err(|e| format!("finish IPC: {e}"))?;
    }
    Ok(buf)
}

/// Check if a fragment has unresolved EXCHANGE_NODE(0 children) that need
/// exchange data from other BEs (i.e., not merged with a leaf fragment).
fn has_unresolved_exchanges(params: &TPipelineFragmentParams) -> Vec<i32> {
    let mut exchange_node_ids = Vec::new();
    let plan = match params.fragment.as_ref().and_then(|f| f.plan.as_ref()) {
        Some(p) => p,
        None => return exchange_node_ids,
    };
    for node in &plan.nodes {
        if node.node_type == TPlanNodeType::EXCHANGE_NODE && node.num_children == 0 {
            exchange_node_ids.push(node.node_id);
        }
    }
    exchange_node_ids
}

/// Get the number of senders for an exchange node from the fragment params.
fn get_num_senders(params: &TPipelineFragmentParams, node_id: i32) -> u32 {
    // per_exch_num_senders: map<TPlanNodeId, i32> — gives sender count per exchange node.
    params
        .per_exch_num_senders
        .get(&node_id)
        .map(|&n| n as u32)
        .or_else(|| params.num_senders.map(|n| n as u32))
        .unwrap_or(1)
}

/// Generate SQL for AGG(finalize) over exchange table.
///
/// When an intermediate fragment has AGG(need_finalize=true) → EXCHANGE, the partial
/// aggregation results are in the exchange table. The finalize needs to apply the
/// merge function (SUM for count/sum, MIN for min, MAX for max) rather than
/// re-running the original aggregate.
///
/// Returns `Some(sql)` if the pattern matches, `None` otherwise.
fn generate_exchange_agg_merge_sql(
    params: &TPipelineFragmentParams,
    table_schemas: &std::collections::HashMap<String, Vec<String>>,
) -> Option<String> {
    let plan = params.fragment.as_ref()?.plan.as_ref()?;
    let nodes = &plan.nodes;

    // Pattern: AGG_NODE(1 child) → EXCHANGE_NODE(0 children)
    if nodes.len() != 2 {
        return None;
    }
    let agg = &nodes[0];
    let exch = &nodes[1];
    if agg.node_type != TPlanNodeType::AGGREGATION_NODE || agg.num_children != 1 {
        return None;
    }
    if exch.node_type != TPlanNodeType::EXCHANGE_NODE || exch.num_children != 0 {
        return None;
    }
    let agg_node = agg.agg_node.as_ref()?;
    if !agg_node.need_finalize {
        return None;
    }

    // Look up the exchange table using the query-scoped name.
    // table_schemas keys are already query-scoped via exchange_table_name().
    let query_lo = params.query_id.lo;
    let table_name = exchange_table_name(query_lo, exch.node_id);
    let columns = table_schemas.get(&table_name)?;

    // Determine grouping columns: first N columns are GROUP BY keys.
    let num_grouping = agg_node.grouping_exprs.as_ref().map(|g| g.len()).unwrap_or(0);
    let num_agg_fns = agg_node.aggregate_functions.len();

    if columns.len() < num_grouping + num_agg_fns {
        tracing::warn!(
            columns = columns.len(), num_grouping, num_agg_fns,
            "exchange table column count mismatch for AGG merge"
        );
        return None;
    }

    // Determine permutation: the exchange table columns may be in a different
    // order than the AGG_FINAL's aggregate_functions. Compute the mapping from
    // AGG_FINAL order → exchange table column positions using slot_ref → slot_id
    // matching (same logic as node_translator.rs reordering).
    let measure_permutation: Vec<usize> = {
        use doris_thrift::exprs::TExprNodeType;

        // Get materialized slots for the exchange node's input tuple,
        // sorted by column_pos (matches DuckDB table column order).
        let input_tuple_id = *exch.row_tuples.first().unwrap_or(&0);
        let mut input_slots: Vec<(i32, i32)> = params.desc_tbl.as_ref()
            .and_then(|dt| dt.slot_descriptors.as_ref())
            .map(|slots| {
                slots.iter()
                    .filter(|s| s.parent == input_tuple_id && s.is_materialized)
                    .map(|s| (s.id, s.column_pos))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        input_slots.sort_by_key(|&(_, pos)| pos);
        let input_slot_ids: Vec<i32> = input_slots.iter().map(|&(id, _)| id).collect();

        if input_slot_ids.len() > num_grouping {
            let measure_slots = &input_slot_ids[num_grouping..];
            let mut perm = Vec::new();
            let mut all_found = true;
            for agg_fn in &agg_node.aggregate_functions {
                let slot_id = agg_fn.nodes.iter()
                    .find(|n| n.node_type == TExprNodeType::SLOT_REF)
                    .and_then(|n| n.slot_ref.as_ref())
                    .map(|sr| sr.slot_id);
                if let Some(sid) = slot_id {
                    if let Some(pos) = measure_slots.iter().position(|&s| s == sid) {
                        perm.push(pos);
                    } else { all_found = false; break; }
                } else { all_found = false; break; }
            }
            if all_found { perm } else { (0..num_agg_fns).collect() }
        } else {
            (0..num_agg_fns).collect()
        }
    };

    // Build SELECT list.
    let mut select_parts = Vec::new();

    // Grouping columns pass through.
    for col in &columns[..num_grouping] {
        select_parts.push(format!("\"{}\"", col));
    }

    // Aggregate columns: apply merge function, using the permutation to
    // map AGG_FINAL's aggregate_functions[i] → exchange table column position.
    for (i, agg_fn_expr) in agg_node.aggregate_functions.iter().enumerate() {
        let col_idx = num_grouping + measure_permutation[i];
        let col = &columns[col_idx];

        // Extract function name from the aggregate expression root node.
        let func_name = agg_fn_expr.nodes.first().and_then(|n| {
            n.fn_.as_ref().map(|f| f.name.function_name.as_str())
        });

        // Map original aggregate → merge function.
        let merge_fn = match func_name {
            Some("count") => "SUM",
            Some("sum") | Some("multi_distinct_sum") => "SUM",
            Some("min") => "MIN",
            Some("max") => "MAX",
            Some("any_value") => "ANY_VALUE",
            Some(other) => {
                tracing::warn!(func = other, "unknown aggregate merge function, defaulting to SUM");
                "SUM"
            }
            None => "SUM",
        };

        select_parts.push(format!("{}(\"{}\") AS \"{}\"", merge_fn, col, col));
    }

    let select = select_parts.join(", ");

    let sql = if num_grouping > 0 {
        let group_cols: Vec<String> = (1..=num_grouping).map(|i| i.to_string()).collect();
        format!(
            "SELECT {} FROM \"{}\" GROUP BY {}",
            select, table_name, group_cols.join(", ")
        )
    } else {
        format!("SELECT {} FROM \"{}\"", select, table_name)
    };

    info!(sql = %sql, "generated exchange AGG merge SQL");
    Some(sql)
}

/// Build a unique exchange table name scoped to a query.
///
/// Uses the low 32 bits of query_id.lo as a hex prefix to avoid collisions
/// when concurrent queries use the same node_id.
fn exchange_table_name(query_lo: i64, node_id: i32) -> String {
    format!("__EXCH_{:08x}_{}", query_lo as u32, node_id)
}

/// Generate SQL for UNION_NODE exchange fragments.
///
/// DuckDB's `from_substrait` is broken for `SetRel` (UNION): it mishandles the
/// union operator, producing wrong results or errors. Since UNION_NODE in exchange
/// fragments just concatenates exchange tables, we bypass substrait and generate
/// the trivial SQL directly: `SELECT * FROM __EXCH_... UNION ALL SELECT * FROM __EXCH_...`
///
/// Non-union exchange patterns (plain EXCHANGE_NODE, SORT over exchange, etc.)
/// go through the normal substrait path.
fn generate_exchange_union_sql(params: &TPipelineFragmentParams) -> Option<String> {
    let plan = params.fragment.as_ref()?.plan.as_ref()?;
    let nodes = &plan.nodes;
    let root = nodes.first()?;
    if root.node_type != TPlanNodeType::UNION_NODE {
        return None;
    }
    let query_lo = params.query_id.lo;
    let num_children = root.num_children as usize;
    let mut parts = Vec::new();
    let mut idx = 1; // skip root node
    for _ in 0..num_children {
        if idx < nodes.len() {
            let child = &nodes[idx];
            let tbl = exchange_table_name(query_lo, child.node_id);
            parts.push(format!("SELECT * FROM \"{}\"", tbl));
            idx += 1;
        }
    }
    if !parts.is_empty() {
        Some(parts.join(" UNION ALL "))
    } else {
        None
    }
}

/// Extract exchange send destinations and partition strategy from fragment params.
///
/// Returns `ExchangeInfo` if this fragment's output should be sent to remote BEs
/// via transmit_block. Returns `None` for result sinks (where FE fetches data
/// directly) or when destinations are empty.
fn extract_exchange_destinations(
    params: &TPipelineFragmentParams,
) -> Option<ExchangeInfo> {
    // Check if fragment has a DATA_STREAM_SINK output
    let fragment = params.fragment.as_ref()?;
    let output_sink = fragment.output_sink.as_ref()?;

    if output_sink.type_ != TDataSinkType::DATA_STREAM_SINK {
        return None;
    }

    let stream_sink = output_sink.stream_sink.as_ref()?;
    let dest_node_id = stream_sink.dest_node_id;

    let destinations = params.destinations.as_ref()?;
    if destinations.is_empty() {
        return None;
    }

    let dests: Vec<ExchangeDest> = destinations
        .iter()
        .filter_map(|d| {
            // Prefer brpc_server address, fall back to server address
            let addr = d
                .brpc_server
                .as_ref()
                .map(|a| format!("{}:{}", a.hostname, a.port))
                .or_else(|| Some(format!("{}:{}", d.server.hostname, d.server.port)))?;
            Some(ExchangeDest {
                brpc_addr: addr,
                finst_id: (d.fragment_instance_id.hi, d.fragment_instance_id.lo),
            })
        })
        .collect();

    if dests.is_empty() {
        return None;
    }

    // Extract partition strategy from TDataStreamSink.output_partition.
    let output_partition = &stream_sink.output_partition;
    let enable_new_shuffle = params
        .query_options
        .as_ref()
        .and_then(|qo| qo.enable_new_shuffle_hash_method)
        .unwrap_or(false);

    let partition = partition_strategy_from_thrift(
        &output_partition.type_,
        &output_partition.partition_exprs,
        dests.len(),
        enable_new_shuffle,
    );

    if !matches!(partition, super::hash_partitioner::PartitionStrategy::Broadcast) {
        tracing::info!(
            dest_node_id,
            num_dests = dests.len(),
            partition_type = ?output_partition.type_,
            "extracted hash-partitioned exchange"
        );
    }

    Some(ExchangeInfo {
        dest_node_id,
        destinations: dests,
        partition,
    })
}

pub struct PBackendServiceHandler {
    state: Arc<BeState>,
    result_store: ResultStore,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
    exchange_buffer: ExchangeBuffer,
    no_cpu_fallback: bool,
    force_cpu: bool,
    nixl_only: bool,
    nixl_agent: Option<Arc<super::nixl_exchange::NixlExchange>>,
    /// Transfer dispatcher for inter-BE exchange (nixl → bRPC fallback).
    transfer_dispatcher: Option<super::transfer_engine::TransferDispatcher>,
    /// This BE's brpc address as seen by other BEs (advertise_host:brpc_port).
    /// Used to detect self-transfer (destination == local BE).
    local_brpc_addr: String,
}

impl PBackendServiceHandler {
    pub fn new(
        state: Arc<BeState>,
        result_store: ResultStore,
        engine: Option<Arc<Mutex<SiriusEngine>>>,
        exchange_buffer: ExchangeBuffer,
        no_cpu_fallback: bool,
        force_cpu: bool,
        nixl_only: bool,
    ) -> Self {
        Self {
            state,
            result_store,
            engine,
            exchange_buffer,
            no_cpu_fallback,
            force_cpu,
            nixl_only,
            nixl_agent: None,
            transfer_dispatcher: None,
            local_brpc_addr: String::new(),
        }
    }

    pub fn with_nixl_agent(mut self, agent: Option<Arc<super::nixl_exchange::NixlExchange>>) -> Self {
        self.nixl_agent = agent;
        self
    }

    pub fn with_transfer_dispatcher(mut self, dispatcher: super::transfer_engine::TransferDispatcher) -> Self {
        self.transfer_dispatcher = Some(dispatcher);
        self
    }

    pub fn with_local_brpc_addr(mut self, addr: String) -> Self {
        self.local_brpc_addr = addr;
        self
    }
}

#[tonic::async_trait]
impl PBackendService for PBackendServiceHandler {
    #[instrument(skip_all, fields(compact, query_id, fragment_id))]
    async fn exec_plan_fragment(
        &self,
        request: Request<PExecPlanFragmentRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        let req = request.into_inner();
        let compact = req.compact.unwrap_or(false);

        let version = req.version.unwrap_or(2); // default = VERSION_2
        let thrift_bytes = match req.request {
            Some(bytes) => bytes,
            None => {
                return Ok(Response::new(PExecPlanFragmentResult {
                    status: err_status("missing request bytes"),
                    ..Default::default()
                }));
            }
        };

        info!(
            compact,
            version,
            len = thrift_bytes.len(),
            first_bytes = ?&thrift_bytes[..thrift_bytes.len().min(32)],
            "received exec_plan_fragment request"
        );

        // Deserialize Thrift fragment params (VERSION_3 may contain multiple fragments).
        let all_params = match deserialize_params(thrift_bytes, compact, version) {
            Ok(p) => p,
            Err(e) => {
                warn!(error = %e, "failed to deserialize fragment params");
                return Ok(Response::new(PExecPlanFragmentResult {
                    status: err_status(&e),
                    ..Default::default()
                }));
            }
        };

        info!(num_fragments = all_params.len(), "deserialized fragment params");

        // Capture the result-sink fragment's instance ID BEFORE merge.
        // The result-sink fragment is fragment_id=0 (EXCHANGE_NODE at root with 0 children,
        // or the single fragment if there's only one). After merge, local_params[0] may
        // belong to a different fragment, so we must capture this first.
        let query_id_key = FinstId {
            hi: all_params[0].query_id.hi,
            lo: all_params[0].query_id.lo,
        };
        let result_sink_finst_id = all_params.iter()
            .find(|p| p.fragment_id == Some(0))
            .or_else(|| all_params.first())
            .and_then(|p| p.local_params.as_ref())
            .and_then(|lp| lp.first())
            .map(|p| FinstId { hi: p.fragment_instance_id.hi, lo: p.fragment_instance_id.lo })
            .filter(|id| id.hi != query_id_key.hi || id.lo != query_id_key.lo);

        // Merge multi-fragment plans for single-BE execution.
        // This replaces EXCHANGE_NODE(0 children) in intermediate fragments with
        // the leaf (scan) fragment's plan, producing a single executable plan.
        let merged_params = merge_fragment_plans(&all_params);
        info!(
            merged_fragments = merged_params.len(),
            "merged fragment plans for single-BE execution"
        );

        // Process each merged fragment.
        for params in &merged_params {
            // Primary key: query_id (FE uses this with enableParallelResultSink=true).
            let finst_id = FinstId {
                hi: params.query_id.hi,
                lo: params.query_id.lo,
            };
            // Secondary key: result-sink fragment's instance ID, captured before merge.
            let fragment_finst_id = result_sink_finst_id;

            // Check for unresolved exchange nodes (receive data from other BEs).
            let exchange_node_ids = has_unresolved_exchanges(params);

            if !exchange_node_ids.is_empty() {
                // This fragment depends on exchange data from other BEs.
                // Register exchange entries and spawn an async task to wait.
                let query_id = (params.query_id.hi, params.query_id.lo);
                let mut notifies = Vec::new();
                for &node_id in &exchange_node_ids {
                    let key = ExchangeKey { query_id, node_id };
                    let num_senders = get_num_senders(params, node_id);
                    info!(
                        ?query_id, node_id, num_senders,
                        "registering exchange for fragment"
                    );
                    let notify = self.exchange_buffer.register(key, num_senders);
                    notifies.push(notify);
                }

                let params = params.clone();
                let engine = match &self.engine {
                    Some(e) => e.clone(),
                    None => {
                        return Ok(Response::new(PExecPlanFragmentResult {
                            status: err_status("engine not initialized"),
                            ..Default::default()
                        }));
                    }
                };
                let store = self.result_store.clone();
                let buffer = self.exchange_buffer.clone();
                let no_cpu_fallback = self.no_cpu_fallback;
                let force_cpu = self.force_cpu;
                let nixl_only = self.nixl_only;
                let nixl_agent = self.nixl_agent.clone();
                let local_brpc_addr = self.local_brpc_addr.clone();
                let exchange_buffer = self.exchange_buffer.clone();

                // Spawn async task: wait for exchange data, decode, load, execute.
                tokio::spawn(async move {
                    // Wait for all exchange nodes to receive all their data.
                    for notify in &notifies {
                        notify.notified().await;
                    }

                    // Check if query was cancelled while we were waiting.
                    if buffer.is_cancelled(query_id.0, query_id.1) {
                        info!(%finst_id, "exchange task cancelled");
                        store.store_error(finst_id, fragment_finst_id, "query cancelled".to_string());
                        buffer.clear_cancelled(query_id.0, query_id.1);
                        return;
                    }

                    info!(%finst_id, "all exchange data received, proceeding with execution");

                    // Decode PBlocks and register exchange tables.
                    let mut table_schemas = std::collections::HashMap::<String, Vec<String>>::new();
                    for &node_id in &exchange_node_ids {
                        let key = ExchangeKey { query_id, node_id };
                        let blocks = buffer.take(&key);
                        let table_name = exchange_table_name(query_id.1, node_id);

                        if blocks.is_empty() {
                            info!(table = %table_name, "exchange has no blocks, creating empty table");
                            continue;
                        }

                        // Log PBlock diagnostic info.
                        let blk0 = &blocks[0];
                        let be_ver = blk0.be_exec_version.unwrap_or(-1);
                        let compressed = blk0.compressed.unwrap_or(false);
                        let cv_len = blk0.column_values.as_ref().map(|v| v.len()).unwrap_or(0);
                        let uncomp = blk0.uncompressed_size.unwrap_or(0);
                        info!(
                            table = %table_name,
                            be_exec_version = be_ver,
                            compressed = compressed,
                            column_values_len = cv_len,
                            uncompressed_size = uncomp,
                            num_column_metas = blk0.column_metas.len(),
                            "exchange PBlock info"
                        );
                        for (i, m) in blk0.column_metas.iter().enumerate() {
                            info!(
                                table = %table_name,
                                col_idx = i,
                                name = m.name.as_deref().unwrap_or(""),
                                type_id = m.r#type.unwrap_or(-1),
                                nullable = m.is_nullable.unwrap_or(false),
                                "exchange PBlock column_meta"
                            );
                        }
                        // Dump first 64 bytes of (decompressed) column_values for debugging.
                        if let Some(cv) = blk0.column_values.as_ref() {
                            let preview_len = cv.len().min(128);
                            let hex: String = cv[..preview_len]
                                .iter()
                                .map(|b| format!("{:02x}", b))
                                .collect::<Vec<_>>()
                                .join(" ");
                            info!(table = %table_name, hex = %hex, "column_values first bytes");
                        }

                        // Extract column metadata from the first block.
                        let col_info = pblock_decoder::extract_column_info(
                            &blocks[0].column_metas,
                        );

                        // Decode all blocks.
                        let decoded = match pblock_decoder::decode_pblocks(&blocks) {
                            Ok(d) => d,
                            Err(e) => {
                                warn!(error = %e, table = %table_name, "failed to decode PBlocks");
                                store.store_error(finst_id, fragment_finst_id, format!("decode PBlocks for {table_name}: {e}"));
                                return;
                            }
                        };

                        info!(
                            table = %table_name,
                            num_rows = decoded.num_rows,
                            num_cols = decoded.columns.len(),
                            "decoded exchange PBlocks"
                        );

                        // Filter out internal Doris columns and FIXEDLENGTHOBJECT (agg state).
                        // __DORIS_ROWID_COL__ is an internal row ID not needed by DuckDB.
                        // FIXEDLENGTHOBJECT columns are opaque aggregate intermediate state.
                        let keep_indices: Vec<usize> = (0..col_info.len())
                            .filter(|&i| {
                                let (name, _) = &col_info[i];
                                let type_id = blocks[0].column_metas.get(i)
                                    .and_then(|m| m.r#type)
                                    .unwrap_or(0);
                                !name.starts_with("__DORIS_")
                                    && type_id != doris_proto::doris::p_generic_type::TypeId::Fixedlengthobject as i32
                            })
                            .collect();

                        let column_names: Vec<String> =
                            keep_indices.iter().map(|&i| {
                                let name = &col_info[i].0;
                                if name.is_empty() {
                                    format!("col{}", i)
                                } else {
                                    name.clone()
                                }
                            }).collect();
                        let column_types_sql: Vec<String> =
                            keep_indices.iter().map(|&i| col_info[i].1.clone()).collect();

                        // Convert decoded columns to Arrow IPC and register via fast path.
                        let ipc_bytes = match pblock_decoder::decoded_columns_to_arrow_ipc(
                            &decoded, &keep_indices,
                        ) {
                            Ok(bytes) => bytes,
                            Err(e) => {
                                warn!(error = %e, table = %table_name, "failed to convert exchange PBlock to Arrow IPC");
                                // Fall back to SQL VALUES path.
                                let column_data_csv: Vec<Vec<String>> = keep_indices.iter()
                                    .map(|&i| pblock_decoder::column_to_sql_values(&decoded.columns[i], decoded.num_rows))
                                    .collect();
                                let engine_guard = engine.lock().unwrap();
                                if let Err(e2) = engine_guard.register_exchange_table(
                                    &table_name, &column_names, &column_types_sql,
                                    decoded.num_rows, &column_data_csv,
                                ) {
                                    warn!(error = %e2, table = %table_name, "SQL VALUES fallback also failed");
                                    store.store_error(finst_id, fragment_finst_id, format!("register exchange table {table_name}: {e2}"));
                                    return;
                                }
                                if let Ok(cols) = engine_guard.get_table_columns(&table_name) {
                                    table_schemas.insert(table_name, cols);
                                }
                                drop(engine_guard);
                                continue;
                            }
                        };

                        info!(
                            table = %table_name,
                            ipc_len = ipc_bytes.len(),
                            num_rows = decoded.num_rows,
                            "converted exchange PBlock to Arrow IPC"
                        );

                        // Register in DuckDB via IPC file (fast path).
                        let engine_guard = engine.lock().unwrap();
                        if let Err(e) = engine_guard.register_exchange_table_from_ipc(
                            &table_name, &ipc_bytes,
                        ) {
                            warn!(error = %e, table = %table_name, "IPC registration failed");
                            store.store_error(finst_id, fragment_finst_id, format!("IPC registration for {table_name}: {e}"));
                            return;
                        }

                        // Get actual columns from DuckDB for Substrait schema.
                        match engine_guard.get_table_columns(&table_name) {
                            Ok(cols) => {
                                table_schemas.insert(table_name, cols);
                            }
                            Err(e) => {
                                warn!(error = %e, "failed to get exchange table columns");
                            }
                        }
                        drop(engine_guard);
                    }

                    // Force DuckDB to commit any pending exchange table transactions.
                    // DuckDB's from_substrait() creates a new connection whose MVCC
                    // snapshot may not include tables from the current connection's
                    // auto-commit transactions. An explicit no-op query forces the
                    // transaction boundary so new connections see all tables.
                    if !table_schemas.is_empty() {
                        let eng = engine.lock().unwrap();
                        let _ = eng.execute_sql("SELECT 42");
                        drop(eng);
                    }

                    // Also register any file tables.
                    let file_scan_infos = extract_file_scan_info(&params);
                    let mut file_scan_map = std::collections::HashMap::<String, plan_translator::FileScanInfo>::new();
                    if !file_scan_infos.is_empty() {
                        let engine_guard = engine.lock().unwrap();
                        for fsi in &file_scan_infos {
                            if fsi.format == "parquet" {
                                // Parquet: use LocalFiles path (no table materialization).
                                match engine_guard.get_parquet_columns(&fsi.files[0].path) {
                                    Ok(columns) => {
                                        info!(
                                            table = %fsi.table_name,
                                            num_files = fsi.files.len(),
                                            "parquet LocalFiles path (async)"
                                        );
                                        table_schemas.insert(fsi.table_name.clone(), columns);
                                        file_scan_map.insert(fsi.table_name.clone(), fsi.clone());
                                    }
                                    Err(e) => {
                                        warn!(error = %e, table = %fsi.table_name,
                                              "get_parquet_columns failed, falling back");
                                        let empty: Vec<String> = vec![];
                                        if let Err(e) = engine_guard.register_file_table(
                                            &fsi.table_name, &fsi.files[0].path, &fsi.format, &empty,
                                        ) {
                                            warn!(error = %e, table = %fsi.table_name, "failed to register file table");
                                            store.store_error(finst_id, fragment_finst_id, format!("register file table '{}': {e}", fsi.table_name));
                                            return;
                                        }
                                        if let Ok(columns) = engine_guard.get_table_columns(&fsi.table_name) {
                                            table_schemas.insert(fsi.table_name.clone(), columns);
                                        }
                                    }
                                }
                            } else {
                                let empty: Vec<String> = vec![];
                                if let Err(e) = engine_guard.register_file_table(
                                    &fsi.table_name, &fsi.files[0].path, &fsi.format, &empty,
                                ) {
                                    warn!(error = %e, table = %fsi.table_name, "failed to register file table");
                                    store.store_error(finst_id, fragment_finst_id, format!("register file table '{}': {e}", fsi.table_name));
                                    return;
                                }
                                if let Ok(columns) = engine_guard.get_table_columns(&fsi.table_name) {
                                    table_schemas.insert(fsi.table_name.clone(), columns);
                                }
                            }
                        }
                        drop(engine_guard);
                    }

                    // Translate and execute.
                    // Priority order:
                    // 1. UNION_NODE: trivial exchange table SQL (DuckDB SetRel broken for substrait)
                    // 2. Everything else (AGG finalize, SORT, JOIN): Substrait path
                    //    AGG finalize uses INTERMEDIATE_TO_RESULT phase so DuckDB rewrites
                    //    count → sum, etc. for merging partial results.
                    let (exec_plan, output_names) =
                        if let Some(read_sql) = generate_exchange_union_sql(&params) {
                            info!(sql = %read_sql, "exchange fragment using CPU-only SQL path");
                            (ExecPlan::SqlCpuOnly(read_sql), None)
                        } else {
                            match plan_translator::translate_fragment(&params, &table_schemas, &file_scan_map) {
                                Ok(plan) => {
                                    // Exchange fragments always use CPU: they read from exchange
                                    // tables (not parquet), so GPU scan pipeline has no operators.
                                    let exec = ExecPlan::SubstraitCpuOnly {
                                        bytes: plan.substrait_bytes,
                                        sort_limit_sql: plan.sort_limit_sql,
                                    };
                                    (exec, Some(plan.output_names))
                                }
                                Err(e) => {
                                    warn!(error = %e, "Substrait translation failed for exchange fragment");
                                    store.store_error(finst_id, fragment_finst_id, format!("Substrait translation: {e}"));
                                    return;
                                }
                            }
                        };

                    // Check if this exchange fragment also needs to forward results.
                    let exchange_dests = extract_exchange_destinations(&params);

                    // Extract slot descriptors for hash partition column resolution.
                    let desc_tbl_slots: Option<Vec<(i32, String)>> = params
                        .desc_tbl
                        .as_ref()
                        .and_then(|dt| dt.slot_descriptors.as_ref())
                        .map(|slots| slots.iter().map(|s| (s.id, s.col_name.clone())).collect());

                    // Extract sender_id from local_params (each BE gets a unique sender_id
                    // for the same fragment, so the receiver's ExchangeBuffer can distinguish senders).
                    let sender_id = params.local_params.as_ref()
                        .and_then(|lp| lp.first())
                        .map(|p| p.sender_id.unwrap_or(0))
                        .unwrap_or(0);

                    // For CPU-only plans (e.g. AGG merge SQL), skip GPU buffer detection
                    // to avoid stale GPU buffers from a previous leaf execution being
                    // mistakenly detected as the current result's location.
                    let is_cpu_only = matches!(&exec_plan, ExecPlan::SqlCpuOnly(_));

                    // If this exchange fragment has destinations and nixl is available,
                    // retain GPU buffers so nixl can use them after query cleanup.
                    let should_retain_exch = exchange_dests.is_some() && nixl_agent.is_some() && !is_cpu_only;
                    let nixl_agent_for_exch_blocking = if should_retain_exch { nixl_agent.clone() } else { None };
                    let engine_for_release = engine.clone();

                    let exec_result = tokio::task::spawn_blocking(move || -> Result<crate::nixl_integration::ExecutionLocation, String> {
                        let engine = engine.lock().unwrap();
                        if should_retain_exch {
                            if let Err(e) = engine.retain_gpu_buffers() {
                                tracing::warn!(error = %e, "failed to set retain_gpu_buffers for exchange fragment");
                            }
                        }
                        let mut ipc_bytes = execute_plan(&engine, exec_plan, no_cpu_fallback, force_cpu)?;
                        if let Some(names) = output_names {
                            ipc_bytes = project_ipc_columns(&ipc_bytes, &names, None)?;
                        }
                        if is_cpu_only {
                            Ok(crate::nixl_integration::ExecutionLocation::Cpu(ipc_bytes))
                        } else {
                            let mut loc = crate::nixl_integration::detect_execution_location(ipc_bytes, &engine);
                            if should_retain_exch {
                                if let Some(staging) = nixl_agent_for_exch_blocking.as_ref().and_then(|a| a.staging()) {
                                    match engine.stage_gpu_buffers(staging.base_addr()) {
                                        Ok(staged) => {
                                            tracing::info!(num_staged = staged.len(), "C++ staged GPU buffers for exchange");
                                            loc.apply_staging(staging.base_addr(), &staged);
                                        }
                                        Err(e) => {
                                            tracing::warn!(error = %e, "C++ stage_gpu_buffers failed for exchange");
                                        }
                                    }
                                }
                            }
                            Ok(loc)
                        }
                    })
                    .await;

                    // Helper to release retained GPU buffers after transfer (or on error).
                    let release_retained_exch = || {
                        if should_retain_exch {
                            let eng = engine_for_release.lock().unwrap();
                            if let Err(e) = eng.release_gpu_buffers() {
                                tracing::warn!(error = %e, "failed to release retained GPU buffers");
                            }
                        }
                    };

                    match exec_result {
                        Ok(Ok(location)) => {
                            if let Some(exch_info) = exchange_dests {
                                let query_id = (params.query_id.hi, params.query_id.lo);
                                if let Err(e) = crate::nixl_integration::send_exchange_with_nixl(
                                    nixl_agent.as_ref(),
                                    location, &exch_info, query_id, sender_id, nixl_only,
                                    &local_brpc_addr, &exchange_buffer,
                                    desc_tbl_slots.as_deref(),
                                ).await {
                                    warn!(error = %e, %finst_id, "exchange forward failed");
                                }
                                release_retained_exch();
                                info!(%finst_id, sender_id, "exchange fragment forward complete");
                            } else {
                                let mut ipc_bytes = location.into_ipc_bytes();

                                // Reorder and pad for late materialization:
                                // VMaterializeNode expects more columns than the exchange
                                // provides, and possibly in a different order. Match IPC
                                // columns to FE output positions by name, inserting NULLs
                                // for late-materialized columns.
                                let fe_names = extract_fe_output_names(&params);
                                if !fe_names.is_empty() {
                                    match reorder_and_pad_ipc(&ipc_bytes, &fe_names) {
                                        Ok(padded) => ipc_bytes = padded,
                                        Err(e) => warn!(error = %e, "IPC reorder+pad failed, storing as-is"),
                                    }
                                }

                                if let Err(e) = store.store_ipc_result(finst_id, &ipc_bytes) {
                                    warn!(error = %e, %finst_id, "failed to store exchange result");
                                } else {
                                    // Also store under fragment instance ID for non-parallel-sink FE.
                                    if let Some(fid) = fragment_finst_id {
                                        if let Some(entry) = store.get(&finst_id) {
                                            store.store_alias(fid, finst_id, entry);
                                        }
                                    }
                                    info!(%finst_id, "exchange fragment execution complete");
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            release_retained_exch();
                            warn!(error = %e, %finst_id, "exchange fragment execution failed");
                            store.store_error(finst_id, fragment_finst_id, format!("exchange execution: {e}"));
                        }
                        Err(e) => {
                            release_retained_exch();
                            warn!(error = %e, "exchange fragment spawn_blocking panicked");
                            store.store_error(finst_id, fragment_finst_id, format!("exchange task panicked: {e}"));
                        }
                    }
                });

                // Return immediately — execution happens asynchronously.
                continue;
            }

            // Standard synchronous execution path (no unresolved exchanges).

            // Step 1: Extract file scan info (all file paths per scan node).
            // Step 2: For parquet: get column names via DESCRIBE (no table materialization).
            //         For non-parquet: register as DuckDB table and get columns.
            // Step 3: Build file_scan_map for parquet LocalFiles path.
            // Step 4: Pass table_schemas + file_scan_map to the Substrait translator.
            let file_scan_infos = extract_file_scan_info(params);
            let mut table_schemas = std::collections::HashMap::<String, Vec<String>>::new();
            let mut file_scan_map = std::collections::HashMap::<String, plan_translator::FileScanInfo>::new();

            if !file_scan_infos.is_empty() {
                if let Some(engine) = &self.engine {
                    let engine_guard = engine.lock().unwrap();
                    for fsi in &file_scan_infos {
                        if fsi.format == "parquet" {
                            // Parquet: use LocalFiles path (no table materialization).
                            // Substrait ReadRel::LocalFiles embeds file paths directly.
                            // GPU: from_substrait → parquet_scan → sirius_physical_parquet_scan
                            // CPU: from_substrait → parquet_scan (DuckDB native)
                            match engine_guard.get_parquet_columns(&fsi.files[0].path) {
                                Ok(columns) => {
                                    info!(
                                        table = %fsi.table_name,
                                        num_files = fsi.files.len(),
                                        first_file = %fsi.files[0].path,
                                        "parquet LocalFiles path (no table materialization)"
                                    );
                                    table_schemas.insert(fsi.table_name.clone(), columns);
                                    file_scan_map.insert(fsi.table_name.clone(), fsi.clone());
                                }
                                Err(e) => {
                                    warn!(error = %e, table = %fsi.table_name,
                                          "get_parquet_columns failed, falling back to table materialization");
                                    let empty: Vec<String> = vec![];
                                    if let Err(e) = engine_guard.register_file_table(&fsi.table_name, &fsi.files[0].path, &fsi.format, &empty) {
                                        warn!(error = %e, table = %fsi.table_name, "failed to register file table");
                                        return Ok(Response::new(PExecPlanFragmentResult {
                                            status: err_status(&format!("register file table '{}': {e}", fsi.table_name)),
                                            ..Default::default()
                                        }));
                                    }
                                    if let Ok(columns) = engine_guard.get_table_columns(&fsi.table_name) {
                                        table_schemas.insert(fsi.table_name.clone(), columns);
                                    }
                                }
                            }
                        } else {
                            // Single-file non-parquet (or single-file parquet on GPU path):
                            // register as DuckDB table.
                            let empty: Vec<String> = vec![];
                            if let Err(e) = engine_guard.register_file_table(&fsi.table_name, &fsi.files[0].path, &fsi.format, &empty) {
                                warn!(error = %e, table = %fsi.table_name, "failed to register file table");
                                return Ok(Response::new(PExecPlanFragmentResult {
                                    status: err_status(&format!("register file table '{}': {e}", fsi.table_name)),
                                    ..Default::default()
                                }));
                            }
                            match engine_guard.get_table_columns(&fsi.table_name) {
                                Ok(columns) => {
                                    info!(
                                        table = %fsi.table_name,
                                        path = %fsi.files[0].path,
                                        columns = ?columns,
                                        "registered file table"
                                    );
                                    table_schemas.insert(fsi.table_name.clone(), columns);
                                }
                                Err(e) => {
                                    warn!(error = %e, table = %fsi.table_name, "failed to get table columns");
                                }
                            }
                        }
                    }
                }
            }

            // Translate to Substrait plan.
            // Constant-only queries (no data tables) use CPU-only Substrait to avoid
            // GPU engine bugs with VirtualTable plans.
            let has_data_tables = !file_scan_infos.is_empty() || !table_schemas.is_empty();
            let (exec_plan, output_names, output_indices) =
                match plan_translator::translate_fragment(params, &table_schemas, &file_scan_map) {
                    Ok(plan) => {
                        info!(bytes = plan.substrait_bytes.len(), has_data_tables, force_cpu = plan.force_cpu_substrait, "translated to Substrait");
                        let exec = if plan.force_cpu_substrait || !has_data_tables {
                            ExecPlan::SubstraitCpuOnly { bytes: plan.substrait_bytes, sort_limit_sql: plan.sort_limit_sql }
                        } else {
                            ExecPlan::Substrait { bytes: plan.substrait_bytes, sort_limit_sql: plan.sort_limit_sql }
                        };
                        (exec, Some(plan.output_names), plan.output_column_indices)
                    }
                    Err(e) => {
                        warn!(error = %e, "Substrait translation failed");
                        return Ok(Response::new(PExecPlanFragmentResult {
                            status: err_status(&format!("plan translation: {e}")),
                            ..Default::default()
                        }));
                    }
                };

            // Execute via Sirius GPU, falling back to DuckDB CPU.
            let engine = match &self.engine {
                Some(e) => e.clone(),
                None => {
                    return Ok(Response::new(PExecPlanFragmentResult {
                        status: err_status("engine not initialized (built without duckdb-bundled?)"),
                        ..Default::default()
                    }));
                }
            };
            let store = self.result_store.clone();
            let no_cpu_fallback = self.no_cpu_fallback;
            let force_cpu = self.force_cpu;

            // Check if this fragment's output should be sent to remote BEs.
            let exchange_dests = extract_exchange_destinations(params);

            // If this leaf has exchange destinations and nixl is available,
            // tell Sirius to retain GPU result buffers past query cleanup
            // so the nixl GPU-direct path can use them.
            let should_retain = exchange_dests.is_some() && self.nixl_agent.is_some();
            let nixl_agent_for_blocking = if should_retain { self.nixl_agent.clone() } else { None };

            // Sirius/DuckDB execution is blocking — run off the async runtime.
            let exec_result = tokio::task::spawn_blocking(move || -> Result<crate::nixl_integration::ExecutionLocation, String> {
                let t_total = std::time::Instant::now();
                let engine = engine.lock().unwrap();
                if should_retain {
                    if let Err(e) = engine.retain_gpu_buffers() {
                        tracing::warn!(error = %e, "failed to set retain_gpu_buffers, nixl may not work");
                    } else {
                        tracing::info!("retain_gpu_buffers set before GPU execution");
                    }
                }
                let t_exec = std::time::Instant::now();
                let mut ipc_bytes = execute_plan(&engine, exec_plan, no_cpu_fallback, force_cpu)?;
                tracing::info!(exec_ms = t_exec.elapsed().as_millis() as u64, ipc_len = ipc_bytes.len(), "execute_plan completed");
                if let Some(names) = output_names {
                    ipc_bytes = project_ipc_columns(&ipc_bytes, &names, output_indices.as_deref())?;
                }
                let t_detect = std::time::Instant::now();
                let mut location = crate::nixl_integration::detect_execution_location(ipc_bytes, &engine);
                tracing::info!(detect_ms = t_detect.elapsed().as_millis() as u64, gpu = matches!(location, crate::nixl_integration::ExecutionLocation::Gpu { .. }), "detect_execution_location done");
                // Try RMM pool registration first (zero-copy path). Falls back to
                // cuMemAlloc copy if pool registration fails.
                if should_retain {
                    // Stage GPU buffers via C++ cudaMemcpy. Pass 0 to let C++ self-allocate
                    // a staging buffer (both src and dst must be in the same CUDA runtime).
                    match engine.stage_gpu_buffers(0) {
                        Ok(staged) if !staged.is_empty() => {
                            tracing::info!(num_staged = staged.len(), "C++ staged GPU buffers");
                            // TODO: register C++ staging ptr with nixl and apply to location
                        }
                        Ok(_) => {}
                        Err(e) => {
                            tracing::warn!(error = %e, "C++ stage_gpu_buffers failed");
                        }
                    }
                }
                tracing::info!(total_ms = t_total.elapsed().as_millis() as u64, "leaf spawn_blocking done");
                Ok(location)
            })
            .await;

            // Helper to release retained GPU buffers after nixl transfer (or on error).
            let release_retained = |engine_arc: &Option<std::sync::Arc<std::sync::Mutex<sirius_ffi::SiriusEngine>>>| {
                if should_retain {
                    if let Some(eng) = engine_arc {
                        let eng = eng.lock().unwrap();
                        if let Err(e) = eng.release_gpu_buffers() {
                            tracing::warn!(error = %e, "failed to release retained GPU buffers");
                        }
                    }
                }
            };

            match exec_result {
                Ok(Ok(location)) => {
                    if let Some(exch_info) = exchange_dests {
                        // Send result to remote BEs via nixl GPU-direct or bRPC fallback.
                        let query_id = (params.query_id.hi, params.query_id.lo);
                        let sender_id = params
                            .local_params
                            .as_ref()
                            .and_then(|lp| lp.first())
                            .map(|p| p.sender_id.unwrap_or(0))
                            .unwrap_or(0);
                        info!(
                            %finst_id,
                            dest_node_id = exch_info.dest_node_id,
                            num_dests = exch_info.destinations.len(),
                            "sending execution result to exchange destinations"
                        );
                        // Extract slot descriptors for hash partition column resolution.
                        let desc_tbl_slots: Option<Vec<(i32, String)>> = params
                            .desc_tbl
                            .as_ref()
                            .and_then(|dt| dt.slot_descriptors.as_ref())
                            .map(|slots| slots.iter().map(|s| (s.id, s.col_name.clone())).collect());
                        if let Err(e) = crate::nixl_integration::send_exchange_with_nixl(
                            self.nixl_agent.as_ref(),
                            location,
                            &exch_info,
                            query_id,
                            sender_id,
                            self.nixl_only,
                            &self.local_brpc_addr,
                            &self.exchange_buffer,
                            desc_tbl_slots.as_deref(),
                        )
                        .await
                        {
                            release_retained(&self.engine);
                            warn!(error = %e, %finst_id, "exchange send failed");
                            return Ok(Response::new(PExecPlanFragmentResult {
                                status: err_status(&format!("exchange send: {e}")),
                                ..Default::default()
                            }));
                        }
                        release_retained(&self.engine);
                        info!(%finst_id, "exchange send complete");
                    } else {
                        // No exchange destinations — store result locally for fetch_data.
                        let ipc_bytes = location.into_ipc_bytes();
                        if let Err(e) = store.store_ipc_result(finst_id, &ipc_bytes) {
                            warn!(error = %e, %finst_id, "failed to store result");
                            return Ok(Response::new(PExecPlanFragmentResult {
                                status: err_status(&format!("store result: {e}")),
                                ..Default::default()
                            }));
                        }
                        // Also store under fragment instance ID for non-parallel-sink FE.
                        if let Some(fid) = fragment_finst_id {
                            if let Some(entry) = store.get(&finst_id) {
                                store.store_alias(fid, finst_id, entry);
                            }
                        }
                        info!(%finst_id, "execution complete, result stored");
                    }
                }
                Ok(Err(e)) => {
                    release_retained(&self.engine);
                    warn!(error = %e, %finst_id, "execution failed");
                    return Ok(Response::new(PExecPlanFragmentResult {
                        status: err_status(&format!("execution: {e}")),
                        ..Default::default()
                    }));
                }
                Err(e) => {
                    release_retained(&self.engine);
                    warn!(error = %e, "spawn_blocking panicked");
                    return Ok(Response::new(PExecPlanFragmentResult {
                        status: err_status(&format!("internal: {e}")),
                        ..Default::default()
                    }));
                }
            }
        }

        Ok(Response::new(PExecPlanFragmentResult {
            status: ok_status(),
            ..Default::default()
        }))
    }

    /// Two-phase execution: prepare phase.
    ///
    /// Doris FE uses a two-phase protocol: `prepare` → `start`. In Sirius, `prepare`
    /// immediately executes the full plan (async exchange tasks wait on Notify regardless).
    /// This is intentionally different from regular Doris BEs where `prepare` sets up
    /// pipeline tasks and `start` triggers them. If Sirius ever needs to support
    /// pipelined execution or coordinated multi-fragment startup, this would need to
    /// actually defer execution to `start`.
    async fn exec_plan_fragment_prepare(
        &self,
        request: Request<PExecPlanFragmentRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        self.exec_plan_fragment(request).await
    }

    /// Two-phase execution: start phase.
    ///
    /// No-op because `prepare` already triggered execution. Async exchange tasks
    /// wait on `Notify` (exchange data arrival), not on the `start` signal.
    async fn exec_plan_fragment_start(
        &self,
        request: Request<PExecPlanFragmentStartRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        let req = request.into_inner();
        info!(query_id = ?req.query_id, "exec_plan_fragment_start (no-op: already executed by prepare)");
        Ok(Response::new(PExecPlanFragmentResult {
            status: ok_status(),
            ..Default::default()
        }))
    }

    /// Cancel a running query fragment.
    ///
    /// Cleans up exchange buffer entries and result store entries for the query,
    /// unblocking any async exchange tasks waiting on Notify.
    async fn cancel_plan_fragment(
        &self,
        request: Request<PCancelPlanFragmentRequest>,
    ) -> Result<Response<PCancelPlanFragmentResult>, Status> {
        let req = request.into_inner();
        info!(query_id = ?req.query_id, fragment_id = ?req.fragment_id, "cancel_plan_fragment");
        if let Some(qid) = &req.query_id {
            self.exchange_buffer.cancel_query(qid.hi, qid.lo);
            self.result_store.remove_query(qid.hi, qid.lo);
        }
        Ok(Response::new(PCancelPlanFragmentResult {
            status: ok_status(),
        }))
    }

    async fn transmit_block(
        &self,
        request: Request<PTransmitDataParams>,
    ) -> Result<Response<PTransmitDataResult>, Status> {
        let req = request.into_inner();
        let query_id = req
            .query_id
            .as_ref()
            .map(|id| (id.hi, id.lo))
            .unwrap_or((0, 0));
        let key = ExchangeKey {
            query_id,
            node_id: req.node_id,
        };

        // Count blocks being buffered.
        let mut block_count = 0u32;

        // Buffer single block (old style).
        if let Some(block) = req.block {
            block_count += 1;
            self.exchange_buffer
                .add_block(&key, req.sender_id, Some(block), false);
        }

        // Buffer multiple blocks (new style).
        for block in req.blocks {
            block_count += 1;
            self.exchange_buffer
                .add_block(&key, req.sender_id, Some(block), false);
        }

        // Signal EOS for this sender.
        let all_done = if req.eos {
            self.exchange_buffer
                .add_block(&key, req.sender_id, None, true)
        } else {
            false
        };

        info!(
            ?query_id,
            node_id = req.node_id,
            sender_id = req.sender_id,
            block_count,
            eos = req.eos,
            all_done,
            "transmit_block"
        );

        Ok(Response::new(PTransmitDataResult {
            status: Some(ok_status()),
            ..Default::default()
        }))
    }

    async fn fetch_arrow_flight_schema(
        &self,
        request: Request<PFetchArrowFlightSchemaRequest>,
    ) -> Result<Response<PFetchArrowFlightSchemaResult>, Status> {
        let req = request.into_inner();
        info!(finst_id = ?req.finst_id, "fetch_arrow_flight_schema");

        let finst_id = match req.finst_id {
            Some(id) => FinstId { hi: id.hi, lo: id.lo },
            None => {
                return Ok(Response::new(PFetchArrowFlightSchemaResult {
                    status: Some(err_status("missing finst_id")),
                    ..Default::default()
                }));
            }
        };

        let entry = match self.result_store.get(&finst_id) {
            Some(e) => e,
            None => {
                warn!(%finst_id, "fetch_arrow_flight_schema: result not found");
                return Ok(Response::new(PFetchArrowFlightSchemaResult {
                    status: Some(err_status("result not found")),
                    ..Default::default()
                }));
            }
        };

        let schema_bytes = entry.schema_ipc_bytes().map_err(|e| {
            Status::internal(format!("failed to serialize schema: {e}"))
        })?;

        Ok(Response::new(PFetchArrowFlightSchemaResult {
            status: Some(ok_status()),
            schema: Some(schema_bytes),
            be_arrow_flight_ip: Some(self.state.advertise_host.as_bytes().to_vec()),
            be_arrow_flight_port: Some(self.state.arrow_flight_port),
        }))
    }

    async fn fetch_data(
        &self,
        request: Request<PFetchDataRequest>,
    ) -> Result<Response<PFetchDataResult>, Status> {
        let req = request.into_inner();
        let finst_id = FinstId {
            hi: req.finst_id.hi,
            lo: req.finst_id.lo,
        };
        info!(%finst_id, "fetch_data");

        // Wait for the result — execution may still be in progress when FE calls fetch_data.
        let entry = match self.result_store.wait_for(&finst_id, std::time::Duration::from_secs(60)).await {
            Some(e) => e,
            None => {
                warn!(%finst_id, "fetch_data: result not found after 60s timeout");
                return Ok(Response::new(PFetchDataResult {
                    status: err_status("result not found"),
                    eos: Some(true),
                    ..Default::default()
                }));
            }
        };

        // Check if the result is an error from async execution.
        if let Some(err_msg) = entry.error_message() {
            warn!(%finst_id, error = %err_msg, "fetch_data: returning error from async execution");
            self.result_store.remove(&finst_id);
            return Ok(Response::new(PFetchDataResult {
                status: err_status(err_msg),
                eos: Some(true),
                ..Default::default()
            }));
        }

        // Convert Arrow data to MySQL text protocol rows and wrap in TResultBatch.
        let mysql_rows = entry.to_mysql_rows();
        info!(%finst_id, num_rows = mysql_rows.len(), "converting to TResultBatch");
        let row_batch_bytes = serialize_result_batch(&mysql_rows, 0)
            .map_err(|e| Status::internal(format!("failed to serialize result batch: {e}")))?;
        info!(
            %finst_id,
            batch_len = row_batch_bytes.len(),
            first_bytes = ?&row_batch_bytes[..row_batch_bytes.len().min(32)],
            "serialized TResultBatch"
        );

        // Remove result after serving it.
        self.result_store.remove(&finst_id);

        Ok(Response::new(PFetchDataResult {
            status: ok_status(),
            packet_seq: Some(0),
            eos: Some(true),
            row_batch: Some(row_batch_bytes),
            ..Default::default()
        }))
    }

    // --- Stub implementations for unsupported methods ---

    async fn fetch_arrow_data(&self, _: Request<PFetchArrowDataRequest>) -> Result<Response<PFetchArrowDataResult>, Status> { Err(unimpl()) }
    async fn tablet_writer_open(&self, _: Request<PTabletWriterOpenRequest>) -> Result<Response<PTabletWriterOpenResult>, Status> { Err(unimpl()) }
    async fn open_load_stream(&self, _: Request<POpenLoadStreamRequest>) -> Result<Response<POpenLoadStreamResponse>, Status> { Err(unimpl()) }
    async fn tablet_writer_add_block(&self, _: Request<PTabletWriterAddBlockRequest>) -> Result<Response<PTabletWriterAddBlockResult>, Status> { Err(unimpl()) }
    async fn tablet_writer_add_block_by_http(&self, _: Request<PEmptyRequest>) -> Result<Response<PTabletWriterAddBlockResult>, Status> { Err(unimpl()) }
    async fn tablet_writer_cancel(&self, _: Request<PTabletWriterCancelRequest>) -> Result<Response<PTabletWriterCancelResult>, Status> { Err(unimpl()) }
    async fn get_info(&self, _: Request<PProxyRequest>) -> Result<Response<PProxyResult>, Status> { Err(unimpl()) }
    async fn update_cache(&self, _: Request<PUpdateCacheRequest>) -> Result<Response<PCacheResponse>, Status> { Err(unimpl()) }
    async fn fetch_cache(&self, _: Request<PFetchCacheRequest>) -> Result<Response<PFetchCacheResult>, Status> { Err(unimpl()) }
    async fn clear_cache(&self, _: Request<PClearCacheRequest>) -> Result<Response<PCacheResponse>, Status> { Err(unimpl()) }
    async fn send_data(&self, _: Request<PSendDataRequest>) -> Result<Response<PSendDataResult>, Status> { Err(unimpl()) }
    async fn commit(&self, _: Request<PCommitRequest>) -> Result<Response<PCommitResult>, Status> { Err(unimpl()) }
    async fn rollback(&self, _: Request<PRollbackRequest>) -> Result<Response<PRollbackResult>, Status> { Err(unimpl()) }
    async fn merge_filter(&self, _: Request<PMergeFilterRequest>) -> Result<Response<PMergeFilterResponse>, Status> { Err(unimpl()) }
    async fn send_filter_size(&self, _: Request<PSendFilterSizeRequest>) -> Result<Response<PSendFilterSizeResponse>, Status> { Err(unimpl()) }
    async fn sync_filter_size(&self, _: Request<PSyncFilterSizeRequest>) -> Result<Response<PSyncFilterSizeResponse>, Status> { Err(unimpl()) }
    async fn apply_filterv2(&self, _: Request<PPublishFilterRequestV2>) -> Result<Response<PPublishFilterResponse>, Status> { Err(unimpl()) }
    async fn fold_constant_expr(&self, _: Request<PConstantExprRequest>) -> Result<Response<PConstantExprResult>, Status> { Err(unimpl()) }
    async fn transmit_block_by_http(&self, _: Request<PEmptyRequest>) -> Result<Response<PTransmitDataResult>, Status> { Err(unimpl()) }
    async fn check_rpc_channel(&self, _: Request<PCheckRpcChannelRequest>) -> Result<Response<PCheckRpcChannelResponse>, Status> { Err(unimpl()) }
    async fn reset_rpc_channel(&self, _: Request<PResetRpcChannelRequest>) -> Result<Response<PResetRpcChannelResponse>, Status> { Err(unimpl()) }
    async fn hand_shake(&self, _: Request<PHandShakeRequest>) -> Result<Response<PHandShakeResponse>, Status> { Err(unimpl()) }
    async fn request_slave_tablet_pull_rowset(&self, _: Request<PTabletWriteSlaveRequest>) -> Result<Response<PTabletWriteSlaveResult>, Status> { Err(unimpl()) }
    async fn response_slave_tablet_pull_rowset(&self, _: Request<PTabletWriteSlaveDoneRequest>) -> Result<Response<PTabletWriteSlaveDoneResult>, Status> { Err(unimpl()) }
    async fn outfile_write_success(&self, _: Request<POutfileWriteSuccessRequest>) -> Result<Response<POutfileWriteSuccessResult>, Status> { Err(unimpl()) }
    async fn fetch_table_schema(
        &self,
        request: Request<PFetchTableSchemaRequest>,
    ) -> Result<Response<PFetchTableSchemaResult>, Status> {
        let req = request.into_inner();
        info!("fetch_table_schema");

        // Deserialize TFileScanRange from Thrift compact-encoded bytes.
        let scan_range_bytes = match req.file_scan_range {
            Some(b) => b,
            None => {
                warn!("fetch_table_schema: missing file_scan_range");
                return Ok(Response::new(PFetchTableSchemaResult {
                    status: Some(err_status("missing file_scan_range")),
                    ..Default::default()
                }));
            }
        };

        let (file_path, file_format) = {
            use thrift::protocol::{TCompactInputProtocol, TBinaryInputProtocol, TSerializable};
            use thrift::transport::TBufferedReadTransport;

            info!(
                len = scan_range_bytes.len(),
                first_bytes = ?&scan_range_bytes[..scan_range_bytes.len().min(64)],
                "fetch_table_schema: raw scan_range bytes"
            );

            // Helper: extract path and format from a deserialized TFileScanRange.
            let extract = |fsr: &doris_thrift::plan_nodes::TFileScanRange| -> (Option<String>, Option<String>) {
                let path_from_ranges = fsr.ranges
                    .as_ref()
                    .and_then(|r| r.first())
                    .and_then(|rd| rd.path.clone());
                let path_from_params = fsr.params
                    .as_ref()
                    .and_then(|p| p.properties.as_ref())
                    .and_then(|props| props.get("file_path").cloned());

                // Extract format from range desc's format_type field.
                let format = fsr.ranges
                    .as_ref()
                    .and_then(|r| r.first())
                    .and_then(|rd| rd.format_type.as_ref())
                    .map(|ft| match *ft {
                        TFileFormatType::FORMAT_PARQUET => "parquet".to_string(),
                        TFileFormatType::FORMAT_ORC => "orc".to_string(),
                        TFileFormatType::FORMAT_JSON => "json".to_string(),
                        TFileFormatType::FORMAT_CSV_PLAIN => "csv".to_string(),
                        other => {
                            warn!(format_type = ?other, "fetch_table_schema: unknown format_type, defaulting to parquet");
                            "parquet".to_string()
                        }
                    });

                (path_from_ranges.or(path_from_params), format)
            };

            // Try compact first, then binary protocol.
            let transport = TBufferedReadTransport::new(Box::new(Cursor::new(scan_range_bytes.clone())));
            let mut protocol = TCompactInputProtocol::new(transport);
            match doris_thrift::plan_nodes::TFileScanRange::read_from_in_protocol(&mut protocol) {
                Ok(fsr) => {
                    info!(
                        has_ranges = fsr.ranges.is_some(),
                        num_ranges = fsr.ranges.as_ref().map(|r| r.len()).unwrap_or(0),
                        has_params = fsr.params.is_some(),
                        "fetch_table_schema: deserialized TFileScanRange"
                    );
                    extract(&fsr)
                }
                Err(e) => {
                    warn!(error = %e, "fetch_table_schema: compact deser failed, trying binary");
                    let transport = TBufferedReadTransport::new(Box::new(Cursor::new(scan_range_bytes)));
                    let mut protocol = TBinaryInputProtocol::new(transport, true);
                    match doris_thrift::plan_nodes::TFileScanRange::read_from_in_protocol(&mut protocol) {
                        Ok(fsr) => {
                            info!(
                                has_ranges = fsr.ranges.is_some(),
                                num_ranges = fsr.ranges.as_ref().map(|r| r.len()).unwrap_or(0),
                                "fetch_table_schema: deserialized TFileScanRange (binary)"
                            );
                            extract(&fsr)
                        }
                        Err(e2) => {
                            warn!(error = %e2, "fetch_table_schema: binary deser also failed");
                            (None, None)
                        }
                    }
                }
            }
        };

        let file_path = match file_path {
            Some(p) => {
                // Strip file:// scheme if present (FE sends file:// URLs for local paths).
                p.strip_prefix("file://").unwrap_or(&p).to_string()
            }
            None => {
                warn!("fetch_table_schema: no file path in TFileScanRange");
                return Ok(Response::new(PFetchTableSchemaResult {
                    status: Some(err_status("no file path in scan range")),
                    ..Default::default()
                }));
            }
        };

        // Determine file format: from TFileScanRange, or infer from file extension.
        let format = file_format.unwrap_or_else(|| {
            let ext_format = if file_path.ends_with(".parquet") || file_path.ends_with(".pq") {
                "parquet"
            } else if file_path.ends_with(".csv") || file_path.ends_with(".tsv") {
                "csv"
            } else if file_path.ends_with(".json") || file_path.ends_with(".jsonl") {
                "json"
            } else if file_path.ends_with(".orc") {
                "orc"
            } else {
                warn!(path = %file_path, "fetch_table_schema: no format_type in scan range, inferring parquet from extension");
                "parquet"
            };
            ext_format.to_string()
        });
        info!(path = %file_path, format = %format, "fetch_table_schema: reading schema");

        // Use DuckDB to read the file schema via a LIMIT 0 query (gets schema without data).
        let engine = match &self.engine {
            Some(e) => e.clone(),
            None => {
                return Ok(Response::new(PFetchTableSchemaResult {
                    status: Some(err_status("engine not initialized")),
                    ..Default::default()
                }));
            }
        };

        let result = tokio::task::spawn_blocking(move || -> Result<(Vec<String>, Vec<PTypeDesc>), String> {
            let engine = engine.lock().unwrap();
            // Get schema via prepared statement metadata (no data read needed).
            let ipc_bytes = engine
                .get_file_schema_ipc(&file_path, &format)
                .map_err(|e| e.to_string())?;

            if ipc_bytes.is_empty() {
                return Err("empty schema IPC from get_file_schema_ipc".to_string());
            }

            // Parse Arrow IPC to get schema.
            use arrow::ipc::reader::StreamReader;
            let reader = StreamReader::try_new(std::io::Cursor::new(&ipc_bytes), None)
                .map_err(|e| format!("parse Arrow IPC: {e}"))?;
            let schema = reader.schema();

            let mut names = Vec::new();
            let mut types = Vec::new();
            for field in schema.fields() {
                names.push(field.name().clone());
                types.push(arrow_type_to_doris(field.data_type()));
            }
            Ok((names, types))
        }).await.map_err(|e| Status::internal(format!("spawn_blocking: {e}")))?
            .map_err(|e| {
                warn!(error = %e, "fetch_table_schema: failed");
                Status::internal(format!("describe: {e}"))
            })?;

        let (column_names, column_types) = result;
        let column_nums = column_names.len() as i32;

        info!(columns = column_nums, names = ?column_names, "fetch_table_schema: got schema");

        Ok(Response::new(PFetchTableSchemaResult {
            status: Some(ok_status()),
            column_nums: Some(column_nums),
            column_names,
            column_types,
        }))
    }
    async fn multiget_data(&self, _: Request<PMultiGetRequest>) -> Result<Response<PMultiGetResponse>, Status> { Err(unimpl()) }
    async fn multiget_data_v2(&self, _: Request<PMultiGetRequestV2>) -> Result<Response<PMultiGetResponseV2>, Status> { Err(unimpl()) }
    async fn get_file_cache_meta_by_tablet_id(&self, _: Request<PGetFileCacheMetaRequest>) -> Result<Response<PGetFileCacheMetaResponse>, Status> { Err(unimpl()) }
    async fn warm_up_rowset(&self, _: Request<PWarmUpRowsetRequest>) -> Result<Response<PWarmUpRowsetResponse>, Status> { Err(unimpl()) }
    async fn recycle_cache(&self, _: Request<PRecycleCacheRequest>) -> Result<Response<PRecycleCacheResponse>, Status> { Err(unimpl()) }
    async fn tablet_fetch_data(&self, _: Request<PTabletKeyLookupRequest>) -> Result<Response<PTabletKeyLookupResponse>, Status> { Err(unimpl()) }
    async fn get_column_ids_by_tablet_ids(&self, _: Request<PFetchColIdsRequest>) -> Result<Response<PFetchColIdsResponse>, Status> { Err(unimpl()) }
    async fn get_tablet_rowset_versions(&self, _: Request<PGetTabletVersionsRequest>) -> Result<Response<PGetTabletVersionsResponse>, Status> { Err(unimpl()) }
    async fn report_stream_load_status(&self, _: Request<PReportStreamLoadStatusRequest>) -> Result<Response<PReportStreamLoadStatusResponse>, Status> { Err(unimpl()) }
    async fn glob(&self, request: Request<PGlobRequest>) -> Result<Response<PGlobResponse>, Status> {
        let req = request.into_inner();
        let pattern = req.pattern.unwrap_or_default();
        info!(pattern = %pattern, "glob");

        // Strip file:// scheme if present (FE sends file:// URLs for local paths).
        let had_scheme = pattern.starts_with("file://");
        let path = pattern.strip_prefix("file://").unwrap_or(&pattern);

        let mut files = vec![];

        // Check if the pattern contains glob wildcards.
        if path.contains('*') || path.contains('?') || path.contains('[') {
            match glob::glob(path) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        if let Ok(meta) = std::fs::metadata(&entry) {
                            if meta.is_file() {
                                let file_path = if had_scheme {
                                    format!("file://{}", entry.display())
                                } else {
                                    entry.display().to_string()
                                };
                                files.push(p_glob_response::PFileInfo {
                                    file: Some(file_path),
                                    size: Some(meta.len() as i64),
                                });
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(pattern = %pattern, error = %e, "glob: pattern error");
                }
            }
        } else {
            // No wildcards — stat the path.
            match std::fs::metadata(path) {
                Ok(meta) if meta.is_dir() => {
                    // Directory: enumerate files inside it so the FE can
                    // distribute individual file scan ranges across BEs.
                    match std::fs::read_dir(path) {
                        Ok(entries) => {
                            for entry in entries.flatten() {
                                if let Ok(m) = entry.metadata() {
                                    if m.is_file() {
                                        let file_path = if had_scheme {
                                            format!("file://{}", entry.path().display())
                                        } else {
                                            entry.path().display().to_string()
                                        };
                                        files.push(p_glob_response::PFileInfo {
                                            file: Some(file_path),
                                            size: Some(m.len() as i64),
                                        });
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            warn!(pattern = %pattern, error = %e, "glob: failed to read directory");
                        }
                    }
                }
                Ok(meta) => {
                    files.push(p_glob_response::PFileInfo {
                        file: Some(pattern.clone()),
                        size: Some(meta.len() as i64),
                    });
                }
                Err(e) => {
                    warn!(pattern = %pattern, error = %e, "glob: file not found");
                }
            }
        }

        info!(pattern = %pattern, count = files.len(), "glob result");

        Ok(Response::new(PGlobResponse {
            status: ok_status(),
            files,
        }))
    }
    async fn group_commit_insert(&self, _: Request<PGroupCommitInsertRequest>) -> Result<Response<PGroupCommitInsertResponse>, Status> { Err(unimpl()) }
    async fn get_wal_queue_size(&self, _: Request<PGetWalQueueSizeRequest>) -> Result<Response<PGetWalQueueSizeResponse>, Status> { Err(unimpl()) }
    async fn fetch_remote_tablet_schema(&self, _: Request<PFetchRemoteSchemaRequest>) -> Result<Response<PFetchRemoteSchemaResponse>, Status> { Err(unimpl()) }
    async fn test_jdbc_connection(&self, _: Request<PJdbcTestConnectionRequest>) -> Result<Response<PJdbcTestConnectionResult>, Status> { Err(unimpl()) }
    async fn alter_vault_sync(&self, _: Request<PAlterVaultSyncRequest>) -> Result<Response<PAlterVaultSyncResponse>, Status> { Err(unimpl()) }
    async fn get_be_resource(&self, _: Request<PGetBeResourceRequest>) -> Result<Response<PGetBeResourceResponse>, Status> { Err(unimpl()) }
    async fn delete_dictionary(&self, _: Request<PDeleteDictionaryRequest>) -> Result<Response<PDeleteDictionaryResponse>, Status> { Err(unimpl()) }
    async fn commit_refresh_dictionary(&self, _: Request<PCommitRefreshDictionaryRequest>) -> Result<Response<PCommitRefreshDictionaryResponse>, Status> { Err(unimpl()) }
    async fn abort_refresh_dictionary(&self, _: Request<PAbortRefreshDictionaryRequest>) -> Result<Response<PAbortRefreshDictionaryResponse>, Status> { Err(unimpl()) }
    async fn get_tablet_rowsets(&self, _: Request<PGetTabletRowsetsRequest>) -> Result<Response<PGetTabletRowsetsResponse>, Status> { Err(unimpl()) }
    async fn fetch_peer_data(&self, _: Request<PFetchPeerDataRequest>) -> Result<Response<PFetchPeerDataResponse>, Status> { Err(unimpl()) }
    async fn request_cdc_client(&self, _: Request<PRequestCdcClientRequest>) -> Result<Response<PRequestCdcClientResult>, Status> { Err(unimpl()) }
}

/// Start the PBackendService server with multi-protocol support.
///
/// Listens on a single TCP port and dispatches connections based on protocol:
/// - bRPC "baidu_std" (magic "PRPC"): inter-BE exchange (transmit_block)
/// - gRPC / HTTP/2 (magic "PRI "): FE→BE calls (exec_plan_fragment, etc.)
pub async fn start_grpc_server(
    listen_addr: &str,
    state: Arc<BeState>,
    result_store: ResultStore,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
    exchange_buffer: ExchangeBuffer,
    no_cpu_fallback: bool,
    force_cpu: bool,
    nixl_only: bool,
    nixl_agent: Option<Arc<super::nixl_exchange::NixlExchange>>,
    local_brpc_addr: String,
) -> Result<(), Box<dyn std::error::Error>> {
    use doris_proto::doris::p_backend_service_server::PBackendServiceServer;
    use tokio_stream::wrappers::ReceiverStream;
    use tokio_stream::StreamExt;

    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    info!(addr = listen_addr, "starting PBackendService multi-protocol server (gRPC + bRPC)");

    // Clone before moving into handler (nixl service also needs these).
    let nixl_agent_for_service = nixl_agent.clone();

    let mut handler = PBackendServiceHandler::new(
        state,
        result_store,
        engine,
        exchange_buffer.clone(),
        no_cpu_fallback,
        force_cpu,
        nixl_only,
    );

    let dispatcher = super::transfer_engine::TransferDispatcher::standard(
        nixl_agent.clone(),
        nixl_only,
    );
    handler = handler
        .with_nixl_agent(nixl_agent)
        .with_transfer_dispatcher(dispatcher)
        .with_local_brpc_addr(local_brpc_addr);

    let svc = PBackendServiceServer::new(handler);

    // Channel to feed gRPC connections to tonic's serve_with_incoming.
    let (grpc_tx, grpc_rx) =
        tokio::sync::mpsc::channel::<tokio::net::TcpStream>(64);

    // Build gRPC server with PBackendService + NixlMetadataService.
    let mut server_builder = tonic::transport::Server::builder()
        .add_service(svc);

    {
        use doris_proto::nixl::NixlMetadataServiceServer;
        let nixl_handler = super::nixl_service::NixlMetadataServiceHandler::new(
            nixl_agent_for_service,
            exchange_buffer.clone(),
        );
        // Increase message size limits for large Arrow IPC payloads in transfer_complete.
        let nixl_svc = NixlMetadataServiceServer::new(nixl_handler)
            .max_decoding_message_size(256 * 1024 * 1024)   // 256 MB
            .max_encoding_message_size(256 * 1024 * 1024);
        server_builder = server_builder.add_service(nixl_svc);
        info!("registered NixlMetadataService on gRPC server");
    }

    // Spawn tonic gRPC server consuming forwarded connections.
    tokio::spawn(async move {
        let incoming = ReceiverStream::new(grpc_rx).map(Ok::<_, std::io::Error>);
        if let Err(e) = server_builder
            .serve_with_incoming(incoming)
            .await
        {
            tracing::error!(error = %e, "tonic gRPC server error");
        }
    });

    // Accept loop: peek first 4 bytes to detect protocol.
    loop {
        let (socket, peer) = listener.accept().await?;
        let mut peek_buf = [0u8; 4];
        match socket.peek(&mut peek_buf).await {
            Ok(n) if n >= 4 && &peek_buf == b"PRPC" => {
                let buf = exchange_buffer.clone();
                tokio::spawn(super::brpc_server::handle_brpc_connection(socket, buf));
            }
            _ => {
                // Assume gRPC (HTTP/2 starts with "PRI * HTTP/2.0...")
                if grpc_tx.send(socket).await.is_err() {
                    warn!(%peer, "gRPC channel closed, dropping connection");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use doris_thrift::data_sinks::{TDataSink, TDataSinkType, TDataStreamSink};
    use doris_thrift::partitions::{TDataPartition, TPartitionType};
    use doris_thrift::plan_nodes::{TPlan, TPlanNode, TPlanNodeType};
    use doris_thrift::types::{TNetworkAddress, TUniqueId};
    use std::collections::BTreeMap;

    /// Create a minimal TPipelineFragmentParams with the given plan nodes.
    fn make_params(nodes: Vec<TPlanNode>) -> TPipelineFragmentParams {
        TPipelineFragmentParams {
            protocol_version:
                doris_thrift::palo_internal_service::PaloInternalServiceVersion::V1,
            query_id: TUniqueId { hi: 1, lo: 2 },
            fragment_id: Some(0),
            per_exch_num_senders: BTreeMap::new(),
            desc_tbl: None,
            resource_info: None,
            destinations: None,
            num_senders: None,
            send_query_statistics_with_every_batch: None,
            coord: None,
            query_globals: None,
            query_options: None,
            import_label: None,
            db_name: None,
            load_job_id: None,
            load_error_hub_info: None,
            fragment_num_on_host: None,
            backend_id: None,
            need_wait_execution_trigger: None,
            instances_sharing_hash_table: None,
            is_simplified_param: None,
            global_dict: None,
            fragment: Some(doris_thrift::planner::TPlanFragment {
                plan: Some(TPlan { nodes }),
                output_exprs: None,
                output_sink: None,
                partition: TDataPartition {
                    type_: TPartitionType::UNPARTITIONED,
                    partition_exprs: None,
                    partition_infos: None,
                },
                min_reservation_bytes: None,
                initial_reservation_total_claims: None,
                query_cache_param: None,
            }),
            local_params: None,
            workload_groups: None,
            txn_conf: None,
            table_name: None,
            file_scan_params: None,
            group_commit: None,
            load_stream_per_node: None,
            total_load_streams: None,
            num_local_sink: None,
            num_buckets: None,
            bucket_seq_to_instance_idx: None,
            per_node_shared_scans: None,
            parallel_instances: None,
            total_instances: None,
            shuffle_idx_to_instance_idx: None,
            is_nereids: None,
            wal_id: None,
            content_length: None,
            current_connect_fe: None,
            topn_filter_source_node_ids: None,
            ai_resources: None,
            is_mow_table: None,
        }
    }

    fn make_node(node_id: i32, node_type: TPlanNodeType, num_children: i32) -> TPlanNode {
        TPlanNode {
            node_id,
            node_type,
            num_children,
            limit: -1,
            row_tuples: vec![0],
            nullable_tuples: vec![],
            conjuncts: None,
            compact_data: false,
            hash_join_node: None,
            agg_node: None,
            sort_node: None,
            merge_node: None,
            exchange_node: None,
            mysql_scan_node: None,
            olap_scan_node: None,
            csv_scan_node: None,
            broker_scan_node: None,
            pre_agg_node: None,
            schema_scan_node: None,
            merge_join_node: None,
            meta_scan_node: None,
            analytic_node: None,
            olap_rewrite_node: None,
            union_node: None,
            resource_profile: None,
            es_scan_node: None,
            repeat_node: None,
            assert_num_rows_node: None,
            intersect_node: None,
            except_node: None,
            odbc_scan_node: None,
            runtime_filters: None,
            group_commit_scan_node: None,
            materialization_node: None,
            vconjunct: None,
            table_function_node: None,
            output_slot_ids: None,
            data_gen_scan_node: None,
            file_scan_node: None,
            jdbc_scan_node: None,
            nested_loop_join_node: None,
            test_external_scan_node: None,
            push_down_agg_type_opt: None,
            push_down_count: None,
            distribute_expr_lists: None,
            is_serial_operator: None,
            projections: None,
            output_tuple_id: None,
            partition_sort_node: None,
            intermediate_projections_list: None,
            intermediate_output_tuple_id_list: None,
            topn_filter_source_node_ids: None,
            nereids_id: None,
        }
    }

    fn make_data_stream_sink(dest_node_id: i32) -> TDataSink {
        TDataSink {
            type_: TDataSinkType::DATA_STREAM_SINK,
            stream_sink: Some(TDataStreamSink {
                dest_node_id,
                output_partition: TDataPartition {
                    type_: TPartitionType::UNPARTITIONED,
                    partition_exprs: None,
                    partition_infos: None,
                },
                ignore_not_found: None,
                output_exprs: None,
                output_tuple_id: None,
                conjuncts: None,
                runtime_filters: None,
                tablet_sink_schema: None,
                tablet_sink_partition: None,
                tablet_sink_location: None,
                tablet_sink_txn_id: None,
                tablet_sink_tuple_id: None,
                tablet_sink_exprs: None,
                is_merge: None,
            }),
            result_sink: None,
            mysql_table_sink: None,
            export_sink: None,
            olap_table_sink: None,
            memory_scratch_sink: None,
            odbc_table_sink: None,
            result_file_sink: None,
            jdbc_table_sink: None,
            multi_cast_stream_sink: None,
            hive_table_sink: None,
            iceberg_table_sink: None,
            dictionary_sink: None,
            blackhole_sink: None,
        }
    }

    // --- has_unresolved_exchanges ---

    #[test]
    fn test_has_unresolved_exchanges_empty_plan() {
        let params = make_params(vec![]);
        assert!(has_unresolved_exchanges(&params).is_empty());
    }

    #[test]
    fn test_has_unresolved_exchanges_no_exchanges() {
        let params = make_params(vec![
            make_node(0, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);
        assert!(has_unresolved_exchanges(&params).is_empty());
    }

    #[test]
    fn test_has_unresolved_exchanges_sender_not_detected() {
        // EXCHANGE_NODE(1 child) = sender, not unresolved.
        let params = make_params(vec![
            make_node(0, TPlanNodeType::EXCHANGE_NODE, 1),
            make_node(1, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);
        assert!(has_unresolved_exchanges(&params).is_empty());
    }

    #[test]
    fn test_has_unresolved_exchanges_receiver_detected() {
        let params = make_params(vec![
            make_node(5, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        assert_eq!(has_unresolved_exchanges(&params), vec![5]);
    }

    #[test]
    fn test_has_unresolved_exchanges_multiple_receivers() {
        // UNION ALL: two EXCHANGE_NODE(0 children) under a UNION_NODE.
        let params = make_params(vec![
            make_node(0, TPlanNodeType::UNION_NODE, 2),
            make_node(1, TPlanNodeType::EXCHANGE_NODE, 0),
            make_node(3, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        assert_eq!(has_unresolved_exchanges(&params), vec![1, 3]);
    }

    #[test]
    fn test_has_unresolved_exchanges_no_fragment() {
        let mut params = make_params(vec![]);
        params.fragment = None;
        assert!(has_unresolved_exchanges(&params).is_empty());
    }

    // --- get_num_senders ---

    #[test]
    fn test_get_num_senders_from_per_exch_map() {
        let mut params = make_params(vec![]);
        params.per_exch_num_senders.insert(5, 3);
        assert_eq!(get_num_senders(&params, 5), 3);
    }

    #[test]
    fn test_get_num_senders_fallback_to_num_senders() {
        let mut params = make_params(vec![]);
        params.num_senders = Some(7);
        assert_eq!(get_num_senders(&params, 99), 7);
    }

    #[test]
    fn test_get_num_senders_default_one() {
        let params = make_params(vec![]);
        assert_eq!(get_num_senders(&params, 99), 1);
    }

    #[test]
    fn test_get_num_senders_per_exch_takes_priority() {
        let mut params = make_params(vec![]);
        params.per_exch_num_senders.insert(5, 2);
        params.num_senders = Some(10);
        // per_exch_num_senders should be preferred over num_senders.
        assert_eq!(get_num_senders(&params, 5), 2);
        // Fallback for unknown node_id.
        assert_eq!(get_num_senders(&params, 99), 10);
    }

    // --- extract_exchange_destinations ---

    #[test]
    fn test_extract_exchange_destinations_none_without_sink() {
        let params = make_params(vec![make_node(0, TPlanNodeType::FILE_SCAN_NODE, 0)]);
        assert!(extract_exchange_destinations(&params).is_none());
    }

    #[test]
    fn test_extract_exchange_destinations_none_for_result_sink() {
        let mut params = make_params(vec![make_node(0, TPlanNodeType::FILE_SCAN_NODE, 0)]);
        params.fragment.as_mut().unwrap().output_sink = Some(TDataSink {
            type_: TDataSinkType::RESULT_SINK,
            stream_sink: None,
            result_sink: None,
            mysql_table_sink: None,
            export_sink: None,
            olap_table_sink: None,
            memory_scratch_sink: None,
            odbc_table_sink: None,
            result_file_sink: None,
            jdbc_table_sink: None,
            multi_cast_stream_sink: None,
            hive_table_sink: None,
            iceberg_table_sink: None,
            dictionary_sink: None,
            blackhole_sink: None,
        });
        assert!(extract_exchange_destinations(&params).is_none());
    }

    #[test]
    fn test_extract_exchange_destinations_with_brpc_server() {
        let mut params = make_params(vec![make_node(0, TPlanNodeType::FILE_SCAN_NODE, 0)]);
        params.fragment.as_mut().unwrap().output_sink = Some(make_data_stream_sink(7));
        params.destinations = Some(vec![
            doris_thrift::data_sinks::TPlanFragmentDestination {
                fragment_instance_id: TUniqueId { hi: 10, lo: 20 },
                server: TNetworkAddress {
                    hostname: "192.168.1.1".to_string(),
                    port: 9060,
                },
                brpc_server: Some(TNetworkAddress {
                    hostname: "192.168.1.1".to_string(),
                    port: 8060,
                }),
            },
        ]);

        let info = extract_exchange_destinations(&params).unwrap();
        assert_eq!(info.dest_node_id, 7);
        assert_eq!(info.destinations.len(), 1);
        assert_eq!(info.destinations[0].brpc_addr, "192.168.1.1:8060");
        assert_eq!(info.destinations[0].finst_id, (10, 20));
    }

    #[test]
    fn test_extract_exchange_destinations_fallback_to_server_addr() {
        let mut params = make_params(vec![make_node(0, TPlanNodeType::FILE_SCAN_NODE, 0)]);
        params.fragment.as_mut().unwrap().output_sink = Some(make_data_stream_sink(1));
        params.destinations = Some(vec![
            doris_thrift::data_sinks::TPlanFragmentDestination {
                fragment_instance_id: TUniqueId { hi: 30, lo: 40 },
                server: TNetworkAddress {
                    hostname: "10.0.0.1".to_string(),
                    port: 9060,
                },
                brpc_server: None,
            },
        ]);

        let info = extract_exchange_destinations(&params).unwrap();
        assert_eq!(info.destinations[0].brpc_addr, "10.0.0.1:9060");
    }

    #[test]
    fn test_extract_exchange_destinations_empty_destinations() {
        let mut params = make_params(vec![make_node(0, TPlanNodeType::FILE_SCAN_NODE, 0)]);
        params.fragment.as_mut().unwrap().output_sink = Some(make_data_stream_sink(1));
        params.destinations = Some(vec![]);
        assert!(extract_exchange_destinations(&params).is_none());
    }

    #[test]
    fn test_extract_exchange_destinations_multiple_dests() {
        let mut params = make_params(vec![make_node(0, TPlanNodeType::FILE_SCAN_NODE, 0)]);
        params.fragment.as_mut().unwrap().output_sink = Some(make_data_stream_sink(5));
        params.destinations = Some(vec![
            doris_thrift::data_sinks::TPlanFragmentDestination {
                fragment_instance_id: TUniqueId { hi: 1, lo: 2 },
                server: TNetworkAddress { hostname: "be1".to_string(), port: 8060 },
                brpc_server: None,
            },
            doris_thrift::data_sinks::TPlanFragmentDestination {
                fragment_instance_id: TUniqueId { hi: 3, lo: 4 },
                server: TNetworkAddress { hostname: "be2".to_string(), port: 8060 },
                brpc_server: None,
            },
        ]);

        let info = extract_exchange_destinations(&params).unwrap();
        assert_eq!(info.dest_node_id, 5);
        assert_eq!(info.destinations.len(), 2);
        assert_eq!(info.destinations[0].brpc_addr, "be1:8060");
        assert_eq!(info.destinations[1].brpc_addr, "be2:8060");
    }

    // --- merge_fragment_plans ---

    #[test]
    fn test_merge_single_leaf_fragment() {
        let params = make_params(vec![
            make_node(0, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);
        let merged = merge_fragment_plans(&[params]);
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_merge_exchange_root_with_leaf() {
        // Fragment 0: EXCHANGE_NODE(0 children) — result collector
        // Fragment 1: FILE_SCAN_NODE — leaf scan
        let root = make_params(vec![
            make_node(0, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        let leaf = make_params(vec![
            make_node(1, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);
        let merged = merge_fragment_plans(&[root, leaf]);
        assert_eq!(merged.len(), 1);
        let plan = merged[0].fragment.as_ref().unwrap().plan.as_ref().unwrap();
        assert!(plan.nodes.iter().any(|n| n.node_type == TPlanNodeType::FILE_SCAN_NODE));
        assert!(!plan.nodes.iter().any(|n| n.node_type == TPlanNodeType::EXCHANGE_NODE));
    }

    #[test]
    fn test_merge_intermediate_with_leaf() {
        // Fragment 0: SORT_NODE → EXCHANGE_NODE(0) — intermediate
        // Fragment 1: FILE_SCAN_NODE — leaf
        let intermediate = make_params(vec![
            make_node(0, TPlanNodeType::SORT_NODE, 1),
            make_node(1, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        let leaf = make_params(vec![
            make_node(2, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);
        let merged = merge_fragment_plans(&[intermediate, leaf]);
        assert_eq!(merged.len(), 1);
        let plan = merged[0].fragment.as_ref().unwrap().plan.as_ref().unwrap();
        assert_eq!(plan.nodes[0].node_type, TPlanNodeType::SORT_NODE);
        assert_eq!(plan.nodes[1].node_type, TPlanNodeType::FILE_SCAN_NODE);
    }

    #[test]
    fn test_merge_skipped_for_remote_exchanges() {
        // UNION ALL: exchange-root has 2 EXCHANGE_NODE(0) but only 1 leaf.
        let mut exchange_root = make_params(vec![
            make_node(0, TPlanNodeType::UNION_NODE, 2),
            make_node(1, TPlanNodeType::EXCHANGE_NODE, 0),
            make_node(3, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        exchange_root.per_exch_num_senders.insert(1, 1);
        exchange_root.per_exch_num_senders.insert(3, 1);

        let leaf = make_params(vec![
            make_node(2, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);

        let merged = merge_fragment_plans(&[exchange_root, leaf]);
        // exchange_count (2) > leaf_count (1) → skip merge.
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merge_empty_input() {
        let merged = merge_fragment_plans(&[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_only_exchange_root_no_leaf() {
        // Exchange-root with no leaf = depends on remote data.
        let root = make_params(vec![
            make_node(0, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        let merged = merge_fragment_plans(&[root]);
        assert_eq!(merged.len(), 1);
        let plan = merged[0].fragment.as_ref().unwrap().plan.as_ref().unwrap();
        assert_eq!(plan.nodes[0].node_type, TPlanNodeType::EXCHANGE_NODE);
    }

    #[test]
    fn test_merge_three_fragment_pipeline() {
        // Typical ORDER BY + LIMIT pipeline:
        // Fragment 0: EXCHANGE_NODE(0) — result collector
        // Fragment 1: SORT_NODE → EXCHANGE_NODE(0) — intermediate
        // Fragment 2: FILE_SCAN_NODE — leaf
        let root = make_params(vec![
            make_node(0, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        let intermediate = make_params(vec![
            make_node(1, TPlanNodeType::SORT_NODE, 1),
            make_node(2, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        let leaf = make_params(vec![
            make_node(3, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);
        let merged = merge_fragment_plans(&[root, intermediate, leaf]);
        // Result: single merged fragment (SORT_NODE → FILE_SCAN_NODE).
        assert_eq!(merged.len(), 1);
        let plan = merged[0].fragment.as_ref().unwrap().plan.as_ref().unwrap();
        assert_eq!(plan.nodes[0].node_type, TPlanNodeType::SORT_NODE);
        assert_eq!(plan.nodes[1].node_type, TPlanNodeType::FILE_SCAN_NODE);
    }

    #[test]
    fn test_merge_three_fragment_all_intermediate() {
        // GROUP BY + ORDER BY pipeline (no pure exchange_root):
        // Fragment 0: SORT_NODE → EXCHANGE_NODE(1) — outermost intermediate
        // Fragment 1: AGG_NODE → EXCHANGE_NODE(3) — innermost intermediate, sends to node 1
        // Fragment 2: FILE_SCAN_NODE — leaf, sends to node 3
        let mut outer = make_params(vec![
            make_node(0, TPlanNodeType::SORT_NODE, 1),
            make_node(1, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        outer.fragment_id = Some(0);

        let mut inner = make_params(vec![
            make_node(2, TPlanNodeType::AGGREGATION_NODE, 1),
            make_node(3, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        inner.fragment_id = Some(1);
        // inner sends its output to exchange node 1 in outer
        inner.fragment.as_mut().unwrap().output_sink = Some(make_data_stream_sink(1));

        let mut leaf = make_params(vec![
            make_node(4, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);
        leaf.fragment_id = Some(2);
        // leaf sends its output to exchange node 3 in inner
        leaf.fragment.as_mut().unwrap().output_sink = Some(make_data_stream_sink(3));

        let merged = merge_fragment_plans(&[outer, inner, leaf]);
        // Should cascade: leaf → inner → outer, producing 1 merged fragment.
        assert_eq!(merged.len(), 1);
        let plan = merged[0].fragment.as_ref().unwrap().plan.as_ref().unwrap();
        // SORT → AGG → FILE_SCAN (3 nodes, exchanges replaced).
        assert_eq!(plan.nodes.len(), 3);
        assert_eq!(plan.nodes[0].node_type, TPlanNodeType::SORT_NODE);
        assert_eq!(plan.nodes[1].node_type, TPlanNodeType::AGGREGATION_NODE);
        assert_eq!(plan.nodes[2].node_type, TPlanNodeType::FILE_SCAN_NODE);
    }

    #[test]
    fn test_merge_skipped_when_per_exch_senders_exceed_leaves() {
        // Single exchange node with 2 senders but only 1 leaf → remote.
        let mut root = make_params(vec![
            make_node(0, TPlanNodeType::EXCHANGE_NODE, 0),
        ]);
        root.per_exch_num_senders.insert(0, 2);

        let leaf = make_params(vec![
            make_node(1, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);

        let merged = merge_fragment_plans(&[root, leaf]);
        // per_exch_num_senders[0] = 2 > 1 leaf → skip merge.
        assert_eq!(merged.len(), 2);
    }

    // --- exchange_table_name ---

    #[test]
    fn test_exchange_table_name_unique_per_query() {
        let name1 = exchange_table_name(0x12345678, 1);
        let name2 = exchange_table_name(0xABCDEF01, 1);
        // Same node_id, different queries → different table names.
        assert_ne!(name1, name2);
        assert!(name1.starts_with("__EXCH_"));
        assert!(name1.ends_with("_1"));
    }

    #[test]
    fn test_exchange_table_name_format() {
        let name = exchange_table_name(0x00FF00FF, 42);
        assert_eq!(name, "__EXCH_00ff00ff_42");
    }
}
