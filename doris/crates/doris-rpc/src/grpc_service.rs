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
use doris_thrift::plan_nodes::{TFileFormatType, TPlanNodeType};
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
fn find_result_finst_id(all_params: &[TPipelineFragmentParams]) -> Option<FinstId> {
    // Strategy 1: find fragment with fragment_id == 0.
    for params in all_params {
        if params.fragment_id == Some(0) {
            if let Some(lp) = params.local_params.as_ref().and_then(|lp| lp.first()) {
                return Some(FinstId {
                    hi: lp.fragment_instance_id.hi,
                    lo: lp.fragment_instance_id.lo,
                });
            }
        }
    }
    // Strategy 2: find the exchange_root fragment (EXCHANGE_NODE at root, 0 children).
    for params in all_params {
        let is_exchange_root = params
            .fragment.as_ref()
            .and_then(|f| f.plan.as_ref())
            .and_then(|p| p.nodes.first())
            .map(|n| n.node_type == TPlanNodeType::EXCHANGE_NODE && n.num_children == 0)
            .unwrap_or(false);
        if is_exchange_root {
            if let Some(lp) = params.local_params.as_ref().and_then(|lp| lp.first()) {
                return Some(FinstId {
                    hi: lp.fragment_instance_id.hi,
                    lo: lp.fragment_instance_id.lo,
                });
            }
        }
    }
    None
}

fn extract_finst_id(params: &TPipelineFragmentParams) -> FinstId {
    // FE calls fetch_data with local_params[0].fragment_instance_id, not query_id.
    // For single-fragment queries these are often the same, but for multi-fragment
    // plans the fragment_instance_id differs from query_id by an offset.
    if let Some(lp) = params.local_params.as_ref().and_then(|lp| lp.first()) {
        return FinstId {
            hi: lp.fragment_instance_id.hi,
            lo: lp.fragment_instance_id.lo,
        };
    }
    // Fallback to query_id if no local_params available.
    FinstId {
        hi: params.query_id.hi,
        lo: params.query_id.lo,
    }
}

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
        // Default to STRING for any other type
        _ => PScalarType { r#type: 23, len: Some(65533), precision: None, scale: None },
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

/// A file-backed table that needs to be loaded into DuckDB before plan execution.
struct FileTable {
    table_name: String,
    file_path: String,
    format: String,
}

/// Extract file tables from fragment params (FILE_SCAN_NODE → file path + format).
///
/// Walks the plan nodes looking for FILE_SCAN_NODE, then extracts the file path
/// from `local_params[0].per_node_scan_ranges` and the format from `file_scan_params`.
fn extract_file_tables(params: &TPipelineFragmentParams) -> Vec<FileTable> {
    let mut tables = Vec::new();

    let fragment = match &params.fragment {
        Some(f) => f,
        None => return tables,
    };
    let plan = match &fragment.plan {
        Some(p) => p,
        None => return tables,
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

        // File path from local_params[*].per_node_scan_ranges[node_id][0].scan_range
        //   .ext_scan_range.file_scan_range.ranges[0].path
        // Search ALL local_params entries (after merge, leaf's entries are appended).
        let file_path = params
            .local_params
            .as_ref()
            .and_then(|lps| {
                lps.iter().find_map(|inst| {
                    inst.per_node_scan_ranges
                        .get(&node_id)?
                        .first()?
                        .scan_range
                        .ext_scan_range
                        .as_ref()?
                        .file_scan_range
                        .as_ref()?
                        .ranges
                        .as_ref()?
                        .first()?
                        .path
                        .clone()
                })
            });

        if let Some(path) = file_path {
            tables.push(FileTable {
                table_name,
                file_path: path,
                format,
            });
        } else {
            tracing::debug!(
                node_id,
                has_local_params = params.local_params.is_some(),
                local_params_len = params.local_params.as_ref().map(|l| l.len()),
                per_node_keys = ?params.local_params.as_ref().and_then(|lp| lp.first()).map(|inst| inst.per_node_scan_ranges.keys().collect::<Vec<_>>()),
                "FILE_SCAN_NODE found but no file path extracted"
            );
        }
    }

    tables
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

    // Check if any fragment has more exchange nodes than we have leaf fragments
    // to fill them. Each EXCHANGE_NODE(0 children) needs one sender; if there are
    // more exchange nodes than leaves, some exchanges receive from remote BEs and
    // we must NOT merge (the exchange-root must wait for remote senders).
    let has_remote_exchanges = |params: &TPipelineFragmentParams| -> bool {
        let plan = match params.fragment.as_ref().and_then(|f| f.plan.as_ref()) {
            Some(p) => p,
            None => return false,
        };
        let exchange_count = plan.nodes.iter().filter(|n| {
            n.node_type == TPlanNodeType::EXCHANGE_NODE && n.num_children == 0
        }).count();

        // Also check per-exchange sender counts: if any single exchange expects
        // more senders than available leaves, it has remote senders.
        let any_excess = plan.nodes.iter().any(|n| {
            n.node_type == TPlanNodeType::EXCHANGE_NODE
                && n.num_children == 0
                && get_num_senders(params, n.node_id) as usize > leaf_fragments.len()
        });

        if exchange_count > leaf_fragments.len() || any_excess {
            info!(
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

    // Cascading merge: sort intermediates by fragment_id descending (innermost first),
    // then merge leaf → inner intermediate → outer intermediate, producing a single
    // fully-merged fragment. This handles 3+ fragment plans (e.g. SORT → AGG → SCAN).
    intermediate_fragments.sort_by(|a, b| {
        let id_a = a.fragment_id.unwrap_or(0);
        let id_b = b.fragment_id.unwrap_or(0);
        id_b.cmp(&id_a) // descending: highest fragment_id (innermost) first
    });

    let mut current_nodes: Vec<_> = leaf_fragments
        .first()
        .and_then(|p| p.fragment.as_ref())
        .and_then(|f| f.plan.as_ref())
        .map(|p| p.nodes.clone())
        .unwrap_or_default();
    let mut current_scan_params = leaf_fragments
        .first()
        .and_then(|p| p.file_scan_params.clone());
    let mut current_local_params = leaf_fragments
        .first()
        .and_then(|p| p.local_params.clone());

    for params in &mut intermediate_fragments {
        // Replace EXCHANGE_NODE(0 children) with current merged plan nodes.
        if let Some(plan) = params
            .fragment
            .as_mut()
            .and_then(|f| f.plan.as_mut())
        {
            let mut merged_nodes = Vec::new();
            for node in &plan.nodes {
                if node.node_type == TPlanNodeType::EXCHANGE_NODE && node.num_children == 0 {
                    if !current_nodes.is_empty() {
                        merged_nodes.extend_from_slice(&current_nodes);
                    } else {
                        merged_nodes.push(node.clone());
                    }
                } else {
                    merged_nodes.push(node.clone());
                }
            }
            plan.nodes = merged_nodes;
            // Update current_nodes to this fragment's merged plan for the next cascade level.
            current_nodes = plan.nodes.clone();
        }

        // Merge file_scan_params from inner level.
        if let Some(inner_scan) = &current_scan_params {
            let merged_scan = params
                .file_scan_params
                .get_or_insert_with(Default::default);
            for (k, v) in inner_scan {
                merged_scan.entry(*k).or_insert_with(|| v.clone());
            }
        }
        current_scan_params = params.file_scan_params.clone();

        // Merge local_params from inner level.
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
    }

    // After cascading, keep only the outermost intermediate (lowest fragment_id = last).
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
        let intermediate_nodes: Vec<_> = intermediate_fragments
            .first()
            .and_then(|p| p.fragment.as_ref())
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

            // Copy file_scan_params and additional local_params from intermediate.
            if let Some(inter) = intermediate_fragments.first() {
                if let Some(inter_scan) = &inter.file_scan_params {
                    let merged_scan = params
                        .file_scan_params
                        .get_or_insert_with(Default::default);
                    for (k, v) in inter_scan {
                        merged_scan.entry(*k).or_insert_with(|| v.clone());
                    }
                }
                // Append intermediate's local_params (scan ranges) but keep
                // exchange_root's local_params[0] first (has the result instance_id).
                if let Some(inter_local) = &inter.local_params {
                    let merged_local = params.local_params.get_or_insert_with(Vec::new);
                    for lp in inter_local {
                        merged_local.push(lp.clone());
                    }
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
    Sql(String),
}

/// Execute a plan via Sirius GPU, falling back to DuckDB CPU.
fn execute_plan(engine: &SiriusEngine, plan: ExecPlan) -> Result<Vec<u8>, String> {
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
            match engine.execute_substrait(plan_bytes, sort_limit_sql.as_deref()) {
                Ok(ipc) => {
                    tracing::info!("executed via gpu_processing_substrait");
                    Ok(ipc)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "gpu_processing_substrait failed, falling back to CPU from_substrait");
                    from_substrait_with_sort(engine, &bytes, sort_limit_sql.as_deref())
                }
            }
        }
        ExecPlan::SubstraitCpuOnly { bytes, sort_limit_sql } => {
            tracing::info!("executing via CPU from_substrait (no data tables, GPU skipped)");
            from_substrait_with_sort(engine, &bytes, sort_limit_sql.as_deref())
        }
        ExecPlan::Sql(sql) => match engine.execute_gpu(&sql) {
            Ok(ipc) => {
                tracing::info!("executed via gpu_execution");
                Ok(ipc)
            }
            Err(e) => {
                tracing::warn!(error = %e, "gpu_execution failed, falling back to direct SQL");
                engine.execute_sql(&sql).map_err(|e| e.to_string())
            }
        },
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
    use substrait::proto::{plan_rel, rel, Plan};

    let mut plan = Plan::decode(bytes).ok()?;
    let relation = plan.relations.first_mut()?;
    let plan_rel::RelType::Root(root) = relation.rel_type.as_mut()? else {
        return None;
    };
    let rel = root.input.as_mut()?;

    // Unwrap FetchRel → SortRel → inner, or SortRel → inner.
    let inner = match rel.rel_type.as_ref()? {
        rel::RelType::Fetch(fetch) => {
            let input = fetch.input.as_ref()?;
            match input.rel_type.as_ref()? {
                rel::RelType::Sort(sort) => sort.input.as_ref()?.as_ref().clone(),
                _ => input.as_ref().clone(),
            }
        }
        rel::RelType::Sort(sort) => sort.input.as_ref()?.as_ref().clone(),
        _ => return None,
    };

    root.input = Some(inner);
    Some(plan.encode_to_vec())
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

    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None)
        .map_err(|e| format!("parse IPC for projection: {e}"))?;
    let schema = reader.schema();
    let schema_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();

    // Determine column indices to project.
    let indices = if let Some(explicit) = explicit_indices {
        tracing::info!(
            indices = ?explicit,
            duckdb_schema = ?schema_names,
            "using explicit column indices"
        );
        explicit.to_vec()
    } else if schema.fields().len() == output_names.len() {
        // Column count matches and no explicit reorder — no projection needed.
        tracing::info!(
            cols = schema.fields().len(),
            duckdb_schema = ?schema_names,
            output_names = ?output_names,
            "IPC column count matches, no projection needed"
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

/// Extract exchange send destinations from fragment params.
///
/// Returns `(dest_node_id, destinations)` if this fragment's output should be
/// sent to remote BEs via transmit_block. Returns `None` for result sinks
/// (where FE fetches data directly) or when destinations are empty.
fn extract_exchange_destinations(
    params: &TPipelineFragmentParams,
) -> Option<(i32, Vec<ExchangeDest>)> {
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

    Some((dest_node_id, dests))
}

pub struct PBackendServiceHandler {
    state: Arc<BeState>,
    result_store: ResultStore,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
    exchange_buffer: ExchangeBuffer,
    #[cfg(feature = "nixl")]
    nixl_agent: Option<Arc<super::nixl_exchange::NixlExchange>>,
}

impl PBackendServiceHandler {
    pub fn new(
        state: Arc<BeState>,
        result_store: ResultStore,
        engine: Option<Arc<Mutex<SiriusEngine>>>,
        exchange_buffer: ExchangeBuffer,
    ) -> Self {
        Self {
            state,
            result_store,
            engine,
            exchange_buffer,
            #[cfg(feature = "nixl")]
            nixl_agent: None,
        }
    }

    #[cfg(feature = "nixl")]
    pub fn with_nixl_agent(mut self, agent: Option<Arc<super::nixl_exchange::NixlExchange>>) -> Self {
        self.nixl_agent = agent;
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

        // Before merge: find the result-sink fragment's finst_id (fragment_id=0).
        // FE always calls fetch_data with this finst_id regardless of merge.
        let result_finst_id = find_result_finst_id(&all_params);

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
            // Determine the finst_id to store the result under.
            // - Single-BE merged plan (1 fragment): FE calls fetch_data with query_id.
            // - Multi-BE exchange (>1 fragment): FE calls fetch_data with the
            //   exchange-root fragment's local_params[0].fragment_instance_id.
            let finst_id = if merged_params.len() == 1 {
                // Single-BE merged: FE uses query_id for fetch_data.
                FinstId {
                    hi: params.query_id.hi,
                    lo: params.query_id.lo,
                }
            } else {
                // Multi-BE exchange: FE uses fragment_instance_id.
                // Prefer the result fragment's finst_id when available.
                result_finst_id.unwrap_or_else(|| extract_finst_id(params))
            };

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
                #[cfg(feature = "nixl")]
                let nixl_agent = self.nixl_agent.clone();

                // Spawn async task: wait for exchange data, decode, load, execute.
                tokio::spawn(async move {
                    // Wait for all exchange nodes to receive all their data.
                    for notify in &notifies {
                        notify.notified().await;
                    }
                    info!(%finst_id, "all exchange data received, proceeding with execution");

                    // Decode PBlocks and register exchange tables.
                    let mut table_schemas = std::collections::HashMap::<String, Vec<String>>::new();
                    for &node_id in &exchange_node_ids {
                        let key = ExchangeKey { query_id, node_id };
                        let blocks = buffer.take(&key);
                        let table_name = format!(
                            "__EXCHANGE_TABLE_{}",
                            node_id
                        );

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

                        // Convert to SQL values for DuckDB insertion (filtered columns only).
                        let column_data_csv: Vec<Vec<String>> = keep_indices.iter()
                            .map(|&i| pblock_decoder::column_to_sql_values(&decoded.columns[i], decoded.num_rows))
                            .collect();

                        // Register in DuckDB engine.
                        let engine_guard = engine.lock().unwrap();
                        if let Err(e) = engine_guard.register_exchange_table(
                            &table_name,
                            &column_names,
                            &column_types_sql,
                            decoded.num_rows,
                            &column_data_csv,
                        ) {
                            warn!(error = %e, table = %table_name, "failed to register exchange table");
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

                    // Also register any file tables.
                    let file_tables = extract_file_tables(&params);
                    if !file_tables.is_empty() {
                        let engine_guard = engine.lock().unwrap();
                        for ft in &file_tables {
                            let empty: Vec<String> = vec![];
                            if let Err(e) = engine_guard.register_file_table(
                                &ft.table_name, &ft.file_path, &ft.format, &empty,
                            ) {
                                warn!(error = %e, table = %ft.table_name, "failed to register file table");
                                return;
                            }
                            if let Ok(columns) = engine_guard.get_table_columns(&ft.table_name) {
                                table_schemas.insert(ft.table_name.clone(), columns);
                            }
                        }
                        drop(engine_guard);
                    }

                    // Translate and execute.
                    // Exchange-receiving fragments use SQL path: SetRel(UNION_ALL)
                    // is not reliably supported by DuckDB's Substrait extension or
                    // the GPU engine, and these fragments just combine pre-computed
                    // exchange data — no GPU acceleration benefit.
                    let use_sql = !exchange_node_ids.is_empty();
                    let has_substrait = !use_sql && engine.lock().unwrap().has_substrait();
                    let (exec_plan, output_names) = if has_substrait {
                        match plan_translator::translate_fragment(&params, &table_schemas) {
                            Ok(plan) => (ExecPlan::Substrait { bytes: plan.substrait_bytes, sort_limit_sql: plan.sort_limit_sql }, Some(plan.output_names)),
                            Err(e) => {
                                warn!(error = %e, "Substrait translation failed for exchange fragment");
                                match plan_translator::translate_fragment_to_sql(&params) {
                                    Ok(sql) => (ExecPlan::Sql(sql), None),
                                    Err(e2) => {
                                        warn!(error = %e2, "SQL translation also failed");
                                        return;
                                    }
                                }
                            }
                        }
                    } else {
                        match plan_translator::translate_fragment_to_sql(&params) {
                            Ok(sql) => {
                                if use_sql {
                                    info!(sql = %sql, "exchange fragment using SQL path");
                                }
                                (ExecPlan::Sql(sql), None)
                            }
                            Err(e) => {
                                warn!(error = %e, "SQL translation failed for exchange fragment");
                                return;
                            }
                        }
                    };

                    // Check if this exchange fragment also needs to forward results.
                    let exchange_dests = extract_exchange_destinations(&params);

                    let exec_result = tokio::task::spawn_blocking(move || -> Result<crate::nixl_integration::ExecutionLocation, String> {
                        let engine = engine.lock().unwrap();
                        let mut ipc_bytes = execute_plan(&engine, exec_plan)?;
                        if let Some(names) = output_names {
                            ipc_bytes = project_ipc_columns(&ipc_bytes, &names, None)?;
                        }
                        Ok(crate::nixl_integration::detect_execution_location(ipc_bytes, &engine))
                    })
                    .await;

                    match exec_result {
                        Ok(Ok(location)) => {
                            if let Some((dest_node_id, dests)) = exchange_dests {
                                let query_id = (params.query_id.hi, params.query_id.lo);
                                if let Err(e) = crate::nixl_integration::send_exchange_with_nixl(
                                    #[cfg(feature = "nixl")]
                                    nixl_agent.as_ref(),
                                    #[cfg(not(feature = "nixl"))]
                                    None,
                                    location, &dests, query_id, dest_node_id, 0,
                                ).await {
                                    warn!(error = %e, %finst_id, "exchange forward failed");
                                }
                                info!(%finst_id, "exchange fragment forward complete");
                            } else {
                                let ipc_bytes = location.into_ipc_bytes();
                                if let Err(e) = store.store_ipc_result(finst_id, &ipc_bytes) {
                                    warn!(error = %e, %finst_id, "failed to store exchange result");
                                } else {
                                    info!(%finst_id, "exchange fragment execution complete");
                                }
                            }
                        }
                        Ok(Err(e)) => warn!(error = %e, %finst_id, "exchange fragment execution failed"),
                        Err(e) => warn!(error = %e, "exchange fragment spawn_blocking panicked"),
                    }
                });

                // Return immediately — execution happens asynchronously.
                continue;
            }

            // Standard synchronous execution path (no unresolved exchanges).

            // Step 1: Register file-backed tables in DuckDB with SELECT * (all columns).
            // Step 2: Get actual column names from DuckDB.
            // Step 3: Pass those as table_schemas to the Substrait translator.
            // This ensures the ReadRel schema and SLOT_REF field references match
            // the DuckDB table, even with TVF late materialization.
            let file_tables = extract_file_tables(params);
            let mut table_schemas = std::collections::HashMap::<String, Vec<String>>::new();

            if !file_tables.is_empty() {
                if let Some(engine) = &self.engine {
                    let engine_guard = engine.lock().unwrap();
                    for ft in &file_tables {
                        // Register with SELECT * to get all file columns.
                        let empty: Vec<String> = vec![];
                        if let Err(e) = engine_guard.register_file_table(&ft.table_name, &ft.file_path, &ft.format, &empty) {
                            warn!(error = %e, table = %ft.table_name, "failed to register file table");
                            return Ok(Response::new(PExecPlanFragmentResult {
                                status: err_status(&format!("register file table '{}': {e}", ft.table_name)),
                                ..Default::default()
                            }));
                        }
                        // Get actual column names from DuckDB.
                        match engine_guard.get_table_columns(&ft.table_name) {
                            Ok(columns) => {
                                info!(
                                    table = %ft.table_name,
                                    path = %ft.file_path,
                                    columns = ?columns,
                                    "registered file table"
                                );
                                table_schemas.insert(ft.table_name.clone(), columns);
                            }
                            Err(e) => {
                                warn!(error = %e, table = %ft.table_name, "failed to get table columns");
                            }
                        }
                    }
                }
            }

            // Try Substrait translation first, fall back to SQL.
            // Constant-only queries (no data tables) use CPU-only Substrait to avoid
            // GPU engine bugs with VirtualTable plans.
            // If substrait extension isn't loaded, skip directly to SQL.
            let has_data_tables = !file_tables.is_empty() || !table_schemas.is_empty();
            let has_substrait = self.engine.as_ref()
                .map(|e| e.lock().unwrap().has_substrait()).unwrap_or(false);
            let (exec_plan, output_names, output_indices) = if has_substrait {
                match plan_translator::translate_fragment(params, &table_schemas) {
                    Ok(plan) => {
                        info!(bytes = plan.substrait_bytes.len(), has_data_tables, "translated to Substrait");
                        let exec = if has_data_tables {
                            ExecPlan::Substrait { bytes: plan.substrait_bytes, sort_limit_sql: plan.sort_limit_sql }
                        } else {
                            ExecPlan::SubstraitCpuOnly { bytes: plan.substrait_bytes, sort_limit_sql: plan.sort_limit_sql }
                        };
                        (exec, Some(plan.output_names), plan.output_column_indices)
                    }
                    Err(e) => {
                        warn!(error = %e, "Substrait translation failed, trying SQL fallback");
                        match plan_translator::translate_fragment_to_sql(params) {
                            Ok(sql) => {
                                info!(sql = %sql, "translated to SQL (fallback)");
                                (ExecPlan::Sql(sql), None, None)
                            }
                            Err(e2) => {
                                warn!(error = %e2, "SQL translation also failed");
                                return Ok(Response::new(PExecPlanFragmentResult {
                                    status: err_status(&format!("plan translation: {e}")),
                                    ..Default::default()
                                }));
                            }
                        }
                    }
                }
            } else {
                match plan_translator::translate_fragment_to_sql(params) {
                    Ok(sql) => {
                        info!(sql = %sql, "translated to SQL (no substrait extension)");
                        (ExecPlan::Sql(sql), None, None)
                    }
                    Err(e) => {
                        warn!(error = %e, "SQL translation failed");
                        return Ok(Response::new(PExecPlanFragmentResult {
                            status: err_status(&format!("plan translation: {e}")),
                            ..Default::default()
                        }));
                    }
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

            // Check if this fragment's output should be sent to remote BEs.
            let exchange_dests = extract_exchange_destinations(params);

            // Sirius/DuckDB execution is blocking — run off the async runtime.
            let exec_result = tokio::task::spawn_blocking(move || -> Result<crate::nixl_integration::ExecutionLocation, String> {
                let engine = engine.lock().unwrap();
                let mut ipc_bytes = execute_plan(&engine, exec_plan)?;
                if let Some(names) = output_names {
                    ipc_bytes = project_ipc_columns(&ipc_bytes, &names, output_indices.as_deref())?;
                }
                Ok(crate::nixl_integration::detect_execution_location(ipc_bytes, &engine))
            })
            .await;

            match exec_result {
                Ok(Ok(location)) => {
                    if let Some((dest_node_id, dests)) = exchange_dests {
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
                            dest_node_id,
                            num_dests = dests.len(),
                            "sending execution result to exchange destinations"
                        );
                        if let Err(e) = crate::nixl_integration::send_exchange_with_nixl(
                            #[cfg(feature = "nixl")]
                            self.nixl_agent.as_ref(),
                            #[cfg(not(feature = "nixl"))]
                            None,
                            location,
                            &dests,
                            query_id,
                            dest_node_id,
                            sender_id,
                        )
                        .await
                        {
                            warn!(error = %e, %finst_id, "exchange send failed");
                            return Ok(Response::new(PExecPlanFragmentResult {
                                status: err_status(&format!("exchange send: {e}")),
                                ..Default::default()
                            }));
                        }
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
                        info!(%finst_id, "execution complete, result stored");
                    }
                }
                Ok(Err(e)) => {
                    warn!(error = %e, %finst_id, "execution failed");
                    return Ok(Response::new(PExecPlanFragmentResult {
                        status: err_status(&format!("execution: {e}")),
                        ..Default::default()
                    }));
                }
                Err(e) => {
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

    async fn exec_plan_fragment_prepare(
        &self,
        request: Request<PExecPlanFragmentRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        self.exec_plan_fragment(request).await
    }

    async fn exec_plan_fragment_start(
        &self,
        request: Request<PExecPlanFragmentStartRequest>,
    ) -> Result<Response<PExecPlanFragmentResult>, Status> {
        let req = request.into_inner();
        info!(query_id = ?req.query_id, "exec_plan_fragment_start");
        Ok(Response::new(PExecPlanFragmentResult {
            status: ok_status(),
            ..Default::default()
        }))
    }

    async fn cancel_plan_fragment(
        &self,
        request: Request<PCancelPlanFragmentRequest>,
    ) -> Result<Response<PCancelPlanFragmentResult>, Status> {
        let req = request.into_inner();
        info!(query_id = ?req.query_id, fragment_id = ?req.fragment_id, "cancel_plan_fragment");
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
            be_arrow_flight_ip: Some(b"127.0.0.1".to_vec()),
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

        let file_path = {
            use thrift::protocol::{TCompactInputProtocol, TBinaryInputProtocol, TSerializable};
            use thrift::transport::TBufferedReadTransport;

            info!(
                len = scan_range_bytes.len(),
                first_bytes = ?&scan_range_bytes[..scan_range_bytes.len().min(64)],
                "fetch_table_schema: raw scan_range bytes"
            );

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
                    // Try to get path from ranges first, then from params.properties
                    let path_from_ranges = fsr.ranges
                        .as_ref()
                        .and_then(|r| r.first())
                        .and_then(|rd| {
                            info!(path = ?rd.path, "fetch_table_schema: range desc");
                            rd.path.clone()
                        });

                    // Also check params.properties for file_path
                    let path_from_params = fsr.params
                        .as_ref()
                        .and_then(|p| {
                            info!(props = ?p.properties, "fetch_table_schema: scan params");
                            p.properties.as_ref()
                        })
                        .and_then(|props| props.get("file_path").cloned());

                    path_from_ranges.or(path_from_params)
                }
                Err(e) => {
                    warn!(error = %e, "fetch_table_schema: compact deser failed, trying binary");
                    // Try binary protocol.
                    let transport = TBufferedReadTransport::new(Box::new(Cursor::new(scan_range_bytes)));
                    let mut protocol = TBinaryInputProtocol::new(transport, true);
                    match doris_thrift::plan_nodes::TFileScanRange::read_from_in_protocol(&mut protocol) {
                        Ok(fsr) => {
                            info!(
                                has_ranges = fsr.ranges.is_some(),
                                num_ranges = fsr.ranges.as_ref().map(|r| r.len()).unwrap_or(0),
                                "fetch_table_schema: deserialized TFileScanRange (binary)"
                            );
                            let path_from_ranges = fsr.ranges
                                .as_ref()
                                .and_then(|r| r.first())
                                .and_then(|rd| rd.path.clone());
                            let path_from_params = fsr.params
                                .as_ref()
                                .and_then(|p| p.properties.as_ref())
                                .and_then(|props| props.get("file_path").cloned());
                            path_from_ranges.or(path_from_params)
                        }
                        Err(e2) => {
                            warn!(error = %e2, "fetch_table_schema: binary deser also failed");
                            None
                        }
                    }
                }
            }
        };

        let file_path = match file_path {
            Some(p) => p,
            None => {
                warn!("fetch_table_schema: no file path in TFileScanRange");
                return Ok(Response::new(PFetchTableSchemaResult {
                    status: Some(err_status("no file path in scan range")),
                    ..Default::default()
                }));
            }
        };

        info!(path = %file_path, "fetch_table_schema: reading schema");

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
                .get_file_schema_ipc(&file_path, "parquet")
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

        // For local files, just stat the path and return it.
        let mut files = vec![];
        match std::fs::metadata(&pattern) {
            Ok(meta) => {
                files.push(p_glob_response::PFileInfo {
                    file: Some(pattern),
                    size: Some(meta.len() as i64),
                });
            }
            Err(e) => {
                warn!(pattern = %pattern, error = %e, "glob: file not found");
            }
        }

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
    #[cfg(feature = "nixl")]
    nixl_agent: Option<Arc<super::nixl_exchange::NixlExchange>>,
    #[cfg(not(feature = "nixl"))]
    _nixl_agent: Option<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    use doris_proto::doris::p_backend_service_server::PBackendServiceServer;
    use tokio_stream::wrappers::ReceiverStream;
    use tokio_stream::StreamExt;

    let listener = tokio::net::TcpListener::bind(listen_addr).await?;
    info!(addr = listen_addr, "starting PBackendService multi-protocol server (gRPC + bRPC)");

    // Clone before moving into handler (nixl service also needs these).
    #[cfg(feature = "nixl")]
    let nixl_agent_for_service = nixl_agent.clone();
    #[cfg(feature = "nixl")]
    let engine_for_service = engine.clone();

    #[allow(unused_mut)]
    let mut handler = PBackendServiceHandler::new(
        state,
        result_store,
        engine,
        exchange_buffer.clone(),
    );

    #[cfg(feature = "nixl")]
    {
        handler = handler.with_nixl_agent(nixl_agent);
    }

    let svc = PBackendServiceServer::new(handler);

    // Channel to feed gRPC connections to tonic's serve_with_incoming.
    let (grpc_tx, grpc_rx) =
        tokio::sync::mpsc::channel::<tokio::net::TcpStream>(64);

    // Build gRPC server with PBackendService + optional NixlMetadataService.
    #[allow(unused_mut)]
    let mut server_builder = tonic::transport::Server::builder()
        .add_service(svc);

    #[cfg(feature = "nixl")]
    {
        use doris_proto::nixl::NixlMetadataServiceServer;
        let nixl_handler = super::nixl_service::NixlMetadataServiceHandler::new(
            nixl_agent_for_service,
            engine_for_service,
            exchange_buffer.clone(),
        );
        server_builder = server_builder.add_service(NixlMetadataServiceServer::new(nixl_handler));
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

        let (dest_node_id, dests) = extract_exchange_destinations(&params).unwrap();
        assert_eq!(dest_node_id, 7);
        assert_eq!(dests.len(), 1);
        assert_eq!(dests[0].brpc_addr, "192.168.1.1:8060");
        assert_eq!(dests[0].finst_id, (10, 20));
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

        let (_, dests) = extract_exchange_destinations(&params).unwrap();
        assert_eq!(dests[0].brpc_addr, "10.0.0.1:9060");
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

        let (node_id, dests) = extract_exchange_destinations(&params).unwrap();
        assert_eq!(node_id, 5);
        assert_eq!(dests.len(), 2);
        assert_eq!(dests[0].brpc_addr, "be1:8060");
        assert_eq!(dests[1].brpc_addr, "be2:8060");
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
        // Fragment 0: SORT_NODE → EXCHANGE_NODE(0) — outermost intermediate
        // Fragment 1: AGG_NODE → EXCHANGE_NODE(0) — innermost intermediate
        // Fragment 2: FILE_SCAN_NODE — leaf
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

        let mut leaf = make_params(vec![
            make_node(4, TPlanNodeType::FILE_SCAN_NODE, 0),
        ]);
        leaf.fragment_id = Some(2);

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
}
