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
use doris_thrift::palo_internal_service::{TPipelineFragmentParams, TPipelineFragmentParamsList};
use doris_thrift::plan_nodes::{TFileFormatType, TPlanNodeType};
use result_formatter::result_store::{FinstId, ResultStore};
use sirius_ffi::SiriusEngine;

use super::heartbeat_service::BeState;

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
/// With `enable_parallel_result_sink` (default: true), the FE uses the `query_id`
/// to fetch results, not the fragment_instance_id. So we always use query_id.
fn extract_finst_id(params: &TPipelineFragmentParams) -> FinstId {
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
        DataType::Date32 | DataType::Date64 => PScalarType { r#type: 28, len: None, precision: None, scale: None },
        DataType::Timestamp(_, _) => PScalarType { r#type: 29, len: None, precision: None, scale: None },
        DataType::Utf8 | DataType::LargeUtf8 => PScalarType { r#type: 16, len: Some(65533), precision: None, scale: None },
        DataType::Decimal128(p, s) => {
            let p = *p as i32;
            let s = *s as i32;
            if p <= 9 {
                PScalarType { r#type: 47, len: None, precision: Some(p), scale: Some(s) }
            } else if p <= 18 {
                PScalarType { r#type: 48, len: None, precision: Some(p), scale: Some(s) }
            } else {
                PScalarType { r#type: 49, len: None, precision: Some(p), scale: Some(s) }
            }
        }
        // Default to VARCHAR for any other type
        _ => PScalarType { r#type: 16, len: Some(65533), precision: None, scale: None },
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

    for params in all_params {
        let plan = match params.fragment.as_ref().and_then(|f| f.plan.as_ref()) {
            Some(p) => p,
            None => continue,
        };
        let root = match plan.nodes.first() {
            Some(r) => r,
            None => continue,
        };

        // Skip result-collector fragments (EXCHANGE at root, 0 children).
        if root.node_type == TPlanNodeType::EXCHANGE_NODE && root.num_children == 0 {
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

    // If no intermediate fragments, just return the leaf fragments.
    if intermediate_fragments.is_empty() {
        return leaf_fragments.into_iter().cloned().collect();
    }

    // Merge: for each intermediate fragment, replace EXCHANGE_NODE(0 children) with leaf plan.
    // For simplicity, use the first leaf fragment's plan for all exchanges.
    let leaf_nodes: Vec<_> = leaf_fragments
        .first()
        .and_then(|p| p.fragment.as_ref())
        .and_then(|f| f.plan.as_ref())
        .map(|p| p.nodes.clone())
        .unwrap_or_default();

    for params in &mut intermediate_fragments {
        if let Some(plan) = params
            .fragment
            .as_mut()
            .and_then(|f| f.plan.as_mut())
        {
            let mut merged_nodes = Vec::new();
            for node in &plan.nodes {
                if node.node_type == TPlanNodeType::EXCHANGE_NODE && node.num_children == 0 {
                    // Replace exchange with the leaf plan nodes.
                    merged_nodes.extend_from_slice(&leaf_nodes);
                } else {
                    merged_nodes.push(node.clone());
                }
            }
            plan.nodes = merged_nodes;
        }

        // Merge file_scan_params and local_params from leaf fragments.
        if let Some(leaf) = leaf_fragments.first() {
            if let Some(leaf_scan_params) = &leaf.file_scan_params {
                let merged_scan_params = params
                    .file_scan_params
                    .get_or_insert_with(Default::default);
                for (k, v) in leaf_scan_params {
                    merged_scan_params.entry(*k).or_insert_with(|| v.clone());
                }
            }
            // Merge local_params (contains per_node_scan_ranges needed for file path extraction).
            if params.local_params.is_none() {
                params.local_params = leaf.local_params.clone();
            } else if let Some(leaf_local) = &leaf.local_params {
                let merged_local = params.local_params.get_or_insert_with(Vec::new);
                for lp in leaf_local {
                    merged_local.push(lp.clone());
                }
            }
        }
    }

    intermediate_fragments
}

pub struct PBackendServiceHandler {
    state: Arc<BeState>,
    result_store: ResultStore,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
}

impl PBackendServiceHandler {
    pub fn new(
        state: Arc<BeState>,
        result_store: ResultStore,
        engine: Option<Arc<Mutex<SiriusEngine>>>,
    ) -> Self {
        Self {
            state,
            result_store,
            engine,
        }
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
            let finst_id = extract_finst_id(params);

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
            enum ExecPlan {
                Substrait(Vec<u8>),
                Sql(String),
            }

            let exec_plan = match plan_translator::translate_fragment(params, &table_schemas) {
                Ok(substrait_bytes) => {
                    info!(bytes = substrait_bytes.len(), "translated to Substrait");
                    ExecPlan::Substrait(substrait_bytes)
                }
                Err(e) => {
                    warn!(error = %e, "Substrait translation failed, trying SQL fallback");
                    match plan_translator::translate_fragment_to_sql(params) {
                        Ok(sql) => {
                            info!(sql = %sql, "translated to SQL (fallback)");
                            ExecPlan::Sql(sql)
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

            // Sirius/DuckDB execution is blocking — run off the async runtime.
            let exec_result = tokio::task::spawn_blocking(move || -> Result<(), String> {
                let engine = engine.lock().unwrap();
                let ipc_bytes = match exec_plan {
                    ExecPlan::Substrait(bytes) => {
                        match engine.execute_substrait(&bytes) {
                            Ok(ipc) => {
                                tracing::info!("executed via gpu_processing_substrait");
                                ipc
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "gpu_processing_substrait failed, falling back to CPU from_substrait");
                                engine.from_substrait(&bytes)
                                    .map_err(|e| e.to_string())?
                            }
                        }
                    }
                    ExecPlan::Sql(sql) => {
                        match engine.execute_gpu(&sql) {
                            Ok(ipc) => {
                                tracing::info!("executed via gpu_execution");
                                ipc
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "gpu_execution failed, falling back to direct SQL");
                                engine.execute_sql(&sql).map_err(|e| e.to_string())?
                            }
                        }
                    }
                };
                store.store_ipc_result(finst_id, &ipc_bytes)?;
                Ok(())
            })
            .await;

            match exec_result {
                Ok(Ok(())) => {
                    info!(%finst_id, "execution complete, result stored");
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
        info!(query_id = ?req.query_id, sender_id = req.sender_id, eos = req.eos, "transmit_block");
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

        let entry = match self.result_store.get(&finst_id) {
            Some(e) => e,
            None => {
                warn!(%finst_id, "fetch_data: result not found");
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
    async fn rerun_fragment(&self, _: Request<PRerunFragmentParams>) -> Result<Response<PRerunFragmentResult>, Status> { Err(unimpl()) }
    async fn reset_global_rf(&self, _: Request<PResetGlobalRfParams>) -> Result<Response<PResetGlobalRfResult>, Status> { Err(unimpl()) }
    async fn transmit_rec_cte_block(&self, _: Request<PTransmitRecCteBlockParams>) -> Result<Response<PTransmitRecCteBlockResult>, Status> { Err(unimpl()) }
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

/// Start the PBackendService gRPC server.
pub async fn start_grpc_server(
    listen_addr: &str,
    state: Arc<BeState>,
    result_store: ResultStore,
    engine: Option<Arc<Mutex<SiriusEngine>>>,
) -> Result<(), Box<dyn std::error::Error>> {
    use doris_proto::doris::p_backend_service_server::PBackendServiceServer;

    let addr = listen_addr.parse()?;
    let handler = PBackendServiceHandler::new(state, result_store, engine);

    info!(addr = listen_addr, "starting PBackendService gRPC server");

    tonic::transport::Server::builder()
        .add_service(PBackendServiceServer::new(handler))
        .serve(addr)
        .await?;

    Ok(())
}
