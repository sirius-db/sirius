//! Engine-derived conformance gate for the translator's partial-state wire-type model
//! (`starrocks-plan-translator`'s `partial_state::wire_columns`).
//!
//! The model is a Rust mirror of what the engine *binds* for each two-phase aggregate's partial
//! state. Its rows were pinned by hand-written tests on both sides — model unit tests in the
//! translator, `[streaming_fragment]` oracle tests in the engine — with nothing mechanical
//! keeping the two in agreement. This suite is that mechanical link: for every model row (every
//! supported two-phase aggregate × input type), it translates the two fragments of the
//! corresponding two-phase query with the real translator, builds each through the real engine
//! build path, and asserts on the engine's own answer (`Fragment::output_types`, the sink
//! column types the hop guard compares):
//!
//! - the PARTIAL fragment's engine-produced sink types must equal the merge fragment's declared
//!   stream schema — the receiver-side rendering of the wire-type model, and exactly the
//!   comparison `relay_from`/`push_packed` make on a live hop;
//! - the MERGE fragment's engine-produced sink types must equal the schema a further downstream
//!   fragment declares from the FE's output slots — the finalizing projection's contract (a
//!   merged integer count/sum binds as HUGEINT inside the fragment and must leave as the
//!   declared BIGINT).
//!
//! A row that fails to translate or build FAILS this suite loudly rather than being skipped:
//! it means the model predicts a wire type for a plan shape that cannot exist (allowlist too
//! broad) or the engine's binding drifted from the model. Either way the answer is a CI failure
//! here, not a refused hop in production.

use starrocks_plan_translator::{ExchangeInput, PlanTranslator, TranslatedPlan};
use starrocks_thrift::descriptors::{TDescriptorTable, TSlotDescriptor, TTupleDescriptor};
use starrocks_thrift::exprs::{TAggregateExpr, TExpr, TExprNode, TExprNodeType, TSlotRef};
use starrocks_thrift::internal_service::{InternalServiceVersion, TExecPlanFragmentParams};
use starrocks_thrift::partitions::{TDataPartition, TPartitionType};
use starrocks_thrift::plan_nodes::{
    TAggregationNode, TExchangeNode, TPlan, TPlanNode, TPlanNodeType,
};
use starrocks_thrift::planner::TPlanFragment;
use starrocks_thrift::types::{
    TFunction, TFunctionBinaryType, TFunctionName, TPrimitiveType, TScalarType, TTypeDesc,
    TTypeNode, TTypeNodeType,
};

/// The exchange node id shared by every constructed fragment; each fragment owns its own
/// engine-side stream, so reuse across fragments is free.
const EXCHANGE_NODE_ID: i32 = 7;

/// One row of the conformance domain: an aggregate function with the FE types it is serialized
/// with. Together the rows instantiate every arm of the wire-type model.
struct Row {
    label: &'static str,
    function: &'static str,
    /// FE type of the aggregate's argument column; `None` is `count(*)`.
    arg: Option<TTypeDesc>,
    /// FE-declared return type — the merge node's output slot type and the serialized
    /// `TFunction.ret_type` the model dispatches on.
    ret: TTypeDesc,
    /// FE-declared intermediate slot type — what the FE *claims* crosses the wire (the model
    /// exists because this lies: DECIMAL128 for decimal sums, VARBINARY for avg).
    intermediate: TTypeDesc,
}

/// Every model row. Arm coverage:
/// - `sum` decimal arm (state FP64): decimal64 and decimal128 inputs;
/// - `sum` integer arm (state I64): every integer width the FE widens to a BIGINT sum;
/// - `sum` floating arm (state FP64): FLOAT and DOUBLE;
/// - `count` arm (state I64): star, a numeric column, a string column;
/// - `min`/`max` identity arm: every scalar type family the translator maps;
/// - `avg` two-column arm (FP64 sum + I64 count): integer, floating, decimal inputs.
fn rows() -> Vec<Row> {
    let mut rows = vec![
        Row {
            label: "sum(DECIMAL64(15,2))",
            function: "sum",
            arg: Some(decimal(TPrimitiveType::DECIMAL64, 15, 2)),
            ret: decimal(TPrimitiveType::DECIMAL128, 38, 2),
            intermediate: decimal(TPrimitiveType::DECIMAL128, 38, 2),
        },
        Row {
            label: "sum(DECIMAL128(38,4))",
            function: "sum",
            arg: Some(decimal(TPrimitiveType::DECIMAL128, 38, 4)),
            ret: decimal(TPrimitiveType::DECIMAL128, 38, 4),
            intermediate: decimal(TPrimitiveType::DECIMAL128, 38, 4),
        },
        Row {
            label: "count(*)",
            function: "count",
            arg: None,
            ret: scalar(TPrimitiveType::BIGINT),
            intermediate: scalar(TPrimitiveType::BIGINT),
        },
        Row {
            label: "count(BIGINT)",
            function: "count",
            arg: Some(scalar(TPrimitiveType::BIGINT)),
            ret: scalar(TPrimitiveType::BIGINT),
            intermediate: scalar(TPrimitiveType::BIGINT),
        },
        Row {
            label: "count(VARCHAR)",
            function: "count",
            arg: Some(scalar(TPrimitiveType::VARCHAR)),
            ret: scalar(TPrimitiveType::BIGINT),
            intermediate: scalar(TPrimitiveType::BIGINT),
        },
    ];
    for primitive in [
        TPrimitiveType::TINYINT,
        TPrimitiveType::SMALLINT,
        TPrimitiveType::INT,
        TPrimitiveType::BIGINT,
    ] {
        rows.push(Row {
            label: sum_label(primitive),
            function: "sum",
            arg: Some(scalar(primitive)),
            ret: scalar(TPrimitiveType::BIGINT),
            intermediate: scalar(TPrimitiveType::BIGINT),
        });
    }
    for primitive in [TPrimitiveType::FLOAT, TPrimitiveType::DOUBLE] {
        rows.push(Row {
            label: sum_label(primitive),
            function: "sum",
            arg: Some(scalar(primitive)),
            ret: scalar(TPrimitiveType::DOUBLE),
            intermediate: scalar(TPrimitiveType::DOUBLE),
        });
    }
    for function in ["min", "max"] {
        for (label, ty) in identity_types(function) {
            rows.push(Row {
                label,
                function,
                arg: Some(ty.clone()),
                ret: ty.clone(),
                intermediate: ty,
            });
        }
    }
    for (label, ty) in [
        ("avg(BIGINT)", scalar(TPrimitiveType::BIGINT)),
        ("avg(DOUBLE)", scalar(TPrimitiveType::DOUBLE)),
        (
            "avg(DECIMAL64(15,2))",
            decimal(TPrimitiveType::DECIMAL64, 15, 2),
        ),
    ] {
        rows.push(Row {
            label,
            function: "avg",
            arg: Some(ty),
            // The FE returns DOUBLE for a non-decimal avg and a max-precision decimal
            // otherwise; both exercise the same model arm, so one decimal row suffices.
            ret: if label == "avg(DECIMAL64(15,2))" {
                decimal(TPrimitiveType::DECIMAL128, 38, 8)
            } else {
                scalar(TPrimitiveType::DOUBLE)
            },
            intermediate: scalar(TPrimitiveType::VARBINARY),
        });
    }
    rows
}

fn sum_label(primitive: TPrimitiveType) -> &'static str {
    match primitive {
        TPrimitiveType::TINYINT => "sum(TINYINT)",
        TPrimitiveType::SMALLINT => "sum(SMALLINT)",
        TPrimitiveType::INT => "sum(INT)",
        TPrimitiveType::BIGINT => "sum(BIGINT)",
        TPrimitiveType::FLOAT => "sum(FLOAT)",
        TPrimitiveType::DOUBLE => "sum(DOUBLE)",
        other => unreachable!("no sum row for {other:?}"),
    }
}

/// The input types the min/max identity arm is instantiated with — one per scalar family the
/// type mapper handles, including the >18-digit decimal whose mapping lowers to FP64.
fn identity_types(function: &'static str) -> Vec<(&'static str, TTypeDesc)> {
    let name = |suffix: &str| -> &'static str {
        // The labels are compile-time; leaking the two dozen formatted strings once per test
        // process is simpler than a static table of fifty entries.
        Box::leak(format!("{function}({suffix})").into_boxed_str())
    };
    vec![
        (name("TINYINT"), scalar(TPrimitiveType::TINYINT)),
        (name("SMALLINT"), scalar(TPrimitiveType::SMALLINT)),
        (name("INT"), scalar(TPrimitiveType::INT)),
        (name("BIGINT"), scalar(TPrimitiveType::BIGINT)),
        (name("FLOAT"), scalar(TPrimitiveType::FLOAT)),
        (name("DOUBLE"), scalar(TPrimitiveType::DOUBLE)),
        (name("DATE"), scalar(TPrimitiveType::DATE)),
        (name("DATETIME"), scalar(TPrimitiveType::DATETIME)),
        (name("VARCHAR"), scalar(TPrimitiveType::VARCHAR)),
        (
            name("DECIMAL32(9,2)"),
            decimal(TPrimitiveType::DECIMAL32, 9, 2),
        ),
        (
            name("DECIMAL64(15,2)"),
            decimal(TPrimitiveType::DECIMAL64, 15, 2),
        ),
        (
            name("DECIMAL128(38,2)"),
            decimal(TPrimitiveType::DECIMAL128, 38, 2),
        ),
    ]
}

/// Every wire_columns row must hold for BOTH engine aggregation paths — the ungrouped
/// (MERGE_AGGREGATE-wrapped) one and the grouped hash one — so each row is built twice.
#[test]
fn wire_type_model_matches_the_engine() {
    let _guard = crate::GPU_ENGINE_TEST_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let context = sirius::SiriusContext::new().expect("bring up the sirius engine context");

    let mut failures = Vec::new();
    for row in rows() {
        for grouped in [false, true] {
            if let Err(reason) = check_row(&context, &row, grouped) {
                failures.push(format!(
                    "{} ({}): {reason}",
                    row.label,
                    if grouped { "grouped" } else { "ungrouped" },
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "the wire-type model and the engine disagree on {} row(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Translates and engine-builds both fragments of one row's two-phase query, comparing the
/// engine's sink types against the receiving side's declared schema on each hop.
fn check_row(context: &sirius::SiriusContext, row: &Row, grouped: bool) -> Result<(), String> {
    let translator = PlanTranslator::new();

    let partial = translator
        .translate_fragment_with_exchange_inputs(
            &partial_params(row, grouped),
            &[exchange_input(input_names(grouped))],
        )
        .map_err(|err| format!("the partial fragment failed to translate: {err}"))?;
    let merge = translator
        .translate_fragment_with_exchange_inputs(
            &merge_params(row, grouped),
            &[exchange_input(partial.output_names.clone())],
        )
        .map_err(|err| format!("the merge fragment failed to translate: {err}"))?;
    // A downstream consumer of the merge fragment's output: translated only for its declared
    // stream schema, which is what a further hop's receiving relay compares against.
    let consumer = translator
        .translate_fragment_with_exchange_inputs(
            &consumer_params(row, grouped),
            &[exchange_input(merge.output_names.clone())],
        )
        .map_err(|err| format!("the downstream consumer failed to translate: {err}"))?;

    // Hop 1: partial sink vs the merge fragment's declared stream — the model's prediction as
    // the receiver renders it.
    let produced = engine_sink_types(context, &partial)
        .map_err(|err| format!("the partial fragment failed to build: {err}"))?;
    let predicted = declared_stream_types(&merge);
    if produced != predicted {
        return Err(format!(
            "partial sink produces {produced:?} but the model declares the merge stream as \
             {predicted:?}"
        ));
    }

    // Hop 2: merge sink vs a downstream fragment's declared stream (the FE output slots) —
    // the finalizing projection's contract.
    let produced = engine_sink_types(context, &merge)
        .map_err(|err| format!("the merge fragment failed to build: {err}"))?;
    let predicted = declared_stream_types(&consumer);
    if produced != predicted {
        return Err(format!(
            "merge sink produces {produced:?} but a downstream fragment declares {predicted:?}"
        ));
    }
    Ok(())
}

/// Builds a translated fragment through the real engine path — the same declare/build calls
/// `engine::run_fragment_inner` makes — and returns its sink column types. Build only: the
/// types are resolved when the plan binds, before any data exists.
fn engine_sink_types(
    context: &sirius::SiriusContext,
    plan: &TranslatedPlan,
) -> Result<Vec<String>, String> {
    let mut fragment = context.fragment().map_err(|err| err.to_string())?;
    for schema in &plan.stream_inputs {
        let stream_id =
            u64::try_from(schema.node_id).map_err(|_| "negative exchange node id".to_string())?;
        for column in &schema.columns {
            fragment
                .declare_input_column(stream_id, &column.name, &column.ty)
                .map_err(|err| err.to_string())?;
        }
    }
    fragment.declare_output(0).map_err(|err| err.to_string())?;
    fragment
        .build(&plan.to_substrait_bytes())
        .map_err(|err| err.to_string())?;
    fragment.output_types().map_err(|err| err.to_string())
    // Dropping the built-but-unrun fragment closes its query lifecycle, freeing the engine for
    // the next build.
}

/// The DuckDB type names a translated fragment declares for its single input stream.
fn declared_stream_types(plan: &TranslatedPlan) -> Vec<String> {
    plan.stream_inputs[0]
        .columns
        .iter()
        .map(|column| column.ty.clone())
        .collect()
}

/// The sender-side output names fed to the partial fragment's exchange input, i.e. the input
/// tuple's column names.
fn input_names(grouped: bool) -> Vec<String> {
    let mut names = Vec::new();
    if grouped {
        names.push("k".to_string());
    }
    names.push("v".to_string());
    names
}

fn exchange_input(names: Vec<String>) -> ExchangeInput {
    ExchangeInput {
        node_id: EXCHANGE_NODE_ID,
        stream_view: sirius::stream_view_name(EXCHANGE_NODE_ID as u64),
        names,
    }
}

/// The partial fragment: an update-phase aggregation (need_finalize = false) over an exchange
/// carrying the argument column, mirroring a scan-side fragment whose input arrived as a
/// stream. Tuple 0 is the input row, tuple 1 the FE's partial output.
fn partial_params(row: &Row, grouped: bool) -> TExecPlanFragmentParams {
    // count(*) has no argument, but the input tuple still needs a column to ship.
    let arg_type = row
        .arg
        .clone()
        .unwrap_or_else(|| scalar(TPrimitiveType::BIGINT));
    let mut slots = Vec::new();
    let mut grouping = Vec::new();
    if grouped {
        slots.push(slot(1, 0, 0, "k", scalar(TPrimitiveType::VARCHAR)));
        grouping.push(slot_ref(1, 0, scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(2, 0, grouped as i32, "v", arg_type.clone()));
    if grouped {
        slots.push(slot(10, 1, 0, "k", scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(11, 1, grouped as i32, "s", row.intermediate.clone()));

    let measure = aggregate_expr(
        row.function,
        row.ret.clone(),
        row.arg.as_ref().map(|ty| slot_ref(2, 0, ty.clone())),
        false,
    );
    let mut aggregate = aggregation_node(1, 1, grouping, vec![measure]);
    aggregate.agg_node.as_mut().unwrap().need_finalize = false;
    params(
        TPlan::new(vec![aggregate, exchange_node(0)]),
        desc_table(vec![0, 1], slots),
    )
}

/// The merge fragment: a merge-phase aggregation (need_finalize = true, is_merge_agg = true)
/// over an exchange carrying the intermediate row. Tuple 2 is the FE's intermediate row (one
/// slot per measure, whatever the state's real width), tuple 3 the FE's final output.
fn merge_params(row: &Row, grouped: bool) -> TExecPlanFragmentParams {
    let mut slots = Vec::new();
    let mut grouping = Vec::new();
    if grouped {
        slots.push(slot(20, 2, 0, "k", scalar(TPrimitiveType::VARCHAR)));
        grouping.push(slot_ref(20, 2, scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(21, 2, grouped as i32, "s", row.intermediate.clone()));
    if grouped {
        slots.push(slot(30, 3, 0, "k", scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(31, 3, grouped as i32, "r", row.ret.clone()));

    // A merge measure always reads its own state slot, count(*) included.
    let measure = aggregate_expr(
        row.function,
        row.ret.clone(),
        Some(slot_ref(21, 2, row.intermediate.clone())),
        true,
    );
    let aggregate = aggregation_node(8, 3, grouping, vec![measure]);
    params(
        TPlan::new(vec![aggregate, exchange_node(2)]),
        desc_table(vec![2, 3], slots),
    )
}

/// A downstream fragment that just reads the merge fragment's output: a bare bound exchange
/// over the FE output tuple, whose declared stream schema is what hop 2 must satisfy.
fn consumer_params(row: &Row, grouped: bool) -> TExecPlanFragmentParams {
    let mut slots = Vec::new();
    if grouped {
        slots.push(slot(30, 3, 0, "k", scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(31, 3, grouped as i32, "r", row.ret.clone()));
    params(
        TPlan::new(vec![exchange_node(3)]),
        desc_table(vec![3], slots),
    )
}

// --- StarRocks thrift construction, mirroring what the FE serializes -----------------------

fn scalar(primitive: TPrimitiveType) -> TTypeDesc {
    TTypeDesc::new(Some(vec![TTypeNode::new(
        TTypeNodeType::SCALAR,
        Some(TScalarType::new(primitive, None, None, None)),
        None,
        None,
    )]))
}

fn decimal(primitive: TPrimitiveType, precision: i32, scale: i32) -> TTypeDesc {
    TTypeDesc::new(Some(vec![TTypeNode::new(
        TTypeNodeType::SCALAR,
        Some(TScalarType::new(
            primitive,
            None,
            Some(precision),
            Some(scale),
        )),
        None,
        None,
    )]))
}

fn slot(id: i32, tuple_id: i32, column_pos: i32, name: &str, ty: TTypeDesc) -> TSlotDescriptor {
    TSlotDescriptor::new(
        Some(id),
        Some(tuple_id),
        Some(ty),
        Some(column_pos),
        None,
        None,
        None,
        Some(name.to_string()),
        None,
        Some(true),
        Some(true),
        Some(true),
        None,
        None,
    )
}

fn desc_table(tuples: Vec<i32>, slots: Vec<TSlotDescriptor>) -> TDescriptorTable {
    TDescriptorTable::new(
        Some(slots),
        tuples
            .into_iter()
            .map(|tuple_id| TTupleDescriptor::new(Some(tuple_id), None, None, None, None))
            .collect(),
        None,
        None,
    )
}

/// A StarRocks expression node with every optional payload cleared.
fn base_expr_node(node_type: TExprNodeType, ty: TTypeDesc, num_children: i32) -> TExprNode {
    TExprNode {
        node_type,
        type_: ty,
        opcode: None,
        num_children,
        agg_expr: None,
        bool_literal: None,
        case_expr: None,
        date_literal: None,
        float_literal: None,
        int_literal: None,
        in_predicate: None,
        is_null_pred: None,
        like_pred: None,
        literal_pred: None,
        slot_ref: None,
        string_literal: None,
        tuple_is_null_pred: None,
        info_func: None,
        decimal_literal: None,
        output_scale: -1,
        fn_call_expr: None,
        large_int_literal: None,
        output_column: None,
        output_type: None,
        vector_opcode: None,
        fn_: None,
        vararg_start_idx: None,
        child_type: None,
        vslot_ref: None,
        used_subfield_names: None,
        binary_literal: None,
        copy_flag: None,
        check_is_out_of_bounds: None,
        use_vectorized: None,
        has_nullable_child: None,
        is_nullable: None,
        child_type_desc: None,
        is_monotonic: None,
        dict_query_expr: None,
        dictionary_get_expr: None,
        is_index_only_filter: None,
        is_nondeterministic: None,
        cast_struct_by_name: None,
    }
}

fn slot_ref(slot_id: i32, tuple_id: i32, ty: TTypeDesc) -> TExpr {
    let mut node = base_expr_node(TExprNodeType::SLOT_REF, ty, 0);
    node.slot_ref = Some(TSlotRef::new(slot_id, tuple_id));
    TExpr::new(vec![node])
}

/// An aggregate measure in flat preorder form, with the phase flag the classifier reads.
fn aggregate_expr(name: &str, ret_type: TTypeDesc, child: Option<TExpr>, is_merge: bool) -> TExpr {
    let num_children = child.as_ref().map(|_| 1).unwrap_or(0);
    let mut node = base_expr_node(TExprNodeType::AGG_EXPR, ret_type.clone(), num_children);
    node.agg_expr = Some(TAggregateExpr::new(is_merge));
    node.fn_ = Some(TFunction::new(
        TFunctionName::new(None, name.to_string()),
        TFunctionBinaryType::BUILTIN,
        Vec::new(),
        ret_type,
        false,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ));
    let mut nodes = vec![node];
    if let Some(child) = child {
        nodes.extend(child.nodes);
    }
    TExpr::new(nodes)
}

fn base_plan_node(
    node_id: i32,
    node_type: TPlanNodeType,
    num_children: i32,
    row_tuples: Vec<i32>,
) -> TPlanNode {
    TPlanNode {
        node_id,
        node_type,
        num_children,
        limit: -1,
        row_tuples,
        nullable_tuples: Vec::new(),
        conjuncts: None,
        compact_data: false,
        common: None,
        hash_join_node: None,
        agg_node: None,
        sort_node: None,
        merge_node: None,
        exchange_node: None,
        mysql_scan_node: None,
        olap_scan_node: None,
        file_scan_node: None,
        schema_scan_node: None,
        meta_scan_node: None,
        analytic_node: None,
        union_node: None,
        resource_profile: None,
        es_scan_node: None,
        repeat_node: None,
        assert_num_rows_node: None,
        intersect_node: None,
        except_node: None,
        merge_join_node: None,
        raw_values_node: None,
        use_vectorized: None,
        hdfs_scan_node: None,
        project_node: None,
        table_function_node: None,
        probe_runtime_filters: None,
        decode_node: None,
        local_rf_waiting_set: None,
        filter_null_value_columns: None,
        need_create_tuple_columns: None,
        jdbc_scan_node: None,
        connector_scan_node: None,
        cross_join_node: None,
        lake_scan_node: None,
        nestloop_join_node: None,
        stream_scan_node: None,
        stream_join_node: None,
        stream_agg_node: None,
        select_node: None,
        fetch_node: None,
        look_up_node: None,
        cache_stats_scan_node: None,
    }
}

/// An unpartitioned (gather) exchange node carrying `tuple_id`.
fn exchange_node(tuple_id: i32) -> TPlanNode {
    let mut node = base_plan_node(
        EXCHANGE_NODE_ID,
        TPlanNodeType::EXCHANGE_NODE,
        0,
        vec![tuple_id],
    );
    node.exchange_node = Some(TExchangeNode::new(
        vec![tuple_id],
        None,
        None,
        Some(TPartitionType::UNPARTITIONED),
        Some(true),
        None,
    ));
    node
}

fn aggregation_node(
    node_id: i32,
    output_tuple: i32,
    grouping: Vec<TExpr>,
    aggregates: Vec<TExpr>,
) -> TPlanNode {
    let mut node = base_plan_node(
        node_id,
        TPlanNodeType::AGGREGATION_NODE,
        1,
        vec![output_tuple],
    );
    node.agg_node = Some(TAggregationNode::new(
        Some(grouping),
        aggregates,
        output_tuple,
        output_tuple,
        true,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ));
    node
}

fn params(plan: TPlan, desc_tbl: TDescriptorTable) -> TExecPlanFragmentParams {
    TExecPlanFragmentParams {
        protocol_version: InternalServiceVersion::V1,
        fragment: Some(TPlanFragment {
            plan: Some(plan),
            output_exprs: None,
            output_sink: None,
            partition: TDataPartition::new(TPartitionType::UNPARTITIONED, None, None, None),
            min_reservation_bytes: None,
            initial_reservation_total_claims: None,
            query_global_dicts: None,
            load_global_dicts: None,
            cache_param: None,
            query_global_dict_exprs: None,
            group_execution_param: None,
        }),
        desc_tbl: Some(desc_tbl),
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
