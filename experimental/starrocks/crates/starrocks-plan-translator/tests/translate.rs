use std::collections::BTreeMap;

use starrocks_plan_translator::{
    ExchangeInput, ExtensionRegistry, PlanTranslator, TranslateError, TranslatedPlan, URN_BOOLEAN,
    URN_COMPARISON, translate_fragment,
};
use starrocks_thrift::data_sinks::{TDataSink, TDataSinkType, TDataStreamSink};
use starrocks_thrift::descriptors::{
    TDescriptorTable, TSlotDescriptor, TTableDescriptor, TTupleDescriptor,
};
use starrocks_thrift::exprs::{
    TAggregateExpr, TBoolLiteral, TCaseExpr, TDateLiteral, TDecimalLiteral, TExpr, TExprNode,
    TExprNodeType, TFloatLiteral, TInPredicate, TIntLiteral, TIsNullPredicate, TSlotRef,
    TStringLiteral,
};
use starrocks_thrift::internal_service::{
    InternalServiceVersion, TExecPlanFragmentParams, TPlanFragmentExecParams, TScanRangeParams,
};
use starrocks_thrift::opcodes::TExprOpcode;
use starrocks_thrift::partitions::{TDataPartition, TPartitionType};
use starrocks_thrift::plan_nodes::{
    TAggregationNode, TBrokerRangeDesc, TBrokerScanRange, TBrokerScanRangeParams, TEqJoinCondition,
    TExchangeNode, TFileFormatType, TFileScanNode, TFileScanType, THashJoinNode, TJoinOp,
    TNestLoopJoinNode, TPlan, TPlanNode, TPlanNodeType, TProjectNode, TScanRange, TSelectNode,
    TSortInfo, TSortNode,
};
use starrocks_thrift::planner::TPlanFragment;
use starrocks_thrift::types::{
    TFileType, TFunction, TFunctionBinaryType, TFunctionName, TPrimitiveType, TScalarType,
    TTableType, TTypeDesc, TTypeNode, TTypeNodeType, TUniqueId,
};
use substrait::proto::{expression, plan_rel, read_rel, rel};

/// Builds a scalar StarRocks type descriptor with no length or decimal metadata.
fn scalar_type(primitive: TPrimitiveType) -> TTypeDesc {
    scalar_type_with(primitive, None, None, None)
}

/// Builds a scalar StarRocks type descriptor with optional width, precision, and scale.
fn scalar_type_with(
    primitive: TPrimitiveType,
    len: Option<i32>,
    precision: Option<i32>,
    scale: Option<i32>,
) -> TTypeDesc {
    TTypeDesc::new(Some(vec![TTypeNode::new(
        TTypeNodeType::SCALAR,
        Some(TScalarType::new(primitive, len, precision, scale)),
        None,
        None,
    )]))
}

/// Builds a non-scalar StarRocks type descriptor for unsupported-type tests.
fn complex_type(kind: TTypeNodeType) -> TTypeDesc {
    TTypeDesc::new(Some(vec![TTypeNode::new(kind, None, None, None)]))
}

/// Builds a materialized slot descriptor owned by a test tuple.
///
/// `column_pos` is always -1: the FE sets it unconditionally and the IDL marks it deprecated, so
/// a fixture carrying a real position would be a shape the translator never sees.
fn slot(id: i32, tuple_id: i32, name: &str, ty: TTypeDesc) -> TSlotDescriptor {
    TSlotDescriptor::new(
        Some(id),
        Some(tuple_id),
        Some(ty),
        Some(-1),
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

/// Builds a StarRocks table descriptor, threading only the fields these tests vary.
fn table_descriptor(id: i64, db: &str, name: &str, num_cols: i32) -> TTableDescriptor {
    TTableDescriptor::new(
        id,
        TTableType::HDFS_TABLE,
        num_cols,
        0,
        name.to_string(),
        db.to_string(),
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
    )
}

/// Builds a minimal descriptor table with the shared `tpch.users` table descriptor.
fn desc_table(tuples: Vec<(i32, Option<i64>)>, slots: Vec<TSlotDescriptor>) -> TDescriptorTable {
    TDescriptorTable::new(
        Some(slots),
        tuples
            .into_iter()
            .map(|(tuple_id, table_id)| {
                TTupleDescriptor::new(Some(tuple_id), None, None, table_id, None)
            })
            .collect(),
        Some(vec![table_descriptor(100, "tpch", "users", 2)]),
        None,
    )
}

/// Builds a StarRocks expression node with every optional expression payload cleared.
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

/// Builds a single-node slot-reference expression in StarRocks flat preorder form.
fn slot_ref(slot_id: i32, tuple_id: i32, ty: TTypeDesc) -> TExpr {
    let mut node = base_expr_node(TExprNodeType::SLOT_REF, ty, 0);
    node.slot_ref = Some(TSlotRef::new(slot_id, tuple_id));
    TExpr::new(vec![node])
}

/// Builds a BIGINT literal expression for comparisons that do not care about width.
fn int_literal(value: i64) -> TExpr {
    int_literal_typed(value, TPrimitiveType::BIGINT)
}

/// Builds an integer literal expression with the requested StarRocks primitive width.
fn int_literal_typed(value: i64, primitive: TPrimitiveType) -> TExpr {
    let mut node = base_expr_node(TExprNodeType::INT_LITERAL, scalar_type(primitive), 0);
    node.int_literal = Some(TIntLiteral::new(value));
    TExpr::new(vec![node])
}

/// Builds a floating-point literal expression with the requested StarRocks primitive width.
fn float_literal_typed(value: f64, primitive: TPrimitiveType) -> TExpr {
    let mut node = base_expr_node(TExprNodeType::FLOAT_LITERAL, scalar_type(primitive), 0);
    node.float_literal = Some(TFloatLiteral::new(value.into()));
    TExpr::new(vec![node])
}

/// Builds a VARCHAR literal expression.
fn string_literal(value: &str) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::STRING_LITERAL,
        scalar_type(TPrimitiveType::VARCHAR),
        0,
    );
    node.string_literal = Some(TStringLiteral::new(value.to_string()));
    TExpr::new(vec![node])
}

/// Builds a BOOLEAN literal expression.
fn bool_literal(value: bool) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::BOOL_LITERAL,
        scalar_type(TPrimitiveType::BOOLEAN),
        0,
    );
    node.bool_literal = Some(TBoolLiteral::new(value));
    TExpr::new(vec![node])
}

/// Builds an arithmetic expression and appends child nodes in preorder.
fn arithmetic(opcode: TExprOpcode, left: TExpr, right: TExpr) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::ARITHMETIC_EXPR,
        scalar_type(TPrimitiveType::BIGINT),
        2,
    );
    node.opcode = Some(opcode);
    let mut nodes = vec![node];
    nodes.extend(left.nodes);
    nodes.extend(right.nodes);
    TExpr::new(nodes)
}

/// Builds a binary predicate expression and appends child nodes in preorder.
fn binary_pred(opcode: TExprOpcode, left: TExpr, right: TExpr) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::BINARY_PRED,
        scalar_type(TPrimitiveType::BOOLEAN),
        2,
    );
    node.opcode = Some(opcode);
    let mut nodes = vec![node];
    nodes.extend(left.nodes);
    nodes.extend(right.nodes);
    TExpr::new(nodes)
}

/// Builds a StarRocks plan node with all node-specific payloads cleared.
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

/// Builds a supported file-scan node for the given tuple.
fn scan_node(node_id: i32, tuple_id: i32) -> TPlanNode {
    let mut node = base_plan_node(node_id, TPlanNodeType::FILE_SCAN_NODE, 0, vec![tuple_id]);
    node.file_scan_node = Some(TFileScanNode::new(tuple_id, None, None, None));
    node
}

/// Builds the execution-fragment params passed to the public translator API.
fn params(
    plan: Option<TPlan>,
    desc_tbl: Option<TDescriptorTable>,
    output_exprs: Option<Vec<TExpr>>,
) -> TExecPlanFragmentParams {
    TExecPlanFragmentParams {
        protocol_version: InternalServiceVersion::V1,
        fragment: Some(TPlanFragment {
            plan,
            output_exprs,
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
        desc_tbl,
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

/// Builds params with an absent fragment to exercise top-level validation.
fn params_without_fragment(desc_tbl: Option<TDescriptorTable>) -> TExecPlanFragmentParams {
    let mut params = params(Some(TPlan::new(vec![scan_node(0, 0)])), desc_tbl, None);
    params.fragment = None;
    params
}

/// Builds the default scan descriptor used by most positive translator tests.
fn base_desc() -> TDescriptorTable {
    desc_table(
        vec![(0, Some(100))],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
        ],
    )
}

/// Extracts the single root relation emitted by these tests.
fn root(plan: &substrait::proto::Plan) -> &substrait::proto::RelRoot {
    match plan.relations[0].rel_type.as_ref().unwrap() {
        plan_rel::RelType::Root(root) => root,
        _ => panic!("expected root relation"),
    }
}

/// Extracts the filter condition from a plan whose root input is a filter relation.
fn filter_condition(plan: &substrait::proto::Plan) -> &substrait::proto::Expression {
    match root(plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    {
        rel::RelType::Filter(filter) => filter.condition.as_deref().unwrap(),
        other => panic!("expected filter rel, got {other:?}"),
    }
}

/// Extracts a scalar-function argument by index for expression-shape assertions.
fn scalar_arg(expr: &substrait::proto::Expression, idx: usize) -> &substrait::proto::Expression {
    let scalar = match expr.rex_type.as_ref().unwrap() {
        expression::RexType::ScalarFunction(scalar) => scalar,
        other => panic!("expected scalar function, got {other:?}"),
    };
    match scalar.arguments[idx].arg_type.as_ref().unwrap() {
        substrait::proto::function_argument::ArgType::Value(expr) => expr,
        other => panic!("expected scalar value argument, got {other:?}"),
    }
}

/// Extracts the literal payload from a Substrait literal expression.
fn literal_type(expr: &substrait::proto::Expression) -> &expression::literal::LiteralType {
    match expr.rex_type.as_ref().unwrap() {
        expression::RexType::Literal(literal) => literal.literal_type.as_ref().unwrap(),
        other => panic!("expected literal, got {other:?}"),
    }
}

/// Verifies a scan-only fragment becomes a Substrait named-table read.
#[test]
fn scan_only_produces_named_table() {
    let translated = PlanTranslator::new()
        .translate_fragment(&params(
            Some(TPlan::new(vec![scan_node(0, 0)])),
            Some(base_desc()),
            None,
        ))
        .unwrap();

    assert!(!translated.to_substrait_bytes().is_empty());
    assert_eq!(translated.output_names, vec!["id", "name"]);
    let input = root(&translated.plan).input.as_ref().unwrap();
    match input.rel_type.as_ref().unwrap() {
        rel::RelType::Read(read) => {
            assert_eq!(read.base_schema.as_ref().unwrap().names, vec!["id", "name"]);
            match read.read_type.as_ref().unwrap() {
                read_rel::ReadType::NamedTable(table) => {
                    assert_eq!(table.names, vec!["tpch", "users"]);
                }
                other => panic!("expected named table, got {other:?}"),
            }
        }
        other => panic!("expected read rel, got {other:?}"),
    }
}

/// Builds a single local broker scan range for `path` with the given format,
/// start offset, size, and optional total file size.
fn broker_scan_range(
    path: &str,
    format: TFileFormatType,
    start_offset: i64,
    size: i64,
    file_size: Option<i64>,
) -> TScanRange {
    let range = TBrokerRangeDesc::new(
        TFileType::FILE_BROKER,
        format,
        false,
        path.to_string(),
        start_offset,
        size,
        None,
        file_size,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let mut params = TBrokerScanRangeParams::new(
        0,
        0,
        0,
        Vec::new(),
        0,
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
    );
    // Production-shaped supported slice: a FILES() query read with direct (non-broker)
    // access; collection rejects anything else.
    params.file_scan_type = Some(TFileScanType::FILES_QUERY);
    params.use_broker = Some(false);
    let broker = TBrokerScanRange::new(vec![range], params, Vec::new(), None, None, None, None);
    TScanRange::new(None, None, Some(broker), None, None, None)
}

/// Builds fragment params whose `node_id` scan carries `scan_range`.
fn params_with_scan_range(
    plan: TPlan,
    desc_tbl: TDescriptorTable,
    node_id: i32,
    scan_range: TScanRange,
) -> TExecPlanFragmentParams {
    let mut fragment_params = params(Some(plan), Some(desc_tbl), None);
    let mut per_node_scan_ranges = BTreeMap::new();
    per_node_scan_ranges.insert(
        node_id,
        vec![TScanRangeParams::new(scan_range, None, None, None)],
    );
    fragment_params.params = Some(TPlanFragmentExecParams::new(
        TUniqueId::new(0, 0),
        TUniqueId::new(0, 0),
        per_node_scan_ranges,
        BTreeMap::new(),
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
    fragment_params
}

/// Builds fragment params whose `node_id` scan carries `scan_range` via the
/// pipeline per-driver-sequence map instead of `per_node_scan_ranges`.
fn params_with_per_driver_scan_range(
    plan: TPlan,
    desc_tbl: TDescriptorTable,
    node_id: i32,
    scan_range: TScanRange,
) -> TExecPlanFragmentParams {
    let mut fragment_params = params(Some(plan), Some(desc_tbl), None);
    let mut per_seq = BTreeMap::new();
    per_seq.insert(0, vec![TScanRangeParams::new(scan_range, None, None, None)]);
    let mut per_driver = BTreeMap::new();
    per_driver.insert(node_id, per_seq);
    fragment_params.params = Some(TPlanFragmentExecParams::new(
        TUniqueId::new(0, 0),
        TUniqueId::new(0, 0),
        BTreeMap::new(),
        BTreeMap::new(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        per_driver,
        None,
        None,
        None,
        None,
    ));
    fragment_params
}

/// Verifies a scan with broker ranges becomes a Substrait `local_files` parquet read.
#[test]
fn scan_with_broker_ranges_produces_local_files() {
    let path = "file:///data/users.parquet";
    let translated = PlanTranslator::new()
        .translate_fragment(&params_with_scan_range(
            TPlan::new(vec![scan_node(0, 0)]),
            base_desc(),
            0,
            broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 0, -1, Some(1024)),
        ))
        .unwrap();

    assert_eq!(translated.output_names, vec!["id", "name"]);
    let input = root(&translated.plan).input.as_ref().unwrap();
    match input.rel_type.as_ref().unwrap() {
        rel::RelType::Read(read) => {
            assert_eq!(read.base_schema.as_ref().unwrap().names, vec!["id", "name"]);
            match read.read_type.as_ref().unwrap() {
                read_rel::ReadType::LocalFiles(local) => {
                    assert_eq!(local.items.len(), 1);
                    let item = &local.items[0];
                    assert!(matches!(
                        item.file_format.as_ref(),
                        Some(read_rel::local_files::file_or_files::FileFormat::Parquet(_))
                    ));
                    match item.path_type.as_ref().unwrap() {
                        read_rel::local_files::file_or_files::PathType::UriFile(uri) => {
                            assert_eq!(uri, path);
                        }
                        other => panic!("expected uri_file, got {other:?}"),
                    }
                }
                other => panic!("expected local files, got {other:?}"),
            }
        }
        other => panic!("expected read rel, got {other:?}"),
    }
}

/// Verifies a non-parquet broker scan range is rejected as unsupported.
#[test]
fn non_parquet_broker_range_is_unsupported() {
    let err = PlanTranslator::new()
        .translate_fragment(&params_with_scan_range(
            TPlan::new(vec![scan_node(0, 0)]),
            base_desc(),
            0,
            broker_scan_range(
                "file:///data/users.orc",
                TFileFormatType::FORMAT_ORC,
                0,
                -1,
                None,
            ),
        ))
        .unwrap_err();
    assert!(matches!(
        err,
        TranslateError::UnsupportedScanRange { node_id: 0, .. }
    ));
}

/// Verifies a byte-range split broker scan range is rejected as unsupported,
/// both for a non-zero start offset and for a first split (offset 0, partial size).
#[test]
fn split_broker_range_is_unsupported() {
    for range in [
        broker_scan_range(
            "file:///data/users.parquet",
            TFileFormatType::FORMAT_PARQUET,
            1024,
            -1,
            None,
        ),
        broker_scan_range(
            "file:///data/users.parquet",
            TFileFormatType::FORMAT_PARQUET,
            0,
            512,
            Some(1024),
        ),
    ] {
        let err = PlanTranslator::new()
            .translate_fragment(&params_with_scan_range(
                TPlan::new(vec![scan_node(0, 0)]),
                base_desc(),
                0,
                range,
            ))
            .unwrap_err();
        assert!(matches!(
            err,
            TranslateError::UnsupportedScanRange { node_id: 0, .. }
        ));
    }
}

/// Verifies incremental scan-range delivery is refused: this CN never receives the rest, so
/// accepting the prefix would silently read a subset of the data.
#[test]
fn has_more_scan_ranges_are_refused() {
    let mut fragment = params_with_scan_range(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        0,
        broker_scan_range(
            "file:///data/users.parquet",
            TFileFormatType::FORMAT_PARQUET,
            0,
            -1,
            Some(1024),
        ),
    );
    fragment
        .params
        .as_mut()
        .unwrap()
        .per_node_scan_ranges
        .get_mut(&0)
        .unwrap()[0]
        .has_more = Some(true);
    let err = PlanTranslator::new()
        .translate_fragment(&fragment)
        .unwrap_err();
    let TranslateError::UnsupportedScanRange { node_id, reason } = err else {
        panic!("expected an unsupported scan range, got {err:?}");
    };
    assert_eq!(node_id, 0);
    assert_eq!(
        reason,
        "incremental scan-range delivery (has_more) is not supported"
    );
}

/// The FE ends a connector scan's assignment with a placeholder `TScanRangeParams` that carries
/// an empty `scan_range`, `empty = true` and `has_more` telling whether more ranges follow. The
/// placeholder itself is skipped; only `has_more = true` refuses the fragment, so the order of the
/// two checks matters and is pinned here.
#[test]
fn empty_placeholder_scan_ranges_are_skipped() {
    let path = "file:///data/users.parquet";
    let placeholder = |has_more: bool| {
        TScanRangeParams::new(TScanRange::default(), None, Some(true), Some(has_more))
    };
    let with_placeholder = |has_more: bool| {
        let mut fragment = params_with_scan_range(
            TPlan::new(vec![scan_node(0, 0)]),
            base_desc(),
            0,
            broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 0, -1, Some(1024)),
        );
        fragment
            .params
            .as_mut()
            .unwrap()
            .per_node_scan_ranges
            .get_mut(&0)
            .unwrap()
            .push(placeholder(has_more));
        fragment
    };

    let translated = PlanTranslator::new()
        .translate_fragment(&with_placeholder(false))
        .expect("an empty placeholder without has_more is skipped");
    let rel::RelType::Read(read) = root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    else {
        panic!("expected local read");
    };
    let Some(read_rel::ReadType::LocalFiles(files)) = read.read_type.as_ref() else {
        panic!("expected local files");
    };
    assert_eq!(files.items.len(), 1);

    let err = PlanTranslator::new()
        .translate_fragment(&with_placeholder(true))
        .unwrap_err();
    let TranslateError::UnsupportedScanRange { reason, .. } = err else {
        panic!("expected an unsupported scan range, got {err:?}");
    };
    assert_eq!(
        reason,
        "incremental scan-range delivery (has_more) is not supported"
    );
}

/// Verifies complete byte-range splits are collapsed to one whole-file local read.
#[test]
fn complete_split_broker_ranges_produce_one_local_file() {
    let path = "file:///data/users.parquet";
    let mut fragment = params_with_scan_range(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        0,
        broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 0, 512, Some(1024)),
    );
    fragment
        .params
        .as_mut()
        .unwrap()
        .per_node_scan_ranges
        .get_mut(&0)
        .unwrap()
        .push(TScanRangeParams::new(
            broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 512, 512, Some(1024)),
            None,
            None,
            None,
        ));
    let translated = PlanTranslator::new().translate_fragment(&fragment).unwrap();
    let rel::RelType::Read(read) = root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    else {
        panic!("expected local read");
    };
    let Some(read_rel::ReadType::LocalFiles(files)) = read.read_type.as_ref() else {
        panic!("expected local files");
    };
    assert_eq!(files.items.len(), 1);
}

/// Builds fragment params whose node-0 scan carries every `(start, size)` split of `path`, all
/// declaring `file_size`, in the order given.
fn params_with_splits(
    path: &str,
    file_size: i64,
    splits: &[(i64, i64)],
) -> TExecPlanFragmentParams {
    let (first, rest) = splits.split_first().expect("at least one split");
    let mut fragment = params_with_scan_range(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        0,
        broker_scan_range(
            path,
            TFileFormatType::FORMAT_PARQUET,
            first.0,
            first.1,
            Some(file_size),
        ),
    );
    let ranges = fragment
        .params
        .as_mut()
        .unwrap()
        .per_node_scan_ranges
        .get_mut(&0)
        .unwrap();
    for &(start, size) in rest {
        ranges.push(TScanRangeParams::new(
            broker_scan_range(
                path,
                TFileFormatType::FORMAT_PARQUET,
                start,
                size,
                Some(file_size),
            ),
            None,
            None,
            None,
        ));
    }
    fragment
}

/// Translates `splits` and returns the `UnsupportedScanRange` reason they were refused with.
fn splits_rejected_because(file_size: i64, splits: &[(i64, i64)]) -> &'static str {
    let err = PlanTranslator::new()
        .translate_fragment(&params_with_splits(
            "file:///data/users.parquet",
            file_size,
            splits,
        ))
        .unwrap_err();
    let TranslateError::UnsupportedScanRange { node_id, reason } = err else {
        panic!("expected an unsupported scan range, got {err:?}");
    };
    assert_eq!(node_id, 0);
    reason
}

/// Splits that leave a hole are refused: collapsing them to a whole-file read would scan the
/// hole, which a sibling instance is already scanning.
#[test]
fn split_broker_ranges_with_a_gap_are_unsupported() {
    assert_eq!(
        splits_rejected_because(1024, &[(0, 256), (512, 512)]),
        "byte-range splits do not tile the parquet file"
    );
}

/// Splits that tile a prefix but stop short of the file are refused: the tail would be dropped.
#[test]
fn split_broker_ranges_covering_only_a_prefix_are_unsupported() {
    assert_eq!(
        splits_rejected_because(1024, &[(0, 256), (256, 256)]),
        "byte-range splits do not tile the parquet file"
    );
}

/// A split that runs past the end of the file is malformed metadata, not a whole-file read. The
/// sweep only asked whether EOF had been reached, so `(512, 1024)` over a 1024-byte file collapsed
/// to a whole-file read instead of being refused.
#[test]
fn split_broker_ranges_extending_past_eof_are_unsupported() {
    assert_eq!(
        splits_rejected_because(1024, &[(0, 512), (512, 1024)]),
        "byte-range split extends past the end of the parquet file"
    );
    assert_eq!(
        splits_rejected_because(1024, &[(0, 2048)]),
        "byte-range split extends past the end of the parquet file"
    );
}

/// A zero-length split covers nothing, so it can never be part of a tiling: `(0, 0)` is refused
/// outright, and a zero-length tail `(1024, -1)` after a whole-file split trips the same rule.
#[test]
fn zero_size_split_is_unsupported() {
    assert_eq!(
        splits_rejected_because(1024, &[(0, 0)]),
        "byte-range splits do not tile the parquet file"
    );
    assert_eq!(
        splits_rejected_because(1024, &[(0, 1024), (1024, -1)]),
        "byte-range splits do not tile the parquet file"
    );
}

/// Two splits that both cover the head of the file are refused. They "cover" every byte, so a
/// sweep that only rejects gaps collapses them into one whole-file read — Sirius would scan the
/// shared row groups once where StarRocks scans them on both instances, and `count(*)` would
/// disagree with no error.
#[test]
fn overlapping_split_broker_ranges_are_unsupported() {
    assert_eq!(
        splits_rejected_because(1024, &[(0, 1024), (0, 512)]),
        "byte-range splits do not tile the parquet file"
    );
}

/// Splits of one file that disagree on how big that file is are refused — the coverage check
/// would otherwise be measured against an arbitrary one of them.
#[test]
fn split_broker_ranges_disagreeing_on_file_size_are_unsupported() {
    let path = "file:///data/users.parquet";
    let mut fragment = params_with_splits(path, 1024, &[(0, 512)]);
    fragment
        .params
        .as_mut()
        .unwrap()
        .per_node_scan_ranges
        .get_mut(&0)
        .unwrap()
        .push(TScanRangeParams::new(
            broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 512, 512, Some(2048)),
            None,
            None,
            None,
        ));
    let err = PlanTranslator::new()
        .translate_fragment(&fragment)
        .unwrap_err();
    let TranslateError::UnsupportedScanRange { reason, .. } = err else {
        panic!("expected an unsupported scan range, got {err:?}");
    };
    assert_eq!(reason, "scan ranges disagree on the parquet file size");
}

/// A split missing its `file_size` reports as missing wherever it lands in the list, not
/// only when it happens to be inserted first. Checking agreement before presence made the
/// message depend on arrival order: the same fragment reported "missing" or "disagree"
/// depending on which range the FE sent first.
#[test]
fn split_broker_range_missing_file_size_after_the_first_is_unsupported() {
    let path = "file:///data/users.parquet";
    let mut fragment = params_with_splits(path, 1024, &[(0, 512)]);
    fragment
        .params
        .as_mut()
        .unwrap()
        .per_node_scan_ranges
        .get_mut(&0)
        .unwrap()
        .push(TScanRangeParams::new(
            broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 512, 512, None),
            None,
            None,
            None,
        ));
    let err = PlanTranslator::new()
        .translate_fragment(&fragment)
        .unwrap_err();
    let TranslateError::UnsupportedScanRange { reason, .. } = err else {
        panic!("expected an unsupported scan range, got {err:?}");
    };
    assert_eq!(reason, "scan range is missing the parquet file size");
}

/// Splits arriving out of order still tile the file: the sweep sorts before checking, and the FE
/// does not promise an order. Without the sort this case would be refused.
#[test]
fn split_broker_ranges_in_descending_order_are_accepted() {
    let translated = PlanTranslator::new()
        .translate_fragment(&params_with_splits(
            "file:///data/users.parquet",
            1024,
            &[(512, 512), (0, 512)],
        ))
        .unwrap();
    let rel::RelType::Read(read) = root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    else {
        panic!("expected local read");
    };
    let Some(read_rel::ReadType::LocalFiles(files)) = read.read_type.as_ref() else {
        panic!("expected local files");
    };
    assert_eq!(files.items.len(), 1);
}

/// The motivating shape: with pipeline dop the FE hands one instance several splits of the same
/// file under different driver sequences. They must be combined across sequences into one
/// whole-file read, not validated per sequence.
#[test]
fn per_driver_split_ranges_across_sequences_produce_one_local_file() {
    let path = "file:///data/users.parquet";
    let mut fragment = params_with_per_driver_scan_range(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        0,
        broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 0, 512, Some(1024)),
    );
    fragment
        .params
        .as_mut()
        .unwrap()
        .node_to_per_driver_seq_scan_ranges
        .as_mut()
        .unwrap()
        .get_mut(&0)
        .unwrap()
        .insert(
            1,
            vec![TScanRangeParams::new(
                broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 512, 512, Some(1024)),
                None,
                None,
                None,
            )],
        );
    let translated = PlanTranslator::new().translate_fragment(&fragment).unwrap();
    let rel::RelType::Read(read) = root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    else {
        panic!("expected local read");
    };
    let Some(read_rel::ReadType::LocalFiles(files)) = read.read_type.as_ref() else {
        panic!("expected local files");
    };
    assert_eq!(files.items.len(), 1);
}

/// Verifies a scan range delivered via the pipeline per-driver-sequence map is
/// collected too (not just `per_node_scan_ranges`).
#[test]
fn per_driver_scan_range_produces_local_files() {
    let path = "file:///data/users.parquet";
    let translated = PlanTranslator::new()
        .translate_fragment(&params_with_per_driver_scan_range(
            TPlan::new(vec![scan_node(0, 0)]),
            base_desc(),
            0,
            broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 0, -1, Some(1024)),
        ))
        .unwrap();

    let input = root(&translated.plan).input.as_ref().unwrap();
    match input.rel_type.as_ref().unwrap() {
        rel::RelType::Read(read) => match read.read_type.as_ref().unwrap() {
            read_rel::ReadType::LocalFiles(local) => {
                assert_eq!(local.items.len(), 1);
                match local.items[0].path_type.as_ref().unwrap() {
                    read_rel::local_files::file_or_files::PathType::UriFile(uri) => {
                        assert_eq!(uri, path);
                    }
                    other => panic!("expected uri_file, got {other:?}"),
                }
            }
            other => panic!("expected local files, got {other:?}"),
        },
        other => panic!("expected read rel, got {other:?}"),
    }
}

/// Asserts a single-node scan over `scan_range` is rejected as an unsupported range.
fn assert_scan_range_unsupported(scan_range: TScanRange) {
    let err = PlanTranslator::new()
        .translate_fragment(&params_with_scan_range(
            TPlan::new(vec![scan_node(0, 0)]),
            base_desc(),
            0,
            scan_range,
        ))
        .unwrap_err();
    assert!(
        matches!(err, TranslateError::UnsupportedScanRange { node_id: 0, .. }),
        "expected UnsupportedScanRange, got {err:?}"
    );
}

/// A whole-file local parquet `FILES()` range used as the base for negative cases.
fn parquet_query_range(path: &str) -> TScanRange {
    broker_scan_range(path, TFileFormatType::FORMAT_PARQUET, 0, -1, Some(1024))
}

/// Verifies a broker range with path-derived columns is rejected as unsupported.
#[test]
fn path_derived_columns_are_unsupported() {
    let mut scan_range = parquet_query_range("file:///data/users.parquet");
    scan_range.broker_scan_range.as_mut().unwrap().ranges[0].columns_from_path =
        Some(vec!["dt".to_string()]);
    assert_scan_range_unsupported(scan_range);
}

/// Verifies a load scan (not a FILES() query) is rejected: only query reads map to
/// a plain `parquet_scan`.
#[test]
fn load_scan_range_is_unsupported() {
    let mut scan_range = parquet_query_range("file:///data/users.parquet");
    scan_range
        .broker_scan_range
        .as_mut()
        .unwrap()
        .params
        .file_scan_type = Some(TFileScanType::LOAD);
    assert_scan_range_unsupported(scan_range);
}

/// Verifies broker-mediated access is rejected: Sirius's reader does not use a broker.
#[test]
fn broker_mediated_scan_range_is_unsupported() {
    let mut scan_range = parquet_query_range("file:///data/users.parquet");
    scan_range
        .broker_scan_range
        .as_mut()
        .unwrap()
        .params
        .use_broker = Some(true);
    assert_scan_range_unsupported(scan_range);
}

/// Verifies flexible (name-based, null-filling) column mapping is rejected.
#[test]
fn flexible_column_mapping_is_unsupported() {
    let mut scan_range = parquet_query_range("file:///data/users.parquet");
    scan_range
        .broker_scan_range
        .as_mut()
        .unwrap()
        .params
        .flexible_column_mapping = Some(true);
    assert_scan_range_unsupported(scan_range);
}

/// Verifies a destination column produced by a transform (here a literal default,
/// not a bare slot reference) is rejected rather than silently dropped.
#[test]
fn dest_column_transform_is_unsupported() {
    let mut scan_range = parquet_query_range("file:///data/users.parquet");
    scan_range
        .broker_scan_range
        .as_mut()
        .unwrap()
        .params
        .expr_of_dest_slot = Some(BTreeMap::from([(1, int_literal(7))]));
    assert_scan_range_unsupported(scan_range);
}

/// Verifies an explicit identity column mapping (bare slot references) is accepted.
#[test]
fn identity_dest_column_mapping_is_supported() {
    let mut scan_range = parquet_query_range("file:///data/users.parquet");
    scan_range
        .broker_scan_range
        .as_mut()
        .unwrap()
        .params
        .expr_of_dest_slot = Some(BTreeMap::from([
        (1, slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        (2, slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))),
    ]));
    let translated = PlanTranslator::new()
        .translate_fragment(&params_with_scan_range(
            TPlan::new(vec![scan_node(0, 0)]),
            base_desc(),
            0,
            scan_range,
        ))
        .unwrap();
    let input = root(&translated.plan).input.as_ref().unwrap();
    assert!(matches!(
        input.rel_type.as_ref().unwrap(),
        rel::RelType::Read(read)
            if matches!(read.read_type.as_ref().unwrap(), read_rel::ReadType::LocalFiles(_))
    ));
}

/// Verifies a remote URI scheme is rejected: credentials/endpoints are not propagated.
#[test]
fn remote_scheme_scan_range_is_unsupported() {
    assert_scan_range_unsupported(parquet_query_range("s3://bucket/users.parquet"));
}

/// Verifies a path with glob metacharacters is rejected: `parquet_scan` would re-expand it.
#[test]
fn glob_path_scan_range_is_unsupported() {
    assert_scan_range_unsupported(parquet_query_range("file:///data/*.parquet"));
}

/// Verifies a bounded-size range with unknown file size is rejected: it cannot be
/// proven to cover the whole file, and the size is dropped by `local_files`.
#[test]
fn bounded_split_with_unknown_file_size_is_unsupported() {
    assert_scan_range_unsupported(broker_scan_range(
        "file:///data/users.parquet",
        TFileFormatType::FORMAT_PARQUET,
        0,
        512,
        None,
    ));
}

/// Verifies an empty (zero-byte) file range is rejected: it is not a readable
/// parquet file, so it must not be passed to `parquet_scan`. Pins the reason too —
/// an empty file used to be reported as a file-size disagreement, which sent the
/// reader hunting for a second, differing range that does not exist.
#[test]
fn empty_file_scan_range_is_unsupported() {
    let err = PlanTranslator::new()
        .translate_fragment(&params_with_scan_range(
            TPlan::new(vec![scan_node(0, 0)]),
            base_desc(),
            0,
            broker_scan_range(
                "file:///data/users.parquet",
                TFileFormatType::FORMAT_PARQUET,
                0,
                0,
                Some(0),
            ),
        ))
        .unwrap_err();
    let TranslateError::UnsupportedScanRange { reason, .. } = err else {
        panic!("expected an unsupported scan range, got {err:?}");
    };
    assert_eq!(
        reason,
        "parquet scan range reports an empty or negative file size"
    );
}

/// Verifies a renamed column mapping is rejected: destination slot 1 ("id") fed
/// from source slot 2 ("name") would have the reader read the wrong column by name.
#[test]
fn renamed_column_mapping_is_unsupported() {
    let mut scan_range = parquet_query_range("file:///data/users.parquet");
    scan_range
        .broker_scan_range
        .as_mut()
        .unwrap()
        .params
        .expr_of_dest_slot = Some(BTreeMap::from([(
        1,
        slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)),
    )]));
    assert_scan_range_unsupported(scan_range);
}

/// Verifies an absent `use_broker` is rejected: it selects the broker filesystem,
/// not the direct access Sirius's reader performs.
#[test]
fn unset_use_broker_scan_range_is_unsupported() {
    let mut scan_range = parquet_query_range("file:///data/users.parquet");
    scan_range
        .broker_scan_range
        .as_mut()
        .unwrap()
        .params
        .use_broker = None;
    assert_scan_range_unsupported(scan_range);
}

/// Verifies a non-broker file descriptor (e.g. a stream) is rejected: it is a load
/// shape, not a readable file scan.
#[test]
fn stream_file_type_scan_range_is_unsupported() {
    let mut scan_range = parquet_query_range("file:///data/users.parquet");
    scan_range.broker_scan_range.as_mut().unwrap().ranges[0].file_type = TFileType::FILE_STREAM;
    assert_scan_range_unsupported(scan_range);
}

/// Verifies a `file://` URI with a non-local authority is rejected.
#[test]
fn remote_authority_file_uri_is_unsupported() {
    assert_scan_range_unsupported(parquet_query_range("file://remote-host/data/users.parquet"));
}

/// Verifies a single-slash remote scheme (no `://`) is still rejected.
#[test]
fn single_slash_remote_scheme_is_unsupported() {
    assert_scan_range_unsupported(parquet_query_range("hdfs:/data/users.parquet"));
}

/// Verifies a scan node appearing in BOTH scan-range maps is rejected: collecting
/// (and reading) its whole-file paths twice would silently duplicate rows.
#[test]
fn node_in_both_scan_range_maps_is_unsupported() {
    let mut fragment_params = params_with_scan_range(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        0,
        parquet_query_range("file:///data/users.parquet"),
    );
    let mut per_seq = BTreeMap::new();
    per_seq.insert(
        0,
        vec![TScanRangeParams::new(
            parquet_query_range("file:///data/users.parquet"),
            None,
            None,
            None,
        )],
    );
    let mut per_driver = BTreeMap::new();
    per_driver.insert(0, per_seq);
    fragment_params
        .params
        .as_mut()
        .unwrap()
        .node_to_per_driver_seq_scan_ranges = Some(per_driver);

    let err = PlanTranslator::new()
        .translate_fragment(&fragment_params)
        .unwrap_err();
    assert!(
        matches!(err, TranslateError::UnsupportedScanRange { node_id: 0, .. }),
        "expected UnsupportedScanRange, got {err:?}"
    );
}

/// Verifies duplicate descriptor names are disambiguated deterministically at the root.
#[test]
fn duplicate_output_names_are_unique_and_match_root() {
    let desc = desc_table(
        vec![(0, Some(100))],
        vec![
            slot(1, 0, "name", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name_1", scalar_type(TPrimitiveType::BIGINT)),
            slot(3, 0, "name", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );

    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    assert_eq!(translated.output_names, vec!["name", "name_1", "name_2"]);
    assert_eq!(root(&translated.plan).names, translated.output_names);
}

/// Verifies translated plans have readable explain/debug output for logs.
#[test]
fn translated_plan_explain_and_debug_are_readable() {
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap();

    let explain = translated.explain().to_string();
    assert!(explain.contains("Read"));
    assert!(explain.contains("users"));

    let debug = format!("{translated:?}");
    assert!(debug.contains("TranslatedPlan"));
    assert!(debug.contains("Read"));
}

/// Verifies select-node conjuncts wrap the child read in a Substrait filter relation.
#[test]
fn scan_filter_wraps_filter_rel() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![binary_pred(
        TExprOpcode::GT,
        slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
        int_literal(10),
    )]);

    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap();

    match root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    {
        rel::RelType::Filter(filter) => {
            assert!(filter.condition.is_some());
            assert!(matches!(
                filter.input.as_ref().unwrap().rel_type.as_ref().unwrap(),
                rel::RelType::Read(_)
            ));
        }
        other => panic!("expected filter rel, got {other:?}"),
    }
}

/// Verifies INT StarRocks literals emit Substrait i32 literals, not widened i64 values.
#[test]
fn integer_literal_preserves_expr_width() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![binary_pred(
        TExprOpcode::GT,
        slot_ref(1, 0, scalar_type(TPrimitiveType::INT)),
        int_literal_typed(10, TPrimitiveType::INT),
    )]);

    let desc = desc_table(
        vec![(0, Some(100))],
        vec![slot(1, 0, "id", scalar_type(TPrimitiveType::INT))],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    assert!(matches!(
        literal_type(scalar_arg(filter_condition(&translated.plan), 1)),
        expression::literal::LiteralType::I32(10)
    ));
}

/// Verifies FLOAT StarRocks literals emit Substrait fp32 literals, not widened fp64 values.
#[test]
fn float_literal_preserves_expr_width() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![binary_pred(
        TExprOpcode::EQ,
        float_literal_typed(1.5, TPrimitiveType::FLOAT),
        float_literal_typed(1.5, TPrimitiveType::FLOAT),
    )]);

    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap();

    assert!(matches!(
        literal_type(scalar_arg(filter_condition(&translated.plan), 0)),
        expression::literal::LiteralType::Fp32(value) if (*value - 1.5).abs() < f32::EPSILON
    ));
}

/// Verifies project expressions follow descriptor output order instead of map key order.
#[test]
fn scan_project_preserves_descriptor_output_order() {
    let mut slot_map = BTreeMap::new();
    slot_map.insert(3, slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)));
    slot_map.insert(4, slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)));

    let mut project = base_plan_node(1, TPlanNodeType::PROJECT_NODE, 1, vec![1]);
    project.project_node = Some(TProjectNode::new(Some(slot_map), None));

    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(3, 1, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(4, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );

    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![project, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    assert_eq!(translated.output_names, vec!["name", "id"]);
    match root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    {
        rel::RelType::Project(project) => assert_eq!(project.expressions.len(), 2),
        other => panic!("expected project rel, got {other:?}"),
    }
}

/// Verifies hidden project expressions are appended in key order and can be
/// referenced by visible expressions without descriptor-table slots.
#[test]
fn project_common_slots_are_materialized_before_visible_expressions() {
    let mut common_slot_map = BTreeMap::new();
    common_slot_map.insert(5, slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)));
    let mut slot_map = BTreeMap::new();
    slot_map.insert(3, slot_ref(5, 1, scalar_type(TPrimitiveType::BIGINT)));

    let mut project = base_plan_node(1, TPlanNodeType::PROJECT_NODE, 1, vec![1]);
    project.project_node = Some(TProjectNode::new(Some(slot_map), Some(common_slot_map)));
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(3, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );

    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![project, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();
    let rel::RelType::Project(visible) = root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    else {
        panic!("expected visible project");
    };
    let expression::RexType::Selection(selection) =
        visible.expressions[0].rex_type.as_ref().unwrap()
    else {
        panic!("expected common-slot selection");
    };
    let expression::field_reference::ReferenceType::DirectReference(segment) =
        selection.reference_type.as_ref().unwrap()
    else {
        panic!("expected direct field reference");
    };
    let expression::reference_segment::ReferenceType::StructField(field) =
        segment.reference_type.as_ref().unwrap()
    else {
        panic!("expected struct field");
    };
    assert_eq!(field.field, 2);

    // The hidden project must pass its two input columns through and append the common slot,
    // and the visible project must then read past all three. Asserting only that a project
    // exists leaves the emit mappings -- the single line this lowering rests on -- unobserved.
    let rel::RelType::Project(hidden) = visible.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected the hidden project under the visible one");
    };
    assert_eq!(hidden.expressions.len(), 1);
    assert_eq!(emit_mapping(hidden.common.as_ref()), vec![0, 1, 2]);
    assert_eq!(emit_mapping(visible.common.as_ref()), vec![3]);
}

/// A later hidden slot may name an earlier one. The translator appends one
/// project per entry in ascending slot id, so the second expression already
/// sees the first column. Putting every hidden expression in one `ProjectRel`
/// would evaluate them all against the scan and this case would break.
#[test]
fn nested_common_slots_are_appended_in_slot_id_order() {
    let bigint = scalar_type(TPrimitiveType::BIGINT);
    let mut common_slot_map = BTreeMap::new();
    common_slot_map.insert(4, slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)));
    common_slot_map.insert(5, slot_ref(1, 0, bigint.clone()));
    common_slot_map.insert(
        6,
        arithmetic(
            TExprOpcode::ADD,
            slot_ref(5, 1, bigint.clone()),
            int_literal(1),
        ),
    );
    let mut slot_map = BTreeMap::new();
    slot_map.insert(3, slot_ref(6, 1, bigint));

    let mut project = base_plan_node(1, TPlanNodeType::PROJECT_NODE, 1, vec![1]);
    project.project_node = Some(TProjectNode::new(Some(slot_map), Some(common_slot_map)));
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(3, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );

    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![project, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();
    let visible = as_project(root(&translated.plan).input.as_ref().unwrap());
    assert_eq!(struct_field(&visible.expressions[0]), 4);
    assert_eq!(emit_mapping(visible.common.as_ref()), vec![5]);

    let cse2 = as_project(visible.input.as_ref().unwrap());
    assert_eq!(struct_field(scalar_arg(&cse2.expressions[0], 0)), 3);
    assert_eq!(emit_mapping(cse2.common.as_ref()), vec![0, 1, 2, 3, 4]);

    let cse1 = as_project(cse2.input.as_ref().unwrap());
    assert_eq!(struct_field(&cse1.expressions[0]), 0);
    assert_eq!(emit_mapping(cse1.common.as_ref()), vec![0, 1, 2, 3]);

    let first = as_project(cse1.input.as_ref().unwrap());
    assert_eq!(struct_field(&first.expressions[0]), 1);
    assert_eq!(emit_mapping(first.common.as_ref()), vec![0, 1, 2]);
}

/// Unwraps a Substrait project relation.
fn as_project(rel: &substrait::proto::Rel) -> &substrait::proto::ProjectRel {
    match rel.rel_type.as_ref().unwrap() {
        rel::RelType::Project(project) => project,
        other => panic!("expected project rel, got {other:?}"),
    }
}

/// Reads the zero-based field index from a Substrait struct-field selection.
fn struct_field(expr: &substrait::proto::Expression) -> i32 {
    let expression::RexType::Selection(selection) = expr.rex_type.as_ref().unwrap() else {
        panic!("expected field selection");
    };
    let expression::field_reference::ReferenceType::DirectReference(segment) =
        selection.reference_type.as_ref().unwrap()
    else {
        panic!("expected direct field reference");
    };
    let expression::reference_segment::ReferenceType::StructField(field) =
        segment.reference_type.as_ref().unwrap()
    else {
        panic!("expected struct field");
    };
    field.field
}

/// Only `PROJECT_NODE` materializes its common slots. A `SELECT_NODE`, hash join or nested-loop
/// join carrying the same field is refused up front with a clear reason, instead of failing later
/// with an opaque descriptor error when a conjunct references one of the shared sub-expressions.
#[test]
fn common_slots_outside_a_project_are_rejected() {
    let common = || {
        let mut map = BTreeMap::new();
        map.insert(5, slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)));
        Some(map)
    };

    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(common()));
    let select_plan = TPlan::new(vec![select, scan_node(0, 0)]);

    let mut hash_join = hash_join_node(TJoinOp::INNER_JOIN);
    hash_join.hash_join_node.as_mut().unwrap().common_slot_map = common();
    let hash_plan = TPlan::new(vec![hash_join, scan_node(0, 0), scan_node(1, 1)]);

    let mut nestloop = nestloop_join_node(
        TJoinOp::INNER_JOIN,
        vec![binary_pred(
            TExprOpcode::LT,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT)),
        )],
    );
    nestloop
        .nestloop_join_node
        .as_mut()
        .unwrap()
        .common_slot_map = common();
    let nestloop_plan = TPlan::new(vec![nestloop, scan_node(0, 0), scan_node(1, 1)]);

    for (label, plan, desc) in [
        ("SELECT_NODE", select_plan, base_desc()),
        ("HASH_JOIN_NODE", hash_plan, join_desc()),
        ("NESTLOOP_JOIN_NODE", nestloop_plan, join_desc()),
    ] {
        let err = translate_fragment(&params(Some(plan), Some(desc), None)).unwrap_err();
        let TranslateError::UnsupportedPlanNode { reason, .. } = err else {
            panic!("{label}: expected an unsupported plan node, got {err:?}");
        };
        assert_eq!(reason, "common slots are only materialized on PROJECT_NODE");
    }
}

/// Verifies fragment output expressions add the final root projection.
#[test]
fn fragment_output_exprs_add_root_projection() {
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(base_desc()),
        Some(vec![
            slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)),
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
        ]),
    ))
    .unwrap();

    assert_eq!(translated.output_names, vec!["name", "id"]);
    match root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    {
        rel::RelType::Project(project) => assert_eq!(project.expressions.len(), 2),
        other => panic!("expected root project rel, got {other:?}"),
    }
}

/// Verifies unsupported plan nodes return a structured unsupported-plan-node error.
#[test]
fn unsupported_merge_join_is_structured_error() {
    let join = base_plan_node(9, TPlanNodeType::MERGE_JOIN_NODE, 0, vec![0]);
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![join])),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        TranslateError::UnsupportedPlanNode {
            node_id: 9,
            node_type,
            ..
        } if node_type == TPlanNodeType::MERGE_JOIN_NODE
    ));
}

/// Verifies unsupported expression nodes return a structured unsupported-expression error.
#[test]
fn unsupported_expression_is_structured_error() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![TExpr::new(vec![base_expr_node(
        TExprNodeType::LAMBDA_FUNCTION_EXPR,
        scalar_type(TPrimitiveType::BOOLEAN),
        0,
    )])]);

    let err = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        TranslateError::UnsupportedExpression {
            node_type,
            ..
        } if node_type == TExprNodeType::LAMBDA_FUNCTION_EXPR
    ));
}

/// Verifies unsupported complex slot types fail during descriptor normalization.
#[test]
fn unsupported_complex_type_is_structured_error() {
    let desc = desc_table(
        vec![(0, Some(100))],
        vec![slot(1, 0, "items", complex_type(TTypeNodeType::ARRAY))],
    );
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        TranslateError::UnsupportedType {
            node_type: Some(node_type),
            ..
        } if node_type == TTypeNodeType::ARRAY
    ));
}

/// Verifies unsupported types on non-materialized slots do not block visible output.
#[test]
fn non_materialized_unsupported_slot_type_is_ignored() {
    let mut hidden_slot = slot(2, 0, "hidden", complex_type(TTypeNodeType::ARRAY));
    hidden_slot.is_materialized = Some(false);

    let desc = desc_table(
        vec![(0, Some(100))],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            hidden_slot,
        ],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    assert_eq!(translated.output_names, vec!["id"]);
}

/// Verifies unsupported LARGEINT slots return a structured type error.
#[test]
fn unsupported_largeint_is_structured_error() {
    let desc = desc_table(
        vec![(0, Some(100))],
        vec![slot(1, 0, "big", scalar_type(TPrimitiveType::LARGEINT))],
    );
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        TranslateError::UnsupportedType {
            primitive: Some(primitive),
            ..
        } if primitive == TPrimitiveType::LARGEINT
    ));
}

/// Verifies DECIMAL256 slots stay unsupported until wider decimal handling is added.
#[test]
fn unsupported_decimal256_is_structured_error() {
    let desc = desc_table(
        vec![(0, Some(100))],
        vec![slot(
            1,
            0,
            "huge_decimal",
            scalar_type_with(TPrimitiveType::DECIMAL256, None, Some(76), Some(0)),
        )],
    );
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        TranslateError::UnsupportedType {
            primitive: Some(primitive),
            ..
        } if primitive == TPrimitiveType::DECIMAL256
    ));
}

/// Verifies project-node conjuncts are rejected until that translation path is supported.
#[test]
fn project_node_conjuncts_are_unsupported() {
    let mut slot_map = BTreeMap::new();
    slot_map.insert(3, slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)));

    let mut project = base_plan_node(1, TPlanNodeType::PROJECT_NODE, 1, vec![1]);
    project.project_node = Some(TProjectNode::new(Some(slot_map), None));
    project.conjuncts = Some(vec![binary_pred(
        TExprOpcode::GT,
        slot_ref(3, 1, scalar_type(TPrimitiveType::BIGINT)),
        int_literal(10),
    )]);

    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(3, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );

    let err = translate_fragment(&params(
        Some(TPlan::new(vec![project, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        TranslateError::UnsupportedPlanNode {
            node_id: 1,
            node_type,
            ..
        } if node_type == TPlanNodeType::PROJECT_NODE
    ));
}

/// Verifies same function names in different extension URNs get distinct anchors.
#[test]
fn extension_registry_keys_functions_by_urn_and_name() {
    let mut registry = ExtensionRegistry::new();
    let boolean_anchor = registry.register_function(URN_BOOLEAN, "overlap");
    let comparison_anchor = registry.register_function(URN_COMPARISON, "overlap");
    let reused_boolean_anchor = registry.register_function(URN_BOOLEAN, "overlap");

    assert_ne!(boolean_anchor, comparison_anchor);
    assert_eq!(boolean_anchor, reused_boolean_anchor);
}

/// Verifies missing descriptor tables fail before plan reconstruction.
#[test]
fn missing_descriptor_table_is_error() {
    let err = translate_fragment(&params(Some(TPlan::new(vec![scan_node(0, 0)])), None, None))
        .unwrap_err();
    assert!(matches!(
        err,
        TranslateError::MissingField {
            context: "TExecPlanFragmentParams",
            field: "desc_tbl"
        }
    ));
}

/// Verifies missing fragment plans fail with a required-field error.
#[test]
fn missing_fragment_plan_is_error() {
    let err = translate_fragment(&params(None, Some(base_desc()), None)).unwrap_err();
    assert!(matches!(
        err,
        TranslateError::MissingField {
            context: "TPlanFragment",
            field: "plan"
        }
    ));
}

/// Verifies missing fragments fail with a top-level required-field error.
#[test]
fn missing_fragment_is_error() {
    let err = translate_fragment(&params_without_fragment(Some(base_desc()))).unwrap_err();
    assert!(matches!(
        err,
        TranslateError::MissingField {
            context: "TExecPlanFragmentParams",
            field: "fragment"
        }
    ));
}

/// Verifies flat preorder child-count mismatches are reported as malformed plans.
#[test]
fn malformed_child_counts_are_errors() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 2, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(err, TranslateError::MalformedPlan(_)));
}

/// Verifies string and boolean literals can participate in filter conjuncts.
#[test]
fn bool_and_string_literals_translate() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![
        binary_pred(
            TExprOpcode::EQ,
            string_literal("alice"),
            string_literal("alice"),
        ),
        bool_literal(true),
    ]);

    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap();
    match root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    {
        rel::RelType::Filter(filter) => assert!(filter.condition.is_some()),
        other => panic!("expected filter rel, got {other:?}"),
    }
}

// --- Helpers and coverage for the expression/scan/type surface ---------------

/// Builds a compound predicate over already-built children in preorder.
fn compound_pred(opcode: TExprOpcode, children: Vec<TExpr>) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::COMPOUND_PRED,
        scalar_type(TPrimitiveType::BOOLEAN),
        children.len() as i32,
    );
    node.opcode = Some(opcode);
    let mut nodes = vec![node];
    for child in children {
        nodes.extend(child.nodes);
    }
    TExpr::new(nodes)
}

/// Builds an `IS [NOT] NULL` predicate over a single child.
fn is_null_pred(is_not_null: bool, child: TExpr) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::IS_NULL_PRED,
        scalar_type(TPrimitiveType::BOOLEAN),
        1,
    );
    node.is_null_pred = Some(TIsNullPredicate::new(is_not_null));
    let mut nodes = vec![node];
    nodes.extend(child.nodes);
    TExpr::new(nodes)
}

/// Builds a cast of `child` to `target` in preorder.
fn cast_expr(target: TTypeDesc, child: TExpr) -> TExpr {
    let node = base_expr_node(TExprNodeType::CAST_EXPR, target, 1);
    let mut nodes = vec![node];
    nodes.extend(child.nodes);
    TExpr::new(nodes)
}

/// Builds a typed NULL literal expression.
fn null_literal(primitive: TPrimitiveType) -> TExpr {
    TExpr::new(vec![base_expr_node(
        TExprNodeType::NULL_LITERAL,
        scalar_type(primitive),
        0,
    )])
}

/// Builds a DECIMAL literal carrying its source string and precision/scale.
fn decimal_literal(value: &str, precision: i32, scale: i32) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::DECIMAL_LITERAL,
        scalar_type_with(
            TPrimitiveType::DECIMAL128,
            None,
            Some(precision),
            Some(scale),
        ),
        0,
    );
    node.decimal_literal = Some(TDecimalLiteral::new(value.to_string(), None));
    TExpr::new(vec![node])
}

/// Builds an HDFS scan node that resolves its tuple from `row_tuples`.
fn hdfs_scan_node(node_id: i32, tuple_id: i32) -> TPlanNode {
    base_plan_node(node_id, TPlanNodeType::HDFS_SCAN_NODE, 0, vec![tuple_id])
}

/// Builds a single-column descriptor whose table carries `db` (empty for fallback tests).
fn desc_with_db(db: &str) -> TDescriptorTable {
    TDescriptorTable::new(
        Some(vec![slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT))]),
        vec![TTupleDescriptor::new(Some(0), None, None, Some(7), None)],
        Some(vec![table_descriptor(7, db, "t", 1)]),
        None,
    )
}

/// Extracts the scalar function from a Substrait expression.
fn scalar_fn(expr: &substrait::proto::Expression) -> &expression::ScalarFunction {
    match expr.rex_type.as_ref().unwrap() {
        expression::RexType::ScalarFunction(scalar) => scalar,
        other => panic!("expected scalar function, got {other:?}"),
    }
}

/// Resolves a function anchor back to its `(extension urn, function name)`.
fn resolved_function(plan: &substrait::proto::Plan, anchor: u32) -> (String, String) {
    use substrait::proto::extensions::simple_extension_declaration::MappingType;
    let func = plan
        .extensions
        .iter()
        .find_map(|decl| match decl.mapping_type.as_ref().unwrap() {
            MappingType::ExtensionFunction(func) if func.function_anchor == anchor => Some(func),
            _ => None,
        })
        .expect("function anchor is declared");
    let urn = plan
        .extension_urns
        .iter()
        .find(|urn| urn.extension_urn_anchor == func.extension_urn_reference)
        .expect("urn anchor is declared")
        .urn
        .clone();
    (urn, func.name.clone())
}

/// Extracts the named-table path emitted by a scan-only plan.
fn read_named_table_names(plan: &substrait::proto::Plan) -> Vec<String> {
    match root(plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    {
        rel::RelType::Read(read) => match read.read_type.as_ref().unwrap() {
            read_rel::ReadType::NamedTable(table) => table.names.clone(),
            other => panic!("expected named table, got {other:?}"),
        },
        other => panic!("expected read rel, got {other:?}"),
    }
}

/// Translates a filter whose single conjunct is `conjunct`.
fn filter_with_conjunct(conjunct: TExpr) -> substrait::proto::Plan {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![conjunct]);
    translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap()
    .plan
}

/// Verifies every comparison opcode maps to its Substrait function name under the comparison URN.
#[test]
fn binary_predicate_opcodes_map_to_comparison_names() {
    for (opcode, expected) in [
        (TExprOpcode::EQ, "equal"),
        (TExprOpcode::NE, "not_equal"),
        (TExprOpcode::LT, "lt"),
        (TExprOpcode::LE, "lte"),
        (TExprOpcode::GT, "gt"),
        (TExprOpcode::GE, "gte"),
    ] {
        let plan = filter_with_conjunct(binary_pred(
            opcode,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            int_literal(10),
        ));
        let scalar = scalar_fn(filter_condition(&plan));
        let (urn, name) = resolved_function(&plan, scalar.function_reference);
        assert_eq!(name, expected, "opcode {opcode:?}");
        assert_eq!(urn, URN_COMPARISON);
    }
}

/// Verifies compound AND/OR/NOT map to boolean functions with the right arity.
#[test]
fn compound_predicates_map_to_boolean_functions() {
    for (opcode, expected, child_count) in [
        (TExprOpcode::COMPOUND_AND, "and", 2usize),
        (TExprOpcode::COMPOUND_OR, "or", 2),
        (TExprOpcode::COMPOUND_NOT, "not", 1),
    ] {
        let children = (0..child_count)
            .map(|_| {
                binary_pred(
                    TExprOpcode::GT,
                    slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
                    int_literal(10),
                )
            })
            .collect();
        let plan = filter_with_conjunct(compound_pred(opcode, children));
        let scalar = scalar_fn(filter_condition(&plan));
        assert_eq!(scalar.arguments.len(), child_count);
        let (urn, name) = resolved_function(&plan, scalar.function_reference);
        assert_eq!(name, expected);
        assert_eq!(urn, URN_BOOLEAN);
    }
}

/// Verifies a compound AND/OR with fewer than two children is rejected.
#[test]
fn compound_and_with_single_child_is_error() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![compound_pred(
        TExprOpcode::COMPOUND_AND,
        vec![bool_literal(true)],
    )]);
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(err, TranslateError::MalformedPlan(_)));
}

/// Verifies multiple filter conjuncts fold into a single boolean `and`.
#[test]
fn multiple_conjuncts_fold_into_boolean_and() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![
        binary_pred(
            TExprOpcode::GT,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            int_literal(10),
        ),
        binary_pred(
            TExprOpcode::LT,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            int_literal(20),
        ),
    ]);
    let plan = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap()
    .plan;
    let scalar = scalar_fn(filter_condition(&plan));
    assert_eq!(scalar.arguments.len(), 2);
    let (urn, name) = resolved_function(&plan, scalar.function_reference);
    assert_eq!(name, "and");
    assert_eq!(urn, URN_BOOLEAN);
}

/// Verifies both IS NULL branches map to the right comparison function.
#[test]
fn is_null_predicates_map_to_comparison_functions() {
    for (is_not_null, expected) in [(false, "is_null"), (true, "is_not_null")] {
        let plan = filter_with_conjunct(is_null_pred(
            is_not_null,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
        ));
        let scalar = scalar_fn(filter_condition(&plan));
        assert_eq!(scalar.arguments.len(), 1);
        let (urn, name) = resolved_function(&plan, scalar.function_reference);
        assert_eq!(name, expected);
        assert_eq!(urn, URN_COMPARISON);
    }
}

/// Verifies a cast emits a throwing Substrait cast to the declared target type.
#[test]
fn cast_expr_emits_throwing_cast_to_target_type() {
    let plan = filter_with_conjunct(binary_pred(
        TExprOpcode::EQ,
        cast_expr(
            scalar_type(TPrimitiveType::INT),
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
        ),
        int_literal_typed(10, TPrimitiveType::INT),
    ));
    match scalar_arg(filter_condition(&plan), 0)
        .rex_type
        .as_ref()
        .unwrap()
    {
        expression::RexType::Cast(cast) => {
            assert_eq!(
                cast.failure_behavior,
                expression::cast::FailureBehavior::ThrowException as i32
            );
            assert!(matches!(
                cast.r#type.as_ref().unwrap().kind.as_ref().unwrap(),
                substrait::proto::r#type::Kind::I32(_)
            ));
        }
        other => panic!("expected cast, got {other:?}"),
    }
}

/// Verifies a cast preserves a declared non-nullable target type.
#[test]
fn cast_expr_preserves_declared_non_nullable_target() {
    let mut cast = base_expr_node(
        TExprNodeType::CAST_EXPR,
        scalar_type(TPrimitiveType::INT),
        1,
    );
    cast.is_nullable = Some(false);
    let mut nodes = vec![cast];
    nodes.extend(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)).nodes);
    let plan = filter_with_conjunct(binary_pred(
        TExprOpcode::EQ,
        TExpr::new(nodes),
        int_literal_typed(10, TPrimitiveType::INT),
    ));
    match scalar_arg(filter_condition(&plan), 0)
        .rex_type
        .as_ref()
        .unwrap()
    {
        expression::RexType::Cast(cast) => {
            match cast.r#type.as_ref().unwrap().kind.as_ref().unwrap() {
                substrait::proto::r#type::Kind::I32(i32_type) => assert_eq!(
                    i32_type.nullability,
                    substrait::proto::r#type::Nullability::Required as i32
                ),
                other => panic!("expected i32 cast type, got {other:?}"),
            }
        }
        other => panic!("expected cast, got {other:?}"),
    }
}

/// Verifies a NULL literal becomes a typed Substrait null.
#[test]
fn null_literal_translates_to_typed_null() {
    let plan = filter_with_conjunct(binary_pred(
        TExprOpcode::EQ,
        slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
        null_literal(TPrimitiveType::BIGINT),
    ));
    match literal_type(scalar_arg(filter_condition(&plan), 1)) {
        expression::literal::LiteralType::Null(ty) => assert!(matches!(
            ty.kind.as_ref().unwrap(),
            substrait::proto::r#type::Kind::I64(_)
        )),
        other => panic!("expected typed null literal, got {other:?}"),
    }
}

/// Verifies decimal literals encode the little-endian unscaled integer with precision/scale.
#[test]
fn decimal_literal_encodes_little_endian_unscaled_value() {
    for (text, precision, scale, expected) in [
        ("1.50", 10, 2, 150i128),
        ("-1.50", 10, 2, -150i128),
        ("1.5", 10, 4, 15000i128),
        ("0", 10, 0, 0i128),
    ] {
        let plan = filter_with_conjunct(binary_pred(
            TExprOpcode::EQ,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            decimal_literal(text, precision, scale),
        ));
        match literal_type(scalar_arg(filter_condition(&plan), 1)) {
            expression::literal::LiteralType::Decimal(decimal) => {
                assert_eq!(
                    decimal.value,
                    expected.to_le_bytes().to_vec(),
                    "value for {text}"
                );
                assert_eq!(decimal.precision, precision);
                assert_eq!(decimal.scale, scale);
            }
            other => panic!("expected decimal literal, got {other:?}"),
        }
    }
}

/// A decimal literal wider than 18 digits follows the slot rule and is emitted as FP64.
#[test]
fn wide_decimal_literal_is_lowered_to_fp64() {
    let plan = filter_with_conjunct(binary_pred(
        TExprOpcode::EQ,
        slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
        decimal_literal("1.5", 19, 2),
    ));
    match literal_type(scalar_arg(filter_condition(&plan), 1)) {
        expression::literal::LiteralType::Fp64(value) => assert_eq!(*value, 1.5),
        other => panic!("expected fp64 literal, got {other:?}"),
    }
}

/// Verifies an integer literal that overflows its declared width is a malformed plan.
#[test]
fn integer_literal_overflowing_declared_width_is_error() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![binary_pred(
        TExprOpcode::EQ,
        slot_ref(1, 0, scalar_type(TPrimitiveType::INT)),
        int_literal_typed(i64::from(i32::MAX) + 1, TPrimitiveType::INT),
    )]);
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(err, TranslateError::MalformedPlan(_)));
}

/// Verifies narrow integer literals keep their Substrait width.
#[test]
fn integer_literals_match_narrow_widths() {
    for primitive in [TPrimitiveType::TINYINT, TPrimitiveType::SMALLINT] {
        let plan = filter_with_conjunct(binary_pred(
            TExprOpcode::EQ,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            int_literal_typed(7, primitive),
        ));
        let lit = literal_type(scalar_arg(filter_condition(&plan), 1));
        match primitive {
            TPrimitiveType::TINYINT => {
                assert!(matches!(lit, expression::literal::LiteralType::I8(7)))
            }
            TPrimitiveType::SMALLINT => {
                assert!(matches!(lit, expression::literal::LiteralType::I16(7)))
            }
            _ => unreachable!(),
        }
    }
}

/// Verifies an HDFS scan node produces the same named-table read as a file scan.
#[test]
fn hdfs_scan_produces_named_table() {
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![hdfs_scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap();
    assert_eq!(translated.output_names, vec!["id", "name"]);
    assert_eq!(
        read_named_table_names(&translated.plan),
        vec!["tpch", "users"]
    );
}

/// Verifies a project emits its expressions starting after the input columns.
#[test]
fn project_emit_mapping_starts_after_input_columns() {
    let mut slot_map = BTreeMap::new();
    slot_map.insert(3, slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)));
    slot_map.insert(4, slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)));

    let mut project = base_plan_node(1, TPlanNodeType::PROJECT_NODE, 1, vec![1]);
    project.project_node = Some(TProjectNode::new(Some(slot_map), None));

    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(3, 1, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(4, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );

    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![project, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    match root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    {
        rel::RelType::Project(project) => {
            let emit = match project.common.as_ref().unwrap().emit_kind.as_ref().unwrap() {
                substrait::proto::rel_common::EmitKind::Emit(emit) => emit,
                other => panic!("expected emit, got {other:?}"),
            };
            // The scan child emits two columns, so the two projected expressions
            // map to indices [2, 3], not [0, 1].
            assert_eq!(emit.output_mapping, vec![2, 3]);
        }
        other => panic!("expected project rel, got {other:?}"),
    }
}

/// Verifies a scan over a tuple with no backing table uses a synthetic name.
#[test]
fn scan_table_name_falls_back_when_table_missing() {
    let desc = desc_table(
        vec![(0, None)],
        vec![slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT))],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();
    assert_eq!(read_named_table_names(&translated.plan), vec!["tuple_0"]);
}

/// Verifies a qualified database produces a two-part named-table path.
#[test]
fn scan_table_name_keeps_qualified_database() {
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(desc_with_db("db")),
        None,
    ))
    .unwrap();
    assert_eq!(read_named_table_names(&translated.plan), vec!["db", "t"]);
}

/// Verifies an empty database name is dropped from the named-table path.
#[test]
fn scan_table_name_omits_empty_database() {
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(desc_with_db("")),
        None,
    ))
    .unwrap();
    assert_eq!(read_named_table_names(&translated.plan), vec!["t"]);
}

/// Verifies trailing expression nodes left by the cursor are rejected.
#[test]
fn expression_with_trailing_nodes_is_error() {
    let mut nodes = slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)).nodes;
    nodes.push(base_expr_node(
        TExprNodeType::INT_LITERAL,
        scalar_type(TPrimitiveType::BIGINT),
        0,
    ));
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![TExpr::new(nodes)]);
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(err, TranslateError::MalformedPlan(_)));
}

/// Verifies a node claiming a child with none following is rejected (cursor under-run).
#[test]
fn expression_missing_child_node_is_error() {
    let mut node = base_expr_node(
        TExprNodeType::COMPOUND_PRED,
        scalar_type(TPrimitiveType::BOOLEAN),
        1,
    );
    node.opcode = Some(TExprOpcode::COMPOUND_NOT);
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![TExpr::new(vec![node])]);
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![select, scan_node(0, 0)])),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(err, TranslateError::MalformedPlan(_)));
}

// ---------------------------------------------------------------------------
// Aggregation, sort, join, and expression coverage for the TPC-H slice.
// ---------------------------------------------------------------------------

/// Builds a builtin StarRocks function payload with the given name and return type.
fn builtin_function(name: &str, ret_type: TTypeDesc) -> TFunction {
    TFunction::new(
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
    )
}

/// Builds an aggregate-function expression (`fn(child)`) in flat preorder form.
fn aggregate_expr(name: &str, ret_type: TTypeDesc, child: Option<TExpr>) -> TExpr {
    let num_children = child.as_ref().map(|_| 1).unwrap_or(0);
    let mut node = base_expr_node(TExprNodeType::AGG_EXPR, ret_type.clone(), num_children);
    node.agg_expr = Some(TAggregateExpr::new(false));
    node.fn_ = Some(builtin_function(name, ret_type));
    let mut nodes = vec![node];
    if let Some(child) = child {
        nodes.extend(child.nodes);
    }
    TExpr::new(nodes)
}

/// Builds a one-phase aggregation node over `output_tuple` with the given keys and aggregates.
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

/// Descriptor with a scan tuple 0 (`id` BIGINT, `name` VARCHAR) and an aggregation output
/// tuple 1 (`name` key, `total` BIGINT).
fn agg_desc() -> TDescriptorTable {
    desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(2, 1, "total", scalar_type(TPrimitiveType::BIGINT)),
        ],
    )
}

/// Verifies one-phase group-by aggregation becomes an `AggregateRel` with the grouping key and
/// a `sum` measure, and that the output row layout switches to the aggregation output tuple.
#[test]
fn aggregation_translates_to_aggregate_rel() {
    let agg = aggregation_node(
        1,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate_expr(
            "sum",
            scalar_type(TPrimitiveType::BIGINT),
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(agg_desc()),
        None,
    ))
    .unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["name", "total"]);
    let rel::RelType::Aggregate(aggregate) =
        root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate relation");
    };
    assert_eq!(aggregate.grouping_expressions.len(), 1);
    assert_eq!(aggregate.groupings.len(), 1);
    assert_eq!(aggregate.groupings[0].expression_references, vec![0]);
    assert_eq!(aggregate.measures.len(), 1);
    let measure = aggregate.measures[0].measure.as_ref().unwrap();
    assert_eq!(measure.arguments.len(), 1);
    assert_eq!(
        measure.invocation,
        substrait::proto::aggregate_function::AggregationInvocation::All as i32
    );
    let names: Vec<_> = extension_function_names(&translated.plan);
    assert!(names.contains(&"sum".to_string()), "{names:?}");
}

/// Verifies a distinct aggregate (StarRocks `multi_distinct_count`) becomes a distinct `count`.
#[test]
fn multi_distinct_count_translates_to_distinct_count() {
    let agg = aggregation_node(
        1,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate_expr(
            "multi_distinct_count",
            scalar_type(TPrimitiveType::BIGINT),
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(agg_desc()),
        None,
    ))
    .unwrap();

    let root = root(&translated.plan);
    let rel::RelType::Aggregate(aggregate) =
        root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate relation");
    };
    let measure = aggregate.measures[0].measure.as_ref().unwrap();
    assert_eq!(
        measure.invocation,
        substrait::proto::aggregate_function::AggregationInvocation::Distinct as i32
    );
    let names = extension_function_names(&translated.plan);
    assert!(names.contains(&"count".to_string()), "{names:?}");
}

/// Descriptor for the phase-classification tests: scan tuple 0 and an aggregation output
/// tuple 1 with a single ungrouped aggregate slot.
fn scalar_agg_desc() -> TDescriptorTable {
    desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "total", scalar_type(TPrimitiveType::BIGINT)),
        ],
    )
}

/// A scalar `sum(id)` aggregation node with per-measure merge flags and the node's
/// `need_finalize`, for driving the phase classifier from tests.
fn phase_aggregation_node(merge_flags: &[bool], need_finalize: bool) -> TPlanNode {
    let aggregates = merge_flags
        .iter()
        .map(|&is_merge| {
            let mut aggregate = aggregate_expr(
                "sum",
                scalar_type(TPrimitiveType::BIGINT),
                Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
            );
            aggregate.nodes[0].agg_expr = Some(TAggregateExpr::new(is_merge));
            aggregate
        })
        .collect();
    let mut node = aggregation_node(1, 1, Vec::new(), aggregates);
    node.agg_node.as_mut().unwrap().need_finalize = need_finalize;
    node
}

fn translate_phase_case(node: TPlanNode) -> Result<TranslatedPlan, TranslateError> {
    translate_fragment(&params(
        Some(TPlan::new(vec![node, scan_node(0, 0)])),
        Some(scalar_agg_desc()),
        None,
    ))
}

/// Verifies a merge aggregation whose child is not an exchange is rejected: its input columns
/// would carry FE-declared types the wire-type override never corrected.
#[test]
fn merge_over_a_scan_is_rejected() {
    let err = translate_phase_case(phase_aggregation_node(&[true], true)).unwrap_err();
    assert!(matches!(err, TranslateError::UnsupportedPlanNode { .. }));
    let message = err.to_string();
    assert!(message.contains("exchange"), "{message}");
    assert!(message.contains("new_planner_agg_stage"), "{message}");
}

/// Builds the merge fragment of a two-phase `sum(decimal), count(*)` query: a merge
/// aggregation (node 8, output tuple 3) over an exchange (node 7) carrying the intermediate
/// tuple 2, whose FE-declared slot types are the ones the wire-type model must override.
fn merge_fragment_params() -> TExecPlanFragmentParams {
    let exchange = exchange_node_with(7, vec![2], None, None);
    let mut sum = aggregate_expr(
        "sum",
        scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(38), Some(2)),
        Some(slot_ref(
            10,
            2,
            scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(38), Some(2)),
        )),
    );
    sum.nodes[0].agg_expr = Some(TAggregateExpr::new(true));
    let mut count = aggregate_expr(
        "count",
        scalar_type(TPrimitiveType::BIGINT),
        Some(slot_ref(11, 2, scalar_type(TPrimitiveType::BIGINT))),
    );
    count.nodes[0].agg_expr = Some(TAggregateExpr::new(true));
    let aggregate = aggregation_node(8, 3, Vec::new(), vec![sum, count]);

    let desc = desc_table(
        vec![(2, None), (3, None)],
        vec![
            // The FE declares the intermediate sum slot as DECIMAL128, the lie the override
            // corrects; the wire column is the partial fragment's FP64 output.
            slot(
                10,
                2,
                "s",
                scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(38), Some(2)),
            ),
            slot(11, 2, "c", scalar_type(TPrimitiveType::BIGINT)),
            slot(
                12,
                3,
                "revenue",
                scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(38), Some(2)),
            ),
            slot(13, 3, "cnt", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    params(
        Some(TPlan::new(vec![aggregate, exchange])),
        Some(desc),
        None,
    )
}

/// Verifies the merge fragment of a two-phase aggregation translates to a plain aggregate
/// with the substituted merge functions (count merges as SUM of partial counts) over an
/// exchange whose declared stream types are the modeled wire types, not the FE's
/// DECIMAL128 intermediate slot type, and that the fragment leaves through the finalizing
/// projection that casts every measure to its FE-declared output-slot type. The engine binds
/// the merged count (a sum over BIGINT) as HUGEINT, so without the cast this fragment's output
/// feeding a further downstream fragment would be refused by the hop's schema guard.
#[test]
fn merge_aggregation_translates_with_substituted_functions() {
    let translated = PlanTranslator::new()
        .translate_fragment_with_exchange_inputs(
            &merge_fragment_params(),
            &[stream_input(7, &["s", "c"])],
        )
        .unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["revenue", "cnt"]);
    let rel::RelType::Project(project) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected the finalizing project over the merge aggregate");
    };
    // One throwing cast per measure, to the FE's output-slot types as the type mapper declares
    // them: the DECIMAL128(38,2) sum slot lowers to FP64 (precision > 18), the same lowering a
    // downstream exchange applies when it derives its stream schema from this very slot, and
    // the count leaves as the BIGINT its slot declares (the engine binds it as HUGEINT).
    assert_eq!(project.expressions.len(), 2);
    let (sum_type, sum_input) = cast_parts(&project.expressions[0]);
    assert!(
        matches!(sum_type, substrait::proto::r#type::Kind::Fp64(_)),
        "{sum_type:?}"
    );
    assert_eq!(field_index(sum_input), 0);
    let (count_type, count_input) = cast_parts(&project.expressions[1]);
    assert!(
        matches!(count_type, substrait::proto::r#type::Kind::I64(_)),
        "{count_type:?}"
    );
    assert_eq!(field_index(count_input), 1);

    let aggregate = root_aggregate(&translated.plan);
    assert_eq!(aggregate.measures.len(), 2);
    for measure in &aggregate.measures {
        assert_eq!(
            measure.measure.as_ref().unwrap().phase,
            substrait::proto::AggregationPhase::IntermediateToResult as i32
        );
    }
    // count merged as count would count rows instead of summing partial counts; the only
    // registered aggregate must be sum.
    let names = extension_function_names(&translated.plan);
    assert!(names.contains(&"sum".to_string()), "{names:?}");
    assert!(!names.contains(&"count".to_string()), "{names:?}");

    // The engine declaration derives from the overridden types: DOUBLE, not DECIMAL(38,2).
    assert_eq!(translated.stream_inputs.len(), 1);
    assert_eq!(
        translated.stream_inputs[0]
            .columns
            .iter()
            .map(|column| (column.name.as_str(), column.ty.as_str()))
            .collect::<Vec<_>>(),
        vec![("s", "DOUBLE"), ("c", "BIGINT")]
    );
}

/// Verifies both fragments of one two-phase query derive the same wire types: the partial
/// fragment's measure output types match the merge fragment's declared stream columns
/// column-for-column (FP64 <-> DOUBLE, I64 <-> BIGINT).
#[test]
fn two_phase_wire_types_agree_end_to_end() {
    // Partial fragment: sum(decimal) + count(*) over a scan, need_finalize = false.
    let mut sum = aggregate_expr(
        "sum",
        scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(38), Some(2)),
        Some(slot_ref(
            1,
            0,
            scalar_type_with(TPrimitiveType::DECIMAL64, None, Some(15), Some(2)),
        )),
    );
    sum.nodes[0].agg_expr = Some(TAggregateExpr::new(false));
    let mut count = aggregate_expr("count", scalar_type(TPrimitiveType::BIGINT), None);
    count.nodes[0].agg_expr = Some(TAggregateExpr::new(false));
    let mut node = aggregation_node(1, 1, Vec::new(), vec![sum, count]);
    node.agg_node.as_mut().unwrap().need_finalize = false;
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(
                1,
                0,
                "price",
                scalar_type_with(TPrimitiveType::DECIMAL64, None, Some(15), Some(2)),
            ),
            slot(
                20,
                1,
                "s",
                scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(38), Some(2)),
            ),
            slot(21, 1, "c", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    let partial = translate_fragment(&params(
        Some(TPlan::new(vec![node, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    let root = root(&partial.plan);
    let rel::RelType::Aggregate(aggregate) =
        root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate relation");
    };
    let partial_kinds: Vec<_> = aggregate
        .measures
        .iter()
        .map(|measure| {
            match measure
                .measure
                .as_ref()
                .unwrap()
                .output_type
                .as_ref()
                .unwrap()
                .kind
                .as_ref()
                .unwrap()
            {
                substrait::proto::r#type::Kind::Fp64(_) => "DOUBLE",
                substrait::proto::r#type::Kind::I64(_) => "BIGINT",
                other => panic!("unexpected partial state kind {other:?}"),
            }
        })
        .collect();

    // Merge fragment of the same query (identical FE-serialized functions).
    let merge = PlanTranslator::new()
        .translate_fragment_with_exchange_inputs(
            &merge_fragment_params(),
            &[stream_input(7, &["s", "c"])],
        )
        .unwrap();
    let merge_types: Vec<_> = merge.stream_inputs[0]
        .columns
        .iter()
        .map(|column| column.ty.as_str())
        .collect();

    assert_eq!(partial_kinds, merge_types);
}

/// Verifies a partial-phase ("update serialize") aggregation translates to a plain aggregate
/// whose measure carries the InitialToIntermediate phase and the modeled partial-state type
/// (I64 for an integer sum), not the FE's declared slot type.
#[test]
fn partial_aggregation_translates_with_the_modeled_state_type() {
    let translated = translate_phase_case(phase_aggregation_node(&[false], false)).unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["total"]);
    let rel::RelType::Aggregate(aggregate) =
        root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate relation");
    };
    assert_eq!(aggregate.measures.len(), 1);
    let measure = aggregate.measures[0].measure.as_ref().unwrap();
    assert_eq!(
        measure.phase,
        substrait::proto::AggregationPhase::InitialToIntermediate as i32
    );
    let output_type = measure.output_type.as_ref().unwrap();
    assert!(
        matches!(
            output_type.kind.as_ref().unwrap(),
            substrait::proto::r#type::Kind::I64(_)
        ),
        "{output_type:?}"
    );
    let names: Vec<_> = extension_function_names(&translated.plan);
    assert!(names.contains(&"sum".to_string()), "{names:?}");
}

/// Names a measure's declared output type the way the engine declares a stream column.
fn measure_output_type(measure: &substrait::proto::aggregate_rel::Measure) -> &'static str {
    match measure
        .measure
        .as_ref()
        .unwrap()
        .output_type
        .as_ref()
        .unwrap()
        .kind
        .as_ref()
        .unwrap()
    {
        substrait::proto::r#type::Kind::Fp64(_) => "DOUBLE",
        substrait::proto::r#type::Kind::I64(_) => "BIGINT",
        substrait::proto::r#type::Kind::Varchar(_) => "VARCHAR",
        other => panic!("unexpected measure output type {other:?}"),
    }
}

/// Extracts the aggregate relation a plan roots at, through an optional finalizing project.
fn root_aggregate(plan: &substrait::proto::Plan) -> &substrait::proto::AggregateRel {
    let mut input = root(plan).input.as_ref().unwrap();
    if let rel::RelType::Project(project) = input.rel_type.as_ref().unwrap() {
        input = project.input.as_ref().unwrap();
    }
    let rel::RelType::Aggregate(aggregate) = input.rel_type.as_ref().unwrap() else {
        panic!("expected an aggregate relation, got {input:?}");
    };
    aggregate
}

/// Verifies a two-phase avg is refused with an error naming the layer that lands it: its
/// Sirius state is a sum and a count, two columns for the one slot the FE allocates, and this
/// layer emits exactly one column per measure.
#[test]
fn two_phase_avg_is_refused_until_the_next_layer() {
    let mut aggregate = aggregate_expr(
        "avg",
        scalar_type(TPrimitiveType::DOUBLE),
        Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
    );
    aggregate.nodes[0].agg_expr = Some(TAggregateExpr::new(false));
    let mut node = aggregation_node(1, 1, Vec::new(), vec![aggregate]);
    node.agg_node.as_mut().unwrap().need_finalize = false;
    let err = translate_phase_case(node).unwrap_err();
    assert!(matches!(err, TranslateError::UnsupportedPlanNode { .. }));
    let message = err.to_string();
    assert!(message.contains("two-phase avg"), "{message}");
    assert!(message.contains("next translator layer"), "{message}");
    assert!(message.contains("new_planner_agg_stage"), "{message}");
}

/// Verifies grouped two-phase aggregation translates: the partial node keeps its grouping key
/// and emits the modeled state type for the measure.
#[test]
fn grouped_two_phase_translates() {
    let mut aggregate = aggregate_expr(
        "sum",
        scalar_type(TPrimitiveType::BIGINT),
        Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
    );
    aggregate.nodes[0].agg_expr = Some(TAggregateExpr::new(false));
    let mut node = aggregation_node(
        1,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate],
    );
    node.agg_node.as_mut().unwrap().need_finalize = false;
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![node, scan_node(0, 0)])),
        Some(agg_desc()),
        None,
    ))
    .unwrap();
    let root = root(&translated.plan);
    let rel::RelType::Aggregate(aggregate) =
        root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate relation");
    };
    assert_eq!(aggregate.grouping_expressions.len(), 1);
    assert_eq!(
        aggregate.measures[0].measure.as_ref().unwrap().phase,
        substrait::proto::AggregationPhase::InitialToIntermediate as i32
    );
}

/// Verifies the one-shot path labels its measures InitialToResult (advisory; the engine
/// ignores phases, but dumped plans should say what each aggregate is).
#[test]
fn one_shot_measures_are_labeled_initial_to_result() {
    let translated = translate_phase_case(phase_aggregation_node(&[false], true)).unwrap();
    let root = root(&translated.plan);
    let rel::RelType::Aggregate(aggregate) =
        root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate relation");
    };
    assert_eq!(
        aggregate.measures[0].measure.as_ref().unwrap().phase,
        substrait::proto::AggregationPhase::InitialToResult as i32
    );
}

/// Verifies the "merge serialize" combination (a 3/4-phase DISTINCT plan's middle stage) is
/// rejected as such.
#[test]
fn merge_serialize_aggregation_is_rejected() {
    let err = translate_phase_case(phase_aggregation_node(&[true], false)).unwrap_err();
    assert!(matches!(err, TranslateError::UnsupportedPlanNode { .. }));
    let message = err.to_string();
    assert!(message.contains("merge-serialize"), "{message}");
}

/// Verifies a node mixing merge and update measures cannot be phase-classified and is rejected.
#[test]
fn mixed_phase_aggregation_is_rejected() {
    let err = translate_phase_case(phase_aggregation_node(&[true, false], true)).unwrap_err();
    assert!(matches!(err, TranslateError::UnsupportedPlanNode { .. }));
    let message = err.to_string();
    assert!(message.contains("mixes merge and update"), "{message}");
}

/// Verifies a top-N sort becomes project (sort tuple) + sort + fetch with the node limit.
#[test]
fn sort_with_limit_becomes_project_sort_fetch() {
    let sort_info = TSortInfo::new(
        vec![slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT))],
        vec![true],
        vec![false],
        None,
    );
    let mut sort = base_plan_node(1, TPlanNodeType::SORT_NODE, 1, vec![1]);
    sort.limit = 5;
    sort.sort_node = Some(TSortNode::new(
        sort_info,
        true,
        Some(0),
        None,
        None,
        None,
        None,
        Some(vec![slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))]),
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
        None,
    ));
    // Sort tuple 1 materializes only the ordering column.
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![sort, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    let root = root(&translated.plan);
    let rel::RelType::Fetch(fetch) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected fetch relation");
    };
    #[allow(deprecated)]
    {
        assert_eq!(
            fetch.count_mode,
            Some(substrait::proto::fetch_rel::CountMode::Count(5))
        );
        assert_eq!(fetch.offset_mode, None);
    }
    let rel::RelType::Sort(sort) = fetch.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected sort under fetch");
    };
    assert_eq!(sort.sorts.len(), 1);
    assert_eq!(
        sort.sorts[0].sort_kind,
        Some(substrait::proto::sort_field::SortKind::Direction(
            substrait::proto::sort_field::SortDirection::AscNullsLast as i32
        ))
    );
    let rel::RelType::Project(_) = sort.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected sort-tuple projection under sort");
    };
}

/// Descriptor for the cross-fragment sort-order tests: scan tuple 0 = {o_orderdate DATE,
/// revenue DOUBLE}; sort tuple 1 re-materializes both, serialized in the order
/// [3 o_orderdate, 5 revenue].
fn sort_wire_desc() -> TDescriptorTable {
    desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "o_orderdate", scalar_type(TPrimitiveType::DATE)),
            slot(2, 0, "revenue", scalar_type(TPrimitiveType::DOUBLE)),
            slot(3, 1, "o_orderdate", scalar_type(TPrimitiveType::DATE)),
            slot(5, 1, "revenue", scalar_type(TPrimitiveType::DOUBLE)),
        ],
    )
}

/// Builds a SORT_NODE over sort tuple 1 with the given ordering and materialization exprs.
fn sort_node_over_tuple_1(ordering: Vec<TExpr>, slot_exprs: Vec<TExpr>) -> TPlanNode {
    let directions = vec![false; ordering.len()];
    let sort_info = TSortInfo::new(ordering, directions.clone(), directions, Some(slot_exprs));
    let mut sort = base_plan_node(1, TPlanNodeType::SORT_NODE, 1, vec![1]);
    sort.sort_node = Some(TSortNode::new(
        sort_info, false, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
    ));
    sort
}

/// The projected field indices of the sort-tuple materialization, in output order.
fn sort_projection_fields(translated: &TranslatedPlan) -> Vec<i32> {
    let root = root(&translated.plan);
    let rel::RelType::Sort(sort) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected sort relation");
    };
    let rel::RelType::Project(project) = sort.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected sort-tuple projection under sort");
    };
    project.expressions.iter().map(field_index).collect()
}

/// An ORDER BY whose key (`revenue`, a measure) is not the sort tuple's leading slot: the FE
/// lists the key's materialization first, but the wire row must ship in the tuple's
/// materialized slot order, because that is the order the receiving fragment declares its
/// stream with. This is the q03/q05 shape ("stream N column 0 is declared DATE but the source
/// sink produces DOUBLE").
#[test]
fn order_by_on_a_non_leading_sort_slot_ships_tuple_slot_order() {
    // FE list order: the ordering key `revenue` (child field 1) first, then `o_orderdate`.
    let sort = sort_node_over_tuple_1(
        vec![slot_ref(5, 1, scalar_type(TPrimitiveType::DOUBLE))],
        vec![
            slot_ref(2, 0, scalar_type(TPrimitiveType::DOUBLE)),
            slot_ref(1, 0, scalar_type(TPrimitiveType::DATE)),
        ],
    );
    let sender = translate_fragment(&params(
        Some(TPlan::new(vec![sort, scan_node(0, 0)])),
        Some(sort_wire_desc()),
        None,
    ))
    .unwrap();

    // The sender ships materialized-slot order: o_orderdate (child field 0) leads.
    assert_eq!(sort_projection_fields(&sender), vec![0, 1]);
    assert_eq!(sender.output_names, vec!["o_orderdate", "revenue"]);
    // The ORDER BY key still resolves to `revenue`, now at column 1 of the shipped row.
    let root = root(&sender.plan);
    let rel::RelType::Sort(sort) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected sort relation");
    };
    assert_eq!(sort.sorts.len(), 1);
    assert_eq!(field_index(sort.sorts[0].expr.as_ref().unwrap()), 1);
    assert_eq!(
        sort.sorts[0].sort_kind,
        Some(substrait::proto::sort_field::SortKind::Direction(
            substrait::proto::sort_field::SortDirection::DescNullsLast as i32
        ))
    );

    // The receiving fragment reads the same sort tuple through an exchange.
    let mut exchange = base_plan_node(14, TPlanNodeType::EXCHANGE_NODE, 0, vec![1]);
    exchange.exchange_node = Some(TExchangeNode::new(
        vec![1],
        None,
        None,
        Some(TPartitionType::UNPARTITIONED),
        Some(true),
        None,
    ));
    let receiver = PlanTranslator::new()
        .translate_fragment_with_exchange_inputs(
            &params(
                Some(TPlan::new(vec![exchange])),
                Some(sort_wire_desc()),
                None,
            ),
            &[ExchangeInput {
                node_id: 14,
                stream_view: "sirius_stream_14".to_string(),
                names: sender.output_names.clone(),
            }],
        )
        .unwrap();

    // Column-for-column: what the sender produces is what the receiver declares.
    // Sender column types come from mapping each projected field into the scan row
    // (field 0 = o_orderdate DATE, field 1 = revenue DOUBLE).
    let scan_types = ["DATE", "DOUBLE"];
    let produced = sort_projection_fields(&sender)
        .into_iter()
        .zip(&sender.output_names)
        .map(|(field, name)| (name.as_str(), scan_types[field as usize]))
        .collect::<Vec<_>>();
    let declared = receiver.stream_inputs[0]
        .columns
        .iter()
        .map(|column| (column.name.as_str(), column.ty.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(produced, declared);
    assert_eq!(
        declared,
        vec![("o_orderdate", "DATE"), ("revenue", "DOUBLE")]
    );
}

/// Control: an ORDER BY on the leading sort-tuple slot already lists the materialization in
/// slot order, so the projection is the identity mapping and nothing moves.
#[test]
fn order_by_on_the_leading_sort_slot_keeps_slot_order() {
    // FE list order: the ordering key `o_orderdate` (child field 0) first -- which is also the
    // sort tuple's leading materialized slot.
    let sort = sort_node_over_tuple_1(
        vec![slot_ref(3, 1, scalar_type(TPrimitiveType::DATE))],
        vec![
            slot_ref(1, 0, scalar_type(TPrimitiveType::DATE)),
            slot_ref(2, 0, scalar_type(TPrimitiveType::DOUBLE)),
        ],
    );
    let sender = translate_fragment(&params(
        Some(TPlan::new(vec![sort, scan_node(0, 0)])),
        Some(sort_wire_desc()),
        None,
    ))
    .unwrap();

    assert_eq!(sort_projection_fields(&sender), vec![0, 1]);
    assert_eq!(sender.output_names, vec!["o_orderdate", "revenue"]);
    let root = root(&sender.plan);
    let rel::RelType::Sort(sort) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected sort relation");
    };
    assert_eq!(field_index(sort.sorts[0].expr.as_ref().unwrap()), 0);
}

/// Descriptor for the q03-shaped aggregation tests: scan tuple 0 lists the columns in query
/// order (l_orderkey BIGINT, o_orderdate DATE, o_shippriority INT, price DOUBLE); the
/// aggregation output tuple 1 and the sort tuple 2 both re-materialize the keys and the sum,
/// serialized in ascending slot-id order [13, 16, 18, 35] -- so the tuples' wire order is not
/// the GROUP BY order [18, 13, 16].
fn q03_desc() -> TDescriptorTable {
    let mut slots = vec![
        slot(18, 0, "l_orderkey", scalar_type(TPrimitiveType::BIGINT)),
        slot(13, 0, "o_orderdate", scalar_type(TPrimitiveType::DATE)),
        slot(16, 0, "o_shippriority", scalar_type(TPrimitiveType::INT)),
        slot(20, 0, "price", scalar_type(TPrimitiveType::DOUBLE)),
    ];
    for tuple_id in [1, 2] {
        slots.extend([
            slot(
                13,
                tuple_id,
                "o_orderdate",
                scalar_type(TPrimitiveType::DATE),
            ),
            slot(
                16,
                tuple_id,
                "o_shippriority",
                scalar_type(TPrimitiveType::INT),
            ),
            slot(
                18,
                tuple_id,
                "l_orderkey",
                scalar_type(TPrimitiveType::BIGINT),
            ),
            slot(35, tuple_id, "revenue", scalar_type(TPrimitiveType::DOUBLE)),
        ]);
    }
    desc_table(vec![(0, Some(100)), (1, None), (2, None)], slots)
}

/// The q03 one-shot aggregation: GROUP BY l_orderkey, o_orderdate, o_shippriority (slot refs
/// 18, 13, 16 -- not ascending) with one sum measure, over scan tuple 0 into output tuple 1.
fn q03_aggregation_node(grouping_slots: &[i32]) -> TPlanNode {
    let scan_types = BTreeMap::from([
        (18, TPrimitiveType::BIGINT),
        (13, TPrimitiveType::DATE),
        (16, TPrimitiveType::INT),
        (20, TPrimitiveType::DOUBLE),
    ]);
    aggregation_node(
        1,
        1,
        grouping_slots
            .iter()
            .map(|slot_id| slot_ref(*slot_id, 0, scalar_type(scan_types[slot_id])))
            .collect(),
        vec![aggregate_expr(
            "sum",
            scalar_type(TPrimitiveType::DOUBLE),
            Some(slot_ref(20, 0, scalar_type(TPrimitiveType::DOUBLE))),
        )],
    )
}

/// The grouping-key field indices an aggregate reads from its input, in emitted order.
fn grouping_fields(aggregate: &substrait::proto::AggregateRel) -> Vec<i32> {
    aggregate
        .grouping_expressions
        .iter()
        .map(field_index)
        .collect()
}

/// Scan-tuple-0 column types of [`q03_desc`], indexed by field.
const Q03_SCAN_TYPES: [&str; 4] = ["BIGINT", "DATE", "INTEGER", "DOUBLE"];

/// GROUP BY keys whose slot ids are not in ascending order (the q03 shape: l_orderkey=18,
/// o_orderdate=13, o_shippriority=16): the FE lists the grouping exprs in GROUP BY order, but
/// every consumer of the output tuple resolves the row through the descriptor's
/// materialized-slot order, so the keys must be emitted in the tuple's order and the next
/// hop's declared stream schema must match the sender column-for-column. This is the q03/q18
/// shape ("stream 14 column 0 is declared DATE but the source sink produces BIGINT").
#[test]
fn group_by_keys_out_of_slot_order_ship_tuple_slot_order() {
    let sender = translate_fragment(&params(
        Some(TPlan::new(vec![
            q03_aggregation_node(&[18, 13, 16]),
            scan_node(0, 0),
        ])),
        Some(q03_desc()),
        None,
    ))
    .unwrap();

    // Keys in materialized order [13, 16, 18]: o_orderdate (scan field 1), o_shippriority
    // (field 2), l_orderkey (field 0) -- not the GROUP BY order [0, 1, 2].
    let aggregate = root_aggregate(&sender.plan);
    let key_fields = grouping_fields(aggregate);
    assert_eq!(key_fields, vec![1, 2, 0]);
    assert_eq!(
        sender.output_names,
        vec!["o_orderdate", "o_shippriority", "l_orderkey", "revenue"]
    );

    // The receiving fragment reads the aggregation output tuple through an exchange.
    let mut exchange = base_plan_node(14, TPlanNodeType::EXCHANGE_NODE, 0, vec![1]);
    exchange.exchange_node = Some(TExchangeNode::new(
        vec![1],
        None,
        None,
        Some(TPartitionType::UNPARTITIONED),
        Some(true),
        None,
    ));
    let receiver = PlanTranslator::new()
        .translate_fragment_with_exchange_inputs(
            &params(Some(TPlan::new(vec![exchange])), Some(q03_desc()), None),
            &[ExchangeInput {
                node_id: 14,
                stream_view: "sirius_stream_14".to_string(),
                names: sender.output_names.clone(),
            }],
        )
        .unwrap();

    // Column-for-column: what the aggregate produces is what the receiver declares.
    let produced = key_fields
        .iter()
        .map(|&field| Q03_SCAN_TYPES[field as usize])
        .chain(aggregate.measures.iter().map(measure_output_type))
        .zip(&sender.output_names)
        .map(|(ty, name)| (name.as_str(), ty))
        .collect::<Vec<_>>();
    let declared = receiver.stream_inputs[0]
        .columns
        .iter()
        .map(|column| (column.name.as_str(), column.ty.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(produced, declared);
    assert_eq!(
        declared,
        vec![
            ("o_orderdate", "DATE"),
            ("o_shippriority", "INTEGER"),
            ("l_orderkey", "BIGINT"),
            ("revenue", "DOUBLE")
        ]
    );
}

/// Builds the q03 TOP-N sort info: ORDER BY revenue DESC (nulls last), o_orderdate ASC (nulls
/// first), ordering over `ordering_tuple` and materializing from `input_tuple` in the FE's
/// list order (ordering keys first, then the leftover payload slots).
fn q03_sort_info(ordering_tuple: i32, input_tuple: i32) -> TSortInfo {
    TSortInfo::new(
        vec![
            slot_ref(35, ordering_tuple, scalar_type(TPrimitiveType::DOUBLE)),
            slot_ref(13, ordering_tuple, scalar_type(TPrimitiveType::DATE)),
        ],
        vec![false, true],
        vec![false, true],
        Some(vec![
            slot_ref(35, input_tuple, scalar_type(TPrimitiveType::DOUBLE)),
            slot_ref(13, input_tuple, scalar_type(TPrimitiveType::DATE)),
            slot_ref(16, input_tuple, scalar_type(TPrimitiveType::INT)),
            slot_ref(18, input_tuple, scalar_type(TPrimitiveType::BIGINT)),
        ]),
    )
}

/// The full q03 sender shape: the one-shot aggregation (keys 18, 13, 16) under a TOP-N with
/// two ordering keys (revenue DESC nulls-last, o_orderdate ASC nulls-first) and a limit. The
/// two sender-side reorders compose: the aggregate emits its keys in tuple order, the sort
/// projection re-materializes that row in the sort tuple's order, and each ordering key
/// resolves to the field really holding its column -- so the wire row matches the receiving
/// merging exchange's declared stream column-for-column.
#[test]
fn topn_over_group_by_keys_out_of_slot_order_resolves_both_sort_keys() {
    let mut sort = base_plan_node(2, TPlanNodeType::SORT_NODE, 1, vec![2]);
    sort.limit = 10;
    sort.sort_node = Some(TSortNode::new(
        q03_sort_info(2, 1),
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
    let sender = translate_fragment(&params(
        Some(TPlan::new(vec![
            sort,
            q03_aggregation_node(&[18, 13, 16]),
            scan_node(0, 0),
        ])),
        Some(q03_desc()),
        None,
    ))
    .unwrap();
    assert_eq!(
        sender.output_names,
        vec!["o_orderdate", "o_shippriority", "l_orderkey", "revenue"]
    );

    // Fetch(limit 10) over Sort over Project(sort tuple) over Aggregate.
    let root = root(&sender.plan);
    let rel::RelType::Fetch(fetch) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected the top-N fetch");
    };
    #[allow(deprecated)]
    {
        assert_eq!(
            fetch.count_mode,
            Some(substrait::proto::fetch_rel::CountMode::Count(10))
        );
    }
    let rel::RelType::Sort(sort) = fetch.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected sort under fetch");
    };
    // Both ordering keys resolve to the fields really holding their columns in the shipped
    // row: revenue at field 3 (DESC nulls-last), o_orderdate at field 0 (ASC nulls-first).
    assert_eq!(sort.sorts.len(), 2);
    assert_eq!(field_index(sort.sorts[0].expr.as_ref().unwrap()), 3);
    assert_eq!(
        sort.sorts[0].sort_kind,
        Some(substrait::proto::sort_field::SortKind::Direction(
            substrait::proto::sort_field::SortDirection::DescNullsLast as i32
        ))
    );
    assert_eq!(field_index(sort.sorts[1].expr.as_ref().unwrap()), 0);
    assert_eq!(
        sort.sorts[1].sort_kind,
        Some(substrait::proto::sort_field::SortKind::Direction(
            substrait::proto::sort_field::SortDirection::AscNullsFirst as i32
        ))
    );
    let rel::RelType::Project(project) = sort.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected sort-tuple projection under sort");
    };
    // The aggregate row and the sort tuple order the same slots the same way, so the
    // materialization projection is the identity.
    let projection_fields = project
        .expressions
        .iter()
        .map(field_index)
        .collect::<Vec<_>>();
    assert_eq!(projection_fields, vec![0, 1, 2, 3]);
    let rel::RelType::Aggregate(aggregate) =
        project.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate under the sort projection");
    };
    let key_fields = grouping_fields(aggregate);
    assert_eq!(key_fields, vec![1, 2, 0]);

    // The receiving fragment merge-sorts the same tuple through a merging exchange.
    let mut exchange = base_plan_node(14, TPlanNodeType::EXCHANGE_NODE, 0, vec![2]);
    exchange.exchange_node = Some(TExchangeNode::new(
        vec![2],
        Some(q03_sort_info(2, 2)),
        Some(0),
        Some(TPartitionType::UNPARTITIONED),
        Some(true),
        None,
    ));
    let receiver = PlanTranslator::new()
        .translate_fragment_with_exchange_inputs(
            &params(Some(TPlan::new(vec![exchange])), Some(q03_desc()), None),
            &[ExchangeInput {
                node_id: 14,
                stream_view: "sirius_stream_14".to_string(),
                names: sender.output_names.clone(),
            }],
        )
        .unwrap();

    // Column-for-column across the hop: the sender's sink row (the sort projection over the
    // aggregate row, whose key columns map into the scan) is what the receiver declares.
    let agg_row_types = key_fields
        .iter()
        .map(|&field| Q03_SCAN_TYPES[field as usize])
        .chain(aggregate.measures.iter().map(measure_output_type))
        .collect::<Vec<_>>();
    let produced = projection_fields
        .iter()
        .map(|&field| agg_row_types[field as usize])
        .zip(&sender.output_names)
        .map(|(ty, name)| (name.as_str(), ty))
        .collect::<Vec<_>>();
    let declared = receiver.stream_inputs[0]
        .columns
        .iter()
        .map(|column| (column.name.as_str(), column.ty.as_str()))
        .collect::<Vec<_>>();
    assert_eq!(produced, declared);
    assert_eq!(
        declared,
        vec![
            ("o_orderdate", "DATE"),
            ("o_shippriority", "INTEGER"),
            ("l_orderkey", "BIGINT"),
            ("revenue", "DOUBLE")
        ]
    );
    // The receiver's merge-sort keys resolve against the declared row the same way.
    let receiver_root = crate::root(&receiver.plan);
    let rel::RelType::Sort(merge_sort) = receiver_root
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    else {
        panic!("expected merging exchange sort");
    };
    assert_eq!(merge_sort.sorts.len(), 2);
    assert_eq!(field_index(merge_sort.sorts[0].expr.as_ref().unwrap()), 3);
    assert_eq!(field_index(merge_sort.sorts[1].expr.as_ref().unwrap()), 0);
}

/// Control: GROUP BY keys already listed in ascending slot-id order translate exactly as
/// before -- the materialized order is the FE order, so nothing moves (the rewritten-q03
/// shape that passes on a live cluster).
#[test]
fn group_by_keys_in_slot_order_keep_their_order() {
    let sender = translate_fragment(&params(
        Some(TPlan::new(vec![
            q03_aggregation_node(&[13, 16, 18]),
            scan_node(0, 0),
        ])),
        Some(q03_desc()),
        None,
    ))
    .unwrap();
    let aggregate = root_aggregate(&sender.plan);
    // The FE-order translation and the materialized-order translation coincide.
    assert_eq!(grouping_fields(aggregate), vec![1, 2, 0]);
    assert_eq!(
        sender.output_names,
        vec!["o_orderdate", "o_shippriority", "l_orderkey", "revenue"]
    );
}

/// A multi-key GROUP BY whose grouping expression is not a bare slot ref cannot be paired
/// with the output key slots, so the key order is unrecoverable and the node must refuse
/// rather than guess.
#[test]
fn non_slot_ref_grouping_expr_with_multiple_keys_is_rejected() {
    let mut agg = q03_aggregation_node(&[18, 13, 16]);
    agg.agg_node
        .as_mut()
        .unwrap()
        .grouping_exprs
        .as_mut()
        .unwrap()[1] = cast_expr(
        scalar_type(TPrimitiveType::DATE),
        slot_ref(13, 0, scalar_type(TPrimitiveType::DATE)),
    );
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(q03_desc()),
        None,
    ))
    .unwrap_err();
    assert!(
        matches!(
            err,
            TranslateError::UnsupportedPlanNode {
                node_type: TPlanNodeType::AGGREGATION_NODE,
                ..
            }
        ),
        "{err:?}"
    );
    assert!(err.to_string().contains("bare slot ref"), "{err}");
}

/// Multi-key grouping refs whose ids do not pair with the output tuple's key slots (a layout
/// the FE never serializes -- each output key slot reuses its grouping ref's id) must refuse
/// loudly: reordering on a wrong pairing would ship wrong columns.
#[test]
fn group_by_keys_that_do_not_pair_with_output_slots_are_rejected() {
    // Grouping refs [18, 20, 16]: slot 20 (the measure argument) is not an output key slot,
    // and key slot 13 pairs with nothing.
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![
            q03_aggregation_node(&[18, 20, 16]),
            scan_node(0, 0),
        ])),
        Some(q03_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(err, TranslateError::Descriptor(_)), "{err:?}");
    assert!(
        err.to_string()
            .contains("output key slot 13 pairs with no grouping expression"),
        "{err}"
    );
}

/// Two-table descriptor for join tests: tuple 0 = users(`a`), tuple 1 = orders(`b`).
fn join_desc() -> TDescriptorTable {
    desc_table(
        vec![(0, Some(100)), (1, Some(100))],
        vec![
            slot(1, 0, "a", scalar_type(TPrimitiveType::BIGINT)),
            slot(1, 1, "b", scalar_type(TPrimitiveType::BIGINT)),
        ],
    )
}

/// Two-table descriptor whose sides differ in width: tuple 0 = users(`a`, `b`), tuple 1 =
/// orders(`c`). A build-side slot lands at field 2, so an index into the concatenated
/// probe-then-build row cannot be confused with a literal `1`.
fn wide_join_desc() -> TDescriptorTable {
    desc_table(
        vec![(0, Some(100)), (1, Some(100))],
        vec![
            slot(1, 0, "a", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "b", scalar_type(TPrimitiveType::BIGINT)),
            slot(1, 1, "c", scalar_type(TPrimitiveType::BIGINT)),
        ],
    )
}

/// Builds a hash-join plan node with one `left = right` equality conjunct.
fn hash_join_node(join_op: TJoinOp) -> TPlanNode {
    let mut join = base_plan_node(2, TPlanNodeType::HASH_JOIN_NODE, 2, vec![0, 1]);
    join.hash_join_node = Some(THashJoinNode::new(
        join_op,
        vec![TEqJoinCondition::new(
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT)),
            Some(TExprOpcode::EQ),
        )],
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
        None,
    ));
    join
}

/// Field indices referenced by a scalar function's arguments, in order.
fn argument_field_indices(scalar: &expression::ScalarFunction) -> Vec<i32> {
    scalar
        .arguments
        .iter()
        .map(|argument| match argument.arg_type.as_ref().unwrap() {
            substrait::proto::function_argument::ArgType::Value(value) => field_index(value),
            other => panic!("unexpected argument {other:?}"),
        })
        .collect()
}

/// Verifies an inner hash join becomes a Substrait join whose equality condition references the
/// concatenated left-then-right row (right side offset by the left width).
#[test]
fn inner_hash_join_translates_to_join_rel() {
    let plan = TPlan::new(vec![
        hash_join_node(TJoinOp::INNER_JOIN),
        scan_node(0, 0),
        scan_node(1, 1),
    ]);
    let translated = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["a", "b"]);
    let rel::RelType::Join(join) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected join relation");
    };
    assert_eq!(
        join.r#type,
        substrait::proto::join_rel::JoinType::Inner as i32
    );
    let expression::RexType::ScalarFunction(equal) =
        join.expression.as_ref().unwrap().rex_type.as_ref().unwrap()
    else {
        panic!("expected scalar function join condition");
    };
    assert_eq!(argument_field_indices(equal), vec![0, 1]);
}

/// Verifies an ON-clause predicate beyond the equality is ANDed into the join condition, and that
/// both operands resolve against the concatenated probe-then-build row.
///
/// Run over the asymmetric descriptor: with a two-column probe side a build-side reference lands
/// at field 2, which a wrong offset (a literal 1, or the build width) cannot reproduce.
#[test]
fn other_join_conjuncts_are_anded_into_the_join_condition() {
    let bigint = || scalar_type(TPrimitiveType::BIGINT);
    let mut join = hash_join_node(TJoinOp::INNER_JOIN);
    join.hash_join_node.as_mut().unwrap().other_join_conjuncts = Some(vec![binary_pred(
        TExprOpcode::LT,
        slot_ref(2, 0, bigint()),
        slot_ref(1, 1, bigint()),
    )]);
    let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);
    let translated = translate_fragment(&params(Some(plan), Some(wide_join_desc()), None)).unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["a", "b", "c"]);
    let rel::RelType::Join(join) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected join relation");
    };
    let conjunction = scalar_fn(join.expression.as_ref().unwrap());
    let (urn, name) = resolved_function(&translated.plan, conjunction.function_reference);
    assert_eq!((urn.as_str(), name.as_str()), (URN_BOOLEAN, "and"));

    let operands: Vec<_> = conjunction
        .arguments
        .iter()
        .map(|argument| match argument.arg_type.as_ref().unwrap() {
            substrait::proto::function_argument::ArgType::Value(value) => scalar_fn(value),
            other => panic!("unexpected argument {other:?}"),
        })
        .collect();
    let names: Vec<_> = operands
        .iter()
        .map(|operand| resolved_function(&translated.plan, operand.function_reference).1)
        .collect();
    assert_eq!(names, vec!["equal", "lt"]);
    // `a = c` then `b < c`: fields 0 and 1 are the probe side, field 2 is the build side.
    assert_eq!(argument_field_indices(operands[0]), vec![0, 2]);
    assert_eq!(argument_field_indices(operands[1]), vec![1, 2]);
}

/// Verifies a join's own conjuncts become a filter over the join, resolved against the
/// concatenated row rather than the probe side alone.
#[test]
fn join_node_conjuncts_become_a_post_join_filter() {
    let mut join = hash_join_node(TJoinOp::INNER_JOIN);
    join.conjuncts = Some(vec![binary_pred(
        TExprOpcode::GT,
        slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT)),
        int_literal(10),
    )]);
    let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);
    let translated = translate_fragment(&params(Some(plan), Some(wide_join_desc()), None)).unwrap();

    let rel::RelType::Filter(filter) = root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    else {
        panic!("expected a filter over the join");
    };
    let rel::RelType::Join(_) = filter.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected the join under the filter");
    };
    let greater = scalar_fn(filter.condition.as_deref().unwrap());
    let substrait::proto::function_argument::ArgType::Value(probed) =
        greater.arguments[0].arg_type.as_ref().unwrap()
    else {
        panic!("expected a value argument");
    };
    assert_eq!(field_index(probed), 2);
}

/// Verifies a left semi join keeps only the probe-side row layout.
#[test]
fn left_semi_join_keeps_probe_layout() {
    let plan = TPlan::new(vec![
        hash_join_node(TJoinOp::LEFT_SEMI_JOIN),
        scan_node(0, 0),
        scan_node(1, 1),
    ]);
    let translated = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["a"]);
    let rel::RelType::Join(join) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected join relation");
    };
    assert_eq!(
        join.r#type,
        substrait::proto::join_rel::JoinType::LeftSemi as i32
    );
}

/// Builds a nested-loop join plan node carrying `conjuncts` as its join predicate.
fn nestloop_join_node(join_op: TJoinOp, conjuncts: Vec<TExpr>) -> TPlanNode {
    let mut join = base_plan_node(2, TPlanNodeType::NESTLOOP_JOIN_NODE, 2, vec![0, 1]);
    join.nestloop_join_node = Some(TNestLoopJoinNode::new(
        Some(join_op),
        None,
        Some(conjuncts),
        None,
        None,
        None,
    ));
    join
}

/// Builds `left OR right`.
fn or_pred(left: TExpr, right: TExpr) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::COMPOUND_PRED,
        scalar_type(TPrimitiveType::BOOLEAN),
        2,
    );
    node.opcode = Some(TExprOpcode::COMPOUND_OR);
    let mut nodes = vec![node];
    nodes.extend(left.nodes);
    nodes.extend(right.nodes);
    TExpr::new(nodes)
}

/// Verifies a cross nested-loop join becomes a filtered constant-key equality join.
#[test]
fn nestloop_join_translates_to_filtered_cross_rel() {
    let join = nestloop_join_node(
        TJoinOp::CROSS_JOIN,
        vec![binary_pred(
            TExprOpcode::LT,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT)),
        )],
    );
    let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);
    let translated = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["a", "b"]);
    let rel::RelType::Filter(filter) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected filter over constant-key join");
    };
    let rel::RelType::Project(project) = filter.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected output projection under filter");
    };
    let rel::RelType::Join(join) = project.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected constant-key join under projection");
    };
    assert_eq!(
        join.r#type,
        substrait::proto::join_rel::JoinType::Inner as i32
    );
}

/// Verifies a nested-loop join that is not inner or cross is rejected: the translation emits a
/// cross product, which keeps no unmatched rows.
#[test]
fn non_inner_nestloop_join_is_rejected() {
    for join_op in [TJoinOp::LEFT_OUTER_JOIN, TJoinOp::LEFT_SEMI_JOIN] {
        let plan = TPlan::new(vec![
            nestloop_join_node(
                join_op,
                vec![binary_pred(
                    TExprOpcode::LT,
                    slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
                    slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT)),
                )],
            ),
            scan_node(0, 0),
            scan_node(1, 1),
        ]);
        let err = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap_err();
        let TranslateError::UnsupportedPlanNode { reason, .. } = err else {
            panic!("{join_op:?}: expected an unsupported plan node, got {err:?}");
        };
        assert_eq!(reason, "only inner/cross nested-loop joins are supported");
    }
}

/// Verifies a nested-loop join still translates when the conjunct would not lift into a
/// comparison join: the synthetic constant key is the join condition, and the original
/// predicate stays a filter.
#[test]
fn nestloop_join_without_a_liftable_comparison_still_translates() {
    let bigint = || scalar_type(TPrimitiveType::BIGINT);
    let probe_side_only = binary_pred(TExprOpcode::LT, slot_ref(1, 0, bigint()), int_literal(10));
    let disjunction = or_pred(
        binary_pred(
            TExprOpcode::LT,
            slot_ref(1, 0, bigint()),
            slot_ref(1, 1, bigint()),
        ),
        binary_pred(
            TExprOpcode::GT,
            slot_ref(1, 0, bigint()),
            slot_ref(1, 1, bigint()),
        ),
    );

    for conjunct in [probe_side_only, disjunction] {
        let plan = TPlan::new(vec![
            nestloop_join_node(TJoinOp::CROSS_JOIN, vec![conjunct]),
            scan_node(0, 0),
            scan_node(1, 1),
        ]);
        let translated = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap();
        let root = root(&translated.plan);
        let rel::RelType::Filter(filter) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
        else {
            panic!("expected filter over constant-key join");
        };
        let rel::RelType::Project(project) =
            filter.input.as_ref().unwrap().rel_type.as_ref().unwrap()
        else {
            panic!("expected output projection under filter");
        };
        let rel::RelType::Join(join) = project.input.as_ref().unwrap().rel_type.as_ref().unwrap()
        else {
            panic!("expected constant-key join under projection");
        };
        assert_eq!(
            join.r#type,
            substrait::proto::join_rel::JoinType::Inner as i32
        );
    }
}

/// Builds an EXCHANGE_NODE `node_id` over `input_row_tuples`, optionally merging (`sort_info`)
/// and skipping `offset` rows.
fn exchange_node_with(
    node_id: i32,
    input_row_tuples: Vec<i32>,
    sort_info: Option<TSortInfo>,
    offset: Option<i64>,
) -> TPlanNode {
    let mut exchange = base_plan_node(
        node_id,
        TPlanNodeType::EXCHANGE_NODE,
        0,
        input_row_tuples.clone(),
    );
    exchange.exchange_node = Some(TExchangeNode::new(
        input_row_tuples,
        sort_info,
        offset,
        Some(TPartitionType::UNPARTITIONED),
        Some(true),
        None,
    ));
    exchange
}

/// Binds exchange node `node_id` to the engine view `sirius_stream_<node_id>` with the given
/// sender names.
fn stream_input(node_id: i32, names: &[&str]) -> ExchangeInput {
    ExchangeInput {
        node_id,
        stream_view: format!("sirius_stream_{node_id}"),
        names: names.iter().map(|name| name.to_string()).collect(),
    }
}

/// Translates `plan` over `desc` with the given input streams bound.
fn translate_with_streams(
    plan: TPlan,
    desc: TDescriptorTable,
    inputs: &[ExchangeInput],
) -> Result<TranslatedPlan, TranslateError> {
    PlanTranslator::new()
        .translate_fragment_with_exchange_inputs(&params(Some(plan), Some(desc), None), inputs)
}

/// Verifies an exchange node without a bound input stream is rejected, naming the node: a
/// fragment translated in isolation has nothing to read the exchange from.
#[test]
fn exchange_node_is_rejected() {
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![exchange_node_with(1, vec![0], None, None)])),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(
        matches!(
            err,
            TranslateError::UnsupportedPlanNode {
                node_id: 1,
                node_type: TPlanNodeType::EXCHANGE_NODE,
                ..
            }
        ),
        "{err:?}"
    );
}

/// Verifies a bound exchange becomes a stream read below the receiver aggregate, and that no
/// file appears anywhere in the plan: the boundary is a stream, not a parquet round-trip.
#[test]
fn bound_exchange_feeds_aggregate_from_a_stream() {
    let aggregate = aggregation_node(
        8,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate_expr(
            "sum",
            scalar_type(TPrimitiveType::BIGINT),
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    let translated = translate_with_streams(
        TPlan::new(vec![aggregate, exchange_node_with(7, vec![0], None, None)]),
        agg_desc(),
        &[stream_input(7, &["id", "name"])],
    )
    .unwrap();

    let root = root(&translated.plan);
    let rel::RelType::Aggregate(aggregate) =
        root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate receiver");
    };
    let rel::RelType::Read(read) = aggregate.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected exchange read");
    };
    let Some(read_rel::ReadType::NamedTable(table)) = read.read_type.as_ref() else {
        panic!("expected a stream read for the exchange input");
    };
    assert_eq!(table.names, vec!["sirius_stream_7"]);
    assert_eq!(read.base_schema.as_ref().unwrap().names, vec!["id", "name"]);
    assert!(translated.output_partition_columns.is_none());

    // The declaration the engine needs, derived from the same schema the read carries.
    assert_eq!(translated.stream_inputs.len(), 1);
    let stream = &translated.stream_inputs[0];
    assert_eq!(stream.node_id, 7);
    assert_eq!(stream.stream_view, "sirius_stream_7");
    assert_eq!(
        stream
            .columns
            .iter()
            .map(|column| (column.name.as_str(), column.ty.as_str()))
            .collect::<Vec<_>>(),
        vec![("id", "BIGINT"), ("name", "VARCHAR")]
    );
}

/// The sender's names win over the receiver's descriptor names: the stream's columns are bound
/// positionally and the fragment's root is named by its own tuple, so a rename at the boundary
/// changes the read's schema and nothing else.
#[test]
fn exchange_columns_take_the_senders_names_positionally() {
    let translated = translate_with_streams(
        TPlan::new(vec![exchange_node_with(7, vec![0], None, None)]),
        base_desc(),
        &[stream_input(7, &["sender_id", "sender_name"])],
    )
    .unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["id", "name"]);
    let rel::RelType::Read(read) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected the exchange read at the root");
    };
    assert_eq!(
        read.base_schema.as_ref().unwrap().names,
        vec!["sender_id", "sender_name"]
    );
    assert_eq!(
        translated.stream_inputs[0]
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        vec!["sender_id", "sender_name"]
    );
}

/// A multi-stage DISTINCT reallocates its columns into a fresh tuple at every stage, but the
/// FE's `buildAggregateTuple` never rebinds `colRefToExpr` for grouping columns, so every ref
/// above the first aggregation keeps naming the tuple from below it (TPC-H q16's
/// `count(distinct ps_suppkey)` reaches its final stage still naming the scan-side tuple). The
/// BE resolves slot refs by slot id alone, so those stale refs must resolve the same way here.
#[test]
fn stale_tuple_ids_from_a_multi_stage_distinct_resolve_by_slot_id() {
    // Four tuples reallocating the same two columns: suppkey keeps slot 2 and brand slot 9
    // through tuple 0 (below the aggregation), tuple 1 (the exchange input), and tuple 2 (the
    // dedup output); tuple 3 is the counting output (brand key, count slot 23).
    let desc = desc_table(
        vec![(0, Some(100)), (1, None), (2, None), (3, None)],
        vec![
            slot(2, 0, "ps_suppkey", scalar_type(TPrimitiveType::BIGINT)),
            slot(9, 0, "p_brand", scalar_type(TPrimitiveType::VARCHAR)),
            slot(2, 1, "ps_suppkey", scalar_type(TPrimitiveType::BIGINT)),
            slot(9, 1, "p_brand", scalar_type(TPrimitiveType::VARCHAR)),
            slot(2, 2, "ps_suppkey", scalar_type(TPrimitiveType::BIGINT)),
            slot(9, 2, "p_brand", scalar_type(TPrimitiveType::VARCHAR)),
            slot(9, 3, "p_brand", scalar_type(TPrimitiveType::VARCHAR)),
            slot(23, 3, "supplier_cnt", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );

    let exchange = exchange_node_with(4, vec![1], None, None);
    // The dedup stage: grouping-only, its refs stale-bound to tuple 0.
    let dedup = aggregation_node(
        5,
        2,
        vec![
            slot_ref(2, 0, scalar_type(TPrimitiveType::BIGINT)),
            slot_ref(9, 0, scalar_type(TPrimitiveType::VARCHAR)),
        ],
        Vec::new(),
    );
    // The counting stage: its count argument names tuple 0 too.
    let count = aggregation_node(
        6,
        3,
        vec![slot_ref(9, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate_expr(
            "count",
            scalar_type(TPrimitiveType::BIGINT),
            Some(slot_ref(2, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    let translated = translate_with_streams(
        TPlan::new(vec![count, dedup, exchange]),
        desc,
        &[stream_input(4, &["ps_suppkey", "p_brand"])],
    )
    .unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["p_brand", "supplier_cnt"]);
    let rel::RelType::Aggregate(counting) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected counting aggregate");
    };
    // The deduped row is [ps_suppkey, p_brand]: the count groups on brand and counts suppkey.
    assert_eq!(counting.grouping_expressions.len(), 1);
    assert_eq!(field_index(&counting.grouping_expressions[0]), 1);
    assert_eq!(counting.measures.len(), 1);
    let measure = counting.measures[0].measure.as_ref().unwrap();
    assert_eq!(measure.arguments.len(), 1);
    let substrait::proto::function_argument::ArgType::Value(argument) =
        measure.arguments[0].arg_type.as_ref().unwrap()
    else {
        panic!("expected value argument");
    };
    assert_eq!(field_index(argument), 0);

    let rel::RelType::Aggregate(dedup) =
        counting.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected dedup aggregate under the count");
    };
    let keys: Vec<_> = dedup.grouping_expressions.iter().map(field_index).collect();
    assert_eq!(keys, vec![0, 1]);
    assert!(dedup.measures.is_empty());
}

/// Verifies a merging exchange globally sorts the stream it reads: the senders' runs arrive in
/// no particular interleaving, and a plain read would drop the cross-fragment ORDER BY.
#[test]
fn merging_exchange_becomes_sort_over_stream_read() {
    let sort_info = TSortInfo::new(
        vec![slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))],
        vec![false],
        vec![false],
        None,
    );
    let translated = translate_with_streams(
        TPlan::new(vec![exchange_node_with(
            7,
            vec![0],
            Some(sort_info),
            Some(0),
        )]),
        base_desc(),
        &[stream_input(7, &["id", "name"])],
    )
    .unwrap();

    let root = root(&translated.plan);
    let rel::RelType::Sort(sort) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected merging exchange sort");
    };
    let rel::RelType::Read(read) = sort.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected an exchange read under the sort");
    };
    assert!(
        matches!(
            read.read_type.as_ref(),
            Some(read_rel::ReadType::NamedTable(_))
        ),
        "a merging exchange must stream its input too, not re-scan a file"
    );
    assert_eq!(sort.sorts.len(), 1);
    assert_eq!(field_index(sort.sorts[0].expr.as_ref().unwrap()), 0);
    assert_eq!(
        sort.sorts[0].sort_kind,
        Some(substrait::proto::sort_field::SortKind::Direction(
            substrait::proto::sort_field::SortDirection::DescNullsLast as i32
        ))
    );
}

/// Without `input_row_tuples` there is no row layout to type the stream with.
#[test]
fn exchange_without_input_row_tuples_is_rejected() {
    let err = translate_with_streams(
        TPlan::new(vec![exchange_node_with(7, Vec::new(), None, None)]),
        base_desc(),
        &[stream_input(7, &["id", "name"])],
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            TranslateError::MissingField {
                context: "TExchangeNode",
                field: "input_row_tuples"
            }
        ),
        "{err:?}"
    );
}

/// The view name is what ties the read to the stream the engine fills. An empty one would bind
/// a table named "" and fail far from the exchange that caused it.
#[test]
fn exchange_bound_to_an_empty_stream_view_is_rejected() {
    let err = translate_with_streams(
        TPlan::new(vec![exchange_node_with(7, vec![0], None, None)]),
        base_desc(),
        &[ExchangeInput {
            node_id: 7,
            stream_view: String::new(),
            names: vec!["id".to_string(), "name".to_string()],
        }],
    )
    .unwrap_err();
    assert!(
        matches!(err, TranslateError::MalformedPlan { .. }),
        "{err:?}"
    );
}

/// The sender's name list must have one entry per column of the declared row layout; a mismatch
/// would otherwise bind columns positionally under the wrong names.
#[test]
fn exchange_names_must_match_the_row_layout_arity() {
    let err = translate_with_streams(
        TPlan::new(vec![exchange_node_with(7, vec![0], None, None)]),
        base_desc(),
        &[stream_input(7, &["only_one"])],
    )
    .unwrap_err();
    assert!(matches!(err, TranslateError::Descriptor(_)), "{err:?}");
}

/// An exchange node carries its own skip offset, not just a sort. It must reach the fetch with an
/// explicit unlimited count, or the consumer decodes the unset count as `LIMIT 0`.
#[test]
#[allow(deprecated)]
fn exchange_offset_becomes_a_fetch() {
    let translated = translate_with_streams(
        TPlan::new(vec![exchange_node_with(7, vec![0], None, Some(5))]),
        base_desc(),
        &[stream_input(7, &["id", "name"])],
    )
    .unwrap();
    let rel::RelType::Fetch(fetch) = root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    else {
        panic!("expected the exchange offset to become a fetch");
    };
    assert_eq!(
        fetch.count_mode,
        Some(substrait::proto::fetch_rel::CountMode::Count(-1))
    );
    assert_eq!(
        fetch.offset_mode,
        Some(substrait::proto::fetch_rel::OffsetMode::Offset(5))
    );
}

/// Builds fragment params whose output sink is a data-stream sink with `partition`.
fn params_with_stream_sink(
    plan: TPlan,
    desc: TDescriptorTable,
    output_exprs: Option<Vec<TExpr>>,
    partition: TDataPartition,
) -> TExecPlanFragmentParams {
    let mut params = params(Some(plan), Some(desc), output_exprs);
    params.fragment.as_mut().unwrap().output_sink = Some(TDataSink::new(
        TDataSinkType::DATA_STREAM_SINK,
        Some(TDataStreamSink::new(
            9, partition, None, None, None, None, None,
        )),
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
    params
}

/// Builds a hash partition over the given key expressions.
fn hash_partition(keys: Vec<TExpr>) -> TDataPartition {
    TDataPartition::new(TPartitionType::HASH_PARTITIONED, Some(keys), None, None)
}

/// A hash-partitioned sink's keys resolve to output column indices in partition-expression
/// order, and an unpartitioned sink resolves to none: the sender hashes exactly the columns the
/// FE named, in the order it named them.
#[test]
fn hash_partitioned_sink_resolves_partition_keys_to_output_columns() {
    let translated = translate_fragment(&params_with_stream_sink(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        None,
        hash_partition(vec![
            slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)),
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
        ]),
    ))
    .unwrap();
    assert_eq!(translated.output_partition_columns, Some(vec![1, 0]));

    let unpartitioned = translate_fragment(&params_with_stream_sink(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        None,
        TDataPartition::new(TPartitionType::UNPARTITIONED, None, None, None),
    ))
    .unwrap();
    assert!(unpartitioned.output_partition_columns.is_none());
}

/// A transformed partition key is refused: this sender would hash a value its peers do not,
/// silently splitting equal keys across destinations.
#[test]
fn hash_partitioned_sink_with_a_transformed_key_is_rejected() {
    let err = translate_fragment(&params_with_stream_sink(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        None,
        hash_partition(vec![arithmetic(
            TExprOpcode::ADD,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            int_literal(1),
        )]),
    ))
    .unwrap_err();
    assert!(
        matches!(err, TranslateError::MalformedPlan { .. }),
        "{err:?}"
    );
}

/// A hash-partitioned sink with no partition expressions has nothing to hash, and one with
/// fragment output expressions has a sink row the FE's slot refs no longer describe. Both are
/// refused rather than guessed at.
#[test]
fn hash_partitioned_sink_without_resolvable_keys_is_rejected() {
    let no_keys = translate_fragment(&params_with_stream_sink(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        None,
        hash_partition(Vec::new()),
    ))
    .unwrap_err();
    assert!(
        matches!(no_keys, TranslateError::MalformedPlan { .. }),
        "{no_keys:?}"
    );

    let reprojected = translate_fragment(&params_with_stream_sink(
        TPlan::new(vec![scan_node(0, 0)]),
        base_desc(),
        Some(vec![slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))]),
        hash_partition(vec![slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))]),
    ))
    .unwrap_err();
    assert!(
        matches!(reprojected, TranslateError::MalformedPlan { .. }),
        "{reprojected:?}"
    );
}

/// Returns every extension function name declared by the plan.
fn extension_function_names(plan: &substrait::proto::Plan) -> Vec<String> {
    use substrait::proto::extensions::simple_extension_declaration::MappingType;
    plan.extensions
        .iter()
        .filter_map(|declaration| match declaration.mapping_type.as_ref() {
            Some(MappingType::ExtensionFunction(function)) => Some(function.name.clone()),
            _ => None,
        })
        .collect()
}

/// Builds a select node filtering the scan with `conjunct`.
fn filtered_scan(conjunct: TExpr) -> TPlan {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![conjunct]);
    TPlan::new(vec![select, scan_node(0, 0)])
}

/// Verifies arithmetic expressions become Substrait arithmetic functions.
#[test]
fn arithmetic_expression_translates() {
    let mut arith = base_expr_node(
        TExprNodeType::ARITHMETIC_EXPR,
        scalar_type(TPrimitiveType::BIGINT),
        2,
    );
    arith.opcode = Some(TExprOpcode::MULTIPLY);
    let mut nodes = vec![arith];
    nodes.extend(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)).nodes);
    nodes.extend(int_literal(2).nodes);
    let product = TExpr::new(nodes);

    let mut pred = base_expr_node(
        TExprNodeType::BINARY_PRED,
        scalar_type(TPrimitiveType::BOOLEAN),
        2,
    );
    pred.opcode = Some(TExprOpcode::GT);
    let mut nodes = vec![pred];
    nodes.extend(product.nodes);
    nodes.extend(int_literal(10).nodes);

    let translated = translate_fragment(&params(
        Some(filtered_scan(TExpr::new(nodes))),
        Some(base_desc()),
        None,
    ))
    .unwrap();
    let names = extension_function_names(&translated.plan);
    assert!(names.contains(&"multiply".to_string()), "{names:?}");
}

/// Verifies a DATE literal becomes a Substrait date literal in days since the epoch.
#[test]
fn date_literal_translates_to_epoch_days() {
    let mut date = base_expr_node(
        TExprNodeType::DATE_LITERAL,
        scalar_type(TPrimitiveType::DATE),
        0,
    );
    date.date_literal = Some(TDateLiteral::new("1998-09-02".to_string()));

    let mut pred = base_expr_node(
        TExprNodeType::BINARY_PRED,
        scalar_type(TPrimitiveType::BOOLEAN),
        2,
    );
    pred.opcode = Some(TExprOpcode::LE);
    let mut nodes = vec![pred];
    nodes.extend(slot_ref(1, 0, scalar_type(TPrimitiveType::DATE)).nodes);
    nodes.push(date);

    let translated = translate_fragment(&params(
        Some(filtered_scan(TExpr::new(nodes))),
        Some(desc_table(
            vec![(0, Some(100))],
            vec![slot(1, 0, "d", scalar_type(TPrimitiveType::DATE))],
        )),
        None,
    ))
    .unwrap();
    let condition = filter_condition(&translated.plan);
    let literal = literal_type(scalar_arg(condition, 1));
    assert_eq!(literal, &expression::literal::LiteralType::Date(10471));
}

/// Verifies `IN` predicates become singular-or-list expressions.
#[test]
fn in_predicate_translates_to_singular_or_list() {
    let mut in_pred = base_expr_node(
        TExprNodeType::IN_PRED,
        scalar_type(TPrimitiveType::BOOLEAN),
        3,
    );
    in_pred.in_predicate = Some(TInPredicate::new(false));
    let mut nodes = vec![in_pred];
    nodes.extend(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)).nodes);
    nodes.extend(int_literal(1).nodes);
    nodes.extend(int_literal(2).nodes);

    let translated = translate_fragment(&params(
        Some(filtered_scan(TExpr::new(nodes))),
        Some(base_desc()),
        None,
    ))
    .unwrap();
    let condition = filter_condition(&translated.plan);
    let expression::RexType::SingularOrList(list) = condition.rex_type.as_ref().unwrap() else {
        panic!("expected singular-or-list");
    };
    assert_eq!(list.options.len(), 2);
}

/// Verifies allowlisted function calls translate and unknown builtins are rejected.
#[test]
fn function_calls_use_allowlist() {
    let build = |name: &str| {
        let mut call = base_expr_node(
            TExprNodeType::FUNCTION_CALL,
            scalar_type(TPrimitiveType::BOOLEAN),
            2,
        );
        call.fn_ = Some(builtin_function(name, scalar_type(TPrimitiveType::BOOLEAN)));
        let mut nodes = vec![call];
        nodes.extend(slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)).nodes);
        nodes.extend(string_literal("%x%").nodes);
        TExpr::new(nodes)
    };

    let translated = translate_fragment(&params(
        Some(filtered_scan(build("like"))),
        Some(base_desc()),
        None,
    ))
    .unwrap();
    let names = extension_function_names(&translated.plan);
    assert!(names.contains(&"like".to_string()), "{names:?}");

    let err = translate_fragment(&params(
        Some(filtered_scan(build("hll_cardinality"))),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(err, TranslateError::MalformedPlan(_)));
}

/// Splits a throwing cast into its target-type kind and input expression.
fn cast_parts(
    expr: &substrait::proto::Expression,
) -> (
    &substrait::proto::r#type::Kind,
    &substrait::proto::Expression,
) {
    let expression::RexType::Cast(cast) = expr.rex_type.as_ref().unwrap() else {
        panic!("expected cast, got {expr:?}");
    };
    assert_eq!(
        cast.failure_behavior,
        expression::cast::FailureBehavior::ThrowException as i32
    );
    (
        cast.r#type.as_ref().unwrap().kind.as_ref().unwrap(),
        cast.input.as_ref().unwrap(),
    )
}

/// Verifies CASE WHEN chains become Substrait if-then expressions with a null default.
#[test]
fn case_expression_translates_to_if_then() {
    let mut case = base_expr_node(
        TExprNodeType::CASE_EXPR,
        scalar_type(TPrimitiveType::BIGINT),
        2,
    );
    case.case_expr = Some(TCaseExpr::new(false, false));
    let mut nodes = vec![case];
    nodes.extend(bool_literal(true).nodes);
    nodes.extend(int_literal(1).nodes);

    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(base_desc()),
        Some(vec![TExpr::new(nodes)]),
    ))
    .unwrap();
    let root = root(&translated.plan);
    let rel::RelType::Project(project) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected projection");
    };
    let expression::RexType::IfThen(if_then) = project.expressions[0].rex_type.as_ref().unwrap()
    else {
        panic!("expected if-then expression");
    };
    assert_eq!(if_then.ifs.len(), 1);
    assert!(
        if_then.r#else.is_some(),
        "CASE without else defaults to null"
    );
}

/// Verifies aggregation-node conjuncts (HAVING) become a filter over the aggregate output.
#[test]
fn aggregation_conjuncts_become_having_filter() {
    let mut agg = aggregation_node(
        1,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate_expr(
            "sum",
            scalar_type(TPrimitiveType::BIGINT),
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    // HAVING total > 10, referencing the aggregation output tuple.
    agg.conjuncts = Some(vec![binary_pred(
        TExprOpcode::GT,
        slot_ref(2, 1, scalar_type(TPrimitiveType::BIGINT)),
        int_literal(10),
    )]);
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(agg_desc()),
        None,
    ))
    .unwrap();

    let root = root(&translated.plan);
    let rel::RelType::Filter(filter) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected HAVING filter over the aggregate");
    };
    let rel::RelType::Aggregate(_) = filter.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate under the HAVING filter");
    };
}

/// Names the extension function a scalar-function expression invokes.
fn scalar_function_name(
    plan: &substrait::proto::Plan,
    expr: &substrait::proto::Expression,
) -> String {
    let expression::RexType::ScalarFunction(call) = expr.rex_type.as_ref().unwrap() else {
        panic!("expected a scalar function, got {expr:?}");
    };
    plan.extensions
        .iter()
        .find_map(|ext| {
            match ext.mapping_type.as_ref().unwrap() {
            substrait::proto::extensions::simple_extension_declaration::MappingType
                ::ExtensionFunction(f) if f.function_anchor == call.function_reference =>
            {
                Some(f.name.clone())
            }
            _ => None,
        }
        })
        .unwrap_or_else(|| panic!("no extension for anchor {}", call.function_reference))
}

/// Verifies each anti join is lowered through the specific supported form it needs, not merely
/// that it translates: a left anti becomes a LEFT join filtered on the build key being NULL, a
/// right anti mirrors that, and a null-aware left anti becomes a MARK join filtered on NOT of the
/// marker column the join appends. Asserting only the output arity cannot tell these apart, and
/// every one of them is a different answer.
#[test]
fn anti_hash_joins_are_lowered() {
    for (join_op, want_type, want_filter, want_filter_field, want_emit) in [
        (
            TJoinOp::LEFT_ANTI_JOIN,
            substrait::proto::join_rel::JoinType::Left,
            "is_null",
            vec![1],
            vec![0],
        ),
        (
            TJoinOp::RIGHT_ANTI_JOIN,
            substrait::proto::join_rel::JoinType::Right,
            "is_null",
            vec![0],
            vec![1],
        ),
        (
            TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN,
            substrait::proto::join_rel::JoinType::LeftMark,
            "not",
            vec![1],
            vec![0],
        ),
    ] {
        let plan = TPlan::new(vec![
            hash_join_node(join_op),
            scan_node(0, 0),
            scan_node(1, 1),
        ]);
        let translated = translate_fragment(&params(Some(plan), Some(join_desc()), None))
            .unwrap_or_else(|err| panic!("{join_op:?}: {err:?}"));

        let root = root(&translated.plan);
        assert_eq!(root.names.len(), 1, "{join_op:?}");
        let rel::RelType::Project(project) =
            root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
        else {
            panic!("{join_op:?}: expected the output projection under the root");
        };
        assert_eq!(
            emit_mapping(project.common.as_ref()),
            want_emit,
            "{join_op:?}"
        );

        let rel::RelType::Filter(filter) =
            project.input.as_ref().unwrap().rel_type.as_ref().unwrap()
        else {
            panic!("{join_op:?}: expected the anti-join filter under the projection");
        };
        assert_eq!(
            scalar_function_name(&translated.plan, filter.condition.as_ref().unwrap()),
            want_filter,
            "{join_op:?}"
        );
        let expression::RexType::ScalarFunction(scalar) = filter
            .condition
            .as_ref()
            .unwrap()
            .rex_type
            .as_ref()
            .unwrap()
        else {
            panic!("{join_op:?}: expected a scalar-function filter");
        };
        assert_eq!(
            argument_field_indices(scalar),
            want_filter_field,
            "{join_op:?}: the filter must test the build key (left anti), the probe key (right \
             anti) or the appended marker (null-aware)"
        );

        let rel::RelType::Join(join) = filter.input.as_ref().unwrap().rel_type.as_ref().unwrap()
        else {
            panic!("{join_op:?}: expected the join under the filter");
        };
        assert_eq!(join.r#type, want_type as i32, "{join_op:?}");
    }
}

/// The outer-join + `is_null(key)` lowering tells an unmatched row by the NULL the join pads the
/// other side with, so the null-tested key must propagate NULL. A column reference does, and so
/// does a cast of one; an arithmetic (or `if`/`case`) expression over the key is refused rather
/// than risk dropping unmatched rows.
#[test]
fn anti_join_with_non_column_key_is_rejected() {
    let times_two = |slot_id: i32, tuple_id: i32| {
        let mut arith = base_expr_node(
            TExprNodeType::ARITHMETIC_EXPR,
            scalar_type(TPrimitiveType::BIGINT),
            2,
        );
        arith.opcode = Some(TExprOpcode::MULTIPLY);
        let mut nodes = vec![arith];
        nodes.extend(slot_ref(slot_id, tuple_id, scalar_type(TPrimitiveType::BIGINT)).nodes);
        nodes.extend(int_literal(2).nodes);
        TExpr::new(nodes)
    };

    // LEFT ANTI null-tests the build (right) key; RIGHT ANTI null-tests the probe (left) key.
    let mut left_anti = hash_join_node(TJoinOp::LEFT_ANTI_JOIN);
    left_anti.hash_join_node.as_mut().unwrap().eq_join_conjuncts[0].right = times_two(1, 1);
    let mut right_anti = hash_join_node(TJoinOp::RIGHT_ANTI_JOIN);
    right_anti
        .hash_join_node
        .as_mut()
        .unwrap()
        .eq_join_conjuncts[0]
        .left = times_two(1, 0);
    for (label, join) in [("LEFT_ANTI", left_anti), ("RIGHT_ANTI", right_anti)] {
        let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);
        let err = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap_err();
        let TranslateError::UnsupportedPlanNode { reason, .. } = err else {
            panic!("{label}: expected an unsupported plan node, got {err:?}");
        };
        assert_eq!(
            reason, "anti join key is not a plain column reference",
            "{label}"
        );
    }

    // A cast over the column keeps NULL as NULL and stays supported.
    let mut cast_key = hash_join_node(TJoinOp::LEFT_ANTI_JOIN);
    cast_key.hash_join_node.as_mut().unwrap().eq_join_conjuncts[0].right = cast_expr(
        scalar_type(TPrimitiveType::BIGINT),
        slot_ref(1, 1, scalar_type(TPrimitiveType::INT)),
    );
    let plan = TPlan::new(vec![cast_key, scan_node(0, 0), scan_node(1, 1)]);
    translate_fragment(&params(Some(plan), Some(join_desc()), None))
        .expect("a cast of a column reference is still a column key");
}

/// A null-aware anti join is only equivalent to `LeftMark + NOT(marker)` when one equality key
/// decides the match. Both executors null out *every* unmatched marker as soon as any build row
/// has a NULL in any key, so a row made definitely non-matching by a second predicate is reported
/// UNKNOWN and dropped. Correlated `NOT IN` (correlation predicate in `other_join_conjuncts`) and
/// tuple `NOT IN` (several eq conjuncts) are therefore refused rather than silently returning too
/// few rows.
#[test]
fn null_aware_anti_join_with_extra_predicates_is_rejected() {
    let extra_eq = || {
        TEqJoinCondition::new(
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT)),
            Some(TExprOpcode::EQ),
        )
    };
    let correlation = || {
        binary_pred(
            TExprOpcode::EQ,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT)),
        )
    };

    // Tuple NOT IN: two equality keys.
    let mut multi_key = hash_join_node(TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN);
    multi_key
        .hash_join_node
        .as_mut()
        .unwrap()
        .eq_join_conjuncts
        .push(extra_eq());

    // Correlated NOT IN: the correlation predicate rides in other_join_conjuncts.
    let mut correlated = hash_join_node(TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN);
    correlated
        .hash_join_node
        .as_mut()
        .unwrap()
        .other_join_conjuncts = Some(vec![correlation()]);

    for (label, join) in [
        ("tuple NOT IN", multi_key),
        ("correlated NOT IN", correlated),
    ] {
        let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);
        let err = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap_err();
        let TranslateError::UnsupportedPlanNode { reason, .. } = err else {
            panic!("{label}: expected an unsupported plan node, got {err:?}");
        };
        assert_eq!(
            reason, "null-aware left anti join with correlated or multi-column keys",
            "{label}"
        );
    }

    // The single-key, no-extra-conjunct form stays supported: there, "unmatched with a NULL on
    // the build side" really is UNKNOWN, so the global rule is exact.
    let plan = TPlan::new(vec![
        hash_join_node(TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN),
        scan_node(0, 0),
        scan_node(1, 1),
    ]);
    translate_fragment(&params(Some(plan), Some(join_desc()), None))
        .expect("plain single-key null-aware anti join is still supported");
}

/// Verifies an unsupported join op is named as the reason even when the plan also carries no join
/// conjuncts, which is the shape some join types arrive in once the FE has folded predicates away.
#[test]
fn unsupported_join_type_is_reported_before_missing_conjuncts() {
    let mut join = hash_join_node(TJoinOp::CROSS_JOIN);
    join.hash_join_node.as_mut().unwrap().eq_join_conjuncts = vec![];
    let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);

    let err = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap_err();
    let TranslateError::UnsupportedPlanNode { reason, .. } = err else {
        panic!("expected an unsupported plan node, got {err:?}");
    };
    assert_eq!(reason, "hash join type is unsupported");
}

/// Asserts an expression is a throwing cast to FP64, the lowering every decimal operand and
/// decimal aggregate argument goes through.
fn assert_fp64_cast(expr: &substrait::proto::Expression) {
    let Some(expression::RexType::Cast(cast)) = expr.rex_type.as_ref() else {
        panic!("expected a cast, got {expr:?}");
    };
    assert!(
        matches!(
            cast.r#type.as_ref().unwrap().kind,
            Some(substrait::proto::r#type::Kind::Fp64(_))
        ),
        "cast target {:?}",
        cast.r#type
    );
    assert_eq!(
        cast.failure_behavior,
        expression::cast::FailureBehavior::ThrowException as i32
    );
}

/// Verifies decimal arithmetic is lowered to throwing FP64 casts for the GPU expression
/// evaluator, and that the result type is FP64 even when the FE result slot stays DECIMAL
/// (precision <= 18, where `map_type_desc` alone would keep it decimal).
#[test]
fn decimal_arithmetic_is_lowered_to_fp64() {
    for decimal in [
        scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(31), Some(4)),
        scalar_type_with(TPrimitiveType::DECIMAL64, None, Some(18), Some(2)),
    ] {
        let mut arith = base_expr_node(TExprNodeType::ARITHMETIC_EXPR, decimal.clone(), 2);
        arith.opcode = Some(TExprOpcode::MULTIPLY);
        let mut nodes = vec![arith];
        nodes.extend(slot_ref(1, 0, decimal.clone()).nodes);
        nodes.extend(slot_ref(1, 0, decimal.clone()).nodes);

        let translated = translate_fragment(&params(
            Some(TPlan::new(vec![scan_node(0, 0)])),
            Some(base_desc()),
            Some(vec![TExpr::new(nodes)]),
        ))
        .unwrap();
        let rel::RelType::Project(project) = root(&translated.plan)
            .input
            .as_ref()
            .unwrap()
            .rel_type
            .as_ref()
            .unwrap()
        else {
            panic!("expected output project");
        };
        let expression::RexType::ScalarFunction(function) =
            project.expressions[0].rex_type.as_ref().unwrap()
        else {
            panic!("expected arithmetic function");
        };
        assert_eq!(function.arguments.len(), 2, "{decimal:?}");
        for argument in &function.arguments {
            let substrait::proto::function_argument::ArgType::Value(value) =
                argument.arg_type.as_ref().unwrap()
            else {
                panic!("expected a value argument");
            };
            assert_fp64_cast(value);
        }
        assert!(
            matches!(
                function.output_type.as_ref().unwrap().kind,
                Some(substrait::proto::r#type::Kind::Fp64(_))
            ),
            "{decimal:?}"
        );
    }
}

/// Translates a one-phase aggregation with one measure over a BIGINT slot and returns that
/// measure's (possibly lowered) argument.
fn single_measure_argument(name: &str, ret_type: TTypeDesc) -> substrait::proto::Expression {
    let agg = aggregation_node(
        1,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate_expr(
            name,
            ret_type,
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(agg_desc()),
        None,
    ))
    .unwrap();
    let rel::RelType::Aggregate(aggregate) = root(&translated.plan)
        .input
        .as_ref()
        .unwrap()
        .rel_type
        .as_ref()
        .unwrap()
    else {
        panic!("expected aggregate");
    };
    let substrait::proto::function_argument::ArgType::Value(value) =
        aggregate.measures[0].measure.as_ref().unwrap().arguments[0]
            .arg_type
            .as_ref()
            .unwrap()
    else {
        panic!("expected a value argument");
    };
    value.clone()
}

/// Verifies a decimal AVG argument is lowered to a throwing FP64 cast for GPU execution.
#[test]
fn decimal_avg_is_lowered_to_fp64() {
    assert_fp64_cast(&single_measure_argument(
        "avg",
        scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(38), Some(8)),
    ));
}

/// Verifies a decimal SUM argument is lowered the same way, including when the FE result slot
/// stays DECIMAL (precision <= 18).
#[test]
fn decimal_sum_is_lowered_to_fp64() {
    assert_fp64_cast(&single_measure_argument(
        "sum",
        scalar_type_with(TPrimitiveType::DECIMAL64, None, Some(18), Some(2)),
    ));
}

/// Verifies an avg that lowers to neither the DOUBLE nor the decimal path (temporal avg, with
/// StarRocks-specific rounding) is refused with a reason that names what is supported.
#[test]
fn temporal_avg_is_rejected() {
    let agg = aggregation_node(
        1,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate_expr(
            "avg",
            scalar_type(TPrimitiveType::DATE),
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(agg_desc()),
        None,
    ))
    .unwrap_err();
    let TranslateError::UnsupportedExpression { node_type, reason } = err else {
        panic!("expected an unsupported expression, got {err:?}");
    };
    assert_eq!(node_type, TExprNodeType::AGG_EXPR);
    assert_eq!(
        reason,
        "avg is only supported where it lowers to the GPU's FP64 avg (DOUBLE and DECIMAL inputs)"
    );
}

/// Verifies partitioned top-N sorts are rejected rather than run as a global sort.
#[test]
fn partitioned_topn_sort_is_rejected() {
    let sort_info = TSortInfo::new(
        vec![slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT))],
        vec![true],
        vec![false],
        Some(vec![slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))]),
    );
    let mut sort = base_plan_node(1, TPlanNodeType::SORT_NODE, 1, vec![1]);
    let mut sort_node = TSortNode::new(
        sort_info, true, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
    );
    sort_node.partition_exprs = Some(vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))]);
    sort_node.partition_limit = Some(3);
    sort.sort_node = Some(sort_node);
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![sort, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap_err();
    assert!(
        matches!(
            err,
            TranslateError::UnsupportedPlanNode {
                node_type: TPlanNodeType::SORT_NODE,
                ..
            }
        ),
        "{err:?}"
    );
}

/// Verifies the sort-tuple materialization is read from `TSortInfo` (the resolved field), not
/// only from the deprecated node-level duplicate.
#[test]
fn sort_tuple_exprs_come_from_sort_info() {
    let sort_info = TSortInfo::new(
        vec![slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT))],
        vec![true],
        vec![false],
        Some(vec![slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))]),
    );
    let mut sort = base_plan_node(1, TPlanNodeType::SORT_NODE, 1, vec![1]);
    sort.sort_node = Some(TSortNode::new(
        sort_info, false, None, None, None, None, None, None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, None, None, None, None,
    ));
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![sort, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();
    let root = root(&translated.plan);
    let rel::RelType::Sort(sort) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected sort relation");
    };
    let rel::RelType::Project(_) = sort.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected sort-tuple projection from TSortInfo exprs");
    };
}

/// Verifies GPU-executor guards: non-constant LIKE patterns, non-constant substring bounds,
/// and ungrouped DISTINCT aggregates are rejected.
#[test]
fn gpu_unsupported_shapes_are_rejected() {
    // LIKE with a column pattern (not a literal).
    let mut like = base_expr_node(
        TExprNodeType::FUNCTION_CALL,
        scalar_type(TPrimitiveType::BOOLEAN),
        2,
    );
    like.fn_ = Some(builtin_function(
        "like",
        scalar_type(TPrimitiveType::BOOLEAN),
    ));
    let mut nodes = vec![like];
    nodes.extend(slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)).nodes);
    nodes.extend(slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)).nodes);
    let err = translate_fragment(&params(
        Some(filtered_scan(TExpr::new(nodes))),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(
        matches!(err, TranslateError::UnsupportedExpression { .. }),
        "{err:?}"
    );

    // substring with a non-constant start.
    let mut substr = base_expr_node(
        TExprNodeType::FUNCTION_CALL,
        scalar_type(TPrimitiveType::VARCHAR),
        3,
    );
    substr.fn_ = Some(builtin_function(
        "substring",
        scalar_type(TPrimitiveType::VARCHAR),
    ));
    let mut nodes = vec![substr];
    nodes.extend(slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)).nodes);
    nodes.extend(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)).nodes);
    nodes.extend(int_literal(2).nodes);
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(base_desc()),
        Some(vec![TExpr::new(nodes)]),
    ))
    .unwrap_err();
    assert!(
        matches!(err, TranslateError::UnsupportedExpression { .. }),
        "{err:?}"
    );

    // DISTINCT aggregate without grouping keys.
    let agg = aggregation_node(
        1,
        1,
        Vec::new(),
        vec![aggregate_expr(
            "multi_distinct_count",
            scalar_type(TPrimitiveType::BIGINT),
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "cnt", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap_err();
    assert!(
        matches!(err, TranslateError::UnsupportedPlanNode { .. }),
        "{err:?}"
    );
}

/// Extracts the struct-field index from a Substrait direct field reference.
fn field_index(expr: &substrait::proto::Expression) -> i32 {
    let Some(expression::RexType::Selection(selection)) = expr.rex_type.as_ref() else {
        panic!("expected a field reference, got {expr:?}");
    };
    let Some(expression::field_reference::ReferenceType::DirectReference(segment)) =
        selection.reference_type.as_ref()
    else {
        panic!("expected a direct reference");
    };
    let Some(expression::reference_segment::ReferenceType::StructField(field)) =
        segment.reference_type.as_ref()
    else {
        panic!("expected a struct field reference");
    };
    field.field
}

/// StarRocks orders an aggregation's output tuple by `groupBys` clause order, which is not
/// sorted by slot id: TPC-H Q18 emits `group by: 2: c_name, 1: c_custkey`. Pins that the
/// translated column order follows the descriptor's wire order rather than ascending slot id.
#[test]
fn aggregation_output_tuple_follows_wire_order_not_slot_id() {
    let agg = aggregation_node(
        1,
        1,
        vec![
            slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)),
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
        ],
        vec![aggregate_expr(
            "sum",
            scalar_type(TPrimitiveType::BIGINT),
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    // Output tuple 1 lists `name` (slot 2) before `id` (slot 1), matching the grouping order.
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(2, 1, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(3, 1, "total", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["name", "id", "total"]);
    let rel::RelType::Aggregate(aggregate) =
        root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate relation");
    };
    // Scan tuple 0 is `id` then `name`, so the keys resolve to fields 1 and 0 in that order.
    assert_eq!(
        aggregate
            .grouping_expressions
            .iter()
            .map(field_index)
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
}

/// StarRocks builds a sort tuple ordering-slots-first, so its wire order is not sorted by slot
/// id. Pins that the sort key resolves against the projection the translator emits.
#[test]
fn sort_tuple_follows_wire_order_not_slot_id() {
    let sort_info = TSortInfo::new(
        vec![slot_ref(2, 1, scalar_type(TPrimitiveType::VARCHAR))],
        vec![true],
        vec![false],
        None,
    );
    let mut sort = base_plan_node(1, TPlanNodeType::SORT_NODE, 1, vec![1]);
    sort.sort_node = Some(TSortNode::new(
        sort_info,
        true,
        Some(0),
        None,
        None,
        None,
        None,
        // Sort-tuple expressions in wire order: the ordering column first, then the payload.
        Some(vec![
            slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR)),
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
        ]),
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
        None,
    ));
    // Sort tuple 1 lists `name` (slot 2) before `id` (slot 1).
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(2, 1, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![sort, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["name", "id"]);
    let rel::RelType::Sort(sorted) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected sort relation");
    };
    // The projection below emits `name` first, so the ordering key is field 0.
    assert_eq!(field_index(sorted.sorts[0].expr.as_ref().unwrap()), 0);
}

/// Name of a Substrait type's kind, for asserting which descriptor slot a measure was paired with.
fn type_kind_name(ty: &substrait::proto::Type) -> &'static str {
    use substrait::proto::r#type::Kind;
    match ty.kind.as_ref().expect("measure output type") {
        Kind::I64(_) => "i64",
        Kind::Fp64(_) => "fp64",
        Kind::String(_) => "string",
        other => panic!("unexpected measure output type {other:?}"),
    }
}

/// StarRocks appends one output-tuple slot per aggregate after the grouping keys, so measure `i`
/// takes its declared output type from slot `keys + i`. The two measures are given different
/// types so that both an off-by-one into the grouping keys and a swap between the measures are
/// visible; with one measure, every wrong slice lands on the same slot.
#[test]
fn each_aggregate_takes_its_own_output_slot() {
    let agg = aggregation_node(
        1,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![
            aggregate_expr(
                "sum",
                scalar_type(TPrimitiveType::DOUBLE),
                Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
            ),
            aggregate_expr(
                "count",
                scalar_type(TPrimitiveType::BIGINT),
                Some(slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))),
            ),
        ],
    );
    // Output tuple 1: grouping key `name` (VARCHAR), then one slot per aggregate in aggregate
    // order — `total` DOUBLE, `n` BIGINT.
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(2, 1, "total", scalar_type(TPrimitiveType::DOUBLE)),
            slot(3, 1, "n", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["name", "total", "n"]);
    let rel::RelType::Aggregate(aggregate) =
        root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected aggregate relation");
    };
    assert_eq!(aggregate.grouping_expressions.len(), 1);
    assert_eq!(aggregate.measures.len(), 2);

    let calls: Vec<_> = aggregate
        .measures
        .iter()
        .map(|m| m.measure.as_ref().unwrap())
        .collect();
    // Each measure keeps its own argument: `sum(id)` reads field 0, `count(name)` field 1.
    let arg_fields: Vec<_> = calls
        .iter()
        .map(|call| {
            let substrait::proto::function_argument::ArgType::Value(expr) =
                call.arguments[0].arg_type.as_ref().unwrap()
            else {
                panic!("expected a value argument");
            };
            field_index(expr)
        })
        .collect();
    assert_eq!(arg_fields, vec![0, 1]);
    // And its own output slot: slice past the grouping keys, in aggregate order.
    let out_kinds: Vec<_> = calls
        .iter()
        .map(|call| type_kind_name(call.output_type.as_ref().unwrap()))
        .collect();
    assert_eq!(out_kinds, vec!["fp64", "i64"]);

    let names = extension_function_names(&translated.plan);
    assert!(names.contains(&"sum".to_string()), "{names:?}");
    assert!(names.contains(&"count".to_string()), "{names:?}");
}

/// Builds a SORT_NODE over sort tuple 1 carrying `limit` and `offset`, sorting on the single
/// BIGINT column the `sort_fetch_desc` fixture materializes.
fn sort_node_with(limit: i64, offset: Option<i64>) -> TPlanNode {
    let sort_info = TSortInfo::new(
        vec![slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT))],
        vec![true],
        vec![false],
        None,
    );
    let mut sort = base_plan_node(1, TPlanNodeType::SORT_NODE, 1, vec![1]);
    sort.limit = limit;
    sort.sort_node = Some(TSortNode::new(
        sort_info,
        true,
        offset,
        None,
        None,
        None,
        None,
        Some(vec![slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))]),
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
        None,
    ));
    sort
}

/// Scan tuple 0 (`id`, `name`) plus a sort tuple 1 materializing only `id`.
fn sort_fetch_desc() -> TDescriptorTable {
    desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
        ],
    )
}

/// Translates a fragment whose only node is a sort with `limit`/`offset`, returning its fetch.
#[allow(deprecated)]
fn fetch_modes(
    limit: i64,
    offset: Option<i64>,
) -> (
    Option<substrait::proto::fetch_rel::CountMode>,
    Option<substrait::proto::fetch_rel::OffsetMode>,
) {
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![
            sort_node_with(limit, offset),
            scan_node(0, 0),
        ])),
        Some(sort_fetch_desc()),
        None,
    ))
    .unwrap();
    let root = root(&translated.plan);
    match root.input.as_ref().unwrap().rel_type.as_ref().unwrap() {
        rel::RelType::Fetch(fetch) => (fetch.count_mode.clone(), fetch.offset_mode.clone()),
        other => panic!("expected a fetch relation, got {other:?}"),
    }
}

/// An offset with no limit must still emit an explicit unlimited count: DuckDB's consumer reads
/// the plain `count` field without checking the oneof, so an unset count decodes as `LIMIT 0` and
/// the query silently returns no rows.
#[test]
#[allow(deprecated)]
fn offset_without_limit_emits_an_unlimited_count() {
    use substrait::proto::fetch_rel::{CountMode, OffsetMode};
    assert_eq!(
        fetch_modes(-1, Some(5)),
        (Some(CountMode::Count(-1)), Some(OffsetMode::Offset(5)))
    );
}

/// `LIMIT n OFFSET m` carries both modes.
#[test]
#[allow(deprecated)]
fn limit_and_offset_emit_both_modes() {
    use substrait::proto::fetch_rel::{CountMode, OffsetMode};
    assert_eq!(
        fetch_modes(10, Some(5)),
        (Some(CountMode::Count(10)), Some(OffsetMode::Offset(5)))
    );
}

/// `LIMIT 0` is a real limit, not the "unset" sentinel: it must reach the plan as `Count(0)`
/// rather than being folded away into an unlimited fetch.
#[test]
#[allow(deprecated)]
fn zero_limit_is_not_treated_as_unlimited() {
    use substrait::proto::fetch_rel::CountMode;
    assert_eq!(fetch_modes(0, Some(0)), (Some(CountMode::Count(0)), None));
}

/// A limit on a non-sort node still becomes a fetch, and it sits *above* that node's conjunct
/// filter — StarRocks applies a scan or aggregation's limit to the rows that passed its
/// predicates, so `Filter(Fetch(..))` would truncate before filtering and return too few rows.
#[test]
#[allow(deprecated)]
fn a_limit_on_an_aggregation_fetches_above_its_having_filter() {
    let mut agg = aggregation_node(
        1,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate_expr(
            "sum",
            scalar_type(TPrimitiveType::BIGINT),
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    agg.limit = 3;
    agg.conjuncts = Some(vec![binary_pred(
        TExprOpcode::GT,
        slot_ref(2, 1, scalar_type(TPrimitiveType::BIGINT)),
        int_literal(10),
    )]);
    let translated = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(agg_desc()),
        None,
    ))
    .unwrap();

    let root = root(&translated.plan);
    let rel::RelType::Fetch(fetch) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected the limit to become a fetch above the aggregation");
    };
    assert_eq!(
        fetch.count_mode,
        Some(substrait::proto::fetch_rel::CountMode::Count(3))
    );
    let rel::RelType::Filter(filter) = fetch.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected the HAVING filter under the fetch");
    };
    let rel::RelType::Aggregate(_) = filter.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected the aggregate under the HAVING filter");
    };
}

/// A sorter carrying a payload tuple beyond the sort tuple is refused: only the first row tuple
/// is translated, so the rest would vanish from the output row.
#[test]
fn sort_with_a_second_row_tuple_is_rejected() {
    let mut sort = sort_node_with(-1, Some(0));
    sort.row_tuples = vec![1, 2];
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![sort, scan_node(0, 0)])),
        Some(sort_fetch_desc()),
        None,
    ))
    .unwrap_err();
    assert!(
        matches!(err, TranslateError::UnsupportedPlanNode { .. }),
        "{err:?}"
    );
}

/// StarRocks can fold a partial aggregation into the sorter; a Substrait sort has nowhere to put
/// it, so the node is refused rather than translated as a plain sort over unaggregated rows.
#[test]
fn sort_with_a_pre_aggregation_payload_is_rejected() {
    for with_slots in [false, true] {
        let mut sort = sort_node_with(-1, Some(0));
        let node = sort.sort_node.as_mut().unwrap();
        if with_slots {
            node.pre_agg_output_slot_id = Some(vec![1]);
        } else {
            node.pre_agg_exprs = Some(vec![aggregate_expr(
                "sum",
                scalar_type(TPrimitiveType::BIGINT),
                Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
            )]);
        }
        let err = translate_fragment(&params(
            Some(TPlan::new(vec![sort, scan_node(0, 0)])),
            Some(sort_fetch_desc()),
            None,
        ))
        .unwrap_err();
        assert!(
            matches!(err, TranslateError::UnsupportedPlanNode { .. }),
            "with_slots={with_slots}: {err:?}"
        );
    }
}

/// A sort carrying its own predicates is refused. StarRocks' sorter applies the limit internally
/// and never evaluates conjuncts — its backend asserts they are absent — so there is no reference
/// answer for whether the predicate runs before or after the truncation, and either choice
/// silently returns a different row set.
#[test]
fn sort_with_conjuncts_is_rejected() {
    let mut sort = sort_node_with(5, Some(0));
    sort.conjuncts = Some(vec![binary_pred(
        TExprOpcode::GT,
        slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT)),
        int_literal(10),
    )]);
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![sort, scan_node(0, 0)])),
        Some(sort_fetch_desc()),
        None,
    ))
    .unwrap_err();
    assert!(
        matches!(err, TranslateError::UnsupportedPlanNode { .. }),
        "{err:?}"
    );
}

/// Reads a relation's `RelCommon` emit mapping — what a consumer actually projects by.
fn emit_mapping(common: Option<&substrait::proto::RelCommon>) -> Vec<i32> {
    let Some(substrait::proto::rel_common::EmitKind::Emit(emit)) =
        common.and_then(|common| common.emit_kind.as_ref())
    else {
        panic!("expected an explicit emit mapping");
    };
    emit.output_mapping.clone()
}

/// A two-column left side and a one-column right side, so the synthetic-key arithmetic cannot be
/// satisfied by more than one formula.
fn asymmetric_join_desc() -> TDescriptorTable {
    desc_table(
        vec![(0, Some(100)), (1, Some(100))],
        vec![
            slot(1, 0, "a", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "a2", scalar_type(TPrimitiveType::BIGINT)),
            slot(1, 1, "b", scalar_type(TPrimitiveType::BIGINT)),
        ],
    )
}

/// Pins the synthetic-key index arithmetic, which is the only thing in this lowering that can be
/// wrong. With one column per side every off-by-one formula produces the same numbers; with a
/// 2x1 descriptor the join row is `[a, a2, key_l, b, key_r]`, so the condition must compare
/// fields 2 and 4 and the projection must emit `[0, 1, 3]` — dropping both synthetic keys.
#[test]
fn constant_key_join_indexes_past_the_synthetic_keys() {
    let join = nestloop_join_node(
        TJoinOp::CROSS_JOIN,
        vec![binary_pred(
            TExprOpcode::LT,
            slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT)),
            slot_ref(1, 1, scalar_type(TPrimitiveType::BIGINT)),
        )],
    );
    let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);
    let translated =
        translate_fragment(&params(Some(plan), Some(asymmetric_join_desc()), None)).unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["a", "a2", "b"]);
    let rel::RelType::Filter(filter) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected filter over constant-key join");
    };
    let rel::RelType::Project(project) = filter.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected output projection under filter");
    };
    // The projection only drops columns; it must not compute anything.
    assert!(project.expressions.is_empty());
    assert_eq!(emit_mapping(project.common.as_ref()), vec![0, 1, 3]);

    let rel::RelType::Join(join) = project.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected constant-key join under projection");
    };
    let expression::RexType::ScalarFunction(condition) =
        join.expression.as_ref().unwrap().rex_type.as_ref().unwrap()
    else {
        panic!("expected a scalar-function join condition");
    };
    let operands: Vec<_> = condition
        .arguments
        .iter()
        .map(|arg| {
            let substrait::proto::function_argument::ArgType::Value(expr) =
                arg.arg_type.as_ref().unwrap()
            else {
                panic!("expected a value argument");
            };
            field_index(expr)
        })
        .collect();
    assert_eq!(operands, vec![2, 4]);
}

/// `SELECT * FROM a, b` — a nested-loop join with no conjuncts at all. This is the shape the PR
/// exists to accept, and it takes the one branch the conjunct-carrying tests never reach: no
/// filter is emitted, so the projection is the root's direct input.
#[test]
fn bare_cross_join_translates_to_constant_key_join() {
    let join = nestloop_join_node(TJoinOp::CROSS_JOIN, Vec::new());
    let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);
    let translated =
        translate_fragment(&params(Some(plan), Some(asymmetric_join_desc()), None)).unwrap();

    let root = root(&translated.plan);
    assert_eq!(root.names, vec!["a", "a2", "b"]);
    let rel::RelType::Project(project) = root.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected the output projection directly under the root, with no filter");
    };
    assert!(project.expressions.is_empty());
    assert_eq!(emit_mapping(project.common.as_ref()), vec![0, 1, 3]);

    let rel::RelType::Join(join) = project.input.as_ref().unwrap().rel_type.as_ref().unwrap()
    else {
        panic!("expected constant-key join under projection");
    };
    assert_eq!(
        join.r#type,
        substrait::proto::join_rel::JoinType::Inner as i32
    );
    // Both operands are the appended literal keys, so the join is a Cartesian product expressed
    // as an equality the GPU planner accepts.
    let expression::RexType::ScalarFunction(condition) =
        join.expression.as_ref().unwrap().rex_type.as_ref().unwrap()
    else {
        panic!("expected a scalar-function join condition");
    };
    let operands: Vec<_> = condition
        .arguments
        .iter()
        .map(|arg| {
            let substrait::proto::function_argument::ArgType::Value(expr) =
                arg.arg_type.as_ref().unwrap()
            else {
                panic!("expected a value argument");
            };
            field_index(expr)
        })
        .collect();
    assert_eq!(operands, vec![2, 4]);
}
