use std::collections::BTreeMap;

use starrocks_plan_translator::{
    ExtensionRegistry, PlanTranslator, TranslateError, URN_BOOLEAN, URN_COMPARISON,
    translate_fragment,
};
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
    TFileFormatType, TFileScanNode, TFileScanType, THashJoinNode, TJoinOp, TNestLoopJoinNode,
    TPlan, TPlanNode, TPlanNodeType, TProjectNode, TScanRange, TSelectNode, TSortInfo, TSortNode,
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
/// parquet file, so it must not be passed to `parquet_scan`.
#[test]
fn empty_file_scan_range_is_unsupported() {
    assert_scan_range_unsupported(broker_scan_range(
        "file:///data/users.parquet",
        TFileFormatType::FORMAT_PARQUET,
        0,
        0,
        Some(0),
    ));
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

/// Verifies a merge-phase aggregate (two-phase aggregation) is rejected.
#[test]
fn merge_aggregation_is_rejected() {
    let mut aggregate = aggregate_expr(
        "sum",
        scalar_type(TPrimitiveType::BIGINT),
        Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
    );
    aggregate.nodes[0].agg_expr = Some(TAggregateExpr::new(true));
    let agg = aggregation_node(1, 1, Vec::new(), vec![aggregate]);
    // Output tuple 1 has two slots but no grouping keys, so use a dedicated descriptor with a
    // single aggregate output slot.
    let desc = desc_table(
        vec![(0, Some(100)), (1, None)],
        vec![
            slot(1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(1, 1, "total", scalar_type(TPrimitiveType::BIGINT)),
        ],
    );
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(desc),
        None,
    ))
    .unwrap_err();
    assert!(matches!(err, TranslateError::UnsupportedExpression { .. }));
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

/// Verifies a cross nested-loop join with a conjunct becomes filter-over-cross-product.
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
        panic!("expected filter over cross product");
    };
    let rel::RelType::Cross(_) = filter.input.as_ref().unwrap().rel_type.as_ref().unwrap() else {
        panic!("expected cross product under filter");
    };
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

/// Verifies a nested-loop join is rejected unless a conjunct survives as a join condition.
///
/// A predicate reading one input is pushed into that input and leaves a bare cross product; a
/// disjunction reading both is demoted to an any-join. The GPU planner has neither operator, so
/// either shape would translate here and then fail at execution.
#[test]
fn nestloop_join_without_a_liftable_comparison_is_rejected() {
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
        let err = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap_err();
        let TranslateError::UnsupportedPlanNode { reason, .. } = err else {
            panic!("expected an unsupported plan node, got {err:?}");
        };
        assert_eq!(
            reason,
            "nested-loop joins need a comparison between the two inputs"
        );
    }
}

/// Verifies an exchange node is still rejected: fragments are translated in isolation and
/// multi-fragment plans are a later milestone.
#[test]
fn exchange_node_is_rejected() {
    let exchange = base_plan_node(1, TPlanNodeType::EXCHANGE_NODE, 0, vec![0]);
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![exchange])),
        Some(base_desc()),
        None,
    ))
    .unwrap_err();
    assert!(matches!(
        err,
        TranslateError::UnsupportedPlanNode {
            node_type: TPlanNodeType::EXCHANGE_NODE,
            ..
        }
    ));
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

/// Verifies anti joins are rejected: the Substrait consumer has no left-anti conversion.
#[test]
fn anti_hash_join_is_rejected() {
    for join_op in [TJoinOp::LEFT_ANTI_JOIN, TJoinOp::NULL_AWARE_LEFT_ANTI_JOIN] {
        let plan = TPlan::new(vec![
            hash_join_node(join_op),
            scan_node(0, 0),
            scan_node(1, 1),
        ]);
        let err = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap_err();
        assert!(
            matches!(err, TranslateError::UnsupportedPlanNode { .. }),
            "{join_op:?}: {err:?}"
        );
    }
}

/// Verifies an unsupported join op is named as the reason even when the plan also carries no join
/// conjuncts, which is the shape an anti join arrives in once the FE has folded its predicate away.
#[test]
fn unsupported_join_type_is_reported_before_missing_conjuncts() {
    let mut join = hash_join_node(TJoinOp::LEFT_ANTI_JOIN);
    join.hash_join_node.as_mut().unwrap().eq_join_conjuncts = vec![];
    let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);

    let err = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap_err();
    let TranslateError::UnsupportedPlanNode { reason, .. } = err else {
        panic!("expected an unsupported plan node, got {err:?}");
    };
    assert_eq!(reason, "hash join type is unsupported");
}

/// Verifies decimal-typed arithmetic is rejected (it crashes the engine's GPU projection).
#[test]
fn decimal_arithmetic_is_rejected() {
    let decimal = scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(31), Some(4));
    let mut arith = base_expr_node(TExprNodeType::ARITHMETIC_EXPR, decimal.clone(), 2);
    arith.opcode = Some(TExprOpcode::MULTIPLY);
    let mut nodes = vec![arith];
    nodes.extend(slot_ref(1, 0, decimal.clone()).nodes);
    nodes.extend(slot_ref(1, 0, decimal).nodes);

    let err = translate_fragment(&params(
        Some(TPlan::new(vec![scan_node(0, 0)])),
        Some(base_desc()),
        Some(vec![TExpr::new(nodes)]),
    ))
    .unwrap_err();
    assert!(
        matches!(
            err,
            TranslateError::UnsupportedExpression {
                node_type: TExprNodeType::ARITHMETIC_EXPR,
                ..
            }
        ),
        "{err:?}"
    );
}

/// Verifies `avg` over decimals is rejected (DuckDB computes a double for decimal inputs).
#[test]
fn decimal_avg_is_rejected() {
    let decimal = scalar_type_with(TPrimitiveType::DECIMAL128, None, Some(38), Some(8));
    let agg = aggregation_node(
        1,
        1,
        vec![slot_ref(2, 0, scalar_type(TPrimitiveType::VARCHAR))],
        vec![aggregate_expr(
            "avg",
            decimal,
            Some(slot_ref(1, 0, scalar_type(TPrimitiveType::BIGINT))),
        )],
    );
    let err = translate_fragment(&params(
        Some(TPlan::new(vec![agg, scan_node(0, 0)])),
        Some(agg_desc()),
        None,
    ))
    .unwrap_err();
    assert!(
        matches!(err, TranslateError::UnsupportedExpression { .. }),
        "{err:?}"
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
/// bare cross joins, and ungrouped DISTINCT aggregates are rejected.
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

    // Nested-loop join without conjuncts (bare cross product).
    let mut join = base_plan_node(2, TPlanNodeType::NESTLOOP_JOIN_NODE, 2, vec![0, 1]);
    join.nestloop_join_node = Some(TNestLoopJoinNode::new(
        Some(TJoinOp::CROSS_JOIN),
        None,
        None,
        None,
        None,
        None,
    ));
    let plan = TPlan::new(vec![join, scan_node(0, 0), scan_node(1, 1)]);
    let err = translate_fragment(&params(Some(plan), Some(join_desc()), None)).unwrap_err();
    assert!(
        matches!(err, TranslateError::UnsupportedPlanNode { .. }),
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
