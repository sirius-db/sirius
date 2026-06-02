use std::collections::BTreeMap;

use starrocks_plan_translator::{
    ExtensionRegistry, PlanTranslator, TranslateError, URN_BOOLEAN, URN_COMPARISON,
    translate_fragment,
};
use starrocks_thrift::descriptors::{
    TDescriptorTable, TSlotDescriptor, TTableDescriptor, TTupleDescriptor,
};
use starrocks_thrift::exprs::{
    TBoolLiteral, TExpr, TExprNode, TExprNodeType, TFloatLiteral, TIntLiteral, TSlotRef,
    TStringLiteral,
};
use starrocks_thrift::internal_service::{InternalServiceVersion, TExecPlanFragmentParams};
use starrocks_thrift::opcodes::TExprOpcode;
use starrocks_thrift::partitions::{TDataPartition, TPartitionType};
use starrocks_thrift::plan_nodes::{
    TFileScanNode, TPlan, TPlanNode, TPlanNodeType, TProjectNode, TSelectNode,
};
use starrocks_thrift::planner::TPlanFragment;
use starrocks_thrift::types::{
    TPrimitiveType, TScalarType, TTableType, TTypeDesc, TTypeNode, TTypeNodeType,
};
use substrait::proto::{expression, plan_rel, read_rel, rel};

fn scalar_type(primitive: TPrimitiveType) -> TTypeDesc {
    scalar_type_with(primitive, None, None, None)
}

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

fn complex_type(kind: TTypeNodeType) -> TTypeDesc {
    TTypeDesc::new(Some(vec![TTypeNode::new(kind, None, None, None)]))
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

fn desc_table(tuples: Vec<(i32, Option<i64>)>, slots: Vec<TSlotDescriptor>) -> TDescriptorTable {
    TDescriptorTable::new(
        Some(slots),
        tuples
            .into_iter()
            .map(|(tuple_id, table_id)| {
                TTupleDescriptor::new(Some(tuple_id), None, None, table_id, None)
            })
            .collect(),
        Some(vec![TTableDescriptor::new(
            100,
            TTableType::HDFS_TABLE,
            2,
            0,
            "users".to_string(),
            "tpch".to_string(),
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
        )]),
        None,
    )
}

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

fn int_literal(value: i64) -> TExpr {
    int_literal_typed(value, TPrimitiveType::BIGINT)
}

fn int_literal_typed(value: i64, primitive: TPrimitiveType) -> TExpr {
    let mut node = base_expr_node(TExprNodeType::INT_LITERAL, scalar_type(primitive), 0);
    node.int_literal = Some(TIntLiteral::new(value));
    TExpr::new(vec![node])
}

fn float_literal_typed(value: f64, primitive: TPrimitiveType) -> TExpr {
    let mut node = base_expr_node(TExprNodeType::FLOAT_LITERAL, scalar_type(primitive), 0);
    node.float_literal = Some(TFloatLiteral::new(value.into()));
    TExpr::new(vec![node])
}

fn string_literal(value: &str) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::STRING_LITERAL,
        scalar_type(TPrimitiveType::VARCHAR),
        0,
    );
    node.string_literal = Some(TStringLiteral::new(value.to_string()));
    TExpr::new(vec![node])
}

fn bool_literal(value: bool) -> TExpr {
    let mut node = base_expr_node(
        TExprNodeType::BOOL_LITERAL,
        scalar_type(TPrimitiveType::BOOLEAN),
        0,
    );
    node.bool_literal = Some(TBoolLiteral::new(value));
    TExpr::new(vec![node])
}

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

fn scan_node(node_id: i32, tuple_id: i32) -> TPlanNode {
    let mut node = base_plan_node(node_id, TPlanNodeType::FILE_SCAN_NODE, 0, vec![tuple_id]);
    node.file_scan_node = Some(TFileScanNode::new(tuple_id, None, None, None));
    node
}

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

fn params_without_fragment(desc_tbl: Option<TDescriptorTable>) -> TExecPlanFragmentParams {
    let mut params = params(Some(TPlan::new(vec![scan_node(0, 0)])), desc_tbl, None);
    params.fragment = None;
    params
}

fn base_desc() -> TDescriptorTable {
    desc_table(
        vec![(0, Some(100))],
        vec![
            slot(1, 0, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, 1, "name", scalar_type(TPrimitiveType::VARCHAR)),
        ],
    )
}

fn root(plan: &substrait::proto::Plan) -> &substrait::proto::RelRoot {
    match plan.relations[0].rel_type.as_ref().unwrap() {
        plan_rel::RelType::Root(root) => root,
        _ => panic!("expected root relation"),
    }
}

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

fn literal_type(expr: &substrait::proto::Expression) -> &expression::literal::LiteralType {
    match expr.rex_type.as_ref().unwrap() {
        expression::RexType::Literal(literal) => literal.literal_type.as_ref().unwrap(),
        other => panic!("expected literal, got {other:?}"),
    }
}

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

#[test]
fn duplicate_output_names_are_unique_and_match_root() {
    let desc = desc_table(
        vec![(0, Some(100))],
        vec![
            slot(1, 0, 0, "name", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, 1, "name_1", scalar_type(TPrimitiveType::BIGINT)),
            slot(3, 0, 2, "name", scalar_type(TPrimitiveType::BIGINT)),
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
        vec![slot(1, 0, 0, "id", scalar_type(TPrimitiveType::INT))],
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
            slot(1, 0, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(2, 0, 1, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(3, 1, 0, "name", scalar_type(TPrimitiveType::VARCHAR)),
            slot(4, 1, 1, "id", scalar_type(TPrimitiveType::BIGINT)),
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

#[test]
fn unsupported_hash_join_is_structured_error() {
    let join = base_plan_node(9, TPlanNodeType::HASH_JOIN_NODE, 0, vec![0]);
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
        } if node_type == TPlanNodeType::HASH_JOIN_NODE
    ));
}

#[test]
fn unsupported_expression_is_structured_error() {
    let mut select = base_plan_node(1, TPlanNodeType::SELECT_NODE, 1, vec![0]);
    select.select_node = Some(TSelectNode::new(None));
    select.conjuncts = Some(vec![TExpr::new(vec![base_expr_node(
        TExprNodeType::FUNCTION_CALL,
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
        } if node_type == TExprNodeType::FUNCTION_CALL
    ));
}

#[test]
fn unsupported_complex_type_is_structured_error() {
    let desc = desc_table(
        vec![(0, Some(100))],
        vec![slot(1, 0, 0, "items", complex_type(TTypeNodeType::ARRAY))],
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

#[test]
fn non_materialized_unsupported_slot_type_is_ignored() {
    let mut hidden_slot = slot(2, 0, 1, "hidden", complex_type(TTypeNodeType::ARRAY));
    hidden_slot.is_materialized = Some(false);

    let desc = desc_table(
        vec![(0, Some(100))],
        vec![
            slot(1, 0, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
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

#[test]
fn unsupported_largeint_is_structured_error() {
    let desc = desc_table(
        vec![(0, Some(100))],
        vec![slot(1, 0, 0, "big", scalar_type(TPrimitiveType::LARGEINT))],
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

#[test]
fn unsupported_decimal256_is_structured_error() {
    let desc = desc_table(
        vec![(0, Some(100))],
        vec![slot(
            1,
            0,
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
            slot(1, 0, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
            slot(3, 1, 0, "id", scalar_type(TPrimitiveType::BIGINT)),
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

#[test]
fn extension_registry_keys_functions_by_urn_and_name() {
    let mut registry = ExtensionRegistry::new();
    let boolean_anchor = registry.register_function(URN_BOOLEAN, "overlap");
    let comparison_anchor = registry.register_function(URN_COMPARISON, "overlap");
    let reused_boolean_anchor = registry.register_function(URN_BOOLEAN, "overlap");

    assert_ne!(boolean_anchor, comparison_anchor);
    assert_eq!(boolean_anchor, reused_boolean_anchor);
}

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
