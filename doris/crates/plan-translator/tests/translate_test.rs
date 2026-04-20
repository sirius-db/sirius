//! Integration test: translate synthetic Doris fragments to Substrait.
//!
//! Tests the public `plan_translator::translate_fragment()` API with
//! full Thrift struct construction (no internal test helpers).

use std::collections::{BTreeMap, HashMap};

use prost::Message;

use doris_thrift::descriptors::*;
use doris_thrift::exprs::*;
use doris_thrift::opcodes::TExprOpcode;
use doris_thrift::palo_internal_service::*;
use doris_thrift::partitions::{TDataPartition, TPartitionType};
use doris_thrift::plan_nodes::*;
use doris_thrift::planner::TPlanFragment;
use doris_thrift::types::*;

use substrait::proto::{expression, plan_rel, rel, Plan};

// ---- Type descriptor helpers ----

fn bigint_type_desc() -> TTypeDesc {
    TTypeDesc {
        types: Some(vec![TTypeNode {
            type_: TTypeNodeType::SCALAR,
            scalar_type: Some(TScalarType {
                type_: TPrimitiveType::BIGINT,
                len: None,
                precision: None,
                scale: None,
                variant_max_subcolumns_count: None,
            }),
            struct_fields: None,
            contains_null: None,
            contains_nulls: None,
        }]),
        is_nullable: Some(true),
        byte_size: None,
        sub_types: None,
        result_is_nullable: None,
        function_name: None,
        be_exec_version: None,
    }
}

fn bool_type_desc() -> TTypeDesc {
    TTypeDesc {
        types: Some(vec![TTypeNode {
            type_: TTypeNodeType::SCALAR,
            scalar_type: Some(TScalarType {
                type_: TPrimitiveType::BOOLEAN,
                len: None,
                precision: None,
                scale: None,
                variant_max_subcolumns_count: None,
            }),
            struct_fields: None,
            contains_null: None,
            contains_nulls: None,
        }]),
        is_nullable: Some(true),
        byte_size: None,
        sub_types: None,
        result_is_nullable: None,
        function_name: None,
        be_exec_version: None,
    }
}

fn varchar_type_desc() -> TTypeDesc {
    TTypeDesc {
        types: Some(vec![TTypeNode {
            type_: TTypeNodeType::SCALAR,
            scalar_type: Some(TScalarType {
                type_: TPrimitiveType::VARCHAR,
                len: Some(65535),
                precision: None,
                scale: None,
                variant_max_subcolumns_count: None,
            }),
            struct_fields: None,
            contains_null: None,
            contains_nulls: None,
        }]),
        is_nullable: Some(true),
        byte_size: None,
        sub_types: None,
        result_is_nullable: None,
        function_name: None,
        be_exec_version: None,
    }
}

// ---- Expression helpers ----

fn make_expr_node(
    node_type: TExprNodeType,
    type_desc: TTypeDesc,
    num_children: i32,
    configure: impl FnOnce(&mut TExprNode),
) -> TExprNode {
    let mut node = TExprNode {
        node_type,
        type_: type_desc,
        num_children,
        output_scale: -1,
        opcode: None,
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
        fn_call_expr: None,
        large_int_literal: None,
        output_column: None,
        output_type: None,
        vector_opcode: None,
        fn_: None,
        vararg_start_idx: None,
        child_type: None,
        is_nullable: None,
        json_literal: None,
        schema_change_expr: None,
        column_ref: None,
        match_predicate: None,
        ipv4_literal: None,
        ipv6_literal: None,
        label: None,
        timev2_literal: None,
        varbinary_literal: None,
        is_cast_nullable: None,
        search_param: None,
    };
    configure(&mut node);
    node
}

fn slot_ref_node(slot_id: i32, tuple_id: i32, type_desc: TTypeDesc) -> TExprNode {
    make_expr_node(TExprNodeType::SLOT_REF, type_desc, 0, |n| {
        n.slot_ref = Some(TSlotRef {
            slot_id,
            tuple_id,
            col_unique_id: None,
            is_virtual_slot: None,
        })
    })
}

fn int_literal_node(value: i64) -> TExprNode {
    make_expr_node(TExprNodeType::INT_LITERAL, bigint_type_desc(), 0, |n| {
        n.int_literal = Some(TIntLiteral { value })
    })
}

fn binary_pred_node(opcode: TExprOpcode) -> TExprNode {
    make_expr_node(TExprNodeType::BINARY_PRED, bool_type_desc(), 2, |n| {
        n.opcode = Some(opcode)
    })
}

// ---- Plan node helpers ----

fn make_plan_node(
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

fn file_scan_plan_node(
    node_id: i32,
    tuple_id: i32,
    table_name: &str,
    conjuncts: Option<Vec<TExpr>>,
) -> TPlanNode {
    let mut node = make_plan_node(node_id, TPlanNodeType::FILE_SCAN_NODE, 0, vec![tuple_id]);
    node.conjuncts = conjuncts;
    node.file_scan_node = Some(TFileScanNode {
        tuple_id: Some(tuple_id),
        table_name: Some(table_name.to_string()),
    });
    node
}

// ---- Descriptor table helpers ----

fn make_desc_table() -> TDescriptorTable {
    let slot0 = TSlotDescriptor {
        id: 0,
        parent: 0,
        slot_type: bigint_type_desc(),
        column_pos: 0,
        byte_offset: 0,
        null_indicator_byte: 0,
        null_indicator_bit: 0,
        col_name: "col1".to_string(),
        slot_idx: 0,
        is_materialized: true,
        col_unique_id: None,
        is_key: None,
        need_materialize: None,
        is_auto_increment: None,
        column_paths: None,
        col_default_value: None,
        primitive_type: None,
        virtual_column_expr: None,
        all_access_paths: None,
        predicate_access_paths: None,
    };

    let slot1 = TSlotDescriptor {
        id: 1,
        parent: 0,
        slot_type: varchar_type_desc(),
        column_pos: 1,
        byte_offset: 8,
        null_indicator_byte: 0,
        null_indicator_bit: 1,
        col_name: "col2".to_string(),
        slot_idx: 1,
        is_materialized: true,
        col_unique_id: None,
        is_key: None,
        need_materialize: None,
        is_auto_increment: None,
        column_paths: None,
        col_default_value: None,
        primitive_type: None,
        virtual_column_expr: None,
        all_access_paths: None,
        predicate_access_paths: None,
    };

    let tuple0 = TTupleDescriptor {
        id: 0,
        byte_size: 16,
        num_null_bytes: 1,
        table_id: None,
        num_null_slots: None,
    };

    TDescriptorTable {
        slot_descriptors: Some(vec![slot0, slot1]),
        tuple_descriptors: vec![tuple0],
        table_descriptors: None,
    }
}

fn make_fragment_params(
    plan_nodes: Vec<TPlanNode>,
    desc_tbl: TDescriptorTable,
) -> TPipelineFragmentParams {
    let plan = TPlan { nodes: plan_nodes };
    let partition = TDataPartition {
        type_: TPartitionType::UNPARTITIONED,
        partition_exprs: None,
        partition_infos: None,
    };
    let fragment = TPlanFragment {
        plan: Some(plan),
        output_exprs: None,
        output_sink: None,
        partition,
        min_reservation_bytes: None,
        initial_reservation_total_claims: None,
        query_cache_param: None,
    };

    TPipelineFragmentParams {
        protocol_version: PaloInternalServiceVersion::V1,
        query_id: TUniqueId { hi: 1, lo: 1 },
        fragment_id: Some(1),
        per_exch_num_senders: BTreeMap::new(),
        desc_tbl: Some(desc_tbl),
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
        fragment: Some(fragment),
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

// ---- Tests ----

#[test]
fn test_translate_scan_with_filter() {
    // Build: SELECT col1, col2 FROM test_table WHERE col1 > 100
    let conjunct = TExpr {
        nodes: vec![
            binary_pred_node(TExprOpcode::GT),
            slot_ref_node(0, 0, bigint_type_desc()),
            int_literal_node(100),
        ],
    };

    let scan_node = file_scan_plan_node(0, 0, "test_table", Some(vec![conjunct]));
    let desc_tbl = make_desc_table();
    let params = make_fragment_params(vec![scan_node], desc_tbl);
    let table_schemas = HashMap::new();

    let result = plan_translator::translate_fragment(&params, &table_schemas, &HashMap::new())
        .expect("translation failed");
    assert!(
        !result.substrait_bytes.is_empty(),
        "plan bytes should not be empty"
    );

    // Decode the Substrait plan and verify structure.
    let plan =
        Plan::decode(result.substrait_bytes.as_slice()).expect("failed to decode Substrait plan");

    // Check version.
    assert_eq!(plan.version.as_ref().unwrap().producer, "sirius-doris-be");

    // Check we have one relation.
    assert_eq!(plan.relations.len(), 1);
    let plan_rel = &plan.relations[0];
    let root = match plan_rel.rel_type.as_ref().unwrap() {
        plan_rel::RelType::Root(r) => r,
        _ => panic!("expected RelRoot"),
    };

    // Check output names.
    assert_eq!(root.names, vec!["col1", "col2"]);

    // The root input should be a FilterRel (from the conjunct) wrapping a ReadRel.
    let top_rel = root.input.as_ref().unwrap();
    match top_rel.rel_type.as_ref().unwrap() {
        rel::RelType::Filter(filter) => {
            // Filter condition should be present.
            assert!(filter.condition.is_some());

            // Filter input should be a ReadRel.
            let input = filter.input.as_ref().unwrap();
            match input.rel_type.as_ref().unwrap() {
                rel::RelType::Read(read) => {
                    let schema = read.base_schema.as_ref().unwrap();
                    assert_eq!(schema.names, vec!["col1", "col2"]);
                }
                other => panic!("expected ReadRel, got {:?}", std::mem::discriminant(other)),
            }

            // Verify filter is a scalar function (gt).
            let cond = filter.condition.as_ref().unwrap();
            match cond.rex_type.as_ref().unwrap() {
                expression::RexType::ScalarFunction(f) => {
                    assert_eq!(f.arguments.len(), 2);
                }
                other => panic!(
                    "expected ScalarFunction, got {:?}",
                    std::mem::discriminant(other)
                ),
            }
        }
        other => panic!(
            "expected FilterRel, got {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Check extensions (should have "gt" function registered).
    assert!(
        !plan.extensions.is_empty(),
        "should have extension functions"
    );
}

#[test]
fn test_translate_scan_without_filter() {
    // Build: SELECT col1, col2 FROM test_table (no WHERE clause)
    let scan_node = file_scan_plan_node(0, 0, "test_table", None);
    let desc_tbl = make_desc_table();
    let params = make_fragment_params(vec![scan_node], desc_tbl);
    let table_schemas = HashMap::new();

    let result = plan_translator::translate_fragment(&params, &table_schemas, &HashMap::new())
        .expect("translation failed");
    let plan = Plan::decode(result.substrait_bytes.as_slice()).expect("failed to decode");

    let root = match plan.relations[0].rel_type.as_ref().unwrap() {
        plan_rel::RelType::Root(r) => r,
        _ => panic!("expected RelRoot"),
    };

    // Without a filter, the top rel should be a ReadRel directly.
    let top_rel = root.input.as_ref().unwrap();
    match top_rel.rel_type.as_ref().unwrap() {
        rel::RelType::Read(read) => {
            let schema = read.base_schema.as_ref().unwrap();
            assert_eq!(schema.names, vec!["col1", "col2"]);
        }
        other => panic!("expected ReadRel, got {:?}", std::mem::discriminant(other)),
    }

    // No extension functions needed when there's no filter.
    assert!(plan.extensions.is_empty());
}

#[test]
fn test_output_names_match_descriptor() {
    // Verify output_names come from descriptor table columns.
    let scan_node = file_scan_plan_node(0, 0, "test_table", None);
    let desc_tbl = make_desc_table();
    let params = make_fragment_params(vec![scan_node], desc_tbl);
    let table_schemas = HashMap::new();

    let result =
        plan_translator::translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();
    assert_eq!(result.output_names, vec!["col1", "col2"]);
}

#[test]
fn test_multiple_filter_predicates() {
    // Build: SELECT col1, col2 FROM test_table WHERE col1 > 100 AND col1 < 200
    let pred1 = TExpr {
        nodes: vec![
            binary_pred_node(TExprOpcode::GT),
            slot_ref_node(0, 0, bigint_type_desc()),
            int_literal_node(100),
        ],
    };
    let pred2 = TExpr {
        nodes: vec![
            binary_pred_node(TExprOpcode::LT),
            slot_ref_node(0, 0, bigint_type_desc()),
            int_literal_node(200),
        ],
    };

    let scan_node = file_scan_plan_node(0, 0, "test_table", Some(vec![pred1, pred2]));
    let desc_tbl = make_desc_table();
    let params = make_fragment_params(vec![scan_node], desc_tbl);
    let table_schemas = HashMap::new();

    let result =
        plan_translator::translate_fragment(&params, &table_schemas, &HashMap::new()).unwrap();
    let plan = Plan::decode(result.substrait_bytes.as_slice()).unwrap();

    let root = match &plan.relations[0].rel_type {
        Some(plan_rel::RelType::Root(r)) => r,
        _ => panic!("expected Root"),
    };

    // Should have a FilterRel at the top.
    let top_rel = root.input.as_ref().unwrap();
    match top_rel.rel_type.as_ref().unwrap() {
        rel::RelType::Filter(filter) => {
            // Multiple conjuncts should be combined with AND.
            let cond = filter.condition.as_ref().unwrap();
            match cond.rex_type.as_ref().unwrap() {
                expression::RexType::ScalarFunction(f) => {
                    // AND function should have 2 arguments (the two predicates).
                    assert_eq!(f.arguments.len(), 2);
                }
                other => panic!(
                    "expected AND ScalarFunction, got {:?}",
                    std::mem::discriminant(other)
                ),
            }
        }
        other => panic!(
            "expected FilterRel, got {:?}",
            std::mem::discriminant(other)
        ),
    }

    // Should have extensions for "gt", "lt", and "and".
    assert!(plan.extensions.len() >= 3);
}
