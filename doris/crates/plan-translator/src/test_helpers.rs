//! Test helpers for constructing Doris Thrift structs.
//!
//! Building Thrift structs manually is verbose (50+ fields per struct).
//! These helpers create minimal valid structs for plan translation tests.

#![cfg(test)]

use std::collections::BTreeMap;

use doris_thrift::descriptors::{TDescriptorTable, TSlotDescriptor, TTupleDescriptor};
use doris_thrift::exprs::{
    TBoolLiteral, TDecimalLiteral, TExpr, TExprNode, TExprNodeType, TFloatLiteral, TIntLiteral,
    TIsNullPredicate, TLiteralPredicate, TSlotRef, TStringLiteral,
};
use doris_thrift::opcodes::TExprOpcode;
use doris_thrift::palo_internal_service::TPipelineFragmentParams;
use doris_thrift::partitions::{TDataPartition, TPartitionType};
use doris_thrift::plan_nodes::{
    TAggregationNode, TEqJoinCondition, TFileScanNode, THashJoinNode, TJoinOp, TNestedLoopJoinNode,
    TPlan, TPlanNode, TPlanNodeType, TSortInfo, TSortNode, TUnionNode,
};
use doris_thrift::planner::TPlanFragment;
use doris_thrift::types::{
    TFunction, TFunctionBinaryType, TFunctionName, TPrimitiveType, TScalarType, TTypeDesc,
    TTypeNode, TTypeNodeType, TUniqueId,
};

use ordered_float::OrderedFloat;

/// Create a minimal TTypeDesc for a primitive type.
pub fn type_desc(ptype: TPrimitiveType) -> TTypeDesc {
    TTypeDesc {
        types: Some(vec![TTypeNode {
            type_: TTypeNodeType::SCALAR,
            scalar_type: Some(TScalarType {
                type_: ptype,
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

/// Create a TTypeDesc for a VARCHAR type.
pub fn varchar_type_desc() -> TTypeDesc {
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

/// Create a TTypeDesc for DECIMAL with precision and scale.
pub fn decimal_type_desc(precision: i32, scale: i32) -> TTypeDesc {
    TTypeDesc {
        types: Some(vec![TTypeNode {
            type_: TTypeNodeType::SCALAR,
            scalar_type: Some(TScalarType {
                type_: TPrimitiveType::DECIMAL128I,
                len: None,
                precision: Some(precision),
                scale: Some(scale),
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

/// Create a TExpr containing a single INT_LITERAL node.
pub fn int_literal_expr(val: i64) -> TExpr {
    TExpr {
        nodes: vec![make_expr_node(
            TExprNodeType::INT_LITERAL,
            type_desc(TPrimitiveType::BIGINT),
            0,
            |n| n.int_literal = Some(TIntLiteral { value: val }),
        )],
    }
}

/// Create a TExpr containing a single STRING_LITERAL node.
pub fn string_literal_expr(val: &str) -> TExpr {
    TExpr {
        nodes: vec![make_expr_node(
            TExprNodeType::STRING_LITERAL,
            varchar_type_desc(),
            0,
            |n| {
                n.string_literal = Some(TStringLiteral {
                    value: val.to_string(),
                })
            },
        )],
    }
}

/// Create a TExpr containing a single FLOAT_LITERAL node.
pub fn float_literal_expr(val: f64) -> TExpr {
    TExpr {
        nodes: vec![make_expr_node(
            TExprNodeType::FLOAT_LITERAL,
            type_desc(TPrimitiveType::DOUBLE),
            0,
            |n| {
                n.float_literal = Some(TFloatLiteral {
                    value: OrderedFloat(val),
                })
            },
        )],
    }
}

/// Create a TExpr containing a single BOOL_LITERAL node.
pub fn bool_literal_expr(val: bool) -> TExpr {
    TExpr {
        nodes: vec![make_expr_node(
            TExprNodeType::BOOL_LITERAL,
            type_desc(TPrimitiveType::BOOLEAN),
            0,
            |n| n.bool_literal = Some(TBoolLiteral { value: val }),
        )],
    }
}

/// Create a TExpr containing a single DECIMAL_LITERAL node.
pub fn decimal_literal_expr(val: &str, precision: i32, scale: i32) -> TExpr {
    TExpr {
        nodes: vec![make_expr_node(
            TExprNodeType::DECIMAL_LITERAL,
            decimal_type_desc(precision, scale),
            0,
            |n| {
                n.decimal_literal = Some(TDecimalLiteral {
                    value: val.to_string(),
                })
            },
        )],
    }
}

/// Create a TExpr containing a SLOT_REF node.
pub fn slot_ref_expr(slot_id: i32, type_d: TTypeDesc) -> TExpr {
    TExpr {
        nodes: vec![make_expr_node(TExprNodeType::SLOT_REF, type_d, 0, |n| {
            n.slot_ref = Some(TSlotRef {
                slot_id,
                tuple_id: 0,
                col_unique_id: None,
                is_virtual_slot: None,
            })
        })],
    }
}

/// Create a TExpr containing a BINARY_PRED (comparison).
pub fn binary_pred_expr(opcode: TExprOpcode, left: TExpr, right: TExpr) -> TExpr {
    let mut nodes = vec![make_expr_node(
        TExprNodeType::BINARY_PRED,
        type_desc(TPrimitiveType::BOOLEAN),
        2,
        |n| n.opcode = Some(opcode),
    )];
    nodes.extend(left.nodes);
    nodes.extend(right.nodes);
    TExpr { nodes }
}

/// Create a TExpr containing a CAST_EXPR wrapping a child.
pub fn cast_expr(target_type: TTypeDesc, child: TExpr) -> TExpr {
    let mut nodes = vec![make_expr_node(
        TExprNodeType::CAST_EXPR,
        target_type,
        1,
        |_| {},
    )];
    nodes.extend(child.nodes);
    TExpr { nodes }
}

/// Create a TExprNode with the given type, setting extra fields via a callback.
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

/// Create a minimal TPlanNode.
pub fn make_plan_node(
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

/// Create a UNION_NODE with constant expression lists.
pub fn make_union_node(
    node_id: i32,
    tuple_id: i32,
    const_expr_lists: Vec<Vec<TExpr>>,
) -> TPlanNode {
    let mut node = make_plan_node(node_id, TPlanNodeType::UNION_NODE, 0, vec![tuple_id]);
    node.union_node = Some(TUnionNode {
        tuple_id,
        result_expr_lists: vec![],
        const_expr_lists,
        first_materialized_child_idx: 0,
    });
    node
}

/// Create an EXCHANGE_NODE with 0 children (exchange receiver).
pub fn make_exchange_node(node_id: i32, row_tuples: Vec<i32>) -> TPlanNode {
    make_plan_node(node_id, TPlanNodeType::EXCHANGE_NODE, 0, row_tuples)
}

/// Create an EMPTY_SET_NODE.
pub fn make_empty_set_node(node_id: i32) -> TPlanNode {
    make_plan_node(node_id, TPlanNodeType::EMPTY_SET_NODE, 0, vec![])
}

/// Create a FILE_SCAN_NODE.
pub fn make_file_scan_node(node_id: i32, tuple_id: i32, table_name: &str) -> TPlanNode {
    let mut node = make_plan_node(node_id, TPlanNodeType::FILE_SCAN_NODE, 0, vec![tuple_id]);
    node.file_scan_node = Some(TFileScanNode {
        tuple_id: Some(tuple_id),
        table_name: Some(table_name.to_string()),
    });
    node
}

/// Create a TPlan from nodes.
pub fn make_plan(nodes: Vec<TPlanNode>) -> TPlan {
    TPlan { nodes }
}

/// Create a minimal TDescriptorTable with tuples and slots.
pub fn make_desc_table(
    tuples: Vec<(i32, Option<i64>)>,
    slots: Vec<(i32, i32, i32, &str, TPrimitiveType)>,
) -> TDescriptorTable {
    let tuple_descriptors = tuples
        .iter()
        .map(|&(id, table_id)| TTupleDescriptor {
            id,
            byte_size: 0,
            num_null_bytes: 0,
            table_id,
            num_null_slots: None,
        })
        .collect();

    let slot_descriptors: Vec<TSlotDescriptor> = slots
        .iter()
        .map(|&(id, parent, col_pos, name, ref ptype)| TSlotDescriptor {
            id,
            parent,
            slot_type: type_desc(*ptype),
            column_pos: col_pos,
            byte_offset: 0,
            null_indicator_byte: 0,
            null_indicator_bit: 0,
            col_name: name.to_string(),
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
        })
        .collect();

    TDescriptorTable {
        slot_descriptors: Some(slot_descriptors),
        tuple_descriptors,
        table_descriptors: None,
    }
}

/// Create a TExpr containing a SLOT_REF node with explicit tuple_id.
pub fn slot_ref_expr_in_tuple(slot_id: i32, tuple_id: i32, type_d: TTypeDesc) -> TExpr {
    TExpr {
        nodes: vec![make_expr_node(TExprNodeType::SLOT_REF, type_d, 0, |n| {
            n.slot_ref = Some(TSlotRef {
                slot_id,
                tuple_id,
                col_unique_id: None,
                is_virtual_slot: None,
            })
        })],
    }
}

/// Create a TExpr containing a NULL_LITERAL node.
pub fn null_literal_expr(ptype: TPrimitiveType) -> TExpr {
    TExpr {
        nodes: vec![make_expr_node(
            TExprNodeType::NULL_LITERAL,
            type_desc(ptype),
            0,
            |_| {},
        )],
    }
}

/// Create a TExpr containing a COMPOUND_PRED (AND/OR/NOT).
pub fn compound_pred_expr(opcode: TExprOpcode, children: Vec<TExpr>) -> TExpr {
    let num_children = children.len() as i32;
    let mut nodes = vec![make_expr_node(
        TExprNodeType::COMPOUND_PRED,
        type_desc(TPrimitiveType::BOOLEAN),
        num_children,
        |n| n.opcode = Some(opcode),
    )];
    for child in children {
        nodes.extend(child.nodes);
    }
    TExpr { nodes }
}

/// Create a TExpr containing a FUNCTION_CALL node.
pub fn function_call_expr(
    func_name: &str,
    ret_type: TTypeDesc,
    arg_types: Vec<TTypeDesc>,
    children: Vec<TExpr>,
) -> TExpr {
    let num_children = children.len() as i32;
    let mut nodes = vec![make_expr_node(
        TExprNodeType::FUNCTION_CALL,
        ret_type.clone(),
        num_children,
        |n| {
            n.fn_ = Some(TFunction {
                name: TFunctionName {
                    db_name: None,
                    function_name: func_name.to_string(),
                },
                binary_type: TFunctionBinaryType::BUILTIN,
                arg_types,
                ret_type,
                has_var_args: false,
                comment: None,
                signature: None,
                hdfs_location: None,
                scalar_fn: None,
                aggregate_fn: None,
                id: None,
                checksum: None,
                vectorized: None,
                is_udtf_function: None,
                is_static_load: None,
                expiration_time: None,
                dict_function: None,
            });
        },
    )];
    for child in children {
        nodes.extend(child.nodes);
    }
    TExpr { nodes }
}

/// Create a TExpr containing an aggregate function (for AGGREGATION_NODE).
pub fn agg_function_expr(
    func_name: &str,
    ret_type: TTypeDesc,
    arg_types: Vec<TTypeDesc>,
    children: Vec<TExpr>,
) -> TExpr {
    // Aggregate function root is the same as FUNCTION_CALL with fn_ metadata.
    function_call_expr(func_name, ret_type, arg_types, children)
}

/// Create a TExpr containing an IS_NULL_PRED.
pub fn is_null_expr(child: TExpr) -> TExpr {
    let mut nodes = vec![make_expr_node(
        TExprNodeType::IS_NULL_PRED,
        type_desc(TPrimitiveType::BOOLEAN),
        1,
        |n| n.is_null_pred = Some(TIsNullPredicate { is_not_null: false }),
    )];
    nodes.extend(child.nodes);
    TExpr { nodes }
}

/// Create a TExpr containing an IN_PRED.
pub fn in_pred_expr(value: TExpr, options: Vec<TExpr>) -> TExpr {
    let num_children = 1 + options.len() as i32;
    let mut nodes = vec![make_expr_node(
        TExprNodeType::IN_PRED,
        type_desc(TPrimitiveType::BOOLEAN),
        num_children,
        |_| {},
    )];
    nodes.extend(value.nodes);
    for opt in options {
        nodes.extend(opt.nodes);
    }
    TExpr { nodes }
}

/// Create a TExpr containing a LITERAL_PRED (constant true/false).
pub fn literal_pred_expr(value: bool) -> TExpr {
    TExpr {
        nodes: vec![make_expr_node(
            TExprNodeType::LITERAL_PRED,
            type_desc(TPrimitiveType::BOOLEAN),
            0,
            |n| {
                n.literal_pred = Some(TLiteralPredicate {
                    value,
                    is_null: false,
                })
            },
        )],
    }
}

/// Create a HASH_JOIN_NODE with eq_join_conjuncts.
pub fn make_hash_join_node(
    node_id: i32,
    join_op: TJoinOp,
    row_tuples: Vec<i32>,
    eq_conjuncts: Vec<(TExpr, TExpr)>,
) -> TPlanNode {
    let eq_join_conjuncts = eq_conjuncts
        .into_iter()
        .map(|(left, right)| TEqJoinCondition {
            left,
            right,
            opcode: None,
        })
        .collect();
    let mut node = make_plan_node(node_id, TPlanNodeType::HASH_JOIN_NODE, 2, row_tuples);
    node.hash_join_node = Some(THashJoinNode {
        join_op,
        eq_join_conjuncts,
        other_join_conjuncts: None,
        add_probe_filters: None,
        vother_join_conjunct: None,
        hash_output_slot_ids: None,
        src_expr_list: None,
        voutput_tuple_id: None,
        vintermediate_tuple_id_list: None,
        is_broadcast_join: None,
        is_mark: None,
        dist_type: None,
        mark_join_conjuncts: None,
        use_specific_projections: None,
    });
    node
}

/// Create an AGGREGATION_NODE.
pub fn make_aggregation_node(
    node_id: i32,
    row_tuples: Vec<i32>,
    grouping_exprs: Option<Vec<TExpr>>,
    aggregate_functions: Vec<TExpr>,
    intermediate_tuple_id: i32,
    output_tuple_id: i32,
    need_finalize: bool,
) -> TPlanNode {
    let mut node = make_plan_node(node_id, TPlanNodeType::AGGREGATION_NODE, 1, row_tuples);
    node.agg_node = Some(TAggregationNode {
        grouping_exprs,
        aggregate_functions,
        intermediate_tuple_id,
        output_tuple_id,
        need_finalize,
        use_streaming_preaggregation: None,
        agg_sort_infos: None,
        is_first_phase: None,
        is_colocate: None,
        agg_sort_info_by_group_key: None,
    });
    node
}

/// Create a SORT_NODE.
pub fn make_sort_node(
    node_id: i32,
    row_tuples: Vec<i32>,
    ordering_exprs: Vec<TExpr>,
    is_asc_order: Vec<bool>,
    nulls_first: Vec<bool>,
    offset: Option<i64>,
    limit: i64,
) -> TPlanNode {
    let mut node = make_plan_node(node_id, TPlanNodeType::SORT_NODE, 1, row_tuples);
    node.limit = limit;
    node.sort_node = Some(TSortNode {
        sort_info: TSortInfo {
            ordering_exprs,
            is_asc_order,
            nulls_first,
            sort_tuple_slot_exprs: None,
            use_two_phase_read: None,
        },
        use_top_n: limit >= 0,
        offset,
        is_default_limit: None,
        use_topn_opt: None,
        merge_by_exchange: None,
        is_analytic_sort: None,
        is_colocate: None,
        algorithm: None,
        use_local_merge: None,
        full_sort_max_buffered_bytes: None,
    });
    node
}

/// Create a CROSS_JOIN_NODE (NESTED_LOOP_JOIN_NODE with CROSS_JOIN).
pub fn make_cross_join_node(node_id: i32, row_tuples: Vec<i32>) -> TPlanNode {
    let mut node = make_plan_node(node_id, TPlanNodeType::CROSS_JOIN_NODE, 2, row_tuples);
    node.nested_loop_join_node = Some(TNestedLoopJoinNode {
        join_op: TJoinOp::CROSS_JOIN,
        src_expr_list: None,
        voutput_tuple_id: None,
        vintermediate_tuple_id_list: None,
        is_output_left_side_only: None,
        vjoin_conjunct: None,
        is_mark: None,
        join_conjuncts: None,
        mark_join_conjuncts: None,
        use_specific_projections: None,
    });
    node
}

/// Create a SELECT_NODE (pass-through with conjuncts applied as filter).
pub fn make_select_node(node_id: i32, row_tuples: Vec<i32>, conjuncts: Vec<TExpr>) -> TPlanNode {
    let mut node = make_plan_node(node_id, TPlanNodeType::SELECT_NODE, 1, row_tuples);
    node.conjuncts = Some(conjuncts);
    node
}

/// Public access to make_expr_node for advanced test construction.
pub fn make_expr_node_pub(
    node_type: TExprNodeType,
    type_desc: TTypeDesc,
    num_children: i32,
    configure: impl FnOnce(&mut TExprNode),
) -> TExprNode {
    make_expr_node(node_type, type_desc, num_children, configure)
}

/// Create a minimal TPipelineFragmentParams.
pub fn make_fragment_params(plan: TPlan, desc_tbl: TDescriptorTable) -> TPipelineFragmentParams {
    TPipelineFragmentParams {
        protocol_version: doris_thrift::palo_internal_service::PaloInternalServiceVersion::V1,
        query_id: TUniqueId { hi: 1, lo: 1 },
        fragment_id: Some(0),
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
        fragment: Some(TPlanFragment {
            plan: Some(plan),
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
