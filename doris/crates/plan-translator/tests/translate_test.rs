//! Integration test: translate a synthetic Doris fragment to Substrait.

use std::collections::BTreeMap;

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

// ---- Helper constructors ----

fn bigint_type_desc() -> TTypeDesc {
    TTypeDesc::new(
        vec![TTypeNode::new(
            TTypeNodeType::SCALAR,
            TScalarType::new(TPrimitiveType::BIGINT, None, None, None, None),
            None,
            None,
            None,
        )],
        true, // is_nullable
        None,
        None,
        None,
        None,
        None,
    )
}

fn bool_type_desc() -> TTypeDesc {
    TTypeDesc::new(
        vec![TTypeNode::new(
            TTypeNodeType::SCALAR,
            TScalarType::new(TPrimitiveType::BOOLEAN, None, None, None, None),
            None,
            None,
            None,
        )],
        true,
        None,
        None,
        None,
        None,
        None,
    )
}

fn varchar_type_desc() -> TTypeDesc {
    TTypeDesc::new(
        vec![TTypeNode::new(
            TTypeNodeType::SCALAR,
            TScalarType::new(TPrimitiveType::VARCHAR, 65535, None, None, None),
            None,
            None,
            None,
        )],
        true,
        None,
        None,
        None,
        None,
        None,
    )
}

/// Build a SLOT_REF expression node.
fn slot_ref_node(slot_id: i32, tuple_id: i32, type_desc: TTypeDesc) -> TExprNode {
    TExprNode::new(
        TExprNodeType::SLOT_REF,
        type_desc,
        None, // opcode
        0,    // num_children
        None, None, None, None, None, // agg, bool, case, date, float
        None, // int_literal
        None, None, None, None, // in, is_null, like, literal_pred
        TSlotRef::new(slot_id, tuple_id, None, None), // slot_ref
        None, None, None, None, // string, tuple_is_null, info_func, decimal
        0, // output_scale
        None, None, None, None, None, // fn_call, large_int, output_col, output_type, vector_opcode
        None, None, None, None, // fn_, vararg, child_type, is_nullable
        None, None, None, None, // json, schema_change, column_ref, match_pred
        None, None, None, None, None, // ipv4, ipv6, label, timev2, varbinary
        None, None, None, // is_cast_nullable, search_param, short_circuit
    )
}

/// Build an INT_LITERAL expression node.
fn int_literal_node(value: i64) -> TExprNode {
    TExprNode::new(
        TExprNodeType::INT_LITERAL,
        bigint_type_desc(),
        None, // opcode
        0,    // num_children
        None, None, None, None, None, // agg, bool, case, date, float
        TIntLiteral::new(value), // int_literal
        None, None, None, None, // in, is_null, like, literal_pred
        None, // slot_ref
        None, None, None, None, // string, tuple_is_null, info_func, decimal
        0, // output_scale
        None, None, None, None, None, // fn_call, large_int, output_col, output_type, vector_opcode
        None, None, None, None, // fn_, vararg, child_type, is_nullable
        None, None, None, None, // json, schema_change, column_ref, match_pred
        None, None, None, None, None, // ipv4, ipv6, label, timev2, varbinary
        None, None, None, // is_cast_nullable, search_param, short_circuit
    )
}

/// Build a BINARY_PRED expression node (comparison operator).
fn binary_pred_node(opcode: TExprOpcode) -> TExprNode {
    TExprNode::new(
        TExprNodeType::BINARY_PRED,
        bool_type_desc(),
        opcode, // opcode
        2,      // num_children
        None, None, None, None, None, // agg, bool, case, date, float
        None, // int_literal
        None, None, None, None, // in, is_null, like, literal_pred
        None, // slot_ref
        None, None, None, None, // string, tuple_is_null, info_func, decimal
        0, // output_scale
        None, None, None, None, None, // fn_call, large_int, output_col, output_type, vector_opcode
        None, None, None, None, // fn_, vararg, child_type, is_nullable
        None, None, None, None, // json, schema_change, column_ref, match_pred
        None, None, None, None, None, // ipv4, ipv6, label, timev2, varbinary
        None, None, None, // is_cast_nullable, search_param, short_circuit
    )
}

/// Build a FILE_SCAN_NODE TPlanNode.
fn file_scan_plan_node(
    node_id: i32,
    tuple_id: i32,
    table_name: &str,
    conjuncts: Option<Vec<TExpr>>,
) -> TPlanNode {
    TPlanNode::new(
        node_id,
        TPlanNodeType::FILE_SCAN_NODE,
        0, // num_children
        -1, // limit
        vec![tuple_id],
        vec![false],
        conjuncts,
        false, // compact_data
        None, None, None, None, None, // hash_join, agg, sort, merge, exchange
        None, None, None, None, None, // mysql, olap, csv, broker, pre_agg
        None, None, None, None, None, // schema, merge_join, meta, analytic, olap_rewrite
        None, None, None, None, None, // union, resource, es, repeat, assert
        None, None, None, None, None, // intersect, except, odbc, runtime_filters, group_commit
        None, None, None, // materialization, rec_cte, vconjunct
        None, None, None, // table_function, output_slot_ids, data_gen
        TFileScanNode::new(tuple_id, table_name.to_string()), // file_scan_node
        None, None, None, // jdbc, nested_loop, test_external
        None, None, None, None, None, // push_down_agg, push_down_count, distribute, serial, rec_cte_scan
        None, None, None, None, None, // projections, output_tuple_id, partition_sort, intermediate_proj, intermediate_tuple
        None, None, // topn_filter, nereids_id
    )
}

/// Construct a minimal TPipelineFragmentParams for testing.
fn make_fragment_params(
    plan_nodes: Vec<TPlanNode>,
    desc_tbl: TDescriptorTable,
) -> TPipelineFragmentParams {
    let plan = TPlan::new(plan_nodes);
    let partition = TDataPartition::new(TPartitionType::UNPARTITIONED, None, None);
    let fragment = TPlanFragment::new(plan, None, None, partition, None, None, None);

    TPipelineFragmentParams::new(
        PaloInternalServiceVersion::V1,
        TUniqueId::new(1, 1),
        1, // fragment_id
        BTreeMap::new(),
        desc_tbl, // desc_tbl
        None, None, None, None, None, // resource, destinations, num_senders, send_stats, coord
        None, None, None, None, None, // query_globals, query_options, import_label, db_name, load_job_id
        None, None, None, None, None, // load_error_hub, fragment_num, backend_id, need_wait, instances_sharing
        None, None, // is_simplified, global_dict
        fragment, // fragment
        None, None, None, None, // local_params, workload_groups, txn_conf, table_name
        None, None, None, None, None, // file_scan_params, group_commit, load_stream, total_load, num_local_sink
        None, None, None, None, None, // num_buckets, bucket_seq, per_node_shared, parallel_inst, total_inst
        None, None, None, None, None, // shuffle_idx, is_nereids, wal_id, content_length, current_fe
        None, None, None, None, // topn_filter, ai_resources, need_notify_close, is_mow_table
    )
}

fn make_desc_table() -> TDescriptorTable {
    let slot0 = TSlotDescriptor::new(
        0, // id (slot_id)
        0, // parent (tuple_id)
        bigint_type_desc(),
        0, // column_pos
        0, // byte_offset
        0, // null_indicator_byte
        0, // null_indicator_bit
        "col1".to_string(),
        0,    // slot_idx
        true, // is_materialized
        None, None, None, None, None, None, None, None, None, None,
    );

    let slot1 = TSlotDescriptor::new(
        1, // id
        0, // parent
        varchar_type_desc(),
        1, // column_pos
        8, // byte_offset
        0, // null_indicator_byte
        1, // null_indicator_bit
        "col2".to_string(),
        1,
        true,
        None, None, None, None, None, None, None, None, None, None,
    );

    let tuple0 = TTupleDescriptor::new(0, 16, 1, None, None);

    TDescriptorTable::new(vec![slot0, slot1], vec![tuple0], None)
}

// ---- Tests ----

#[test]
fn test_translate_scan_with_filter() {
    // Build: SELECT col1, col2 FROM test_table WHERE col1 > 100
    let conjunct = TExpr::new(vec![
        binary_pred_node(TExprOpcode::GT),
        slot_ref_node(0, 0, bigint_type_desc()),
        int_literal_node(100),
    ]);

    let scan_node = file_scan_plan_node(0, 0, "test_table", Some(vec![conjunct]));
    let desc_tbl = make_desc_table();
    let params = make_fragment_params(vec![scan_node], desc_tbl);

    let plan_bytes = plan_translator::translate_fragment(&params).expect("translation failed");
    assert!(!plan_bytes.is_empty(), "plan bytes should not be empty");

    // Decode the Substrait plan and verify structure.
    let plan = Plan::decode(plan_bytes.as_slice()).expect("failed to decode Substrait plan");

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
        other => panic!("expected FilterRel, got {:?}", std::mem::discriminant(other)),
    }

    // Check extensions (should have "gt" function registered).
    assert!(!plan.extensions.is_empty(), "should have extension functions");
}

#[test]
fn test_translate_scan_without_filter() {
    // Build: SELECT col1, col2 FROM test_table (no WHERE clause)
    let scan_node = file_scan_plan_node(0, 0, "test_table", None);
    let desc_tbl = make_desc_table();
    let params = make_fragment_params(vec![scan_node], desc_tbl);

    let plan_bytes = plan_translator::translate_fragment(&params).expect("translation failed");
    let plan = Plan::decode(plan_bytes.as_slice()).expect("failed to decode");

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
