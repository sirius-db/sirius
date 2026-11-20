// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#![cfg(not(miri))]
use sirius::{
    Comparison, DataType, Expression, Field, Plan, PlanError, PlanNode, PlanNodeKind, Schema,
};

#[test]
fn retained_children_and_schema_survive_their_parents() {
    let field = Field {
        name: "id",
        data_type: DataType::Int64,
        nullable: true,
    };
    let schema = Schema::new(&[field]).unwrap();
    let scan = PlanNode::scan_parquet("missing.parquet", &schema).unwrap();
    let rhs = Expression::literal_int64(10).unwrap();
    let predicate = scan
        .column(0)
        .unwrap()
        .compare(Comparison::Greater, &rhs)
        .unwrap();
    assert_eq!(predicate.data_type(), DataType::Boolean);
    assert!(predicate.nullable());
    let plan = Plan::from_root(&scan.filter(&predicate).unwrap()).unwrap();
    assert!(plan.describe().contains("missing.parquet"));
    let root = plan.root();
    drop((plan, predicate, rhs, scan, schema));
    assert_eq!(root.kind(), PlanNodeKind::Filter);
    let input = root.input().unwrap();
    drop(root);
    assert_eq!(input.kind(), PlanNodeKind::ParquetScan);
    assert!(input.input().is_none());
    let schema = input.schema();
    drop(input);
    assert_eq!(schema.field_count(), 1);
    assert_eq!(schema.field(0).unwrap(), field);
    assert!(schema.field(1).is_err());
}

#[test]
fn columns_are_scoped_to_the_node_that_created_them() {
    let schema = Schema::new(&[Field {
        name: "flag",
        data_type: DataType::Boolean,
        nullable: false,
    }])
    .unwrap();
    let left = PlanNode::scan_parquet("a.parquet", &schema).unwrap();
    let right = PlanNode::scan_parquet("a.parquet", &schema).unwrap();
    let predicate = left
        .column(0)
        .unwrap()
        .compare(
            Comparison::Equal,
            &Expression::literal_boolean(true).unwrap(),
        )
        .unwrap();
    assert!(
        matches!(right.filter(&predicate), Err(PlanError::InvalidArgument(message)) if !message.is_empty())
    );
}
