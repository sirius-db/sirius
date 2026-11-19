// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
use sirius::{Comparison, DataType, Expression, Field, Plan, PlanError, PlanNode, Schema};

fn main() -> Result<(), PlanError> {
    let schema = Schema::new(&[Field {
        name: "id",
        data_type: DataType::Int64,
        nullable: true,
    }])?;
    let scan = PlanNode::scan_parquet("data.parquet", &schema)?;
    let predicate = scan
        .column(0)?
        .compare(Comparison::Greater, &Expression::literal_int64(10)?)?;
    let plan = Plan::from_root(&scan.filter(&predicate)?)?;
    println!("{}", plan.describe());
    Ok(())
}
