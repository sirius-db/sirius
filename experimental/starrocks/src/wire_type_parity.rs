//! Engine-derived conformance gate for the translator's wire-type contract: the partial-state
//! model (`starrocks-plan-translator`'s `partial_state::wire_type`) and the sink conformance
//! projection that casts every data-stream-sink fragment's output row to the FE-declared wire
//! types.
//!
//! The model is a Rust mirror of what the engine *binds* for each two-phase aggregate's partial
//! state, and the conformance projection exists because the translator cannot prove its own
//! view of an expression's type is what the engine binds (a `year()` the FE declared SMALLINT
//! binds BIGINT). Nothing mechanical kept either in agreement with the engine. This suite is
//! that mechanical link: for every model row (every supported two-phase aggregate × input type)
//! and for every expression-column row (the q08/q09 class), it translates the fragments of the
//! corresponding query with the real translator, builds each through the real engine build
//! path, and asserts on the engine's own answer (`Fragment::output_types`, the sink column
//! types the hop guard compares):
//!
//! - a sender fragment's engine-produced sink types must equal the receiving fragment's
//!   declared stream schema — exactly the comparison `relay_from`/`push_packed` make on a live
//!   hop;
//! - the MERGE fragment's engine-produced sink types must equal the schema a further downstream
//!   fragment declares from the FE's output slots — the finalizing projection's contract (a
//!   merged integer count/sum binds as HUGEINT inside the fragment and must leave as the
//!   declared BIGINT).
//!
//! A row that fails to translate or build FAILS this suite loudly rather than being skipped:
//! it means the contract predicts a wire type for a plan shape that cannot exist (allowlist too
//! broad) or the engine's binding drifted from the model. Either way the answer is a CI failure
//! here, not a refused hop in production.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Date32Array, Float64Array, Int16Array, RecordBatch, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use starrocks_plan_translator::{ExchangeInput, PlanTranslator, TranslatedPlan};
use starrocks_thrift::data_sinks::{TDataSink, TDataSinkType, TDataStreamSink};
use starrocks_thrift::descriptors::{TDescriptorTable, TSlotDescriptor, TTupleDescriptor};
use starrocks_thrift::exprs::{TAggregateExpr, TExpr, TExprNode, TExprNodeType, TSlotRef};
use starrocks_thrift::internal_service::{
    InternalServiceVersion, TExecPlanFragmentParams, TPlanFragmentExecParams, TScanRangeParams,
};
use starrocks_thrift::opcodes::TExprOpcode;
use starrocks_thrift::partitions::{TDataPartition, TPartitionType};
use starrocks_thrift::plan_nodes::{
    TAggregationNode, TBrokerRangeDesc, TBrokerScanRange, TBrokerScanRangeParams, TExchangeNode,
    TFileFormatType, TFileScanNode, TFileScanType, TPlan, TPlanNode, TPlanNodeType, TProjectNode,
    TScanRange,
};
use starrocks_thrift::planner::TPlanFragment;
use starrocks_thrift::types::{
    TFileType, TFunction, TFunctionBinaryType, TFunctionName, TPrimitiveType, TScalarType,
    TTypeDesc, TTypeNode, TTypeNodeType, TUniqueId,
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
    /// exists because this lies: DECIMAL128 for decimal sums).
    intermediate: TTypeDesc,
}

/// Every model row. Arm coverage:
/// - `sum` decimal arm (state FP64): decimal64 and decimal128 inputs;
/// - `sum` integer arm (state I64): every integer width the FE widens to a BIGINT sum;
/// - `sum` floating arm (state FP64): FLOAT and DOUBLE;
/// - `count` arm (state I64): star, a numeric column, a string column;
/// - `min`/`max` identity arm: every scalar type family the translator maps.
///
/// avg is absent by refusal, not omission: this branch does not model its two-column state, and
/// `two_phase_avg_is_refused_at_translation` pins that both avg-shaped fragments fail loudly.
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

/// Every wire-type row must hold for BOTH engine aggregation paths — the ungrouped
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

/// The demo's avg rows are not silently dropped from the domain: a two-phase avg ships a
/// two-column state this branch does not model, so BOTH avg-shaped fragments must refuse at
/// translation, loudly naming avg.
#[test]
fn two_phase_avg_is_refused_at_translation() {
    let row = Row {
        label: "avg(BIGINT)",
        function: "avg",
        arg: Some(scalar(TPrimitiveType::BIGINT)),
        ret: scalar(TPrimitiveType::DOUBLE),
        intermediate: scalar(TPrimitiveType::VARBINARY),
    };
    let translator = PlanTranslator::new();
    for grouped in [false, true] {
        let err = translator
            .translate_fragment_with_exchange_inputs(
                &partial_params(&row, grouped),
                &[exchange_input(input_names(grouped))],
            )
            .expect_err("the avg partial fragment must refuse translation");
        assert!(err.to_string().contains("avg"), "{err}");
        let err = translator
            .translate_fragment_with_exchange_inputs(
                &merge_params(&row, grouped),
                &[exchange_input(input_names(grouped))],
            )
            .expect_err("the avg merge fragment must refuse translation");
        assert!(err.to_string().contains("avg"), "{err}");
    }
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

/// One expression-column row of the conformance domain — the q08/q09 class: a projected
/// expression whose FE-declared slot type is not what the engine binds for the expression.
struct ExprRow {
    label: &'static str,
    /// Input tuple columns `(name, FE type)` arriving over the exchange, slots 1..=n of
    /// tuple 0.
    inputs: Vec<(&'static str, TTypeDesc)>,
    /// The projected expression over the input tuple, in flat preorder form.
    expr: TExpr,
    /// FE-declared type of the expression's output slot.
    declared: TTypeDesc,
}

/// The expression rows. Each pairs a translator lowering with the engine binding it produces:
/// `year()` binds BIGINT against a SMALLINT slot (the exact q08/q09 `o_year` column), a
/// SMALLINT addition binds SMALLINT against an INT slot, and a decimal addition is lowered to
/// FP64 against a DECIMAL64(13,2) slot.
fn expr_rows() -> Vec<ExprRow> {
    let dec12 = || decimal(TPrimitiveType::DECIMAL64, 12, 2);
    vec![
        ExprRow {
            label: "year(DATE) declared SMALLINT",
            inputs: vec![("d", scalar(TPrimitiveType::DATE))],
            expr: year_call(
                slot_ref(1, 0, scalar(TPrimitiveType::DATE)),
                scalar(TPrimitiveType::SMALLINT),
            ),
            declared: scalar(TPrimitiveType::SMALLINT),
        },
        ExprRow {
            label: "add(SMALLINT, SMALLINT) declared INT",
            inputs: vec![
                ("a", scalar(TPrimitiveType::SMALLINT)),
                ("b", scalar(TPrimitiveType::SMALLINT)),
            ],
            expr: add_expr(
                slot_ref(1, 0, scalar(TPrimitiveType::SMALLINT)),
                slot_ref(2, 0, scalar(TPrimitiveType::SMALLINT)),
                scalar(TPrimitiveType::INT),
            ),
            declared: scalar(TPrimitiveType::INT),
        },
        ExprRow {
            label: "add(DECIMAL64(12,2), DECIMAL64(12,2)) declared DECIMAL64(13,2)",
            inputs: vec![("a", dec12()), ("b", dec12())],
            expr: add_expr(
                slot_ref(1, 0, dec12()),
                slot_ref(2, 0, dec12()),
                decimal(TPrimitiveType::DECIMAL64, 13, 2),
            ),
            declared: decimal(TPrimitiveType::DECIMAL64, 13, 2),
        },
    ]
}

/// Every expression row must conform at the sink in BOTH sender shapes: a project-rooted
/// fragment (the expression column ships directly) and a partial-aggregation fragment grouped
/// by the expression column (the exact q09 leaf shape). Both hops of the aggregated shape are
/// checked through the real engine.
#[test]
fn expression_wire_types_conform_at_the_sink() {
    let _guard = crate::GPU_ENGINE_TEST_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let context = sirius::SiriusContext::new().expect("bring up the sirius engine context");

    let mut failures = Vec::new();
    for row in expr_rows() {
        if let Err(reason) = check_expr_row_projected(&context, &row) {
            failures.push(format!("{} (project-rooted): {reason}", row.label));
        }
        if let Err(reason) = check_expr_row_aggregated(&context, &row) {
            failures.push(format!("{} (partial-agg-keyed): {reason}", row.label));
        }
    }
    assert!(
        failures.is_empty(),
        "expression wire types and the engine disagree on {} case(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Project-rooted sender: EXCHANGE(input tuple) -> PROJECT(expr into the declared slot) with a
/// stream sink; the receiver is a bare bound exchange over the project tuple.
fn check_expr_row_projected(context: &sirius::SiriusContext, row: &ExprRow) -> Result<(), String> {
    let translator = PlanTranslator::new();
    let names = row
        .inputs
        .iter()
        .map(|(name, _)| name.to_string())
        .collect::<Vec<_>>();
    let sender = translator
        .translate_fragment_with_exchange_inputs(
            &projected_expr_params(row),
            &[exchange_input(names)],
        )
        .map_err(|err| format!("the sender failed to translate: {err}"))?;
    let consumer = translator
        .translate_fragment_with_exchange_inputs(
            &bound_exchange_params(1, vec![slot(10, 1, "e", row.declared.clone())]),
            &[exchange_input(sender.output_names.clone())],
        )
        .map_err(|err| format!("the consumer failed to translate: {err}"))?;

    let produced = engine_sink_types(context, &sender)
        .map_err(|err| format!("the sender failed to build: {err}"))?;
    let declared = declared_stream_types(&consumer);
    if produced != declared {
        return Err(format!(
            "the sink produces {produced:?} but the receiver declares {declared:?}"
        ));
    }
    Ok(())
}

/// Partial-agg-keyed sender (the q09 leaf shape): EXCHANGE -> PROJECT(expr slot + a DOUBLE
/// value column) -> partial AGG grouped by the expression slot, with a stream sink; then the
/// merge fragment over it, itself stream-sinked to a bare consumer. Both hops are checked.
fn check_expr_row_aggregated(context: &sirius::SiriusContext, row: &ExprRow) -> Result<(), String> {
    let translator = PlanTranslator::new();
    let names = row
        .inputs
        .iter()
        .map(|(name, _)| name.to_string())
        .chain(std::iter::once("v".to_string()))
        .collect::<Vec<_>>();
    let partial = translator
        .translate_fragment_with_exchange_inputs(
            &aggregated_expr_partial_params(row),
            &[exchange_input(names)],
        )
        .map_err(|err| format!("the partial fragment failed to translate: {err}"))?;
    let merge = translator
        .translate_fragment_with_exchange_inputs(
            &aggregated_expr_merge_params(row),
            &[exchange_input(partial.output_names.clone())],
        )
        .map_err(|err| format!("the merge fragment failed to translate: {err}"))?;
    let consumer = translator
        .translate_fragment_with_exchange_inputs(
            &bound_exchange_params(
                3,
                vec![
                    slot(10, 3, "k2", row.declared.clone()),
                    slot(31, 3, "r", scalar(TPrimitiveType::DOUBLE)),
                ],
            ),
            &[exchange_input(merge.output_names.clone())],
        )
        .map_err(|err| format!("the consumer failed to translate: {err}"))?;

    let produced = engine_sink_types(context, &partial)
        .map_err(|err| format!("the partial fragment failed to build: {err}"))?;
    let declared = declared_stream_types(&merge);
    if produced != declared {
        return Err(format!(
            "the partial sink produces {produced:?} but the merge stream declares {declared:?}"
        ));
    }

    let produced = engine_sink_types(context, &merge)
        .map_err(|err| format!("the merge fragment failed to build: {err}"))?;
    let declared = declared_stream_types(&consumer);
    if produced != declared {
        return Err(format!(
            "the merge sink produces {produced:?} but a downstream fragment declares \
             {declared:?}"
        ));
    }
    Ok(())
}

/// The execution pin for the conformance cast kernels: the exact q09 leaf — a parquet scan,
/// a projected `year()` into a SMALLINT slot, a grouped partial sum — must sink the
/// FE-declared wire row, pass the relay's schema guard, and merge to exact values with the
/// key arriving as Int16.
#[test]
fn q09_shaped_year_key_survives_the_hop_with_correct_values() {
    let _guard = crate::GPU_ENGINE_TEST_LOCK
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let context = sirius::SiriusContext::new().expect("bring up the sirius engine context");

    let dir = tempfile::tempdir().unwrap();
    let orders = dir.path().join("orders.parquet");
    write_orders_parquet(&orders);

    let translator = PlanTranslator::new();
    let leaf = translator
        .translate_fragment(&q09_leaf_params(orders.to_str().unwrap()))
        .expect("translate the q09-shaped leaf");
    let merge = translator
        .translate_fragment_with_exchange_inputs(
            &q09_merge_params(),
            &[exchange_input(leaf.output_names.clone())],
        )
        .expect("translate the q09-shaped merge");

    // The leaf must leave in exactly the wire row the merge declares its stream with — the
    // comparison relay_from makes before moving any batch.
    let expected_wire = vec![
        "VARCHAR".to_string(),
        "SMALLINT".to_string(),
        "DOUBLE".to_string(),
    ];
    assert_eq!(declared_stream_types(&merge), expected_wire);
    let mut sender = context.fragment().unwrap();
    sender.declare_output(0).unwrap();
    sender.build(&leaf.to_substrait_bytes()).unwrap();
    assert_eq!(sender.output_types().unwrap(), expected_wire);
    sender.run().expect("run the conformed leaf on the GPU");

    let mut receiver = context.fragment().unwrap();
    for column in &merge.stream_inputs[0].columns {
        receiver
            .declare_input_column(EXCHANGE_NODE_ID as u64, &column.name, &column.ty)
            .unwrap();
    }
    receiver.build(&merge.to_substrait_bytes()).unwrap();
    receiver
        .relay_from(&mut sender, 0, EXCHANGE_NODE_ID as u64, 0)
        .expect("the hop must pass the relay schema guard");
    receiver.run().expect("run the merge on the GPU");
    let result = receiver.result_to_arrow().expect("collect the merge rows");

    assert_eq!(result.schema.field(1).data_type(), &DataType::Int16);
    let mut rows = Vec::new();
    for batch in &result.batches {
        let nations = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("nation column is Utf8");
        let years = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int16Array>()
            .expect("o_year column is Int16");
        let sums = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("sum column is Float64");
        for index in 0..batch.num_rows() {
            rows.push((
                nations.value(index).to_string(),
                years.value(index),
                sums.value(index),
            ));
        }
    }
    rows.sort_by(|left, right| left.partial_cmp(right).unwrap());
    assert_eq!(
        rows,
        vec![
            ("BRAZIL".to_string(), 1995, 15.0),
            ("BRAZIL".to_string(), 1996, 7.0),
            ("PERU".to_string(), 1995, 2.0),
        ]
    );
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
/// stream. Tuple 0 is the input row, tuple 1 the FE's partial output. Carries a stream sink:
/// the sink conformance projection only fires on data-stream-sink fragments, which is also the
/// only shape a partial fragment has in production.
fn partial_params(row: &Row, grouped: bool) -> TExecPlanFragmentParams {
    // count(*) has no argument, but the input tuple still needs a column to ship.
    let arg_type = row
        .arg
        .clone()
        .unwrap_or_else(|| scalar(TPrimitiveType::BIGINT));
    let mut slots = Vec::new();
    let mut grouping = Vec::new();
    if grouped {
        slots.push(slot(1, 0, "k", scalar(TPrimitiveType::VARCHAR)));
        grouping.push(slot_ref(1, 0, scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(2, 0, "v", arg_type.clone()));
    if grouped {
        slots.push(slot(10, 1, "k", scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(11, 1, "s", row.intermediate.clone()));

    let measure = aggregate_expr(
        row.function,
        row.ret.clone(),
        row.arg.as_ref().map(|ty| slot_ref(2, 0, ty.clone())),
        false,
    );
    let mut aggregate = aggregation_node(1, 1, grouping, vec![measure]);
    aggregate.agg_node.as_mut().unwrap().need_finalize = false;
    with_stream_sink(params(
        TPlan::new(vec![aggregate, exchange_node(0)]),
        desc_table(vec![0, 1], slots),
    ))
}

/// The merge fragment: a merge-phase aggregation (need_finalize = true, is_merge_agg = true)
/// over an exchange carrying the intermediate row, itself stream-sinked to a further hop.
/// Tuple 2 is the FE's intermediate row, tuple 3 the FE's final output.
fn merge_params(row: &Row, grouped: bool) -> TExecPlanFragmentParams {
    let mut slots = Vec::new();
    let mut grouping = Vec::new();
    if grouped {
        slots.push(slot(20, 2, "k", scalar(TPrimitiveType::VARCHAR)));
        grouping.push(slot_ref(20, 2, scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(21, 2, "s", row.intermediate.clone()));
    if grouped {
        slots.push(slot(30, 3, "k", scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(31, 3, "r", row.ret.clone()));

    // A merge measure always reads its own state slot, count(*) included.
    let measure = aggregate_expr(
        row.function,
        row.ret.clone(),
        Some(slot_ref(21, 2, row.intermediate.clone())),
        true,
    );
    let aggregate = aggregation_node(8, 3, grouping, vec![measure]);
    with_stream_sink(params(
        TPlan::new(vec![aggregate, exchange_node(2)]),
        desc_table(vec![2, 3], slots),
    ))
}

/// A downstream fragment that just reads the merge fragment's output: a bare bound exchange
/// over the FE output tuple, whose declared stream schema is what hop 2 must satisfy. Sink-less
/// on purpose — only its declared stream is read.
fn consumer_params(row: &Row, grouped: bool) -> TExecPlanFragmentParams {
    let mut slots = Vec::new();
    if grouped {
        slots.push(slot(30, 3, "k", scalar(TPrimitiveType::VARCHAR)));
    }
    slots.push(slot(31, 3, "r", row.ret.clone()));
    bound_exchange_params(3, slots)
}

/// A sink-less fragment that is just a bound exchange over `tuple_id`.
fn bound_exchange_params(tuple_id: i32, slots: Vec<TSlotDescriptor>) -> TExecPlanFragmentParams {
    params(
        TPlan::new(vec![exchange_node(tuple_id)]),
        desc_table(vec![tuple_id], slots),
    )
}

/// Project-rooted sender for one expression row: EXCHANGE(tuple 0) -> PROJECT(tuple 1: the
/// expression into its declared slot), with a stream sink.
fn projected_expr_params(row: &ExprRow) -> TExecPlanFragmentParams {
    let mut slots = input_slots(row);
    slots.push(slot(10, 1, "e", row.declared.clone()));
    let mut slot_map = BTreeMap::new();
    slot_map.insert(10, row.expr.clone());
    with_stream_sink(params(
        TPlan::new(vec![project_node(1, 1, slot_map), exchange_node(0)]),
        desc_table(vec![0, 1], slots),
    ))
}

/// Partial-agg-keyed sender for one expression row: EXCHANGE(tuple 0, inputs + v DOUBLE) ->
/// PROJECT(tuple 1: k2 = expr, v) -> partial AGG(tuple 2: group by k2, sum(v)), with a stream
/// sink.
fn aggregated_expr_partial_params(row: &ExprRow) -> TExecPlanFragmentParams {
    let mut slots = input_slots(row);
    let value_slot = row.inputs.len() as i32 + 1;
    slots.push(slot(value_slot, 0, "v", scalar(TPrimitiveType::DOUBLE)));
    slots.push(slot(10, 1, "k2", row.declared.clone()));
    slots.push(slot(11, 1, "v", scalar(TPrimitiveType::DOUBLE)));
    slots.push(slot(10, 2, "k2", row.declared.clone()));
    slots.push(slot(21, 2, "s", scalar(TPrimitiveType::DOUBLE)));

    let mut slot_map = BTreeMap::new();
    slot_map.insert(10, row.expr.clone());
    slot_map.insert(11, slot_ref(value_slot, 0, scalar(TPrimitiveType::DOUBLE)));

    let measure = aggregate_expr(
        "sum",
        scalar(TPrimitiveType::DOUBLE),
        Some(slot_ref(11, 1, scalar(TPrimitiveType::DOUBLE))),
        false,
    );
    let mut aggregate = aggregation_node(
        2,
        2,
        vec![slot_ref(10, 1, row.declared.clone())],
        vec![measure],
    );
    aggregate.agg_node.as_mut().unwrap().need_finalize = false;
    with_stream_sink(params(
        TPlan::new(vec![
            aggregate,
            project_node(1, 1, slot_map),
            exchange_node(0),
        ]),
        desc_table(vec![0, 1, 2], slots),
    ))
}

/// The merge half over [`aggregated_expr_partial_params`]'s output: EXCHANGE(tuple 2) -> merge
/// AGG(tuple 3: group by k2, sum(s)), with a stream sink.
fn aggregated_expr_merge_params(row: &ExprRow) -> TExecPlanFragmentParams {
    let slots = vec![
        slot(10, 2, "k2", row.declared.clone()),
        slot(21, 2, "s", scalar(TPrimitiveType::DOUBLE)),
        slot(10, 3, "k2", row.declared.clone()),
        slot(31, 3, "r", scalar(TPrimitiveType::DOUBLE)),
    ];
    let measure = aggregate_expr(
        "sum",
        scalar(TPrimitiveType::DOUBLE),
        Some(slot_ref(21, 2, scalar(TPrimitiveType::DOUBLE))),
        true,
    );
    let aggregate = aggregation_node(
        8,
        3,
        vec![slot_ref(10, 2, row.declared.clone())],
        vec![measure],
    );
    with_stream_sink(params(
        TPlan::new(vec![aggregate, exchange_node(2)]),
        desc_table(vec![2, 3], slots),
    ))
}

/// One slot per expression-row input column, slots 1..=n of tuple 0.
fn input_slots(row: &ExprRow) -> Vec<TSlotDescriptor> {
    row.inputs
        .iter()
        .enumerate()
        .map(|(index, (name, ty))| slot(index as i32 + 1, 0, name, ty.clone()))
        .collect()
}

/// The q09-shaped leaf: FILE_SCAN(tuple 0: nation, o_orderdate, amount) -> PROJECT(tuple 1:
/// nation, o_year = year(o_orderdate) SMALLINT, amount) -> partial AGG(tuple 2: keys nation +
/// o_year, sum(amount)), with a stream sink.
fn q09_leaf_params(parquet: &str) -> TExecPlanFragmentParams {
    let desc = desc_table(
        vec![0, 1, 2],
        vec![
            slot(1, 0, "nation", scalar(TPrimitiveType::VARCHAR)),
            slot(2, 0, "o_orderdate", scalar(TPrimitiveType::DATE)),
            slot(3, 0, "amount", scalar(TPrimitiveType::DOUBLE)),
            slot(10, 1, "nation", scalar(TPrimitiveType::VARCHAR)),
            slot(11, 1, "o_year", scalar(TPrimitiveType::SMALLINT)),
            slot(12, 1, "amount", scalar(TPrimitiveType::DOUBLE)),
            slot(10, 2, "nation", scalar(TPrimitiveType::VARCHAR)),
            slot(11, 2, "o_year", scalar(TPrimitiveType::SMALLINT)),
            slot(22, 2, "sum_amount", scalar(TPrimitiveType::DOUBLE)),
        ],
    );
    let mut slot_map = BTreeMap::new();
    slot_map.insert(10, slot_ref(1, 0, scalar(TPrimitiveType::VARCHAR)));
    slot_map.insert(
        11,
        year_call(
            slot_ref(2, 0, scalar(TPrimitiveType::DATE)),
            scalar(TPrimitiveType::SMALLINT),
        ),
    );
    slot_map.insert(12, slot_ref(3, 0, scalar(TPrimitiveType::DOUBLE)));

    let measure = aggregate_expr(
        "sum",
        scalar(TPrimitiveType::DOUBLE),
        Some(slot_ref(12, 1, scalar(TPrimitiveType::DOUBLE))),
        false,
    );
    let mut aggregate = aggregation_node(
        2,
        2,
        vec![
            slot_ref(10, 1, scalar(TPrimitiveType::VARCHAR)),
            slot_ref(11, 1, scalar(TPrimitiveType::SMALLINT)),
        ],
        vec![measure],
    );
    aggregate.agg_node.as_mut().unwrap().need_finalize = false;
    let plan = TPlan::new(vec![
        aggregate,
        project_node(1, 1, slot_map),
        scan_node(0, 0),
    ]);
    with_stream_sink(with_scan_range(
        params(plan, desc),
        0,
        broker_scan_range(parquet),
    ))
}

/// The q09-shaped merge over the leaf's stream: EXCHANGE(tuple 2) -> merge AGG(tuple 3).
/// Sink-less: it is the RESULT fragment of the value test, collected as Arrow.
fn q09_merge_params() -> TExecPlanFragmentParams {
    let desc = desc_table(
        vec![2, 3],
        vec![
            slot(10, 2, "nation", scalar(TPrimitiveType::VARCHAR)),
            slot(11, 2, "o_year", scalar(TPrimitiveType::SMALLINT)),
            slot(22, 2, "sum_amount", scalar(TPrimitiveType::DOUBLE)),
            slot(10, 3, "nation", scalar(TPrimitiveType::VARCHAR)),
            slot(11, 3, "o_year", scalar(TPrimitiveType::SMALLINT)),
            slot(32, 3, "sum_amount", scalar(TPrimitiveType::DOUBLE)),
        ],
    );
    let measure = aggregate_expr(
        "sum",
        scalar(TPrimitiveType::DOUBLE),
        Some(slot_ref(22, 2, scalar(TPrimitiveType::DOUBLE))),
        true,
    );
    let aggregate = aggregation_node(
        8,
        3,
        vec![
            slot_ref(10, 2, scalar(TPrimitiveType::VARCHAR)),
            slot_ref(11, 2, scalar(TPrimitiveType::SMALLINT)),
        ],
        vec![measure],
    );
    params(TPlan::new(vec![aggregate, exchange_node(2)]), desc)
}

/// Writes the value-test parquet fixture: nation/date/amount rows spanning two BRAZIL years
/// and one PERU row. Date32 is days since epoch (1995-01-01 = 9131, 1996-01-01 = 9496).
fn write_orders_parquet(path: &std::path::Path) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("nation", DataType::Utf8, false),
        Field::new("o_orderdate", DataType::Date32, false),
        Field::new("amount", DataType::Float64, false),
    ]));
    let nations: ArrayRef = Arc::new(StringArray::from(vec![
        "BRAZIL", "BRAZIL", "BRAZIL", "PERU",
    ]));
    // 1995-03-15, 1995-07-01, 1996-01-02, 1995-12-31.
    let dates: ArrayRef = Arc::new(Date32Array::from(vec![
        9131 + 73,
        9131 + 181,
        9496 + 1,
        9131 + 364,
    ]));
    let amounts: ArrayRef = Arc::new(Float64Array::from(vec![10.0, 5.0, 7.0, 2.0]));
    let batch = RecordBatch::try_new(schema.clone(), vec![nations, dates, amounts]).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
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

/// Builds a materialized slot descriptor owned by a test tuple. `column_pos` is always -1: the
/// FE sets it unconditionally and the IDL marks it deprecated, so a fixture carrying a real
/// position would be a shape the translator never sees.
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

/// `year(<child>)` FUNCTION_CALL declared with `declared` (the FE says SMALLINT).
fn year_call(child: TExpr, declared: TTypeDesc) -> TExpr {
    let mut node = base_expr_node(TExprNodeType::FUNCTION_CALL, declared.clone(), 1);
    node.fn_ = Some(builtin_function("year", declared));
    let mut nodes = vec![node];
    nodes.extend(child.nodes);
    TExpr::new(nodes)
}

/// `<left> + <right>` ARITHMETIC_EXPR declared with `declared`.
fn add_expr(left: TExpr, right: TExpr, declared: TTypeDesc) -> TExpr {
    let mut node = base_expr_node(TExprNodeType::ARITHMETIC_EXPR, declared, 2);
    node.opcode = Some(TExprOpcode::ADD);
    let mut nodes = vec![node];
    nodes.extend(left.nodes);
    nodes.extend(right.nodes);
    TExpr::new(nodes)
}

/// An aggregate measure in flat preorder form, with the phase flag the classifier reads.
fn aggregate_expr(name: &str, ret_type: TTypeDesc, child: Option<TExpr>, is_merge: bool) -> TExpr {
    let num_children = child.as_ref().map(|_| 1).unwrap_or(0);
    let mut node = base_expr_node(TExprNodeType::AGG_EXPR, ret_type.clone(), num_children);
    node.agg_expr = Some(TAggregateExpr::new(is_merge));
    node.fn_ = Some(builtin_function(name, ret_type));
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

fn scan_node(node_id: i32, tuple_id: i32) -> TPlanNode {
    let mut node = base_plan_node(node_id, TPlanNodeType::FILE_SCAN_NODE, 0, vec![tuple_id]);
    node.file_scan_node = Some(TFileScanNode::new(tuple_id, None, None, None));
    node
}

fn project_node(node_id: i32, output_tuple: i32, slot_map: BTreeMap<i32, TExpr>) -> TPlanNode {
    let mut node = base_plan_node(node_id, TPlanNodeType::PROJECT_NODE, 1, vec![output_tuple]);
    node.project_node = Some(TProjectNode::new(Some(slot_map), None));
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

/// Attaches an unpartitioned data-stream sink, making the fragment a wire sender: the sink
/// conformance projection only fires on data-stream-sink fragments.
fn with_stream_sink(mut fragment_params: TExecPlanFragmentParams) -> TExecPlanFragmentParams {
    fragment_params.fragment.as_mut().unwrap().output_sink = Some(TDataSink::new(
        TDataSinkType::DATA_STREAM_SINK,
        TDataStreamSink::new(
            EXCHANGE_NODE_ID,
            TDataPartition::new(TPartitionType::UNPARTITIONED, None, None, None),
            None,
            None,
            None,
            None,
            None,
        ),
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
    fragment_params
}

/// A single local broker scan range for `path`: a FILES() query read with direct access, the
/// production-shaped supported slice.
fn broker_scan_range(path: &str) -> TScanRange {
    let file_size = std::fs::metadata(path).unwrap().len() as i64;
    let range = TBrokerRangeDesc::new(
        TFileType::FILE_BROKER,
        TFileFormatType::FORMAT_PARQUET,
        false,
        path.to_string(),
        0,
        -1,
        None,
        Some(file_size),
        None,
        None,
        None,
        None,
        None,
        None,
    );
    let mut range_params = TBrokerScanRangeParams::new(
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
    range_params.file_scan_type = Some(TFileScanType::FILES_QUERY);
    range_params.use_broker = Some(false);
    let broker = TBrokerScanRange::new(
        vec![range],
        range_params,
        Vec::new(),
        None,
        None,
        None,
        None,
    );
    TScanRange::new(None, None, Some(broker), None, None, None)
}

/// Attaches `scan_range` to the fragment's `node_id` scan.
fn with_scan_range(
    mut fragment_params: TExecPlanFragmentParams,
    node_id: i32,
    scan_range: TScanRange,
) -> TExecPlanFragmentParams {
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
