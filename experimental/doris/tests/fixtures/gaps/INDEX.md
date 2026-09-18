# Fragment corpus: sql/gaps

Captured from Doris FE 4.1.4 with sql/session.sql, dataset /tmp/tpch-sf1 (SF1 parquet).
Per query: `query.sql`, `explain.txt`, the raw dispatch (`batch-NN-request.tcompact`, a
TCompact `TPipelineFragmentParamsList` exactly as `exec_plan_fragment` received it) and
`batch-NN-summary.txt` (root fragment first). Pretty-print a dispatch with
`cargo run -p sirius-doris-be --no-default-features --bin dump-fragments -- <tcompact>`.

| query | fragments | shapes (root first) |
|---|---|---|
| g11-window | 2 | fragment_id=1 sink=RESULT_SINK nodes=[ANALYTIC_EVAL_NODE > EXCHANGE_NODE]<br>fragment_id=0 sink=DATA_STREAM_SINK nodes=[SORT_NODE > FILE_SCAN_NODE] |
| g12-union-all | 4 | fragment_id=4 sink=RESULT_SINK nodes=[EXCHANGE_NODE]<br>fragment_id=3 sink=DATA_STREAM_SINK nodes=[UNION_NODE > EXCHANGE_NODE > EXCHANGE_NODE]<br>fragment_id=2 sink=DATA_STREAM_SINK nodes=[FILE_SCAN_NODE]<br>fragment_id=0 sink=DATA_STREAM_SINK nodes=[FILE_SCAN_NODE] |
| g12-union-distinct | 5 | fragment_id=5 sink=RESULT_SINK nodes=[EXCHANGE_NODE]<br>fragment_id=4 sink=DATA_STREAM_SINK nodes=[AGGREGATION_NODE > EXCHANGE_NODE]<br>fragment_id=3 sink=DATA_STREAM_SINK nodes=[AGGREGATION_NODE > UNION_NODE > EXCHANGE_NODE > EXCHANGE_NODE]<br>fragment_id=2 sink=DATA_STREAM_SINK nodes=[FILE_SCAN_NODE]<br>fragment_id=0 sink=DATA_STREAM_SINK nodes=[FILE_SCAN_NODE] |
| g13-distinct-topn | 3 | fragment_id=2 sink=RESULT_SINK nodes=[EXCHANGE_NODE]<br>fragment_id=1 sink=DATA_STREAM_SINK nodes=[SORT_NODE > AGGREGATION_NODE > EXCHANGE_NODE]<br>fragment_id=0 sink=DATA_STREAM_SINK nodes=[AGGREGATION_NODE > FILE_SCAN_NODE] |
| g13-distinct | 3 | fragment_id=2 sink=RESULT_SINK nodes=[EXCHANGE_NODE]<br>fragment_id=1 sink=DATA_STREAM_SINK nodes=[AGGREGATION_NODE > EXCHANGE_NODE]<br>fragment_id=0 sink=DATA_STREAM_SINK nodes=[AGGREGATION_NODE > FILE_SCAN_NODE] |

## Coverage

- plan nodes: EXCHANGE_NODE (12), FILE_SCAN_NODE (7), AGGREGATION_NODE (6), UNION_NODE (2), SORT_NODE (2), ANALYTIC_EVAL_NODE (1)
- expression nodes: NULL_LITERAL (95), SLOT_REF (57), COMPOUND_PRED (5), FUNCTION_CALL (4), BOOL_LITERAL (1), BINARY_PRED (1), AGG_EXPR (1)
  (`NULL_LITERAL` is almost entirely `TFileScanRangeParams.default_value_of_src_slot`, one per scanned column)
- functions (scalar + aggregate, TFunctionName.function_name): is_null_pred (2), is_not_null_pred (2), row_number (1), eq (1)
- join ops: 
- sinks: DATA_STREAM_SINK (12), RESULT_SINK (5)
