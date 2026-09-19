-- Session variables for the benchmark's native Doris BE (plan-doc experiments/sf10-bench,
-- system A): every variable that sql/session.sql pins for the Sirius backend goes back to
-- its Doris 4.1.4 default (fe/fe-core/.../qe/SessionVariable.java), so Doris runs the way it
-- ships — the ClickBench "default configuration, no tuning" rule. Applied GLOBAL by
-- scripts/run-tpch.sh --session-sql before each native run (the GLOBAL values persist in the
-- FE's metadata, so the two files must always undo each other).

-- Parallelism: instances per fragment auto by core count, local shuffle on.
SET GLOBAL parallel_pipeline_task_num = 0;
SET GLOBAL enable_local_shuffle = true;

-- Parallel result sink (default on).
SET GLOBAL enable_parallel_result_sink = true;

-- Runtime filters as shipped (TVF scans do not consume them, catalog tables do).
SET GLOBAL runtime_filter_mode = 'GLOBAL';

-- CTE materialization and lazy top-n materialization as shipped.
SET GLOBAL enable_cte_materialize = true;
SET GLOBAL topn_lazy_materialization_threshold = 1024;

-- File splitting as shipped (0 = the BE's own split size).
SET GLOBAL file_split_size = 0;

-- Unchanged from the default (false) either way.
SET GLOBAL enable_fold_constant_by_be = false;
SET GLOBAL enable_profile = false;

-- The only non-default: the same generous timeout as sql/session.sql (default 900 s).
SET GLOBAL query_timeout = 3600;

-- Back to the default two-level file splitting (sql/session.sql turns the BE side off and
-- makes the FE side whole-file; sql/session-native-split.sql turns the BE side off only).
SET GLOBAL file_split_size_on_be = 67108864;
SET GLOBAL file_split_size_on_fe = 536870912;

-- No FE result cache (4.1.4 default on): with it, a repeated query on internal tables is
-- answered from the FE in ~15 ms (HitSqlCache=true in the audit log) and the hot rounds of the
-- native-olap reference measure the cache, not the engine. local() queries never hit it.
SET GLOBAL enable_sql_cache = false;
