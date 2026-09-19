-- sql/session-native.sql plus one switch: FE-side file splitting. Doris 4.1.4's default
-- (enable_file_scanner_v2 + file_split_size_on_be = 64 MB) ships one whole-file range per
-- parquet file to the BE and lets the BE cut it into scanner splits; for a local() TVF over a
-- single 162 MB (SF1) / 2.4 GB (SF10) lineitem file the BE ends up with one working scanner
-- (profile: 64 scanners, one with all the running time), so the scan is single-threaded.
-- file_split_size_on_be = 0 turns BE-side splitting off and the FE cuts the file into 32/64 MB
-- ranges as before 4.1 — 16 scanners, Q1 on SF1 1.4 s → 0.65 s. This is the benchmark's
-- "native-split" system (plan §6.3: not a tuning of the engine, a workaround for the single
-- scanner); the stock default is the "native" system.
SET GLOBAL parallel_pipeline_task_num = 0;
SET GLOBAL enable_local_shuffle = true;
SET GLOBAL enable_parallel_result_sink = true;
SET GLOBAL runtime_filter_mode = 'GLOBAL';
SET GLOBAL enable_cte_materialize = true;
SET GLOBAL topn_lazy_materialization_threshold = 1024;
SET GLOBAL file_split_size = 0;
SET GLOBAL enable_fold_constant_by_be = false;
SET GLOBAL enable_profile = false;
SET GLOBAL query_timeout = 3600;

SET GLOBAL file_split_size_on_be = 0;
SET GLOBAL file_split_size_on_fe = 536870912;

-- No FE result cache (4.1.4 default on): with it, a repeated query on internal tables is
-- answered from the FE in ~15 ms (HitSqlCache=true in the audit log) and the hot rounds of the
-- native-olap reference measure the cache, not the engine. local() queries never hit it.
SET GLOBAL enable_sql_cache = false;
