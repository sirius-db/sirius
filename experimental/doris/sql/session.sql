-- Global session defaults that keep the FE's plan shapes inside what the Sirius backend
-- translates (pseudo-be-feasibility.md §3.3). Applied once per FE bootstrap by
-- scripts/run-tpch.sh; GLOBAL so every new connection inherits them.

-- One instance per fragment per backend (0 = auto by cores); the backend runs a fragment as
-- one Substrait plan, and this also keeps local shuffle out of the picture.
SET GLOBAL parallel_pipeline_task_num = 1;
SET GLOBAL enable_local_shuffle = false;

-- The result fragment gathers on one backend; fetch_data then has a single result stream
-- keyed by the result fragment's instance id (see result_store.rs).
SET GLOBAL enable_parallel_result_sink = false;

-- No runtime filters (TVF scans never produce them anyway; catalog tables would).
SET GLOBAL runtime_filter_mode = 'OFF';

-- No CTE materialization (MULTI_CAST_DATA_STREAM_SINK) and no lazy top-n materialization
-- (MATERIALIZATION_NODE): both are outside the translator's surface.
SET GLOBAL enable_cte_materialize = false;
SET GLOBAL topn_lazy_materialization_threshold = -1;

-- One split per parquet file: the engine reads whole files, byte ranges are ignored.
SET GLOBAL file_split_size = 1099511627776;

-- Keep the FE from asking the backend to fold constants or collect profiles.
SET GLOBAL enable_fold_constant_by_be = false;
SET GLOBAL enable_profile = false;

-- Generous timeout for large scale factors.
SET GLOBAL query_timeout = 3600;
