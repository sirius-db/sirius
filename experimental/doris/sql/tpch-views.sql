-- TPC-H tables as views over parquet files read through the local() table-valued function
-- (ADR-011 D-4: local() + shared_storage=true, every backend sees the same paths).
-- @@TPCH_DIR@@ is substituted by scripts/run-tpch.sh with the dataset directory, whose
-- layout is <dir>/<table>/part.N.parquet (test/tpch_performance/generate_tpch_data.sh).

CREATE DATABASE IF NOT EXISTS tpch;
USE tpch;

CREATE VIEW IF NOT EXISTS nation AS
    SELECT * FROM local("file_path" = "@@TPCH_DIR@@/nation/*.parquet", "format" = "parquet", "shared_storage" = "true");
CREATE VIEW IF NOT EXISTS region AS
    SELECT * FROM local("file_path" = "@@TPCH_DIR@@/region/*.parquet", "format" = "parquet", "shared_storage" = "true");
CREATE VIEW IF NOT EXISTS part AS
    SELECT * FROM local("file_path" = "@@TPCH_DIR@@/part/*.parquet", "format" = "parquet", "shared_storage" = "true");
CREATE VIEW IF NOT EXISTS supplier AS
    SELECT * FROM local("file_path" = "@@TPCH_DIR@@/supplier/*.parquet", "format" = "parquet", "shared_storage" = "true");
CREATE VIEW IF NOT EXISTS partsupp AS
    SELECT * FROM local("file_path" = "@@TPCH_DIR@@/partsupp/*.parquet", "format" = "parquet", "shared_storage" = "true");
CREATE VIEW IF NOT EXISTS customer AS
    SELECT * FROM local("file_path" = "@@TPCH_DIR@@/customer/*.parquet", "format" = "parquet", "shared_storage" = "true");
CREATE VIEW IF NOT EXISTS orders AS
    SELECT * FROM local("file_path" = "@@TPCH_DIR@@/orders/*.parquet", "format" = "parquet", "shared_storage" = "true");
CREATE VIEW IF NOT EXISTS lineitem AS
    SELECT * FROM local("file_path" = "@@TPCH_DIR@@/lineitem/*.parquet", "format" = "parquet", "shared_storage" = "true");

-- Q15's revenue view, as in Doris's tools/tpch-tools/ddl (a view, not a CTE, so the
-- planner never needs enable_cte_materialize).
CREATE VIEW IF NOT EXISTS revenue0 (supplier_no, total_revenue) AS
SELECT
    l_suppkey,
    sum(l_extendedprice * (1 - l_discount))
FROM
    lineitem
WHERE
    l_shipdate >= date '1996-01-01'
    AND l_shipdate < date '1996-01-01' + interval '3' month
GROUP BY
    l_suppkey;
