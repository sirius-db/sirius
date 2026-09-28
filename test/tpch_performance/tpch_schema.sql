-- TPC-H schema for loading classic dbgen .tbl output into DuckDB.
--
-- Key widths follow spec clause 1.3.1: an identifier must hold every key value
-- generated for its column and support at least 2,147,483,647 unique values.
-- Orderkeys are sparsely populated across 6,000,000 * SF, so they exceed 2^31
-- above ~SF358 and must be BIGINT. Every other key is densely populated and
-- stays inside INTEGER well past SF1000 (partkey SF*200,000, custkey
-- SF*150,000, suppkey SF*10,000). Clause 1.3.1 exempts identifier columns from
-- the datatype-consistency rule so they can be sized individually.
--
-- This matches sirius-db/tpchgen-rs, which emits Int32 for every key except
-- l_orderkey/o_orderkey, so the parquet and duckdb datasets share one key
-- layout. It diverges from DuckDB's tpch extension, which makes all keys
-- BIGINT: a database built here is no longer column-type-interchangeable with
-- one built by CALL dbgen(). Narrower keys are what let the SF1000 working set
-- fit -- l_partkey and l_suppkey alone are 48 GB smaller at that scale.
--
-- Fixed-width text is VARCHAR, which is how DuckDB stores CHAR anyway.

CREATE TABLE region (
    r_regionkey     INTEGER,
    r_name          VARCHAR,
    r_comment       VARCHAR
);

CREATE TABLE nation (
    n_nationkey     INTEGER,
    n_name          VARCHAR,
    n_regionkey     INTEGER,
    n_comment       VARCHAR
);

CREATE TABLE supplier (
    s_suppkey       INTEGER,
    s_name          VARCHAR,
    s_address       VARCHAR,
    s_nationkey     INTEGER,
    s_phone         VARCHAR,
    s_acctbal       DECIMAL(15,2),
    s_comment       VARCHAR
);

CREATE TABLE customer (
    c_custkey       INTEGER,
    c_name          VARCHAR,
    c_address       VARCHAR,
    c_nationkey     INTEGER,
    c_phone         VARCHAR,
    c_acctbal       DECIMAL(15,2),
    c_mktsegment    VARCHAR,
    c_comment       VARCHAR
);

CREATE TABLE part (
    p_partkey       INTEGER,
    p_name          VARCHAR,
    p_mfgr          VARCHAR,
    p_brand         VARCHAR,
    p_type          VARCHAR,
    p_size          INTEGER,
    p_container     VARCHAR,
    p_retailprice   DECIMAL(15,2),
    p_comment       VARCHAR
);

CREATE TABLE partsupp (
    ps_partkey      INTEGER,
    ps_suppkey      INTEGER,
    ps_availqty     BIGINT,
    ps_supplycost   DECIMAL(15,2),
    ps_comment      VARCHAR
);

CREATE TABLE orders (
    o_orderkey      BIGINT,
    o_custkey       INTEGER,
    o_orderstatus   VARCHAR,
    o_totalprice    DECIMAL(15,2),
    o_orderdate     DATE,
    o_orderpriority VARCHAR,
    o_clerk         VARCHAR,
    o_shippriority  INTEGER,
    o_comment       VARCHAR
);

CREATE TABLE lineitem (
    l_orderkey      BIGINT,
    l_partkey       INTEGER,
    l_suppkey       INTEGER,
    l_linenumber    BIGINT,
    l_quantity      DECIMAL(15,2),
    l_extendedprice DECIMAL(15,2),
    l_discount      DECIMAL(15,2),
    l_tax           DECIMAL(15,2),
    l_returnflag    VARCHAR,
    l_linestatus    VARCHAR,
    l_shipdate      DATE,
    l_commitdate    DATE,
    l_receiptdate   DATE,
    l_shipinstruct  VARCHAR,
    l_shipmode      VARCHAR,
    l_comment       VARCHAR
);
