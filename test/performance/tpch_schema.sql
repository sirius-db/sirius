-- TPC-H schema for loading classic dbgen .tbl output into DuckDB.
--
-- This file uses INTEGER keys for small datasets. Before loading .tbl files,
-- generate_tpch_data.sh runs tpch_schema.py to widen every primary and foreign
-- key in a family together when its full dataset range exceeds INTEGER.
-- The resulting widths match the adaptive defaults in sirius-db/tpchgen-rs.
-- Nation and region keys always fit INTEGER.
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
    o_orderkey      INTEGER,
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
    l_orderkey      INTEGER,
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
