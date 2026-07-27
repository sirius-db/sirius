-- TPC-H schema for loading classic dbgen .tbl output into DuckDB.
--
-- Types match what DuckDB's tpch extension creates, so a database built from
-- dbgen and one built by the extension are interchangeable. That means keys are
-- BIGINT rather than dss.ddl's INTEGER (orderkeys pass 2^31 above ~SF300) and
-- fixed-width text is VARCHAR, which is how DuckDB stores CHAR anyway.

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
    s_suppkey       BIGINT,
    s_name          VARCHAR,
    s_address       VARCHAR,
    s_nationkey     INTEGER,
    s_phone         VARCHAR,
    s_acctbal       DECIMAL(15,2),
    s_comment       VARCHAR
);

CREATE TABLE customer (
    c_custkey       BIGINT,
    c_name          VARCHAR,
    c_address       VARCHAR,
    c_nationkey     INTEGER,
    c_phone         VARCHAR,
    c_acctbal       DECIMAL(15,2),
    c_mktsegment    VARCHAR,
    c_comment       VARCHAR
);

CREATE TABLE part (
    p_partkey       BIGINT,
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
    ps_partkey      BIGINT,
    ps_suppkey      BIGINT,
    ps_availqty     BIGINT,
    ps_supplycost   DECIMAL(15,2),
    ps_comment      VARCHAR
);

CREATE TABLE orders (
    o_orderkey      BIGINT,
    o_custkey       BIGINT,
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
    l_partkey       BIGINT,
    l_suppkey       BIGINT,
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
