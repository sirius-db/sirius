DROP TABLE IF EXISTS nation;
DROP TABLE IF EXISTS region;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS supplier;
DROP TABLE IF EXISTS partsupp;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS lineitem;

CREATE TABLE nation  ( n_nationkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                       n_name       CHAR(25) NOT NULL,
                       n_regionkey  INTEGER NOT NULL,
                       n_comment    VARCHAR(152));

CREATE TABLE region  ( r_regionkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                       r_name       CHAR(25) NOT NULL,
                       r_comment    VARCHAR(152));

CREATE TABLE part  ( p_partkey     BIGINT NOT NULL UNIQUE PRIMARY KEY,
                     p_name        VARCHAR(55) NOT NULL,
                     p_mfgr        CHAR(25) NOT NULL,
                     p_brand       CHAR(10) NOT NULL,
                     p_type        VARCHAR(25) NOT NULL,
                     p_size        INTEGER NOT NULL,
                     p_container   CHAR(10) NOT NULL,
                     p_retailprice DECIMAL(15,2) NOT NULL,
                     p_comment     VARCHAR(23) NOT NULL );

CREATE TABLE supplier ( s_suppkey     BIGINT NOT NULL UNIQUE PRIMARY KEY,
                        s_name        CHAR(25) NOT NULL,
                        s_address     VARCHAR(40) NOT NULL,
                        s_nationkey   INTEGER NOT NULL,
                        s_phone       CHAR(15) NOT NULL,
                        s_acctbal     DECIMAL(15,2) NOT NULL,
                        s_comment     VARCHAR(101) NOT NULL);

CREATE TABLE partsupp ( ps_partkey     BIGINT NOT NULL,
                        ps_suppkey     BIGINT NOT NULL,
                        ps_availqty    INTEGER NOT NULL,
                        ps_supplycost  DECIMAL(15,2)  NOT NULL,
                        ps_comment     VARCHAR(199) NOT NULL,
                        CONSTRAINT PS_PARTSUPPKEY UNIQUE(PS_PARTKEY, PS_SUPPKEY) );

CREATE TABLE customer ( c_custkey     INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        c_name        VARCHAR(25) NOT NULL,
                        c_address     VARCHAR(40) NOT NULL,
                        c_nationkey   INTEGER NOT NULL,
                        c_phone       CHAR(15) NOT NULL,
                        c_acctbal     DECIMAL(15,2)   NOT NULL,
                        c_mktsegment  CHAR(10) NOT NULL,
                        c_comment     VARCHAR(117) NOT NULL);

CREATE TABLE orders  ( o_orderkey       BIGINT NOT NULL UNIQUE PRIMARY KEY,
                       o_custkey        INTEGER NOT NULL,
                       o_orderstatus    CHAR(1) NOT NULL,
                       o_totalprice     DECIMAL(15,2) NOT NULL,
                       o_orderdate      DATE NOT NULL,
                       o_orderpriority  CHAR(15) NOT NULL,
                       o_clerk          CHAR(15) NOT NULL,
                       o_shippriority   INTEGER NOT NULL,
                       o_comment        VARCHAR(79) NOT NULL);

CREATE TABLE lineitem ( l_orderkey    BIGINT NOT NULL,
                        l_partkey     BIGINT NOT NULL,
                        l_suppkey     BIGINT NOT NULL,
                        l_linenumber  INTEGER NOT NULL,
                        l_quantity    DECIMAL(15,2) NOT NULL,
                        l_extendedprice  DECIMAL(15,2) NOT NULL,
                        l_discount    DECIMAL(15,2) NOT NULL,
                        l_tax         DECIMAL(15,2) NOT NULL,
                        l_returnflag  CHAR(1) NOT NULL,
                        l_linestatus  CHAR(1) NOT NULL,
                        l_shipdate    DATE NOT NULL,
                        l_commitdate  DATE NOT NULL,
                        l_receiptdate DATE NOT NULL,
                        l_shipinstruct CHAR(25) NOT NULL,
                        l_shipmode     CHAR(10) NOT NULL,
                        l_comment      VARCHAR(44) NOT NULL);

COPY lineitem FROM 'test_datasets/tpch-dbgen/s1/lineitem.tbl' WITH (HEADER false, DELIMITER '|');
COPY orders FROM 'test_datasets/tpch-dbgen/s1/orders.tbl' WITH (HEADER false, DELIMITER '|');
COPY supplier FROM 'test_datasets/tpch-dbgen/s1/supplier.tbl' WITH (HEADER false, DELIMITER '|');
COPY part FROM 'test_datasets/tpch-dbgen/s1/part.tbl' WITH (HEADER false, DELIMITER '|');
COPY customer FROM 'test_datasets/tpch-dbgen/s1/customer.tbl' WITH (HEADER false, DELIMITER '|');
COPY partsupp FROM 'test_datasets/tpch-dbgen/s1/partsupp.tbl' WITH (HEADER false, DELIMITER '|');
COPY nation FROM 'test_datasets/tpch-dbgen/s1/nation.tbl' WITH (HEADER false, DELIMITER '|');
COPY region FROM 'test_datasets/tpch-dbgen/s1/region.tbl' WITH (HEADER false, DELIMITER '|');
