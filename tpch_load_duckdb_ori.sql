-- Copyright 2025 Sirius Contributors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

DROP TABLE IF EXISTS LINEITEM;
DROP TABLE IF EXISTS ORDERS;
DROP TABLE IF EXISTS CUSTOMER;
DROP TABLE IF EXISTS PARTSUPP;
DROP TABLE IF EXISTS SUPPLIER;
DROP TABLE IF EXISTS PART;
DROP TABLE IF EXISTS NATION;
DROP TABLE IF EXISTS REGION;


CREATE TABLE REGION  (
    R_REGIONKEY  INTEGER NOT NULL,
    R_NAME       CHAR(25) NOT NULL,
    R_COMMENT    VARCHAR(152),
);

CREATE TABLE NATION  (
    N_NATIONKEY  INTEGER NOT NULL,
    N_NAME       CHAR(25) NOT NULL,
    N_REGIONKEY  INTEGER NOT NULL,
    N_COMMENT    VARCHAR(152),
);

CREATE TABLE PART  (
    P_PARTKEY     INTEGER NOT NULL,
    P_NAME        VARCHAR(55) NOT NULL,
    P_MFGR        INTEGER NOT NULL,
    P_BRAND       INTEGER NOT NULL,
    P_TYPE        INTEGER NOT NULL,
    P_SIZE        INTEGER NOT NULL,
    P_CONTAINER   INTEGER NOT NULL,
    P_RETAILPRICE DECIMAL(15,2) NOT NULL,
    P_COMMENT     VARCHAR(23) NOT NULL,
);

CREATE TABLE SUPPLIER (
    S_SUPPKEY     INTEGER NOT NULL,
    S_NAME        CHAR(25) NOT NULL,
    S_ADDRESS     VARCHAR(40) NOT NULL,
    S_NATIONKEY   INTEGER NOT NULL,
    S_PHONE       CHAR(15) NOT NULL,
    S_ACCTBAL     DECIMAL(15,2) NOT NULL,
    S_COMMENT     VARCHAR(101) NOT NULL,
);

CREATE TABLE PARTSUPP (
    PS_PARTKEY     INTEGER NOT NULL,
    PS_SUPPKEY     INTEGER NOT NULL,
    PS_AVAILQTY    INTEGER NOT NULL,
    PS_SUPPLYCOST  DECIMAL(15,2)  NOT NULL,
    PS_COMMENT     VARCHAR(199) NOT NULL,
);

CREATE TABLE CUSTOMER (
    C_CUSTKEY     INTEGER NOT NULL,
    C_NAME        VARCHAR(25) NOT NULL,
    C_ADDRESS     VARCHAR(40) NOT NULL,
    C_NATIONKEY   INTEGER NOT NULL,
    C_PHONE       CHAR(15) NOT NULL,
    C_ACCTBAL     DECIMAL(15,2)   NOT NULL,
    C_MKTSEGMENT  INTEGER NOT NULL,
    C_COMMENT     VARCHAR(117) NOT NULL,
);

CREATE TABLE ORDERS  (
    O_ORDERKEY       INTEGER NOT NULL,
    O_CUSTKEY        INTEGER NOT NULL,
    O_ORDERSTATUS    INTEGER NOT NULL,
    O_TOTALPRICE     DECIMAL(15,2) NOT NULL,
    O_ORDERDATE      INTEGER NOT NULL,
    O_ORDERPRIORITY  INTEGER NOT NULL,
    O_CLERK          INTEGER NOT NULL,
    O_SHIPPRIORITY   INTEGER NOT NULL,
    O_COMMENT        VARCHAR(79) NOT NULL,
);

CREATE TABLE LINEITEM (
    L_ORDERKEY    INTEGER NOT NULL,
    L_PARTKEY     INTEGER NOT NULL,
    L_SUPPKEY     INTEGER NOT NULL,
    L_LINENUMBER  INTEGER NOT NULL,
    L_QUANTITY    INTEGER NOT NULL,
    L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,
    L_DISCOUNT    DECIMAL(15,2) NOT NULL,
    L_TAX         DECIMAL(15,2) NOT NULL,
    L_RETURNFLAG  INTEGER NOT NULL,
    L_LINESTATUS  INTEGER NOT NULL,
    L_SHIPDATE    INTEGER NOT NULL,
    L_COMMITDATE  INTEGER NOT NULL,
    L_RECEIPTDATE INTEGER NOT NULL,
    L_SHIPINSTRUCT INTEGER NOT NULL,
    L_SHIPMODE     INTEGER NOT NULL,
    L_COMMENT      VARCHAR(44) NOT NULL,
);

COPY lineitem FROM '/home/ubuntu/new-crystal/tpch_dataset_creator/dbgen/s1/lineitem.tbl' WITH (HEADER false, DELIMITER '|');
COPY orders FROM '/home/ubuntu/new-crystal/tpch_dataset_creator/dbgen/s1/orders.tbl' WITH (HEADER false, DELIMITER '|');
COPY supplier FROM '/home/ubuntu/new-crystal/tpch_dataset_creator/dbgen/s1/supplier.tbl' WITH (HEADER false, DELIMITER '|');
COPY part FROM '/home/ubuntu/new-crystal/tpch_dataset_creator/dbgen/s1/part.tbl' WITH (HEADER false, DELIMITER '|');
COPY customer FROM '/home/ubuntu/new-crystal/tpch_dataset_creator/dbgen/s1/customer.tbl' WITH (HEADER false, DELIMITER '|');
COPY partsupp FROM '/home/ubuntu/new-crystal/tpch_dataset_creator/dbgen/s1/partsupp.tbl' WITH (HEADER false, DELIMITER '|');
COPY nation FROM '/home/ubuntu/new-crystal/tpch_dataset_creator/dbgen/s1/nation.tbl' WITH (HEADER false, DELIMITER '|');
COPY region FROM '/home/ubuntu/new-crystal/tpch_dataset_creator/dbgen/s1/region.tbl' WITH (HEADER false, DELIMITER '|');
