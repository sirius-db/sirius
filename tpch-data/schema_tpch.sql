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

    -- PRIMARY KEY (R_REGIONKEY)
);

CREATE TABLE NATION  (
    N_NATIONKEY  INTEGER NOT NULL,
    N_NAME       CHAR(25) NOT NULL,
    N_REGIONKEY  INTEGER NOT NULL,
    N_COMMENT    VARCHAR(152),

    -- PRIMARY KEY (N_NATIONKEY),

    -- CONSTRAINT NATION_FK1 FOREIGN KEY (N_REGIONKEY) references REGION(R_REGIONKEY)
);

CREATE TABLE PART  (
    P_PARTKEY     INTEGER NOT NULL,
    P_NAME        VARCHAR(55) NOT NULL,
    P_MFGR        CHAR(25) NOT NULL,
    P_BRAND       CHAR(10) NOT NULL,
    P_TYPE        VARCHAR(25) NOT NULL,
    P_SIZE        INTEGER NOT NULL,
    P_CONTAINER   CHAR(10) NOT NULL,
    P_RETAILPRICE DECIMAL(15,2) NOT NULL,
    P_COMMENT     VARCHAR(23) NOT NULL,

    -- PRIMARY KEY (P_PARTKEY)
);

CREATE TABLE SUPPLIER (
    S_SUPPKEY     INTEGER NOT NULL,
    S_NAME        CHAR(25) NOT NULL,
    S_ADDRESS     VARCHAR(40) NOT NULL,
    S_NATIONKEY   INTEGER NOT NULL,
    S_PHONE       CHAR(15) NOT NULL,
    S_ACCTBAL     DECIMAL(15,2) NOT NULL,
    S_COMMENT     VARCHAR(101) NOT NULL,

    -- PRIMARY KEY (S_SUPPKEY),
    -- CONSTRAINT SUPPLIER_FK1 FOREIGN KEY (S_NATIONKEY) references NATION(N_NATIONKEY)
);

CREATE TABLE PARTSUPP (
    PS_PARTKEY     INTEGER NOT NULL,
    PS_SUPPKEY     INTEGER NOT NULL,
    PS_AVAILQTY    INTEGER NOT NULL,
    PS_SUPPLYCOST  DECIMAL(15,2)  NOT NULL,
    PS_COMMENT     VARCHAR(199) NOT NULL,

    -- PRIMARY KEY (PS_PARTKEY,PS_SUPPKEY),

    -- CONSTRAINT PARTSUPP_FK1 FOREIGN KEY (PS_SUPPKEY) references SUPPLIER(S_SUPPKEY),
    -- CONSTRAINT PARTSUPP_FK2 FOREIGN KEY (PS_PARTKEY) references PART(P_PARTKEY)
);

CREATE TABLE CUSTOMER (
    C_CUSTKEY     INTEGER NOT NULL,
    C_NAME        VARCHAR(25) NOT NULL,
    C_ADDRESS     VARCHAR(40) NOT NULL,
    C_NATIONKEY   INTEGER NOT NULL,
    C_PHONE       CHAR(15) NOT NULL,
    C_ACCTBAL     DECIMAL(15,2)   NOT NULL,
    C_MKTSEGMENT  CHAR(10) NOT NULL,
    C_COMMENT     VARCHAR(117) NOT NULL,

    -- PRIMARY KEY (C_CUSTKEY),

    -- CONSTRAINT CUSTOMER_FK1 FOREIGN KEY (C_NATIONKEY) references NATION(N_NATIONKEY)
);

CREATE TABLE ORDERS  (
    O_ORDERKEY       INTEGER NOT NULL,
    O_CUSTKEY        INTEGER NOT NULL,
    O_ORDERSTATUS    CHAR(1) NOT NULL,
    O_TOTALPRICE     DECIMAL(15,2) NOT NULL,
    O_ORDERDATE      DATE NOT NULL,
    O_ORDERPRIORITY  CHAR(15) NOT NULL,
    O_CLERK          CHAR(15) NOT NULL,
    O_SHIPPRIORITY   INTEGER NOT NULL,
    O_COMMENT        VARCHAR(79) NOT NULL,

    -- PRIMARY KEY (O_ORDERKEY),

    -- CONSTRAINT ORDERS_FK1 FOREIGN KEY (O_CUSTKEY) references CUSTOMER(C_CUSTKEY)
);

CREATE TABLE LINEITEM (
    L_ORDERKEY    INTEGER NOT NULL,
    L_PARTKEY     INTEGER NOT NULL,
    L_SUPPKEY     INTEGER NOT NULL,
    L_LINENUMBER  INTEGER NOT NULL,
    L_QUANTITY    DECIMAL(15,2) NOT NULL,
    L_EXTENDEDPRICE  DECIMAL(15,2) NOT NULL,
    L_DISCOUNT    DECIMAL(15,2) NOT NULL,
    L_TAX         DECIMAL(15,2) NOT NULL,
    L_RETURNFLAG  CHAR(1) NOT NULL,
    L_LINESTATUS  CHAR(1) NOT NULL,
    L_SHIPDATE    DATE NOT NULL,
    L_COMMITDATE  DATE NOT NULL,
    L_RECEIPTDATE DATE NOT NULL,
    L_SHIPINSTRUCT CHAR(25) NOT NULL,
    L_SHIPMODE     CHAR(10) NOT NULL,
    L_COMMENT      VARCHAR(44) NOT NULL,

    -- PRIMARY KEY (L_ORDERKEY,L_LINENUMBER),

    -- CONSTRAINT LINEITEM_FK1 FOREIGN KEY (L_ORDERKEY)  references ORDERS(O_ORDERKEY),
    -- CONSTRAINT LINEITEM_FK2 FOREIGN KEY (L_PARTKEY,L_SUPPKEY) references PARTSUPP(PS_PARTKEY,PS_SUPPKEY)
);
