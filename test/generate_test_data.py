import duckdb
import os
import sys

if __name__ == "__main__":
  con = duckdb.connect('performance_test.duckdb', config={"allow_unsigned_extensions": "true"})
#   con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
  con.execute("load '/mnt/nvme/sirius/build/release/extension/sirius/sirius.duckdb_extension'")
  
  SF = sys.argv[1]
  command = f"cd dbgen && ./dbgen -f -s {SF} && mv *.tbl perf_test/"
  
  print("Generating TPC-H data...")
  os.system("mkdir -p dbgen/perf_test")
  os.system("rm -f dbgen/perf_test/*")
  os.system(command)

  print("Creating Region, Nation, Part, Supplier, Partsupp, Customer, Orders, Lineitem tables...")
  con.execute('''
  CREATE TABLE REGION  (
      R_REGIONKEY  BIGINT NOT NULL UNIQUE PRIMARY KEY,
      R_NAME       CHAR(25) NOT NULL,
      R_COMMENT    VARCHAR(152) NOT NULL,
  );''')

  con.execute('''
  CREATE TABLE NATION  (
      N_NATIONKEY  BIGINT NOT NULL UNIQUE PRIMARY KEY,
      N_NAME       CHAR(25) NOT NULL,
      N_REGIONKEY  BIGINT NOT NULL,
      N_COMMENT    VARCHAR(152) NOT NULL,
  );''')

  con.execute('''
  CREATE TABLE PART  (
      P_PARTKEY     BIGINT NOT NULL UNIQUE PRIMARY KEY,
      P_NAME        VARCHAR(55) NOT NULL,
      P_MFGR        BIGINT NOT NULL,
      P_BRAND       BIGINT NOT NULL,
      P_TYPE        BIGINT NOT NULL,
      P_SIZE        BIGINT NOT NULL,
      P_CONTAINER   BIGINT NOT NULL,
      P_RETAILPRICE DOUBLE NOT NULL,
      P_COMMENT     VARCHAR(23) NOT NULL,
  );''')

  con.execute('''
  CREATE TABLE SUPPLIER (
      S_SUPPKEY     BIGINT NOT NULL UNIQUE PRIMARY KEY,
      S_NAME        CHAR(25) NOT NULL,
      S_ADDRESS     VARCHAR(40) NOT NULL,
      S_NATIONKEY   BIGINT NOT NULL,
      S_PHONE       CHAR(15) NOT NULL,
      S_ACCTBAL     DOUBLE NOT NULL,
      S_COMMENT     VARCHAR(101) NOT NULL,
  );''')

  con.execute('''
  CREATE TABLE PARTSUPP (
      PS_PARTKEY     BIGINT NOT NULL,
      PS_SUPPKEY     BIGINT NOT NULL,
      PS_AVAILQTY    DOUBLE NOT NULL,
      PS_SUPPLYCOST  DOUBLE  NOT NULL,
      PS_COMMENT     VARCHAR(199) NOT NULL,
      CONSTRAINT PS_PARTSUPPKEY UNIQUE(PS_PARTKEY, PS_SUPPKEY)
  );''')

  con.execute('''
  CREATE TABLE CUSTOMER (
      C_CUSTKEY     BIGINT NOT NULL UNIQUE PRIMARY KEY,
      C_NAME        VARCHAR(25) NOT NULL,
      C_ADDRESS     VARCHAR(40) NOT NULL,
      C_NATIONKEY   BIGINT NOT NULL,
      C_PHONE       CHAR(15) NOT NULL,
      C_ACCTBAL     DOUBLE NOT NULL,
      C_MKTSEGMENT  BIGINT NOT NULL,
      C_COMMENT     VARCHAR(117) NOT NULL,
  );''')

  con.execute('''
  CREATE TABLE ORDERS  (
      O_ORDERKEY       BIGINT NOT NULL UNIQUE PRIMARY KEY,
      O_CUSTKEY        BIGINT NOT NULL,
      O_ORDERSTATUS    BIGINT NOT NULL,
      O_TOTALPRICE     DOUBLE NOT NULL,
      O_ORDERDATE      BIGINT NOT NULL,
      O_ORDERPRIORITY  BIGINT NOT NULL,
      O_CLERK          BIGINT NOT NULL,
      O_SHIPPRIORITY   BIGINT NOT NULL,
      O_COMMENT        VARCHAR(79) NOT NULL,
  );''')

  con.execute('''
  CREATE TABLE LINEITEM (
      L_ORDERKEY    BIGINT NOT NULL,
      L_PARTKEY     BIGINT NOT NULL,
      L_SUPPKEY     BIGINT NOT NULL,
      L_LINENUMBER  BIGINT NOT NULL,
      L_QUANTITY    DOUBLE NOT NULL,
      L_EXTENDEDPRICE  DOUBLE NOT NULL,
      L_DISCOUNT    DOUBLE NOT NULL,
      L_TAX         DOUBLE NOT NULL,
      L_RETURNFLAG  BIGINT NOT NULL,
      L_LINESTATUS  BIGINT NOT NULL,
      L_SHIPDATE    BIGINT NOT NULL,
      L_COMMITDATE  BIGINT NOT NULL,
      L_RECEIPTDATE BIGINT NOT NULL,
      L_SHIPINSTRUCT BIGINT NOT NULL,
      L_SHIPMODE     BIGINT NOT NULL,
      L_COMMENT      VARCHAR(44) NOT NULL,
  );''')
  
  print("Copying data into tables...")

  con.execute('''
  COPY lineitem FROM 'dbgen/perf_test/lineitem.tbl' WITH (HEADER false, DELIMITER '|')
  ''')

  con.execute('''
  COPY orders FROM 'dbgen/perf_test/orders.tbl' WITH (HEADER false, DELIMITER '|')
  ''')

  con.execute('''
  COPY supplier FROM 'dbgen/perf_test/supplier.tbl' WITH (HEADER false, DELIMITER '|')
  ''')

  con.execute('''
  COPY part FROM 'dbgen/perf_test/part.tbl' WITH (HEADER false, DELIMITER '|')
  ''')

  con.execute('''
  COPY customer FROM 'dbgen/perf_test/customer.tbl' WITH (HEADER false, DELIMITER '|')
  ''')

  con.execute('''
  COPY partsupp FROM 'dbgen/perf_test/partsupp.tbl' WITH (HEADER false, DELIMITER '|')
  ''')

  con.execute('''
  COPY nation FROM 'dbgen/perf_test/nation.tbl' WITH (HEADER false, DELIMITER '|')
  ''')

  con.execute('''
  COPY region FROM 'dbgen/perf_test/region.tbl' WITH (HEADER false, DELIMITER '|')
  ''')
  
  con.close()