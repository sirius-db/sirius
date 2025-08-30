# =============================================================================
# Copyright 2025, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

import duckdb
import os
import sys

if __name__ == "__main__":
    con = duckdb.connect('performance_test.duckdb', config={"allow_unsigned_extensions": "true"})
    #   con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    extension_path = os.path.join(
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
      'build/release/extension/sirius/sirius.duckdb_extension')
    con.execute("load '{}'".format(extension_path))

    SF = sys.argv[1]
    command = f"cd test_datasets/tpch-dbgen && ./dbgen -f -s {SF} && mv *.tbl perf_test/"

    print("Generating TPC-H data...")
    os.system("mkdir -p test_datasets/tpch-dbgen/perf_test")
    os.system("rm -f test_datasets/tpch-dbgen/perf_test/*")
    os.system(command)

    print("Creating Region, Nation, Part, Supplier, Partsupp, Customer, Orders, Lineitem tables...")
    con.execute('DROP TABLE IF EXISTS region;')
    con.execute('DROP TABLE IF EXISTS nation;')
    con.execute('DROP TABLE IF EXISTS part;')
    con.execute('DROP TABLE IF EXISTS supplier;')
    con.execute('DROP TABLE IF EXISTS partsupp;')
    con.execute('DROP TABLE IF EXISTS customer;')
    con.execute('DROP TABLE IF EXISTS orders;')
    con.execute('DROP TABLE IF EXISTS lineitem;')

    con.execute('''
    CREATE TABLE nation  ( 
                        n_nationkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        n_name       CHAR(25) NOT NULL,
                        n_regionkey  INTEGER NOT NULL,
                        n_comment    VARCHAR(152));
    ''')

    con.execute('''
    CREATE TABLE region  ( 
                        r_regionkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        r_name       CHAR(25) NOT NULL,
                        r_comment    VARCHAR(152));
    ''')

    con.execute('''
    CREATE TABLE part  ( 
                        p_partkey     BIGINT NOT NULL UNIQUE PRIMARY KEY,
                        p_name        VARCHAR(55) NOT NULL,
                        p_mfgr        CHAR(25) NOT NULL,
                        p_brand       CHAR(10) NOT NULL,
                        p_type        VARCHAR(25) NOT NULL,
                        p_size        INTEGER NOT NULL,
                        p_container   CHAR(10) NOT NULL,
                        p_retailprice DECIMAL(15,2) NOT NULL,
                        p_comment     VARCHAR(23) NOT NULL 
    );''')

    con.execute('''
    CREATE TABLE supplier ( 
                        s_suppkey     BIGINT NOT NULL UNIQUE PRIMARY KEY,
                        s_name        CHAR(25) NOT NULL,
                        s_address     VARCHAR(40) NOT NULL,
                        s_nationkey   INTEGER NOT NULL,
                        s_phone       CHAR(15) NOT NULL,
                        s_acctbal     DECIMAL(15,2) NOT NULL,
                        s_comment     VARCHAR(101) NOT NULL
    );''')

    con.execute('''
    CREATE TABLE partsupp ( 
                        ps_partkey     BIGINT NOT NULL,
                        ps_suppkey     BIGINT NOT NULL,
                        ps_availqty    INTEGER NOT NULL,
                        ps_supplycost  DECIMAL(15,2)  NOT NULL,
                        ps_comment     VARCHAR(199) NOT NULL,
                        CONSTRAINT PS_PARTSUPPKEY UNIQUE(PS_PARTKEY, PS_SUPPKEY)
    );''')

    con.execute('''
    CREATE TABLE customer ( 
                        c_custkey     INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        c_name        VARCHAR(25) NOT NULL,
                        c_address     VARCHAR(40) NOT NULL,
                        c_nationkey   INTEGER NOT NULL,
                        c_phone       CHAR(15) NOT NULL,
                        c_acctbal     DECIMAL(15,2)   NOT NULL,
                        c_mktsegment  CHAR(10) NOT NULL,
                        c_comment     VARCHAR(117) NOT NULL
    );''')

    con.execute('''
    CREATE TABLE orders  ( 
                        o_orderkey       BIGINT NOT NULL UNIQUE PRIMARY KEY,
                        o_custkey        INTEGER NOT NULL,
                        o_orderstatus    CHAR(1) NOT NULL,
                        o_totalprice     DECIMAL(15,2) NOT NULL,
                        o_orderdate      DATE NOT NULL,
                        o_orderpriority  CHAR(15) NOT NULL,
                        o_clerk          CHAR(15) NOT NULL,
                        o_shippriority   INTEGER NOT NULL,
                        o_comment        VARCHAR(79) NOT NULL
    );''')

    con.execute('''
    CREATE TABLE lineitem ( 
                        l_orderkey    BIGINT NOT NULL,
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
                        l_comment      VARCHAR(44) NOT NULL
    );''')
  
    print("Copying data into tables...")

    con.execute('''
    COPY lineitem FROM 'test_datasets/tpch-dbgen/perf_test/lineitem.tbl' WITH (HEADER false, DELIMITER '|')
    ''')

    con.execute('''
    COPY orders FROM 'test_datasets/tpch-dbgen/perf_test/orders.tbl' WITH (HEADER false, DELIMITER '|')
    ''')

    con.execute('''
    COPY supplier FROM 'test_datasets/tpch-dbgen/perf_test/supplier.tbl' WITH (HEADER false, DELIMITER '|')
    ''')

    con.execute('''
    COPY part FROM 'test_datasets/tpch-dbgen/perf_test/part.tbl' WITH (HEADER false, DELIMITER '|')
    ''')

    con.execute('''
    COPY customer FROM 'test_datasets/tpch-dbgen/perf_test/customer.tbl' WITH (HEADER false, DELIMITER '|')
    ''')

    con.execute('''
    COPY partsupp FROM 'test_datasets/tpch-dbgen/perf_test/partsupp.tbl' WITH (HEADER false, DELIMITER '|')
    ''')

    con.execute('''
    COPY nation FROM 'test_datasets/tpch-dbgen/perf_test/nation.tbl' WITH (HEADER false, DELIMITER '|')
    ''')

    con.execute('''
    COPY region FROM 'test_datasets/tpch-dbgen/perf_test/region.tbl' WITH (HEADER false, DELIMITER '|')
    ''')
  
    con.close()