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

# Import queries from same directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
import queries as _queries_module

QUERIES = _queries_module.QUERIES


def _q11_ratio_for_sf(sf):
    """Q11 ratio: 0.0001 for sf=1; inversely proportional to sf."""
    return 0.0001 / float(sf)


def _generate_tpch_queries(sf, out_root):
    """Write orig/ and gpu/ query files under out_root using QUERIES and scale-dependent Q11 ratio."""
    orig_dir = os.path.join(out_root, "orig")
    gpu_dir = os.path.join(out_root, "gpu")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(gpu_dir, exist_ok=True)

    q11_ratio = _q11_ratio_for_sf(sf)

    for name, sql in QUERIES.items():
        # Q11: substitute ratio with scale-factor-dependent value
        if name == "q11":
            sql = sql.replace(
                "0.0001000000", f"{q11_ratio:.10f}".rstrip("0").rstrip(".")
            )

        # Orig: unmodified query
        orig_path = os.path.join(orig_dir, f"{name}.sql")
        with open(orig_path, "w") as f:
            f.write(sql.strip())
            f.write("\n")

        # GPU: wrap in call gpu_execution('...'); (escape single quotes by doubling)
        escaped = sql.strip().replace("'", "''")
        gpu_sql = f"call gpu_execution('{escaped}');\n"
        gpu_path = os.path.join(gpu_dir, f"{name}.sql")
        with open(gpu_path, "w") as f:
            f.write(gpu_sql)


if __name__ == "__main__":
    con = duckdb.connect(
        "performance_test.duckdb", config={"allow_unsigned_extensions": "true"}
    )
    #   con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    extension_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "build/release/extension/sirius/sirius.duckdb_extension",
    )
    con.execute("load '{}'".format(extension_path))

    if len(sys.argv) < 4:
        print(
            "Usage: python generate_test_data_tpchgen-rs.py <SF> <partitions> <format>"
        )
        sys.exit(1)

    if not os.path.exists("test_datasets"):
        print("Run this script from the Sirius top-level directory")

    SF = sys.argv[1]
    partitions = sys.argv[2]
    format = sys.argv[3]

    data_dir = f"test_datasets/tpchgen-rs/sf{SF}/p{partitions}/{format}"
    command = f"tpchgen-cli -s {SF} --parts {partitions} --format {format} --output-dir {data_dir}"

    if os.path.exists(data_dir):
        print(
            f"Data for SF={SF}, partitions={partitions}, format={format} already exists in {data_dir}, skipping generation..."
        )
    else:
        print("Generating TPC-H data using tpchgen-rs...")
        os.system(f"rm -rf {data_dir}")
        os.system(f"mkdir -p {data_dir}")
        os.system(command)

    print(
        "Creating Region, Nation, Part, Supplier, Partsupp, Customer, Orders, Lineitem tables..."
    )
    con.execute("DROP TABLE IF EXISTS region;")
    con.execute("DROP TABLE IF EXISTS nation;")
    con.execute("DROP TABLE IF EXISTS part;")
    con.execute("DROP TABLE IF EXISTS supplier;")
    con.execute("DROP TABLE IF EXISTS partsupp;")
    con.execute("DROP TABLE IF EXISTS customer;")
    con.execute("DROP TABLE IF EXISTS orders;")
    con.execute("DROP TABLE IF EXISTS lineitem;")

    con.execute(
        """
    CREATE TABLE nation  (
                        n_nationkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        n_name       CHAR(25) NOT NULL,
                        n_regionkey  INTEGER NOT NULL,
                        n_comment    VARCHAR(152));
    """
    )

    con.execute(
        """
    CREATE TABLE region  (
                        r_regionkey  INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        r_name       CHAR(25) NOT NULL,
                        r_comment    VARCHAR(152));
    """
    )

    con.execute(
        """
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
    );"""
    )

    con.execute(
        """
    CREATE TABLE supplier (
                        s_suppkey     BIGINT NOT NULL UNIQUE PRIMARY KEY,
                        s_name        CHAR(25) NOT NULL,
                        s_address     VARCHAR(40) NOT NULL,
                        s_nationkey   INTEGER NOT NULL,
                        s_phone       CHAR(15) NOT NULL,
                        s_acctbal     DECIMAL(15,2) NOT NULL,
                        s_comment     VARCHAR(101) NOT NULL
    );"""
    )

    con.execute(
        """
    CREATE TABLE partsupp (
                        ps_partkey     BIGINT NOT NULL,
                        ps_suppkey     BIGINT NOT NULL,
                        ps_availqty    INTEGER NOT NULL,
                        ps_supplycost  DECIMAL(15,2)  NOT NULL,
                        ps_comment     VARCHAR(199) NOT NULL,
                        CONSTRAINT PS_PARTSUPPKEY UNIQUE(PS_PARTKEY, PS_SUPPKEY)
    );"""
    )

    con.execute(
        """
    CREATE TABLE customer (
                        c_custkey     INTEGER NOT NULL UNIQUE PRIMARY KEY,
                        c_name        VARCHAR(25) NOT NULL,
                        c_address     VARCHAR(40) NOT NULL,
                        c_nationkey   INTEGER NOT NULL,
                        c_phone       CHAR(15) NOT NULL,
                        c_acctbal     DECIMAL(15,2)   NOT NULL,
                        c_mktsegment  CHAR(10) NOT NULL,
                        c_comment     VARCHAR(117) NOT NULL
    );"""
    )

    con.execute(
        """
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
    );"""
    )

    con.execute(
        """
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
    );"""
    )

    print("Copying data into tables...")

    lineitem_file = f"{data_dir}/lineitem/lineitem.*.{format}"
    orders_file = f"{data_dir}/orders/orders.*.{format}"
    supplier_file = f"{data_dir}/supplier/supplier.*.{format}"
    part_file = f"{data_dir}/part/part.*.{format}"
    customer_file = f"{data_dir}/customer/customer.*.{format}"
    partsupp_file = f"{data_dir}/partsupp/partsupp.*.{format}"
    nation_file = f"{data_dir}/nation/nation.*.{format}"
    region_file = f"{data_dir}/region/region.*.{format}"

    format_options = "WITH (HEADER true, DELIMITER ',')" if format == "csv" else ""

    con.execute(
        f"""
    COPY lineitem FROM '{lineitem_file}' {format_options}
    """
    )

    con.execute(
        f"""
    COPY orders FROM '{orders_file}' {format_options}
    """
    )

    con.execute(
        f"""
    COPY supplier FROM '{supplier_file}' {format_options}
    """
    )

    con.execute(
        f"""
    COPY part FROM '{part_file}' {format_options}
    """
    )

    con.execute(
        f"""
    COPY customer FROM '{customer_file}' {format_options}
    """
    )

    con.execute(
        f"""
    COPY partsupp FROM '{partsupp_file}' {format_options}
    """
    )

    con.execute(
        f"""
    COPY nation FROM '{nation_file}' {format_options}
    """
    )

    con.execute(
        f"""
    COPY region FROM '{region_file}' {format_options}
    """
    )

    # Generate TPC-H query files: orig (unmodified) and gpu (wrapped in call gpu_execution(...))
    tpch_queries_dir = os.path.join(_script_dir, "tpch_queries")
    _generate_tpch_queries(SF, tpch_queries_dir)
    print(f"Wrote TPC-H queries to {tpch_queries_dir}/orig and {tpch_queries_dir}/gpu")

    con.close()
