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
con = duckdb.connect('tpch_s1.duckdb', config={"allow_unsigned_extensions": "true"})
con.execute("load '/mnt/nvme/sirius/build/release/extension/sirius/sirius.duckdb_extension'")
con.execute("call gpu_buffer_init('1 GB', '1 GB')")
# con.execute("create table T(A int, B double);");
# con.execute("insert into T values(1, 1.2);");
print(con.execute("select n_name from nation").fetchall())
print(con.execute("call gpu_processing('select n_name from nation')").fetchall())
print(con.execute("call gpu_processing('select \
    l_orderkey, \
    sum(l_extendedprice * (1 - l_discount)) as revenue, \
    o_orderdate, \
    o_shippriority \
from \
    customer, \
    orders, \
    lineitem \
where \
    c_mktsegment = 1 \
    and c_custkey = o_custkey \
    and l_orderkey = o_orderkey \
    and o_orderdate < 19950315 \
    and l_shipdate > 19950315 \
group by \
    l_orderkey, \
    o_orderdate, \
    o_shippriority \
order by \
    revenue desc, \
    o_orderdate')").fetchall())
con.close()