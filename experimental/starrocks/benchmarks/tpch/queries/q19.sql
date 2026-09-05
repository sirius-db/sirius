WITH
customer AS (SELECT * FROM FILES("path"="file://__TPCH_DATA__/customer/*.parquet","format"="parquet")),
lineitem AS (SELECT * FROM FILES("path"="file://__TPCH_DATA__/lineitem/*.parquet","format"="parquet")),
nation AS (SELECT * FROM FILES("path"="file://__TPCH_DATA__/nation/*.parquet","format"="parquet")),
orders AS (SELECT * FROM FILES("path"="file://__TPCH_DATA__/orders/*.parquet","format"="parquet")),
part AS (SELECT * FROM FILES("path"="file://__TPCH_DATA__/part/*.parquet","format"="parquet")),
partsupp AS (SELECT * FROM FILES("path"="file://__TPCH_DATA__/partsupp/*.parquet","format"="parquet")),
region AS (SELECT * FROM FILES("path"="file://__TPCH_DATA__/region/*.parquet","format"="parquet")),
supplier AS (SELECT * FROM FILES("path"="file://__TPCH_DATA__/supplier/*.parquet","format"="parquet"))
SELECT
    sum(l_extendedprice * (1 - l_discount)) AS revenue
FROM
    lineitem,
    part
WHERE (p_partkey = l_partkey
    AND p_brand = 'Brand#12'
    AND p_container IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
    AND l_quantity >= 1
    AND l_quantity <= 1 + 10
    AND p_size BETWEEN 1 AND 5
    AND l_shipmode IN ('AIR', 'AIR REG')
    AND l_shipinstruct = 'DELIVER IN PERSON')
    OR (p_partkey = l_partkey
        AND p_brand = 'Brand#23'
        AND p_container IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        AND l_quantity >= 10
        AND l_quantity <= 10 + 10
        AND p_size BETWEEN 1 AND 10
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON')
    OR (p_partkey = l_partkey
        AND p_brand = 'Brand#34'
        AND p_container IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        AND l_quantity >= 20
        AND l_quantity <= 20 + 10
        AND p_size BETWEEN 1 AND 15
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON');
