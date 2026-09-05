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
    l_shipmode,
    sum(
        CASE WHEN o_orderpriority = '1-URGENT'
            OR o_orderpriority = '2-HIGH' THEN
            1
        ELSE
            0
        END) AS high_line_count,
    sum(
        CASE WHEN o_orderpriority <> '1-URGENT'
            AND o_orderpriority <> '2-HIGH' THEN
            1
        ELSE
            0
        END) AS low_line_count
FROM
    orders,
    lineitem
WHERE
    o_orderkey = l_orderkey
    AND l_shipmode IN ('MAIL', 'SHIP')
    AND l_commitdate < l_receiptdate
    AND l_shipdate < l_commitdate
    AND l_receiptdate >= CAST('1994-01-01' AS date)
    AND l_receiptdate < CAST('1995-01-01' AS date)
GROUP BY
    l_shipmode
ORDER BY
    l_shipmode;
