
#  — bring up the cluster (fe + cn + engine) first:
pixi run cluster        # (the task description says run this in another terminal first)


# In terminal 2 — open the MySQL client against the local FE:
pixi run client         # → mysql --host 127.0.0.1 --port 9030 --user root

Note: SET new_planner_agg_stage = 1; need to enable multi-fragment aggregation.
inside the cli client you can run the following commands:

show compute nodes;
SET new_planner_agg_stage = 1;
select * from files('path'='file:///home/ubuntu/git/sirius/scratch/tpch_sf1/nation/part.0.parquet', 'format'='parquet'); 

WITH lineitem AS (
  SELECT *
  FROM FILES(
    "path"="file:///home/ubuntu/git/sirius/scratch/tpch_sf1/lineitem/part.0.parquet",
    "format"="parquet"
  )
)
SELECT
  sum(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= date '1997-01-01'
  AND l_shipdate < date '1998-01-01'
  AND l_discount BETWEEN 0.03 - 0.01 AND 0.03 + 0.01
  AND l_quantity < 24;


