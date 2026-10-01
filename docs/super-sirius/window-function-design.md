
**DuckDB Plan**

The plan we get from DuckDB for a Window function is basically just one relation called WINDOW. It can hold multiple window functions, even if they have different window shapes (different ordering or partitioning).

For example:
SELECT
    l_orderkey,
    l_partkey,
    l_shipdate,
    l_quantity,
    l_extendedprice,
    l_discount,
    SUM(l_extendedprice) OVER bounded_range AS bounded_extended_price_sum,
    AVG(l_discount) OVER bounded_range AS bounded_discount_avg,
    SUM(l_quantity) OVER (
        ORDER BY l_shipdate
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_quantity
FROM lineitem
WINDOW bounded_range AS (
    PARTITION BY l_partkey
    ORDER BY l_shipdate
    RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND INTERVAL 7 DAYS FOLLOWING
)

Produces:
[
  {
    "name": "PROJECTION",
    "children": [
      {
        "name": "WINDOW",
        "children": [
          {
            "name": "SEQ_SCAN",
            "children": [],
            "extra_info": {
              "Filters": "",
              "Table": "tpch_sf1.main.lineitem",
              "Type": "Sequential Scan",
              "Estimated Cardinality": "6001215"
            }
          }
        ],
        "extra_info": {
          "Expressions": [
            "sum(l_extendedprice) OVER (PARTITION BY l_partkey ORDER BY CAST(l_shipdate AS TIMESTAMP) ASC NULLS LAST RANGE BETWEEN (l_shipdate - '7 days'::INTERVAL) PRECEDING AND (l_shipdate + '7 days'::INTERVAL) FOLLOWING)",
            "avg(l_discount) OVER (PARTITION BY l_partkey ORDER BY CAST(l_shipdate AS TIMESTAMP) ASC NULLS LAST RANGE BETWEEN (l_shipdate - '7 days'::INTERVAL) PRECEDING AND (l_shipdate + '7 days'::INTERVAL) FOLLOWING)",
            "sum(l_quantity) OVER (ORDER BY l_shipdate ASC NULLS LAST ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)"
          ],
          "Estimated Cardinality": "6001215"
        }
      }
    ],
    "extra_info": {
      "Expressions": [
        "l_orderkey",
        "l_partkey",
        "l_shipdate",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "bounded_extended_price_sum",
        "bounded_discount_avg",
        "cumulative_quantity"
      ],
      "Estimated Cardinality": "6001215"
    }
  }
]

**Proposed Sirius Physical Plan Shape**

For each WINDOW relation we receive from DuckDB, we need to identify all the unique window data orderings that are in the window relation. Two window data orderings are considered the same if they have the same partition by and order by clause. For each unique window data ordering, we will do a data ordering stage followed by the Sirius physical window operator. This means that if a query has multiple different window data orderings in one WINDOW relation, then we have to split the dag into multiple branches and then join these branches at the end. Initially we should strive to support just one to keep things simpler.

The data ordering stage will consist of either a full global sort, a hash partitioning or nothing depending on the data ordering clauses in the window function:
- If it has a Partition By and Order By clause:
	- Do a global sort wherein the sort keys are the Partition By clause keys and Order By clause keys combined. BUT in the global sort, the range partitioning (a.k.a. sort partitioning) keys are only the Partition By clause keys
- If it has a Order By clause and no Partition By clause:
	- Do a global sort using the order by clause keys
- If it has Partition By clause and no Order By clause:
	- Do a hash partitioning using the Partition By clause keys
- If it has no Partition By clause and no Order By clause:
	- Do not do anything to the data

Then after the data ordering stage we will have the Sirius Physical Window operator.


**Proposed Sirius Physical Window Operator**

The Window Operator has the following functions. Which functions it needs to run, depends on the window type. Some of these functions can be run at the same time in one task, some depend on other functions running first as a separate task. This means that for each data batch that arrives at the Window Operator, we may need to do different task types, like a mini state machine per batch.

The functions:
- determine_edges_for_ranges
	- Used only when windows are RANGE based (as opposed to ROWS based), and at least one Window edge is BOUNDED.
	- They just see what values are at the ends so that we can see how far the halo extends into other batches.	-
- copy_halo / find_and_copy_halo
	- Used when windows are ROWS / RANGE based as at least one Window edge is BOUNDED
	- This is used for copying a segment of the batch (halo) to then be used on another batch
	- find_and_copy_halo is used for RANGE windows and requires determine_edges_for_ranges to be complete
	- copy_halo is used for ROW windows and can be done independently
- concat_halo
	- concatenate the halo from the preceding and/or following batch onto the current batch
	- requires all copy_halo or find_and_copy_halo to be completed
- compute_local_window_aggregate
	- perform a window aggregate on the batch
	- If halos are required for the window type, then this needs to be done after concat_halo
- trim_halo
	- Remove the rows that correspond to the halo which was concatenated
	- requires compute_local_window_aggregate to be completed
- copy_unbounded_partial_aggregation
	- Used when there is at least one UNBOUNDED window edge
	- This is used for copying a partial aggregation representing a batch for an unbounded edge.
	- requires compute_local_window_aggregate to be completed
- calculate_unbounded_cumulative_partial_aggregation
	- used to combine multiple results from copy_unbounded_partial_aggregation
	- hard barrier. Requires all copy_unbounded_partial_aggregation to have completed.
- apply_unbounded_partial_agg
	- Apply the cumulative unbounded partial aggregations calculated for all the preceding and/or following batches onto the current batch.
	- requires compute_local_window_aggregate and calculate_unbounded_cumulative_partial_aggregation to be completed

In the case of something like a window that has no Partition By clause and we do have one bounded and one unbounded edge, then we would have to do all the different functions. In that case you can see the dependencies between the function types. In this case we would have to do 4 different tasks for each batch:
1. determine_edges_for_ranges
2. find_and_copy_halo
3. concat_halo, compute_local_window_aggregate, trim_halo, copy_unbounded_partial_aggregation
4. calculate_unbounded_cumulative_partial_aggregation  (one task only for all. Not per batch)
5. apply_unbounded_partial_agg

These dependencies happen because the  information from one task of one batch is needed for another batch. i.e. We need to determine_edges_for_ranges on batch A, to be able to do find_and_copy_halo for batch B, which is needed for concat_halo on batch A.


**Design Considerations and design options**
The above design would require the following small Sirius architectural changes:
- Because we would have multiple different task types, we would need to be able to track memory consumption for different task types independently since they would have very different weights
- We would need a data repository to live inside the operator in order to hold the halos. This would then need to be exposed to the downgrade manager

We could alternatively separate the Sirius Physical Window operator proposed into multiple different  operators in order to avoid these changes and to make it more "modular". BUT I think that would make the design significantly more complex.

**Other design options**
The above design is structured for maximum task parallelism and so that every batch can be done in whatever order, subject to the dependencies of the task types.
Alternatively we could reshape the tasks so that they are more daisy chained, but that would only be more effective if we are in a highly memory constrained scenario and/or with few GPUs or if GPUs are already at a high occupancy.

**Notes regarding the scalability of this design**
The design proposed above would be limited in the size of shape of the data it can process. If there is a Partition By clause, one or more whole partitions would have to be able to fit into a single batch. Meaning a single partition value could not be able to be spread between multiple batches. This becomes a problem where there is a very large amount of data, with a Partition By clause that has very low cardinality (low number of unique values).
That being said the proposed design can be altered to be able to support better scaling properties (a partition can span multiple batches). But would be a bit more complex and slower (would require a bit more calculation steps when we do have a Partition By clause).

Another limitation on scalability is that when there are bounded windows, the design requires to be able to hold a batch plus halos holding the bounded extension to the batch, all within one single batch. With very large bounded windows this can result in an OOM.

 See example in action: window-function-design-example.ods
