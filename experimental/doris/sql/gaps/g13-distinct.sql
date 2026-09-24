-- G-13: SELECT DISTINCT is compiled by Nereids into a two-phase AGGREGATION_NODE with grouping
-- keys and no aggregate functions; the stitcher folds it and the translator emits an
-- AggregateRel without measures (a group-by, never DuckDB's LogicalDistinct).
select distinct n_regionkey from nation;
