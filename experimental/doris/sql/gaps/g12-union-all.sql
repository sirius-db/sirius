-- G-12: UNION ALL plans to UNION_NODE, rejected by name (the DuckDB Substrait consumer takes
-- two-input SetRels only and Sirius has no set operator).
select n_nationkey as k from nation union all select r_regionkey from region;
