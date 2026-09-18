-- G-13: DISTINCT over two keys with ORDER BY + LIMIT (top-n over the folded group-by, then a
-- merging exchange).
select distinct n_regionkey, n_name from nation order by n_regionkey limit 3;
