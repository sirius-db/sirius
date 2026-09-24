-- Outer-join padding must retain rows 1 and 2 even though the build side contains NULL.
select n.n_nationkey
from nation n
left join (
    select case when r_regionkey = 1 then cast(null as int) else r_regionkey end as k
    from region where r_regionkey in (0, 1)) r on n.n_nationkey = r.k
where n.n_nationkey in (0, 1, 2) and r.k is null
order by n.n_nationkey;
