-- NOT EXISTS ignores NULL build keys and retains rows 1 and 2.
select n.n_nationkey
from nation n
where n.n_nationkey in (0, 1, 2)
  and not exists (
      select 1 from region r
      where (case when r.r_regionkey = 1 then cast(null as int) else r.r_regionkey end)
            = n.n_nationkey)
order by n.n_nationkey;
