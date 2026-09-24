-- A NULL on the build side makes every nonmatching NOT IN probe UNKNOWN.
select n_nationkey
from nation
where n_nationkey in (0, 1, 2)
  and n_nationkey not in (
      select case when r_regionkey = 1 then cast(null as int) else r_regionkey end
      from region where r_regionkey in (0, 1))
order by n_nationkey;
