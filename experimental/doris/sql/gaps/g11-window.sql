-- G-11: window functions plan to ANALYTIC_EVAL_NODE, which the translator names in its
-- rejection (Sirius has no window operator).
select n_name, row_number() over (order by n_nationkey) as rn from nation;
