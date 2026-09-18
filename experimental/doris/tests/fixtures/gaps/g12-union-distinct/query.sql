-- G-12: UNION (distinct) plans to a two-phase group-by over UNION_NODE; the UNION_NODE is
-- still what gets rejected.
select n_nationkey as k from nation union select r_regionkey from region;
