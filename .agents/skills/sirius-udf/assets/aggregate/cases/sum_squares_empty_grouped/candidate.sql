SELECT group_key, label, sum_squares(value) AS result FROM aggregate_input
GROUP BY group_key, label ORDER BY group_key NULLS FIRST, label NULLS FIRST;
