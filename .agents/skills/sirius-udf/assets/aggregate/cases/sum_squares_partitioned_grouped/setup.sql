-- Bounded integer-valued doubles preserve exact binary64 partial sums.
-- Shuffled physical order and skewed keys exercise partition merging and batches.
CREATE TABLE aggregate_input AS
SELECT i::BIGINT AS id,
       CASE WHEN i % 101 = 0 THEN NULL::BIGINT
            WHEN i % 5 <> 0 THEN 0::BIGINT ELSE (i % 1024)::BIGINT END AS group_key,
       CASE WHEN i % 107 = 0 THEN NULL::VARCHAR
            WHEN i % 2 = 0 THEN 'even' ELSE 'odd' END AS label,
       CASE WHEN i % 13 = 0 THEN NULL::DOUBLE
            ELSE ((i % 97) - 48)::DOUBLE END AS value,
       CASE WHEN i % 17 = 0 THEN NULL::DOUBLE
            ELSE ((i % 7) + 1)::DOUBLE END AS weight
FROM range(@ROWS@) AS source(i)
ORDER BY (i * 104729) % @ROWS@;
