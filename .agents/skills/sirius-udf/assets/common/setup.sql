-- Independent synthetic data. The runner replaces @ROWS@ before executing.
CREATE TABLE sample_input AS
SELECT i::BIGINT AS id,
       CASE WHEN i % 13 = 0 THEN NULL
            ELSE ((i % 97) - 48)::DOUBLE / 4.0 END AS value
FROM range(@ROWS@) AS input(i);
