-- Issue #2590; quarantine: mixed-type DECIMAL COALESCE arms; last failed: 1604e3fe.
CREATE TABLE d (rid INTEGER, v DECIMAL(18,4));
INSERT INTO d VALUES (0, -668140.6850), (1, 35609.9250);
SELECT coalesce(v, (SELECT sum(CAST(99.99 AS DECIMAL(12,2))) FROM d)) AS result FROM d ORDER BY rid;
