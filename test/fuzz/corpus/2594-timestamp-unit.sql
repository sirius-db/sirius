-- Issue #2594; quarantine: features.casts.timestamp_units; last failed: 1604e3fe.
CREATE TABLE t (ts TIMESTAMP);
INSERT INTO t VALUES (TIMESTAMP '2020-01-01 23:59:59.999999');
SELECT CAST(ts AS TIMESTAMP_S) AS v FROM t;
