-- Issue #2152; quarantine: features.casts.float_to_int; last failed: 1604e3fe.
CREATE TABLE t (f FLOAT);
INSERT INTO t VALUES (-1.5), (2.5);
SELECT CAST(f AS BIGINT) AS v FROM t ORDER BY f;
