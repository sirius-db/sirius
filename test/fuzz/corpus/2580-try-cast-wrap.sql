-- Issue #2580; quarantine: features.casts.integer_narrowing; last failed: 1604e3fe.
CREATE TABLE t (i INTEGER);
INSERT INTO t VALUES (255), (-1);
SELECT TRY_CAST(i AS TINYINT) AS v FROM t ORDER BY i;
