-- Issue #2085; quarantine: data.edge_values.nan_inf; last failed: 1604e3fe.
CREATE TABLE t (g INTEGER, v FLOAT);
INSERT INTO t VALUES (1, CAST('NaN' AS FLOAT)), (1, 2.0), (1, CAST('Infinity' AS FLOAT)), (2, CAST('NaN' AS FLOAT)), (2, -1.0);
SELECT g, min(v) AS lo, max(v) AS hi FROM t GROUP BY g ORDER BY g;
