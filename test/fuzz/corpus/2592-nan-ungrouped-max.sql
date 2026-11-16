-- Issue #2592; quarantine: data.edge_values.nan_inf and features.aggregates.float_division_minmax; last failed: 1604e3fe.
CREATE TABLE t (v FLOAT);
INSERT INTO t VALUES (0.0), (1.0);
SELECT max(v / v) AS hi FROM t;
