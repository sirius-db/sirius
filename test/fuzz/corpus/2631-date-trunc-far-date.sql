-- Issue #2631; quarantine: data.edge_values.date_beyond_ns; last failed: 1604e3fe.
CREATE TABLE t (d DATE);
INSERT INTO t VALUES (DATE '9999-12-31');
SELECT date_trunc('minute', CAST(d AS TIMESTAMP)) AS v FROM t;
