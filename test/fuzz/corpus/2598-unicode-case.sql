-- Issue #2598; quarantine: data.edge_values.unicode_case; last failed: 1604e3fe.
CREATE TABLE t (s VARCHAR);
INSERT INTO t VALUES ('straße'), ('İstanbul');
SELECT upper(s) AS v FROM t ORDER BY s;
