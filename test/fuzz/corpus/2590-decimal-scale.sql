-- Issue #2590; quarantine: features.casts.decimal_scale; last failed: 1604e3fe.
CREATE TABLE d (rid INTEGER, v DECIMAL(18,18));
INSERT INTO d VALUES (0, 0.999999999999999999), (1, -0.999999999999999999);
SELECT CAST(v AS DECIMAL(18,4)) AS result FROM d ORDER BY rid;
