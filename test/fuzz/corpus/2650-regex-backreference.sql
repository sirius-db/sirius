-- Issue #2650; quarantine: features.scalar_functions.regexp_backreferences; last failed: 1604e3fe.
CREATE TABLE d (rid INTEGER, v VARCHAR);
INSERT INTO d VALUES (0, 'AA'), (1, '1本😀tj本aOyP6PI O');
SELECT regexp_replace(v, '[A-Z]', '\0\0') AS result FROM d ORDER BY rid;
