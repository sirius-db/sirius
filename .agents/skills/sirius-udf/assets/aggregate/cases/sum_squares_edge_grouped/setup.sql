-- Independently synthetic fixtures, including duplicate, NULL and singleton groups.
CREATE TABLE aggregate_input (
    id BIGINT, group_key BIGINT, label VARCHAR, value DOUBLE, weight DOUBLE
);
INSERT INTO aggregate_input VALUES
    (0, NULL, NULL, 2, 1), (1, NULL, NULL, 2, 1),
    (2, 0, 'all_null', NULL, NULL), (3, 0, 'all_null', NULL, 4),
    (4, 1, 'singleton', -3, 2),
    (5, 2, 'zero_weight', 5, 0), (6, 2, 'zero_weight', -7, 0),
    (7, 3, 'cancel_weight', 5, 1), (8, 3, 'cancel_weight', 9, -1),
    (9, 4, 'partial_null', NULL, 2), (10, 4, 'partial_null', 7, NULL),
    (11, 4, 'partial_null', 4, 2),
    (12, 5, NULL, 0, 1), (13, 5, NULL, -4, 1),
    (14, 6, 'large_exact', 1048576, 2), (15, 6, 'large_exact', -1048576, 2);
