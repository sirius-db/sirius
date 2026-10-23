CREATE TABLE aggregate_input (
    id BIGINT, group_key BIGINT, label VARCHAR, value DOUBLE, weight DOUBLE
);
INSERT INTO aggregate_input VALUES
    (0, 0, 'fractional', 0.1, 0.3), (1, 0, 'fractional', 0.2, 0.7),
    (2, 0, 'fractional', -0.3, 0.1), (3, 0, 'fractional', 0.5, 0.9);
