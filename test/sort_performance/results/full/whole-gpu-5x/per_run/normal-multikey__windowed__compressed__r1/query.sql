SET gpu_execution = true;
SET sort_merge_mode = 'windowed';
SET spill_compression = true;
COPY (SELECT * FROM sirius_generate(
  num_batches := 2253,
  rows_per_batch := 8000000,
  seed := 42,
  columns := [
    {name: 'k1', type: 'INTEGER', dist: 'normal', p1: 0, p2: 1000000},
    {name: 'k2', type: 'BIGINT', dist: 'uniform', p1: 0, p2: 1000000000000000},
    {name: 's', type: 'VARCHAR', dist: 'uniform', p1: 8, p2: 16}
  ]
) ORDER BY k1, k2, s)
TO '/dev/null' (FORMAT sirius_discard);
