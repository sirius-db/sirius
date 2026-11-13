SET gpu_execution = true;
SET sort_merge_mode = 'windowed';
SET spill_compression = true;
COPY (SELECT * FROM sirius_generate(
  num_batches := 101,
  rows_per_batch := 8000000,
  seed := 42,
  columns := [
    {name: 'k', type: 'BIGINT', dist: 'zipf', p1: 1.1, cardinality: 1000000000},
    {name: 'v', type: 'DOUBLE', dist: 'uniform', p1: 0, p2: 1}
  ]
) ORDER BY k)
TO '/dev/null' (FORMAT sirius_discard, check_order true);
