SET gpu_execution = true;
SET sort_merge_mode = 'windowed';
SET spill_compression = false;
COPY (SELECT * FROM sirius_generate(
  num_batches := 73,
  rows_per_batch := 8000000,
  seed := 42,
  columns := [
    {name: 's', type: 'VARCHAR', dist: 'few_distinct', p1: 18, p2: 18}
  ]
) ORDER BY s)
TO '/dev/null' (FORMAT sirius_discard, check_order true);
