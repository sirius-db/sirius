# Sort merge benchmark

- Host: `felipe-TRX50-AERO-D`, GPU index 0
- GPU limit (`sirius.memory.gpu.usage_limit_bytes`): 85899345920 bytes (80.00 GiB); data: 2x the limit
- `spill_min_batch_bytes` in the compressed arm: 1048576
- Validation (`check_order` + row counts): off
- A `one_shot` run that ends in `oom`, `error`, `timeout` or `crashed` is an outcome of the comparison, not a harness failure.

## uniform-1key

BIGINT uniform key + DOUBLE payload; `ORDER BY k`; 1342 x 8000000 rows = 10736000000 rows, 16 bytes/row, 159.98 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 449.0 | 97246 | 310.30 | 142.98 | 201 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |

## zipf-1key

BIGINT zipf(s=1.1, cardinality 1e9) key + DOUBLE payload; `ORDER BY k`; 1342 x 8000000 rows = 10736000000 rows, 16 bytes/row, 159.98 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 770.5 | 97246 | 9813.95 | 194.81 | 400 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
