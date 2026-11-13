# Sort merge benchmark

- Host: `felipe-TRX50-AERO-D`, GPU index 0
- GPU limit (`sirius.memory.gpu.usage_limit_bytes`): 85899345920 bytes (80.00 GiB); data: 0.4x the limit
- `spill_min_batch_bytes` in the compressed arm: 1048576
- Validation (`check_order` + row counts): off
- A `one_shot` run that ends in `oom`, `error`, `timeout` or `crashed` is an outcome of the comparison, not a harness failure.

## normal-multikey

INTEGER normal + BIGINT uniform + VARCHAR(8-16) keys; `ORDER BY k1, k2, s`; 153 x 8000000 rows = 1224000000 rows, 28 bytes/row, 31.92 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 120.9 | 90320 | 15.96 | 0.00 | 115 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | false | 2 | oom | 204.0 | 82544 | 0.00 | 0.00 | 200 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | false | 3 | oom | 204.6 | 89872 | 0.00 | 0.00 | 200 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |

## sequential

BIGINT sequential (already sorted) key + DOUBLE payload; `ORDER BY k`; 268 x 8000000 rows = 2144000000 rows, 16 bytes/row, 31.95 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 67.6 | 86368 | 0.00 | 0.00 | 100 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | false | 2 | oom | 67.5 | 86368 | 0.00 | 0.00 | 100 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | false | 3 | oom | 67.2 | 87454 | 0.00 | 0.00 | 100 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
