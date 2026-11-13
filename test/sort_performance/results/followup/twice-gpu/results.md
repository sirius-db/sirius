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
| windowed | false | 1 | ok | 61.3 | 95998 | 168.38 | 41.41 | 6 | 6 | 64466 | 2569 | 30-419265 | 0 | 0.00 | - |
| auto | false | 1 | ok | 54.4 | 97246 | 153.10 | 42.63 | 46 | 46 | 53210 | 2160 | 32-515217 | 0 | 0.00 | - |
| one_shot | false | 1 | error | 128.1 | 97246 | 197.03 | 118.80 | 5 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: std::bad_cast |

## zipf-1key

BIGINT zipf(s=1.1, cardinality 1e9) key + DOUBLE payload; `ORDER BY k`; 1342 x 8000000 rows = 10736000000 rows, 16 bytes/row, 159.98 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed | false | 1 | ok | 67.2 | 97246 | 144.03 | 27.32 | 19 | 19 | 172990 | 2863 | 108-156382 | 0 | 0.00 | - |
| auto | false | 1 | ok | 56.1 | 97246 | 142.68 | 26.24 | 69 | 69 | 124904 | 2867 | 62-218083 | 0 | 0.00 | - |
| one_shot | false | 1 | oom | 769.5 | 97246 | 9838.01 | 172.02 | 401 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
