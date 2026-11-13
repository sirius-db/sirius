# Sort merge benchmark

- Host: `felipe-TRX50-AERO-D`, GPU index 0
- GPU limit (`sirius.memory.gpu.usage_limit_bytes`): 4294967296 bytes (4.00 GiB); data: 3x the limit
- `spill_min_batch_bytes` in the compressed arm: 1048576
- Validation (`check_order` + row counts): on
- A `one_shot` run that ends in `oom`, `error`, `timeout` or `crashed` is an outcome of the comparison, not a harness failure.

## uniform-1key

BIGINT uniform key + DOUBLE payload; `ORDER BY k`; 101 x 8000000 rows = 808000000 rows, 16 bytes/row, 12.04 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed | false | 1 | ok | 6.6 | 5476 | 15.95 | 14.13 | 0 | 0 | 23722 | 192 | 131-66725 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 8.6 | 5678 | 18.57 | 7.29 | 0 | 0 | 22095 | 193 | 369-67397 | 7461 | 18.92 | - |

## normal-multikey

INTEGER normal + BIGINT uniform + VARCHAR(8-16) keys; `ORDER BY k1, k2, s`; 58 x 8000000 rows = 464000000 rows, 28 bytes/row, 12.10 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed | false | 1 | ok | 6.7 | 5494 | 19.69 | 14.56 | 0 | 0 | 19709 | 486 | 51-70569 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 13.5 | 5182 | 29.13 | 1.63 | 3 | 0 | 55285 | 1332 | 15-73069 | 1931 | 24.07 | - |
