# Sort merge benchmark

- Host: `felipe-TRX50-AERO-D`, GPU index 0
- GPU limit (`sirius.memory.gpu.usage_limit_bytes`): 4294967296 bytes (4.00 GiB); data: 10x the limit
- `spill_min_batch_bytes` in the compressed arm: 1048576
- Validation (`check_order` + row counts): on
- A `one_shot` run that ends in `oom`, `error`, `timeout` or `crashed` is an outcome of the comparison, not a harness failure.

## uniform-1key

BIGINT uniform key + DOUBLE payload; `ORDER BY k`; 336 x 8000000 rows = 2688000000 rows, 16 bytes/row, 40.05 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed | false | 1 | ok | 47.4 | 5860 | 91.77 | 115.98 | 0 | 0 | 137467 | 3801 | 8-77690 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 57.9 | 6094 | 159.08 | 88.50 | 3 | 3 | 169238 | 5270 | 23-77443 | 21975 | 158.80 | - |

## zipf-1key

BIGINT zipf(s=1.1, cardinality 1e9) key + DOUBLE payload; `ORDER BY k`; 336 x 8000000 rows = 2688000000 rows, 16 bytes/row, 40.05 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed | false | 1 | ok | 30.9 | 5860 | 52.02 | 76.59 | 1 | 1 | 89346 | 917 | 278-68351 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 94.9 | 5998 | 248.73 | 84.69 | 0 | 0 | 402046 | 12547 | 2-71877 | 24766 | 248.54 | - |

## normal-multikey

INTEGER normal + BIGINT uniform + VARCHAR(8-16) keys; `ORDER BY k1, k2, s`; 192 x 8000000 rows = 1536000000 rows, 28 bytes/row, 40.05 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed | false | 1 | ok | 48.4 | 5814 | 86.18 | 116.80 | 0 | 0 | 79606 | 3979 | 15-78035 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 241.7 | 5760 | 153.76 | 45.95 | 3 | 3 | 924297 | 12123 | 1-82291 | 28249 | 130.61 | - |
