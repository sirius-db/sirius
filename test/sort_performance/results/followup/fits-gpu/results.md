# Sort merge benchmark

- Host: `felipe-TRX50-AERO-D`, GPU index 0
- GPU limit (`sirius.memory.gpu.usage_limit_bytes`): 85899345920 bytes (80.00 GiB); data: 0.4x the limit
- `spill_min_batch_bytes` in the compressed arm: 1048576
- Validation (`check_order` + row counts): off
- A `one_shot` run that ends in `oom`, `error`, `timeout` or `crashed` is an outcome of the comparison, not a harness failure.

## uniform-1key

BIGINT uniform key + DOUBLE payload; `ORDER BY k`; 268 x 8000000 rows = 2144000000 rows, 16 bytes/row, 31.95 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | ok | 2.7 | 83006 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| one_shot | false | 2 | ok | 2.7 | 82526 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| one_shot | false | 3 | ok | 2.7 | 82686 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| windowed | false | 1 | ok | 3.5 | 82526 | 0.00 | 0.00 | 0 | 0 | 2102 | 80 | 5721-2099414 | 0 | 0.00 | - |
| windowed | false | 2 | ok | 3.5 | 82526 | 0.00 | 0.00 | 0 | 0 | 2102 | 80 | 5721-2099414 | 0 | 0.00 | - |
| windowed | false | 3 | ok | 3.5 | 82526 | 0.00 | 0.00 | 0 | 0 | 2102 | 80 | 5721-2099414 | 0 | 0.00 | - |
| auto | false | 1 | ok | 3.5 | 82526 | 0.00 | 0.00 | 0 | 0 | 2102 | 80 | 4204-2099414 | 0 | 0.00 | - |
| auto | false | 2 | ok | 3.5 | 82526 | 0.00 | 0.00 | 0 | 0 | 2102 | 80 | 5721-2099414 | 0 | 0.00 | - |
| auto | false | 3 | ok | 3.5 | 82526 | 0.00 | 0.00 | 0 | 0 | 2102 | 80 | 5721-2099414 | 0 | 0.00 | - |

## zipf-1key

BIGINT zipf(s=1.1, cardinality 1e9) key + DOUBLE payload; `ORDER BY k`; 268 x 8000000 rows = 2144000000 rows, 16 bytes/row, 31.95 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | ok | 2.2 | 82526 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| one_shot | false | 2 | ok | 2.2 | 82526 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| one_shot | false | 3 | ok | 2.2 | 82526 | 0.00 | 0.00 | 0 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| windowed | false | 1 | ok | 2.7 | 82526 | 0.00 | 0.00 | 0 | 0 | 2088 | 80 | 23341-2099414 | 0 | 0.00 | - |
| windowed | false | 2 | ok | 2.7 | 82526 | 0.00 | 0.00 | 0 | 0 | 2088 | 80 | 23341-2099414 | 0 | 0.00 | - |
| windowed | false | 3 | ok | 2.7 | 82526 | 0.00 | 0.00 | 0 | 0 | 2088 | 80 | 23341-2099414 | 0 | 0.00 | - |
| auto | false | 1 | ok | 2.7 | 82526 | 0.00 | 0.00 | 0 | 0 | 2088 | 80 | 23341-2099414 | 0 | 0.00 | - |
| auto | false | 2 | ok | 2.7 | 82526 | 0.00 | 0.00 | 0 | 0 | 2088 | 80 | 23341-2099414 | 0 | 0.00 | - |
| auto | false | 3 | ok | 2.7 | 82526 | 0.00 | 0.00 | 0 | 0 | 2088 | 80 | 23341-2099414 | 0 | 0.00 | - |

## normal-multikey

INTEGER normal + BIGINT uniform + VARCHAR(8-16) keys; `ORDER BY k1, k2, s`; 153 x 8000000 rows = 1224000000 rows, 28 bytes/row, 31.92 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | error | 114.0 | 90388 | 15.96 | 0.00 | 110 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: std::bad_cast |
| one_shot | false | 2 | oom | 147.2 | 89966 | 15.96 | 0.00 | 141 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | false | 3 | oom | 203.5 | 82544 | 0.00 | 0.00 | 200 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| windowed | false | 1 | ok | 6.7 | 82544 | 0.00 | 0.00 | 0 | 0 | 1968 | 70 | 127948-1200411 | 0 | 0.00 | - |
| windowed | false | 2 | ok | 6.7 | 82544 | 0.00 | 0.00 | 0 | 0 | 1968 | 70 | 3075-1200411 | 0 | 0.00 | - |
| windowed | false | 3 | ok | 6.7 | 82544 | 0.00 | 0.00 | 0 | 0 | 1969 | 70 | 2740-1200411 | 0 | 0.00 | - |
| auto | false | 1 | ok | 6.7 | 82544 | 0.00 | 0.00 | 0 | 0 | 1968 | 70 | 127948-1200411 | 0 | 0.00 | - |
| auto | false | 2 | ok | 6.7 | 82544 | 0.00 | 0.00 | 0 | 0 | 1969 | 70 | 2740-1200411 | 0 | 0.00 | - |
| auto | false | 3 | ok | 6.7 | 82544 | 0.00 | 0.00 | 0 | 0 | 1969 | 70 | 2740-1200411 | 0 | 0.00 | - |

## few_distinct-strings

VARCHAR few_distinct key (18 distinct values, length 18); `ORDER BY s`; 195 x 8000000 rows = 1560000000 rows, 22 bytes/row, 31.96 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 161.1 | 82748 | 15.98 | 0.00 | 199 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | false | 2 | oom | 163.6 | 82724 | 34.87 | 0.00 | 200 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | false | 3 | oom | 160.8 | 82526 | 0.00 | 0.00 | 200 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| windowed | false | 1 | ok | 14.2 | 82526 | 0.00 | 0.00 | 0 | 0 | 2219 | 87 | 3229-1460010 | 0 | 0.00 | - |
| windowed | false | 2 | ok | 14.2 | 82526 | 0.00 | 0.00 | 0 | 0 | 2219 | 87 | 3229-1460010 | 0 | 0.00 | - |
| windowed | false | 3 | ok | 14.2 | 82526 | 0.00 | 0.00 | 0 | 0 | 2219 | 87 | 3229-1460010 | 0 | 0.00 | - |
| auto | false | 1 | ok | 14.2 | 82526 | 0.00 | 0.00 | 0 | 0 | 2219 | 87 | 3229-1460010 | 0 | 0.00 | - |
| auto | false | 2 | ok | 14.2 | 82526 | 0.00 | 0.00 | 0 | 0 | 2219 | 87 | 3229-1460010 | 0 | 0.00 | - |
| auto | false | 3 | ok | 14.2 | 82526 | 0.00 | 0.00 | 0 | 0 | 2219 | 87 | 3229-1460010 | 0 | 0.00 | - |

## sequential

BIGINT sequential (already sorted) key + DOUBLE payload; `ORDER BY k`; 268 x 8000000 rows = 2144000000 rows, 16 bytes/row, 31.95 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | error | 1.2 | 86688 | 0.00 | 0.00 | 1 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: std::bad_cast |
| one_shot | false | 2 | error | 1.2 | 86304 | 0.00 | 0.00 | 1 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: std::bad_cast |
| one_shot | false | 3 | error | 1.2 | 86528 | 0.00 | 0.00 | 1 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: std::bad_cast |
| windowed | false | 1 | ok | 1.4 | 82526 | 0.00 | 0.00 | 0 | 0 | 2057 | 80 | 1759270-14929208 | 0 | 0.00 | - |
| windowed | false | 2 | ok | 1.4 | 82526 | 0.00 | 0.00 | 0 | 0 | 2057 | 80 | 1759270-14929208 | 0 | 0.00 | - |
| windowed | false | 3 | ok | 1.4 | 82526 | 0.00 | 0.00 | 0 | 0 | 2057 | 80 | 1759270-14929208 | 0 | 0.00 | - |
| auto | false | 1 | ok | 1.4 | 82526 | 0.00 | 0.00 | 0 | 0 | 2052 | 80 | 1062940-2010878 | 0 | 0.00 | - |
| auto | false | 2 | ok | 1.4 | 82526 | 0.00 | 0.00 | 0 | 0 | 2052 | 80 | 1062940-2010878 | 0 | 0.00 | - |
| auto | false | 3 | ok | 1.4 | 82526 | 0.00 | 0.00 | 0 | 0 | 2052 | 80 | 1062940-2010878 | 0 | 0.00 | - |
