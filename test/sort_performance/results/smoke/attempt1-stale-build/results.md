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
| one_shot | false | 1 | error | 0.0 | 4696 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| one_shot | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | false | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |

## zipf-1key

BIGINT zipf(s=1.1, cardinality 1e9) key + DOUBLE payload; `ORDER BY k`; 101 x 8000000 rows = 808000000 rows, 16 bytes/row, 12.04 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| one_shot | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | false | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |

## normal-multikey

INTEGER normal + BIGINT uniform + VARCHAR(8-16) keys; `ORDER BY k1, k2, s`; 58 x 8000000 rows = 464000000 rows, 28 bytes/row, 12.10 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| one_shot | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | false | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |

## few_distinct-strings

VARCHAR few_distinct key (18 distinct values, length 18); `ORDER BY s`; 73 x 8000000 rows = 584000000 rows, 22 bytes/row, 11.97 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| one_shot | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | false | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |

## sequential

BIGINT sequential (already sorted) key + DOUBLE payload; `ORDER BY k`; 101 x 8000000 rows = 808000000 rows, 16 bytes/row, 12.04 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| one_shot | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | false | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
| windowed | true | 1 | error | 0.0 | 4682 | 0.00 | 0.00 | 0 | - | - | - | - | - | - | InvalidInputException: Invalid Input Error: sirius_generate requires Sirius GPU execution (SET gpu_execution = true) |
