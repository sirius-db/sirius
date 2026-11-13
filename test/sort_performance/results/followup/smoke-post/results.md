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
| one_shot | false | 1 | ok | 2.3 | 6614 | 12.00 | 0.00 | 0 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| one_shot | true | 1 | ok | 3.1 | 6906 | 10.99 | 0.00 | 1 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| windowed | false | 1 | ok | 8.2 | 5778 | 38.87 | 0.00 | 0 | 0 | 56953 | 1419 | 92-78845 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 7.8 | 5686 | 29.29 | 0.00 | 0 | 0 | 42864 | 761 | 23-70928 | 3140 | 29.29 | - |

## zipf-1key

BIGINT zipf(s=1.1, cardinality 1e9) key + DOUBLE payload; `ORDER BY k`; 101 x 8000000 rows = 808000000 rows, 16 bytes/row, 12.04 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 8.8 | 6934 | 23.06 | 0.00 | 202 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | true | 1 | oom | 13.0 | 7450 | 40.92 | 0.00 | 297 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| windowed | false | 1 | ok | 4.9 | 5714 | 22.42 | 0.00 | 0 | 0 | 30959 | 329 | 294-68519 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 7.8 | 5680 | 31.32 | 0.00 | 0 | 0 | 46786 | 787 | 20-67211 | 2352 | 31.06 | - |

## normal-multikey

INTEGER normal + BIGINT uniform + VARCHAR(8-16) keys; `ORDER BY k1, k2, s`; 58 x 8000000 rows = 464000000 rows, 28 bytes/row, 12.10 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | ok | 2.4 | 6182 | 23.15 | 0.00 | 0 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| one_shot | true | 1 | ok | 4.3 | 6258 | 23.12 | 0.00 | 3 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| windowed | false | 1 | ok | 5.2 | 5316 | 23.13 | 0.00 | 0 | 0 | 17034 | 386 | 95-75071 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 7.7 | 5414 | 33.11 | 0.00 | 0 | 0 | 22089 | 571 | 1-83407 | 1961 | 27.24 | - |

## few_distinct-strings

VARCHAR few_distinct key (18 distinct values, length 18); `ORDER BY s`; 73 x 8000000 rows = 584000000 rows, 22 bytes/row, 11.97 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 67.8 | 5530 | 932.25 | 0.00 | 812 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | true | 1 | oom | 57.5 | 5926 | 570.08 | 0.00 | 970 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| windowed | false | 1 | ok | 7.6 | 4950 | 17.84 | 0.00 | 0 | 0 | 16899 | 173 | 5773-73412 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 7.8 | 4934 | 19.19 | 0.00 | 0 | 0 | 19505 | 330 | 358-69104 | 1630 | 15.58 | - |

## sequential

BIGINT sequential (already sorted) key + DOUBLE payload; `ORDER BY k`; 101 x 8000000 rows = 808000000 rows, 16 bytes/row, 12.04 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 6.0 | 5042 | 10.13 | 0.00 | 200 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | true | 1 | oom | 5.9 | 5076 | 13.58 | 0.00 | 121 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| windowed | false | 1 | ok | 9.4 | 4976 | 25.28 | 0.00 | 0 | 0 | 31117 | 306 | 7000-615385 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 7.5 | 5070 | 17.08 | 0.00 | 0 | 0 | 21741 | 153 | 41709-1925736 | 461 | 17.20 | - |
