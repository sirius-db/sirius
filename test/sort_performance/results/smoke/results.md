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
| one_shot | false | 1 | ok | 3.2 | 7030 | 11.94 | 0.00 | 0 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| one_shot | true | 1 | ok | 3.1 | 6778 | 11.87 | 0.00 | 1 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| windowed | false | 1 | ok | 5.5 | 6038 | 19.28 | 0.00 | 0 | 0 | 23717 | 230 | 109-69396 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 6.8 | 5846 | 21.49 | 0.00 | 0 | 0 | 28401 | 326 | 68-71930 | 4633 | 21.42 | - |

## zipf-1key

BIGINT zipf(s=1.1, cardinality 1e9) key + DOUBLE payload; `ORDER BY k`; 101 x 8000000 rows = 808000000 rows, 16 bytes/row, 12.04 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 8.8 | 7638 | 23.40 | 0.00 | 203 | - | - | - | - | - | - | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | true | 1 | oom | 11.6 | 7510 | 30.41 | 0.00 | 277 | - | - | - | - | - | - | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| windowed | false | 1 | ok | 6.1 | 5714 | 28.74 | 0.00 | 0 | 0 | 42674 | 774 | 66-82644 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 72.4 | 5776 | 32.07 | 0.00 | 0 | 0 | 665741 | 1421 | 5-71922 | 2342 | 31.96 | - |

## normal-multikey

INTEGER normal + BIGINT uniform + VARCHAR(8-16) keys; `ORDER BY k1, k2, s`; 58 x 8000000 rows = 464000000 rows, 28 bytes/row, 12.10 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | ok | 2.3 | 6536 | 23.01 | 0.00 | 1 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| one_shot | true | 1 | ok | 4.5 | 6612 | 23.55 | 0.00 | 2 | 0 | 0 | 0 | 0-0 | 0 | 0.00 | - |
| windowed | false | 1 | ok | 5.5 | 5316 | 25.62 | 0.00 | 0 | 0 | 20174 | 538 | 444-70551 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 17.9 | 5186 | 37.97 | 0.00 | 0 | 0 | 84282 | 1104 | 49-71693 | 1852 | 32.42 | - |

## few_distinct-strings

VARCHAR few_distinct key (18 distinct values, length 18); `ORDER BY s`; 73 x 8000000 rows = 584000000 rows, 22 bytes/row, 11.97 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 82.3 | 5530 | 876.08 | 0.00 | 902 | - | - | - | - | - | - | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | true | 1 | oom | 81.4 | 5766 | 1002.95 | 0.00 | 1001 | - | - | - | - | - | - | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| windowed | false | 1 | ok | 7.4 | 4822 | 18.91 | 0.00 | 0 | 0 | 16681 | 173 | 642-71751 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 8.1 | 4902 | 21.19 | 0.00 | 0 | 0 | 21512 | 366 | 1248-71323 | 1767 | 17.24 | - |

## sequential

BIGINT sequential (already sorted) key + DOUBLE payload; `ORDER BY k`; 101 x 8000000 rows = 808000000 rows, 16 bytes/row, 12.04 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| one_shot | false | 1 | oom | 6.5 | 5010 | 21.37 | 0.00 | 100 | - | - | - | - | - | - | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| one_shot | true | 1 | oom | 6.0 | 5140 | 17.30 | 0.00 | 113 | - | - | - | - | - | - | Error: CPU fallback is not supported: sirius_generate: stream source has no CPU body. Underlying GPU error: GPU pipeline task exceeded maximum retry limit (1... |
| windowed | false | 1 | ok | 7.2 | 5072 | 19.28 | 0.00 | 0 | 0 | 22330 | 174 | 14399-2092400 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 9.0 | 4976 | 26.92 | 0.00 | 0 | 0 | 29405 | 350 | 2841-615385 | 633 | 26.80 | - |
