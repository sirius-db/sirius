# Sort merge benchmark

- Host: `felipe-TRX50-AERO-D`, GPU index 0
- GPU limit (`sirius.memory.gpu.usage_limit_bytes`): 25769803776 bytes (24.00 GiB); data: 10x the limit
- `spill_min_batch_bytes` in the compressed arm: 1048576
- Validation (`check_order` + row counts): off
- A `one_shot` run that ends in `oom`, `error`, `timeout` or `crashed` is an outcome of the comparison, not a harness failure.

## uniform-1key

BIGINT uniform key + DOUBLE payload; `ORDER BY k`; 2013 x 8000000 rows = 16104000000 rows, 16 bytes/row, 239.97 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed | false | 1 | ok | 423.0 | 30972 | 397.37 | 300.87 | 0 | 0 | 229677 | 4464 | 10-204704 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 306.0 | 35512 | 464.20 | 144.18 | 0 | 0 | 402460 | 4464 | 1-150804 | 320649 | 450.38 | - |

## zipf-1key

BIGINT zipf(s=1.1, cardinality 1e9) key + DOUBLE payload; `ORDER BY k`; 2013 x 8000000 rows = 16104000000 rows, 16 bytes/row, 239.97 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed | false | 1 | ok | 465.3 | 29660 | 470.60 | 326.74 | 0 | 0 | 558157 | 5660 | 1-104776 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 203.9 | 29750 | 484.82 | 67.52 | 0 | 0 | 489693 | 5003 | 3-138061 | 219253 | 484.74 | - |

## normal-multikey

INTEGER normal + BIGINT uniform + VARCHAR(8-16) keys; `ORDER BY k1, k2, s`; 1150 x 8000000 rows = 9200000000 rows, 28 bytes/row, 239.91 GiB.

| mode | compression | rep | status | wall s | peak GPU MiB | to host GiB | to disk GiB | OOM retries | shrinks | rounds | intermediate tasks | rows/round min-max | runs read compressed | compressed refill GiB | error |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| windowed | false | 1 | ok | 614.5 | 29454 | 474.18 | 313.60 | 0 | 0 | 326955 | 5329 | 3-94258 | 0 | 0.00 | - |
| windowed | true | 1 | ok | 500.4 | 29512 | 481.62 | 66.36 | 5 | 5 | 274947 | 4337 | 5-140938 | 274718 | 412.09 | - |
