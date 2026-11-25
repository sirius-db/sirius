# Selected M9gd DuckDB results

Selected paired measurements, with each iteration summed independently. Earlier dated measurements remain preserved.

| Scale | Iteration 1 sum (s) | Iteration 2 sum (s) | Source |
| --- | ---: | ---: | --- |
| SF100 | 33.992618 | 8.177413 | [Native / 64 KiB](20261125-native64k/RESULTS.md) |
| SF300 | 67.280689 | 16.110636 | [Native / 64 KiB](20261125-native64k/RESULTS.md) |
| SF1000 | 375.285785 | 50.114462 | [Native / 64 KiB](20261125-native64k/RESULTS.md) |
| SF3000 | 1241.309344 | 325.166412 | [Native / 64 KiB](20261125-native64k/RESULTS.md) |
| SF10000 | 5496.954199 | 3788.048400 | [Native / 64 KiB](20261125-native64k/RESULTS.md); **Q12 256 GiB**, others 296.7 GiB; assembled with retries |

SF10000 is not a uniform-memory full-suite run. Its per-query source and memory budget are explicit in the dated records. There is no mixing of per-query minima. SF100/SF300 are new CPU measurements; SF1000/SF3000/SF10000 replace the earlier higher sums.
