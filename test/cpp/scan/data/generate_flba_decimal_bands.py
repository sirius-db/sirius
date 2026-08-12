# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
"""Regenerate the FIXED_LEN_BYTE_ARRAY decimal fixtures used by the parquet pushdown tests.

The fixtures exist because no writer available to the tests themselves can produce a NARROW
FIXED_LEN_BYTE_ARRAY decimal. DuckDB stores every decimal wider than 18 digits as a 16-byte
FIXED_LEN_BYTE_ARRAY regardless of precision -- DECIMAL(19,2), DECIMAL(25,2) and DECIMAL(38,4)
all come out 16 bytes -- so a DuckDB-written fixture can only ever exercise that one width.
pyarrow instead sizes the field to the precision, giving 4-, 8- and 16-byte decimals, which is
what makes sign extension observable at each width rather than only at the widest one.

Each file carries `id` plus one `amount` column whose row groups hold disjoint bands:
row group g covers [(g - 5) * 10000, (g - 5) * 10000 + 2047], so the first five row groups are
entirely negative. Every 97th amount is NULL. A predicate selecting negative amounts must keep
exactly the first five row groups, which is only possible when the statistics for a negative
band keep their sign.

Run with any environment providing pyarrow:

    python test/cpp/scan/data/generate_flba_decimal_bands.py
"""

from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROWS_PER_GROUP = 2048
GROUP_COUNT = 10
NEGATIVE_GROUPS = 5
NULL_EVERY = 97

# (file stem, precision, scale, expected FIXED_LEN_BYTE_ARRAY byte width)
FIXTURES = (
    ("flba4_decimal_bands", 9, 2, 4),
    ("flba8_decimal_bands", 18, 2, 8),
    ("flba16_decimal_bands", 38, 4, 16),
)


def main() -> None:
    out_dir = Path(__file__).parent
    row_count = ROWS_PER_GROUP * GROUP_COUNT
    ids = list(range(row_count))
    raw = [
        (
            None
            if i % NULL_EVERY == 0
            else ((i // ROWS_PER_GROUP) - NEGATIVE_GROUPS) * 10000
            + (i % ROWS_PER_GROUP)
        )
        for i in ids
    ]

    for stem, precision, scale, expected_width in FIXTURES:
        quantum = Decimal(1).scaleb(-scale)
        amounts = [None if v is None else Decimal(v).quantize(quantum) for v in raw]
        table = pa.table(
            {
                "id": pa.array(ids, pa.int32()),
                "amount": pa.array(amounts, pa.decimal128(precision, scale)),
            }
        )
        path = out_dir / f"{stem}.parquet"
        # store_decimal_as_integer defaults to False; stated because writing the decimal as an
        # integer instead would defeat the entire purpose of these fixtures.
        pq.write_table(
            table,
            path,
            row_group_size=ROWS_PER_GROUP,
            compression="zstd",
            store_decimal_as_integer=False,
        )

        # The point of the fixture is the physical encoding, so verify it rather than trust it.
        column = pq.ParquetFile(path).schema.column(1)
        assert column.physical_type == "FIXED_LEN_BYTE_ARRAY", column.physical_type
        assert (
            column.length == expected_width
        ), f"{stem}: got {column.length}, want {expected_width}"
        print(
            f"{path.name}: DECIMAL({precision},{scale}) as FIXED_LEN_BYTE_ARRAY({column.length})"
        )


if __name__ == "__main__":
    main()
