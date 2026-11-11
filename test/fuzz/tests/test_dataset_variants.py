"""Physical dataset choices stay stable and replayable."""

import unittest

from siriusfuzz.config import Data, Parquet, Pin
from siriusfuzz.dataset_variants import choose, comment, from_dataset_sql


class DatasetVariantTests(unittest.TestCase):
    def test_choices_are_replayable_and_pins_target_tables(self):
        data = Data(
            pin=Pin(tables_fraction=0.5),
            parquet=Parquet(tables_fraction=0.5),
        )
        names = [f"t{i}" for i in range(6)]
        first = choose(data, names, 20261111)
        self.assertEqual(first, choose(data, names, 20261111))
        self.assertFalse(set(first["pin"]) & set(first["parquet"]))
        self.assertEqual(set(first["pin"]) | set(first["parquet"]), set(names))
        self.assertEqual(from_dataset_sql(comment(first) + "CREATE TABLE t0(k INT);"), first)


if __name__ == "__main__":
    unittest.main()
