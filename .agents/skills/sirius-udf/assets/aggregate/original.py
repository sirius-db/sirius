# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Independent synthetic aggregate sources: update, partial-state merge, finalize.

These are executable CPU originals, not SQL rewrites or customer implementations.
Both accept finite doubles in the supplied cases. Values are bounded so exact
binary64 integer accumulation remains representable for the generated data.
"""


class SumSquares:
    def __init__(self):
        self.total = 0.0
        self.seen = False

    def update(self, value):
        if value is not None:
            self.total += value * value
            self.seen = True

    def merge(self, other):
        self.total += other.total
        self.seen = self.seen or other.seen

    def finalize(self):
        return self.total if self.seen else None


class WeightedMean:
    def __init__(self):
        self.numerator = 0.0
        self.denominator = 0.0
        self.seen = False

    def update(self, value, weight):
        if value is not None and weight is not None:
            self.numerator += value * weight
            self.denominator += weight
            self.seen = True

    def merge(self, other):
        self.numerator += other.numerator
        self.denominator += other.denominator
        self.seen = self.seen or other.seen

    def finalize(self):
        if not self.seen or self.denominator == 0.0:
            return None
        return self.numerator / self.denominator
