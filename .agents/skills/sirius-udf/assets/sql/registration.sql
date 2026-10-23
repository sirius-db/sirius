-- Synthetic deterministic scalar macro; no native build is required.
CREATE MACRO sample_adjust(x) AS (x * x + 0.5::DOUBLE);
