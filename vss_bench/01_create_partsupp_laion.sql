-- Generate a 1M-row partsupp_laion mirror for single-table vector-search benchmarking.
-- Real Exqutor schema: TPC-H partsupp columns + two embedding columns
--   ps_image_embedding FLOAT[96]   (DEEP-style image vectors)
--   ps_text_embedding  FLOAT[768]  (WIKI-style text vectors)
-- Vector *values* are random: brute-force VSS runtime is data-independent, so random
-- vectors give the same timing (and exact, recall=1.0 results) as the real LAION/DEEP sets.
--
-- Run ONCE, on CPU (data generation needs no GPU). It writes partsupp_laion.duckdb:
--   ./build/release/duckdb partsupp_laion.duckdb -unsigned -f vss_bench/01_create_partsupp_laion.sql

SET gpu_execution = false;       -- build the data on CPU; avoids GPU interception
SELECT setseed(0.42);            -- reproducible random vectors

DROP TABLE IF EXISTS partsupp_laion;

CREATE TABLE partsupp_laion AS
SELECT
  (i % 200000)::INT + 1                                AS ps_partkey,
  (i % 10000)::INT + 1                                 AS ps_suppkey,
  (i % 9999)::INT + 1                                  AS ps_availqty,
  (random() * 1000.0)::DOUBLE                          AS ps_supplycost,
  apply(range(96),  x -> random()::FLOAT)::FLOAT[96]   AS ps_image_embedding,
  apply(range(768), x -> random()::FLOAT)::FLOAT[768]  AS ps_text_embedding
FROM range(500) t(i);

CHECKPOINT;   -- persist to on-disk blocks so Sirius's native GPU scan can read it

SELECT count(*) AS rows,
       count(ps_image_embedding) AS img_vecs,
       count(ps_text_embedding)  AS txt_vecs
FROM partsupp_laion;
