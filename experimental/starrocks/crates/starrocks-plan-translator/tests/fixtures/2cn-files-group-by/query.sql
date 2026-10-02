-- FE + 2 CNs, SIRIUS_CN_TRANSLATE_ONLY + SIRIUS_CN_DUMP_FRAGMENTS.
-- Resulting fragments: FileScan + partial SUM (HASH N=2) → merge SUM → gather EXCHANGE.
SELECT region, SUM(amount)
FROM FILES("path"="file:///opt/dlami/nvme/tmp/sirius-2cn-plan-dump/sales/sales_*.parquet","format"="parquet")
GROUP BY region;
