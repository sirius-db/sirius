
# 0. clone + data (TPC-H parquet at $TPCH_DATA/<table>/*.parquet)
git clone <repo> && cd <repo>            # branch demo-multi-cn
git submodule update --init --recursive
export TPCH_DATA=/path/to/tpch_sf1

# 1. Engine A — Sirius GPU CNs
cd experimental/starrocks
pixi run cluster2                        # own terminal; first run builds everything
mysql -h127.0.0.1 -P9030 -uroot -e "SHOW COMPUTE NODES;"   # wait for 2x Alive=true
RESTART_CMD='pkill -f "[s]irius-starrocks-cn"; pkill -f "[S]tarRocksFE"; sleep 5; \
  (cd $PWD && nohup pixi run cluster2 >/tmp/c2.log 2>&1 &); sleep 20' \
  ./benchmarks/tpch/bench.sh /tmp/bench/A/timings.csv 3
pkill -f '[s]irius-starrocks-cn'; pkill -f '[S]tarRocksFE'   # down; nvidia-smi = 0 MiB

# 2. Engine B — stock StarRocks (same FE port; A must be down)
JAVA_HOME=/usr/lib/jvm/<jdk17+> ./benchmarks/tpch/setup-engine-b.sh
~/starrocks-bench/fe/bin/start_fe.sh --daemon
~/starrocks-bench/be1/bin/start_be.sh --daemon && ~/starrocks-bench/be2/bin/start_be.sh --daemon
mysql -h127.0.0.1 -P9030 -uroot -e 'ALTER SYSTEM ADD BACKEND "127.0.0.1:9050"; ALTER SYSTEM ADD BACKEND "127.0.0.1:9052";'
./benchmarks/tpch/bench.sh /tmp/bench/B/timings.csv 3        # BEs, not CNs — CNs can't scan FILES()
pkill -f '[s]tarrocks_be'; pkill -f '[S]tarRocksFE'

# 3. Compare → markdown table + log-scale plot
./benchmarks/tpch/analyze.py /tmp/bench/A/timings.csv /tmp/bench/B/timings.csv results.md plot.png

Or skip all of it: TPCH_DATA=/path/to/tpch_sf1 ./benchmarks/tpch/run-comparison.sh does steps 1–3 in one command. Full trap list (readiness gates, port sharing, known flakes) is in benchmarks/tpch/REPRODUCE.md; reference numbers to compare against are in benchmarks/tpch/results/sf1-2026-08-07.md.
