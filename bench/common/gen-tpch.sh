#!/usr/bin/env bash
# Generate a TPC-H parquet dataset at any scale factor.
#
# Defaults to f64 monetary columns. decimal128 activates Sirius's decimal-lowering drift,
# which scores ~11/22 against a DuckDB oracle instead of ~21/22, so f64 is what the
# benchmark baselines use. Pass --decimal only when you are deliberately testing that path.
set -euo pipefail

SCALE=""
DECIMAL_TYPE=f64
OUT=""
ROOT=${TPCH_ROOT:-/opt/dlami/nvme/tpch}
GEN=${TPCHGEN:-$ROOT/tpchgen-rs/target/release/tpchgen-cli}
PYTHON=${TPCH_PYTHON:-$ROOT/venv/bin/python}
TARGET_PART_GB=${TARGET_PART_GB:-4.3}
FORCE=0
VERIFY=1

usage() {
    cat <<USAGE
usage: gen-tpch.sh -s <scale> [options]

  -s, --scale <N>       scale factor (required)
  -o, --output <dir>    output directory
                        (default: \$TPCH_ROOT/tpch_parquet_sf<N>[_f64])
      --decimal         decimal128 monetary columns (default: f64)
      --force           overwrite an existing output directory
      --no-verify       skip the row-count check
  -h, --help

env: TPCH_ROOT (default /opt/dlami/nvme/tpch), TPCHGEN, TPCH_PYTHON, TARGET_PART_GB
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        -s|--scale)   SCALE=${2:?}; shift 2 ;;
        -o|--output)  OUT=${2:?}; shift 2 ;;
        --decimal)    DECIMAL_TYPE=decimal128; shift ;;
        --force)      FORCE=1; shift ;;
        --no-verify)  VERIFY=0; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die() { echo "FATAL: $*" >&2; exit 1; }

[ -n "$SCALE" ] || { usage >&2; die "-s/--scale is required"; }
[[ "$SCALE" =~ ^[0-9]+$ ]] && [ "$SCALE" -gt 0 ] || die "scale must be a positive integer, got '$SCALE'"
[ -x "$GEN" ] || die "tpchgen-cli not executable at $GEN (set TPCHGEN)"

if [ -z "$OUT" ]; then
    OUT=$ROOT/tpch_parquet_sf$SCALE
    [ "$DECIMAL_TYPE" = f64 ] && OUT=${OUT}_f64
fi

if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    [ "$FORCE" = 1 ] || die "$OUT exists and is not empty; pass --force to overwrite"
    rm -rf "$OUT"
fi
mkdir -p "$OUT"

# Approximate compressed GB per unit of scale factor, measured from the SF300 and SF500 sets.
# Part counts derived from these reproduce the existing SF100/SF300/SF500 layouts.
gb_per_sf() {
    case "$1" in
        lineitem) echo 0.258 ;;
        orders)   echo 0.075 ;;
        partsupp) echo 0.047 ;;
        customer) echo 0.0138 ;;
        part)     echo 0.0068 ;;
        supplier) echo 0.0009 ;;
        *)        echo 0 ;;
    esac
}

parts_for() {
    local table=$1
    case "$table" in nation|region) echo 1; return ;; esac
    "$PYTHON" - "$(gb_per_sf "$table")" "$SCALE" "$TARGET_PART_GB" <<'PY'
import sys
per_sf, scale, target = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
print(max(1, round(per_sf * scale / target)))
PY
}

TABLES="region nation supplier part customer partsupp orders lineitem"

echo "scale=$SCALE  decimal=$DECIMAL_TYPE  out=$OUT"
for t in $TABLES; do
    n=$(parts_for "$t")
    echo "[$(date +%T)] $t parts=$n"
    "$GEN" -s "$SCALE" -T "$t" -f parquet --parts "$n" \
           --decimal-column-type "$DECIMAL_TYPE" -o "$OUT"
done

cat > "$OUT/gen-info.json" <<JSON
{
  "scale_factor": $SCALE,
  "decimal_column_type": "$DECIMAL_TYPE",
  "target_part_gb": $TARGET_PART_GB,
  "generated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "generator": "$($GEN --version 2>/dev/null | head -1)"
}
JSON

echo "[$(date +%T)] generated $(find "$OUT" -name '*.parquet' | wc -l) files, $(du -sh "$OUT" | cut -f1)"

[ "$VERIFY" = 1 ] || { echo "verification skipped"; exit 0; }
[ -x "$PYTHON" ] || die "no python with duckdb at $PYTHON (set TPCH_PYTHON or pass --no-verify)"

# Every table but lineitem has an exact per-SF row count. lineitem's varies slightly with the
# generator's data distribution, so it is checked against a tolerance instead.
"$PYTHON" - "$OUT" "$SCALE" "$DECIMAL_TYPE" <<'PY'
import sys, duckdb
out, scale, dtype = sys.argv[1], int(sys.argv[2]), sys.argv[3]
exact = {"orders":1500000,"customer":150000,"part":200000,"partsupp":800000,"supplier":10000}
con, bad = duckdb.connect(), []
for t, per in sorted(exact.items()):
    n = con.sql(f"select count(*) from read_parquet('{out}/{t}/*.parquet')").fetchone()[0]
    e = per*scale
    ok = n == e
    bad += [] if ok else [t]
    print(f"{t:10s} {n:>15,} expected {e:>15,} {'OK' if ok else 'MISMATCH'}")
for t, e in (("nation",25),("region",5)):
    n = con.sql(f"select count(*) from read_parquet('{out}/{t}/*.parquet')").fetchone()[0]
    bad += [] if n == e else [t]
    print(f"{t:10s} {n:>15,} expected {e:>15,} {'OK' if n==e else 'MISMATCH'}")
n = con.sql(f"select count(*) from read_parquet('{out}/lineitem/*.parquet')").fetchone()[0]
e = 6001215*scale
ok = abs(n-e)/e < 0.001
bad += [] if ok else ["lineitem"]
print(f"{'lineitem':10s} {n:>15,} ~expected {e:>15,} ({(n-e)/e*100:+.3f}%) {'OK' if ok else 'MISMATCH'}")

want = "DOUBLE" if dtype == "f64" else "DECIMAL"
got = dict(con.sql(
    f"select column_name, column_type from (describe select * from "
    f"read_parquet('{out}/lineitem/*.parquet')) "
    f"where column_name in ('l_quantity','l_extendedprice','l_discount','l_tax')").fetchall())
wrong = [c for c, t in got.items() if want not in t]
print(f"monetary columns: {sorted(set(got.values()))} expected {want}")
if wrong: bad.append("column-types")
if bad:
    print("FAILED: " + ", ".join(sorted(set(bad)))); sys.exit(1)
print("all row counts and column types OK")
PY
