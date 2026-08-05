#!/usr/bin/env bash
# run_calibration.sh - eater-off baselines, per-domain calibration sweeps, and
# the 3x3 cross-talk matrix for the membw throttle tools. Emits tab-separated
# rows:  <label> \t RESULT domain=... gbps=...
#
# Each measurement is ~VSECS+SETTLE seconds; the full script keeps GPU-busy
# time to roughly a minute. Re-checks GPU idleness before every GPU section.
#
# Env overrides: VSECS (victim secs, default 2.5), SETTLE (eater settle,
# default 0.7), SECTIONS (space list: baseline hbm dram c2c xtalk, default all).
set -u
cd "$(dirname "$0")"

VSECS=${VSECS:-2.5}
SETTLE=${SETTLE:-0.7}
SECTIONS=${SECTIONS:-"baseline hbm dram c2c xtalk"}

# Fixed "aggressive" eater rates for the cross-talk matrix (~80-90% of the
# flat-out max measured on this box; see docs/membw-throttle.md).
XT_HBM=${XT_HBM:-4000}
XT_DRAM=${XT_DRAM:-200}
XT_C2C=${XT_C2C:-300}

check_idle() {
	local n
	n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
	while [ "${n:-0}" -gt 0 ]; do
		echo "# GPU busy with $n other process(es); waiting 30s (ctrl-c to abort)" >&2
		sleep 30
		n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
	done
}

# pair <label> <eater-args|-> <victim-args>
# Reports the victim RESULT line plus eater_gbps= (the eater's achieved rate
# averaged over CSV rows after the settle window, i.e. while contended).
pair() {
	local label="$1" eater="$2" victim="$3" epid="" ecsv=""
	if [ "$eater" != "-" ]; then
		ecsv=$(mktemp)
		# shellcheck disable=SC2086
		./membw_eater $eater --csv "$ecsv" 2>/dev/null &
		epid=$!
		sleep "$SETTLE"
	fi
	local res
	# shellcheck disable=SC2086
	res=$(./membw_victim $victim --secs "$VSECS" --quiet 2>/dev/null | grep ^RESULT)
	local egbps="-"
	if [ -n "$epid" ]; then
		kill -INT "$epid" 2>/dev/null
		wait "$epid" 2>/dev/null
		egbps=$(awk -F, -v s="$SETTLE" 'NR>1 && $2>=s+0.2 {a+=$6; n++} END {if (n) printf "%.1f", a/n; else print "-"}' "$ecsv")
		rm -f "$ecsv"
	fi
	printf '%s\t%s eater_gbps=%s\n' "$label" "$res" "$egbps"
}

has() { case " $SECTIONS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

DRAM_EATER="--domain dram --threads 16 --cpu-start 36"
DRAM_VICTIM="--domain dram --threads 8 --cpu-start 0"

if has baseline; then
	check_idle
	echo "# --- baselines (eater off) ---"
	pair "base/hbm-sm"  - "--domain hbm --engine sm"
	pair "base/hbm-ce"  - "--domain hbm --engine ce"
	pair "base/dram"    - "$DRAM_VICTIM"
	pair "base/c2c-h2d" - "--domain c2c --engine h2d"
	pair "base/c2c-d2h" - "--domain c2c --engine d2h"
fi

if has hbm; then
	check_idle
	echo "# --- hbm calibration: victim=hbm-sm vs hbm eater rate ---"
	for r in 250 500 1000 1500 2000 4000 max; do
		pair "cal/hbm/sm@$r" "--domain hbm --engine sm --gbps $r" "--domain hbm --engine sm"
	done
	echo "# --- hbm eater engine comparison (ce vs sm) ---"
	for r in 4000 max; do
		pair "cal/hbm/ce@$r" "--domain hbm --engine ce --gbps $r" "--domain hbm --engine sm"
	done
fi

if has dram; then
	echo "# --- dram calibration: victim=dram(8thr) vs dram eater rate ---"
	for r in 60 120 180 max; do
		pair "cal/dram@$r" "$DRAM_EATER --gbps $r" "$DRAM_VICTIM"
	done
fi

if has c2c; then
	check_idle
	echo "# --- c2c calibration: victim=c2c-h2d vs c2c-h2d eater rate ---"
	for r in 100 200 300 max; do
		pair "cal/c2c/h2d@$r" "--domain c2c --engine h2d --gbps $r" "--domain c2c --engine h2d"
	done
fi

if has xtalk; then
	check_idle
	echo "# --- cross-talk matrix: each eater at fixed rate vs all victims ---"
	for v in "hbm --engine sm" "dram-VICTIM" "c2c --engine h2d"; do
		if [ "$v" = "dram-VICTIM" ]; then vic="$DRAM_VICTIM"; vname="dram"; else vic="--domain $v"; vname=${v%% *}; fi
		pair "xt/hbm@$XT_HBM->$vname"   "--domain hbm --engine sm --gbps $XT_HBM" "$vic"
		pair "xt/dram@$XT_DRAM->$vname" "$DRAM_EATER --gbps $XT_DRAM"             "$vic"
		pair "xt/c2c@$XT_C2C->$vname"   "--domain c2c --engine h2d --gbps $XT_C2C" "$vic"
	done
fi
