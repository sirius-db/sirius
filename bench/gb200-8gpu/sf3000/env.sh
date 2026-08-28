# 8-CN SF3000 knobs. Not yet swept on this pair.
#
# 4-CN SF3000 preset is 120/36/16 (engine-a.env). 8 CNs need less arena per node than 4,
# but SF1000 8-CN still needed 32 GiB (16 GiB exhausted). Do not drop below 36 GiB
# until a high-water line says so. Dataset 1.2 T > one box of LPDDR → HOST_MEM=16GiB.
# Occupancy 120+36+0.76 = 156.8 / 184 = 85 %.
# pipeline_dop=18 is topology, not scale.

export SCALE_FACTOR=3000
export NUM_CNS_PER_HOST=${NUM_CNS_PER_HOST:-4}
export NUM_CNS=${NUM_CNS:-8}

export GPU_MEM=${GPU_MEM:-120GiB}
export HOST_MEM=${HOST_MEM:-16GiB}
export STAGING=${STAGING:-36GiB}
export SIRIUS_EXCHANGE_STAGING_BYTES=${SIRIUS_EXCHANGE_STAGING_BYTES:-$STAGING}

export SIRIUS_EXCHANGE_STAGING_ARENA=${SIRIUS_EXCHANGE_STAGING_ARENA:-fabric}
export UCX_TLS=${UCX_TLS:-cuda_copy,cuda_ipc,tcp,self}
export SIRIUS_CN_USE_SIRIUS_DATASOURCE=${SIRIUS_CN_USE_SIRIUS_DATASOURCE:-true}

export SIRIUS_QUERY_WATCHDOG_SECS=${SIRIUS_QUERY_WATCHDOG_SECS:-600}
export SIRIUS_CN_RPC_TIMEOUT_SECS=${SIRIUS_CN_RPC_TIMEOUT_SECS:-900}
export SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS=${SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS:-600}

export PIPELINE_DOP=${PIPELINE_DOP:-18}
export FE_QUERY_TIMEOUT=${FE_QUERY_TIMEOUT:-18000}
export QUERY_TIMEOUT=${QUERY_TIMEOUT:-5400}
export COLD_TIMEOUT=${COLD_TIMEOUT:-18000}
export MIN_BACKENDS=${MIN_BACKENDS:-8}

export TPCH_DATA=${TPCH_DATA:-/scratch/sirius/datasets/tpch_sf3000}
export Q11_FRACTION=${Q11_FRACTION:-0.000000033333}
