# Benchmark runner image for Sirius TPC-H benchmarks.
#
# Provides the pixi environment manager, which installs the full CUDA toolchain
# and all Sirius build dependencies. The CUDA runtime is injected at container
# startup by nvidia-container-toolkit on the host — no CUDA base layer needed.
#
# Built natively for linux/amd64 (T4 runners, developer workstations) and
# linux/arm64 (DGX Spark) and published as a single multi-arch manifest at:
#   ghcr.io/sirius-db/sirius/benchmark-runner:latest
#
# BAKE_SCCACHE=true triggers a full dev build during image construction to
# pre-warm the sccache. Pass this in the nightly workflow only — a developer
# pulling the image gets yesterday's dev cache and only compiles the delta.
# Default is false so toolchain-only rebuilds are fast.
#
# Local usage with act:
#   act --container-options "--gpus all" \
#       -P self-hosted=ghcr.io/sirius-db/sirius/benchmark-runner:latest
#
# Or with a local sccache mount (skips cache bake, uses your own cache):
#   act --container-options "--gpus all -v $HOME/.cache/sccache:/root/.cache/sccache"

FROM nvidia/cuda:13.0.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pixi — manages the full CUDA toolchain and all Sirius dependencies.
# Version is pinned to match check.yml (pixi-version: v0.65.0).
ARG PIXI_VERSION=0.65.0
RUN curl -fsSL https://pixi.prefix.dev/install.sh \
    | PIXI_VERSION=${PIXI_VERSION} bash -s -- --no-modify-path
ENV PATH="/root/.pixi/bin:$PATH"

# sccache cache directory. Populated by the nightly bake (BAKE_SCCACHE=true)
# or by mounting a local cache directory at runtime.
ENV SCCACHE_DIR=/root/.cache/sccache

# --- Nightly sccache bake ---
# Clone dev, run a full build to populate sccache, then discard the source and
# build artifacts. Only the sccache directory is kept in the image layer.
# This reduces incremental build time for PRs and on-demand benchmarks.
ARG BAKE_SCCACHE=false
ARG SIRIUS_REF=dev
RUN if [ "$BAKE_SCCACHE" = "true" ]; then \
    echo "Baking sccache from sirius@${SIRIUS_REF}..." && \
    git clone --recurse-submodules --depth=1 --branch ${SIRIUS_REF} \
        https://github.com/sirius-db/sirius.git /tmp/sirius-build && \
    cd /tmp/sirius-build && \
    pixi install && \
    pixi run -- bash -c 'cd duckdb && cmake --preset release' && \
    pixi run -- cmake --build build/release \
        --target sirius_loadable_extension --target duckdb \
        -j$(nproc) && \
    echo "sccache stats:" && pixi run -- sccache --show-stats && \
    rm -rf /tmp/sirius-build; \
fi
