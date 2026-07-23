#!/usr/bin/env bash
# =============================================================================
# detect_gpu_arch.sh — Auto-detect the AMD GPU architecture for ROCm builds.
#
# This script is sourced by build_rocm_deps.sh and run_tpch_rocm.sh so they
# work on ANY AMD GPU (MI300 gfx942, MI250 gfx90a, MI100 gfx908, RDNA3
# gfx1100, etc.) instead of hardcoding gfx942.
#
# Detection order:
#   1. Honor an explicit --gpu-arch flag or pre-set GPU_ARCH env var.
#   2. Parse rocminfo / rocm_agent_enumerator for the first gfx* token.
#   3. Fall back to gfx942 (MI300, the primary target) when no GPU is found
#      (CI containers, build-only boxes without a GPU).
#
# After sourcing, the following variables are set:
#   GPU_ARCH         — e.g. "gfx1100"
#   ROCM_AMDGPU_TARGETS — same value (for hipDF/hipMM CMake)
#   GPU_TARGETS      — same value (for hipCUB/rocThrust CMake)
#
# Usage:
#   source scripts/detect_gpu_arch.sh
#   # or:
#   GPU_ARCH=$(bash scripts/detect_gpu_arch.sh)
# =============================================================================

# Don't re-run if already detected (sourced multiple times)
if [ -n "${GPU_ARCH:-}" ] && [ -z "${DETECT_GPU_ARCH_FORCE:-}" ]; then
  return 0 2>/dev/null || true
fi

detect_gpu_arch() {
  local explicit="${GPU_ARCH:-}"

  # 1. Honor pre-set env var (highest priority)
  if [ -n "$explicit" ]; then
    echo "$explicit"
    return 0
  fi

  # 2. Try rocm_agent_enumerator (fast, ships with ROCm)
  if command -v rocm_agent_enumerator &>/dev/null; then
    local arch
    arch=$(rocm_agent_enumerator 2>/dev/null | grep -oE 'gfx[0-9]+[a-z]?[0-9]*' | head -1)
    if [ -n "$arch" ]; then
      echo "$arch"
      return 0
    fi
  fi

  # 3. Try rocminfo (slower but always available with ROCm)
  if command -v rocminfo &>/dev/null; then
    local arch
    # Look for "Name: gfxXXXX" in the agent section (skip CPU agents)
    arch=$(rocminfo 2>/dev/null | grep -E "^\s*Name:\s+gfx" | head -1 | grep -oE 'gfx[0-9]+[a-z]?[0-9]*')
    if [ -n "$arch" ]; then
      echo "$arch"
      return 0
    fi
  fi

  # 4. Try /sys/class/kfd/kfd/topology/nodes/*/name (no rocminfo needed)
  if ls /sys/class/kfd/kfd/topology/nodes/*/name &>/dev/null 2>&1; then
    local arch
    arch=$(cat /sys/class/kfd/kfd/topology/nodes/*/name 2>/dev/null | grep -oE 'gfx[0-9]+[a-z]?[0-9]*' | head -1)
    if [ -n "$arch" ]; then
      echo "$arch"
      return 0
    fi
  fi

  # 5. Fallback: gfx942 (MI300, the primary target — safe default for
  # build-only boxes without a GPU, since the binaries won't run anyway)
  echo "gfx942"
}

# If sourced, set the variables. If executed directly, just print the arch.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  # Executed directly — print and exit
  detect_gpu_arch
else
  # Sourced — set variables
  GPU_ARCH=$(detect_gpu_arch)
  export ROCM_AMDGPU_TARGETS="$GPU_ARCH"
  export GPU_TARGETS="$GPU_ARCH"
  echo "[detect_gpu_arch] Detected GPU architecture: $GPU_ARCH"
fi
