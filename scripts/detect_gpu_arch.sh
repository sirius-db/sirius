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

  # 5. No GPU detected — ask the user instead of silently defaulting to gfx942.
  # Falling back to gfx942 on a non-MI300 box produces wrong-fatbinaries that
  # fail at runtime with confusing errors. Prompting forces an explicit choice.
  if [ -t 0 ]; then
    # Interactive shell — prompt the user
    echo "" >&2
    echo "[detect_gpu_arch] No AMD GPU detected via rocminfo, rocm_agent_enumerator," >&2
    echo "[detect_gpu_arch] or /sys/class/kfd. This may be a build-only container." >&2
    echo "" >&2
    echo "  All AMD GPU architectures:" >&2
    echo "    gfx950  — MI350 (Instinct, CDNA4)" >&2
    echo "    gfx942  — MI300X / MI300A (Instinct, CDNA3)" >&2
    echo "    gfx940  — MI300 (Instinct, CDNA3 pre-production)" >&2
    echo "    gfx90a  — MI250X / MI250 (Instinct, CDNA2)" >&2
    echo "    gfx908  — MI100 (Instinct, CDNA1)" >&2
    echo "    gfx906  — Fiji / FirePro S9300 x2 (GCN 3rd)" >&2
    echo "    gfx803  — Polaris 10/11 (RX 470/480/570/580, GCN 4th)" >&2
    echo "    gfx900  — Vega 10 (Radeon Vega FE / RX Vega 64, GCN 5th)" >&2
    echo "    gfx906  — Vega 20 (Radeon VII, GCN 5th)" >&2
    echo "    gfx1010 — RDNA1 (RX 5000 series)" >&2
    echo "    gfx1030 — RDNA2 (RX 6800/6900 series)" >&2
    echo "    gfx1031 — RDNA2 (RX 6600/6700 series)" >&2
    echo "    gfx1032 — RDNA2 (RX 6400/6500 series)" >&2
    echo "    gfx1100 — RDNA3 (RX 7900 XT/XTX)" >&2
    echo "    gfx1101 — RDNA3 (RX 7700/7800 series)" >&2
    echo "    gfx1102 — RDNA3 (RX 7600 series)" >&2
    echo "    gfx1103 — RDNA3 (RX 7400/7900M series)" >&2
    echo "    gfx1150 — RDNA3.5 (Ryzen 800M APU integrated)" >&2
    echo "    gfx1200 — RDNA4 (RX 9000 series)" >&2
    echo "" >&2
    read -p "[detect_gpu_arch] Enter GPU architecture (or press Enter for gfx942): " user_arch >&2
    if [ -n "$user_arch" ]; then
      echo "$user_arch"
    else
      echo "gfx942"
    fi
  else
    # Non-interactive (piped, CI, etc.) — can't prompt, default to gfx942
    echo "[detect_gpu_arch] WARNING: No GPU detected and stdin is not a TTY." >&2
    echo "[detect_gpu_arch] Defaulting to gfx942 (MI300). Override with GPU_ARCH=<arch>" >&2
    echo "gfx942"
  fi
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
  echo "[detect_gpu_arch] GPU architecture: $GPU_ARCH"
fi
