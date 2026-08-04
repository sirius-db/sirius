#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-FileCopyrightText: Copyright (c) 2026, Sirius Contributors.
# SPDX-License-Identifier: Apache-2.0
# Reports relevant environment information useful for diagnosing and
# debugging Sirius issues.
# Usage:
# "bash print_env.sh" - prints to stdout
# "bash print_env.sh > env.txt" - prints to file "env.txt"

MISSING_RUNTIME=()
MISSING_BUILD=()
MISSING_OPTIONAL=()

check_tools() {
    type "git"        &>/dev/null || MISSING_RUNTIME+=("git")
    type "nvidia-smi" &>/dev/null || MISSING_RUNTIME+=("nvidia-smi")
    type "pixi"       &>/dev/null || MISSING_RUNTIME+=("pixi")
    type "nvcc"       &>/dev/null || MISSING_BUILD+=("nvcc")
    type "cmake"      &>/dev/null || MISSING_BUILD+=("cmake")
    type "g++"        &>/dev/null || MISSING_BUILD+=("g++")
    type "clang"      &>/dev/null || MISSING_BUILD+=("clang")
    type "rustc"      &>/dev/null || MISSING_OPTIONAL+=("rustc")
    type "cargo"      &>/dev/null || MISSING_OPTIONAL+=("cargo")
}

print_env() {
echo "***git***"
if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]; then
    git log --decorate -n 1
    echo "***git submodules***"
    git submodule status --recursive
else
    echo "Not inside a git repository"
fi
echo

echo "***OS information***"
cat /etc/*-release 2>/dev/null
uname -a
echo

echo "***GPU information & topology***"
if type "nvidia-smi" &> /dev/null; then
    nvidia-smi
    nvidia-smi topo -m
else
    echo "nvidia-smi not found"
fi
echo

echo "***CPU***"
if type "lscpu" &> /dev/null; then
    lscpu
else
    sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "CPU info unavailable"
fi
echo

echo "***CMake***"
if type "cmake" &> /dev/null; then
    which cmake && cmake --version
else
    echo "cmake not found"
fi
echo

echo "***g++***"
if type "g++" &> /dev/null; then
    which g++ && g++ --version
else
    echo "g++ not found"
fi
echo

echo "***clang***"
if type "clang" &> /dev/null; then
    which clang && clang --version
else
    echo "clang not found"
fi
echo

echo "***nvcc***"
if type "nvcc" &> /dev/null; then
    which nvcc && nvcc --version
else
    echo "nvcc not found"
fi
echo

echo "***Rust***"
if type "rustc" &> /dev/null; then
    which rustc && rustc --version
else
    echo "rustc not found"
fi
echo
if type "cargo" &> /dev/null; then
    which cargo && cargo --version
else
    echo "cargo not found"
fi
echo

echo "***pixi***"
if type "pixi" &> /dev/null; then
    which pixi && pixi --version
else
    echo "pixi not found"
fi
echo

echo "***DuckDB***"
if [ -f "build/release/duckdb" ]; then
    build/release/duckdb --version
elif [ -f "build/debug/duckdb" ]; then
    build/debug/duckdb --version
else
    echo "DuckDB binary not found in build/"
fi
echo

echo "***Sirius build type***"
if [ -d "build/release" ]; then
    echo "release build present"
elif [ -d "build/debug" ]; then
    echo "debug build present"
else
    echo "No build directory found"
fi
echo

echo "***Environment variables***"
printf '%-32s: %s\n' PATH "$PATH"
printf '%-32s: %s\n' LD_LIBRARY_PATH "$LD_LIBRARY_PATH"
printf '%-32s: %s\n' CUDA_VISIBLE_DEVICES "$CUDA_VISIBLE_DEVICES"
printf '%-32s: %s\n' CUDA_CACHE_PATH "$CUDA_CACHE_PATH"
echo

echo "***pixi info***"
if type "pixi" &> /dev/null; then
    pixi info
else
    echo "pixi not found"
fi
echo

echo "***pixi list***"
if type "pixi" &> /dev/null; then
    pixi list
else
    echo "pixi not found"
fi
echo
}

check_tools

if [ ${#MISSING_RUNTIME[@]} -gt 0 ]; then
    echo "ERROR:   Missing runtime tools required to run Sirius: ${MISSING_RUNTIME[*]}"
    echo "Please install the missing tools and try again."
    exit 1
fi
if [ ${#MISSING_BUILD[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: Missing build tools (not required to run Sirius): ${MISSING_BUILD[*]}"
fi
if [ ${#MISSING_OPTIONAL[@]} -gt 0 ]; then
    echo ""
    echo "NOTE:    Optional tools not found: ${MISSING_OPTIONAL[*]}"
fi

echo "<details><summary>Click here to see environment details</summary><pre>"
echo "     "
print_env | while read -r line; do
    echo "     $line"
done
echo "</pre></details>"
