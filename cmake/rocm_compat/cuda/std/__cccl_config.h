/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! Minimal CCCL config for ROCm. Provides the _CCCL_* macros that Sirius's
//! detail headers use for device/host/inline annotations.
//!
//! The real CCCL __cccl_config.h defines version macros and platform detection.
//! On stock ROCm (no hipDF/CCCL), this minimal shim provides the annotation
//! macros so headers like bit_unpack.cuh, byte_copy.cuh, etc. that use
//! _CCCL_DEVICE / _CCCL_FORCEINLINE compile.
//!
//! When hipDF is installed, its CCCL __cccl_config.h should take precedence.
//! This file only resolves if the shim is found first (BEFORE on include path).

#pragma once

// Try hipDF's real CCCL config first (if installed via cudf::cudf includes).
#if __has_include_next(<cuda/std/__cccl_config.h>)
  #include_next <cuda/std/__cccl_config.h>
#endif

// Fallback: define _CCCL_* macros only if not already defined by real CCCL.
#ifndef _CCCL_DEVICE
#define _CCCL_DEVICE __device__
#endif
#ifndef _CCCL_HOST
#define _CCCL_HOST __host__
#endif
#ifndef _CCCL_GLOBAL
#define _CCCL_GLOBAL __global__
#endif
#ifndef _CCCL_FORCEINLINE
#define _CCCL_FORCEINLINE __forceinline__
#endif
#ifndef _CCCL_EXECUTION_CHECK
#define _CCCL_EXECUTION_CHECK
#endif

// CCCL version stub (not used by Sirius, but prevents undefined-macro issues).
#ifndef _CCCL_VER_MAJOR
#define _CCCL_VER_MAJOR 2
#define _CCCL_VER_MINOR 0
#define _CCCL_VER_PATCH 0
#endif
