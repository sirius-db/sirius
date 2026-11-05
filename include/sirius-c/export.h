// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
/** @file
 * @brief Visibility and language linkage helpers for the public C API. */
#pragma once
#ifndef SIRIUS_EXPORT
#define SIRIUS_EXPORT __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
#define SIRIUS_C_NOEXCEPT noexcept
#else
#define SIRIUS_C_NOEXCEPT
#endif
