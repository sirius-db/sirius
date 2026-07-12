/*
 * Copyright 2026, rocm-cuda-compat contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */

//! @file
//! NVML → ROCm shim for cuCascade topology_discovery.
//!
//! cuCascade's topology_discovery.cpp uses NVML (NVIDIA Management Library)
//! for GPU enumeration, name/UUID queries, PCI bus info, and peer topology.
//! ROCm's equivalents are:
//!   - hipGetDeviceCount / hipGetDeviceProperties (device enumeration)
//!   - hipDeviceGetPCIBusId (PCI bus)
//!   - hipDeviceCanAccessPeer (peer access)
//!   - rocm_smi (rocm-smi library) for UUID, power, temperature
//!
//! This shim provides the NVML API surface cuCascade uses, backed by HIP calls.
//! It's a thin compatibility layer — not a full NVML implementation.

#pragma once

#include <cuda_runtime.h>  // shim → hip runtime
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

// NVML return codes
typedef int nvmlReturn_t;
#define NVML_SUCCESS 0
#define NVML_ERROR_UNINITIALIZED 1
#define NVML_ERROR_NO_PERMISSION 2
#define NVML_ERROR_NOT_FOUND 3
#define NVML_ERROR_NOT_SUPPORTED 4
#define NVML_ERROR_GPU_NOT_FOUND 5
#define NVML_ERROR_INVALID_ARGUMENT 6
#define NVML_ERROR_UNKNOWN 999

// NVML device handle — wraps a device index
typedef struct nvmlDevice_st* nvmlDevice_t;

// NVML constants
#define NVML_DEVICE_UUID_BUFFER_SIZE 80
#define NVML_DEVICE_NAME_BUFFER_SIZE 64
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32

// NVML clock types
typedef enum {
  NVML_CLOCK_GRAPHICS = 0,
  NVML_CLOCK_SM = 1,
  NVML_CLOCK_MEM = 2,
} nvmlClockType_t;

// NVML compute mode
typedef enum {
  NVML_COMPUTEMODE_DEFAULT = 0,
  NVML_COMPUTEMODE_EXCLUSIVE_THREAD = 1,
  NVML_COMPUTEMODE_PROHIBITED = 2,
  NVML_COMPUTEMODE_EXCLUSIVE_PROCESS = 3,
} nvmlComputeMode_t;

// NVML PCI info
typedef struct {
  char busIdLegacy[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  unsigned int domain;
  unsigned int bus;
  unsigned int device;
  unsigned int pciDeviceId;
  unsigned int pciSubSystemId;
} nvmlPciInfo_t;

// --- NVML API functions (backed by HIP) ---

inline nvmlReturn_t nvmlInit_v2() { return NVML_SUCCESS; }
inline nvmlReturn_t nvmlShutdown() { return NVML_SUCCESS; }
inline nvmlReturn_t nvmlInit() { return NVML_SUCCESS; }

inline nvmlReturn_t nvmlDeviceGetCount(unsigned int* deviceCount) {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess) return NVML_ERROR_UNKNOWN;
  *deviceCount = static_cast<unsigned int>(count);
  return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
  // Store the index as the handle (cast through uintptr_t for safety)
  *device = reinterpret_cast<nvmlDevice_t>(static_cast<uintptr_t>(index + 1));
  return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
  unsigned int index = static_cast<unsigned int>(reinterpret_cast<uintptr_t>(device)) - 1;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, index) != hipSuccess) return NVML_ERROR_UNKNOWN;
  strncpy(name, prop.name, length - 1);
  name[length - 1] = '\0';
  return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) {
  unsigned int index = static_cast<unsigned int>(reinterpret_cast<uintptr_t>(device)) - 1;
  // ROCm doesn't have a direct UUID API; use device index as a synthetic UUID
  snprintf(uuid, length, "GPU-%08x", index);
  return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, /* nvmlMemory_t* */ void* memory) {
  // Simplified: just return success. Full implementation would use hipDeviceGetAttribute.
  return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  unsigned int index = static_cast<unsigned int>(reinterpret_cast<uintptr_t>(device)) - 1;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE] = {0};
  if (hipDeviceGetPCIBusId(busId, sizeof(busId), index) != hipSuccess) return NVML_ERROR_UNKNOWN;
  memset(pci, 0, sizeof(*pci));
  strncpy(pci->busIdLegacy, busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE - 1);
  return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) {
  *clock = 0;
  return NVML_SUCCESS;
}

inline nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode) {
  *mode = NVML_COMPUTEMODE_DEFAULT;
  return NVML_SUCCESS;
}

// Helper to get the device index from a handle
inline unsigned int nvml_device_index(nvmlDevice_t device) {
  return static_cast<unsigned int>(reinterpret_cast<uintptr_t>(device)) - 1;
}
