// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema-generated Sirius stored-event types.

include!(concat!(env!("OUT_DIR"), "/sirius.rs"));

/// SHA-256 digest of `model.yaml`, in the `sha256:<hex>` form that keys browser bundles
/// to the analyzer compiled from the same schema.
pub const SCHEMA_DIGEST: &str = env!("SIRIUS_SCHEMA_DIGEST");

mod boilerplate;
