// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    // Records this checkout's git revision in `QUENT_SOURCE_COMMIT`, which becomes the
    // `analyzer_revision` of generated bundles.
    quent_build_info::emit_source();
}
