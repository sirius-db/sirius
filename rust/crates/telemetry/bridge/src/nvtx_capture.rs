// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use nvtx_bridge::NvtxEventEntity;
use quent_instrumentation::{ContextInner, ObserverInner};
use quent_io::{ExporterOptions, FileSystemExporterOptions, FileSystemFormat};

#[cxx::bridge(namespace = "sirius::telemetry::detail")]
mod ffi {
    #[namespace = "quent::detail::uuid"]
    unsafe extern "C++" {
        include!("telemetry-bridge/gen/uuid.rs.h");
        type UUID = crate::bridge::uuid::ffi::UUID;
    }

    extern "Rust" {
        type NvtxCapture;

        fn owns_injection_hook(self: &NvtxCapture) -> bool;
        fn nvtx_capture_none() -> Box<NvtxCapture>;
        fn nvtx_capture_ndjson(context_id: &UUID, output_dir: String) -> Result<Box<NvtxCapture>>;
        fn nvtx_capture_msgpack(context_id: &UUID, output_dir: String) -> Result<Box<NvtxCapture>>;
        fn nvtx_capture_postcard(context_id: &UUID, output_dir: String)
        -> Result<Box<NvtxCapture>>;
    }
}

/// Owns the non-schema NVTX observer for a Sirius instrumentation context.
pub struct NvtxCapture {
    _pipeline: Option<ObserverInner<NvtxEventEntity>>,
    owns_injection_hook: bool,
}

impl NvtxCapture {
    fn owns_injection_hook(&self) -> bool {
        self.owns_injection_hook
    }
}

fn nvtx_capture_none() -> Box<NvtxCapture> {
    Box::new(NvtxCapture {
        _pipeline: None,
        owns_injection_hook: false,
    })
}

fn nvtx_capture_ndjson(
    context_id: &ffi::UUID,
    output_dir: String,
) -> Result<Box<NvtxCapture>, String> {
    nvtx_capture_filesystem(context_id, output_dir, FileSystemFormat::Ndjson)
}

fn nvtx_capture_msgpack(
    context_id: &ffi::UUID,
    output_dir: String,
) -> Result<Box<NvtxCapture>, String> {
    nvtx_capture_filesystem(context_id, output_dir, FileSystemFormat::Msgpack)
}

fn nvtx_capture_postcard(
    context_id: &ffi::UUID,
    output_dir: String,
) -> Result<Box<NvtxCapture>, String> {
    nvtx_capture_filesystem(context_id, output_dir, FileSystemFormat::Postcard)
}

fn nvtx_capture_filesystem(
    context_id: &ffi::UUID,
    output_dir: String,
    format: FileSystemFormat,
) -> Result<Box<NvtxCapture>, String> {
    let context_id = quent_instrumentation::Uuid::from(*context_id);
    let options =
        ExporterOptions::FileSystem(FileSystemExporterOptions::new(format, output_dir.into()));
    let context = ContextInner::try_new(context_id).map_err(|error| error.to_string())?;
    let pipeline = context
        .block_on(context.observer::<NvtxEventEntity>(&options))
        .map_err(|error| error.to_string())?;
    let sender = pipeline.sender();

    // The injection hook is process-global and one-shot. A later Sirius
    // context cannot replace the first context's capture destination.
    let owns_injection_hook =
        match nvtx_injection::install_hook(move |event| sender.emit(context_id, event)) {
            Ok(()) => true,
            Err(nvtx_injection::InstallHookError::AlreadyInstalled) => false,
        };

    Ok(Box::new(NvtxCapture {
        _pipeline: Some(pipeline),
        owns_injection_hook,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn exports_injected_nvtx_events() {
        let context_id = quent_instrumentation::Uuid::now_v7();
        let bridge_id = context_id.into();
        let output_dir = std::env::temp_dir().join(format!("sirius_nvtx_{context_id}"));

        let capture = nvtx_capture_ndjson(&bridge_id, output_dir.to_string_lossy().into_owned())
            .expect("create NVTX capture");
        assert!(capture.owns_injection_hook());

        nvtx::mark(c"sirius-nvtx-capture-test");
        drop(capture);

        let stream_dir = output_dir.join(context_id.to_string()).join("NvtxEvent");
        let records = fs::read_dir(&stream_dir)
            .expect("read NVTX stream directory")
            .map(|entry| fs::read_to_string(entry.expect("read NVTX stream entry").path()))
            .collect::<Result<String, _>>()
            .expect("read NVTX stream");

        assert!(records.contains("sirius-nvtx-capture-test"));
        assert!(records.contains("Mark"));

        fs::remove_dir_all(output_dir).expect("remove NVTX test output");
    }
}
