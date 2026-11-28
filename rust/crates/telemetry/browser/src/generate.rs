// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native construction of Sirius browser bundles from collector output.

use std::path::Path;

use quent_browser_bundle::Capabilities;
use quent_browser_bundle_build::{BuildOptions, GenerationSummary};
use sirius_telemetry_store::{SCHEMA_DIGEST, Sirius};

use crate::{MODEL_NAME, engine_id};

/// Recorded in bundle metadata so a bundle names the analyzer revision that wrote it.
pub const ANALYZER_REVISION: &str = match option_env!("QUENT_SOURCE_COMMIT") {
    Some(commit) => commit,
    None => env!("CARGO_PKG_VERSION"),
};

/// Convert a completed collector directory or Quent ZIP to a browser bundle.
///
/// The input must contain exactly one Sirius engine. Every context directory under
/// the input root is included.
pub fn generate_bundle(
    input: &Path,
    output: &Path,
) -> Result<GenerationSummary, quent_browser_bundle_build::Error> {
    quent_browser_bundle_build::generate_bundle::<Sirius>(
        input,
        output,
        &BuildOptions {
            model_name: MODEL_NAME,
            schema_digest: SCHEMA_DIGEST,
            analyzer_revision: ANALYZER_REVISION,
            capabilities: Capabilities {
                data_flow: true,
                // NVTX ranges are served by the native server, not stored in the bundle.
                nvtx: false,
            },
        },
        engine_id,
    )
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use quent_events::Event;
    use quent_io::{
        ExporterOptions,
        filesystem::{self, Format},
    };
    use quent_query_engine_analyzer::ui::UiAnalyzer;
    use sirius_telemetry_analyzer::SiriusUiAnalyzer;
    use sirius_telemetry_instrumentation as instrumentation;
    use sirius_telemetry_store::SiriusEvent;
    use uuid::Uuid;

    use super::*;

    /// Write one context holding an engine (optionally), a query group, a finished query,
    /// and its plan and worker.
    fn write_context(root: &Path, with_engine: bool) -> (Uuid, Uuid, Uuid) {
        let context = instrumentation::Context::<instrumentation::Sirius>::try_new(
            ExporterOptions::FileSystem(filesystem::exporter::Options::new(
                Format::Ndjson,
                root.to_path_buf(),
            )),
        )
        .unwrap();
        let mut engine = context.observer::<instrumentation::Engine>().handle();
        if with_engine {
            engine.init(Some("Sirius".into())).unwrap();
        }
        let mut group = context.observer::<instrumentation::QueryGroup>().handle();
        group
            .declaration("group".into(), engine.as_entity_ref())
            .unwrap();
        let query = context
            .observer::<instrumentation::Query>()
            .handle()
            .init("select".into(), group.as_entity_ref())
            .planning()
            .executing()
            .finished();
        let mut worker = context.observer::<instrumentation::Worker>().handle();
        worker
            .init(engine.as_entity_ref(), "42".into(), "worker".into())
            .unwrap();
        let mut plan = context.observer::<instrumentation::Plan>().handle();
        plan.declaration(
            query.as_entity_ref(),
            "plan".into(),
            vec![],
            Some(worker.as_entity_ref()),
        )
        .unwrap();
        (context.id(), engine.id(), query.id())
    }

    #[test]
    fn converts_collector_output_and_the_analyzer_loads_it() {
        let temp = tempfile::tempdir().unwrap();
        let (context_id, engine_id, query_id) = write_context(temp.path(), true);
        let output = temp.path().join("nested").join("trace.quent-browser");

        let summary = generate_bundle(temp.path(), &output).unwrap();
        let bytes = std::fs::read(&output).unwrap();
        let decoded = quent_browser_bundle::decode(&bytes).unwrap();
        let events: Vec<Event<SiriusEvent>> = postcard::from_bytes(decoded.payload).unwrap();

        assert_eq!(summary.engine_id, engine_id);
        assert_eq!(summary.context_count, 1);
        assert_eq!(summary.event_count, events.len());
        assert_eq!(summary.schema_digest, SCHEMA_DIGEST);
        assert!(summary.event_count >= 3, "{}", summary.event_count);
        assert_eq!(decoded.metadata.model_name, MODEL_NAME);
        assert_eq!(decoded.metadata.schema_digest, SCHEMA_DIGEST);
        assert_eq!(decoded.metadata.codec, "postcard");
        assert_eq!(decoded.metadata.engine_id, engine_id);
        assert_eq!(decoded.metadata.context_ids, vec![context_id]);
        assert_eq!(decoded.metadata.analyzer_revision, ANALYZER_REVISION);
        assert_eq!(
            decoded.metadata.capabilities,
            Capabilities {
                data_flow: true,
                nvtx: false
            }
        );
        assert_eq!(
            decoded.metadata.model_provenance.unwrap()["model"]["name"],
            "sirius"
        );

        crate::load_browser_service(&bytes).expect("analyzer accepts the generated bundle");
        let analyzer = SiriusUiAnalyzer::try_new(engine_id, events.into_iter()).unwrap();
        assert_eq!(analyzer.query_bundle(query_id).unwrap().query_id, query_id);
    }

    #[test]
    fn converts_a_zip_archive() {
        let temp = tempfile::tempdir().unwrap();
        let collected = temp.path().join("collected");
        let (context_id, engine_id, _) = write_context(&collected, true);

        let archive = temp.path().join("trace.zip");
        let mut zip = zip::ZipWriter::new(std::fs::File::create(&archive).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        for entry in walk(&collected) {
            let name = entry.strip_prefix(&collected).unwrap().to_str().unwrap();
            zip.start_file(name, options).unwrap();
            zip.write_all(&std::fs::read(&entry).unwrap()).unwrap();
        }
        zip.finish().unwrap();
        let output = temp.path().join("trace.quent-browser");

        let summary = generate_bundle(&archive, &output).unwrap();
        assert_eq!(summary.engine_id, engine_id);
        assert_eq!(summary.context_count, 1);
        let bytes = std::fs::read(&output).unwrap();
        let decoded = quent_browser_bundle::decode(&bytes).unwrap();
        assert_eq!(decoded.metadata.context_ids, vec![context_id]);
        crate::load_browser_service(&bytes).expect("analyzer accepts the generated bundle");
    }

    #[test]
    fn rejects_input_without_an_engine() {
        let temp = tempfile::tempdir().unwrap();
        write_context(temp.path(), false);

        let error =
            generate_bundle(temp.path(), &temp.path().join("trace.quent-browser")).unwrap_err();
        assert!(error.to_string().contains("exactly one engine"), "{error}");
    }

    #[test]
    fn rejects_input_with_more_than_one_engine() {
        let temp = tempfile::tempdir().unwrap();
        write_context(temp.path(), true);
        write_context(temp.path(), true);

        let error =
            generate_bundle(temp.path(), &temp.path().join("trace.quent-browser")).unwrap_err();
        assert!(error.to_string().contains("exactly one engine"), "{error}");
    }

    #[test]
    fn rejects_input_without_contexts() {
        let temp = tempfile::tempdir().unwrap();

        let error =
            generate_bundle(temp.path(), &temp.path().join("trace.quent-browser")).unwrap_err();
        assert!(error.to_string().contains("context"), "{error}");
    }

    /// All regular files under `dir`, in a stable order.
    fn walk(dir: &Path) -> Vec<std::path::PathBuf> {
        let mut paths = std::fs::read_dir(dir)
            .unwrap()
            .flat_map(|entry| {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(&path)
                } else {
                    vec![path]
                }
            })
            .collect::<Vec<_>>();
        paths.sort();
        paths
    }
}
