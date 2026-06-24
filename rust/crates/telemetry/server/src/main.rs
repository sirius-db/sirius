use std::{
    net::ToSocketAddrs,
    path::{Path, PathBuf},
};

use clap::Parser;
use instrumentation_model::SiriusEvent;
use quent_collector::server::CollectorServiceOptions;
use quent_exporter::{
    ExporterOptions, FileSystemExporterOptions, FileSystemFormat, FileSystemImporterOptions,
    ImporterOptions, create_importer,
};
use quent_query_engine_server::{
    analyzer_service_router, collector_service,
    error::{ServerError, ServerResult},
    initialize_tracing,
};
use sirius_telemetry_analyzer::SiriusUiAnalyzer;
use tokio::net::TcpListener;
use uuid::Uuid;

mod defaults {
    pub(crate) const QUENT_COLLECTOR_ADDRESS: &str = "[::]:7836";
    pub(crate) const QUENT_ANALYZER_ADDRESS: &str = "[::]:8080";
}

mod env {
    pub(crate) const QUENT_COLLECTOR_ADDRESS: &str = "QUENT_COLLECTOR_ADDRESS";
    pub(crate) const QUENT_COLLECTOR_OUTPUT_DIR: &str = "QUENT_COLLECTOR_OUTPUT_DIR";
    pub(crate) const QUENT_COLLECTOR_EXPORTER: &str = "QUENT_COLLECTOR_EXPORTER";
    pub(crate) const QUENT_ANALYZER_ADDRESS: &str = "QUENT_ANALYZER_ADDRESS";
    pub(crate) const QUENT_ANALYZER_CORS_ADDRESS: &str = "QUENT_ANALYZER_CORS_ADDRESS";
}

// Probed (highest-fidelity first) when auto-detecting the format of an existing
// context directory, independent of the configured export format.
const TELEMETRY_FORMATS: [(&str, FileSystemFormat); 3] = [
    ("postcard", FileSystemFormat::Postcard),
    ("msgpack", FileSystemFormat::Msgpack),
    ("ndjson", FileSystemFormat::Ndjson),
];

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "info")]
    log_level: String,

    #[arg(long, default_value = defaults::QUENT_COLLECTOR_ADDRESS, env = env::QUENT_COLLECTOR_ADDRESS)]
    collector_address: String,

    #[arg(long, default_value = "ndjson", env = env::QUENT_COLLECTOR_EXPORTER)]
    exporter: String,

    #[arg(long, default_value = "data", env = env::QUENT_COLLECTOR_OUTPUT_DIR)]
    output_dir: PathBuf,

    #[arg(long, default_value = defaults::QUENT_ANALYZER_ADDRESS, env = env::QUENT_ANALYZER_ADDRESS)]
    analyzer_address: String,

    #[arg(long, env = env::QUENT_ANALYZER_CORS_ADDRESS)]
    cors_address: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args {
        log_level,
        cors_address,
        collector_address,
        exporter,
        output_dir,
        analyzer_address,
    } = Args::parse();

    initialize_tracing(&log_level);

    let collector_addr = collector_address
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| format!("unable to resolve socket address: {collector_address}"))?;
    let analyzer_addr = analyzer_address
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| format!("unable to resolve socket address: {analyzer_address}"))?;

    let importer_output_dir = output_dir.clone();
    let lister_output_dir = output_dir.clone();

    let format = match exporter.as_str() {
        "ndjson" => FileSystemFormat::Ndjson,
        "msgpack" => FileSystemFormat::Msgpack,
        "postcard" => FileSystemFormat::Postcard,
        other => return Err(format!("unknown exporter: {other}").into()),
    };
    // Each context exports to `output_dir/<context-id>/events.<ext>` (plus a
    // `model.qmi` provenance sidecar), so the collector writes under output_dir.
    let exporter = ExporterOptions::FileSystem(FileSystemExporterOptions {
        format,
        root: output_dir,
    });

    let collector = async {
        collector_service::<SiriusEvent>(CollectorServiceOptions { exporter })?
            .serve(collector_addr)
            .await
            .map_err(|error| -> Box<dyn std::error::Error> { Box::new(error) })
    };

    let lister = move || list_engine_ids(&lister_output_dir);

    let importer = move |engine_id| {
        let importer = importer_options_for_engine(&importer_output_dir, engine_id)?;
        Ok(Box::new(create_importer::<SiriusEvent>(&importer)?) as Box<dyn Iterator<Item = _>>)
    };

    let analyzer = async {
        axum::serve(
            TcpListener::bind(analyzer_addr).await?,
            analyzer_service_router::<SiriusUiAnalyzer>(
                Box::new(importer),
                Box::new(lister),
                cors_address,
            )?
            .into_make_service(),
        )
        .await?;
        Ok::<(), Box<dyn std::error::Error>>(())
    };

    tracing::info!("listening on {collector_addr} and {analyzer_addr}");
    tokio::try_join!(collector, analyzer)?;
    Ok(())
}

/// Detect which event-file format a context directory holds, by probing for
/// `events.<ext>`. `None` if the directory has no recognized event file.
fn detect_format(dir: &Path) -> Option<FileSystemFormat> {
    TELEMETRY_FORMATS
        .iter()
        .find(|(extension, _)| dir.join(format!("events.{extension}")).is_file())
        .map(|(_, format)| *format)
}

/// Map a file extension to its telemetry format, if recognized.
fn format_for_ext(extension: &str) -> Option<FileSystemFormat> {
    TELEMETRY_FORMATS
        .iter()
        .find(|(known, _)| *known == extension)
        .map(|(_, format)| *format)
}

/// Legacy flat telemetry files `<output_dir>/<id>.<ext>`, written before the
/// per-context-directory layout. Returns `(file_path, format)` sorted by path.
fn flat_files(output_dir: &Path) -> ServerResult<Vec<(PathBuf, FileSystemFormat)>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(output_dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        if let Some(format) = path
            .extension()
            .and_then(|extension| extension.to_str())
            .and_then(format_for_ext)
        {
            files.push((path, format));
        }
    }
    files.sort_by(|(a, _), (b, _)| a.cmp(b));
    Ok(files)
}

/// The uuid encoded in a flat telemetry file's stem (`<id>.<ext>`), if any.
fn file_stem_uuid(path: &Path) -> Option<Uuid> {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| Uuid::parse_str(stem).ok())
}

/// Per-context export directories under `output_dir`: those named by a uuid that
/// contain a recognized event file. Returns `(dir_id, dir_path, format)` sorted
/// by id for stable listing.
fn context_dirs(output_dir: &Path) -> ServerResult<Vec<(Uuid, PathBuf, FileSystemFormat)>> {
    let mut dirs = Vec::new();
    for entry in std::fs::read_dir(output_dir)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let Some(dir_id) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| Uuid::parse_str(name).ok())
        else {
            continue;
        };
        if let Some(format) = detect_format(&path) {
            dirs.push((dir_id, path, format));
        }
    }
    dirs.sort_by_key(|(dir_id, _, _)| *dir_id);
    Ok(dirs)
}

/// The id of the engine instance recorded in a telemetry capture's events (the
/// id of its `Engine` event), or `None` if no such event is present. `path` is a
/// context directory or a flat event file — the importer accepts either.
fn extract_engine_id(path: &Path, format: FileSystemFormat) -> ServerResult<Option<Uuid>> {
    let importer = ImporterOptions::FileSystem(FileSystemImporterOptions {
        format,
        path: path.to_path_buf(),
    });
    for event in create_importer::<SiriusEvent>(&importer)? {
        if let SiriusEvent::Engine(_) = event.data {
            return Ok(Some(event.id));
        }
    }
    Ok(None)
}

/// Engine instance ids discoverable under `output_dir`.
///
/// A context directory is keyed on its real `Engine` event id, never the
/// directory name (an unrelated context id since Quent dropped the engine id at
/// context creation). Legacy flat files were named by the engine id, so their
/// stem is a valid fallback. Captures only appear once flushed (exporters buffer
/// until `force_flush` on drop), so a still-running context is not yet listed.
fn list_engine_ids(output_dir: &Path) -> ServerResult<Vec<Uuid>> {
    let mut ids = std::collections::HashSet::new();
    for (_, path, format) in context_dirs(output_dir)? {
        if let Some(id) = extract_engine_id(&path, format)? {
            ids.insert(id);
        }
    }
    for (path, format) in flat_files(output_dir)? {
        if let Some(id) = extract_engine_id(&path, format)?.or_else(|| file_stem_uuid(&path)) {
            ids.insert(id);
        }
    }
    Ok(ids.into_iter().collect())
}

fn importer_options_for_engine(
    output_dir: &Path,
    engine_id: Uuid,
) -> ServerResult<ImporterOptions> {
    for (_, path, format) in context_dirs(output_dir)? {
        if extract_engine_id(&path, format)? == Some(engine_id) {
            return Ok(ImporterOptions::FileSystem(FileSystemImporterOptions {
                format,
                path,
            }));
        }
    }

    // Legacy flat files: match on engine id, or on the `<id>.<ext>` stem.
    for (path, format) in flat_files(output_dir)? {
        if extract_engine_id(&path, format)? == Some(engine_id)
            || file_stem_uuid(&path) == Some(engine_id)
        {
            return Ok(ImporterOptions::FileSystem(FileSystemImporterOptions {
                format,
                path,
            }));
        }
    }

    // Fall back to a directory named directly by the engine id.
    let path = output_dir.join(engine_id.to_string());
    if let Some(format) = detect_format(&path) {
        return Ok(ImporterOptions::FileSystem(FileSystemImporterOptions {
            format,
            path,
        }));
    }

    Err(ServerError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("no telemetry directory found for engine {engine_id}"),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An events file with no readable `Engine` event must not surface the
    /// context directory id as an engine id.
    #[test]
    fn empty_context_dir_is_not_listed_as_engine() {
        let root = std::env::temp_dir().join(format!("sirius_telemetry_list_{}", Uuid::now_v7()));
        let context_id = Uuid::now_v7();
        let context_dir = root.join(context_id.to_string());
        std::fs::create_dir_all(&context_dir).unwrap();
        std::fs::write(context_dir.join("events.ndjson"), b"").unwrap();

        let ids = list_engine_ids(&root).unwrap();

        std::fs::remove_dir_all(&root).unwrap();
        assert!(ids.is_empty(), "context id leaked as engine id: {ids:?}");
    }
}
