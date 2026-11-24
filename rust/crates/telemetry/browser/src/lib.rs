// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Browser-hosted API facade for Sirius Quent traces.

#[cfg(feature = "generate")]
pub mod generate;

use std::collections::HashSet;

use quent_events::Event;
use quent_query_engine_analyzer::ui::UiAnalyzer;
use quent_query_engine_browser::{BrowserError, BrowserService};
use sirius_telemetry_analyzer::SiriusUiAnalyzer;
use sirius_telemetry_store::{EngineEvent, SCHEMA_DIGEST, SiriusEvent};
use uuid::Uuid;

/// Browser-bundle model name shared by the producer and this analyzer.
pub const MODEL_NAME: &str = "sirius";

pub(crate) fn engine_id(event: &Event<SiriusEvent>) -> Option<Uuid> {
    matches!(event.data, SiriusEvent::Engine(EngineEvent::Init { .. })).then_some(event.id)
}

fn load_browser_service(bytes: &[u8]) -> Result<BrowserService<SiriusUiAnalyzer>, BrowserError> {
    let bundle = quent_browser_bundle::decode(bytes).map_err(bundle_decode_error)?;
    if bundle.metadata.model_name != MODEL_NAME {
        return Err(BrowserError::incompatible_bundle(format!(
            "browser bundle model mismatch: expected {MODEL_NAME}, got {}",
            bundle.metadata.model_name
        )));
    }
    if bundle.metadata.schema_digest != SCHEMA_DIGEST {
        return Err(BrowserError::incompatible_bundle(format!(
            "browser bundle schema mismatch: expected {SCHEMA_DIGEST}, got {}",
            bundle.metadata.schema_digest
        )));
    }
    if bundle.metadata.codec != "postcard" {
        return Err(BrowserError::incompatible_bundle(format!(
            "unsupported sirius event codec: {}",
            bundle.metadata.codec
        )));
    }

    let events: Vec<Event<SiriusEvent>> =
        postcard::from_bytes(bundle.payload).map_err(|error| {
            BrowserError::invalid_bundle(format!("unable to decode sirius events: {error}"))
        })?;
    let engine_ids = events.iter().filter_map(engine_id).collect::<HashSet<_>>();
    if engine_ids.len() != 1 {
        return Err(BrowserError::invalid_bundle(format!(
            "expected exactly one sirius engine, found {}",
            engine_ids.len()
        )));
    }
    let engine_id = *engine_ids.iter().next().expect("one engine was validated");
    if engine_id != bundle.metadata.engine_id {
        return Err(BrowserError::invalid_bundle(
            "browser bundle engine ID does not match its events",
        ));
    }
    let analyzer =
        SiriusUiAnalyzer::try_new(engine_id, events.into_iter()).map_err(BrowserError::from)?;
    Ok(BrowserService::new(
        analyzer,
        engine_id,
        bundle.metadata.context_ids,
    ))
}

fn bundle_decode_error(error: quent_browser_bundle::Error) -> BrowserError {
    let message = format!("unable to decode browser bundle: {error}");
    match error {
        quent_browser_bundle::Error::UnsupportedVersion(_) => {
            BrowserError::incompatible_bundle(message)
        }
        quent_browser_bundle::Error::Io(_) => BrowserError::failure(message),
        quent_browser_bundle::Error::Truncated
        | quent_browser_bundle::Error::InvalidMagic
        | quent_browser_bundle::Error::LengthOverflow
        | quent_browser_bundle::Error::InvalidLength
        | quent_browser_bundle::Error::Checksum
        | quent_browser_bundle::Error::Metadata(_) => BrowserError::invalid_bundle(message),
    }
}

quent_query_engine_browser::export_wasm_browser_server!(SiriusUiAnalyzer, load_browser_service);

#[cfg(test)]
mod tests {
    use quent_browser_bundle::{Capabilities, FORMAT_VERSION, Metadata};
    use uuid::Uuid;

    use super::*;

    fn metadata(model_name: &str, schema_digest: &str, engine_id: Uuid) -> Metadata {
        Metadata {
            format_version: FORMAT_VERSION,
            model_name: model_name.into(),
            schema_digest: schema_digest.into(),
            model_provenance: None,
            analyzer_revision: "test".into(),
            quent_revision: "test".into(),
            codec: "postcard".into(),
            engine_id,
            context_ids: vec![],
            capabilities: Capabilities::default(),
        }
    }

    fn bundle(model_name: &str, schema_digest: &str) -> Vec<u8> {
        quent_browser_bundle::encode(
            &metadata(model_name, schema_digest, Uuid::nil()),
            // Not a valid event list: identity mismatches must be reported first.
            b"\xff\xff\xff",
        )
        .unwrap()
    }

    fn engine_init(id: Uuid) -> Event<SiriusEvent> {
        Event::new(
            id,
            1,
            SiriusEvent::Engine(EngineEvent::Init { label: None }),
        )
    }

    fn bundle_of(events: &[Event<SiriusEvent>], engine_id: Uuid) -> Vec<u8> {
        quent_browser_bundle::encode(
            &metadata(MODEL_NAME, SCHEMA_DIGEST, engine_id),
            &postcard::to_allocvec(&events).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn rejects_schema_digest_mismatch_before_decoding_events() {
        let error = load_browser_service(&bundle(MODEL_NAME, "sha256:other"))
            .err()
            .unwrap();
        assert_eq!(error.code(), "incompatible-bundle");
        assert!(error.to_string().contains("schema mismatch"), "{error}");
    }

    #[test]
    fn rejects_model_mismatch_before_decoding_events() {
        let error = load_browser_service(&bundle("other-model", SCHEMA_DIGEST))
            .err()
            .unwrap();
        assert_eq!(error.code(), "incompatible-bundle");
        assert!(error.to_string().contains("model mismatch"), "{error}");
    }

    #[test]
    fn accepts_matching_schema_digest_and_proceeds_to_decoding() {
        let error = load_browser_service(&bundle(MODEL_NAME, SCHEMA_DIGEST))
            .err()
            .unwrap();
        assert_eq!(error.code(), "invalid-bundle");
        assert!(
            error.to_string().contains("unable to decode sirius events"),
            "{error}"
        );
    }

    #[test]
    fn rejects_truncated_bundle() {
        let error = load_browser_service(&[0, 1, 2]).err().unwrap();
        assert_eq!(error.code(), "invalid-bundle");
        assert!(error.to_string().contains("truncated"), "{error}");
    }

    #[test]
    fn rejects_unsupported_codec() {
        let mut metadata = metadata(MODEL_NAME, SCHEMA_DIGEST, Uuid::nil());
        metadata.codec = "msgpack".into();
        let error = load_browser_service(&quent_browser_bundle::encode(&metadata, b"").unwrap())
            .err()
            .unwrap();
        assert_eq!(error.code(), "incompatible-bundle");
        assert!(error.to_string().contains("codec"), "{error}");
    }

    #[test]
    fn rejects_bundle_without_exactly_one_engine() {
        let engine = Uuid::now_v7();
        let none = load_browser_service(&bundle_of(&[], engine)).err().unwrap();
        assert_eq!(none.code(), "invalid-bundle");
        assert!(none.to_string().contains("found 0"), "{none}");

        let two = load_browser_service(&bundle_of(
            &[engine_init(engine), engine_init(Uuid::now_v7())],
            engine,
        ))
        .err()
        .unwrap();
        assert_eq!(two.code(), "invalid-bundle");
        assert!(two.to_string().contains("found 2"), "{two}");
    }

    #[test]
    fn rejects_engine_id_that_differs_from_metadata() {
        let error = load_browser_service(&bundle_of(&[engine_init(Uuid::now_v7())], Uuid::nil()))
            .err()
            .unwrap();
        assert_eq!(error.code(), "invalid-bundle");
        assert!(error.to_string().contains("engine ID"), "{error}");
    }

    #[test]
    fn loads_a_bundle_with_one_engine() {
        let engine = Uuid::now_v7();
        load_browser_service(&bundle_of(&[engine_init(engine)], engine))
            .expect("analyzer accepts a matching bundle");
    }
}
