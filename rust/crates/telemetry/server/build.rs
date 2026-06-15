use quent_query_engine_ui::QueryBundle;
use quent_ui::timeline::{
    request::{BulkTimelineRequest, SingleTimelineRequest},
    response::{BulkTimelinesResponse, SingleTimelineResponse},
};
use sirius_telemetry_ui::{EntityRef, QueryFilter, TaskFilter};
use ts_rs::TS;

const TS_OUT_DIR: &str = "./ts-bindings/";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    <QueryBundle<EntityRef> as TS>::export_all_to(TS_OUT_DIR)?;
    <SingleTimelineRequest<QueryFilter, TaskFilter> as TS>::export_all_to(TS_OUT_DIR)?;
    <SingleTimelineResponse as TS>::export_all_to(TS_OUT_DIR)?;
    <BulkTimelineRequest<QueryFilter, TaskFilter> as TS>::export_all_to(TS_OUT_DIR)?;
    <BulkTimelinesResponse as TS>::export_all_to(TS_OUT_DIR)?;
    Ok(())
}
