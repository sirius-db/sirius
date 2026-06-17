use quent_query_engine_ui::QueryBundle;
use quent_ui::timeline::{
    request::{BulkTimelineRequest, SingleTimelineRequest},
    response::{BulkTimelinesResponse, SingleTimelineResponse},
};
use quent_query_engine_ui::{OperatorFilter, QueryFilter};
use quent_simulator_ui::EntityRef;
use ts_rs::TS;

const TS_OUT_DIR: &str = "./ts-bindings/";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    <QueryBundle<EntityRef> as TS>::export_all_to(TS_OUT_DIR)?;
    <SingleTimelineRequest<QueryFilter, OperatorFilter> as TS>::export_all_to(TS_OUT_DIR)?;
    <SingleTimelineResponse as TS>::export_all_to(TS_OUT_DIR)?;
    <BulkTimelineRequest<QueryFilter, OperatorFilter> as TS>::export_all_to(TS_OUT_DIR)?;
    <BulkTimelinesResponse as TS>::export_all_to(TS_OUT_DIR)?;
    Ok(())
}
